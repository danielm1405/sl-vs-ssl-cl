# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import json
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from cassle.args.utils import N_CLASSES_PER_DATASET
from cassle.args.knn import parse_args_knn
from cassle.utils.classification_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_transforms,
)
from cassle.methods import METHODS
from cassle.utils.knn import WeightedKNNClassifier


@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module) -> Tuple[torch.Tensor]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """

    model.eval()
    backbone_features, labels = [], []
    for im, lab in tqdm(loader):
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        outs = model(im)
        backbone_features.append(outs["feats"].detach())
        labels.append(lab)
    model.train()
    backbone_features = torch.cat(backbone_features)
    labels = torch.cat(labels)
    return backbone_features, labels


@torch.no_grad()
def run_knn(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    k: int,
    T: float,
    distance_fx: str,
) -> Tuple[float]:
    """Runs offline knn on a train and a test dataset.

    Args:
        train_features (torch.Tensor, optional): train features.
        train_targets (torch.Tensor, optional): train targets.
        test_features (torch.Tensor, optional): test features.
        test_targets (torch.Tensor, optional): test targets.
        k (int): number of neighbors.
        T (float): temperature for the exponential. Only used with cosine
            distance.
        distance_fx (str): distance function.

    Returns:
        Tuple[float]: tuple containing the the knn acc@1 and acc@5 for the model.
    """

    # build knn
    knn = WeightedKNNClassifier(
        k=k,
        T=T,
        distance_fx=distance_fx,
    )

    # add features
    knn(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
    )

    # compute
    results_dict = knn.compute()

    # free up memory
    del knn

    return results_dict['val_knn_acc1'], results_dict['val_knn_acc5']


def search_knn(model, dataset, data_dir, train_dir, val_dir, batch_size, ks, distance_functions, temperatures):
    # prepare data - use same augs in train and val!
    _, T_val = prepare_transforms(dataset)
    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_val,
        T_val,
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
    )

    # extract features
    train_features, train_targets = extract_features(train_loader, model)
    test_features, test_targets = extract_features(val_loader, model)

    # run k-nn for all possible combinations of parameters
    results = {}
    for k in ks:
        for distance_fx in distance_functions:
            Ts = temperatures if distance_fx == "cosine" else [None]
            for T in Ts:
                # print("---")
                # print(f"Running k-NN with params: distance_fx={distance_fx}, k={k}, T={T}...")
                acc1, acc5 = run_knn(
                    train_features=train_features,
                    train_targets=train_targets,
                    test_features=test_features,
                    test_targets=test_targets,
                    k=k,
                    T=T,
                    distance_fx=distance_fx,
                )
                # print(f"Result: acc@1={acc1}, acc@5={acc5}")
                
                results[f"{k=}-{distance_fx}-{T=}"] = acc1

    return results


def eval_single_dataset(model, dataset, data_dir, train_dir, val_dir, batch_size, ks, distance_functions, temperatures):
    results = search_knn(model, dataset, data_dir, train_dir, val_dir, batch_size, ks, distance_functions, temperatures)
    
    print("\n---\n")
    print("ALL RESULTS:\n{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in results.items()) + "}")

    # BEST RESULTS
    best_val_knn_acc1 = max(results.items(), key=lambda k: k[1])
    print(f"BEST RESULTS: {best_val_knn_acc1}")
    
    wandb.log({'best_val_knn_acc1': best_val_knn_acc1[1]})
    wandb.log({'best_val_knn_params': best_val_knn_acc1[0]})
  
  
def eval_multidataset(model, datasets, data_dir, train_dir, val_dir, batch_size, ks, distance_functions, temperatures):
    for task_idx, dataset in enumerate(datasets):
        results = search_knn(model, dataset, data_dir, train_dir, val_dir, batch_size, ks, distance_functions, temperatures)
        
        print("\n---\n")
        print("ALL RESULTS:\n{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in results.items()) + "}")

        # BEST RESULTS
        best_val_knn_acc1 = max(results.items(), key=lambda k: k[1])
        print(f"BEST RESULTS for task{task_idx}: {best_val_knn_acc1}")
        
        wandb.log({f'best_val_knn_acc1_task{task_idx}': best_val_knn_acc1[1]})
        wandb.log({f'best_val_knn_params_task{task_idx}': best_val_knn_acc1[0]})
    
      

def main():
    args = parse_args_knn()

    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    if str(ckpt_dir).endswith(".ckpt"):
        ckpt_dir = ckpt_dir.parent
    args_path = ckpt_dir / "args.json"
    ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)

    dataset = method_args['dataset']
    method_args['task_idx'] = 0
    n_total_classes = N_CLASSES_PER_DATASET[dataset] if method_args['datasets'] == [] else \
        sum([N_CLASSES_PER_DATASET[d] for d in method_args['datasets']])
    
    # build the model
    model = METHODS[method_args["method"]](
        **method_args, tasks=[list(range(n_total_classes))]
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        if 'heads' in k or 'projector' in k:
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model.cuda()

    wandb.init(id=str(ckpt_dir).split('/')[-1], resume="must")
    if method_args['datasets'] == []:
        eval_single_dataset(model, dataset, args.data_dir, args.train_dir, args.val_dir, args.batch_size,
                            args.k, args.distance_function, args.temperature)
    else:
        eval_multidataset(model, method_args['datasets'], args.data_dir, args.train_dir, args.val_dir, args.batch_size,
                          args.k, args.distance_function, args.temperature)


if __name__ == "__main__":
    main()
