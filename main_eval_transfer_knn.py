import os
import types
from tqdm import tqdm
from pathlib import Path
import re

import numpy as np
import torchvision.transforms as T
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50

from cassle.utils.transfer_datasets import load_datasets
from cassle.utils.classification_dataloader import (
    prepare_dataloaders,
)
from main_knn import search_knn


def get_backbone():
    backbone = resnet18()
    # backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    # backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    return backbone


def get_initialized_backbone(ckpt_path):
    state = torch.load(ckpt_path)["state_dict"]
    for k in list(state.keys()):
        if k.startswith("frozen_"):
            pass
        elif "encoder" in k:
            state[k.replace("encoder.", "")] = state[k]
        del state[k]
    backbone = get_backbone()
    backbone.load_state_dict(state, strict=True)
    backbone.to("cuda")
    backbone.eval()
    return backbone


@torch.no_grad()
def extract_features(loader, model):
    model.eval()
    backbone_features, labels = [], []
    for im, lab in tqdm(loader):
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        outs = model(im)
        backbone_features.append(outs.detach())
        labels.append(lab)
    model.train()
    backbone_features = torch.cat(backbone_features)
    labels = torch.cat(labels)
    return backbone_features, labels


def directory_find(atom, root="."):
    for path, dirs, files in os.walk(root):
        if atom in dirs:
            return os.path.join(path, atom)


def find_numbers_after_substring(S, s):
    # Create a regular expression pattern to match 's' followed by numbers.
    pattern = re.compile(rf"{re.escape(s)}(\d+)")

    # Use re.findall to find all matches in the string.
    matches = pattern.findall(S)

    # Extract and return the numbers from the matches.
    numbers = [match for match in matches]

    return numbers


def get_all_checkpoints(exp_ids, required_tasks):
    out_ckpts = {}
    for _task_idx in required_tasks:
        out_ckpts[_task_idx] = {}
        for method_name in exp_ids.keys():
            out_ckpts[_task_idx][method_name] = []

    for method_name, child_exp_ids in exp_ids.items():
        for child_exp_id in child_exp_ids:
            child_exp_dir = directory_find(child_exp_id)
            parent_dir = Path(child_exp_dir).parent
            # print(f'Searching parent dir: {parent_dir}')
            items = os.listdir(parent_dir)
            exp_names = [
                item for item in items if os.path.isdir(os.path.join(parent_dir, item))
            ]
            # print(f'Found {len(exp_names)} experiments: {exp_names}')

            task_ckpts = {}
            for exp_name in exp_names:
                filenames = os.listdir(parent_dir / exp_name)
                ckpts = [
                    filename
                    for filename in filenames
                    if str(parent_dir / exp_name / filename).endswith(".ckpt")
                ]
                if len(ckpts) == 0:
                    print(f"Checkpoint not found in {parent_dir / exp_name}!")
                    return False
                if len(ckpts) > 1:
                    print(
                        f"Multiple checkpoints found in {parent_dir / exp_name}: {ckpts}!"
                    )
                    return False

                task_idx = find_numbers_after_substring(ckpts[0], "task")[0]
                task_ckpts[int(task_idx)] = str(parent_dir / exp_name / ckpts[0])

            if not set(required_tasks).issubset(set(task_ckpts.keys())):
                print(
                    f"Missing checkpoints for tasks {set(required_tasks) - set(task_ckpts.keys())}!"
                )
                return False

            for task_idx in required_tasks:
                out_ckpts[task_idx][method_name].append(task_ckpts[task_idx])

    return out_ckpts


def main():
    data_dir = "/home/data"
    eval_datasets = [
        'food101',
        'pets',
        'flowers',
        'caltech101',
        'cars',
        'aircraft',
        'dtd',
        "cub200",
    ]

    # these are IDs ofany experiment (task) from a given task sequence
    ckpt_ids = {
        # SL
        "SL": [
            "l05vzps7",  
            "swze51f1",
            "no31nu9m",
        ],
        "SL+LWF": [
            "k5fzizxw",
            "ulmxuuof",
            "eabih1kf",
        ],
        "SL+PFR": [
            "g59g8d4z",
            "4irpwgsp",
            "umo196pk",
        ],
        # SL+MLP
        "SL+MLP": [
            "pnsvrp9v",
            "3z15t5rh",
            "fdu94f3o",
        ],
        "SL+MLP+LWF": [
            "dsykgoxb",
            "oksbngg2",
            "qvegd1ug",
        ],
        "SL+MLP+PFR": [
            "tky21t6k",
            "5mumsnio",
            "alivjlha",
        ],
        # SSL
        "SSL": [
            "ipd3saw3",
            "x71vjniv",
            "37j458wf",
        ],
        "SSL+PFR": [
            "q7jefoe2",
            "mm9w3xes",
            "zxlj7qlt",
        ],
        "SSL+CaSSLe": [
            "39j8bh5x",
            "fou92ki2",
            "35kik0d5",
        ],
    }

    tasks = range(5)

    all_ckpts = get_all_checkpoints(ckpt_ids, tasks)

    # knn search params
    ks = [5, 10, 20, 50, 100, 200]
    distance_functions = ["euclidean", "cosine"]
    temperatures = [0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 1]

    results = {}

    # iterate over tasks
    for task_idx in tasks:
        task_ckpts = all_ckpts[task_idx]
        results[task_idx] = {}
        
        # iterate over methods
        for method_name, ckpts in task_ckpts.items():
            results[task_idx][method_name] = {}

            # iterate over eval datasets
            for eval_dataset in eval_datasets:
                results[task_idx][method_name][eval_dataset] = []
                ds_dict = load_datasets(eval_dataset, data_dir)

                bs = 100
                n_workers = 4

                train_dl = DataLoader(
                    ds_dict["trainval"],
                    batch_size=bs,
                    num_workers=n_workers,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False,
                )
                test_dl = DataLoader(
                    ds_dict["test"],
                    batch_size=bs,
                    num_workers=n_workers,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False,
                )

                # iterate over seeds
                for ckpt in ckpts:
                    print(
                        f"\nEvaluating:\n\t{method_name=}\n\ton {eval_dataset=}\n\tfrom ckpt {ckpt=}"
                    )

                    try:
                        model = get_initialized_backbone(ckpt)

                        # extract features
                        train_features, train_targets = extract_features(
                            train_dl, model
                        )
                        test_features, test_targets = extract_features(test_dl, model)

                        _r = search_knn(
                            train_features,
                            train_targets,
                            test_features,
                            test_targets,
                            ks,
                            distance_functions,
                            temperatures,
                        )
                        print(f"All results: {_r}")

                        _best = max(_r.items(), key=lambda k: k[1])[1]
                        print(f"Best results: {_best}")

                        results[task_idx][method_name][eval_dataset].append(_best)
                    except Exception as exc:
                        print(f"EVAL FAILED!\n{exc}")
                        results[task_idx][method_name][eval_dataset].append("FAILED")

    print(f"FINAL RESULTS\n{results}")


if __name__ == "__main__":
    main()

