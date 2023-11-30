import os
import types
from typing import Sequence

import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

from pytorch_lightning import seed_everything

from cassle.utils.classification_dataloader import prepare_data


def get_backbone():
    backbone = resnet18()
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    return backbone


def get_initialized_backbone(ckpt_path):
    state = torch.load(ckpt_path)["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder.", "")] = state[k]
        del state[k]
    backbone = get_backbone()
    backbone.load_state_dict(state, strict=False)
    backbone.to("cuda")
    return backbone


def get_per_class_embeddings_and_classes(backbone, loader):
    embeddings, classes = [], []
    backbone.eval()
    for X, Y in loader:
        X = X.to("cuda")
        Y = Y.to("cuda")
        with torch.no_grad():
            e = backbone(X)
        embeddings.append(e)
        classes.append(Y)
    embeddings = torch.cat(embeddings)
    classes = torch.cat(classes)
    
    per_class_embeddings = {i: torch.empty((0,512)) for i in torch.unique(classes)}
    for cls, embds in per_class_embeddings.items():
        ind = classes == cls
        per_class_embeddings[cls] = embeddings[ind]
    
    return per_class_embeddings, classes


def calc_prototypes(per_class_embeddings):
    return {cls: embds.mean(dim=0) for cls, embds in per_class_embeddings.items()}


def calc_accuracies(per_class_embeddings, prototypes):
    accs = {}
    for cls, embds in per_class_embeddings.items():
        closests = []
        for embd in embds:
            closest_class = -1
            min_dist = 10000000000

            for prot_cls, prot in prototypes.items():
                d = torch.norm(embd - prot, p=2)
                if d < min_dist:
                    closest_class = prot_cls
                    min_dist = d

            closests.append(closest_class)

        closests = torch.tensor(closests, device='cuda')
        accs[cls] = sum(closests == cls) / closests.shape[0]
        
    avg_acc = sum(accs.values()) / len(accs.values())

    return accs, avg_acc


def calc_old_and_new_NMC(datasets: Sequence[str], ckpt_paths: dict):
    print(f"#######################################")
    print(f"Checkpoint paths:\n{ckpt_paths}\n")

    for dataset in datasets:

        print(f"### Dataset: {dataset} ###\n")

        train_loader, val_loader = prepare_data(
            dataset,
            data_dir="data",
            train_dir=None,
            val_dir=None,
            batch_size=1000,
            num_workers=4,
            semi_supervised=None,
        )

        # t0 acc
        t0_backbone = get_initialized_backbone(ckpt_paths[0])
        t0_per_class_embeddings, t0_classes = get_per_class_embeddings_and_classes(t0_backbone, train_loader)
        prototypes_after_t0 = calc_prototypes(t0_per_class_embeddings)

        val_t0_per_class_embeddings, val_t0_classes = get_per_class_embeddings_and_classes(t0_backbone, val_loader)
        
        _, old_bb_old_proto = calc_accuracies(val_t0_per_class_embeddings,
                                                            prototypes_after_t0)
        
        # t1 acc
        t1_backbone = get_initialized_backbone(ckpt_paths[1])
        t1_per_class_embeddings, t1_classes = get_per_class_embeddings_and_classes(t1_backbone, train_loader)
        prototypes_after_t1 = calc_prototypes(t1_per_class_embeddings)

        val_t1_per_class_embeddings, val_t1_classes = get_per_class_embeddings_and_classes(t1_backbone, val_loader)

        # using old prototypes
        _, new_bb_old_proto = calc_accuracies(val_t1_per_class_embeddings,
                                                                prototypes_after_t0)

        # using new prototypes
        _, new_bb_new_proto = calc_accuracies(val_t1_per_class_embeddings,
                                                                prototypes_after_t1)

        print(f"old bb, old proto acc: {100 * old_bb_old_proto:.1f}\n"
            f"new bb, old proto acc: {100 * new_bb_old_proto:.1f}\n"
            f"new bb, new proto acc: {100 * new_bb_new_proto:.1f}\n\n")

    print(f"#######################################\n\n")

def main():
    seed_everything(5)

    experiments = [
        # sl
        {
            'datasets': ['cifar10', 'svhn'],
            'ckpt_paths': {
                0: "X.ckpt",
                1: "X.ckpt"
            }
        },
        {
            'datasets': ['cifar10', 'cifar100'],
            'ckpt_paths': {
                0: "X.ckpt",
                1: "X.ckpt"
            }
        },
        # barlow
        {
            'datasets': ['cifar10', 'svhn'],
            'ckpt_paths': {
                0: "X.ckpt",
                1: "X.ckpt"
            }
        },
        {
            'datasets': ['cifar10', 'cifar100'],
            'ckpt_paths': {
                0: "",
                1: ""
            }
        },
        # sl+mlp
        {
            'datasets': ['cifar10', 'svhn'],
            'ckpt_paths': {
                0: "X.ckpt",
                1: "X.ckpt"
            }
        },
        {
            'datasets': ['cifar10', 'cifar100'],
            'ckpt_paths': {
                0: "",
                1: ""
            }
        },
    ]

    for exp in experiments:
        try:
            calc_old_and_new_NMC(**exp)
        except Exception as ex:
            print(ex)


if __name__ == "__main__":
    main()
