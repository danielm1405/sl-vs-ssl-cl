import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from cassle.args.setup import parse_args_pretrain
from cassle.methods import METHODS
from cassle.distillers import DISTILLERS

from cassle.utils.checkpointer import Checkpointer
from cassle.utils.classification_dataloader import prepare_val_dataset, drop_last

from cassle.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_transform,
    ConcatDatasetWithMetadata,
)
from main_knn import eval_multidataset as knn_seach_eval


def main():
    args = parse_args_pretrain()

    seed_everything(args.seed)

    # online eval dataset reloads when task dataset is over
    args.multiple_trainloader_mode = "min_size"

    datasets = args.datasets
    print(f"{datasets=}")

    tasks = []  # tasks will be handled later
    # collect datasets
    train_datasets, online_eval_datasets, val_datasets = [], [], []

    # pretrain and online eval dataloaders
    for dataset in datasets:
        # asymmetric augmentations
        if args.unique_augs > 1:
            transform = [
                prepare_transform(dataset, multicrop=args.multicrop, **kwargs)
                for kwargs in args.transform_kwargs
            ]
        else:
            transform = prepare_transform(
                dataset, multicrop=args.multicrop, **args.transform_kwargs
            )

        online_eval_transform = transform[-1] if isinstance(transform, list) else transform
        task_transform = prepare_n_crop_transform(transform, num_crops=args.num_crops)

        _train_dataset, _online_eval_dataset = prepare_datasets(
            dataset,
            task_transform=task_transform,
            online_eval_transform=online_eval_transform,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            no_labels=args.no_labels,
        )

        _val_dataset = prepare_val_dataset(
            dataset,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
        )

        drop_last(_val_dataset, args.batch_size)

        next_target_base = 0
        if train_datasets:
            next_target_base = int(torch.max(torch.Tensor(train_datasets[-1].targets))) + 1

        for ds in [_online_eval_dataset, _val_dataset]:
            if isinstance(ds.targets, torch.Tensor) or isinstance(ds.targets, np.ndarray):
                ds.targets += next_target_base
            elif isinstance(ds.targets, list):
                ds.targets = [t + next_target_base for t in ds.targets]
            else:
                raise NotImplementedError(f"Type not supported: {ds.targets=}")

        tasks.append(torch.unique(torch.tensor(_val_dataset.targets)))

        print(f"{dataset=}: {len(_train_dataset)}, {len(_online_eval_dataset)}, {len(_val_dataset)}")

        train_datasets.append(_train_dataset)
        online_eval_datasets.append(_online_eval_dataset)
        val_datasets.append(_val_dataset)

    online_eval_dataset = ConcatDatasetWithMetadata(online_eval_datasets)
    val_dataset = ConcatDatasetWithMetadata(val_datasets)

    tasks = tuple(tasks)
    args.num_classes = len(torch.unique(torch.cat(tasks)))

    task_dataset = train_datasets[args.task_idx]

    # make loaders for datasets
    task_loader = prepare_dataloader(
        task_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    args.online_eval_batch_size = len(online_eval_dataset) // len(task_loader) \
        if args.eval_using_whole_dataset else int(args.batch_size)
    online_eval_loader = prepare_dataloader(
        online_eval_dataset,
        batch_size=args.online_eval_batch_size,
        num_workers=args.num_workers,
    )
    train_loaders = {
        f"task{args.task_idx}": task_loader,
        "online_eval": online_eval_loader
    }

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    assert args.method in METHODS, f"Choose from {METHODS.keys()}"
    MethodClass = METHODS[args.method]

    if args.distiller:
        MethodClass = DISTILLERS[args.distiller](MethodClass)

    # tasks - define classes for splitting
    model = MethodClass(**args.__dict__, tasks=tasks if args.split_strategy == "class" else None)

    # only one resume mode can be true
    assert [args.resume_from_checkpoint, args.pretrained_model].count(True) <= 1

    if args.resume_from_checkpoint:
        pass  # handled by the trainer
    elif args.pretrained_model:
        print(f"Loading previous task checkpoint {args.pretrained_model}...")
        state_dict = torch.load(args.pretrained_model, map_location="cpu")["state_dict"]
        if args.reset_projector:
            for k in list(state_dict.keys()):
                if "projector" in k:
                    print(f"Removing params {k}")
                    del state_dict[k]
        model.load_state_dict(state_dict, strict=False)

    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{args.name}-task{args.task_idx}",
            project=args.project,
            entity=args.entity,
            offline=args.offline,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    if args.save_checkpoint:
        ckpt = Checkpointer(
            args,
            logdir=args.checkpoint_dir,
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        checkpoint_callback=False,
        terminate_on_nan=True,
    )

    model.current_task_idx = args.task_idx

    trainer.fit(model, train_loaders, val_loader)
    
    model.cuda()
    knn_seach_eval(model, datasets, args.data_dir, args.train_dir, args.val_dir, args.batch_size,
                   ks=[5, 10, 20, 50, 100, 200],
                   distance_functions=['euclidean', 'cosine'],
                   temperatures=[0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 1])


if __name__ == "__main__":
    main()
