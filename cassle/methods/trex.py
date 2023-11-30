import argparse
from typing import Any, List, Sequence, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from cassle.losses.barlow import barlow_loss_func
from cassle.methods.base import BaseModel
from cassle.utils.metrics import weighted_mean
from cassle.utils.projectors import PROJECTORS


class TReX(BaseModel):
    def __init__(
        self,
        num_classes: int,
        mlp_projector: str = None,
        **kwargs
    ):
        super().__init__(num_classes=num_classes, **kwargs)

        tasks = kwargs['tasks']
        task_idx = kwargs['task_idx']
        self.heads = nn.ModuleList()
        for t in range(task_idx + 1):
            head_layers = []
            if mlp_projector:
                head_layers.append(PROJECTORS[mlp_projector](ft_dim=self.features_dim, bottleneck_dim=self.features_dim))
                
            head_layers.append(ClfLayer(self.features_dim, len(tasks[t])))
            self.heads.append(nn.Sequential(*head_layers))

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(TReX, TReX).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("TReX")
        parser.add_argument(
            "--mlp-projector",
            type=str,
            choices=list(PROJECTORS.keys()) + [None],
            default='trex'
        )
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        return super().learnable_params + self.extra_learnable_params
    
    @property
    def extra_learnable_params(self) -> List[dict]:
        return [
            {
                "name": "head",
                "params": self.heads[-1].parameters()
            },
        ]

    def on_train_start(self):
        super().on_train_start()

        for head in self.heads[:-1]:
            for p in head.parameters():
                p.requires_grad = False
                
        self.curr_task_labels, _ = self.tasks[self.current_task_idx].to(self.device).sort()
        self.global_to_local_label_map = {k.item(): v for v, k in enumerate(self.curr_task_labels)}

    def forward(self, X, *args, **kwargs):
        out = super().forward(X, *args, **kwargs)
        z = self.heads[-1](out["feats"])
        return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        out = super().training_step(batch, batch_idx)
        loss_cls = out["loss"]

        _, _, labels = batch[f"task{self.current_task_idx}"]

        feats1, feats2 = out["feats"]

        logits1 = self.heads[-1](feats1)
        logits2 = self.heads[-1](feats2)
        
        logits = torch.cat([logits1, logits2])
        labels = torch.cat([labels, labels])

        CE_loss = F.cross_entropy(logits, labels)

        loss = out["loss"] + CE_loss
        
        pred = logits.argmax(dim=1)
        acc1 = (pred == labels).sum() / pred.shape[0] * 100.0

        self.log_dict(
            {
                "train_supervised_loss": CE_loss,
                "train_cls_loss": out["loss"],
                "train_total_loss": loss,
                "train_sup_acc1": acc1,
            },
            sync_dist=True
        )

        out.update(
            {
                "loss": loss,
                "loss_cls": loss_cls,
                "loss_ssl": CE_loss,
                "z": [logits1, logits2]
            }
        )

        return out

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        if self.online_eval and not self.trainer.sanity_checking:
            out = super().validation_step(batch, batch_idx)
            
            *_, targets = batch
            
            curr_task_idx = [x in self.curr_task_labels for x in targets]
            
            curr_task_targets = targets[curr_task_idx]
            curr_task_feats = out['feats'][curr_task_idx]
            
            # slow remapping :(
            for i in range(len(curr_task_targets)):
                curr_task_targets[i] = self.global_to_local_label_map[curr_task_targets[i].item()]
            
            logits = self.heads[-1](curr_task_feats)
            pred = logits.argmax(dim=1)
            acc1 = (pred == curr_task_targets).sum() / pred.shape[0] * 100.0
            
            return {**out,
                    'val_sup_acc1': acc1,
                    'curr_task_samples_in_batch': curr_task_targets.shape[0]}

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        if self.online_eval and not self.trainer.sanity_checking:
            super().validation_epoch_end(outs)

            val_sup_acc1 = weighted_mean(outs, "val_sup_acc1", "curr_task_samples_in_batch")
            self.log_dict({"val_sup_acc1": val_sup_acc1})


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def extra_repr(self):
        return "dim={}".format(self.dim)

    def forward(self, x):
        return nn.functional.normalize(x, dim=self.dim, p=2)


class ClfLayer(nn.Module):
    def __init__(self, emb_dim, n_classes, tau=0.1):
        super().__init__()
        self.tau = tau

        self.norm = nn.Identity()
        if tau > 0:
            self.norm = L2Norm(dim=1)

        self.fc = nn.Linear(emb_dim, n_classes, bias=False)

    def forward(self, x):
        # no temperature scaling
        if self.tau <= 0:
            return self.fc(x)

        # temperature scaling with l2-normalized weights
        x = self.norm(x)
        w = self.norm(self.fc.weight)
        o = (x @ w.t()) / self.tau
        return o

    def extra_repr(self):
        return "tau={}".format(self.tau)
