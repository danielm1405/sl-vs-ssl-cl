import argparse
from copy import deepcopy
from typing import Any, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cassle.distillers.base import base_distill_wrapper


def pfr_distill_wrapper(Method=object):
    class PFRDistillWrapper(Method):
        def __init__(
            self,
            distill_lamb: float,
            distill_projector_type: str,
            distill_proj_hidden_dim: int,
            **kwargs
        ) -> None:
            super().__init__(**kwargs)

            self.distill_lamb = distill_lamb
            self.distill_projector_type = distill_projector_type

            if self.distill_projector_type == "mlp":
                self.distill_projector = nn.Sequential(
                    nn.Linear(self.features_dim, distill_proj_hidden_dim, bias=False),
                    nn.BatchNorm1d(distill_proj_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(distill_proj_hidden_dim, self.features_dim),
                )
            elif self.distill_projector_type == "identity":
                self.distill_projector = nn.Identity()
                
            self.frozen_encoder = deepcopy(self.encoder)

        @staticmethod
        def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("pfr")

            parser.add_argument("--distill_lamb", type=float, default=1)
            parser.add_argument("--distill_projector_type", type=str, default="mlp",
                                choices=["mlp", "identity"])
            parser.add_argument("--distill_proj_hidden_dim", type=int, default=256)

            return parent_parser

        @property
        def learnable_params(self) -> List[dict]:
            """Adds projector parameters to parent's learnable parameters.

            Returns:
                List[dict]: list of learnable parameters.
            """
            return super().learnable_params + [
                {
                    "name": "distill_projector",
                    "params": self.distill_projector.parameters()
                },
            ]

        def on_train_start(self):
            super().on_train_start()
            if self.current_task_idx > 0:
                self.frozen_encoder = deepcopy(self.encoder)
                for pg in self.frozen_encoder.parameters():
                    pg.requires_grad = False

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:         
            out = super().training_step(batch, batch_idx)
            feats1, feats2 = out["feats"]
            
            _, (X1, X2), _ = batch[f"task{self.current_task_idx}"]
            frozen_feats1, frozen_feats2 = self.frozen_encoder(X1), self.frozen_encoder(X2)

            p1, p2 = self.distill_projector(feats1), self.distill_projector(feats2)

            distill_loss = -(
                F.cosine_similarity(p1, frozen_feats1).mean() + \
                F.cosine_similarity(p2, frozen_feats2).mean()
            ) / 2

            self.log_dict(
                {
                    "train_distill_loss": distill_loss,
                }
            )

            return out["loss"] + self.distill_lamb * distill_loss

    return PFRDistillWrapper
