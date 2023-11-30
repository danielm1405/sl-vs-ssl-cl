import argparse
from typing import Any, List, Sequence, Optional

import torch
import torch.nn as nn
from cassle.losses.barlow import barlow_loss_func
from cassle.methods.base import BaseModel


class BarlowTwins(BaseModel):
    def __init__(
        self,
        proj_hidden_dim: int,
        output_dim: int,
        lamb: float,
        scale_loss: float,
        reset_extra_params: bool,
        **kwargs
    ):
        """Implements Barlow Twins (https://arxiv.org/abs/2103.03230)

        Args:
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            output_dim (int): number of dimensions of projected features.
            lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
            scale_loss (float): scaling factor of the loss.
        """

        super().__init__(**kwargs)

        self.lamb = lamb
        self.scale_loss = scale_loss
        self.reset_extra_params = reset_extra_params

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(BarlowTwins, BarlowTwins).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("barlow_twins")

        # projector
        parser.add_argument("--output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--lamb", type=float, default=5e-3)
        parser.add_argument("--scale_loss", type=float, default=0.025)

        parser.add_argument("--reset_extra_params", action="store_true")

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        return super().learnable_params + self.extra_learnable_params
    
    @property
    def extra_learnable_params(self) -> List[dict]:
        return [
            {
                "name": "projector",
                "params": self.projector.parameters()
            },
        ]

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)

        if self.reset_extra_params:
            for model in [self.projector]:
                for layer in model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

    def forward(self, X, *args, **kwargs):
        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        loss_cls = out["loss"]

        feats1, feats2 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        # ------- barlow twins loss -------
        barlow_loss = barlow_loss_func(z1, z2, lamb=self.lamb, scale_loss=self.scale_loss)

        loss = out["loss"] + barlow_loss

        if self.online_eval:
            self.log_dict(
                {
                    "train_barlow_loss": barlow_loss,
                    "train_cls_loss": out["loss"],
                    "train_total_loss": loss,
                },
                sync_dist=True
            )

        out.update(
            {
                "loss": loss,
                "loss_cls": loss_cls,
                "loss_ssl": barlow_loss,
                "z": [z1, z2]
            }
        )

        return out
