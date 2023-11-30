import argparse
from copy import deepcopy
from typing import Any, Sequence
import torch


def lwf_wrapper(Method=object):
    class LWFWrapper(Method):
        def __init__(
            self,
            distill_lamb: float,
            **kwargs
        ) -> None:
            super().__init__(**kwargs)
            
            assert hasattr(self, "heads"), "'heads' are required for LWF"

            self.distill_lamb = distill_lamb
            self.frozen_encoder = deepcopy(self.encoder)

        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("lwf_distiller")

            parser.add_argument("--distill_lamb", type=float, default=1.0)

            return parent_parser
        
        def on_train_start(self):
            super().on_train_start()

            if self.current_task_idx > 0:
                self.frozen_encoder = deepcopy(self.encoder)
                for pg in self.frozen_encoder.parameters():
                    pg.requires_grad = False

                    
        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            _, (X1, X2), _ = batch[f"task{self.current_task_idx}"]

            out = super().training_step(batch, batch_idx)
            
            frozen_feats1 = self.frozen_encoder(X1)
            frozen_feats2 = self.frozen_encoder(X2)
            feats1, feats2 = out["feats"]

            logits, frozen_logits = [], []
            for head in self.heads[:-1]:
                logits.append(head(feats1))
                logits.append(head(feats2))
                frozen_logits.append(head(frozen_feats1))
                frozen_logits.append(head(frozen_feats2))

            distill_loss = self.cross_entropy(torch.cat(logits, dim=1),
                                              torch.cat(frozen_logits, dim=1),
                                              exp=0.5)
                            
            loss = out['loss'] + self.distill_lamb * distill_loss
            
            self.log("train_distill_loss", distill_loss, on_epoch=True)
            
            return loss
        
        def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
            """Calculates cross-entropy with temperature scaling"""
            out = torch.nn.functional.softmax(outputs, dim=1)
            tar = torch.nn.functional.softmax(targets, dim=1)
            if exp != 1:
                out = out.pow(exp)
                out = out / out.sum(1).view(-1, 1).expand_as(out)
                tar = tar.pow(exp)
                tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
            out = out + eps / out.size(1)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            ce = -(tar * out.log()).sum(1)
            if size_average:
                ce = ce.mean()
            return ce
    
    return LWFWrapper
