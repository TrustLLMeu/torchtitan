from typing import Callable

import torch

from torchtitan.models.MoEllama.moellama import MoEInputs


def moe_loss(
        pred: MoEInputs,
        labels: torch.Tensor,
        loss_fn: Callable,
) -> torch.Tensor:
    """Sequence-wise auxiliary loss-enhanced loss function for MoE Transformer
    model training.
    """
    assert isinstance(pred, dict)
    loss = loss_fn(pred, labels)
    loss += pred["aux_loss"]
    return loss
