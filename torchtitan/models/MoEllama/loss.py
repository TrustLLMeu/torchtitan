import torch

from torchtitan.components.loss import LossFunction
from torchtitan.models.MoEllama.moellama import MoEInputsDict


def moe_loss(
        pred: MoEInputsDict,
        labels: torch.Tensor,
        loss_fn: LossFunction,
) -> torch.Tensor:
    """Sequence-wise auxiliary loss-enhanced loss function for MoE Transformer
    model training.
    """
    assert isinstance(pred, dict)
    loss = loss_fn(pred, labels)
    loss += pred["aux_loss"]
    return loss
