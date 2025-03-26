from typing import NotRequired, TypedDict

import torch


class TransformerInputsDict(TypedDict):
    tokens_list: list[torch.Tensor | None] | torch.Tensor


TransformerInputs = torch.Tensor | TransformerInputsDict


class MTPInputsDict(TransformerInputsDict):
    orig_tokens: NotRequired[torch.Tensor | None]
    prev_embed: NotRequired[torch.Tensor | None]


MTPInputs = torch.Tensor | MTPInputsDict


class MoEInputsDict(TransformerInputsDict):
    tokens_list: list[torch.Tensor | None] | torch.Tensor
    aux_loss: NotRequired[torch.Tensor]
    moe_entropy_per_layer: NotRequired[torch.Tensor]


MoEInputs = torch.Tensor | MoEInputsDict
