import functools
from typing import Optional

import torch
from torch.distributed.tensor import DTensor, Replicate
import torch.nn as nn

INIT_FN_TYPES = ["trunc_normal", "normal", "orthogonal", "scion_normal"]


# Deliberately throw away `mean` and `std` arguments.
def _wrap_ignore_mean_std(fn):

    @functools.wraps(fn)
    def wrapped_fn(tensor, mean=None, std=None, *args, **kwargs):
        return fn(tensor, *args, **kwargs)

    return wrapped_fn


# Deliberately throw away the `generator` argument.
def _wrap_ignore_generator(fn):

    @functools.wraps(fn)
    def wrapped_fn(tensor, *args, generator=None, **kwargs):
        return fn(tensor, *args, **kwargs)

    return wrapped_fn


def orthogonal_(params, gain: float = 1.0, generator: Optional[torch.Generator] = None):
    if not isinstance(params.data, DTensor):
        return nn.init.orthogonal_(params, gain=gain, generator=generator)
    else:
        temp_tensor = torch.empty(params.shape)  # full shape
        torch.nn.init.orthogonal_(temp_tensor, gain=gain, generator=generator)

        # Create a replicated DTensor
        replicated = DTensor.from_local(
            temp_tensor,
            placements=[Replicate()] * params.device_mesh.ndim,
            device_mesh=params.device_mesh,
        )

        # Reshard to match original dtensor's placement
        resharded = replicated.redistribute(placements=params.placements)

        # Copy values to original dtensor
        params.copy_(resharded)
        return params


def scion_normal_(
        tensor,
        mean: float = 0.0,
        std: float = 1.0,
        norm_axis: int = 1,
        eps: float = 1e-12,
        generator: Optional[torch.Generator] = None,
):
    nn.init.normal_(
        tensor,
        mean=mean,
        std=std,
        generator=generator,
    )
    with torch.no_grad():
        divisor = torch.rsqrt(tensor.pow(2).sum(axis=norm_axis, keepdim=True) + eps)
        tensor.mul_(divisor)


def build_init_fn(init_fn_type: str):
    """
    Builds the specified initialization function based on `init_fn_type`.

    Args:
        init_fn_type (str): The type of normalization layer to build.
            Supported types: trunc_normal, normal

    Returns:
        The built initialization function.

    Raises:
        NotImplementedError: If an unknown `init_fn_type` is provided.
    """
    init_fn_type = init_fn_type.lower()  # Normalize to lowercase

    if init_fn_type == "trunc_normal":
        return nn.init.trunc_normal_
    elif init_fn_type == "normal":
        return nn.init.normal_
    elif init_fn_type == "zeros":
        return _wrap_ignore_generator(_wrap_ignore_mean_std(nn.init.zeros_))
    elif init_fn_type == "orthogonal":
        return _wrap_ignore_mean_std(orthogonal_)
    elif init_fn_type == "scion_normal":
        return scion_normal_
    else:
        raise NotImplementedError(f"Unknown `init_fn_type`: '{init_fn_type}'")
