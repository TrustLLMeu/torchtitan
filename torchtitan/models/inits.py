import functools
import math
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor
import torch.nn as nn

INIT_FN_TYPES = [
    "trunc_normal",
    "normal",
    "orthogonal",
    "scaled_orthogonal",
    "scion_normal",
    "scion_normal_input",
    "scion_normal_output",
]


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


def orthogonal_(param, gain: float = 1.0, generator: Optional[torch.Generator] = None):
    with torch.no_grad():
        if not isinstance(param.data, DTensor):
            return nn.init.orthogonal_(param, gain=gain, generator=generator)

        # rank 0 makes the full tensor
        full_shape = tuple(param.shape)
        device = param.device
        if dist.get_rank() == 0:
            temp = torch.empty(full_shape, device=device)
            nn.init.orthogonal_(temp, gain=gain, generator=generator)
        else:
            # allocate an uninitialized placeholder
            temp = torch.empty(full_shape, device=device)

        # broadcast the full tensor from rank 0 to everyone
        dist.broadcast(temp, src=0)

        # shard it into a DTensor matching param’s placements/device_mesh
        local_shard = distribute_tensor(
            temp,
            placements=param.placements,
            device_mesh=param.device_mesh
        )

        # copy into the parameter
        param.data.copy_(local_shard)
        return param


def scaled_orthogonal_(param, gain: float = 1.0, generator: Optional[torch.Generator] = None):
    """
    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.xavier_uniform_(w.T, ...)``.
    """
    with torch.no_grad():
        assert param.ndim == 2, \
            "Fan in and fan out can not be computed for tensor with other than 2 dimensions"
        fan_out, fan_in = param.shape
        scale = math.sqrt(fan_out / fan_in)
        gain *= scale

    return orthogonal_(param, gain, generator)


def scion_normal_(
        tensor,
        mean: float = 0.0,
        std: float = 1.0,
        norm_axis: int = 1,
        eps: float = 1e-12,
        scale_type: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
):
    assert tensor.ndim == 2, "Tensor for scion_normal_ init must have 2 dimensions"
    nn.init.normal_(
        tensor,
        mean=mean,
        std=std,
        generator=generator,
    )
    if scale_type is None:
        scale = 1.0
    elif scale_type == "input":
        scale = math.sqrt(tensor.shape[norm_axis])
    elif scale_type == "output":
        scale = 1 / math.sqrt(tensor.shape[norm_axis])
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    with torch.no_grad():
        scale = scale * torch.rsqrt(tensor.pow(2).sum(axis=norm_axis, keepdim=True) + eps)
        tensor.mul_(scale)


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

    def _wrap_orthogonal(fn):

        @functools.wraps(fn)
        def wrapped_fn(tensor, mean=None, std=1, *args, **kwargs):
            return fn(tensor, gain=std, *args, **kwargs)

        return wrapped_fn

    if init_fn_type == "trunc_normal":
        return nn.init.trunc_normal_
    elif init_fn_type == "normal":
        return nn.init.normal_
    elif init_fn_type == "zeros":
        return _wrap_ignore_generator(_wrap_ignore_mean_std(nn.init.zeros_))
    elif init_fn_type == "orthogonal":
        return _wrap_orthogonal(orthogonal_)
    elif init_fn_type == "scaled_orthogonal":
        return _wrap_orthogonal(scaled_orthogonal_)
    elif init_fn_type == "scion_normal":
        return scion_normal_
    elif init_fn_type == "scion_normal_input":
        return functools.partial(
            scion_normal_,
            scale_type="input",
            norm_axis=1,
        )
    elif init_fn_type == "scion_normal_output":
        return functools.partial(
            scion_normal_,
            scale_type="output",
            norm_axis=1,
        )
    else:
        raise NotImplementedError(f"Unknown `init_fn_type`: '{init_fn_type}'")
