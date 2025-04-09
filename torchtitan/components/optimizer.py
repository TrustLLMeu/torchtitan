# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import functools
import re
from typing import Any, Generic, Iterator, TypeVar

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from torchtitan.components.ft import FTManager, has_torchft
from torchtitan.config_manager import JobConfig
from torchtitan.optimizers import DistributedMuon, DistributedMuonV2, Muon, Scion

__all__ = [
    "OptimizersContainer",
    "build_optimizers",
]


if has_torchft:
    import torchft as ft


T = TypeVar("T", bound=Optimizer)


def _extract_param_groups(
    model: torch.nn.Module,
    optimizer_config: dict[str, Any] | None = None,
):
    param_groups_config: list[dict[str, Any]] | None = (
        optimizer_config.pop("param_groups", None)
        if optimizer_config is not None
        else None
    )

    if optimizer_config is None or not param_groups_config:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = []
        param_dict = OrderedDict((n, p) for n, p in model.named_parameters() if p.requires_grad)

        for param_group_config in param_groups_config:
            str_match = param_group_config.pop("param_str_match")
            filter_fn = functools.partial(re.search, str_match)
            param_names = [n for n in param_dict.keys() if filter_fn(n)]
            group_params = {
                "params": [param_dict.pop(n) for n in param_names],
                "param_names": param_names,
            }
            assert len(group_params["params"]) == len(group_params["param_names"])
            group_params.update(param_group_config)
            params.append(group_params)

        param_names = list(param_dict.keys())
        params.insert(
            0,
            {"params": [param_dict.pop(n) for n in param_names], "param_names": param_names},
        )
        assert not param_dict
    return params


class OptimizersContainer(Optimizer, Stateful, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    optimizers: list[T]
    model_parts: list[nn.Module]

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        param_groups_config = optimizer_kwargs.get("param_groups", None)
        for model in self.model_parts:
            # copy parts we will pop from to preserve settings across model parts
            kwargs = optimizer_kwargs.copy()
            if "param_groups" in optimizer_kwargs:
                kwargs["param_groups"] = (
                    param_groups_config.copy()
                    if param_groups_config is not None
                    else None
                )

            # for muon, we need to pass model as well

            is_muon = issubclass(optimizer_cls, (Muon, DistributedMuon, DistributedMuonV2))
            extra_kwargs = kwargs.pop("extra_kwargs")

            params = _extract_param_groups(model, kwargs)
            if is_muon:
                kwargs.update(extra_kwargs)
                self.optimizers.append(optimizer_cls(params, model, **kwargs))
            else:
                self.optimizers.append(optimizer_cls(params, **kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        # Do not separately save the external settings in
        # optimizer defaults.
        del optimizer_kwargs["param_groups"]
        optimizer_kwargs.update(optimizer_kwargs.pop("extra_kwargs", {}))
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            if not (isinstance(optimizer, Scion) and optimizer.is_light):
                optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)


class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.model_parts = model_parts

        optim_dict = {}
        for model in self.model_parts:
            for p in model.parameters():
                if p.requires_grad:
                    optim_dict[p] = optimizer_cls([p], **optimizer_kwargs)
                all_params.append(p)

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())

        self._validate_length(
            sum(len(list(model.parameters())) for model in self.model_parts)
        )
        self._post_init(all_params, optimizer_kwargs)

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass


class FTOptimizersContainer(OptimizersContainer):
    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        ft_manager: "ft.Manager",
    ) -> None:
        super().__init__(model_parts, optimizer_cls, optimizer_kwargs)

        # Force to initialize the optimizer state so that `optim.step()`
        # won't be called by state_dict() and load_state_dict().
        _ = {
            k: v
            for sd in map(get_optimizer_state_dict, model_parts, self.optimizers)
            for k, v in sd.items()
        }
        self.cache_state_dict: dict[str, Any] = {}
        self._ft_optimizer = ft.Optimizer(ft_manager, self)
        self._call_from_ft: bool = False

    def init_cache_state_dict(self) -> None:
        self.cache_state_dict = super().state_dict()

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # We have to invalidate the `cache_state_dict` because optimizer uses
        # assign instead of copy when doing `load_state_dict()`. Without
        # invalidating the `cache_state_dict`, there will be memory leakage.
        self.cache_state_dict = {}
        super().load_state_dict(state_dict)
        self.init_cache_state_dict()

    def step(self, *args, **kwargs) -> None:
        """Calling the correct step() depending on the caller.

        TorchFT's OptimizerWrapper.step() is designed to be callled only once
        per train step per ft.Manager regardless how many optimizers are used.
        Hence we will need to appropriately dispatch the call.
        """
        if self._call_from_ft:
            super().step(*args, **kwargs)
        else:
            self._call_from_ft = True
            self._ft_optimizer.step(*args, **kwargs)
            self._call_from_ft = False

    def zero_grad(self, *args, **kwargs) -> None:
        """Calling the correct zero_grad() depending on the caller.

        Check the comment in ``step()``.
        """
        if self._call_from_ft:
            super().zero_grad(*args, **kwargs)
        else:
            self._call_from_ft = True
            self._ft_optimizer.zero_grad(*args, **kwargs)
            self._call_from_ft = False


def build_optimizers(
    model_parts: list[nn.Module],
    job_config: JobConfig,
    ft_manager: FTManager,
    extra_kwargs: dict[str, Any],
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``job_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        job_config (JobConfig): Job config containing the optimizer name and parameters.
    """
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    if optim_in_bwd and job_config.parallelism.pipeline_parallel_degree > 1:
        raise NotImplementedError(
            "Optimizers in backward is not supported with pipeline parallelism."
        )
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    eps = job_config.optimizer.eps
    weight_decay = job_config.optimizer.weight_decay

    if name in ["Adam", "AdamW", "Muon", "DistributedMuon", "DistributedMuonV2"]:
        optim_implementation = job_config.optimizer.implementation
        assert optim_implementation in ["fused", "foreach", "for-loop"]

        fused = optim_implementation == "fused"
        foreach = optim_implementation == "foreach"

        optimizer_kwargs = {
            "lr": lr,
            "eps": eps,
            "betas": (0.9, 0.95),
            "weight_decay": weight_decay,
            "fused": fused,
            "foreach": foreach,
        }
    elif name == "Scion":
        backend_steps = job_config.optimizer.backend_steps
        momentum = job_config.optimizer.momentum
        nesterov = job_config.optimizer.nesterov
        is_light = job_config.optimizer.is_light
        is_unconstrained = job_config.optimizer.is_unconstrained

        optimizer_kwargs = {
            "is_light": is_light,
            "is_unconstrained": is_unconstrained,
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "eps": eps,
            "norm_factor": "spectral",
            "backend": "newtonschulz5",
            "backend_steps": backend_steps,
        }
    else:
        raise NotImplementedError(f"Optimizer {name} not added.")

    # Configure parameter group settings
    embed_lr = job_config.optimizer.embed_lr
    embed_str_match = job_config.optimizer.embed_str_match
    if embed_lr is not None and embed_str_match:
        param_groups_config = optimizer_kwargs.setdefault("param_groups", [])
        param_groups_config.append({
            "param_str_match": embed_str_match,
            "lr": embed_lr,
            "norm_factor": "embed_sqrt",
            "backend": "identity",
        })
    unembed_lr = job_config.optimizer.unembed_lr
    unembed_str_match = job_config.optimizer.unembed_str_match
    if unembed_lr is not None and unembed_str_match:
        param_groups_config = optimizer_kwargs.setdefault("param_groups", [])
        param_groups_config.append({
            "param_str_match": unembed_str_match,
            "lr": unembed_lr,
            "norm_factor": "unembed_sqrt",
            "backend": "identity",
        })

    optimizer_kwargs["extra_kwargs"] = extra_kwargs

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "Muon": Muon,
        "DistributedMuon": DistributedMuon,
        "DistributedMuonV2": DistributedMuonV2,
        "Scion": Scion,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]

    if optim_in_bwd and ft_manager.enabled:
        raise ValueError("TorchFT is not supported with optimizers in backward.")
    elif optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )
    elif ft_manager.enabled:
        return FTOptimizersContainer(
            model_parts, optimizer_cls, optimizer_kwargs, ft_manager.manager
        )
    else:
        return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)
