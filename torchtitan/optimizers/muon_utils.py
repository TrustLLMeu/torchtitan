from typing import cast, List, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import (
    _device_dtype_check_for_fused,
    _get_scalar_dtype,
)


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (torch.linalg.norm(X) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def _init_adamw_group(
    self,
    group,
    params_with_grad,
    grads,
    amsgrad,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
):
    has_complex = False
    for p in group["params"]:
        if p.grad is None and not self.state[p]["use_muon"]:
            continue
        has_complex |= torch.is_complex(p)
        params_with_grad.append(p)
        if p.grad.is_sparse:
            raise RuntimeError("AdamW does not support sparse gradients")
        grads.append(p.grad)

        state = self.state[p]

        # State initialization
        if "exp_avg" not in state:
            if group["adamw_fused"]:
                _device_dtype_check_for_fused(p)
            # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
            # This is because kernel launches are costly on CUDA and XLA.
            state["step"] = (
                torch.zeros(
                    (),
                    dtype=_get_scalar_dtype(is_fused=group["adamw_fused"]),
                    device=p.device,
                )
                if group["adamw_capturable"] or group["adamw_fused"]
                else torch.tensor(0.0, dtype=_get_scalar_dtype())
            )
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state["exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state["max_exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])

        if group["adamw_amsgrad"]:
            max_exp_avg_sqs.append(state["max_exp_avg_sq"])
        if group["adamw_differentiable"] and state["step"].requires_grad:
            raise RuntimeError(
                "`requires_grad` is not supported for `step` in differentiable mode"
            )

        # Foreach without capturable does not support a tensor lr
        if (
            group["adamw_foreach"]
            and isinstance(group["adamw_lr"], Tensor)
            and not group["adamw_capturable"]
        ):
            raise RuntimeError(
                "lr as a Tensor is not supported for capturable=False and foreach=True"
            )

        state_steps.append(state["step"])
    return has_complex


@torch.no_grad()
def update_adamw(self):
    """
    Optimized AdamW implementation using PyTorch's _functional.adamw or foreach operations.
    """
    for group in self.param_groups:
        params_with_grad: List[Tensor] = []
        grads: List[Tensor] = []
        exp_avgs: List[Tensor] = []
        exp_avg_sqs: List[Tensor] = []
        max_exp_avg_sqs: List[Tensor] = []
        state_steps: List[Tensor] = []
        amsgrad: bool = group["adamw_amsgrad"]
        beta1, beta2 = cast(Tuple[float, float], group["adamw_betas"])

        has_complex = _init_adamw_group(
            self,
            group,
            params_with_grad,
            grads,
            amsgrad,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
        )

        torch.optim._functional.adamw(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=group["adamw_lr"],
            weight_decay=group["adamw_weight_decay"],
            eps=group["adamw_eps"],
            maximize=group["adamw_maximize"],
            foreach=group["adamw_foreach"],
            capturable=group["adamw_capturable"],
            differentiable=group["adamw_differentiable"],
            fused=group["adamw_fused"],
            grad_scale=getattr(self, "grad_scale", None),
            found_inf=getattr(self, "found_inf", None),
            has_complex=has_complex,
        )
