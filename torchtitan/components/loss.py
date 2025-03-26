# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Union

import torch

from torchtitan.config_manager import JobConfig
from torchtitan.models.llama.model import MTPInputs, TransformerInputs


def cross_entropy_loss(
        pred: Union[torch.Tensor, list[torch.Tensor], TransformerInputs],
        labels: torch.Tensor,
) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    if isinstance(pred, dict):
        pred = pred["tokens_list"]
    if isinstance(pred, list):
        pred = pred[0]
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


# TODO: compiling loss function causes CUDA errors, turning off for now
# compiled_cross_entropy_loss = torch.compile(cross_entropy_loss)


def multi_token_cross_entropy_loss(
        preds: Union[list[torch.Tensor], MTPInputs],
        labels: torch.Tensor,
        loss_fn: Callable,
        job_config: JobConfig,
) -> torch.Tensor:
    """Multi-token cross-entropy loss function for Transformer model training.

    Based on DeepSeek-V3 technical report: https://arxiv.org/abs/2412.19437.
    """
    if isinstance(preds, dict):
        preds = preds["tokens_list"]
    assert isinstance(preds, list)
    clm_loss = loss_fn(preds[0], labels[:, :job_config.training.seq_len])

    mtp_loss = 0
    for (label_offset, pred) in enumerate(preds[1:], 1):
        loss = loss_fn(
            pred,
            labels[:, label_offset:label_offset + job_config.training.seq_len],
        )
        # Take average over MTP predictions.
        loss = loss / job_config.training.num_mtp_tokens
        mtp_loss = mtp_loss + loss
    return clm_loss + mtp_loss * job_config.training.mtp_loss_weight
