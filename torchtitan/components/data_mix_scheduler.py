# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.config import JobConfig

__all__ = [
    "DataMixScheduler",
]


class DataMixScheduler:
    """
    Lets make this stateless for now, to make the change of data mix easier.

    """

    def __init__(
        self,
        dataloader,
        mixing_configs,
    ):
        self.dataloader = dataloader
        self.mixing_configs = mixing_configs
        self.step_milestones = sorted(mixing_configs.keys(), reverse=True)

    def get_weights_at_step(self, current_step: int):
        for step in self.step_milestones:
            if current_step >= step:
                return self.mixing_configs[step]
        # In theory this should never happen since we assume
        # there is at least a step 0, but just in case:
        # fall back to the earliest config
        first_step = self.step_milestones[0]
        return self.mixing_configs[first_step]

    def step(self, current_step: int):
        current_weights = self.get_weights_at_step(current_step)
        self.dataloader.dataset.set_weights(current_weights)


def build_data_mix_scheduler(dataloader: BaseDataLoader, job_config: JobConfig):

    mixing_scheduler_configs = job_config.training.data_mixing_scheduler_configs
    mixing_configs = None
    if mixing_scheduler_configs:
        if os.path.isfile(mixing_scheduler_configs):
            try:
                mixing_configs = json.load(open(mixing_scheduler_configs))
                mixing_configs = {int(k): v for k, v in mixing_configs.items()}
            except Exception as e:
                pass

    """
    mixing_configs should be organized like:
    {
        0: [weights_for_dataset_0, weights_for_dataset_1, ...],
        500: [weights_for_dataset_0, weights_for_dataset_1, ...],
        step: [weights_for_dataset_0, weights_for_dataset_1, ...],
    }
    """
    if mixing_configs is None:
        mixing_configs = {
            0: dataloader.dataset.weights,
        }
    else:
        assert (
            0 in mixing_configs
        ), "mixing_configs must contain at least one entry for step 0"

        for step, weights in mixing_configs.items():
            assert len(weights) == len(dataloader.dataset.datasets), (
                f"weights must have the same length as datasets get len(datasets) = "
                f"{len(dataloader.dataset.datasets)} and len(weights) = "
                f"{len(weights)}"
            )

    return DataMixScheduler(dataloader, mixing_configs)
