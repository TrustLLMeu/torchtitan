# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger


def _load_c4_dataset(dataset_path: str):
    """Load C4 dataset with default configuration."""
    return _load_simple_dataset(
        dataset_path,
        dataset_name="en",
        dataset_files=None,
        dataset_split="train",
        dataset_streaming=True,
    )


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return _process_simple_text(sample, "text")


def _load_simple_dataset(
        dataset_path: str,
        dataset_name: Optional[str],
        dataset_files: Union[str, Sequence[str], None],
        dataset_split: str,
        dataset_streaming: bool,
):
    """Load a simple custom dataset with its configuration."""
    return load_dataset(
        dataset_path,
        name=dataset_name,
        data_files=dataset_files,
        split=dataset_split,
        streaming=dataset_streaming,
    )


def _process_simple_text(sample: dict[str, Any], key: str) -> str:
    """Process a simple custom dataset's sample text."""
    return sample[key]


@dataclass
class DatasetArgs:
    path: str
    name: Optional[str]
    files: Union[str, Sequence[str], None]
    split: str
    streaming: bool
    key: str


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    text_processor: Callable


# Add your dataset here here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetArgs(
        path="allenai/c4",
        name="en",
        files=None,
        split="train",
        streaming=True,
        key="text",
    ),
    "c4_test": DatasetArgs(
        path="tests/assets/c4_test",
        name=None,
        files=None,
        split="train",
        streaming=False,
        key="text",
    ),
    "fineweb": DatasetArgs(
        path="HuggingFaceFW/fineweb",
        name="default",
        files=None,
        split="train",
        streaming=True,
        key="text",
    ),
    "simple_custom": None,
}


def _validate_dataset(
    dataset_name: str,
    dataset_path: str,
    dataset_inner_name: Optional[str],
    dataset_files: Union[str, Sequence[str], None],
    dataset_split: str,
    dataset_streaming: bool,
    dataset_key: str,
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    if config is None:
        config = DatasetArgs(
            path=dataset_path,
            name=dataset_inner_name,
            files=dataset_files,
            split=dataset_split,
            streaming=dataset_streaming,
            key=dataset_key,
        )
    if not isinstance(config, DatasetConfig):
        assert isinstance(config, DatasetArgs)
        old_config = config
        config = DatasetConfig(
            path=old_config.path,
            loader=lambda path: _load_simple_dataset(
                path,
                old_config.name,
                old_config.files,
                old_config.split,
                old_config.streaming,
            ),
            text_processor=lambda sample: _process_simple_text(sample, old_config.key),
        )

    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.text_processor


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        dataset_inner_name: Optional[str] = None,
        dataset_files: Union[str, Sequence[str], None] = None,
        dataset_split: str = "train",
        dataset_streaming: bool = False,
        dataset_key: str = "text",
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_inner_name=dataset_inner_name,
            dataset_files=dataset_files,
            dataset_split=dataset_split,
            dataset_streaming=dataset_streaming,
            dataset_key=dataset_key,
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.infinite = infinite
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0

    def __len__(self):
        return len(self._data)

    def _get_data_iter(self):
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        it = iter(self._data)
        for _ in range(self._sample_idx):
            next(it)
        return it

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                self._sample_idx += 1
                yield sample_tokens

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self):
        return {"sample_idx": self._sample_idx}


class GreedyPackedDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset: IterableDataset,
        seq_len: int = 2048,
        infinite: bool = False,
        num_mtp_tokens: int = 0,
    ) -> None:
        self._data = dataset
        self.seq_len = seq_len
        self.infinite = infinite
        self.num_mtp_tokens = num_mtp_tokens

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: list[int] = []

    @property
    def dataset_name(self):
        return self._data.dataset_name

    def _get_data_iter(self):
        # We don't use the sample index because we defer skipping to the
        # sub-dataset.
        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len + self.num_mtp_tokens

        while True:
            for sample_tokens in self._get_data_iter():
                self._all_tokens.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield input, label

            if not self.infinite:
                logger.warning(f"Packed dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Packed dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]
        self._data.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {
            "token_buffer": self._all_tokens,
            "sample_idx": self._sample_idx,
            "dataset": self._data.state_dict(),
        }


def build_hf_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.batch_size
    seq_len = job_config.training.seq_len
    num_mtp_tokens = job_config.training.num_mtp_tokens
    dataset_inner_name = job_config.training.dataset_inner_name
    dataset_split = job_config.training.dataset_split
    dataset_files = job_config.training.dataset_files
    dataset_streaming = job_config.training.dataset_streaming
    dataset_key = job_config.training.dataset_key

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        dataset_inner_name=dataset_inner_name,
        dataset_files=dataset_files,
        dataset_split=dataset_split,
        dataset_streaming=dataset_streaming,
        dataset_key=dataset_key,
    )

    hf_ds = GreedyPackedDataset(
        dataset=hf_ds,
        seq_len=seq_len,
        infinite=infinite,
        num_mtp_tokens=num_mtp_tokens,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )
