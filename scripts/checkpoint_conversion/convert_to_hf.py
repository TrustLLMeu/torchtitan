# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This new code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this new tree.

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torchtitan.protocols.train_spec as train_spec_module
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP
from dataclasses import is_dataclass, fields
import torchtitan.models  # noqa: F401

import json
import shutil
import os


def update_dataclass_from_dict(target, data):
    """
    Recursively update `target` dataclass in-place using values from `data` (a dict).
    - Only updates fields that exist on `target`.
    - If a field on `target` is a dataclass and the corresponding value in `data`
      is a dict, it recurses instead of overwriting the dataclass instance.
    """
    if not is_dataclass(target):
        raise TypeError("target must be a dataclass instance")

    for f in fields(target):
        name = f.name
        if name not in data:
            continue

        new_val = data[name]
        old_val = getattr(target, name)

        # If the existing value is a dataclass and new_val is a dict, recurse
        if is_dataclass(old_val) and isinstance(new_val, dict):
            update_dataclass_from_dict(old_val, new_val)
        else:
            # Otherwise just overwrite the value
            setattr(target, name, new_val)


def try_to_copy_tokenizer(output_dir, hf_assets_path):
    """
    if these files exist in the hf_assets_path, then copy them to the output_dir
    """

    tokenizer_assests_lists = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    for asset in tokenizer_assests_lists:
        if os.path.exists(os.path.join(hf_assets_path, asset)):
            shutil.copy(
                os.path.join(hf_assets_path, asset), os.path.join(output_dir, asset)
            )


@torch.inference_mode()
def convert_to_hf(
    input_dir,
    output_dir,
    actual_model_args,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
):
    # load model and model args so that we can get the state dict shape
    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]

    actual_model_args = json.load(open(actual_model_args, "r"))
    update_dataclass_from_dict(model_args, actual_model_args)

    with torch.device("cpu"):
        actual_model = train_spec.model_cls(model_args)
    model = ModelWrapper(actual_model)

    sd_adapter = train_spec.state_dict_adapter(model_args, hf_assets_path)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."

    # allocate state dict memory with empty weights to load checkpoint
    state_dict = model._get_state_dict()
    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
    )

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # map and apply export dtype if needed
    old_dtype = TORCH_DTYPE_MAP[export_dtype]
    if old_dtype != torch.float32:
        hf_state_dict = {k: v.to(old_dtype) for k, v in hf_state_dict.items()}

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )

    try_to_copy_tokenizer(output_dir, hf_assets_path)

    if train_spec.hf_assets_setup_fn is not None:
        train_spec.hf_assets_setup_fn(actual_model, model_args, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with DCP weights."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
    )
    parser.add_argument("model_args_file", type=str, help="Model args file.")
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory. This is used to get the model.safetensors.index.json mapping",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for HF checkpoint (default: float32)",
    )
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_args_file,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
    )
