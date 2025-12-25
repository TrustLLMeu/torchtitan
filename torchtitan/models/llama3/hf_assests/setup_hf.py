# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil


def copy_and_overwrite_model_config(model, model_args, dst_path: str):
    # check where am i, and copy these file to dst path:
    #  configuration_staging_llama.py and  modeling_staging_llama.py
    # to the dst path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_config_path = os.path.join(current_dir, "configuration_staging_llama.py")
    src_modeling_path = os.path.join(current_dir, "modeling_staging_llama.py")

    # separate paths for the files in dst
    dst_config_py_path = os.path.join(dst_path, "configuration_staging_llama.py")
    dst_modeling_py_path = os.path.join(dst_path, "modeling_staging_llama.py")
    dst_config_json_path = os.path.join(dst_path, "config.json")

    new_config = overwrite_config(model, model_args)

    if os.path.exists(dst_path):
        shutil.copy(src_config_path, dst_config_py_path)
        shutil.copy(src_modeling_path, dst_modeling_py_path)
        with open(dst_config_json_path, "w") as f:
            json.dump(new_config, f, indent=4)


def overwrite_config(model, model_args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = json.load(open(os.path.join(current_dir, "config.json")))

    attention = model.layers["0"].attention
    ffn = model.layers["0"].feed_forward

    default_config["num_hidden_layers"] = model.n_layers
    default_config["vocab_size"] = model.vocab_size
    default_config["rms_norm_eps"] = model_args.norm_eps
    default_config["qk_norm"] = model_args.qk_norm
    default_config["norm_everywhere"] = model_args.norm_everywhere
    default_config["max_position_embeddings"] = model_args.max_seq_len

    default_config["num_attention_heads"] = attention.n_heads
    default_config["num_key_value_heads"] = attention.n_kv_heads
    default_config["head_dim"] = attention.head_dim
    default_config["rope_theta"] = model_args.rope_theta

    default_config["intermediate_size"] = ffn.hidden_dim
    default_config["hidden_size"] = ffn.dim

    return default_config
