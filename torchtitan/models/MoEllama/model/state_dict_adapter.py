# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any
from torch.distributed.tensor import DTensor

logger = logging.getLogger()

from torchtitan.models.utils import MoEStateDictAdapter

from .args import MoEModelArgs


class MoEllamaStateDictAdapter(MoEStateDictAdapter):
    def __init__(
        self,
        model_args: MoEModelArgs,
        hf_assets_path: str | None,
    ):
        super().__init__(model_args, hf_assets_path)

        self.model_args = model_args
        self.hf_assets_path = hf_assets_path
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
            "model.layers.{}.mlp.router.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.expert_bias": "layers.{}.moe.expert_bias",
            "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
        }

    # HuggingFace permutation function (exact copy from their conversion script)
    def _permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
            .clone()
        )

    def _reverse_permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        n_heads = self.model_args.n_heads
        n_kv_heads = (
            self.model_args.n_kv_heads
            if self.model_args.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_args.dim
        head_dim = dim // n_heads
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "load_balance_loss" in key:
                continue
            if "tokens_per_expert" in key:
                continue
            if "router_entropy" in key:
                continue
            if "acc_fwd_times" in key:
                continue

            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if "moe.experts" in key:
                    # Store the GroupedExperts Weight metadata for from_hf()
                    if isinstance(value, DTensor):
                        self.grouped_expert_weight_placements[abstract_key] = (
                            value.placements
                        )
                        self.grouped_expert_weight_shape[abstract_key] = value.shape

                        # Split GroupedExperts weight to local individual expert weights
                        local_expert_fqn = self._get_local_experts_weights(
                            new_key,
                            abstract_key,
                            layer_num,
                            value,
                        )
                        hf_state_dict.update(local_expert_fqn)

                    else:
                        # keep this path for offline conversion
                        split_values = self._split_experts_weights(
                            value, self.model_args.moe_args.num_experts
                        )

                        for expert_num in range(self.model_args.moe_args.num_experts):
                            expert_new_key = new_key.format(layer_num, expert_num)
                            hf_state_dict[expert_new_key] = split_values[
                                expert_num
                            ].squeeze()
                else:
                    if abstract_key == "layers.{}.attention.wq.weight":
                        value = self._permute(value, n_heads)
                    if abstract_key == "layers.{}.attention.wk.weight":
                        key_value_dim = head_dim * n_kv_heads
                        value = self._permute(value, n_kv_heads, key_value_dim, dim)

                    if new_key is None:
                        continue
                    new_key = new_key.format(layer_num)
                    hf_state_dict[new_key] = value
            else:
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("from_hf is not implemented for MoEllama")
        n_heads = self.model_args.n_heads
        n_kv_heads = (
            self.model_args.n_kv_heads
            if self.model_args.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_args.dim
        head_dim = dim // n_heads
        state_dict: dict[str, Any] = {}

        # Temporary storage for HF MoE expert weights before regrouping:
        # keyed by (layer_num, param_kind) where param_kind in {"w1","w2","w3"}
        grouped_experts: dict[tuple[str, str], dict[int, Any]] = defaultdict(dict)

        num_experts = self.model_args.moe_args.num_experts

        for key, value in hf_state_dict.items():
            if "layers" in key:
                # collect all numeric indices (layer, expert, ...)
                nums = re.findall(r"\d+", key)
                if not nums:
                    # shouldn't happen, but be defensive
                    continue
                layer_num = nums[0]

                # Generalise *all* numeric indices so MoE patterns match too
                abstract_key = re.sub(r"(\d+)", "{}", key)

                # --- MoE experts (two indices: layer + expert) ---
                if ".mlp.experts." in key:
                    if len(nums) < 2:
                        logger.warning(
                            "Found MoE expert key without expert index: %s", key
                        )
                        continue
                    expert_num = int(nums[1])

                    new_key_template = self.from_hf_map.get(abstract_key, None)
                    if new_key_template is None:
                        # nothing to do (unknown key)
                        continue

                    # Identify which of w1 / w2 / w3 this is
                    if ".gate_proj.weight" in key:
                        param_kind = "w1"
                    elif ".up_proj.weight" in key:
                        param_kind = "w3"
                    elif ".down_proj.weight" in key:
                        param_kind = "w2"
                    else:
                        logger.warning(
                            "Unrecognized MoE expert projection key: %s", key
                        )
                        continue

                    grouped_experts[(layer_num, param_kind)][expert_num] = value
                    continue  # don't write directly into state_dict yet

                # --- Non-expert layer parameters (attention, FFN, router, shared_experts, etc.) ---
                new_key_template = self.from_hf_map.get(abstract_key, None)
                if new_key_template is None:
                    continue

                # Undo RoPE permutations
                if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                    value = self._reverse_permute(value, n_heads)
                elif abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(value, n_kv_heads, key_value_dim, dim)

                if new_key_template is None:
                    # e.g. rotary_emb.inv_freq
                    continue

                new_key = new_key_template.format(layer_num)
                state_dict[new_key] = value
            else:
                new_key = self.from_hf_map.get(key, None)
                if new_key is None:
                    continue
                state_dict[new_key] = value

        # --- Rebuild grouped-expert weights from per-expert HF tensors ---
        for (layer_num, param_kind), experts_dict in grouped_experts.items():
            # Expect indices [0, num_experts-1]; warn if incomplete
            missing = [i for i in range(num_experts) if i not in experts_dict]
            if missing:
                logger.warning(
                    "Missing experts %s for layer %s param %s when regrouping MoE weights",
                    missing,
                    layer_num,
                    param_kind,
                )

            # Order by expert index; only keep those we actually have
            ordered_expert_ids = sorted(experts_dict.keys())
            ordered_weights = [experts_dict[i] for i in ordered_expert_ids]

            # Stack along expert dimension to invert _split_experts_weights
            grouped_weight = torch.stack(ordered_weights, dim=0)

            native_key = f"layers.{layer_num}.moe.experts.{param_kind}"
            state_dict[native_key] = grouped_weight

        return state_dict
