# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Sequence

from transformers import AutoConfig, AutoTokenizer

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger


class HFTokenizer(Tokenizer):
    """
    Tokenizing and encoding/decoding text using a HuggingFace tokenizer.

    Args:
        tokenizer_dir (str): The directory containing the tokenizer's files.
    """

    def __init__(self, tokenizer_dir: str):
        super().__init__()
        assert os.path.exists(
            tokenizer_dir,
        ), f"The tokenizer's directory does not exist: {tokenizer_dir}"
        assert os.path.isdir(tokenizer_dir), tokenizer_dir

        self.config = AutoConfig.from_pretrained(tokenizer_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        self._n_words: int = self.tokenizer.vocab_size
        # BOS / EOS token IDs
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id
        self.pad_id: int = self.tokenizer.pad_token_id
        logger.info(
            f"HFTokenizer built: #words {self.n_words}, BOS ID {self.bos_id}, EOS ID {self.eos_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
    ) -> list[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            list[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.tokenizer.encode(s, add_special_tokens=False)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(t)


def build_hf_tokenizer(job_config: JobConfig) -> HFTokenizer:
    return HFTokenizer(job_config.model.tokenizer_path)
