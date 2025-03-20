import tempfile
from typing import Dict, List, Optional

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger


class ByteTokenizer(Tokenizer):
    """
    Tokenize and encode/decode text using a byte-level tokenizer.

    Args:
        bos_token (str): String to use for representing the
            beginning-of-sequence token.
        eos_token (str): String to use for representing the
            end-of-sequence token.
        special_tokens (Optional[List[str]]): optional list of
            additional tokens to add
    """

    NUM_BYTE_VALUES = 256

    def __init__(
            self,
            bos_token: str = "〈BOS〉",
            eos_token: str = "〈EOS〉",
            special_tokens: Optional[List[str]] = None,
    ):
        # We don't need a file, so we hack the super constructor with a
        # tmpfile.
        with tempfile.NamedTemporaryFile() as f:
            super().__init__(f.name)

        if special_tokens is None:
            special_tokens = []
        if bos_token not in special_tokens:
            special_tokens.insert(0, bos_token)
        if eos_token not in special_tokens:
            special_tokens.insert(1, eos_token)

        self.special_tokens = {
            tok: i + ByteTokenizer.NUM_BYTE_VALUES
            for (i, tok) in enumerate(special_tokens)
        }
        self._bos_id = self.special_tokens[bos_token]
        self._eos_id = self.special_tokens[eos_token]
        self._pad_id = -1

        vocab_size = ByteTokenizer.NUM_BYTE_VALUES + len(self.special_tokens)
        self._n_words = vocab_size

        self._vocab = {chr(i): i for i in range(ByteTokenizer.NUM_BYTE_VALUES)}
        self._vocab.update(self.special_tokens)

        assert len(self._vocab) == vocab_size, (
            f"unexpected vocabulary size; make sure none of the specified "
            f"special tokens collide with the original "
            f"{ByteTokenizer.NUM_BYTE_VALUES} ASCII symbols"
        )

        self._inv_vocab = {v: k for (k, v) in self._vocab.items()}
        logger.info(
            f"ByteTokenizer built: #words {self.n_words}, BOS ID {self._bos_id}, "
            f"EOS ID {self._eos_id}"
        )

    def encode(self, text: str, *, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            text (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        # This always byte-tokenizes even sequences that would result in
        # special tokens. This means it's impossible to obtain special
        # tokens from tokenization.
        tokens = list(text.encode(errors="replace"))
        if bos:
            tokens.insert(0, self._bos_id)
        if eos:
            tokens.append(self._eos_id)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # This is a bit awkward, but we try to prevent UTF-8 shenanigans.
        curr_bytestr = []
        text_parts = []
        for val in tokens:
            if val >= ByteTokenizer.NUM_BYTE_VALUES:
                if curr_bytestr:
                    text_part = bytes(curr_bytestr).decode(errors="replace")
                    text_parts.append(text_part)
                    curr_bytestr.clear()
                text_parts.append(self._inv_vocab[val])
            else:
                curr_bytestr.append(val)
        text_parts.append(bytes(curr_bytestr).decode(errors="replace"))
        return "".join(text_parts)

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    @property
    def inv_vocab(self) -> Dict[int, str]:
        return self._inv_vocab


def build_byte_tokenizer(job_config: JobConfig) -> ByteTokenizer:
    return ByteTokenizer()
