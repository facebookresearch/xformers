# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os

from sentencepiece import SentencePieceProcessor


class Tokenizer:
    """Encoding/decoding text using SentencePiece."""

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a SentencePiece model.

        Args:
            model_path (str): The path to the SentencePiece model file.
        """
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        print(
            f"loaded SentencePiece model: "
            f"#words: {self.n_words} - "
            f"bos id: {self.bos_id} - "
            f"eos id: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> list[int]:
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
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: list[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (list[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)
