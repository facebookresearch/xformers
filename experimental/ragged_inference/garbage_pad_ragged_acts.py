# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def garbage_pad_ragged_acts_kernel(
    ragged_acts_ptr,
    ragged_acts_offset_per_seq_ptr,
    n_ctx_per_seq_ptr,
    padded_acts_ptr,
    **meta,  # Optional meta-parameters for the kernel
):
    BLOCK_SIZE = meta["d_model"]  # How many inputs each program should process
    # There are multiple 'program's processing different data. We identify which program
    # we are here

    seq_idx = tl.program_id(axis=0)
    ctx_idx = tl.program_id(axis=1)

    # This program will process inputs that are offset from the initial data.
    # for instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].

    ragged_acts_offset_ptr = ragged_acts_offset_per_seq_ptr + seq_idx
    ragged_acts_offset = tl.load(ragged_acts_offset_ptr)

    # Create a mask to guard memory operations against out-of-bounds accesses
    n_ctx_in_this_seq_ptr = n_ctx_per_seq_ptr + seq_idx
    n_ctx_in_this_seq = tl.load(n_ctx_in_this_seq_ptr)
    ctx_idx_too_large_mask = ctx_idx < n_ctx_in_this_seq

    # Note that offsets is a list of pointers
    ragged_acts_offsets = ragged_acts_offset + tl.arange(0, BLOCK_SIZE)

    # Load ragged acts, since we use a BLOCK_SIZE of d_model, the only out of bounds
    # that we can have is if the n_ctx value is too big
    acts = tl.load(ragged_acts_ptr + ragged_acts_offsets, mask=ctx_idx_too_large_mask)

    # Calculate the offsets for the padded acts
    n_ctx_max = meta["n_ctx_max"]
    padded_acts_offset = n_ctx_max * seq_idx * BLOCK_SIZE

    # Write things back, again masking out the sections that would be garbage
    tl.store(padded_acts_ptr + padded_acts_offset, acts, mask=ctx_idx_too_large_mask)


class RaggedActivations:
    def __init__(self, raw_tensor: torch.Tensor, n_ctx_per_seq: List[int]):
        self.raw_tensor = raw_tensor
        self.n_ctx_per_seq = n_ctx_per_seq

    @property
    def n_seqs(self):
        return len(self.n_ctx_per_seq)

    @property
    def max_n_ctx_per_seq(self):
        return max(self.n_ctx_per_seq)

    @property
    def dtype(self):
        return self.raw_tensor.dtype

    @property
    def device(self):
        return self.raw_tensor.device

    @classmethod
    def from_list(cls, tensors: List[torch.Tensor]):
        """Tensors must all be of shape [n_ctx, d_model]."""
        return cls(
            raw_tensor=torch.cat(tensors),
            n_ctx_per_seq=[tensor.shape[0] for tensor in tensors],
        )

    def iter_full_tensors(self):
        idx_so_far = 0
        for n_ctx_in_this_seq in self.n_ctx_per_seq:
            yield self.raw_tensor[idx_so_far : idx_so_far + n_ctx_in_this_seq]
            idx_so_far += n_ctx_in_this_seq

    def to_garbage_padded(self) -> torch.Tensor:
        """
        Create a tensor of shape (n_seqs, n_ctx_max, d_model) where the
        sequences are right-padded with garbage data
        """
        n_seqs = len(self.n_ctx_per_seq)
        n_ctx_max = max(self.n_ctx_per_seq)

        n_dim = self.raw_tensor.shape[-1]
        # TODO: flag use zeros for garbage
        padded_acts = torch.zeros(
            n_seqs, n_ctx_max, n_dim, dtype=self.raw_tensor.dtype, device="cuda"
        )

        idx_so_far = 0
        for seq_idx, n_ctx_in_this_seq in enumerate(self.n_ctx_per_seq):
            this_seq = self.raw_tensor[idx_so_far : idx_so_far + n_ctx_in_this_seq]
            padded_acts[seq_idx, :n_ctx_in_this_seq, :] = this_seq
            idx_so_far += n_ctx_in_this_seq

        return padded_acts

    def triton_to_garbage_padded(self) -> torch.Tensor:
        """
        Create a tensor of shape (n_seqs, n_ctx_max, d_model) where the
        sequences are right-padded with garbage data
        """
        n_seqs = len(self.n_ctx_per_seq)
        n_ctx_max = max(self.n_ctx_per_seq)

        ragged_acts = self.raw_tensor
        d_model = ragged_acts.shape[-1]
        padded_acts = torch.empty(
            n_seqs, n_ctx_max, d_model, dtype=ragged_acts.dtype, device="cuda"
        )

        # We just use one program per n_ctx position for simplicity
        assert d_model >= 128, f"bad {d_model=}"
        assert d_model <= 8 * 1024, f"bad {d_model=}"
        assert d_model % 32 == 0, f"bad {d_model=}"

        # We use numpy here because it's a bit faster
        n_ctx_per_seq = self.n_ctx_per_seq
        ragged_acts_offset_per_seq = get_acts_offset_per_seq(n_ctx_per_seq)

        # The SPMD launch grid denotes the number of kernel instances that run in parallel.
        # It is analogous to CUDA launch grids. It can be either Tuple[int], or
        # Callable(metaparameters) -> Tuple[int]
        #
        # In this case, we use a 2D grid where the size is n_ctx
        grid_2d = (n_seqs, n_ctx_max)

        # NOTE:
        #  - each torch.tensor object is implicitly converted into a pointer to its
        #       first element.
        #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a
        #       callable GPU kernel

        # [breakpoint()]
        garbage_pad_ragged_acts_kernel[grid_2d](
            ragged_acts,
            torch.tensor(ragged_acts_offset_per_seq, device="cuda"),
            torch.tensor(self.n_ctx_per_seq, device="cuda"),
            padded_acts,
            d_model=d_model,
            n_ctx_max=n_ctx_max,
        )
        return padded_acts


def get_acts_offset_per_seq(n_ctx_per_seq):
    n_ctx_per_seq_shifted = np.array([0] + n_ctx_per_seq[:-1])
    ragged_acts_offset_per_seq = n_ctx_per_seq_shifted.cumsum(axis=0)
    return ragged_acts_offset_per_seq


"""

# TODO: Build LUT
seq_idx = 1
ctx_idx = 0

ragged_offset = 1

# How to do a list of tensors?
#

# TODO: Add the QK dotprod to get scores
#  - Start with a ragged tensor for the keys also
#  - Using a list of tensors as the Keys
#  - Using sequences

# 16x16x256


# scores [n_seq, n_ctx_keys_max, n_ctx_queries_max]


# final_out [n_seq, n_ctx_keys_max, d_model]
"""
