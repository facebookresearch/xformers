# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector
    y_ptr,  # *Pointer* to second input vector
    output_ptr,  # *Pointer* to output vector
    n_elements,  # Size of the vector
    **meta,  # Optional meta-parameters for the kernel
):
    BLOCK_SIZE = meta["BLOCK_SIZE"]  # How many inputs each program should process
    # There are multiple 'program's processing different data. We identify which program
    # we are here

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
    # This program will process inputs that are offset from the initial data.
    # for instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].

    # Note that offsets is a list of pointers
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements

    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):

    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda

    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or
    # Callable(metaparameters) -> Tuple[int]
    #
    # In this case, we use a 1D grid where the size is the number of blocks

    def grid_fn(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # NOTE:
    #  - each torch.tensor object is implicitly converted into a pointer to its
    #       first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a
    #       callable GPU kernel
    kernel_for_our_grid = add_kernel[grid_fn]

    # We must pass meta-parameters as keywords arguments
    _ = kernel_for_our_grid(x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


class RaggedActivations:
    def __init__(self, raw_tensor: torch.Tensor, n_ctx_per_seq: List[int]):
        self.raw_tensor = raw_tensor
        self.n_ctx_per_seq = n_ctx_per_seq

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
        padded_acts = torch.empty(
            n_seqs, n_ctx_max, n_dim, dtype=self.raw_tensor.dtype, device="cuda"
        )

        idx_so_far = 0
        for seq_idx, n_ctx_in_this_seq in enumerate(self.n_ctx_per_seq):
            this_seq = self.raw_tensor[idx_so_far : idx_so_far + n_ctx_in_this_seq]
            padded_acts[seq_idx, :n_ctx_in_this_seq, :] = this_seq
            idx_so_far += n_ctx_in_this_seq

        return padded_acts
