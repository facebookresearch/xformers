# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple, Type

import torch

from xformers.ops import fmha

from .fmha import (
    flash,
    flash3,
    memory_efficient_attention_forward_requires_grad,
    memory_efficient_attention_partial,
    merge_attentions,
    triton_splitk,
)
from .fmha.attn_bias import (
    AttentionBias,
    PagedBlockDiagonalGappyKeysMask,
    PagedBlockDiagonalPaddedKeysMask,
)
from .fmha.common import AttentionFwOpBase
from .fmha.dispatch import _get_use_fa3, fa3_available


@dataclass
class TreeAttnMetadata:
    """
    tree_choices: definition of the tree, tuples sorted by length, each corresponding
        to a node. See the docstring of TreeAttnMetadata.from_tree_choices.
    attention_bias: Medusa-style tree attention bias as an explicit tensor
        of shape (tree_size, tree_size), where tree_size is the total number
        of nodes in the tree. It can be used as a spec_attn_bias ("right"
        or "suffix" attention part) in tree_attention.
        See tree_attention_with_sync for a usage example.
    tree_indices: 1D tensor of size tree_size which maps tree nodes to draft tokens.
        Tree nodes are assumed to be in the same order as in tree_choices
        (see TreeAttnMetadata.from_tree_choices).
    retrieval_indices: a tensor of shape (number of leaves, depth + 1), where one
        row corresponds to one path, and contains indices of the tree nodes
        on that path. Paths are padded with -1 from the right.
        The paths (row dim) are unsorted.
    path_lengths: real lengths for each of the paths.
    tree_seq_position_ids: 1D tensor of size tree_size which indicates which head
        a node belongs to. Equivalently, it shows the sequence position of the
        node within the corresponding path.
    parent_node_indices: 1D tensor of size tree_size which for each node contains
        position of its parent + 1. For root node(s) it contains 0.
    child_node_indices: a tensor of shape (tree_size, max_num_children_per_node),
        in which each row contains indices of children of the corresponding node.
        Rows corresponding to nodes which have less than max_num_children_per_node
        children are padded by repeating the last child index.
        For leaf nodes the values are meaningless and filled with 0.
    num_children_per_node: 1D tensor of size tree_size which contains the number of
        children for each node.
    candidate_idx: 1D tensor of size tree_size, contains index of each node among its "siblings".
        Takes values from 0 to the number of children of the parent node minus 1.
    num_nodes_per_level: 1D tensor of the number of nodes at each level (including root).
    num_children_per_node_at_level: List of 1D tensors, each containing the number of children at the tree level.
    subtree_size: List of integers, each containing the number of nodes in the subtree at the tree level.
    Example:
        Tree choices
          `[(0,), (0, 0), (0, 1), (0, 2), (1,), (1, 0), (1, 1), (1, 2), (2,), (2, 0), (2, 1), (2, 2)]`
        represents a tree that looks like this:
            0
            |-- 1
            |   |-- 4
            |   |-- 5
            |   |-- 6
            |
            |-- 2
            |   |-- 7
            |   |-- 8
            |   |-- 9
            |
            |-- 3
                |-- 10
                |-- 11
                |-- 12

        with TreeAttnMetadata
            tree_indices=tensor([0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6])
            retrieval_indices=tensor([[ 0,  1,  5],
                                      [ 0,  2,  9],
                                      [ 0,  3, 11],
                                      [ 0,  1,  4],
                                      [ 0,  2,  8],
                                      [ 0,  3, 10],
                                      [ 0,  1,  6],
                                      [ 0,  2,  7],
                                      [ 0,  3, 12]])
            path_lengths=[3, 3, 3, 3, 3, 3, 3, 3, 3]
            tree_seq_position_ids=tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
            child_node_indices=tensor([[ 0,  1,  2],
                                       [ 3,  4,  5],
                                       [ 6,  7,  8],
                                       [ 9, 10, 11],
                                       [ 0,  0,  0],
                                       ...
                                       [ 0,  0,  0]])
            num_children_per_node=tensor([3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            candidate_idx=tensor([0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
            num_nodes_per_level=tensor([1, 3, 3])
            num_children_per_node_at_level=[tensor([3]), tensor([3, 3, 3]), tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])]
            subtree_sizes=[1, 4, 13]
    """

    tree_choices: Sequence[Tuple[int, ...]]
    attention_bias: torch.Tensor
    tree_indices: torch.Tensor
    retrieval_indices: torch.Tensor
    path_lengths: List[int]
    tree_seq_position_ids: torch.Tensor
    parent_node_indices: torch.Tensor
    child_node_indices: torch.Tensor
    num_children_per_node: torch.Tensor
    candidate_idx: torch.Tensor
    num_nodes_per_level: torch.Tensor
    num_nodes_per_level_cpu: torch.Tensor
    num_children_per_node_at_level: List[torch.Tensor]
    num_children_per_node_at_level_cpu: List[torch.Tensor]
    subtree_sizes: List[int]

    @classmethod
    @lru_cache
    def from_tree_choices_cached(
        cls,
        tree_choices: Tuple[Tuple[int, ...]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> "TreeAttnMetadata":
        return cls.from_tree_choices(tree_choices, dtype, device)

    @classmethod
    def from_tree_choices(
        cls,
        tree_choices: Sequence[Tuple[int, ...]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> "TreeAttnMetadata":
        """
        Args:
            tree_choices: tree description in the style of
                https://github.com/FasterDecoding/Medusa/blob/5e9805386/medusa/model/medusa_choices.py
                A typical tree description would look like:
                [(node0, node1, ...), (node0, node2), (node0, node3), (node1, node3), ..., (node0, node2, ..., nodeN)]
                Every tuple is corresponds to one node in the tree, encoded as a path from one of the root nodes to the
                node in question.
                For example, a node encoded as (1, 0, 3, ..., 2) is understood as:
                list all the root nodes and take node number 1
                list all children of that node and take node number 0
                list all children of that node and take node number 3
                ...
                list all children of that node and take node number 2 - that's the node encoded by this tuple.

            dtype: data type of the output mask tensor.
            device: device of the output tensors.
        Returns:
            TreeAttnMetadata object with all the fields.
        """
        # from https://github.com/SafeAILab/EAGLE/blob/e98fc7c/model/utils.py#L89C1-L117C1
        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

        depth_counts = _get_depth_counts(sorted_tree_choices)
        tree_indices = _prepare_tree_indices(sorted_tree_choices, depth_counts, device)
        retrieval_indices, path_lengths = _prepare_retrieval_indices(
            sorted_tree_choices, device
        )
        tree_seq_position_ids = _prepare_tree_position_ids(
            depth_counts, tree_len, device
        )
        tree_attn_mask = _prepare_tree_attn_bias(
            sorted_tree_choices, depth_counts, dtype, device
        )
        parent_node_indices = _prepare_parent_node_indices(sorted_tree_choices, device)
        child_node_indices, num_children_per_node = _prepare_child_node_indices(
            sorted_tree_choices, device
        )
        candidate_idx = _prepare_candidate_idx(sorted_tree_choices, device)

        num_nodes_per_level = _get_num_nodes_per_level(depth_counts, device)
        num_nodes_per_level_cpu = num_nodes_per_level.cpu()
        (
            subtree_sizes,
            num_children_per_node_at_level,
        ) = _get_subtree_size_and_num_children_per_node_at_level(
            num_nodes_per_level, num_children_per_node, device
        )
        num_children_per_node_at_level_cpu = [
            row.cpu() for row in num_children_per_node_at_level
        ]
        return TreeAttnMetadata(
            sorted_tree_choices,
            tree_attn_mask,
            tree_indices,
            retrieval_indices,
            path_lengths,
            tree_seq_position_ids,
            parent_node_indices,
            child_node_indices,
            num_children_per_node,
            candidate_idx,
            num_nodes_per_level,
            num_nodes_per_level_cpu,
            num_children_per_node_at_level,
            num_children_per_node_at_level_cpu,
            subtree_sizes,
        )


def _get_subtree_size_and_num_children_per_node_at_level(
    num_nodes_per_level: torch.Tensor,
    num_children_per_node: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[List[int], List[torch.Tensor]]:
    depth: int = len(num_nodes_per_level)
    subtree_sizes: List[int] = [
        1,
    ]
    num_children_per_node_at_level: List[torch.Tensor] = [
        num_children_per_node[0].unsqueeze(0)
    ]
    for i in range(1, depth):
        subtree_sizes.append(int(torch.sum(num_nodes_per_level[: (i + 1)])))
        num_children_per_node_at_level.append(
            num_children_per_node[subtree_sizes[i - 1] : subtree_sizes[i]]
        )
    return subtree_sizes, num_children_per_node_at_level


def _get_depth_counts(sorted_tree_choices: List[Tuple[int, ...]]) -> List[int]:
    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    return depth_counts


def _get_num_nodes_per_level(
    depth_counts: List[int], device: Optional[torch.device]
) -> torch.Tensor:
    depth_counts_tensor: torch.Tensor = torch.tensor([1] + depth_counts, device=device)
    return depth_counts_tensor[depth_counts_tensor != 0]


def _prepare_tree_attn_bias(
    sorted_tree_choices: List[Tuple[int, ...]],
    depth_counts: List[int],
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
) -> torch.Tensor:
    """
    Construct a Medusa-style tree attention bias as an explicit tensor.
    It can be used as a spec_attn_bias ("right" or "suffix" attention part)
    in tree_attention. See run_tree_attention_inner in test for a usage example.
    Args:
        sorted_tree_choices: tree description in the style of
            https://github.com/FasterDecoding/Medusa/blob/5e9805386/medusa/model/medusa_choices.py
            A typical tree description would look like:
            [(node0, node1, ...), (node0, node2), (node0, node3), (node1, node3), ..., (node0, node2, ..., nodeN)]
            Every tuple is corresponds to one node in the tree, encoded as a path from one of the root nodes to the
            node in question. Passed in sorted order.
            For example, a node encoded as (1, 0, 3, ..., 2) is understood as:
            list all the root nodes and take node number 1
            list all children of that node and take node number 0
            list all children of that node and take node number 3
            ...
            list all children of that node and take node number 2 - that's the node encoded by this tuple
        depth_counts: a list of integers, where the i-th element is the number of choices with depth i.
        dtype: data type of the output tensor.
        device: device of the output tensor.
    Returns:
        attention bias of shape (tree_size, tree_size),
        where tree_size is the total number of nodes in the tree.
    """
    # +1 comes from the addtional root node
    tree_len = len(sorted_tree_choices) + 1
    tree_attn_mask = torch.full(
        (tree_len, tree_len), -torch.inf, device=device, dtype=dtype
    )

    mask_val = 0
    for i in range(tree_len):
        tree_attn_mask[i, i] = mask_val

    tree_attn_mask[:, 0] = mask_val
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            # retrieve ancestor position
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(
                    sorted_tree_choices.index(cur_tree_choice[: c + 1]) + 1
                )
            tree_attn_mask[j + start + 1, ancestor_idx] = mask_val
        start += depth_counts[i]
    return tree_attn_mask


def _prepare_tree_indices(
    sorted_tree_choices: List[Tuple[int, ...]],
    depth_counts: List[int],
    device: Optional[torch.device],
) -> torch.Tensor:
    """
    Construct an index tensor for choices in the tree and their corresponding index in the draft tokens.
    Args:
        sorted_tree_choices: sorted from tree_choices input of prepare_tree_attn_metadata function
        depth_counts: a list of integers, where the i-th element is the number of choices with depth i.
        device: device of the output tensor.
    Returns:
        tree indices of shape (tree_len,). See docstring of TreeAttnMetadata for details.
    """
    # Generate tree indices for the tree_choices structure
    # add root node from main head prediction to the tree_len
    tree_len = len(sorted_tree_choices) + 1
    tree_indices = torch.zeros(tree_len, device=device, dtype=torch.long)
    tree_indices[0] = 0
    start, max_idx_prev_level = 0, 0
    for i in range(len(depth_counts)):
        cur_offset = max_idx_prev_level
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            tree_idx = cur_tree_choice[-1] + cur_offset + 1
            tree_indices[start + j + 1] = tree_idx
            max_idx_prev_level = max(tree_idx, max_idx_prev_level)
        start += depth_counts[i]
    return tree_indices


def _prepare_retrieval_indices(
    tree_choices: List[Tuple[int, ...]], device: Optional[torch.device]
) -> Tuple[torch.Tensor, List[int]]:
    """
    Convert tree definition from the format used by Medusa and EAGLE (tree_choices, see docstring of
    TreeAttnMetadata.from_tree_choices) to a list of paths:
    [
        (node_index0_path0, node_index1_path0, ...),
        (node_index0_path1, node_index1_path1, ...),
        ...
    ]
    where each value is an index of a node inside the corresponding level of a tree.
    Returns:
        retrieval indices of shape (number of leaves, depth + 1)
        length of each path.
    """
    tree_depth = max(len(node) for node in tree_choices) + 1 if tree_choices else 1

    leaves = set(tree_choices)

    for node in tree_choices[::-1]:
        if node[:-1] in leaves:
            leaves.remove(node[:-1])

    paths, path_lengths = [], []
    for leaf in leaves:
        path = [0] + [
            tree_choices.index(leaf[:level]) + 1 for level in range(1, len(leaf) + 1)
        ]
        path_lengths.append(len(path))
        paths.append(path + [-1] * (tree_depth - len(path)))
    paths_tensor = torch.tensor(paths, dtype=torch.long, device=device)
    return paths_tensor, path_lengths


def _prepare_tree_position_ids(
    depth_counts: List[int], tree_len: int, device: Optional[torch.device]
) -> torch.Tensor:
    """
    Construct sequence position of each node within its path, can be used for positional embedding.
    Args:
        depth_counts: number of nodes at each of the levels of the tree.
        tree_len: total number of nodes in the tree including the root.
        device: device of the output tensor.
    Returns:
        tree position ids of shape (tree_len,). See docstring of TreeAttnMetadata for details.
    """
    tree_position_ids = torch.zeros(tree_len, dtype=torch.int32, device=device)
    start = 0
    for i in range(len(depth_counts)):
        tree_position_ids[start + 1 : start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    return tree_position_ids


def _prepare_parent_node_indices(
    sorted_tree_choices: List[Tuple[int, ...]], device: Optional[torch.device]
) -> torch.Tensor:
    ancestor_idx = []
    for cur_medusa_choice in sorted_tree_choices:
        try:
            ancestor_idx.append(sorted_tree_choices.index(cur_medusa_choice[:-1]) + 1)
        except ValueError:
            ancestor_idx.append(0)
    return torch.tensor(ancestor_idx, dtype=torch.long, device=device)


def _prepare_child_node_indices(
    tree_choices: List[Tuple[int, ...]], device: Optional[torch.device]
) -> Tuple[torch.Tensor, torch.Tensor]:
    res = []
    num_children_per_node = []
    for x in [()] + tree_choices:
        curr_children = [
            i
            for i, y in enumerate(tree_choices)
            if len(x) + 1 == len(y) and y[:-1] == x
        ]
        num_children_per_node.append(len(curr_children))
        if curr_children:
            res.append(curr_children)
        else:
            res.append([0])

    # pad all children lists by repeating the last element
    max_num_children = max(len(x) for x in res)
    res = [x + x[-1:] * (max_num_children - len(x)) for x in res]

    # Check that all nodes have the same number of children.
    assert all(len(x) == len(res[0]) for x in res)
    return (
        torch.tensor(res, dtype=torch.long, device=device),
        torch.tensor(num_children_per_node, dtype=torch.long, device=device),
    )


def _prepare_candidate_idx(
    tree_choices: List[Tuple[int, ...]], device: Optional[torch.device]
) -> torch.Tensor:
    candidate_idx = [
        sum(
            curr_node[:-1] == another_node[:-1]
            for another_node in tree_choices[:curr_node_idx]
        )
        for curr_node_idx, curr_node in enumerate(tree_choices)
    ]
    return torch.tensor(candidate_idx, dtype=torch.long, device=device)


def use_triton_splitk_for_prefix(B: int, G: int, tree_size: int) -> bool:
    """
    Heuristic to decide whether to use Triton Split-k or default (Flash Attention) for prefix attention.
    """
    return (
        (B * G <= 128 and tree_size <= 64)
        or (B * G < 4 and tree_size < 100)
        or B * G < 2
    )


def select_prefix_op(
    B: int,
    G: int,
    tree_size: int,
    autotune: bool,
    attn_bias: AttentionBias,
    kv_cache_dtype: torch.dtype,
) -> Optional[Type[AttentionFwOpBase]]:
    """
    Heuristic to decide whether to use Triton Split-k or default (Flash Attention) for prefix attention.
    """
    triton_splitk_op = SplitKAutotune if autotune else triton_splitk.FwOp
    if torch.version.hip:
        # TODO: further tune heuristics once CK splitK is ready
        return triton_splitk_op

    # Triton Split-k is not present in the dispatcher list for some shapes.
    # However, we need to dispatch to it if no other op is available.
    # FA3 splitKV doesn't yet support gappy or paged biases.
    fa3_splitkv_supported = isinstance(
        attn_bias, flash3.FwOp_KVSplit.SUPPORTED_ATTN_BIAS_TYPES  # type: ignore
    )
    fa3_supported = isinstance(attn_bias, flash3.FwOp.SUPPORTED_ATTN_BIAS_TYPES)  # type: ignore
    flash2_supported = isinstance(attn_bias, flash.FwOp.SUPPORTED_ATTN_BIAS_TYPES)  # type: ignore
    if not (fa3_splitkv_supported or fa3_supported or flash2_supported):
        return triton_splitk_op

    assert torch.version.cuda
    use_fa3 = _get_use_fa3() and fa3_available()
    # override heuristics for quantized kv cache for decode
    if kv_cache_dtype == torch.uint8:
        return triton_splitk_op
    # select FA3 when bs >= 64
    if B >= 64 and use_fa3:
        if fa3_splitkv_supported:
            return flash3.FwOp_KVSplit
        return flash3.FwOp
    elif use_triton_splitk_for_prefix(B, G, tree_size):
        return triton_splitk_op
    else:
        # use default heuristics from xformers
        return None


def tree_attention(
    q: torch.Tensor,
    spec_k: torch.Tensor,
    spec_v: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    spec_attn_bias: torch.Tensor,
    prefix_attn_bias: AttentionBias,
    prefix_op: Optional[Type[AttentionFwOpBase]] = None,
    suffix_op: Optional[Type[AttentionFwOpBase]] = None,
    autotune: bool = False,
    quantized_kv_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Compute Medusa/EAGLE/Hydra-style tree attention.
    Notice that this function takes as arguments biases for the left (prefix)
    and right (speculative suffix) parts of the attention.
    This way we avoid creating these biases on the fly, and
    allow this function to be used in performance-critical decoding
    jobs, including in CUDA graph mode. In the latter case one should
    construct the biases once, and update prefix_attn_bias with
    current seqlens before every graph replay; spec_attn_bias stays static,
    as it's determined by the tree structure.
    Args:
        q: query from speculative tokens, of shape (B, tree_size_q, (G), H, D)
        spec_k, spec_v: keys/values from speculative tokens, each of shape (B, tree_size_kv, (G), H, D).
            If tree_size_q < tree_size_kv, we assume the end of the query sequence aligns with end the k/v sequence,
            like in "from-bottom-right" attention masks. Such rectangular attention masks can be used when we are
            adding new nodes to the tree, and want to avoid recomputing attention for the existing nodes. For example,
            this can be used during draft token generation in EAGLE.
        cache_k/cache_v: queries/keys/values from the existing context, each of shape (B, Mk, (G), H, D)
        spec_attn_bias: attention bias of the "right" part of the attention (tree_size_q x spec tokens).
            This would typically be a an explicit tensor mask, precomputed once and not changing during decoding
        prefix_attn_bias: attention bias of the "left" part of the attention (tree_size_q x existing context).
            This bias would typically be block-diagonal padded non-causal (BlockDiagonalPaddedKeysMask), and it
            changes at every decoding step as K/V sequence lengths grow during decoding.
        prefix_op: attention backend which will be passed to memory_efficient_attention to compute prefix attention.
                   If None, will use Triton Split-K or Flash Attention depending on the heuristics.
        suffix_op: same as prefix_op, but for the suffix.
        autotune: If True, Triton Split-K will use autotuning when chosen
            as a default backend for prefix/suffix attention.
    Returns:
        attention output of shape (B, tree_size_q, (G), H, D)

    :Usage example:

        See also tree_attention_with_sync in tests/test_tree_attention.py

    .. code-block:: python

        # Create an attention bias for the prefix part of the attention
        prefix_attn_bias = BlockDiagonalPaddedKeysMask.from_seqlens(
            q_seqlen=[tree_size_q for _ in range(B)], kv_seqlen=kv_lens, kv_padding=Mk
        )
        # Create an explit attention bias for the speculative part of the attention
        spec_attn_bias = TreeAttnMetadata.from_tree_choices(tree_choices, q.dtype, q.device).attention_bias
        attn_output = tree_attention(
            q, spec_k, spec_v, cache_k, cache_v, spec_attn_bias, prefix_attn_bias
        )
    """

    is_bmhk = q.ndim == 4
    if is_bmhk:
        q = q.unsqueeze(2)
        spec_k, spec_v = spec_k.unsqueeze(2), spec_v.unsqueeze(2)
        cache_k, cache_v = cache_k.unsqueeze(2), cache_v.unsqueeze(2)

    B, tree_size_q, G, H, D = q.shape
    Bkv, Mk, G1, H1, D1 = cache_k.shape
    tree_size_q1, tree_size_kv = spec_attn_bias.shape
    if isinstance(
        prefix_attn_bias,
        (PagedBlockDiagonalPaddedKeysMask, PagedBlockDiagonalGappyKeysMask),
    ):
        assert Bkv == 1
    else:
        assert Bkv == B

    assert H == H1 and D == D1 and G == G1
    assert cache_k.shape == cache_v.shape
    assert (
        tree_size_q1 == tree_size_q <= tree_size_kv
    ), f"{tree_size_q1=} {tree_size_q=} {tree_size_kv=}"
    assert (
        q.shape[2:] == spec_k.shape[2:] == spec_v.shape[2:]
    ), f"{q.shape=} {spec_k.shape=} {spec_v.shape=}"

    spec_attn_bias = spec_attn_bias.expand(B, G, H, tree_size_q, tree_size_kv)

    triton_splitk_op = SplitKAutotune if autotune else triton_splitk.FwOp

    # TODO: improve this heuristic
    if prefix_op is None:
        prefix_op = select_prefix_op(
            B, G, tree_size_kv, autotune, prefix_attn_bias, cache_k.dtype
        )
    if cache_k.dtype == torch.uint8:
        assert quantized_kv_scales is not None
        assert prefix_op is triton_splitk.FwOp
        fp8_inp = triton_splitk.InputsFp8(
            query=q.view(1, B * tree_size_q, G, H, D),
            key=cache_k.view(1, Bkv * Mk, G, H, D).view(torch.int32),
            value=cache_v.view(1, Bkv * Mk, G, H, D).view(torch.int32),
            attn_bias=prefix_attn_bias,
            k_fp8_scale_shift=quantized_kv_scales[0].view(1, Bkv * Mk, G, H),
            v_fp8_scale_shift=quantized_kv_scales[1].view(1, Bkv * Mk, G, H),
            is_partial=True,
        )
        out, ctx = fmha._memory_efficient_attention_forward_requires_grad(
            fp8_inp,
            op=prefix_op,
        )
        attn_prefix, lse_prefix = out, ctx.lse
    else:
        attn_prefix, lse_prefix = memory_efficient_attention_partial(
            q.view(1, B * tree_size_q, G, H, D),
            cache_k.view(1, Bkv * Mk, G, H, D),
            cache_v.view(1, Bkv * Mk, G, H, D),
            attn_bias=prefix_attn_bias,
            op=prefix_op,
        )
    attn_prefix = attn_prefix.view(B, tree_size_q, G, H, D)
    lse_prefix = lse_prefix.view(G, H, B, tree_size_q).permute(2, 0, 1, 3)

    # attn_suffix ~ (B, tree_size_q, G, H, D)
    # lse_suffix ~ (B, G, H, tree_size_q)
    attn_suffix, lse_suffix = memory_efficient_attention_forward_requires_grad(
        q,
        spec_k,
        spec_v,
        attn_bias=spec_attn_bias,
        op=suffix_op or triton_splitk_op,
    )

    # attn_output ~ [B, tree_size_q, G, H, D]
    # attn input [B, M, G, H, Kq]
    # lse input [B, G, H, M]
    attn_output, _ = merge_attentions(
        [attn_prefix, attn_suffix],
        [lse_prefix, lse_suffix],
        output_dtype=q.dtype,
    )

    if is_bmhk:
        attn_output = attn_output.squeeze(2)
    return attn_output


class SplitKAutotune(triton_splitk.FwOp):
    AUTOTUNE = True


@lru_cache
def construct_full_tree_choices(
    tree_depth: int, branching: int
) -> List[Tuple[int, ...]]:
    """
    Construct a full tree of a given depth where each node (except for leaves) has a given number of children.
    The format is compatible with that used by Medusa and EAGLE:
    https://github.com/FasterDecoding/Medusa/blob/5e98053/medusa/model/medusa_choices.py
    For detailed description, see docstring of
    xformers.ops.tree_attention.TreeAttnMetadata.from_tree_choices .
    """
    return construct_tree_choices(branching=[branching] * tree_depth)


def construct_tree_choices(
    branching: List[int],
) -> List[Tuple[int, ...]]:
    """
    Construct a tree based on given branching factor for each non-root level.
    """
    choices: List[Tuple[int, ...]] = []
    for i in range(len(branching)):
        choices.extend(itertools.product(*[range(branching[k]) for k in range(i + 1)]))
    return choices


def get_full_tree_size(tree_depth: int, branching: int) -> int:
    """
    Number of nodes in a full tree of a given depth (including the root node) and branching factor.
    """
    return sum(branching**i for i in range(tree_depth))
