import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import estimate_matmul_time, prune_num_stages


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


BLOCK_Q = 16
BLOCK_K = 16
BLOCK_D = 64


def get_fast_dev_configs():
    return [
        triton.Config(
            {"BLOCK_Q": 64, "BLOCK_K": 32, "BLOCK_D": 32},
            num_stages=5,
            num_warps=2,
        )
    ]


@triton.autotune(
    # configs=get_all_configs(),
    configs=get_fast_dev_configs(),
    key=["n_ctx_q", "n_ctx_k", "d_head"],
    prune_configs_by={
        "prune_num_stages_by": prune_num_stages,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    },
)
@triton.jit
def _kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    n_ctx_q,
    n_ctx_k,  # N
    d_head,
    stride_ctx_q,
    stride_ctx_k,
    stride_out_q,
    stride_out_k,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):

    # matrix multiplication
    pid = tl.program_id(0)

    # Determine the number of blocks in the grid
    grid_k = (n_ctx_k + BLOCK_K - 1) // BLOCK_K

    pid_q = pid // grid_k
    pid_k = pid % grid_k

    # do matrix multiplication
    rq = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    rq = tl.max_contiguous(tl.multiple_of(rq % n_ctx_q, BLOCK_Q), BLOCK_Q)

    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rk = tl.max_contiguous(tl.multiple_of(rk % n_ctx_k, BLOCK_K), BLOCK_K)

    # Iterate through blocks of the d_head dimension and accumulate values into acc
    acc_tile = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    rd = tl.arange(0, BLOCK_D)

    q_ptr_tile = q_ptr + (rq[:, None] * stride_ctx_q + rd[None, :])
    k_ptr_tile = k_ptr + (rd[:, None] + rk[None, :] * stride_ctx_k)

    for d_max_offset in range(d_head, 0, -BLOCK_D):
        q_tile = tl.load(q_ptr_tile, mask=rd[None, :] < d_max_offset, other=0.0)
        k_tile = tl.load(k_ptr_tile, mask=rd[:, None] < d_max_offset, other=0.0)

        # In einsum notation, the following does: qd,dk->qk
        acc_tile += tl.dot(q_tile, k_tile)

        q_ptr_tile += BLOCK_D
        k_ptr_tile += BLOCK_D

    acc_tile = acc_tile.to(scores_ptr.dtype.element_ty)

    # We rematerialize rq and rk here because it allows them to be deallocated above
    # instead of being kept in registers throughout the inner for-loop
    rq = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    scores_offset_tile = rq[:, None] * stride_out_q + rk[None, :] * stride_out_k
    scores_ptr_tile = scores_ptr + scores_offset_tile

    mask = (rq < n_ctx_q)[:, None] & (rk < n_ctx_k)[None, :]

    tl.store(scores_ptr_tile, acc_tile, mask=mask)


def qk_dotprod_v2(query, key):
    device = query.device

    # handle non-contiguous inputs if necessary
    if query.stride(0) > 1 and query.stride(1) > 1:
        query = query.contiguous()
    if key.stride(0) > 1 and key.stride(1) > 1:
        key = key.contiguous()

    # check constraints
    n_ctx_q, d_head = query.shape
    n_ctx_k, d_head_k = key.shape
    assert d_head == d_head_k, f"{query.shape=} {key.shape=}"

    # allocates output
    scores_out = torch.empty((n_ctx_q, n_ctx_k), device=device, dtype=query.dtype)

    # Stride along the d_head dimension must be 1
    assert query.stride(1) == 1, f"{query.stride(1)}"
    assert key.stride(1) == 1, f"{key.stride(1)}"

    # We need to know the grid to make our lookup tables
    grid = (triton.cdiv(n_ctx_q, BLOCK_Q) * triton.cdiv(n_ctx_k, BLOCK_K),)

    # Create lookup tables
    pid_to_query_input_token_offset = [0, 4, 5, 8, 12]
    pid_to_key_input_token_offset = ...
    # pid_to_seq_idx = [0, 0, 1, 2, 2]

    # pid_to_output_k_block

    _kernel[grid](
        query,
        key,
        scores_out,
        n_ctx_q,
        n_ctx_k,
        d_head,
        query.stride(0),  # stride_ctx_q
        key.stride(0),  # stride_ctx_k
        scores_out.stride(0),  # stride_out_q
        scores_out.stride(1),  # stride_out_k
    )
    return scores_out
