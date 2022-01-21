import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import estimate_matmul_time, prune_num_stages


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_K": block_n,
                                "BLOCK_D": block_k,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )

    return configs


def get_all_configs():
    return [
        # basic configs for compute-bound matmuls
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_K": 256, "BLOCK_D": 32},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_K": 128, "BLOCK_D": 32},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_K": 64, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_K": 32, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 32, "BLOCK_D": 32},
            num_stages=5,
            num_warps=2,
        ),
    ] + get_configs_io_bound()


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
    key=["n_ctx_q", "n_ctx_k", "d_model"],
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
    d_model,
    stride_ctx_q,
    stride_ctx_k,
    stride_d,  # Stride along the d_model_per_head dim
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

    # Iterate through blocks of the d_model dimension and accumulate values into acc
    acc_tile = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    rd = tl.arange(0, BLOCK_D)

    q_ptr_tile = q_ptr + (rq[:, None] * stride_ctx_q + rd[None, :] * stride_d)
    k_ptr_tile = k_ptr + (rd[:, None] * stride_d + rk[None, :] * stride_ctx_k)

    for d_max_offset in range(d_model, 0, -BLOCK_D):
        q_tile = tl.load(q_ptr_tile, mask=rd[None, :] < d_max_offset, other=0.0)
        k_tile = tl.load(k_ptr_tile, mask=rd[:, None] < d_max_offset, other=0.0)

        # In einsum notation, the following does: qd,dk->qk
        acc_tile += tl.dot(q_tile, k_tile)

        q_ptr_tile += BLOCK_D * stride_d
        k_ptr_tile += BLOCK_D * stride_d

    acc_tile = acc_tile.to(scores_ptr.dtype.element_ty)

    # We rematerialize rq and rk here because it allows them to be deallocated above
    # instead of being kept in registers throughout the inner for-loop
    rq = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    scores_offset_tile = rq[:, None] * stride_out_q + rk[None, :] * stride_out_k
    scores_ptr_tile = scores_ptr + scores_offset_tile

    mask = (rq < n_ctx_q)[:, None] & (rk < n_ctx_k)[None, :]

    tl.store(scores_ptr_tile, acc_tile, mask=mask)


def qk_dotprod(query, key):
    device = query.device

    # handle non-contiguous inputs if necessary
    if query.stride(0) > 1 and query.stride(1) > 1:
        query = query.contiguous()
    if key.stride(0) > 1 and key.stride(1) > 1:
        key = key.contiguous()

    # check constraints
    n_ctx_q, d_model = query.shape
    n_ctx_k, d_model_k = key.shape
    assert d_model == d_model_k, f"{query.shape=} {key.shape=}"

    # allocates output
    scores_out = torch.empty((n_ctx_q, n_ctx_k), device=device, dtype=query.dtype)

    # Stride along the d_model dimension
    stride_d = query.stride(1)
    assert stride_d == key.stride(1), f"{stride_d=}, {key.stride(1)=}"

    # launch kernel
    def grid(META):
        return (
            triton.cdiv(n_ctx_q, META["BLOCK_Q"])
            * triton.cdiv(n_ctx_k, META["BLOCK_K"]),
        )

    _kernel[grid](
        query,
        key,
        scores_out,
        n_ctx_q,
        n_ctx_k,
        d_model,
        query.stride(0),  # stride_ctx_q
        key.stride(0),  # stride_ctx_k
        stride_d,  # stride_d
        scores_out.stride(0),  # stride_out_q
        scores_out.stride(1),  # stride_out_k
    )
    return scores_out
