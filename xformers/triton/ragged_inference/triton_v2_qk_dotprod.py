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
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
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
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_stages=5,
            num_warps=2,
        ),
    ] + get_configs_io_bound()


def get_fast_dev_configs():
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_stages=5,
            num_warps=2,
        )
    ]



@triton.autotune(
    # configs=get_all_configs(),
    configs=get_fast_dev_configs(),
    key=["n_ctx_q", "n_ctx_k", "n_dim"],
    prune_configs_by={
        "prune_num_stages_by": prune_num_stages,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    },
)
@triton.jit
def _kernel(
    query_ptr,
    key_ptr,
    scores_out_ptr,
    n_ctx_q, # M
    n_ctx_k, # N
    n_dim, # K
    stride_ctx_q,
    stride_d_model_for_q,
    stride_d_model_for_k,
    stride_ctx_k,
    stride_out_ctx_q,
    stride_out_ctx_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):

    # matrix multiplication
    pid = tl.program_id(0)

    # Determine the number of blocks in the grid
    grid_n = (n_ctx_k + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % n_ctx_q, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % n_ctx_k, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # pointers
    query_ptr = query_ptr + (ram[:, None] * stride_ctx_q + rk[None, :] * stride_d_model_for_q)
    key_ptr = key_ptr + (rk[:, None] * stride_d_model_for_k + rbn[None, :] * stride_ctx_k)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(n_dim, 0, -BLOCK_K):

        a = tl.load(query_ptr, mask=rk[None, :] < k, other=0.)
        b = tl.load(key_ptr, mask=rk[:, None] < k, other=0.)

        acc += tl.dot(a, b)
        query_ptr += BLOCK_K * stride_d_model_for_q
        key_ptr += BLOCK_K * stride_d_model_for_k
    acc = acc.to(scores_out_ptr.dtype.element_ty)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    scores_out_ptr = scores_out_ptr + (rm[:, None] * stride_out_ctx_q + rn[None, :] * stride_out_ctx_k)
    mask = (rm < n_ctx_q)[:, None] & (rn < n_ctx_k)[None, :]

    tl.store(scores_out_ptr, acc, mask=mask)


def qk_dotprod(query, key):
    device = query.device
    key_T = key.T

    # handle non-contiguous inputs if necessary
    if query.stride(0) > 1 and query.stride(1) > 1:
        query = query.contiguous()
    if key_T.stride(0) > 1 and key_T.stride(1) > 1:
        key_T = key_T.contiguous()

    # checks constraints
    assert query.shape[1] == key_T.shape[0], f"incompatible dimensions, {query.shape=} {key_T.shape=}"

    n_ctx_q, n_dim = query.shape
    _, n_ctx_k = key_T.shape

    # allocates output
    scores_out = torch.empty((n_ctx_q, n_ctx_k), device=device, dtype=query.dtype)

    # launch kernel
    def grid(META):
        return (triton.cdiv(n_ctx_q, META["BLOCK_M"]) * triton.cdiv(n_ctx_k, META["BLOCK_N"]),)

    _kernel[grid](
        query,
        key_T,
        scores_out,
        n_ctx_q,
        n_ctx_k,
        n_dim,
        query.stride(0),
        query.stride(1),
        key_T.stride(0),
        key_T.stride(1),
        scores_out.stride(0),
        scores_out.stride(1),
    )
    return scores_out
