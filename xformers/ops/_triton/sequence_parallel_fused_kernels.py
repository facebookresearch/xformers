# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from typing import List, Optional, Set, Tuple, cast

import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time


def init_to_zero(*names):
    def result(nargs):
        if nargs["blocks_done_counters"].numel() > 0:
            nargs["blocks_done_counters"].zero_()
        for name in names:
            nargs[name].zero_()

    return result


def gen_config(
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    warps: int,
    split_k: int = 1,
    group_m: int = 8,
) -> triton.Config:
    """A more compact way to define a triton.Config, so it fits on one line"""

    return triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "SPLIT_K": split_k,
            "GROUP_M": group_m,
        },
        num_stages=stages,
        num_warps=warps,
        pre_hook=init_to_zero("C1", "C2", "C3") if split_k > 1 else init_to_zero(),
    )


BASIC_MATMUL_CONFIGS = [
    gen_config(block_m=128, block_n=256, block_k=32, stages=3, warps=8),
    gen_config(block_m=256, block_n=128, block_k=32, stages=3, warps=8),
    gen_config(block_m=256, block_n=64, block_k=32, stages=4, warps=4),
    gen_config(block_m=64, block_n=256, block_k=32, stages=4, warps=4),
    gen_config(block_m=128, block_n=128, block_k=32, stages=4, warps=4),
    gen_config(block_m=128, block_n=64, block_k=32, stages=4, warps=4),
    gen_config(block_m=64, block_n=128, block_k=32, stages=4, warps=4),
    gen_config(block_m=128, block_n=32, block_k=32, stages=4, warps=4),
    gen_config(block_m=64, block_n=32, block_k=32, stages=5, warps=2),
]


INT8_MATMUL_CONFIGS = [
    gen_config(block_m=128, block_n=256, block_k=128, stages=3, warps=8),
    gen_config(block_m=256, block_n=128, block_k=128, stages=3, warps=8),
    gen_config(block_m=256, block_n=64, block_k=128, stages=4, warps=4),
    gen_config(block_m=64, block_n=256, block_k=128, stages=4, warps=4),
    gen_config(block_m=128, block_n=128, block_k=128, stages=4, warps=4),
    gen_config(block_m=128, block_n=64, block_k=64, stages=4, warps=4),
    gen_config(block_m=64, block_n=128, block_k=64, stages=4, warps=4),
    gen_config(block_m=128, block_n=32, block_k=64, stages=4, warps=4),
    gen_config(block_m=64, block_n=32, block_k=64, stages=5, warps=2),
]


IO_BOUND_MATMUL_CONFIGS_STAGES = [2, 3, 4, 5, 6]
IO_BOUND_MATMUL_CONFIGS_BLOCK_M = [16, 32]
IO_BOUND_MATMUL_CONFIGS_BLOCK_K = [32, 64]
IO_BOUND_MATMUL_CONFIGS_BLOCK_N = [32, 64, 128, 256]
IO_BOUND_MATMUL_CONFIGS_SPLIT_K = [1, 2, 4, 8, 16]


IO_BOUND_MATMUL_CONFIGS = [
    gen_config(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        warps=2 if block_n <= 64 else 4,
        split_k=split_k,
    )
    for stages, block_m, block_k, block_n, split_k in itertools.product(
        IO_BOUND_MATMUL_CONFIGS_STAGES,
        IO_BOUND_MATMUL_CONFIGS_BLOCK_M,
        IO_BOUND_MATMUL_CONFIGS_BLOCK_K,
        IO_BOUND_MATMUL_CONFIGS_BLOCK_N,
        IO_BOUND_MATMUL_CONFIGS_SPLIT_K,
    )
]


TRITON_CONFIGS = BASIC_MATMUL_CONFIGS + INT8_MATMUL_CONFIGS + IO_BOUND_MATMUL_CONFIGS


BACKWARDS_WITH_ME_FIRST = 0
FORWARDS_WITH_ME_LAST = 1

NUM_SPINS_BETWEEN_TIMEOUT_CHECKS = 1000


@triton.jit
def determine_tile(
    A,
    B1,
    B2,
    B3,
    C1,
    C2,
    C3,
    A_my_shard,
    C1_my_shard,
    C2_my_shard,
    C3_my_shard,
    M,
    N1,
    N2,
    N3,
    my_rank,
    world_size,
    direction,
    stride_am,
    stride_bk1,
    stride_bk2,
    stride_bk3,
    stride_bn1,
    stride_bn2,
    stride_bn3,
    stride_cm1,
    stride_cm2,
    stride_cm3,
    stride_cn1,
    stride_cn2,
    stride_cn3,
    BLOCK_M,
    BLOCK_N,
    GROUP_M,
):
    # tl.device_assert(M % world_size == 0)
    M_per_rank = M // world_size
    # matrix multiplication
    pid = tl.program_id(0)
    grid_m_per_rank = tl.cdiv(M_per_rank, BLOCK_M)
    grid_n1 = tl.cdiv(N1, BLOCK_N)
    grid_n2 = tl.cdiv(N2, BLOCK_N)
    grid_n3 = tl.cdiv(N3, BLOCK_N)
    grid_n = grid_n1 + grid_n2 + grid_n3

    # Blocks with lower pid will be executed first (this isn't a documented
    # guarantee, but seems to happen in practice, and Triton already leverages
    # it for its swizzling just below). We want the first blocks to operate on
    # the local rank's shard, since it's immediately available, then once that's
    # all done operate on the first remote contribution to arrive (the one from
    # my_rank - 1), etc. Thus we change the pointers to A and C, and the value
    # of pid, as needed to operate on the input in the order we want.
    blocks_per_rank = grid_m_per_rank * grid_n
    if direction == BACKWARDS_WITH_ME_FIRST:
        other_rank = (my_rank - (pid // blocks_per_rank) + world_size) % world_size
    else:  # direction == FORWARDS_WITH_ME_LAST:
        other_rank = (my_rank + (pid // blocks_per_rank + 1)) % world_size
    pid = pid % blocks_per_rank

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m_per_rank - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    B = tl.where(pid_n < grid_n1, B1, tl.where(pid_n < grid_n1 + grid_n2, B2, B3))
    C = tl.where(pid_n < grid_n1, C1, tl.where(pid_n < grid_n1 + grid_n2, C2, C3))
    C_my_shard = tl.where(
        pid_n < grid_n1,
        C1_my_shard,
        tl.where(pid_n < grid_n1 + grid_n2, C2_my_shard, C3_my_shard),
    )
    stride_bk = tl.where(
        pid_n < grid_n1,
        stride_bk1,
        tl.where(pid_n < grid_n1 + grid_n2, stride_bk2, stride_bk3),
    )
    stride_bn = tl.where(
        pid_n < grid_n1,
        stride_bn1,
        tl.where(pid_n < grid_n1 + grid_n2, stride_bn2, stride_bn3),
    )
    stride_cm = tl.where(
        pid_n < grid_n1,
        stride_cm1,
        tl.where(pid_n < grid_n1 + grid_n2, stride_cm2, stride_cm3),
    )
    stride_cn = tl.where(
        pid_n < grid_n1,
        stride_cn1,
        tl.where(pid_n < grid_n1 + grid_n2, stride_cn2, stride_cn3),
    )
    N = tl.where(pid_n < grid_n1, N1, tl.where(pid_n < grid_n1 + grid_n2, N2, N3))
    pid_n = tl.where(
        pid_n < grid_n1,
        pid_n,
        tl.where(pid_n < grid_n1 + grid_n2, pid_n - grid_n1, pid_n - grid_n1 - grid_n2),
    )

    A = tl.where(
        other_rank == my_rank, A_my_shard, A + other_rank * M_per_rank * stride_am
    )
    C = tl.where(
        other_rank == my_rank, C_my_shard, C + other_rank * M_per_rank * stride_cm
    )

    return (
        A,
        B,
        C,
        M_per_rank,
        N,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        pid_m,
        pid_n,
        other_rank,
        blocks_per_rank,
    )


@triton.jit
def wait_for_recv(
    seq_num,
    wait_counters,
    other_rank,
    my_rank,
    stripe,
    num_stripes,
    _wait,
    do_wait,
    timeout_ns,
):
    if (_wait and do_wait) and other_rank != my_rank:
        wait_counter = wait_counters + other_rank * num_stripes + stripe
        start_time_ns = tl.extra.cuda.globaltimer()
        num_spins = 0
        # There's no atomic_load, hence we simulate it with a CAS.
        while tl.atomic_cas(wait_counter, 0, 0) != seq_num:
            num_spins += 1
            if num_spins == NUM_SPINS_BETWEEN_TIMEOUT_CHECKS:
                if tl.extra.cuda.globaltimer() - start_time_ns > timeout_ns:
                    tl.device_assert(
                        False,
                        "xFormers's fused kernels for sequence parallelism "
                        "timed out waiting for a peer GPU. To prevent "
                        "downstream computations from operating on corrupted "
                        "data, we're bringing the CUDA context down with us.",
                    )
                num_spins = 0


@triton.jit
def do_matmul(
    A,
    B,
    C,
    pid_m,
    pid_n,
    pid_z,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    ACC_TYPE,
    SPLIT_K,
    EVEN_K,
):
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


@triton.jit
def trigger_send(
    seq_num,
    blocks_done_counters,
    write_counters,
    other_rank,
    my_rank,
    num_stripes,
    stripe,
    num_blocks_3d,
    _wait,
    do_write,
):
    if (_wait and do_write) and other_rank != my_rank:
        num_blocks_done = (
            tl.atomic_add(
                blocks_done_counters + other_rank + tl.arange(0, 1),
                1,
                sem="acq_rel",
            )
            + 1
        )
        tl.atomic_xchg(
            write_counters + other_rank * num_stripes + stripe + tl.arange(0, 1),
            seq_num,
            mask=num_blocks_done == num_blocks_3d,
            sem="release",
        )


def our_estimate_matmul_time(B1, C1, N1, N2, N3, **kwargs):
    """Call into Triton's upstream cost model, with the right args

    The upstream function expects arguments to have certain names. Since we
    renamed a few of them in our implementation, we rename them back.

    At the time of writing (July 2023) the arguments that Triton expects are:
    M, N, K, A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages.

    """
    return estimate_matmul_time(N=N1 + N2 + N3, B=B1, C=C1, **kwargs)


@triton.autotune(
    configs=TRITON_CONFIGS,
    key=["M", "N1", "N2", "N3", "K"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": our_estimate_matmul_time,
        "top_k": 10,
    },
    reset_to_zero=["blocks_done_counters"],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit(
    do_not_specialize=[
        "wait_counters",
        "blocks_done_counters",
        "write_counters",
        "do_wait",
        "do_write",
        "direction",
        "stripe",
        "seq_num",
        "num_stripes",
        "_wait",
        "my_rank",
        "world_size",
        "timeout_ns",
    ],
    debug=True,  # To avoid stripping device asserts
)
def _xformers_seqpar_matmul_kernel(
    A_my_shard,
    A,  # DO NOT CHANGE NAME: MUST MATCH PERF MODEL
    B1,
    B2,
    B3,
    C1,
    C2,
    C3,
    C1_my_shard,
    C2_my_shard,
    C3_my_shard,
    wait_counters,
    blocks_done_counters,
    write_counters,
    M,  # DO NOT CHANGE NAME: MUST MATCH PERF MODEL
    N1,
    N2,
    N3,
    K,  # DO NOT CHANGE NAME: MUST MATCH PERF MODEL
    stride_am,
    stride_ak,
    stride_bk1,
    stride_bk2,
    stride_bk3,
    stride_bn1,
    stride_bn2,
    stride_bn3,
    stride_cm1,
    stride_cm2,
    stride_cm3,
    stride_cn1,
    stride_cn2,
    stride_cn3,
    do_wait,
    do_write,
    direction,
    stripe,
    seq_num,
    num_stripes,
    _wait,
    my_rank,
    world_size,
    timeout_ns,
    BLOCK_M: tl.constexpr,  # DO NOT CHANGE NAME: MUST MATCH PERF MODEL
    BLOCK_N: tl.constexpr,  # DO NOT CHANGE NAME: MUST MATCH PERF MODEL
    BLOCK_K: tl.constexpr,  # DO NOT CHANGE NAME: MUST MATCH PERF MODEL
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,  # DO NOT CHANGE NAME: MUST MATCH PERF MODEL
    EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    (
        A,
        B,
        C,
        M,
        N,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        pid_m,
        pid_n,
        other_rank,
        num_blocks_2d,
    ) = determine_tile(
        A,
        B1,
        B2,
        B3,
        C1,
        C2,
        C3,
        A_my_shard,
        C1_my_shard,
        C2_my_shard,
        C3_my_shard,
        M,
        N1,
        N2,
        N3,
        my_rank,
        world_size,
        direction,
        stride_am,
        stride_bk1,
        stride_bk2,
        stride_bk3,
        stride_bn1,
        stride_bn2,
        stride_bn3,
        stride_cm1,
        stride_cm2,
        stride_cm3,
        stride_cn1,
        stride_cn2,
        stride_cn3,
        BLOCK_M,
        BLOCK_N,
        GROUP_M,
    )
    pid_z = tl.program_id(1)

    wait_for_recv(
        seq_num,
        wait_counters,
        other_rank,
        my_rank,
        stripe,
        num_stripes,
        _wait,
        do_wait,
        timeout_ns,
    )

    do_matmul(
        A,
        B,
        C,
        pid_m,
        pid_n,
        pid_z,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        ACC_TYPE,
        SPLIT_K,
        EVEN_K,
    )

    trigger_send(
        seq_num,
        blocks_done_counters,
        write_counters,
        other_rank,
        my_rank,
        num_stripes,
        stripe,
        num_blocks_2d * SPLIT_K,
        _wait,
        do_write,
    )


AUTOTUNED_SIZES: Set[Tuple[int, Tuple[int, ...], int, torch.dtype]] = set()


def common_alignment(*args):
    for div in [16, 8, 4, 2]:
        if all(a % div == 0 for a in args):
            return div
    return 1


def _launch_triton_matmul(
    a_my_shard: Optional[torch.Tensor],
    a: torch.Tensor,
    bs: List[torch.Tensor],
    cs: List[torch.Tensor],
    cs_my_shard: Optional[List[torch.Tensor]],
    my_rank: int,
    world_size: int,
    wait_counters: Optional[torch.Tensor],
    write_counters: Optional[torch.Tensor],
    direction: int,
    stripe: int,
    seq_num: int,
    num_stripes: int,
    timeout_s: int,
    _wait: bool = True,
) -> None:
    # checks constraints
    assert 0 <= my_rank < world_size
    assert 0 <= stripe < num_stripes and 0 <= seq_num < 2**8
    assert direction in (BACKWARDS_WITH_ME_FIRST, FORWARDS_WITH_ME_LAST)

    assert len(bs) == len(cs)
    assert a.ndim == 2
    assert all(b.ndim == 2 for b in bs)
    assert all(c.ndim == 2 for c in cs)
    M, K = a.shape
    Ns = [b.shape[1] for b in bs]
    assert all(b.shape[0] == K for b in bs)
    assert all(c.shape[0] == M for c in cs)
    assert all(c.shape[1] == N for c, N in zip(cs, Ns))
    stride_am, stride_ak = cast(Tuple[int, int], a.stride())
    strides_bk, strides_bn = zip(*(cast(Tuple[int, int], b.stride()) for b in bs))
    strides_cm, strides_cn = zip(*(cast(Tuple[int, int], c.stride()) for c in cs))
    assert stride_am == 1 or stride_ak == 1
    assert all(s == 1 for s in strides_bk) or all(s == 1 for s in strides_bn)
    assert all(s == 1 for s in strides_cm) or all(s == 1 for s in strides_cn)

    if a_my_shard is not None:
        assert a_my_shard.ndim == 2
        assert a_my_shard.shape[0] * world_size == a.shape[0]
        assert a_my_shard.shape[1] == a.shape[1]
        assert a_my_shard.stride() == a.stride()
    else:
        assert a.shape[0] % world_size == 0
        a_my_shard = a.tensor_split(world_size)[my_rank]

    if cs_my_shard is not None:
        assert len(cs_my_shard) == len(cs)
        assert all(c_my_shard.ndim == 2 for c_my_shard in cs_my_shard)
        assert all(
            c_my_shard.shape[0] * world_size == c.shape[0]
            for c, c_my_shard in zip(cs, cs_my_shard)
        )
        assert all(
            c_my_shard.shape[1] == c.shape[1] for c, c_my_shard in zip(cs, cs_my_shard)
        )
        assert all(
            c_my_shard.stride() == c.stride() for c, c_my_shard in zip(cs, cs_my_shard)
        )
    else:
        assert all(c.shape[0] % world_size == 0 for c in cs)
        cs_my_shard = [c.tensor_split(world_size)[my_rank] for c in cs]

    if wait_counters is not None:
        assert wait_counters.shape == (world_size, num_stripes)
        assert wait_counters.dtype is torch.int
        assert wait_counters.is_contiguous()
        do_wait = True
    else:
        do_wait = False
        wait_counters = torch.empty((0,), dtype=torch.int, device=a.device)

    if write_counters is not None:
        assert write_counters.shape == (world_size, num_stripes)
        assert write_counters.dtype is torch.int
        assert write_counters.is_contiguous()
        do_write = True
        blocks_done_counters = torch.empty(
            (world_size,), dtype=torch.int, device=a.device
        )
    else:
        do_write = False
        write_counters = torch.empty((0,), dtype=torch.int, device=a.device)
        blocks_done_counters = torch.empty((0,), dtype=torch.int, device=a.device)

    # accumulator types
    assert all(c.dtype == cs[0].dtype for c in cs)
    ACC_TYPE = (
        tl.float32
        if cs[0].dtype in [torch.float16, torch.bfloat16, torch.float32]
        else tl.int32
    )

    # launch kernel
    def grid(META):
        return (
            world_size
            * triton.cdiv(M // world_size, META["BLOCK_M"])
            * sum(triton.cdiv(N, META["BLOCK_N"]) for N in Ns),
            META["SPLIT_K"],
        )

    # Can be raised if needed.
    assert len(bs) <= 3

    # We auto-tune the kernel's tiling and other parameters for each set of
    # sizes. However, auto-tuning performs a device sync (it has to retrieve
    # timings), which can be problematic: the kernel may busy-wait for something
    # that will only be scheduled later, and the sync would never return. Thus,
    # for auto-tuning, we'd like to set _wait to False, and then set it to True
    # for the real run. (We assume that the kernel is idempotent, and that it
    # won't have a wildly different perf profile when it runs on garbage data
    # compared to real data).

    # Define the args/kwargs corresponding to the default invocation.
    # We can't just use kwargs because Triton expects some args as positional.
    args = (
        a_my_shard,
        a,
        bs[0],
        bs[min(1, len(bs) - 1)],
        bs[min(2, len(bs) - 1)],
        cs[0],
        cs[min(1, len(cs) - 1)],
        cs[min(2, len(cs) - 1)],
        cs_my_shard[0],
        cs_my_shard[min(1, len(cs_my_shard) - 1)],
        cs_my_shard[min(2, len(cs_my_shard) - 1)],
        wait_counters,
        blocks_done_counters,
        write_counters,
        M,
        Ns[0],
        Ns[1] if len(Ns) >= 2 else 0,
        Ns[2] if len(Ns) >= 3 else 0,
        K,
    )
    kwargs = dict(
        stride_am=stride_am,
        stride_ak=stride_ak,
        stride_bk1=strides_bk[0],
        stride_bk2=strides_bk[min(1, len(strides_bk) - 1)],
        stride_bk3=strides_bk[min(2, len(strides_bk) - 1)],
        stride_bn1=strides_bn[0],
        stride_bn2=strides_bn[min(1, len(strides_bn) - 1)],
        stride_bn3=strides_bn[min(2, len(strides_bn) - 1)],
        stride_cm1=strides_cm[0],
        stride_cm2=strides_cm[min(1, len(strides_cm) - 1)],
        stride_cm3=strides_cm[min(2, len(strides_cm) - 1)],
        stride_cn1=strides_cn[0],
        stride_cn2=strides_cn[min(1, len(strides_cn) - 1)],
        stride_cn3=strides_cn[min(2, len(strides_cn) - 1)],
        do_wait=do_wait,
        do_write=do_write,
        direction=direction,
        stripe=stripe,
        seq_num=seq_num,
        num_stripes=num_stripes,
        _wait=_wait,
        my_rank=my_rank,
        world_size=world_size,
        timeout_ns=timeout_s * 1_000_000_000,
        ACC_TYPE=ACC_TYPE,
    )

    # Run without waiting to auto-tune this set of sizes, if needed
    if (M, tuple(Ns), K, cs[0].dtype) not in AUTOTUNED_SIZES:
        kwargs["_wait"] = False
        _xformers_seqpar_matmul_kernel[grid](*args, **kwargs)
        kwargs["_wait"] = _wait
        AUTOTUNED_SIZES.add((M, tuple(Ns), K, cs[0].dtype))

    # Run the actual kernel
    _xformers_seqpar_matmul_kernel[grid](*args, **kwargs)
