import gc
import math
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import triton
from triton.ops.blocksparse import matmul as blocksparse_matmul
from yaml import BlockSequenceStartToken

from xformers.components.attention.attention_patterns import (
    axial_2d_pattern,
    causal_1d_pattern,
    global_token_pattern,
    local_1d_pattern,
    local_2d_pattern,
)
from xformers.components.attention.core import SparseCS, _matmul_with_mask

device = "cuda"
TestCase = namedtuple("TestCase", ["prepare_callable", "mask", "config", "name"])

##############################################
# Plotting utilities
##############################################


def pretty_barplot(results, title, units: str, filename=None, dash_key=""):
    """Graph out the contents of a dict.
    Dash key means that if the result label has this key, then it will be displayed with a dash"""

    if not filename:
        filename = title + ".png"

    # Sanitize the filename
    filename = (
        filename.replace(" ", "_").replace("/", "_").replace("-", "_").replace(":", "")
    )

    xlabels = list(results.keys())
    # Gather all the results in "collumns"
    workloads: Dict[str, Any] = {k: [] for v in results.values() for k in v.keys()}
    for v in results.values():
        for k in v.keys():
            workloads[k].append(float(v[k]))

    options = list(workloads.keys())
    group_len = len(options)
    for key in workloads.keys():
        num_groups = len(workloads[key])
        break
    group_width = group_len + 1

    # Make sure that the plot is big enough
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)

    for idx in range(group_len):
        option = options[idx]
        values = workloads[option]
        xloc = np.arange(1 + idx, group_width * num_groups, group_width)
        plt.bar(xloc, values, width=1, edgecolor="black")

    plt.title(title)
    plt.legend(list(workloads.keys()), loc="upper right")
    plt.ylabel(units)

    ax = plt.gca()
    xticks_loc = np.arange(
        1 + (group_len - 1) / 2.0, group_width * num_groups, group_width
    )
    ax.set_xticks(xticks_loc, xlabels)
    plt.xticks(rotation=45)

    plt.setp(ax.xaxis.get_majorticklabels(), ha="right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.xaxis.grid(color="gray", linestyle="dashed")

    plt.savefig(filename, bbox_inches="tight")
    plt.close(f)


def plot_mask(mask, config, filename):
    sparsity = get_sparsity(mask)
    batch_size = config.batch_size
    num_heads = config.num_heads
    seq_len = config.seq_length

    proxy = torch.ones(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool)
    proxy = triton.testing.mask_tensor(proxy, mask, config.block_size, False)
    proxy = proxy[0][0]

    f = plt.figure()
    plt.imshow(proxy.logical_not(), cmap="gray")
    plt.suptitle("Sparsity = " + str(sparsity) + "%")
    plt.savefig(filename)
    plt.close(f)


##############################################
# Mask and testing utilities
##############################################


def get_mask(MaskGenType, config, config_setter=[]):
    mask_config = Configuration()
    mask_config.init(config)

    # Get the mask
    mask_generator = MaskGenType(mask_config)
    for (key, value) in config_setter:
        mask_generator.set_config_attr(key, value)
    if not mask_generator.is_valid_config():
        return None

    return mask_generator()


def densify_mask(mask, config):
    num_heads = config.num_heads
    seq_length = config.seq_length
    block_size = config.block_size
    dense_mask = torch.zeros(num_heads, seq_length, seq_length)
    for (h, i, j) in zip(*mask.nonzero(as_tuple=True)):
        dense_mask[
            h,
            i * block_size : (i + 1) * block_size,
            j * block_size : (j + 1) * block_size,
        ] = mask[h, i, j]
    return dense_mask


def mask_tensor(a, mask, config):
    return triton.testing.mask_tensor(a, mask, config.block_size, 0.0)


def sparsify_tensor(a, mask, config):
    return triton.testing.sparsify_tensor(a, mask, config.block_size)


def get_sparsity(mask):
    return round((1.0 - mask.sum().item() / mask.numel()) * 100)


##############################################
# Mask Generation
##############################################


@dataclass
class Configuration(object):
    batch_size: int = 32
    num_heads: int = 12
    seq_length: int = 2048
    hidden_size: int = 768  # hidden_size = n_heads * projection_hidden_dimension
    block_size: int = 64

    @property
    def blocked_seq_length(self):
        return int(self.seq_length / self.block_size)

    def init(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return f"bs={self.batch_size},h={self.num_heads},k={self.hidden_size},seq={self.seq_length},bl={self.block_size}"
        # return "-".join([str(getattr(self, x.name)) for x in fields(self)])


class AttentionMask(object):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = Configuration()
        self.config = config

    def is_blocked(self):
        return self.config.block_size != 1

    def is_valid_config(self, keep_blocked=True):
        return True

    def expand(self, mask):
        if mask.ndim == 2:
            return mask.unsqueeze(0).expand(self.config.num_heads, -1, -1)

    def gen_mask(self, keep_blocked=True):
        raise NotImplementedError("Abstract data class")

    def set_config_attr(self, key, value):
        setattr(self.config, key, value)

    def __str__(self):
        raise NotImplementedError("Abstract data type")

    def __call__(self):
        mask = self.gen_mask()
        return mask, self.config, str(self)


class RandomAttentionMask(AttentionMask):
    """
    This is a Random mask. Useful for performance and memory analysis.
    """

    def __init__(self, config=None):
        super(RandomAttentionMask, self).__init__(config)
        self.set_config_attr("mask_prob", 0.5)

    def gen_mask(self, keep_blocked=True):
        seq_length = self.config.seq_length
        if keep_blocked:
            seq_length = self.config.blocked_seq_length

        mask = torch.rand(seq_length, seq_length) > self.config.mask_prob
        return self.expand(mask)

    def __str__(self):
        return "random"


class LowerTriangularAttentionMask(AttentionMask):
    """
    This is a lower triangular mask. This is common in decoder only models.
    This should reduce the computation and memory to roughly half as close to
    half of the mask is zero.

    The mask stays same for each head and each input.

    Nit pick (TODO) - While blocking, we need to ensure that the blocks along
    the diagonals are themselves lower triangular blocks. But, for performance
    measurement, this is ok to ignore as we treat the whole block as useful
    values.
    """

    def __init__(self, config=None):
        super(LowerTriangularAttentionMask, self).__init__(config)

    def gen_mask(self, keep_blocked=True):
        seq_length = self.config.seq_length
        if keep_blocked:
            seq_length = self.config.blocked_seq_length
        return self.expand(causal_1d_pattern(seq_length))

    def __str__(self):
        return "lower_triangular"


class BigBirdAttentionMask(AttentionMask):
    """
    BigBird mask are composed of three types of masks - random, global and window.
    For more details, refer to https://arxiv.org/pdf/2007.14062.pdf

    One point to note is that mask is per head here. So, mask is 3D tensor.
    (num_heads, seq_length, seq_length).
    """

    def __init__(self, config=None):
        super(BigBirdAttentionMask, self).__init__(config)
        self.mask_per_head = True
        self.set_config_attr("num_global_tokens", 2 * self.config.block_size)
        self.set_config_attr("num_random_tokens", 3 * self.config.block_size)
        self.set_config_attr("num_window_tokens", 3 * self.config.block_size)

    def gen_global_mask(self, seq_length):
        # Global tokens are tokens that attend to all tokens and to whom all tokens attend to in the sequence
        num_global_blocks = self.config.num_global_tokens // self.config.block_size
        mask_indices = torch.randint(0, seq_length - 1, size=(num_global_blocks,))
        mask_indices = torch.unique(mask_indices)
        query_mask = torch.zeros(seq_length).to(dtype=torch.bool)
        query_mask.scatter_(0, mask_indices, True)
        return global_token_pattern(query_mask)

    def gen_random_mask(self, seq_length):
        # Each query token attends over r random number of tokens
        num_random_blocks = self.config.num_random_tokens // self.config.block_size
        mask_indices = torch.randint(
            0, seq_length - 1, size=(seq_length, num_random_blocks)
        )
        random_mask = torch.zeros(seq_length, seq_length).to(dtype=torch.bool)
        random_mask.scatter_(1, mask_indices, True)
        return random_mask

    def gen_window_mask(self, seq_length):
        num_window_blocks = self.config.num_window_tokens // self.config.block_size
        if num_window_blocks % 2 == 0:
            num_window_blocks += 1
        return local_1d_pattern(seq_length, num_window_blocks)

    def gen_mask(self, keep_blocked=True):
        seq_length = self.config.seq_length
        if keep_blocked == True:
            seq_length = self.config.blocked_seq_length
        assert (
            keep_blocked == True
        ), "Not implemented, call to_dense later to get full tensor"

        if self.mask_per_head:
            head_masks = []

            for _ in range(self.config.num_heads):
                global_mask = self.gen_global_mask(seq_length)
                random_mask = self.gen_random_mask(seq_length)
                window_mask = self.gen_window_mask(seq_length)

                mask = global_mask + random_mask + window_mask
                head_masks.append(mask)

            mask = torch.stack(head_masks)
        else:
            global_mask = self.gen_global_mask(seq_length)
            random_mask = self.gen_random_mask(seq_length)
            window_mask = self.gen_window_mask(seq_length)

            mask = global_mask + random_mask + window_mask
            mask = self.expand(mask)
        return mask

    def __str__(self):
        return "bigbird"


class AxialAttentionMask(AttentionMask):
    """
    BigBird mask are composed of three types of masks - random, global and window.
    For more details, refer to https://arxiv.org/pdf/2007.14062.pdf

    One point to note is that mask is per head here. So, mask is 3D tensor.
    (num_heads, seq_length, seq_length).
    """

    def __init__(self, config=None):
        super(AxialAttentionMask, self).__init__(config)
        if config is None:
            self.set_config_attr("seq_length", 1024)

    def is_valid_config(self, keep_blocked=True):
        seq_length = self.config.seq_length
        if keep_blocked:
            seq_length = self.config.blocked_seq_length
        H = int(math.sqrt(seq_length))
        if H * H == seq_length:
            return True
        return False

    def gen_mask(self, keep_blocked=True):
        seq_length = self.config.seq_length
        if keep_blocked:
            seq_length = self.config.blocked_seq_length
        H = int(math.sqrt(seq_length))
        assert H * H == seq_length, f"H={H}, seq_length={seq_length}"
        return self.expand(axial_2d_pattern(H, H))

    def __str__(self):
        return "axial"


class LocalAttentionMask(AttentionMask):
    """
    BigBird mask are composed of three types of masks - random, global and window.
    For more details, refer to https://arxiv.org/pdf/2007.14062.pdf

    One point to note is that mask is per head here. So, mask is 3D tensor.
    (num_heads, seq_length, seq_length).
    """

    def __init__(self, config=None):
        super(LocalAttentionMask, self).__init__(config)
        self.set_config_attr("num_local_blocks", 3)
        if config is None:
            self.set_config_attr("seq_length", 1024)

    def is_valid_config(self, keep_blocked=True):
        seq_length = self.config.seq_length
        if keep_blocked:
            seq_length = self.config.blocked_seq_length
        H = int(math.sqrt(seq_length))
        if H * H == seq_length:
            return True
        return False

    def gen_mask(self, keep_blocked=True):
        seq_length = self.config.seq_length
        if keep_blocked:
            seq_length = self.config.blocked_seq_length
        H = int(math.sqrt(seq_length))
        assert H * H == seq_length, f"H={H}, seq_length={seq_length}"
        return self.expand(local_2d_pattern(H, H, self.config.num_local_blocks))

    def __str__(self):
        return "local"


##############################################
# Class to organize the experiments
##############################################


class Experiment(object):
    def __init__(self, mode, dtype, do_accuracy_check):
        self.mode = mode
        self.dtype = dtype
        self.do_accuracy_check = do_accuracy_check

    def reset_results(self):
        self.results = {}
        self.results["flops"] = {}
        self.results["time"] = {}
        self.results["memory"] = {}
        self.results["speedup"] = {}
        self.results["memory_savings"] = {}

    def do_mem(sel, fn):
        # bookeeping
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # actually run the function
        fn()
        fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() // 2 ** 20

    def gen_config(self):
        raise NotImplementedError("Not setup")

    def plot(self, sparsity, pattern_name):
        raise NotImplementedError("Not setup")

    def run(self):
        raise NotImplementedError("Not setup")

    def bench_all(
        self, a, b, tests, mask_config, sparsity, baseline_name, op_flops, dict_key
    ):

        if self.do_accuracy_check:
            self.check_all(tests, a, b)

        for testcase in tests:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                fn = testcase.prepare_callable(a, b, testcase.mask, testcase.config)
                ms = triton.testing.do_bench(fn)[0]
                flops = op_flops / ms * 1e3  # TFlop per second
                mem = self.do_mem(fn)
            except:
                # raise
                ms = -1
                flops = -1
                mem = -1

            def add_kv(d, d_key, d_value):
                d_value = max(0, d_value)
                if d_key not in d:
                    d[d_key] = {}
                d[d_key][testcase.name] = d_value

            # Write into results
            # dict_key = f"sp={sparsity}%,{mask_config}"
            add_kv(self.results["time"], dict_key, ms)
            add_kv(self.results["flops"], dict_key, flops)
            add_kv(self.results["memory"], dict_key, mem)

            speedup = self.results["time"][dict_key][baseline_name] / ms
            memory_savings = self.results["memory"][dict_key][baseline_name] / mem
            add_kv(self.results["speedup"], dict_key, speedup)
            add_kv(self.results["flops"], dict_key, flops)
            add_kv(self.results["memory_savings"], dict_key, memory_savings)

            print(
                f"{testcase.name} --> {mask_config}, sparsity={sparsity}, ops={op_flops}, time={ms}, tflops={flops}, mem={mem}"
            )

    def get_inputs(self, config, device="cuda"):
        # if mode = sddmm, a, b = query, key
        # if mode = spmm, a, b = attn, value
        if self.mode == "sddmm":
            return [
                torch.randn(
                    config.batch_size,
                    config.num_heads,
                    config.seq_length,
                    config.hidden_size // config.num_heads,
                    device=device,
                    dtype=self.dtype,
                )
                for _ in range(2)
            ]
        else:
            assert self.mode == "spmm"
            attn = torch.randn(
                config.batch_size,
                config.num_heads,
                config.seq_length,
                config.seq_length,
                device=device,
                dtype=self.dtype,
            )
            value = torch.randn(
                config.batch_size,
                config.num_heads,
                config.seq_length,
                config.hidden_size // config.num_heads,
                device=device,
                dtype=self.dtype,
            )
            return [attn, value]

    def torch_matmul_callable(self, a, b, mask, config):
        input_a = mask_tensor(a, mask, config) if self.mode == "spmm" else a
        input_b = b.transpose(-1, -2) if self.mode == "sddmm" else b

        def torch_fn():
            return torch.matmul(input_a, input_b)

        return torch_fn

    def get_triton_fn(self, mask, config, mode="sddmm"):
        if mode == "sddmm":
            return blocksparse_matmul(
                layout=mask,
                block=config.block_size,
                mode="sdd",
                device="cuda",
                trans_a=False,
                trans_b=True,
            )
        else:
            assert mode == "spmm"
            return blocksparse_matmul(
                layout=mask,
                block=config.block_size,
                mode="dsd",
                device="cuda",
                trans_a=False,
                trans_b=False,
            )

    def triton_callable(self, a, b, mask, config):
        triton_kernel = self.get_triton_fn(mask, config, self.mode)
        input_a = sparsify_tensor(a, mask, config) if self.mode == "spmm" else a
        input_b = b

        def triton_fn():
            return triton_kernel(input_a, input_b)

        return triton_fn

    def prepare_sputnik_inputs(self, query, key, config, mask):
        # - sparse / sputnik
        mask_cs = torch.ones(
            [config.batch_size, config.num_heads, config.seq_length, config.seq_length],
            dtype=torch.bool,
            device="cuda",
        )
        mask_cs = triton.testing.mask_tensor(
            mask_cs, mask, config.block_size, value=False
        )
        # Sputnik kernels only handle fp32
        query_cs = query.flatten(start_dim=0, end_dim=1).to(torch.float32)
        key_cs = key.flatten(start_dim=0, end_dim=1).to(torch.float32)

        query_cs = query_cs.contiguous()
        key_cs = key_cs.transpose(-2, -1)

        sparse_mask_cs = SparseCS(
            mask_cs.flatten(start_dim=0, end_dim=1).contiguous(),
            device=torch.device("cuda"),
        )
        return query_cs, key_cs, sparse_mask_cs

    def sputnik_callable(self, a, b, mask, config):
        assert self.mode == "sddmm"
        a_cs, b_cs, sparse_mask_cs = self.prepare_sputnik_inputs(a, b, config, mask)

        def sputnik_fn():
            return _matmul_with_mask(a_cs, b_cs, sparse_mask_cs)

        return sputnik_fn

    def get_op_flops(self, mask, config):
        # Measure total compute ops
        op_flops = (
            2  # FMA
            * config.batch_size  # batched matmul
            * (config.hidden_size // config.num_heads)  # Reduce dimension
            * float(mask.sum())
            * config.block_size
            * config.block_size  # Effective seq length * seq_length
            * 1e-12  # TFlops
        )
        return op_flops

    def check_all(self, tests, a, b):
        ref_test = tests[0]
        ref_out = ref_test.prepare_callable(a, b, ref_test.mask, ref_test.config)()

        res_test = tests[1]
        res_out = res_test.prepare_callable(a, b, res_test.mask, res_test.config)()

        self.check_accuracy(ref_out, res_out, ref_test.mask, ref_test.config)

    def check_accuracy(self, ref_full, res_bsr, mask, config):
        if self.mode == "sddmm":
            # Get the dense representation of the bsr tensor
            # Use triton sparse * dense multiplication to get the dense tensor back
            sparse_dot_dsd = blocksparse_matmul(
                layout=mask,
                block=config.block_size,
                mode="dsd",
                device="cuda",
                trans_a=False,
                trans_b=False,
            )
            identity = torch.eye(
                config.seq_length, config.seq_length, device=device, dtype=self.dtype
            )

            identity = identity.expand(config.batch_size, config.num_heads, -1, -1)
            res = sparse_dot_dsd(res_bsr, identity)

            # Get the res where values are masked. Expand the blocked mask
            # ref = triton.testing.mask_tensor(ref_full, mask, config.block_size)
            full_mask = densify_mask(mask, config)
            ref = ref_full * full_mask.to(dtype=self.dtype, device=device)
            try:
                assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)
            except RuntimeError:
                pass
            except AssertionError:
                raise
        else:
            assert self.mode == "spmm"
            # Both are dense outputs
            try:
                assert torch.allclose(ref_full, res_bsr, atol=1e-3, rtol=1e-3)
            except RuntimeError:
                pass
            except AssertionError:
                import pdb

                pdb.set_trace()
                raise


class DifferentPatternExperiment(Experiment):
    """
    In this experiment, we check if sparsity pattern (like bigbird, lower triangular
    etc) changes the performance of different kernels. The idea is to check if
    changing sparsity pattern, while keeping total sparsity ratio same, leads to any
    perforamnce differences.

    We will perform two experiments

    1) LowerTraingularMask vs RandomMask - Both have ~50% sparsity.
    2) BigBird Mask vs RandomMask - Both have same sparsity.
    """

    def __init__(self, mode, dtype, do_accuracy_check):
        super(DifferentPatternExperiment, self).__init__(mode, dtype, do_accuracy_check)

    def gen_config(self):
        batch_sizes = [32]
        heads = [16]
        seq_lengths = [1024, 2048]
        block_sizes = [64]
        hidden_sizes = [1024, 4096, 8192]
        for batch in batch_sizes:
            for hidden_size in hidden_sizes:
                for head in heads:
                    for seq in seq_lengths:
                        for block in block_sizes:
                            entry = {
                                "batch_size": batch,
                                "num_heads": head,
                                "seq_length": seq,
                                "block_size": block,
                                "hidden_size": hidden_size,
                            }
                            yield entry

    def plot(self, sparsity, pattern_name):
        pretty_barplot(
            self.results["speedup"],
            title=f"Same Sparsity ({sparsity}%) - speedup - {self.mode} and {self.dtype}",
            filename=f"same_sparsity_{self.mode}_{self.dtype}_{pattern_name}_time.svg",
            dash_key="pytorch",
            units="Speedup normalized to torch_matmul",
        )

        pretty_barplot(
            self.results["flops"],
            title=f"Same Sparsity ({sparsity}%) - throughput - {self.mode} and {self.dtype}",
            filename=f"same_sparsity_{self.mode}_{self.dtype}_{pattern_name}_flops.svg",
            dash_key="pytorch",
            units="TFlops/s",
        )

        pretty_barplot(
            self.results["memory_savings"],
            title=f"Same Sparsity ({sparsity}%) - memory savings - {self.mode} and {self.dtype}",
            filename=f"same_sparsity_{self.mode}_{self.dtype}_{pattern_name}_memory.svg",
            dash_key="pytorch",
            units="Memory savings normalized to torch_matmul",
        )

    def run(self):
        for MaskGenType in [LowerTriangularAttentionMask, BigBirdAttentionMask]:
            self.reset_results()
            for config in self.gen_config():
                # Get pattern mask
                pattern_mask, pattern_config, pattern_name = get_mask(
                    MaskGenType, config
                )
                sparsity = get_sparsity(pattern_mask)
                mask_prob = sparsity / 100

                # Get random mask
                random_mask, random_config, _ = get_mask(
                    RandomAttentionMask,
                    config,
                    [("mask_prob", mask_prob)],
                )
                print(f"{pattern_name} sparsity", get_sparsity(pattern_mask))
                print("Random sparsity", get_sparsity(random_mask))

                # Create input tensors
                a, b = self.get_inputs(random_config)

                tests = []
                baseline_name = "torch-matmul"
                tests.append(
                    TestCase(
                        self.torch_matmul_callable,
                        random_mask,
                        random_config,
                        f"{baseline_name}",
                    )
                )
                tests.append(
                    TestCase(
                        self.triton_callable,
                        random_mask,
                        random_config,
                        f"triton-random",
                    )
                )
                tests.append(
                    TestCase(
                        self.triton_callable,
                        pattern_mask,
                        pattern_config,
                        f"triton-{pattern_name}",
                    )
                )
                if self.mode == "sddmm":
                    tests.append(
                        TestCase(
                            self.sputnik_callable,
                            random_mask,
                            random_config,
                            f"sputnik-random",
                        )
                    )
                    tests.append(
                        TestCase(
                            self.sputnik_callable,
                            pattern_mask,
                            pattern_config,
                            f"sputnik-{pattern_name}",
                        )
                    )

                self.bench_all(
                    a,
                    b,
                    tests,
                    random_config,
                    sparsity,
                    baseline_name,
                    self.get_op_flops(random_mask, random_config),
                    f"{random_config}",
                )

                self.plot(sparsity, pattern_name)


class VarySparsityExperiment(Experiment):
    """
    In this experiment, we check how sparsity ration affects the performance.
    """

    def __init__(self, mode, dtype, do_accuracy_check):
        super(VarySparsityExperiment, self).__init__(mode, dtype, do_accuracy_check)

    def gen_config(self):
        batch_sizes = [32]
        heads = [16]
        seq_lengths = [2048]
        hidden_sizes = [1024, 8192]
        block_sizes = [64]
        for batch in batch_sizes:
            for seq in seq_lengths:
                for head in heads:
                    for block in block_sizes:
                        for hidden_size in hidden_sizes:
                            entry = {
                                "batch_size": batch,
                                "num_heads": head,
                                "seq_length": seq,
                                "block_size": block,
                                "hidden_size": hidden_size,
                            }
                            yield entry

    def plot(self, sparsity, pattern_name):
        pretty_barplot(
            self.results["speedup"],
            title=f"Varying Sparsity - speedup - {self.mode} and {self.dtype}",
            filename=f"vary_sparsity_{self.mode}_{self.dtype}_{pattern_name}_time.svg",
            dash_key="pytorch",
            units="Speedup normalized to torch_matmul",
        )

        pretty_barplot(
            self.results["flops"],
            title=f"Varying Sparsity - throughput - {self.mode} and {self.dtype}",
            filename=f"vary_sparsity_{self.mode}_{self.dtype}_{pattern_name}_flops.svg",
            dash_key="pytorch",
            units="TFlops/s",
        )

        pretty_barplot(
            self.results["memory_savings"],
            title=f"Varying Sparsity - memory savings - {self.mode} and {self.dtype}",
            filename=f"vary_sparsity_{self.mode}_{self.dtype}_{pattern_name}_memory.svg",
            dash_key="pytorch",
            units="Memory savings normalized to torch_matmul",
        )

    def run(self):
        self.reset_results()
        for config in self.gen_config():
            for x in range(10, 100, 10):
                mask_prob = x / 100.0
                # Get random mask
                random_mask, random_config, _ = get_mask(
                    RandomAttentionMask,
                    config,
                    [
                        ("mask_prob", mask_prob),
                    ],
                )
                sparsity = get_sparsity(random_mask)
                print("Random sparsity", get_sparsity(random_mask))

                # Create input tensors
                a, b = self.get_inputs(random_config)

                tests = []
                baseline_name = "torch-matmul"
                tests.append(
                    TestCase(
                        self.torch_matmul_callable,
                        random_mask,
                        random_config,
                        f"{baseline_name}",
                    )
                )
                tests.append(
                    TestCase(
                        self.triton_callable,
                        random_mask,
                        random_config,
                        f"triton-random",
                    )
                )
                if self.mode == "sddmm":
                    tests.append(
                        TestCase(
                            self.sputnik_callable,
                            random_mask,
                            random_config,
                            f"sputnik-random",
                        )
                    )
                dict_key = f"sp={mask_prob},{random_config}"
                self.bench_all(
                    a,
                    b,
                    tests,
                    random_config,
                    sparsity,
                    baseline_name,
                    self.get_op_flops(random_mask, random_config),
                    dict_key,
                )
        self.plot(None, "random")


class BlockSizeExperiment(Experiment):
    """
    In this experiment, we analyze how increasing the block size affects
    performance.  We will take the lower triangular pattern. As we increase the
    batch size, the blocks near the diagonal have to do more unnecessary
    computation (the effective sparsity starts decreasing).
    """

    def __init__(self, mode, dtype, do_accuracy_check):
        super(BlockSizeExperiment, self).__init__(mode, dtype, do_accuracy_check)

    def gen_config(self):
        batch_sizes = [32]
        heads = [16]
        seq_lengths = [2048]
        block_sizes = [32, 64, 128, 256]
        hidden_sizes = [1024, 8192]
        for batch in batch_sizes:
            for seq in seq_lengths:
                for hidden_size in hidden_sizes:
                    for block in block_sizes:
                        for head in heads:
                            entry = {
                                "batch_size": batch,
                                "num_heads": head,
                                "seq_length": seq,
                                "block_size": block,
                                "hidden_size": hidden_size,
                            }
                            yield entry

    def plot(self, sparsity, pattern_name):
        pretty_barplot(
            self.results["speedup"],
            title=f"Varying Block size - speedup - {self.mode} and {self.dtype}",
            filename=f"vary_block_size_{self.mode}_{self.dtype}_{pattern_name}_time.svg",
            dash_key="pytorch",
            units="Speedup normalized to torch matmul",
        )

        pretty_barplot(
            self.results["flops"],
            title=f"Varying Block size - throughput - {self.mode} and {self.dtype}",
            filename=f"vary_block_size_{self.mode}_{self.dtype}_{pattern_name}_flops.svg",
            dash_key="pytorch",
            units="TFlops/s",
        )

        pretty_barplot(
            self.results["memory_savings"],
            title=f"Varying Block size - memory_savings - {self.mode} and {self.dtype}",
            filename=f"vary_block_size_{self.mode}_{self.dtype}_{pattern_name}_memory.svg",
            dash_key="pytorch",
            units="Memory savings normalized to torch matmul",
        )

    def get_op_flops(self, mask, config):
        # Op flops here refer to the original non blocked attention mask, where
        # no unnecessary elements are unmasked. We can compute this by computing
        # the total flops of batch matmul and then multiply by (n+1)/2n.
        num_masked_elems = (config.seq_length + 1) / (2.0 * config.seq_length)
        op_flops = (
            2  # FMA
            * config.batch_size  # batched matmul
            * config.num_heads
            * (config.hidden_size // config.num_heads)  # Reduce dimension
            * config.seq_length
            * config.seq_length
            * num_masked_elems
            * 1e-12  # TFlops
        )
        return op_flops

    def run(self):
        self.reset_results()
        for config in self.gen_config():
            lt_mask, lt_config, lt_name = get_mask(
                LowerTriangularAttentionMask,
                config,
            )
            sparsity = get_sparsity(lt_mask)
            print("Effective sparsity", sparsity)
            if lt_config.seq_length == 2048:
                plot_mask(lt_mask, lt_config, f"lt_mask_{lt_config.block_size}.svg")

            # Create input tensors
            a, b = self.get_inputs(lt_config)

            tests = []
            baseline_name = "torch-matmul"
            tests.append(
                TestCase(
                    self.torch_matmul_callable, lt_mask, lt_config, f"{baseline_name}"
                )
            )
            tests.append(
                TestCase(self.triton_callable, lt_mask, lt_config, f"triton-random")
            )
            if self.mode == "sddmm":
                tests.append(
                    TestCase(
                        self.sputnik_callable, lt_mask, lt_config, f"sputnik-random"
                    )
                )
            dict_key = f"sp={sparsity}%,{lt_config}"
            self.bench_all(
                a,
                b,
                tests,
                lt_config,
                sparsity,
                baseline_name,
                self.get_op_flops(lt_mask, lt_config),
                dict_key,
            )
        self.plot(None, lt_name)


if __name__ == "__main__":
    for MaskGen in [
        RandomAttentionMask,
        LowerTriangularAttentionMask,
        BigBirdAttentionMask,
        AxialAttentionMask,
        LocalAttentionMask,
    ]:
        mask_gen = MaskGen()
        mask, config, name = mask_gen()
        plot_mask(mask, config, f"{name}.svg")

    for mode in ["sddmm", "spmm"]:
        DifferentPatternExperiment(mode, torch.float16, True).run()
        VarySparsityExperiment(mode, torch.float16, True).run()
        BlockSizeExperiment(mode, torch.float16, True).run()
