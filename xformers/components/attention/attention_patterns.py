import numpy as np
import torch


# generic nd cases
def _generate_nd_grid(*sizes):
    coords = [torch.arange(s) for s in sizes]
    return torch.meshgrid(*coords)


def local_nd_distance(*sizes, p=2.0, weights=None):
    if weights is None:
        weights = (1,) * len(sizes)
    assert len(sizes) == len(weights)
    grid = _generate_nd_grid(*sizes)
    grid = [i.flatten() * w for i, w in zip(grid, weights)]
    grid = torch.stack(grid, dim=1).float()
    d = torch.cdist(grid, grid, p=p)
    return d


def local_nd_gaussian_distribution(*sizes, sigma=1):
    d = local_nd_distance(*sizes, p=2.0) ** 2
    d = torch.exp(-0.5 * sigma ** (-2.0) * d)
    return d


def local_nd_pattern(*sizes, distance, p=2.0):
    d = local_nd_distance(*sizes, p=p)
    return d < distance


def axial_nd_pattern(*sizes):
    # axial is a special case with p=0 and distance=2
    d = local_nd_distance(*sizes, p=0)
    return d < 2


def random_pattern_from_probability_matrix(dist_matrix, nnz):
    att = torch.zeros_like(dist_matrix, dtype=torch.bool)
    # PyTorch multinomial wrongly doesn't support sampling when number of categories
    # is > 2^24, arguing that it's because it's the max representable consecutive element
    # in fp32 and that the kernels use float32. This is actually not true, and the kernels
    # should work fine if double tensor is passed on CPU. This is a bug that was introduced
    # in https://github.com/pytorch/pytorch/commit/bf04c2ca2f591d98ce57816f0ef0cd20a21bbf66
    # when unifying the checks between CPU and CUDA. For now, just fall-back to numpy
    if dist_matrix.numel() > 2 ** 24:
        dist_matrix = dist_matrix.double()
        dist_matrix /= dist_matrix.sum()
        idxs = np.random.choice(
            dist_matrix.numel(), nnz, p=dist_matrix.flatten(), replace=False
        )
        idxs = torch.as_tensor(idxs)
    else:
        idxs = torch.multinomial(dist_matrix.flatten(), nnz, replacement=False)
    att.view(-1)[idxs] = True
    return att


def global_token_pattern(attention_query_mask: torch.Tensor) -> torch.Tensor:
    assert attention_query_mask.ndim == 1
    assert attention_query_mask.dtype == torch.bool
    attention_query_mask = attention_query_mask[None, :]
    mask = attention_query_mask | attention_query_mask.transpose(1, 0)
    return mask


def random_pattern(attn_size: int, sparsity: float) -> torch.Tensor:
    assert 0 < sparsity < 1
    mask = torch.rand(attn_size, attn_size) > sparsity
    return mask


# 1d-specific cases
def local_1d_pattern(attn_size: int, window_size: int) -> torch.Tensor:
    assert (
        window_size % 2 == 1
    ), "The window size is assumed to be odd (counts self-attention + 2 wings)"
    h_win_size = window_size // 2 + 1
    return local_nd_pattern(attn_size, distance=h_win_size, p=1.0)


def causal_1d_pattern(attn_size: int) -> torch.Tensor:
    mask = torch.tril(torch.ones(attn_size, attn_size, dtype=torch.bool))
    return mask


# 2d-specific cases
def horizontal_axial_2d_distance(H, W, p=2.0):
    d = local_nd_distance(H, W, p=p, weights=(1, 0))
    return d


def vertical_axial_2d_distance(H, W, p=2.0):
    d = local_nd_distance(H, W, p=p, weights=(0, 1))
    return d


def local_2d_distance(H, W, p=2.0):
    return local_nd_distance(H, W, p=p)


def local_2d_gausian_distribution(H, W, sigma=1):
    return local_nd_gaussian_distribution(H, W, sigma=sigma)


def local_2d_pattern(H, W, distance, p=2.0):
    return local_nd_pattern(H, W, distance=distance, p=p)


def axial_2d_pattern(H, W):
    return axial_nd_pattern(H, W)
