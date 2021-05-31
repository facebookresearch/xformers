import pytest
import torch

import xformers.components.attention.attention_patterns as AP


# baseline implementations
def _local_1d_pattern(attn_size: int, window_size: int) -> torch.Tensor:
    assert (
        window_size % 2 == 1
    ), "The window size is assumed to be odd (counts self-attention + 2 wings)"
    h_win_size = window_size // 2

    attn_shape = (attn_size, attn_size)
    full_attn = torch.ones(attn_shape, dtype=torch.bool)

    mask = torch.tril(full_attn, diagonal=h_win_size)
    mask &= ~torch.tril(full_attn, diagonal=-(h_win_size + 1))
    return mask


def _generate_2d_grid(H, W):
    i = torch.arange(H)
    j = torch.arange(W)
    i, j = torch.meshgrid(i, j)
    return i, j


def _horizontal_axial_2d_distance(H, W, p=2.0):
    i, _ = _generate_2d_grid(H, W)
    ij = i.reshape(-1, 1).float()
    d = torch.cdist(ij, ij, p=p)
    return d


def _vertical_axial_2d_distance(H, W, p=2.0):
    _, j = _generate_2d_grid(H, W)
    ij = j.reshape(-1, 1).float()
    d = torch.cdist(ij, ij, p=p)
    return d


def _local_2d_distance(H, W, p=2.0):
    # axial is a special case with p=0 and distance=2
    i, j = _generate_2d_grid(H, W)
    ij = torch.stack([i.flatten(), j.flatten()], 1).float()
    d = torch.cdist(ij, ij, p=p)
    return d


def _local_2d_gaussian_distribution(H, W, sigma=1.0):
    d = _local_2d_distance(H, W, p=2.0) ** 2
    d = torch.exp(-0.5 * sigma ** (-2.0) * d)
    return d


@pytest.mark.parametrize("window_size", [3, 7, 11])
@pytest.mark.parametrize("attn_size", [50, 51, 64])
def test_local_1d_pattern(attn_size, window_size):
    mask = AP.local_1d_pattern(attn_size, window_size).float()
    mask_ref = _local_1d_pattern(attn_size, window_size).float()
    assert torch.allclose(mask, mask_ref)


@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("W", [5, 7, 10])
@pytest.mark.parametrize("H", [5, 7, 10])
def test_horizontal_axial_2d_distance(H, W, p):
    d = AP.horizontal_axial_2d_distance(H, W, p=p)
    d_ref = _horizontal_axial_2d_distance(H, W, p=p)
    assert torch.allclose(d, d_ref)


@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("W", [5, 7, 10])
@pytest.mark.parametrize("H", [5, 7, 10])
def test_vertical_axial_2d_distance(H, W, p):
    d = AP.vertical_axial_2d_distance(H, W, p=p)
    d_ref = _vertical_axial_2d_distance(H, W, p=p)
    assert torch.allclose(d, d_ref)


@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("W", [5, 7, 10])
@pytest.mark.parametrize("H", [5, 7, 10])
def test_local_2d_distance(H, W, p):
    d = AP.local_2d_distance(H, W, p=p)
    d_ref = _local_2d_distance(H, W, p=p)
    assert torch.allclose(d, d_ref)


@pytest.mark.parametrize("sigma", [0.5, 1, 2])
@pytest.mark.parametrize("W", [5, 7, 10])
@pytest.mark.parametrize("H", [5, 7, 10])
def test_local_2d_gaussian_distribution(H, W, sigma):
    d = AP.local_2d_gausian_distribution(H, W, sigma=sigma)
    d_ref = _local_2d_gaussian_distribution(H, W, sigma=sigma)
    assert torch.allclose(d, d_ref)
