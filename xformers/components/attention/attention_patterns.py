import torch


def local_1d_pattern(attn_size: int, window_size: int) -> torch.Tensor:
    assert (
        window_size % 2 == 1
    ), "The window size is assumed to be odd (counts self-attention + 2 wings)"
    h_win_size = window_size // 2

    attn_shape = (attn_size, attn_size)
    full_attn = torch.ones(attn_shape, dtype=torch.bool)

    mask = torch.tril(full_attn, diagonal=h_win_size)
    mask &= ~torch.tril(full_attn, diagonal=-(h_win_size + 1))
    return mask


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


def causal_1d_pattern(attn_size: int) -> torch.Tensor:
    mask = torch.tril(torch.ones(attn_size, attn_size, dtype=torch.bool))
    return mask
