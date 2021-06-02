import torch

from xformers.components.attention import Attention, AttentionConfig, register_attention


@register_attention("fourier_mix", AttentionConfig)
class FourierMix(Attention):
    def __init__(self, *args, **kwargs):
        """
        FFT-based pseudo-attention mechanism, from
        "
        "FNet: Mixing Tokens with Fourier Transforms"
        Lee-Thorp et al., 2021, https://arxiv.org/pdf/2105.03824.pdf
        """
        super().__init__()

    def forward(self, q: torch.Tensor, *args, **kwargs):
        fourier_hidden = torch.fft.fft(q, dim=-1)  # FFT on the embedding dimension
        fourier_sequence = torch.fft.fft(
            fourier_hidden, dim=1
        )  # FFT on the sequence dimension
        return torch.real(fourier_sequence)  # only keep the real part, as suggested
