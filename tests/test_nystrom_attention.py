import torch

from xformers.components.attention import NystromAttention, ScaledDotProduct


def test_nystrom_attention():
    b, s, d = 8, 900, 384
    seed = 42
    torch.random.manual_seed(seed)

    def test_close_to_sdp():
        # Make sure that Nystrom and Normal attention are not too far off.
        a = torch.rand(b, s, d)
        nystrom_config = {
            "name": "nystrom",
            "dropout": 0.0,
            "num_landmarks": 30,
            "num_heads": 2,
        }

        sdp_config = {
            "name": "scaled_dot_product",
            "dropout": 0.0,
        }

        nystrom_attention = NystromAttention(**nystrom_config)
        sdp_attention = ScaledDotProduct(**sdp_config)

        r_nystrom = nystrom_attention(a, a, a)
        r_sdp = sdp_attention(a, a, a)

        assert torch.allclose(r_nystrom, r_sdp, rtol=0.005, atol=1e-2)

    test_close_to_sdp()
