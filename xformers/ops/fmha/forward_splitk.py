import torch
from typing import Any, List, Set, Tuple, Optional
from xformers.ops.common import get_xformers_operator, register_operator
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from xformers.ops.fmha.common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1

@register_operator
class FwOp(AttentionFwOpBase):
    
    OPERATOR = get_xformers_operator("efficient_attention_forward_decoder_splitk_ck")
    SUPPORTED_DEVICES = {"cuda"}
    SUPPORTED_DTYPES = {
        torch.half,
        torch.bfloat16,
    }  # Those are dtypes of Q. In the quantized case K/V has dtype int32
    SUPPORTED_MAX_K = 128
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
    }
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    NAME = "ck_splitKF"

    SPLIT_K: Optional[int] = None
    BLOCK_M = 16
    BLOCK_N = 64

    NUM_GROUPS = 1  # Default quantization is row-wise

    @classmethod
    def shape_not_supported_reasons(
        cls, Mq: int, Mkv: int, K: int, Kv: int
    ) -> List[str]:
        reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
        if K not in {16, 32, 64, 128}:
            reasons.append(f"Embed dim {K} not supported")
        return reasons

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        if d.key.dtype != torch.int32:
            check_lastdim_alignment_stride1(reasons, "key", d.key, 8)
            check_lastdim_alignment_stride1(reasons, "value", d.value, 8)
        if cls.OPERATOR is None:
            reasons.append("triton is not available")
        if d.device.type == "cuda":
            # Has only been tested on 8.0 / 9.0.
            if torch.cuda.get_device_capability(d.device) < (7, 0):
                reasons.append(
                    "requires GPU with sm80 minimum compute capacity, e.g., A100/H100/L4"
                )

        q_len = d.query.shape[1]
        if isinstance(d.attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask):
            seqinfo = d.attn_bias.q_seqinfo
            if q_len != seqinfo.seqstart_py[-1]:
                reasons.append(
                    f"Expected total {seqinfo.seqstart_py[-1]} queries not {q_len}"
                )
            q_len = seqinfo.min_seqlen
            if q_len != seqinfo.max_seqlen:
                reasons.append(
                    "Variable query len is not supported in the presence of causal mask."
                )

        if d.key.ndim in [4, 5] and d.key.shape[-2] != 1:
            if d.key.stride(-2) == 0 and d.value.stride(-2) == 0 and q_len > 1:
                reasons.append("multiquery is only supported with query seqlen=1")

        if d.attn_bias is not None and q_len > 1:
            reasons.append(
                "query with seqlen > 1 is not supported in the presence of causal mask"
            )
        return reasons

    @classmethod
    def get_split_k(cls, B: int, H: int, Mk: int) -> int:
        """Heuristic for the number of splits"""
        bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
        split_k = max(Mk, 1024) // bh
        max_chunk_size = 64 if Mk <= 512 and bh <= 64 else 128
        while split_k > 0 and Mk / split_k < max_chunk_size:
            split_k = split_k // 2
        split_k = min(split_k, 64)
        split_k = max(split_k, 1)
        return split_k

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        attn_bias = inp.attn_bias
        seq_len = None
        q, k, v = inp.get_qkv_in_bmghk()
        
        if attn_bias is not None:
            assert isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
            seq_len = attn_bias.k_seqinfo.seqlen
            B = len(seq_len)
            G, H, Kq = q.shape[-3:]
            Kkv = v.shape[-1]

            # assume kv has been padded
            q = q.reshape(B, -1, G, H, Kq)
            k = k.reshape(B, -1, G, H, Kkv)
            v = v.reshape(B, -1, G, H, Kkv)
        
        mqa_swap_seqlen_head = False
        if k.shape[3] > 1 and k.stride(3) == 0 and v.stride(3) == 0:
            mqa_swap_seqlen_head = True
            assert q.shape[1] == 1
            q = q.transpose(1, 3)
            k = k[:, :, :, :1]
            v = v[:, :, :, :1]

        Lk = k.shape[-1]

        B, Mk, G, H, Kkv = k.shape
        B, M, G, H, Kq = q.shape
        assert Lk == Kq, f"Keys have head dim {Lk} but queries have head dim {Kq}"

        BLOCK_M = cls.BLOCK_M
        BLOCK_N = cls.BLOCK_N
        if cls.SPLIT_K is not None:
            split_k = cls.SPLIT_K
        else:
            # Use heuristics
            split_k = cls.get_split_k(B, H, Mk)

        M_ceil = (M + BLOCK_M - 1) // BLOCK_M * BLOCK_M

        # o_splitk = torch.empty(
        #     [B * G * H, split_k, M_ceil, Kq], dtype=torch.float32, device=q.device
        # )
        # metadata = torch.empty(
        #     [B * G * H, 2, split_k, M_ceil], dtype=torch.float32, device=q.device
        # )

        if inp.scale is not None:
            qk_scale = inp.scale
        else:
            qk_scale = torch.rsqrt(torch.tensor(k.shape[-1], dtype=torch.float32))

        out = cls.OPERATOR(query=q, key=k, value=v, seq_positions=seq_len, scale=qk_scale, split_k=split_k)

        return out, None

