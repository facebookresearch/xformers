import torch

from xformers.components.attention import BlockSparseAttention,BlockSparseLocalAttention, BlockNoglobalAttention
from xformers.components import MultiHeadDispatch

# BATCH = 1
# HEADS = 16
# SEQ = 4096
# EMB = 64
# BLOCK_SIZE = 512
# DROPOUT = 0.1

BATCH = 1
HEADS = 16
SEQ = 8192
EMB = 64 * HEADS
BLOCK_SIZE = 512
BLOCK_UNIT = 64
DROPOUT = 0
dtype = torch.float16

blocks = SEQ // BLOCK_UNIT


def pattern_to_layout(mask: torch.Tensor, block_size: int) -> torch.Tensor:
    r"""
    Given a mask pattern and blocksize, return the corresponding layout
    which makes sure that all the positives in the mask are covered
    """
    assert mask.ndim >= 2, "We're expecting [Heads, Seq, Seq] or [Seq, Seq]"
    _should_squeeze = False

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
        _should_squeeze = True

    assert (
        mask.shape[1] % block_size == 0 and mask.shape[2] % block_size == 0
    ), "We're only handling masks divisible by block_size"

    # Now mark the mask
    layout = torch.nn.functional.max_pool2d(
        mask.to(torch.float), kernel_size=block_size, stride=block_size
    )
    layout = layout.to(torch.long)

    if _should_squeeze:
        layout.squeeze_(0)

    return layout

local_mask = torch.zeros(HEADS, SEQ, SEQ)
for block_start in range(0, SEQ, BLOCK_SIZE):
    local_mask[:, block_start:block_start+BLOCK_SIZE, block_start:block_start+BLOCK_SIZE] = 1
local_layout_64 = pattern_to_layout(local_mask, 64)
local_layout_32 = pattern_to_layout(local_mask, 32)

attention_32 = BlockSparseAttention(layout=local_layout_32, block_size=32, dropout=DROPOUT, num_heads=HEADS)
attention_64 = BlockSparseAttention(layout=local_layout_64, block_size=64, dropout=DROPOUT, num_heads=HEADS)

test = torch.rand((2*HEADS, SEQ-100, 64)).cuda().half()

# att_32 = attention_32(test, test, test)[0]
# att_64 = attention_64(test, test, test)[0]


# # no issue here
# assert (attention_32(test, test, test) != attention_64(test, test, test)).sum() == 0


# test = test.transpose(1,2).reshape(2, SEQ, -1).half()

# multi_head_32 = (
#     MultiHeadDispatch(
#         seq_len=SEQ,
#         dim_model=EMB,
#         residual_dropout=0,
#         num_heads=HEADS,
#         attention=attention_32,
#     )
#     .cuda()
#     .half()
# )

# multi_head_64 = (
#     MultiHeadDispatch(
#         seq_len=SEQ,
#         dim_model=EMB,
#         residual_dropout=0,
#         num_heads=HEADS,
#         attention=attention_64,
#     )
#     .cuda()
#     .half()
# )


# att_val_32 = multi_head_32(query=test, key=test, value=test)
# att_val_64 = multi_head_64(query=test, key=test, value=test)

# # error here
# assert (att_val_32 != att_val_64).sum() == 0

# def build_local_layout(HEADS, block_size, block_unit, seq_len):
#     local_block_units = block_size // block_unit
#     layout = torch.zeros(HEADS, seq_len // block_unit, seq_len // block_unit)
#     for block_start_idx in range(0, seq_len // block_unit, local_block_units):
#         layout[:,block_start_idx:block_start_idx + local_block_units, block_start_idx:block_start_idx + local_block_units] = 1
#     return layout


# layout_64 = build_local_layout(HEADS, BLOCK_SIZE, 64, SEQ)
# layout_32 = build_local_layout(HEADS, BLOCK_SIZE, 32, SEQ)

# assert (layout_32 == local_layout_32).float().mean() == 1
# assert (layout_64 == local_layout_64).float().mean() == 1




attention_32 = BlockSparseLocalAttention(seq_len=SEQ, block_size=BLOCK_SIZE, dropout=DROPOUT, num_heads=HEADS, block_unit=32)
attention_64 = BlockSparseLocalAttention(seq_len=SEQ, block_size=BLOCK_SIZE, dropout=DROPOUT, num_heads=HEADS, block_unit=64)

attention_base = BlockNoglobalAttention(dropout=DROPOUT, num_heads=HEADS, block_size=BLOCK_SIZE)

assert (attention_32(test, test, test) != attention_64(test, test, test)).sum() == 0

diff_index = (attention_32(test, test, test) != attention_base(test, test, test)).nonzero(as_tuple=True)

breakpoint()
assert (attention_32(test, test, test) != attention_base(test, test, test)).sum() == 0
