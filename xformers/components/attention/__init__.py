from pathlib import Path

from xformers.components.attention.base import Attention  # noqa
from xformers.components.attention.base import AttentionConfig  # noqa
from xformers.utils import import_all_modules

# automatically import any Python files in the directory
# FIXME: @lefaudeux
import_all_modules(str(Path(__file__).parent), "xformers.components.attention")

from xformers.components.attention.multi_head_attention import (  # noqa
    MultiHeadAttention,
)
