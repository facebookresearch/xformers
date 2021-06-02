from pathlib import Path

from xformers.components.attention import AttentionConfig  # noqa
from xformers.utils import import_all_modules

from .activations import Activation
from .attention import ATTENTION_REGISTRY, AttentionConfig  # noqa
from .multi_head_dispatch import MultiHeadDispatch, MultiHeadDispatchConfig  # noqa

__all__ = ["MultiHeadDispatch", "Activation"]

# automatically import any Python files in the directory
import_all_modules(str(Path(__file__).parent), "xformers.components")


def build_multi_head_attention(
    config: AttentionConfig, multi_head_config: MultiHeadDispatchConfig
):
    """Builds an attention from a config.

    This assumes a 'name' key in the config which is used to determine what
    attention class to instantiate. For instance, a config `{"name": "my_attention",
    "foo": "bar"}` will find a class that was registered as "my_attention"
    (see :func:`register_attention`) and call .from_config on it."""

    multi_head_config.attention = ATTENTION_REGISTRY[
        config.name
    ].constructor.from_config(config)
    return MultiHeadDispatch.from_config(multi_head_config)
