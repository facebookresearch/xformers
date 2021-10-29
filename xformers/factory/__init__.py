# register factory configs
from hydra.core.config_store import ConfigStore

from xformers.components import MultiHeadDispatchConfig  # noqa
from xformers.components.attention import AttentionConfig  # noqa
from xformers.components.feedforward import FeedforwardConfig  # noqa
from xformers.components.positional_embedding import PositionEmbeddingConfig  # noqa

from .block_factory import xFormerDecoderConfig  # noqa; noqa
from .block_factory import xFormerEncoderBlock, xFormerDecoderBlock  # noqa
from .block_factory import xFormerEncoderConfig  # noqa
from .block_factory import xFormerBlockConfig
from .model_factory import xFormer, xFormerConfig, xFormerStackConfig  # noqa

cs = ConfigStore.instance()
cs.store(name="decoder_schema", group="xformers", node=xFormerDecoderConfig)
cs.store(name="encoder_schema", group="xformers", node=xFormerEncoderConfig)
cs.store(name="block_schema", group="xformers", node=xFormerBlockConfig)
cs.store(name="xformers_schema", group="xformers", node=xFormerConfig)
cs.store(name="stack_schema", group="xformers", node=xFormerStackConfig)
