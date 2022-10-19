# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# register components configs into Hydra ConfigStore
# component config classes could be used to validate configs
import logging

from hydra.core.config_store import ConfigStore
from omegaconf.errors import ValidationError

from xformers.components.attention import ATTENTION_REGISTRY
from xformers.components.feedforward import FEEDFORWARD_REGISTRY
from xformers.components.positional_embedding import POSITION_EMBEDDING_REGISTRY

logger = logging.getLogger("xformers")


def import_xformer_config_schema():
    """
    Best effort - OmegaConf supports limited typing, so we may fail to import
    certain config classes. For example, pytorch typing are not supported.
    """
    cs = ConfigStore.instance()

    for k, v in {
        "ff": FEEDFORWARD_REGISTRY,
        "pe": POSITION_EMBEDDING_REGISTRY,
        "attention": ATTENTION_REGISTRY,
    }.items():
        for kk in v.keys():
            try:
                cs.store(name=f"{kk}_schema", node=v[kk].config, group=f"xformers/{k}")
            except ValidationError as e:
                logger.debug(f"Error registering {kk}_schema, error: {e}")
