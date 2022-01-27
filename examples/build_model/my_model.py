# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from omegaconf import DictConfig

from xformers.factory.hydra_helper import import_xformer_config_schema


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.xformer, _convert_="all")
    print(
        f"Built a model with {len(cfg.xformer.stack_configs)} stack: {cfg.xformer.stack_configs.keys()}"
    )
    print(model)


if __name__ == "__main__":
    # optional - only needed when you want to use xformer config dataclass
    # to validate config values.
    import_xformer_config_schema()
    my_app()
