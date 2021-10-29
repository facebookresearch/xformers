from dataclasses import dataclass

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from xformers.factory.model_factory import xFormer, xFormerConfig, xFormerStackConfig

BATCH = 8
SEQ = 1024
EMB = 384
VOCAB = 64


@dataclass
class Config:
    encoder: xFormerStackConfig
    decoder: xFormerStackConfig
    dim_model: int
    vocab_size: int
    seq_len: int


cs = ConfigStore.instance()

# base_config will validate the schema/typing in primary config - conf/config.yaml
cs.store(name="base_config", node=Config)


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: Config) -> None:
    encoder_cfg = OmegaConf.to_container(cfg.encoder, resolve=True)
    decoder_cfg = OmegaConf.to_container(cfg.decoder, resolve=True)
    config = xFormerConfig([encoder_cfg, decoder_cfg])  # type: ignore
    model = xFormer.from_config(config)
    print(f"build model with encoder attention {type(model.encoders[0].mha.attention)}")
    #  Test out with dummy inputs
    x = (torch.rand((BATCH, SEQ)) * VOCAB).abs().to(torch.int)
    y = model(src=x, tgt=x)
    print(type(y))


if __name__ == "__main__":
    my_app()
