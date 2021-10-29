from typing import Any, Dict, List
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from xformers.factory.block_factory import xFormerEncoderBlock, xFormerEncoderConfig

BATCH = 8


@hydra.main(config_path=".", config_name="encoder_config")
def my_app(cfg: DictConfig) -> None:
    # resolve to standard python dict
    config = OmegaConf.to_container(cfg.encoder, resolve=True)
    encoder_config: xFormerEncoderConfig = xFormerEncoderConfig(**config)  # type: ignore
    encoder = xFormerEncoderBlock(encoder_config)

    #  Test out with dummy inputs
    x = (torch.rand((BATCH, cfg.seq_len)) * cfg.vocab_size).abs().to(torch.int)
    y = encoder(x, x, x)
    print(y)


if __name__ == "__main__":
    my_app()
