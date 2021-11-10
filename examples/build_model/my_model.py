import hydra
from omegaconf import DictConfig

from xformers.factory.hydra_helper import import_xformer_config_schema


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.xformer, _convert_="all")
    print(model)


if __name__ == "__main__":
    # optional - only needed when you want to use xformer config dataclass
    # to validate config values.
    import_xformer_config_schema()
    my_app()
