import hydra
from pathlib import Path
from omegaconf import DictConfig

from trainer import Trainer


@hydra.main(config_path=f"{str(Path(__file__).parent.absolute())}/config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
