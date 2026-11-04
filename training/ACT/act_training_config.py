"""ACT training configuration without the shared trainer's EMA module."""

from dataclasses import dataclass
from typing import ClassVar

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


@dataclass(frozen=True)
class _DisabledEMA:
    # The shared trainer reads cfg.ema.enable. Keep only this disabled runtime
    # sentinel; it has no weights, decay settings, CLI options or saved fields.
    enable: bool = False
    use_for_eval: bool = False


@dataclass
class ACTTrainPipelineConfig(TrainPipelineConfig):
    # ClassVar removes the inherited field from dataclass parsing/serialization.
    ema: ClassVar[_DisabledEMA] = _DisabledEMA()


def install_act_training_config(train_module) -> None:
    """Use ACT's schema while retaining LeRobot's CLI handling and train loop."""
    original_train = train_module.train.__wrapped__

    @parser.wrap()
    def train_act(cfg: ACTTrainPipelineConfig):
        return original_train(cfg)

    train_module.train = train_act
