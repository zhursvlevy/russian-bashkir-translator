import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
import torch
from pytorch_models.src.models.components.config import T5BaseConfig

from pytorch_models.src.train import train

IS_CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.parametrize(
        ("device",),
        [
            pytest.param("cpu", id="test run on cpu"),
            pytest.param("gpu", id="test run on gpu", marks=pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA is not available"))
         ]
)
def test_train_fast_dev_run(cfg_train: DictConfig, device: str) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = device
    train(cfg_train)



def test_model_from_config(cfg_train: DictConfig) -> None:
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
    cfg_train.model.net.weights = None
    cfg_train.model.net.config = T5BaseConfig()
    train(cfg_train)
