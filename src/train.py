import os
import json

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from .data.ba_ru_dataset import BaRuDataset
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast

from .config import ExperimentConfig, DataConfig


def train(cfg: ExperimentConfig):

    train_dataset = BaRuDataset(cfg.data.train_data)
    val_dataset = BaRuDataset(cfg.data.val_data)

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size)

    print(next(iter(train_loader)))


if __name__ == "__main__":

    experiment_config = ExperimentConfig(DataConfig(
        train_data="data/dataset/splits/train.parquet",
        val_data="data/dataset/splits/val.parquet"
    ))

    train(experiment_config)