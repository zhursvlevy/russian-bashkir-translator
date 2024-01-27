import os
import json
import yaml

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from .data.ba_ru_dataset import BaRuDataset
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast

from .config import ExperimentConfig, DataConfig, TokenizerConfig


def train(cfg: ExperimentConfig):

    # data
    train_dataset = BaRuDataset(cfg.data.train_data)
    val_dataset = BaRuDataset(cfg.data.val_data)

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size)

    # tokenization
    ru_tokenizer = T5TokenizerFast.from_pretrained(cfg.tokenizer.ba_tokenizer)
    ba_tokenizer = T5TokenizerFast.from_pretrained(cfg.tokenizer.ru_tokenizer)

    ba, ru = next(iter(train_loader))

    with open("configs/model/t5-small.yaml") as f:
        config = yaml.safe_load(f)
    
    config = T5Config(**config)
    model = T5ForConditionalGeneration(config)

    src = ru_tokenizer(ru, 
                       padding="longest",
                       max_length=128,
                       truncation=True, 
                       return_tensors="pt").input_ids
    tgt = ba_tokenizer(ba,
                       padding="longest",
                       max_length=128,
                       truncation=True, 
                       return_tensors="pt").input_ids

    tgt[tgt == ba_tokenizer.pad_token_id] = -100

    print(src)
    print(tgt)

    loss = model(input_ids=src, labels=tgt).loss

    print(loss)
if __name__ == "__main__":

    experiment_config = ExperimentConfig(data=DataConfig(
        train_data="data/dataset/splits/train.parquet",
        val_data="data/dataset/splits/val.parquet"
    ), tokenizer=TokenizerConfig("data/ru_t5_tokenizer",
                                 "data/ba_t5_tokenizer")
                                 )

    train(experiment_config)