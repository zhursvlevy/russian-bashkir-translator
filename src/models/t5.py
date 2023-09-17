import os
import sys
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import yaml

from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast
from src.data.ba_ru_dataset import BaRuDataset


import pandas as pd


if __name__ == "__main__":

    ru_tokenizer = T5TokenizerFast.from_pretrained("data/ru_t5_tokenizer")
    ba_tokenizer = T5TokenizerFast.from_pretrained("data/ba_t5_tokenizer")

    dset = BaRuDataset("data/dataset/train-00000-of-00001-cb5cc9a04cc776c6.parquet")

    ba, ru = dset[0]

    print(type(ba))

    src = ru_tokenizer(ru, return_tensors="pt").input_ids
    tgt = ba_tokenizer(ba, return_tensors="pt").input_ids

    print(src, tgt)

    with open("configs/model/t5-small.yaml") as f:
        config = yaml.safe_load(f)
    
    config = T5Config(**config)
    model = T5ForConditionalGeneration(config)

    loss = model(input_ids=src, labels=tgt).loss
    # print(model)
    print(loss)