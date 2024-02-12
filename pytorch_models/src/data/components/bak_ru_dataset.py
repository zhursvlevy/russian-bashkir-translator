from typing import Tuple

import torch
from transformers import BasicTokenizer
from torch.utils.data import Dataset
import pandas as pd
from enum import Enum


class Language(str, Enum):
    RU = "русский"
    BAK = "башкирский"


class BakRuDataset(Dataset):
    IGNORE_INDEX = -100

    def __init__(self, path2dset: str, target_lang: Language, tokenizer: BasicTokenizer, max_len: int) -> None:
        self.dset = pd.read_parquet(path2dset)
        self.target_lang = target_lang
        self.prefix = self._set_prefix(Language.RU, Language.BAK) \
            if target_lang == Language.BAK else self._set_prefix(Language.BAK, Language.RU)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        row = self.dset.iloc[idx]
        ru = row["ru"]
        bak = row["ba"]
        source, target = (ru, bak) if self.target_lang == Language.BAK else (bak, ru)
        
        source = self.tokenizer(
            self.prefix + source,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
            )
        target = self.tokenizer(
            target,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
            )
        labels = target.input_ids
        labels[labels == self.tokenizer.pad_token_id] = self.IGNORE_INDEX
        return source.input_ids.squeeze(0), source.attention_mask.squeeze(0), labels.squeeze(0)

    @staticmethod
    def _set_prefix(from_lang: Language, to_lang: Language) -> str:
        return  "-".join([from_lang.value, to_lang.value]) + ": "

