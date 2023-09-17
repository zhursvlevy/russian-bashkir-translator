from transformers import T5Tokenizer, T5TokenizerFast
from torch.utils.data import Dataset

import hydra
from omegaconf import DictConfig

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.ba_ru_dataset import BaRuDataset


def batch_iterator(dataset: Dataset, batch_size=64, lang="ba"):
    for i in range(0, len(dataset), batch_size):
        if lang == "ba":
            yield dataset[i : i + batch_size][0]
        elif lang == "ru":
            yield dataset[i : i + batch_size][1]
        else:
            raise("lang must be 'ba' or 'ru'")


@hydra.main(version_base="1.3", config_path="../configs", config_name="tokenizer-train")
def train(cfg: DictConfig) -> None:

    fast_tokenizer = T5TokenizerFast.from_pretrained(cfg.tokenizer.pretrained)
    dset = BaRuDataset(cfg.data.path)

    new_fast_tokenizer = fast_tokenizer.train_new_from_iterator(
        text_iterator=batch_iterator(dataset=dset, lang=cfg.tokenizer.lang),
        vocab_size=cfg.tokenizer.vocab_size,
        show_progress=True,
    )

    new_fast_tokenizer.save_pretrained(cfg.tokenizer.save_dir)

if __name__ == "__main__":

    train()