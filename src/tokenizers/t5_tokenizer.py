from transformers import  T5TokenizerFast

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from data.ba_ru_dataset import BaRuDataset

dset = BaRuDataset("data/train-00000-of-00001-cb5cc9a04cc776c6.parquet")

fast_tokenizer = T5TokenizerFast.from_pretrained("data/ru_t5_tokenizer")

sent = dset[25][1]

print(sent)
print(fast_tokenizer.tokenize(sent))