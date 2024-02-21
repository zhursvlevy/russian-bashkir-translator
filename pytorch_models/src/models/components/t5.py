from typing import Optional

from transformers import T5ForConditionalGeneration, T5Config
import torch
from torch import nn

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class T5BaseModel(torch.nn.Module):

    def __init__(self, weights: Optional[str] = None, config: Optional[T5Config] = None) -> None:
        super().__init__()
        if weights:
            self.model = T5ForConditionalGeneration.from_pretrained(weights)
        else:
            self.model = T5ForConditionalGeneration(config)
            self.model.shared = nn.Embedding(config.vocab_size, config.d_model)
            self.model.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model.generate(input_ids, attention_mask=attention_mask, do_sample=False)