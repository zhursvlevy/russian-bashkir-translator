from typing import Optional

from transformers import T5ForConditionalGeneration, T5Config
from pytorch_models.src.models.components.config import T5BaseConfig, PAD_TOKEN_ID
import torch
from torch import nn

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class T5BaseModel(torch.nn.Module):

    def __init__(self, weights: Optional[str] = None, config: Optional[T5BaseConfig] = None) -> None:
        super().__init__()
        if weights:
            self.model = T5ForConditionalGeneration.from_pretrained(weights)
        else:
            self.t5_config = T5Config(
                d_model=config.d_model,
                d_kv=config.d_kv,
                d_ff=config.d_ff,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                num_decoder_layers=config.num_decoder_layers,
                dropout_rate=config.dropout_rate,
                model_type=config.model_type,
                relative_attention_num_buckets=config.relative_attention_num_buckets,
                n_positions=config.n_positions,
                output_past=config.output_past,
                vocab_size=config.vocab_size,
                layer_norm_epsilon=config.layer_norm_epsilon,
                decoder_start_token_id=PAD_TOKEN_ID
            )

            self.model = T5ForConditionalGeneration(self.t5_config)
            self.model.shared = nn.Embedding(config.vocab_size, config.d_model)
            self.model.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model.generate(input_ids, attention_mask=attention_mask, do_sample=False)