from dataclasses import dataclass


PAD_TOKEN_ID = 0

@dataclass
class T5BaseConfig:
    d_ff: int =  2048
    d_kv: int = 64
    d_model: int = 512
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 0.000001
    model_type: str = "t5"
    n_positions: int = 512
    num_heads: int = 8
    num_layers: int = 6
    output_past: bool = True
    num_decoder_layers: int = num_layers
    relative_attention_num_buckets: int = 32
    vocab_size: int = 32128
    