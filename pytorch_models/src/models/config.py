from dataclasses import dataclass


@dataclass
class T5BaseConfig:
    d_ff: int =  2048
    d_kv: int = 64
    d_model: int = 512
    decoder_start_token_id: int = 0
    dropout_rate: float = 0.1
    eos_token_id: int = 1
    initializer_factor: float = 1.0
    is_encoder_decoder: bool = True
    layer_norm_epsilon: float = 0.000001
    model_type: str = "t5"
    n_positions: int = 512
    num_heads: int = 8
    num_layers: int = 6
    output_past: bool = True
    pad_token_id: int = 0
    relative_attention_num_buckets: int = 32
    vocab_size: int = 32128
    