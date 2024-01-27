from dataclasses import dataclass


@dataclass
class DataConfig:
    train_data: str
    val_data: str
    batch_size: int = 4
    shuffle: bool = True

@dataclass
class TokenizerConfig:
    ru_tokenizer: str
    ba_tokenizer: str

@dataclass
class ExperimentConfig:
    data: DataConfig
    tokenizer: TokenizerConfig
