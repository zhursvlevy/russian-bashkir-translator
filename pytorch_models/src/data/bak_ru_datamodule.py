from typing import Any, Dict, Optional

from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import BasicTokenizer

from src.data.components.bak_ru_dataset import BakRuDataset, Language


class BakRuDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        tokenizer: BasicTokenizer,
        max_len: int = 512,
        batch_size: int = 64,
        num_workers: int = 0,
        train_size: int = 50000,
        val_size: int = 1000,
        test_size: int = 1000,  
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:

        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_path = list(Path(self.hparams.data_dir).glob("train*"))
            val_path = list(Path(self.hparams.data_dir).glob("val*"))
            test_path = list(Path(self.hparams.data_dir).glob("test*"))
            if len(train_path) == 0:
                raise("train dataset must be provided")
            if len(val_path) == 0:
                raise("val dataset must be provided")
            if len(train_path) > 1:
                raise("Only one train file must be provided. More than one train dataset file found")
            if len(val_path) > 1:
                raise("Only one val file must be provided. More than one val dataset file found") 
            ru_bak_trainset = BakRuDataset(
                train_path[0], 
                Language.BAK, 
                tokenizer=self.hparams.tokenizer, 
                max_len=self.hparams.max_len,
                num_samples=self.hparams.train_size
                )
            bak_ru_trainset = BakRuDataset(
                train_path[0],
                Language.RU, 
                tokenizer=self.hparams.tokenizer, 
                max_len=self.hparams.max_len,
                num_samples=self.hparams.train_size
                )
            ru_bak_valset = BakRuDataset(
                val_path[0], 
                Language.BAK, 
                tokenizer=self.hparams.tokenizer, 
                max_len=self.hparams.max_len,
                num_samples=self.hparams.val_size
                )
            bak_ru_valset = BakRuDataset(
                val_path[0], 
                Language.RU, 
                tokenizer=self.hparams.tokenizer, 
                max_len=self.hparams.max_len,
                num_samples=self.hparams.val_size
                )
            self.data_train = ConcatDataset(datasets=[bak_ru_trainset, ru_bak_trainset])
            self.data_val = ConcatDataset(datasets=[bak_ru_valset, ru_bak_valset])
            if len(test_path) == 1:
                ru_bak_testset = BakRuDataset(
                    test_path[0], 
                    Language.BAK, 
                    tokenizer=self.hparams.tokenizer, 
                    max_len=self.hparams.max_len,
                    num_samples=self.hparams.test_size
                )
                bak_ru_testset = BakRuDataset(
                    test_path[0],
                    Language.RU,
                    tokenizer=self.hparams.tokenizer, 
                    max_len=self.hparams.max_len,
                    num_samples=self.hparams.test_size
                )
                self.data_test = ConcatDataset(datasets=[bak_ru_testset, ru_bak_testset])

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
