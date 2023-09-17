from torch.utils.data import Dataset
import pandas as pd

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class BaRuDataset(Dataset):

    def __init__(self,
                 path2dset: str,
                 transforms=None) -> None:
        super().__init__()

        self.dset = pd.read_parquet(path2dset)

    def __len__(self):
        return len(self.dset)
    
    def __getitem__(self, idx: int):
        
        row = self.dset.iloc[idx]

        return row["ba"], row["ru"]
    

if __name__ == "__main__":
    
    dset = BaRuDataset("data/train-00000-of-00001-cb5cc9a04cc776c6.parquet")
    print(dset[:2][1])