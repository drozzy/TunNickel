# %% Dataset and Datamodule
import torch
import pytorch_lightning as pl
import h5py
from torch.utils.data import DataLoader
WINDOW_SIZE = 200
class KinematicDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (torch.tensor(self.x[idx], dtype=torch.float32), 
                torch.tensor(self.y[idx], dtype=torch.long) 
        )

def read_h5_dataset(filepath):
        f = h5py.File(filepath, "r")
        return KinematicDataset(f['x'], f['y'])
            
class KinematicsDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
    
    def setup(self, stage=None):
        print("Setting up 2% train data.")

        train_path= f"../dataset/Suturing/kinematics/trainpy_{WINDOW_SIZE}.jld"
        val_path= f"../dataset/Suturing/kinematics/valpy_{WINDOW_SIZE}.jld"
        test_path= f"../dataset/Suturing/kinematics/testpy_{WINDOW_SIZE}.jld"

        self.train_dataset = read_h5_dataset(train_path)
        self.val_dataset = read_h5_dataset(val_path)
        self.test_dataset = read_h5_dataset(test_path)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=128)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=128)
# %%
dm = KinematicsDataModule()
dm.prepare_data()
dm.setup()
print("Train, Val, Test:")
len(dm.train_dataset), len(dm.val_dataset), len(dm.test_dataset)
# next(iter(dm.test_dataloader()))
# %%

# %%

# %%

# %%
