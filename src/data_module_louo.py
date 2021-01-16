# %% Dataset and Datamodule
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_louo import pad_collate, LouoDataset            
class LouoDataModule(pl.LightningDataModule):
    def __init__(self, task="Knot_Tying", user_out="1_Out"):
        super().__init__()
        self.task = task
        self.user_out = user_out
    
    def setup(self, stage=None):
        self.train_dataset = LouoDataset(task=self.task, 
                user_out=self.user_out, train=True)
        self.val_dataset = LouoDataset(task=self.task, 
                user_out=self.user_out, train=False)
        # Use the same dataset for testing - just to plot the best validation score
        self.test_dataset = LouoDataset(task=self.task, 
                user_out=self.user_out, train=False)

        self.num_gestures = self.train_dataset.num_gestures
        
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=128, collate_fn=pad_collate)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=128, collate_fn=pad_collate)
# %%
# dm = LouoDataModule()
# dm.prepare_data()
# dm.setup()
# print("Train, Val:")
# len(dm.train_dataset), len(dm.val_dataset)
# next(iter(dm.test_dataloader()))
# %%

# %%

# %%

# %%
