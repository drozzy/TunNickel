# %% Dataset and Datamodule
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import pad_collate, JigsawsDataset            
class JigsawsDataModule(pl.LightningDataModule):
    def __init__(self, task="Knot_Tying", user_out="1_Out"):
        super().__init__()
        self.task = task
        self.user_out = user_out
    
    def setup(self, stage=None):
        ds = JigsawsDataset(task=self.task, user_out=self.user_out, train=True)
        self.num_gestures = ds.num_gestures
        n = len(ds)
        nval = int(0.1*n)
        ntrain = n - nval
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(ds, [ntrain, nval], generator=torch.Generator().manual_seed(42))

        self.test_dataset = JigsawsDataset(task=self.task, user_out=self.user_out, train=False)
        
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=128, collate_fn=pad_collate)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=128, collate_fn=pad_collate)
# %%
# dm = JigsawsDataModule()
# dm.prepare_data()
# dm.setup()
# print("Train, Val:")
# len(dm.train_dataset), len(dm.val_dataset)
# next(iter(dm.test_dataloader()))
# %%

# %%

# %%

# %%
