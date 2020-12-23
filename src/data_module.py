# %% Dataset and Datamodule
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import pad_collate, JigsawsDataset            
class JigsawsDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
    
    def setup(self, stage=None):
        self.train_dataset = JigsawsDataset(train=True)
        self.val_dataset = JigsawsDataset(train=False)
        # self.test_dataset = read_h5_dataset(test_path)
        
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=128, collate_fn=pad_collate)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=128)
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
