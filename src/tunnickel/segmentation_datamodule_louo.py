# %% Datamodule
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from segmentation_dataset_louo import SegmentationDataset

from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    # trial_kinematic_data, labels, mask <- batch
    xx = [torch.tensor(b[0], dtype=torch.float32) for b in batch]
    yy = [torch.tensor(b[1], dtype=torch.int64) for b in batch]
    xx_mask = [torch.tensor(b[2], dtype=torch.int64) for b in batch]

    # yy = torch.tensor([b['label'] for b in batch])
    # xx = [torch.tensor(d['segment'], dtype=torch.float32) for d in batch]
    xx_pad = pad_sequence(xx, batch_first=True)
    yy_pad = pad_sequence(yy, batch_first=True)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_mask_pad = pad_sequence(xx_mask, batch_first=True)

    return xx_pad, yy_pad, x_lens, y_lens, xx_mask_pad

class SegmentationDataModuleLouo(pl.LightningDataModule):
    def __init__(self, task="Knot_Tying", user_out="1_Out", train_batch_size=1, test_batch_size=8):
        super().__init__()
        self.task = task
        self.user_out = user_out        
        self.train_batch_size = train_batch_size
        self.test_batch_size=test_batch_size
    
    def setup(self, stage=None):
        self.train_dataset = SegmentationDataset(task=self.task, user_out=self.user_out, train=True)

        self.val_dataset = SegmentationDataset(task=self.task, user_out=self.user_out, train=False)

        # Use the same dataset for testing - just to plot the best validation score
        self.test_dataset = SegmentationDataset(task=self.task, user_out=self.user_out, train=False)
        
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.train_batch_size, shuffle=True, collate_fn=pad_collate)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.test_batch_size, collate_fn=pad_collate)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.test_batch_size, collate_fn=pad_collate)
# %%
dm = SegmentationDataModuleLouo()
dm.prepare_data()
dm.setup()
print("Train, Test:")
lent, lenv = len(dm.train_dataset), len(dm.val_dataset)
print(f"Train length: {lent}, Val length: {lenv}")
# # %%
next(iter(dm.test_dataloader()))
# %%
xx_pad, yy_pad, x_lens, y_lens, xx_mask_pad = next(iter(dm.train_dataloader()))
xx_pad.size()
# %%
