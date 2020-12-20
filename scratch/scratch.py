# %%
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# %%
x = torch.randn(500, 100)
y = torch.randn(500, 1)

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.x = torch.randn(1000, 64, 64)
        self.y = torch.randn(1000, 1)

    def __getitem__(self, index):
        return {"feature": self.x[index], "label": self.y[index]}

    def __len__(self):
        return self.x.shape[0]

# %%
ds = MyDataset()
dl = DataLoader(ds, batch_size=50)
# %%
batch = next(iter(dl))
# %%
batch['feature'].shape, batch['label'].shape
# %%
