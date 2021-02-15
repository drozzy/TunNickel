# %% Imports
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from data import NUM_LABELS
# %%
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS):
        super().__init__()
        hidden_size = num_classes
        self.lstm = nn.LSTM(input_size=num_features, 
            hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out
# %%
m = Model(num_classes=1, num_features=1)
# (batch, seq_len, input_size)
batch = 1
seq_len = 1
input_size = 2
xs = torch.tensor([ [ [10.1] ]])
# print(xs.shape)
y = m(xs)
y = y.view(y.shape[0], -1)
# print(y[:, -1, :].shape)
target = torch.tensor([[0]], dtype=torch.int64)
target = target.reshape(-1)
print(y.shape, target.shape)
y
# %%
F.cross_entropy(y, target)
# %%
# %%
