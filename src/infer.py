# %% 
from tunnickel.model import ANODE_Linear, NODE_Linear, Linear_Model, LSTM_Model, Module, S_NODE_CNN, S_ANODE_CNN, ANODE_CNN, NODE_CNN, NODE_LSTM, S_NODE_LSTM, S_ANODE_LSTM, ANODE_LSTM
from tunnickel.data import USERS, NUM_LABELS
from pytorch_lightning import Trainer
from importlib import resources
from tunnickel.train import train
import numpy as np
import argparse
import torch
from tunnickel.data import TrialDataset, TrialsDataModule
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as PF
# %%
KINEMATICS_USECOLS = None
MAX_EPOCHS = 10_000
BATCH_SIZE = 32
PATIENCE = 500
NUM_WORKERS = 16
DOWNSAMPLE_FACTOR = 6
NUM_FEATURES = len(KINEMATICS_USECOLS) if KINEMATICS_USECOLS is not None else 76


def get_predictions(model, x):
    pred_logits = model(x)
    return torch.argmax(pred_logits, dim=-1)

def compute_padded_accuracy(preds, target, lens):
    acc = pl.metrics.Accuracy()
    for p, t, l in zip(preds, target, lens):
        p2 = p[:l].reshape(-1)
        t2 = t[:l].reshape(-1)
        acc(p2, t2)    
    return acc.compute()

model = LSTM_Model(num_features=NUM_FEATURES, num_classes=NUM_LABELS, hidden_size=1024)

# %%
# d = TrialDataset(users="B")
# x, y = d[1]
# print(x.shape)
# print(y.shape)
# x = x[22:32]
# y = y[22:32]
# # print(x)


# model.eval()
# # pred_logits = model(x)
# pred = get_predictions(model, x)

# print("Target/Pred:")
# print(y)
# print(pred)
# target = y
# acc = pl.metrics.Accuracy()
# acc(pred, target)
# print(f"Accuracy: {acc.compute()}")

# %% Now trydata loader
dm = TrialsDataModule(test_batch_size=2, usecols=KINEMATICS_USECOLS, downsample_factor=600)
dm.prepare_data()
dm.setup()

it = iter(dm.test_dataloader())

x, target, lens = next(it)

print(x.shape)
print(target.shape)
# raise ValueError()
preds = get_predictions(model, x)
torch.set_printoptions(edgeitems=1000)
print("Target/preds")
print(target)
print(preds)

acc = pl.metrics.Accuracy()
a = acc(target, preds)
print(f"Naive (wrong) Accuracy: {a}")

a2 = compute_padded_accuracy(preds, target, lens)
print(f"Padded (correct) accuracy: {a2}")

# x = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
# y = torch.tensor([0, 1, 2, 2], dtype=torch.int64)
# print(PF.accuracy(x, y))
# tensor(0.7500)
