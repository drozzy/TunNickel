# %%
import importlib, data
importlib.reload(data)
from data import *

dm = TrialsDataModule("Suturing", train_batch_size=3, test_users=USERS[0:3])
dm.setup()
xs, ys, lengths = next(iter(dm.train_dataloader()))
lengths
# %%
