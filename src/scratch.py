# %% 
from importlib import resources

from tunnickel.train import train
from tunnickel.data import TrialsDataModule, USERS
with resources.path("tunnickel", f"Suturing") as trials_dir:
    train(test_users=[USERS[0]], max_epochs=3, trials_dir=trials_dir)
# %%
