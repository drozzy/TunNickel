# %% 
from importlib import resources

from tunnickel.train import train_with_users
from tunnickel.data import TrialsDataModule, USERS
with resources.path("tunnickel", f"Suturing") as trials_dir:
    train_with_users(test_users=[USERS[0]], max_epochs=1, trials_dir=trials_dir)
# %%
