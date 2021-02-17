# %% 
from importlib import resources, reload
import tunnickel
reload(tunnickel)
from tunnickel.train import train
from tunnickel.data import TrialsDataModule, USERS
with resources.path("tunnickel", f"Suturing") as trials_dir:
    result = train(test_users=[USERS[0]], max_epochs=1, trials_dir=trials_dir)

    acc = 100*result['test_acc_epoch']
    print(f"The final accuracy was: {acc}")
# %%
