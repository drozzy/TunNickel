# %%
from tunnickel.model import Model, Module
from tunnickel.data import TrialsDataModule, USERS, NUM_LABELS
import torch
from pytorch_lightning import Trainer
import torch.nn.functional as F
from importlib import resources
import torch

def train(test_users, max_epochs, trials_dir, batch_size = 3):    
    trainer = Trainer(max_epochs=max_epochs)
    mo = Module(num_features=76, num_classes=NUM_LABELS)
    
    dm = TrialsDataModule(trials_dir, test_users=test_users, train_batch_size=batch_size)

    result = trainer.fit(mo, datamodule=dm)

    result = trainer.test()
    # return result['test_acc']
# %%
# from tunnickel.model import Model, Module
# from importlib import resources
# import torch
# from pytorch_lightning import Trainer
# import torch.nn.functional as F
# from tunnickel.data import TrialsDataModule, USERS, NUM_LABELS
# from importlib import resources
# import torch
# # %%

# trainer = Trainer(max_epochs=2)
# with resources.path("tunnickel", f"Suturing") as trials_dir:
#     mo = Module(num_features=76, num_classes=NUM_LABELS)
    
#     batch_size = 3
#     dm = TrialsDataModule(trials_dir, test_users=[USERS[0]], train_batch_size=batch_size)
   
#     result = trainer.fit(mo, datamodule=dm)

# trainer.test()
# # %%