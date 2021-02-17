# %%
from tunnickel.model import Module
from tunnickel.data import TrialsDataModule, USERS, NUM_LABELS
import torch
from pytorch_lightning import Trainer
import torch.nn.functional as F
from importlib import resources
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


def create_trainer(patience, max_epochs, gpus):
    early_stop_callback = EarlyStopping(
        monitor='val_acc_epoch',
        patience=patience,
        mode='max'
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_acc_epoch',
        mode='max'
    )
    trainer = pl.Trainer(gpus=gpus, max_epochs=max_epochs, 
        callbacks=[early_stop_callback, checkpoint_callback])
    return trainer
    
def train(test_users, model, max_epochs, trials_dir, batch_size = 3, patience=100, gpus=0, num_workers=0):    
    trainer = create_trainer(patience, max_epochs, gpus)
    mo = Module(model=model)
    
    dm = TrialsDataModule(trials_dir, test_users=test_users, train_batch_size=batch_size,
        num_workers=num_workers)

    result = trainer.fit(mo, datamodule=dm)

    result = trainer.test()
    return result[0]
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