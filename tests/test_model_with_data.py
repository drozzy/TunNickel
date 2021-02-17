from tunnickel.model import Model, Module
from importlib import resources
import torch
from pytorch_lightning import Trainer
import torch.nn.functional as F
from tunnickel.data import TrialsDataModule, USERS, NUM_LABELS
from importlib import resources
import torch

def test_model_and_data_module_before_train():
    trainer = Trainer()
    with resources.path("tunnickel", f"Suturing") as trials_dir:
        mo = Module(num_features=76, num_classes=NUM_LABELS)
        
        batch_size = 3
        dm = TrialsDataModule(trials_dir, test_users=[USERS[0]], train_batch_size=batch_size)
        dm.prepare_data()
        dm.setup(None)
    
        result = trainer.test(mo, datamodule=dm)
        assert type(result[0]) == dict
