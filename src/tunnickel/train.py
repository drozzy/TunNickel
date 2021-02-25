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
from pytorch_lightning import loggers as pl_loggers
from tunnickel.model import ANODE_Linear, NODE, Linear_Model, LSTM_Model, Module, S_NODE_CNN, S_ANODE_CNN, ANODE_CNN, NODE_CNN, NODE_LSTM, S_NODE_LSTM, S_ANODE_LSTM, ANODE_LSTM

def create_model(model_name, num_features, num_classes, params_min=140_000, params_max=160_000):
    if model_name == 'ANODE-LSTM':
        model = ANODE_LSTM(num_features=num_features, num_classes=num_classes, hidden_size=1024)
    elif model_name == 'S-ANODE-LSTM':
        model = S_ANODE_LSTM(num_features=num_features, num_classes=num_classes, hidden_size=1024)
    elif model_name == 'LSTM':
        model = LSTM_Model(num_features=num_features, num_classes=num_classes, hidden_size=1256)
    elif model_name == 'Linear':
        model = Linear_Model(num_features=num_features, num_classes=num_classes, hidden_size=1596)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params}")
    assert params > params_min, "Increase num of parameters in train.py to model."
    assert params < params_max, "Decrease num of parameters in train.py to model."

    return model


def create_trainer(model_name, patience, max_epochs, gpus, deterministic):
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
    experiment_name = model_name
    tb_logger = pl_loggers.TensorBoardLogger('logs/', name=experiment_name)

    trainer = pl.Trainer(logger=tb_logger, deterministic=deterministic, gpus=gpus, max_epochs=max_epochs, 
        callbacks=[early_stop_callback, checkpoint_callback])
    return trainer
    
def train(model_name, test_users, max_epochs, trials_dir, batch_size, patience, gpus, num_workers, downsample_factor, usecols, 
        deterministic, num_features, num_classes):    
    trainer = create_trainer(model_name, patience, max_epochs, gpus, deterministic)
    model = create_model(model_name, num_features, num_classes)
    
    mo = Module(model=model)
    
    dm = TrialsDataModule(trials_dir, test_users=test_users, train_batch_size=batch_size,
        num_workers=num_workers, downsample_factor=downsample_factor, usecols=usecols)

    result = trainer.fit(mo, datamodule=dm)

    result = trainer.test()
    return result
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
