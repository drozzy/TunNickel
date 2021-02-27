# %%
from tunnickel.data import TrialsDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from tunnickel.model import LSTM2_NODE, CNN, S_ANODE_Linear, ANODE_Linear, NODE_Linear, Linear_Model, LSTM_Model, NODE_CNN, S_ANODE_LSTM, ANODE_LSTM
from torchdyn.models import *
from torchdyn import *
from pytorch_lightning.loggers import WandbLogger
import wandb 
from tunnickel.ode_rnn_rubanova import ODE_RNN_Rubanova
from tunnickel.module import Module 

def create_model(model_name, num_features, num_classes, min_params, max_params, dropout):
    if model_name == 'ANODE-LSTM':
        model = ANODE_LSTM(num_features=num_features, num_classes=num_classes, hidden_size=1024)
    elif model_name == 'S-ANODE-LSTM':
        model = S_ANODE_LSTM(num_features=num_features, num_classes=num_classes, hidden_size=1024)
    elif model_name == 'LSTM':
        model = LSTM_Model(num_features=num_features, num_classes=num_classes, hidden_size=512, dropout=dropout)
    elif model_name == 'Linear':
        model = Linear_Model(num_features=num_features, num_classes=num_classes, hidden_size=1596)
    elif model_name == 'ODE-RNN-Rubanova':
        model = ODE_RNN_Rubanova(num_features=num_features, num_classes=num_classes, hidden_size=136)
    elif model_name == 'NODE-Linear':
        model = NODE_Linear(num_features=num_features, num_classes=num_classes, hidden_size=1556)
    elif model_name == 'ANODE-Linear':
        model = ANODE_Linear(num_features=num_features, num_classes=num_classes, hidden_size=1556)
    elif model_name == 'S-ANODE-Linear':
        model = S_ANODE_Linear(num_features=num_features, num_classes=num_classes, hidden_size=1416)
    elif model_name == 'NODE-CNN':
        model = NODE_CNN(num_features=num_features, num_classes=num_classes, hidden_size=1446)
    elif model_name == 'CNN':
        model = CNN(num_features=num_features, num_classes=num_classes, hidden_size=1445)
    elif model_name == 'LSTM2_NODE':
        model = LSTM2_NODE(num_features=num_features, num_classes=num_classes, hidden_size=93)
    else:
        raise ValueError("No such model exists: %s" % model_name)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model parameters: {params}")
    assert params > min_params, f"Model has {params} but need at least {min_params}. Increase num of parameters in train.py to model."
    assert params < max_params, f"Model has {params} but max is {max_params}. Decrease num of parameters in train.py to model."

    return model


def create_trainer(project_name, experiment_name, model_name, patience, max_epochs, gpus, enable_logging):
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
    
    if enable_logging:
        logger = WandbLogger(project=project_name, name=experiment_name, group=model_name)  
    else:
        logger = None
    
    trainer = pl.Trainer(logger=logger, gpus=gpus, max_epochs=max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback])
    return trainer
    
def train(project_name, experiment_name, model_name, test_users, max_epochs, trials_dir, batch_size, patience, gpus, num_workers, downsample_factor, usecols, 
        num_features, num_classes, enable_logging, min_params, max_params, dropout):    

    trainer = create_trainer(project_name, experiment_name, model_name, patience, max_epochs, gpus, enable_logging)
    model = create_model(model_name, num_features, num_classes, min_params, max_params, dropout)
    
    mo = Module(model=model)

    dm = TrialsDataModule(trials_dir, test_users=test_users, train_batch_size=batch_size,
        num_workers=num_workers, downsample_factor=downsample_factor, usecols=usecols)
    
    result = trainer.fit(mo, datamodule=dm)

    result = trainer.test()

    
    if enable_logging:
        wandb.config.max_epochs = max_epochs
        wandb.config.batch_size = batch_size
        wandb.config.patience = patience
        wandb.config.downsample_factor = downsample_factor
        wandb.config.num_features = num_features
        wandb.config.num_classes = num_classes
        wandb.config.dropout = dropout
        
        wandb.save('tunnickel/model.py')
        wandb.save('tunnickel/data.py')
        wandb.save('tunnickel/train.py')
        wandb.finish()  # Otherwise we keep writing to the same run!
    return result
    # return result['test_acc']
