# %% Imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_module import JigsawsDataModule
from pytorch_lightning.loggers import WandbLogger
import datetime
from m01_lstm import LSTMModel
from m02_cnn import CnnModel
from m03_neuralode_cnn import NeuralODECnnModel
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

# %% Train
data = JigsawsDataModule(user_out="1_Out") # for cross validation
data.prepare_data()
data.setup()

models = [
    CnnModel(num_classes=data.num_gestures),
    LSTMModel(num_classes=data.num_gestures),
    NeuralODECnnModel(num_classes=data.num_gestures)
]


EPOCHS = 1000

def run_name(model_obj):
    name = type(model_obj).__name__.split("Model")[0]
    timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')[1]
    return f"{name} {timestamp}" 

for model in models:
    early_stop_callback = EarlyStopping(
        monitor='val_acc_epoch',
        patience=500,
        mode='max'
    )
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_acc_epoch',
        mode='max'
    )
    name = run_name(model)
    trainer = pl.Trainer(gpus=1, max_epochs=EPOCHS, 
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=WandbLogger(name))

    trainer.fit(model, data)

    results = trainer.test(datamodule=data);

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
    wandb.run.summary["num_parameters"] = num_parameters
    wandb.finish()

# %%
