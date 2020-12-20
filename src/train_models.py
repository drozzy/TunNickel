# %% Imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_module import KinematicsDataModule
from pytorch_lightning.loggers import WandbLogger
import datetime
from m02_lstm import LSTMModel
from m03_cnn import CnnModel
from m03_neuralode_cnn import NeuralODECnnModel
from m05_anode import ANodeModel
from pytorch_lightning.callbacks import ModelCheckpoint

# %% Models 
models = [
    CnnModel(),
    LSTMModel(),
    NeuralODECnnModel()
]

# %% Test out  a model
m = CnnModel()
data = KinematicsDataModule()
data.prepare_data()
data.setup()
x,y = next(iter(data.train_dataloader()))
print("LABELS ARE:")
print(y)
print(x.shape, y.shape)
outs = m(x)
print(outs.shape)
print(y)
print(outs[0])
# %% Train
data = KinematicsDataModule()

EPOCHS = 1000

def run_name(model_obj):
    name = type(model_obj).__name__.split("Model")[0]
    timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')[1]
    return f"{name} {timestamp}" 

for model in models:
    early_stop_callback = EarlyStopping(
        monitor='val_acc_epoch',
        patience=25,
        verbose=False,
        mode='max'
    )
    n = run_name(model)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor='val_acc_epoch',
        mode='max'
    )
    
    trainer = pl.Trainer(gpus=1, max_epochs=EPOCHS, 
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=WandbLogger(run_name(model)))
    trainer.fit(model, data)
    results = trainer.test(datamodule=data);

# %%

trainer.test(datamodule=data);

# %%
