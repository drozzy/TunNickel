# %% Imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_module import JigsawsDataModule
# from pytorch_lightning.loggers import WandbLogger
import datetime
from m01_lstm import LSTMModel
from pytorch_lightning.callbacks import ModelCheckpoint

# %% Model
data = JigsawsDataModule(user_out="1_Out") # for cross validation
data.prepare_data()
data.setup()
model = LSTMModel(data.num_gestures)

EPOCHS = 10

def run_name(model_obj):
    name = type(model_obj).__name__.split("Model")[0]
    timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')[1]
    return f"{name} {timestamp}" 

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

trainer = pl.Trainer(gpus=0, max_epochs=EPOCHS, 
    callbacks=[early_stop_callback, checkpoint_callback])

trainer.fit(model, data)

# %% Test 
trainer.test(datamodule=data);

# %%
# %%
#%% Test things out
# xx,yy,xlens = next(iter(data.train_dataloader()))
# print("LABELS ARE:")
# print(yy[0:5])
# print(xx.shape, yy.shape)
# outs = model(xx)
# print(outs.shape)
# print(outs[0])
import torch
torch.utils.data.random_split(range(10), [0.1, 0.9], generator=torch.Generator().manual_seed(42))
# %%
