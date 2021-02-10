# %% Imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from segmentation_dataset_louo import SegmentationDataset
from segmentation_model_lstm import SegmentationLSTMModel
from segmentation_datamodule_louo import SegmentationDataModuleLouo
# from pytorch_lightning.loggers import WandbLogger
import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.loggers import TensorBoardLogger


dm = SegmentationDataModuleLouo(train_batch_size=1)
# dm.prepare_data()
# dm.setup()
# dl = dm.train_dataloader()
model = SegmentationLSTMModel()
max_epochs = 200
logger = TensorBoardLogger('tb_logs', name='SegLstm')
trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, logger=logger)


trainer.fit(model, dm)

# %% Test 
#trainer.test(datamodule=data);

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
#import torch
#torch.utils.data.random_split(range(10), [0.1, 0.9], generator=torch.Generator().manual_seed(42))
# %%
