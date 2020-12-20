# 0-Augmentation:
# https://torchdyn.readthedocs.io/en/latest/tutorials/04_augmentation_strategies.html#Neural-ODE-(0-augmentation)

# %% Imports
import torch.nn.functional as F
import pytorch_lightning as pl
from torchdyn.models import *
from torchdyn import *
import torch.nn as nn
from data_module import KinematicsDataModule
WINDOW_SIZE = 400
class ANodeModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        f = nn.Sequential(
            nn.Linear(16+5, 32),
            nn.Tanh(),
            nn.Linear(32, 16+5)
        )

        self.neuralDE = NeuralDE(f, sensitivity='adjoint', 
            solver='dopri5')

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(76*WINDOW_SIZE, 16),
            nn.ReLU(),
            Augmenter(augment_dims=5),
            self.neuralDE,
            nn.Linear(16+5, 25)
        )

        self.model = model
        self.accuracy_train = pl.metrics.Accuracy()
        self.accuracy_test = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy_train(y_hat, y))
    
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy_train.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log('val_loss', loss)
        self.log('val_acc_step', self.accuracy_val(y_hat, y))
    
        return loss

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.accuracy_val.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log('test_loss', loss)
        self.log('test_acc_step', self.accuracy_test(y_hat, y))
    
        return loss

    def test_epoch_end(self, outs):
        self.log('test_acc_epoch', self.accuracy_test.compute())

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)
