# %% Imports
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from tunnickel.data import NUM_LABELS
# %%
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, 
            hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out

class Module(pl.LightningModule):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, lr=0.001):
        super().__init__()
        self.model = Model(num_features, num_classes)

        self.accuracy_train = pl.metrics.Accuracy()
        self.accuracy_test = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()

    def forward(self, x):
        y = self.model(x)
        return y
    
    def training_step(self, batch, batch_idx):
        loss, pred, target = self.get_loss(batch)
        self.log("train_acc_step", self.accuracy_train(pred, target))

        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.accuracy_train.compute())

    def validation_step(self, batch, batch_idx):
        loss, pred, target = self.get_loss(batch)
        self.log('val_loss', loss)
        self.log("val_acc_step", self.accuracy_val(pred, target))

    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.accuracy_val.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, pred, target = self.get_loss(batch)
        self.log('test_loss', loss)
        self.log("test_acc_step", self.accuracy_test(pred, target))

    def test_epoch_end(self, outs):
        self.log("test_acc_epoch", self.accuracy_test.compute(), prog_bar=True)
        
    def get_loss(self, batch):
        x, target, lens = batch
        b, s = x.shape[0], x.shape[1]

        pred = self(x)                  # batch x seq_len x classes
        pred = pred.reshape(b*s, -1)    # batch*seq_len, classes

        target = target.reshape(-1)     # batch x seq_len -> batch*seq_len

        loss = F.cross_entropy(pred, target)
        return loss, pred, target

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
# %%