# %% Imports
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torch
# %% Baseline linear model
class LSTMModel(pl.LightningModule):
    def __init__(self, num_classes, lr):
        super().__init__()
        hidden = 86
        self.lr = lr
        self.lstm = nn.LSTM(input_size=76, hidden_size=hidden, batch_first=True)
        self.final = nn.Linear(hidden, num_classes)
        self.accuracy_train = pl.metrics.Accuracy()
        self.accuracy_test = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1]
        
        return self.final(x)

    def training_step(self, batch, batch_idx):
        x, y, xlens = batch
        y_pred_logits = self(x)
        loss = F.cross_entropy(y_pred_logits, y)
        
        self.log('train_loss', loss)
        self.log("train_acc_step", self.accuracy_train(y_pred_logits, y))
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.accuracy_train.compute())

    def validation_step(self, batch, batch_idx):
        x, y, xlens = batch
        y_pred_logits = self(x)
        loss = F.cross_entropy(y_pred_logits, y)
        
        self.log('val_loss', loss)
        self.log("val_acc_step", self.accuracy_val(y_pred_logits, y))
        return loss
    
    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.accuracy_val.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, xlens = batch
        y_pred_logits = self(x)
        loss = F.cross_entropy(y_pred_logits, y)
        
        self.log('test_loss', loss)
        self.log("test_acc_step", self.accuracy_test(y_pred_logits, y))
        return loss
    
    def test_epoch_end(self, outs):
        self.log("test_acc_epoch", self.accuracy_test.compute())


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# %%
