# %% Imports
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class SegmentationLSTMModel(pl.LightningModule):
    def __init__(self, lr=0.001, num_classes=15):
        super().__init__()
        self.num_classes=num_classes
        hidden = 86
        self.lr = lr
        self.lstm = nn.LSTM(input_size=76, hidden_size=hidden, batch_first=True)
        self.final = nn.Linear(hidden, num_classes)
        self.accuracy_train = pl.metrics.Accuracy()
        self.accuracy_test = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()

    def forward(self, batch):
        (xx_pad, _yy_pad, x_lens, _y_lens, _xx_mask_pad) = batch
        xx_packed = pack(xx_pad, x_lens, batch_first=True, enforce_sorted=False)
        
        yy_pred_packed, _ = self.lstm(xx_packed)
        yy_pred, _yy_pred_lens = unpack(yy_pred_packed)
        yy_pred = self.final(yy_pred)

        return yy_pred.permute(1, 0, 2)


    def training_step(self, batch, batch_idx):
        # x, y, xlens = batch
        _xx_pad, yy_pad, _x_lens, _y_lens, xx_mask_pad = batch
        yy_pred = self(batch)
        # print(f"yy_pred shape: {yy_pred.shape}")
        yy_pred = yy_pred.reshape(-1, self.num_classes)
        yy_target = yy_pad.view(-1)

        # print(f"yy pred: {yy_pred.shape}")
        # print(f"yy_target: {yy_target.shape}")
        loss = F.cross_entropy(yy_pred, yy_target)
        
        self.log('train_loss', loss)
        self.log("train_acc_step", self.accuracy_train(yy_pred, yy_target))
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.accuracy_train.compute())

    # def validation_step(self, batch, batch_idx):
    #     x, y, xlens = batch
    #     y_pred_logits = self(x)
    #     loss = F.cross_entropy(y_pred_logits, y)
        
    #     self.log('val_loss', loss)
    #     self.log("val_acc_step", self.accuracy_val(y_pred_logits, y))
    #     return loss
    
    # def validation_epoch_end(self, outs):
    #     self.log("val_acc_epoch", self.accuracy_val.compute(), prog_bar=True)

    # def test_step(self, batch, batch_idx):
    #     x, y, xlens = batch
    #     y_pred_logits = self(x)
    #     loss = F.cross_entropy(y_pred_logits, y)
        
    #     self.log('test_loss', loss)
    #     self.log("test_acc_step", self.accuracy_test(y_pred_logits, y))
    #     return loss
    
    # def test_epoch_end(self, outs):
    #     self.log("test_acc_epoch", self.accuracy_test.compute())


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# %%
