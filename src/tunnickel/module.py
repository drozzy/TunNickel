# %% Imports
import torch.nn.functional as F
import pytorch_lightning as pl
import torch


class Module(pl.LightningModule):
    def __init__(self, model, lr=0.001):
        super().__init__()
        self.model = model
        self.lr = lr
        self.accuracy_train = pl.metrics.Accuracy()
        self.accuracy_test = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()
        self.save_hyperparameters()

    def forward(self, x):
        y = self.model(x)
        return y
    
    def training_step(self, batch, batch_idx):
        loss, _pred, _target, _lens = self.get_loss(batch)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, target, lens = self.get_loss(batch)
        self.log('val_loss', loss)
        
        for p, t, l in zip(pred, target, lens):
            p2 = p[:l].reshape(-1)
            t2 = t[:l].reshape(-1)
            self.accuracy_val(p2, t2)

    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.accuracy_val.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, pred, target, lens = self.get_loss(batch)
        self.log('test_loss', loss)
        
        for p, t, l in zip(pred, target, lens):
            p2 = p[:l].reshape(-1)
            t2 = t[:l].reshape(-1)
            self.accuracy_test(p2, t2)

    def test_epoch_end(self, outs):
        self.log("test_acc_epoch", self.accuracy_test.compute(), prog_bar=True)
        
    def get_loss(self, batch):
        x, target_orig, lens = batch
        b, s = x.shape[0], x.shape[1]
        pred_logits = self(x)                  # batch x seq_len x classes
        
        pred = torch.argmax(pred_logits, dim=2)
        pred_logits = pred_logits.reshape(b*s, -1)    # batch*seq_len, classes
        
        target = target_orig.reshape(-1)     # batch x seq_len -> batch*seq_len
       
        loss = F.cross_entropy(pred_logits, target, ignore_index=-1)
        return loss, pred, target_orig, lens

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)        
        return optimizer