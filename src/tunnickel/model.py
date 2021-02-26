# %% Imports
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import accuracy
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from tunnickel.data import NUM_LABELS
from torchdyn.models import *
from torchdyn import *
# %%
import torch.nn.functional as F

class LstmField(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.func = nn.LSTM(input_size=num_features, hidden_size=num_features, batch_first=True)

    def forward(self, x):        
        y, _ = self.func(x)
        return y

class LSTM_Model(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, batch_first=True, num_layers=2, dropout=0.5)
        # self.penultimate = nn.Linear(num_features, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        # x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class Linear_Model(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        self.m = nn.Linear(num_features, num_features)
        self.penultimate = nn.Linear(num_features, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.m(x)
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class NODE_Linear(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        self.func = nn.Linear(in_features=num_features, out_features=num_features)
        self.neuralOde = NeuralODE(self.func)

        self.penultimate = nn.Linear(num_features, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.neuralOde(x)
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class LSTM2_NODE(nn.Module):
    """ LSTM2-NODE uses [x_t, h_{t-1}, h_t]  as input to NODE. """
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, batch_first=True)
        self.func = nn.Linear(in_features=2*hidden_size+num_features, out_features=2*hidden_size+num_features)
        self.neuralOde = NeuralODE(self.func)

        self.penultimate = nn.Linear(hidden_size*2+num_features, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h, _ = self.lstm(x)
        h_prev = torch.roll(h, shifts=1, dims=1)
        x = torch.cat((x, h_prev, h), dim=2)
        
        x = self.neuralOde(x)
        
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

# %%
class ANODE_Linear(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        aug_dims = 8
        self.func = nn.Linear(in_features=num_features+aug_dims, out_features=num_features+aug_dims)
        self.neuralOde = NeuralODE(self.func)

        self.m = nn.Sequential(
            Augmenter(augment_dims=aug_dims, augment_idx=2),
            self.neuralOde
        )
        self.penultimate = nn.Linear(num_features+aug_dims, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.m(x)
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x
class S_ANODE_Linear(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        aug_dims = 8
        self.func = nn.Linear(in_features=num_features+aug_dims, out_features=num_features+aug_dims)
        self.neuralOde = NeuralODE(self.func)

        self.m = nn.Sequential(
            Augmenter(augment_dims=aug_dims, augment_idx=2),
            self.neuralOde
        )
        self.skip_aug = Augmenter(augment_dims=aug_dims, augment_idx=2)

        self.penultimate = nn.Linear(num_features+aug_dims, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x_orig = x
        x = self.m(x)
        x = x + self.skip_aug(x_orig)
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class NODE_LSTM(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        self.func = LstmField(num_features=num_features)
        self.neuralOde = NeuralODE(self.func)

        self.penultimate = nn.Linear(num_features, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = self.neuralOde(x)
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class S_NODE_LSTM(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        self.func = LstmField(num_features=num_features)
        self.neuralOde = NeuralODE(self.func)

        self.penultimate = nn.Linear(num_features, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x_orig = x
        x = self.neuralOde(x)
        x = x + x_orig
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class S_ANODE_LSTM(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        aug_dims = 8
        self.func = LstmField(num_features=num_features+aug_dims)
        self.neuralOde = NeuralODE(self.func)

        self.m = nn.Sequential(
            Augmenter(augment_dims=aug_dims, augment_idx=2),
            self.neuralOde
        )
        self.skip_aug = Augmenter(augment_dims=aug_dims, augment_idx=2)

        self.penultimate = nn.Linear(num_features+aug_dims, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x_orig = x
        x = self.m(x)
        x = x + self.skip_aug(x_orig)
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class ANODE_LSTM(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        aug_dims = 8
        self.func = LstmField(num_features=num_features+aug_dims)
        self.neuralOde = NeuralODE(self.func)

        self.m = nn.Sequential(
            Augmenter(augment_dims=aug_dims, augment_idx=2),
            self.neuralOde
        )

        self.penultimate = nn.Linear(num_features+aug_dims, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.m(x)
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class CNN(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        self.func = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.penultimate = nn.Linear(num_features, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.func(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.penultimate(x))

        x = self.final(x)
         
        return x

class NODE_CNN(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        self.func = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.neuralOde = NeuralODE(self.func)

        self.m = nn.Sequential(
            self.neuralOde
        )
        self.penultimate = nn.Linear(num_features, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.m(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.penultimate(x))

        x = self.final(x)
         
        return x

class ANODE_CNN(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        aug_dims = 8
        self.func = nn.Sequential(
            nn.Conv1d(in_channels=num_features+aug_dims, out_channels=num_features+aug_dims, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.neuralOde = NeuralODE(self.func)

        self.m = nn.Sequential(
            Augmenter(augment_dims=aug_dims, augment_idx=1),
            self.neuralOde
        )
        self.penultimate = nn.Linear(num_features+aug_dims, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.m(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.penultimate(x))

        x = self.final(x)
         
        return x

class S_NODE_CNN(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        self.func = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.neuralOde = NeuralODE(self.func)

        self.m = nn.Sequential(
            self.neuralOde
        )
        self.penultimate = nn.Linear(num_features, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        x_orig = x
        x = x.permute(0, 2, 1)
        x = self.m(x)
        x = x.permute(0, 2, 1)
        x = x + x_orig
        x = torch.relu(self.penultimate(x))

        x = self.final(x)
         
        return x

class S_ANODE_CNN(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        aug_dims = 8
        self.func = nn.Sequential(
            nn.Conv1d(in_channels=num_features+aug_dims, out_channels=num_features+aug_dims, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.neuralOde = NeuralODE(self.func)

        self.m = nn.Sequential(
            Augmenter(augment_dims=aug_dims, augment_idx=1),
            self.neuralOde
        )
        self.skip_aug = Augmenter(augment_dims=aug_dims, augment_idx=2)
        self.penultimate = nn.Linear(num_features+aug_dims, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        x_orig = x
        x = x.permute(0, 2, 1)
        x = self.m(x)
        x = x.permute(0, 2, 1)
        x_orig = self.skip_aug(x_orig)
        x = x + x_orig
        x = torch.relu(self.penultimate(x))

        x = self.final(x)
         
        return x

class Module(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.accuracy_train = pl.metrics.Accuracy()
        self.accuracy_test = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()

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
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
# %%
