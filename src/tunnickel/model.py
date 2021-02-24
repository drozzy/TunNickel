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

class LstmModel(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32, dropout=0):
        super().__init__()
        self.m = LstmField(num_features=num_features)
        self.penultimate = nn.Linear(num_features, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.m(x)
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class LinearModel(nn.Module):
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

class LinearNeuralOdeModel(nn.Module):
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

class LinearAugNeuralOdeModel(nn.Module):
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

class LstmNeuralOdeModel(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32, s_span=torch.linspace(0, 1, 2)):
        super().__init__()
        self.func = LstmField(num_features=num_features)
        self.neuralOde = NeuralODE(self.func, s_span=s_span)

        # self.m = nn.Sequential(
        #     self.neuralOde
        # )
        self.penultimate = nn.Linear(num_features, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = self.neuralOde(x)
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class LstmResNeuralOdeModel(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32):
        super().__init__()
        self.func = LstmField(num_features=num_features)
        self.neuralOde = NeuralODE(self.func)

        # self.m = nn.Sequential(
        #     self.neuralOde
        # )
        self.penultimate = nn.Linear(num_features, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x_orig = x
        x = self.neuralOde(x)
        x = x + x_orig
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class LstmResAugNeuralOdeModel(nn.Module):
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

class LstmAugNeuralOdeModel(nn.Module):
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


class PlainNeuralOdeModel(nn.Module):
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

class AugNeuralOdeModel(nn.Module):
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

class ResNeuralOdeModel(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32, skip_weight=0.1):
        super().__init__()
        self.func = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.skip_weight = skip_weight
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
        x = x + self.skip_weight*x_orig
        x = torch.relu(self.penultimate(x))

        x = self.final(x)
         
        return x

class ResAugNeuralOdeModel(nn.Module):
    def __init__(self, num_features=76, num_classes=NUM_LABELS, hidden_size=32, skip_weight=0.1):
        super().__init__()
        aug_dims = 8
        self.func = nn.Sequential(
            nn.Conv1d(in_channels=num_features+aug_dims, out_channels=num_features+aug_dims, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.skip_weight = skip_weight
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
        x = x + self.skip_weight*x_orig
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
        loss, pred, target, lens = self.get_loss(batch)
        #accs = []
        #for p, t, l in zip(pred, target, lens):
            #self.log("train_acc_step", self.accuracy_train(p[:l], t[:l]))

        self.log("train_loss", loss)

        return loss


    def validation_step(self, batch, batch_idx):
        loss, pred, target, lens = self.get_loss(batch)
        self.log('val_loss', loss)
        accs = []
        for p, t, l in zip(pred, target, lens):
            p2 = p[:l].reshape(-1)
            t2 = t[:l].reshape(-1)
            self.log("val_acc_step", self.accuracy_val(p2, t2))

    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.accuracy_val.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, pred, target, lens = self.get_loss(batch)
        self.log('test_loss', loss)
        accs = []
        for p, t, l in zip(pred, target, lens):
            p2 = p[:l].reshape(-1)
            t2 = t[:l].reshape(-1)
            self.log("test_acc_step", self.accuracy_test(p2, t2))

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
