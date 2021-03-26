# %% Imports
import torch.nn as nn
import torch
from torchdyn.models import *
from torchdyn import *
from torch.nn import functional as F


class LstmField(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()        
        self.func = nn.LSTM(input_size=num_features, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):        
        y, _ = self.func(x)
        return y


class LSTM_Model(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, dropout):
        super().__init__()

        self.d1 = torch.nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(input_size=num_features, hidden_size=int(hidden_size),batch_first=True,bidirectional=True,)
        self.lstm2 = nn.LSTM(input_size=int(hidden_size*2), hidden_size=hidden_size, batch_first=True,bidirectional=True,)

        self.final = nn.Linear(int(hidden_size*2), num_classes)

    def forward(self, x):
        x = self.d1(x)
        x, _ = self.lstm1(x)
        x = self.d1(x)
        x, _ = self.lstm2(x)
        x = self.d1(x)
        x, _ = self.lstm2(x)
        x = self.d1(x)

        x = self.final(x)
         
        return x



class LSTM_NODE(nn.Module):
    """Good LSTM model followed by the NODE"""
    def __init__(self, num_features, num_classes, hidden_size, dropout):
        super().__init__()

        self.d1 = torch.nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(input_size=num_features, hidden_size=int(hidden_size),batch_first=True,bidirectional=True,)
        self.lstm2 = nn.LSTM(input_size=int(hidden_size*2), hidden_size=hidden_size, batch_first=True,bidirectional=True,)

        self.field = nn.Linear(in_features=2*hidden_size, out_features=2*hidden_size)
        self.neuralOde = NeuralODE(self.field)

        self.final = nn.Linear(int(hidden_size*2), num_classes)

    def forward(self, x):
        x = self.d1(x)
        x, _ = self.lstm1(x)
        x = self.d1(x)
        x, _ = self.lstm2(x)
        x = self.d1(x)
        x, _ = self.lstm2(x)
        x = self.d1(x)

        x = torch.relu(self.neuralOde(x))
        x = self.final(x)
         
        return x

class LSTM2_NODE(nn.Module):
    """ LSTM2-NODE uses [x_t, h_{t-1}, h_t]  as input to NODE. """
    def __init__(self, num_features, num_classes, hidden_size):
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
        
class Linear_Model(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size):
        super().__init__()
        self.m = nn.Linear(num_features, hidden_size)
        self.penultimate = nn.Linear(hidden_size, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.m(x)
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class NODE_Linear(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size):
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



class ANODE_Linear(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size):
        super().__init__()
        aug_dims = 5
        self.func = nn.Sequential(
                    nn.Linear(num_features+aug_dims, 1024),
                    nn.Tanh(),
                    nn.Linear(1024, num_features+aug_dims))

        self.neuralOde = NeuralODE(self.func)

        self.m = nn.Sequential(
            Augmenter(augment_func=nn.Linear(num_features, aug_dims), augment_idx=2),
            self.neuralOde,
            nn.Linear(num_features+aug_dims, num_classes)
        )

    def forward(self, x):
        return self.m(x)


class S_ANODE_Linear(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size):
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
    def __init__(self, num_features, num_classes, hidden_size):
        super().__init__()
        self.func = LstmField(num_features=num_features, hidden_size=hidden_size)
        self.neuralOde = NeuralODE(self.func)

        self.penultimate = nn.Linear(hidden_size, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = self.neuralOde(x)
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class S_NODE_LSTM(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size):
        super().__init__()
        self.func = LstmField(num_features=num_features, hidden_size=hidden_size)
        self.neuralOde = NeuralODE(self.func)

        self.penultimate = nn.Linear(hidden_size, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x_orig = x
        x = self.neuralOde(x)
        x = x + x_orig
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class S_ANODE_LSTM(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size):
        super().__init__()
        aug_dims = 8
        self.func = LstmField(num_features=num_features+aug_dims, hidden_size=hidden_size)
        self.neuralOde = NeuralODE(self.func)

        self.m = nn.Sequential(
            Augmenter(augment_dims=aug_dims, augment_idx=2),
            self.neuralOde
        )
        self.skip_aug = Augmenter(augment_dims=aug_dims, augment_idx=2)

        self.penultimate = nn.Linear(hidden_size+aug_dims, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x_orig = x
        x = self.m(x)
        x = x + self.skip_aug(x_orig)
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class ANODE_LSTM(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size):
        super().__init__()
        aug_dims = 8
        self.func = LstmField(num_features=num_features+aug_dims, hidden_size=hidden_size)
        self.neuralOde = NeuralODE(self.func)

        self.m = nn.Sequential(
            Augmenter(augment_dims=aug_dims, augment_idx=2),
            self.neuralOde
        )

        self.penultimate = nn.Linear(hidden_size+aug_dims, hidden_size)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.m(x)
        x = torch.relu(self.penultimate(x))
        x = self.final(x)
         
        return x

class CNN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size):
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
    def __init__(self, num_features, num_classes, hidden_size):
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
    def __init__(self, num_features, num_classes, hidden_size):
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
    def __init__(self, num_features, num_classes, hidden_size):
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
    def __init__(self, num_features, num_classes, hidden_size):
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
