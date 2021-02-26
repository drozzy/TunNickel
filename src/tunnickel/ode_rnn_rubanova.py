# Model based Paper https://github.com/YuliaRubanova/latent_ode
# starting from code here: https://github.com/DiffEqML/torchdyn/blob/7a13ff3e7d2314ab98c41ebdd8654ac2ca48fca0/torchdyn/models/hybrid.py
# %%
import torch
import torch.nn as nn
from torchdyn.models import *
from torchdyn import *

class ODE_RNN_Rubanova(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size):
        super().__init__()

        self.hidden_dim = hidden_size
       
        f = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        torch.nn.Tanh())

        self.flow = NeuralODE(f)

        self.jump = torch.nn.LSTMCell(num_features, self.hidden_dim)
 
        self.linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, num_classes)

        self.jump_func = self._jump_latent_cell
        
    def forward(self, x):
        h = c = self._init_latent(x)
        Y = torch.zeros(x.shape[0], *h.shape)
        
        for t, x_t in enumerate(x):
            h = self.flow(h)

            h, c = self.jump_func(x_t, h, c)
            
            Y[t] = h
        Y = torch.relu(self.linear(Y))
        Y = self.out(Y)
        return Y

    def _init_latent(self, x):
        x = x[0]
        return torch.zeros(x.shape[0], self.jump.hidden_size).to(x.device)

    def _jump_latent_cell(self, *args):
        x_t, h, c = args[:3]
        return self.jump(x_t, (h, c))
# %%
