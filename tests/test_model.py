from tunnickel.model import LSTM_Model, NODE_LSTM, Module
from importlib import resources
import torch
import torch.nn.functional as F


def test_neural_ode_model_forward():    
    # (batch, seq_len, input_size)
    batch = 12
    seq_len = 7
    input_size = 5
    num_classes = 10
    m = NODE_LSTM(num_classes=num_classes, num_features=input_size)
    
    x = torch.rand((batch, seq_len, input_size))
    assert x.shape[0] == batch
    y = m(x) # 12 x 7 x 10 -> N x C = 12*7 x 10
    assert y.shape[0] == batch
    
    y = y.reshape(batch * seq_len, -1)
    assert y.shape[1] == num_classes

    target = torch.randint(num_classes-1, (batch, seq_len,)) # 12 x 7 -> N
    target = target.reshape(-1)
    assert target.shape == (batch * seq_len,)
    
    loss = F.cross_entropy(y, target)
    assert type(loss) == torch.Tensor
    assert loss.shape == torch.Size([])
    

def test_lstm_model_forward():    
    # (batch, seq_len, input_size)
    batch = 12
    seq_len = 7
    input_size = 5
    num_classes = 10
    m = LSTM_Model(num_classes=num_classes, num_features=input_size)
    
    x = torch.rand((batch, seq_len, input_size))
    assert x.shape[0] == batch
    y = m(x) # 12 x 7 x 10 -> N x C = 12*7 x 10
    assert y.shape[0] == batch
    
    y = y.reshape(batch * seq_len, -1)
    assert y.shape[1] == num_classes

    target = torch.randint(num_classes-1, (batch, seq_len,)) # 12 x 7 -> N
    target = target.reshape(-1)
    assert target.shape == (batch * seq_len,)
    
    loss = F.cross_entropy(y, target)
    assert type(loss) == torch.Tensor
    assert loss.shape == torch.Size([])
    
def test_module_with_lstm_model_forward():
    m = LSTM_Model(num_classes=1, num_features=1)
    mo = Module(model=m)
    # (batch, seq_len, input_size)
    batch = 2
    xs = torch.tensor([ [ [0.1]], [ [10.1] ]])
    y = mo(xs)
    assert xs.shape[0] == y.shape[0] == batch