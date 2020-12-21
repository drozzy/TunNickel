# %%
import torch
import os
from torch.utils.data import Dataset, DataLoader
from numpy import genfromtxt

# %%

def parse_line(line):
    spec, label = line.split()
    n1,n2,n3,p1,p2 = spec.split("_")
    p2,_ = p2.split(".")

    fname = f"{n1}_{n2}_{n3}.txt"
    return fname, int(p1)-1, int(p2)-1

def parse_file(fname):
    data = []
    for line in open(fname):
        fname, from_line, to_line = parse_line(line)
        data.append({'fname': fname, 
            'from_line': from_line, 
            'to_line': to_line})
    return data

# %%
class JigsawsDataset(Dataset):
    def __init__(self, fname, task):
        self.fname = fname
        self.task = task

        self.data = parse_file(fname)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# %%

ds = JigsawsDataset(train_txt, task)
# %%
def trial_path(task, fname):
    trialdir = os.path.join(datasetdir, task, "kinematics", "AllGestures")
    return os.path.join(trialdir, fname)
# %%
task = "Knot_Tying"
datasetdir = "dataset"
path = os.path.join(datasetdir, "Experimental_setup", task, "Balanced", "GestureClassification", "UserOut", "1_Out", "itr_1")
train_txt = os.path.join(path, "Train.txt")

fname = 'Knot_Tying_I002.txt'
tpath = trial_path(task, fname)
ds[0]
# %%
line = open(tpath).readlines()[0]
# %%
[float(x) for x in line.split()]
# %%
# %% Read all the kinematics data into memory

my_data = genfromtxt(tpath)
# %%

trial_data = {}
for fname in os.listdir(os.path.join(datasetdir, task, "kinematics", "AllGestures")):
    trial_data[fname] = genfromtxt(trial_path(task, fname))
    
# %%
trial_data['Knot_Tying_F002.txt'].shape
# %%
