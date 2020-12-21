# %%
import torch
import os
from torch.utils.data import Dataset, DataLoader
from numpy import genfromtxt
import numpy as np
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
def trial_dir(task):
    return os.path.join(datasetdir, task, "kinematics", "AllGestures")

def trial_path(task, fname):
    trialdir = os.path.join(datasetdir, task, "kinematics", "AllGestures")
    return os.path.join(trialdir, fname)
# %%


def cache_task_data(task):
    for fname in os.listdir(os.path.join(datasetdir, task, "kinematics", "AllGestures")):
        d = genfromtxt(trial_path(task, fname))
        trial_data[fname] = d
        oname = os.path.join("cached", task, f"{fname}.npy")
        np.save(oname, d)


def read_cached(task):
    trial_data = {}
    for fname in os.listdir(f"cached/{task}"):
        d = np.load(os.path.join(f"cached/{task}", fname))
        trial_name, _ = fname.split(".npy") 
        trial_data[trial_name] = d
    return trial_data
        

def read_task_data(task):
    if not os.path.exists(f"cached/{task}"):
        os.makedirs(f"cached/{task}", exist_ok=True)
        cache_task_data(task)
    return read_cached(task)

read_task_data("Knot_Tying");#.keys()
# %%
class JigsawsDataset(Dataset):
    """
    fname - Experiment setup filename. e.g.:
        "dataset/Experimental_setup/Knot_Tying/Balanced/GestureClassification/UserOut/1_Out/itr_1/Train.txt"
    task - Task, e.g. "Knot_Tying"
    """
    def __init__(self, fname, task):
        self.fname = fname
        self.task = task
        self.task_data = read_task_data(task)
        self.data = parse_file(fname)

    def __getitem__(self, index):
        e = self.data[index]
        fname = e['fname']
        from_line, to_line = e['from_line'], e['to_line']
        return {'segment' : self.task_data[fname][from_line:to_line]}

    def __len__(self):
        return len(self.data)

task = "Knot_Tying"
datasetdir = "dataset"
path = os.path.join(datasetdir, "Experimental_setup", task, "Balanced", "GestureClassification", "UserOut", "1_Out", "itr_1")
train_txt = os.path.join(path, "Train.txt")
ds = JigsawsDataset(train_txt, task)

fname = 'Knot_Tying_I002.txt'
tpath = trial_path(task, fname)
ds[0]['segment'].shape

# %%
ds[1]['segment'].shape
# %%
