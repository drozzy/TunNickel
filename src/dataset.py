# %%
from genericpath import exists
import torch
import os
from torch.utils.data import Dataset, DataLoader
from numpy import genfromtxt
import numpy as np

# %% Task Data - read all of it into memory
def cache_task_data(task):
    for fname in os.listdir(f"dataset/{task}/kinematics/AllGestures"):
        data = genfromtxt(f"dataset/{task}/kinematics/AllGestures/{fname}")
        to = f"cached/{task}/{fname}.npy"
        np.save(to, data)

def read_cached(task):
    data = {}
    for fname in os.listdir(f"cached/{task}"):
        d = np.load(f"cached/{task}/{fname}")
        trial_name, _ = fname.split(".npy") 
        data[trial_name] = d
    return data
        
def read_task_data(task):
    if not os.path.exists(f"cached/{task}"):
        os.makedirs(f"cached/{task}", exist_ok=True)
        cache_task_data(task)
    return read_cached(task)

def read_data():
    data = {}
    for task in os.listdir("dataset/Experimental_Setup"):
        data[task] = read_task_data(task)
    return data

task_data = read_data()

# %% Experimental Setup data
def parse_line(line, task):
    spec, label = line.split()
    # print(spec)
    trial, from_line, to_line_txt = spec.split(f"{task}_")[1].split("_")
    to_line,_ = to_line_txt.split(".")

    gesture = f"{trial}_{from_line}_{to_line}.txt"
    return gesture, int(from_line)-1, int(to_line)-1, label

def parse_file(fname, task):    
    data = []
    for line in open(fname):
        gesture, from_line, to_line, label = parse_line(line, task)
        data.append({'gesture': gesture, 
            'from_line': from_line, 
            'to_line': to_line,
            'label': label})
    return data

def read_itr(task, user_out, itr):
    modes = {}
    # for m in ['Train', 'Test']:
    for mode in os.listdir(f"dataset/Experimental_setup/{task}/Balanced/GestureClassification/UserOut/{user_out}/{itr}"):
        fname = f"dataset/Experimental_setup/{task}/Balanced/GestureClassification/UserOut/{user_out}/{itr}/{mode}"
        modes[mode] = parse_file(fname, task)
    return modes

def read_user_out(task, user_out):
    itrs = {}
    for itr in os.listdir(f"dataset/Experimental_setup/{task}/Balanced/GestureClassification/UserOut/{user_out}"):
        itrs[itr] = read_itr(task, user_out, itr)
    return itrs

def read_exp_task(task):
    user_outs = {}
    for user_out in os.listdir(f"dataset/Experimental_setup/{task}/Balanced/GestureClassification/UserOut"):
        user_outs[user_out] = read_user_out(task, user_out)
    return user_outs

def read_experimental_setup():
    exp_tasks = {}
    for task in os.listdir("dataset/Experimental_Setup"):
        exp_tasks[task] = read_exp_task(task)
    return exp_tasks
    
# tasks, tasks_setup = experimental_setup()
exp_tasks = read_experimental_setup()
# %%

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
