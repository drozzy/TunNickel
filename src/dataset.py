# %%
from sklearn import preprocessing
from genericpath import exists
import torch
import os
from torch.utils.data import Dataset, DataLoader
from numpy import genfromtxt
import numpy as np


# Task Data - read all of it into memory
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
    for task in os.listdir("dataset/Experimental_setup"):
        data[task] = read_task_data(task)
    return data

# task_data = read_data()
# # %%
# task_data.keys()

# %% Experimental Setup data
def parse_line(line, task):
    spec, label = line.split()
    # print(spec)
    trial, from_line, to_line_txt = spec.split(f"{task}_")[1].split("_")
    to_line,_ = to_line_txt.split(".")

    # trial_fname = f"{task}_{trial}.txt"
    return trial, int(from_line)-1, int(to_line)-1, label

def parse_file(fname, task):    
    data = []
    for line in open(fname):
        trial, from_line, to_line, label = parse_line(line, task)
        data.append({'trial': trial, 
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
    return read_itr(task, user_out, "itr_1")

def read_exp_task(task):
    user_outs = {}
    for user_out in os.listdir(f"dataset/Experimental_setup/{task}/Balanced/GestureClassification/UserOut"):
        user_outs[user_out] = read_user_out(task, user_out)
    return user_outs

def read_experimental_setup():
    exp_tasks = {}
    for task in os.listdir("dataset/Experimental_setup"):
        exp_tasks[task] = read_exp_task(task)
    return exp_tasks
# %%
# # tasks, tasks_setup = experimental_setup()
# exp_setup = read_experimental_setup()
# # %%
# gg = [f"G{i}" for i in range(1,16)]
# gg
def oneint(value, all_values):
    i = all_values.index(value)
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(all_values)
    return targets[i]

# oneint("G1", gg)

# TODO: Rename to "ClassificationDataset"
class JigsawsDataset(Dataset):
    def __init__(self, task="Knot_Tying", user_out="1_Out", train=True):        
        self.task = task
        self.user_out = user_out
        self.num_gestures = 15
        self.gestures = [f"G{i}" for i in range(1, self.num_gestures+1)]

        le = preprocessing.LabelEncoder()
        self.targets = le.fit_transform(self.gestures)

        if train:
            self.mode = 'Train.txt'
        else:
            self.mode = 'Test.txt'

        self.task_data = read_data()
        self.experimental_setup = read_experimental_setup()

    def __getitem__(self, index):
        e = self.task_data
        e = self.experimental_setup[self.task][self.user_out][self.mode][index]
        trial = e['trial']

        from_line, to_line = e['from_line'], e['to_line']
        trial_fname = f"{self.task}_{trial}.txt"

        segment = self.task_data[self.task][trial_fname][from_line:to_line]
        label = e['label']
        i = self.gestures.index(label)
        label_int = self.targets[i]
        return {'segment' : segment, 'label': label_int}

    def __len__(self):
        return len(self.experimental_setup[self.task][self.user_out][self.mode])

class SegmentationDataset(Dataset):
    def __init__(self, task="Knot_Tying", user_out="1_Out", train=True):        
        self.task = task
        self.user_out = user_out
        self.num_gestures = 15
        self.gestures = [f"G{i}" for i in range(1, self.num_gestures+1)]

        le = preprocessing.LabelEncoder()
        self.targets = le.fit_transform(self.gestures)

        if train:
            self.mode = 'Train.txt'
        else:
            self.mode = 'Test.txt'

        self.data = read_data()
        self.trial_data = self.data[task]
        self.trials = sorted(self.trial_data.keys())

        self.experimental_setup = read_experimental_setup()

    def __getitem__(self, index):
        trial = self.trials[index]
        # TODO: Add label as well
        return self.trial_data[trial]

    def __len__(self):
        return len(self.trials)
# %%

# task = "Knot_Tying"
# datasetdir = "dataset"
# path = os.path.join(datasetdir, "Experimental_setup", task, "Balanced", "GestureClassification", "UserOut", "1_Out", "itr_1")
# train_txt = os.path.join(path, "Train.txt")
# ds = JigsawsDataset()
# # %%
# ds[1]
# # %%
# dl = DataLoader(ds)
# d = next(iter(dl))
# d['segment'].shape

# %% Segmentation dataset test

task = "Knot_Tying"
datasetdir = "dataset"
path = os.path.join(datasetdir, "Experimental_setup", task, "Balanced", "GestureClassification", "UserOut", "1_Out", "itr_1")
train_txt = os.path.join(path, "Train.txt")
ds = SegmentationDataset()
ds[0].shape
# %%


# %%
# dl = DataLoader(ds, batch_size=2)
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    yy = torch.tensor([b['label'] for b in batch])
    xx = [torch.tensor(d['segment'], dtype=torch.float32) for d in batch]
    xx_pad = pad_sequence(xx, batch_first=True)
    x_lens = [len(x) for x in xx]

    return xx_pad, yy, x_lens


# dl = DataLoader(dataset=ds, batch_size=32, shuffle=True, collate_fn=pad_collate)
# xx, yy, x_lens = next(iter(dl))
# print(xx.shape)
# print(yy.shape)
# print(x_lens)
# %%
