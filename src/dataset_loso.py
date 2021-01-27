# %%
from sklearn import preprocessing
import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from read_data import read_data

# %%
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

def read_itr(task, super_trial_out, itr):
    modes = {}
    # for m in ['Train', 'Test']:
    for mode in os.listdir(f"dataset/Experimental_setup/{task}/Balanced/GestureClassification/SuperTrialOut/{super_trial_out}/{itr}"):
        fname = f"dataset/Experimental_setup/{task}/Balanced/GestureClassification/SuperTrialOut/{super_trial_out}/{itr}/{mode}"
        modes[mode] = parse_file(fname, task)
    return modes

def read_super_trial_out(task, super_trial_out):
    return read_itr(task, super_trial_out, "itr_1")

def read_exp_task(task):
    super_trial_outs = {}
    for super_trial_out in os.listdir(f"dataset/Experimental_setup/{task}/Balanced/GestureClassification/SuperTrialOut"):
        super_trial_outs[super_trial_out] = read_super_trial_out(task, super_trial_out)
    return super_trial_outs

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

class LosoDataset(Dataset):
    def __init__(self, task="Knot_Tying", super_trial_out="1_Out", train=True, num_gestures = 15):        
        self.task = task
        self.super_trial_out = super_trial_out
        self.num_gestures = num_gestures
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
        e = self.experimental_setup[self.task][self.super_trial_out][self.mode][index]
        trial = e['trial']

        from_line, to_line = e['from_line'], e['to_line']
        trial_fname = f"{self.task}_{trial}.txt"

        segment = self.task_data[self.task][trial_fname][from_line:to_line]
        label = e['label']
        i = self.gestures.index(label)
        label_int = self.targets[i]
        return {'segment' : segment, 'label': label_int}

    def __len__(self):
        return len(self.experimental_setup[self.task][self.super_trial_out][self.mode])
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

# %%

# ds = LosoDataset()
# ds[0]

# %%
