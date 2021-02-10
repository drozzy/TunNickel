# %%
# Method to read the data for all the tasks.
from genericpath import exists
import os
from numpy import genfromtxt
import numpy as np

def read_kinematic_data():
    data = {}
    for task in os.listdir("dataset/Experimental_setup"):
        data[task] = _read_kinematic_task_data(task)
    return data

# %% Task Data - read all of it into memory
def _cache_kinematic_task_data(task):
    for fname in os.listdir(f"dataset/{task}/kinematics/AllGestures"):
        data = genfromtxt(f"dataset/{task}/kinematics/AllGestures/{fname}")
        to = f"cached/{task}/{fname}.npy"
        np.save(to, data)

def _read_cached_kinematic_data(task):
    data = {}
    for fname in os.listdir(f"cached/{task}"):
        d = np.load(f"cached/{task}/{fname}")
        trial_name, _ = fname.split(".npy") 
        data[trial_name] = d
    return data
        
def _read_kinematic_task_data(task):
    if not os.path.exists(f"cached/{task}"):
        os.makedirs(f"cached/{task}", exist_ok=True)
        _cache_kinematic_task_data(task)
    return _read_cached_kinematic_data(task)

