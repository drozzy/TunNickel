# %%
# Method to read the data for all the tasks.
from genericpath import exists
import os
from numpy import genfromtxt
import numpy as np
from torch.utils.data import Dataset
from importlib import resources

USERS = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
ORIG_LABEL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
NEW_LABEL_IDS  = [i-1 for i in ORIG_LABEL_IDS]
LABELS = [f"G{i}" for i in ORIG_LABEL_IDS]
NUM_LABELS = len(LABELS)

USER_TO_TRIALS = {
    'B': [1, 2, 3, 4, 5],
    'C': [1, 2, 3, 4, 5],
    'D': [1, 2, 3, 4, 5],
    'E': [1, 2, 3, 4, 5],
    'F': [1, 2, 3, 4, 5],
    'G': [1, 2, 3, 4, 5],
    'H': [1,    3, 4, 5],
    'I': [1, 2, 3, 4, 5]
}

def trial_name(user, trial):
    """ Make a filename for a particular user and a trial.

    Args:
        user: A string.
        trial: An integer.

    Returns:
        A string.
    """
    return f"Suturing_{user}{trial:03}.txt"

def trial_names_for_users(users):
    """ Make a list of trial names corresponding to users.

    Make a sorted list of trial names for given users.

    Args:
        users: A list of strings.
    
    Returns: 
        A list of strings.
    """
    trial_names = []
    for user in users:
        for trial in USER_TO_TRIALS[user]:
            trial_names.append(trial_name(user, trial))
    return sorted(trial_names)

def read_kinematic_data(trial_name):
    with resources.path("tunnickel", f"Suturing") as path:
        return genfromtxt(f"{path}/kinematics/AllGestures/{trial_name}")

def read_transcription_data(trial_name):
    """ Read transcription data
 
     Args:
        trial_name: A string.

    Returns:
        A 2-D NumPy array of shape (T, 3) where T is the time axis,
        the first two columns define the range and the last column 
        is a original integer label.
    """
    with resources.path("tunnickel", f"Suturing") as path:
        return genfromtxt(f"{path}/transcriptions/{trial_name}", dtype=np.int,
            converters={2:lambda g: int(g.decode('utf-8')[1:])})


class TrialDataset(Dataset):
    def __init__(self, users):
        """ Create a dataset of trials corresponding to the given users.

        Args:
            users: A list of strings.
        """
        self.users = users
        self.trials = trial_names_for_users(users)

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial = self.trials[idx]
        x = read_kinematic_data(trial)
        return x

def read_data_and_labels(trial_name):
    kd = read_kinematic_data(trial_name)
    td = read_transcription_data(trial_name)

    seq_len = kd.shape[0]
    frames = np.arange(1, seq_len+1, dtype=np.int)

    labels = np.zeros(frames.shape, dtype=np.int)

    for start, end, label in td:
        to_label = (frames >= start) & (frames <= end)
        labels[to_label] = label
    
    labeled_only_mask = (labels != 0)

    return kd[labeled_only_mask], labels[labeled_only_mask]

# %%
# read_kinematic_data("Suturing_B001.txt")
# # read_transcription_data("Suturing_B001.txt").shape

# read_data_and_labels("Suturing_B001.txt")
# %%
