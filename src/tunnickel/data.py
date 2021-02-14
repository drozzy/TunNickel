# %%
# Method to read the data for all the tasks.
from genericpath import exists
import os
from numpy import genfromtxt
import numpy as np
from torch.utils.data import Dataset

USERS = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
ORIG_LABEL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
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

def read_kinematic_data(trial_name, trials_dir):
    path = f"{trials_dir}/kinematics/AllGestures/{trial_name}"
    path_binary = f"{path}.npy"
    if os.path.exists(path_binary):
        return np.load(path_binary)
    else:
        d = genfromtxt(path)
        np.save(path, d)
        return d
        

def read_transcription_data(trial_name, trials_dir):
    """ Read transcription data
 
     Args:
        trial_name: A string.

    Returns:
        A 2-D NumPy array of shape (T, 3) where T is the time axis,
        the first two columns define the range and the last column 
        is a original gesture id as an integer.
    """
    path = f"{trials_dir}/transcriptions/{trial_name}"
    path_binary = f"{path}.npy"
    if os.path.exists(path_binary):
        return np.load(path_binary)
    else:
        d = genfromtxt(path, dtype=np.int,
            converters={2:lambda g: int(g.decode('utf-8')[1:])})
        np.save(path, d)
        return d


class TrialDataset(Dataset):
    def __init__(self, users, trials_dir="tunnickel/Suturing"):
        """ Create a dataset of trials corresponding to the given users.

        Args:
            users: A list of strings.
            trials_dir: Path to the dir that contains trails. E.g. "Suturing"
        """
        self.trials_dir = trials_dir
        self.users = users
        self.trials = trial_names_for_users(users)

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial = self.trials[idx]
        x, y = read_data_and_labels(trial, self.trials_dir)
        return x, y

def read_data_and_labels(trial_name, trials_dir="Suturing"):
    """ Load data and labels.

    Args:
        trial_name: A string.

    Returns:
        A tuple with the first element being the data
        which is of the shape (seq_len, 76) and the second
        element being the 0-based class labels of shape (seq_len,).
    """
    kinematic_data = read_kinematic_data(trial_name, trials_dir)
    transcription_data = read_transcription_data(trial_name, trials_dir)

    seq_len = kinematic_data.shape[0]
    frames = np.arange(1, seq_len+1, dtype=np.int)

    gesture_ids = np.zeros(frames.shape, dtype=np.int)

    for start, end, gesture_id in transcription_data:
        to_label = (frames >= start) & (frames <= end)
        gesture_ids[to_label] = gesture_id
    
    labeled_only_mask = (gesture_ids != 0)

    labels = gesture_ids[labeled_only_mask] - 1
    return kinematic_data[labeled_only_mask], labels

# %%
# read_kinematic_data("Suturing_B001.txt")
# # read_transcription_data("Suturing_B001.txt").shape

# read_data_and_labels("Suturing_B001.txt")
# %%
# %%
# %% 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    xs = [torch.tensor(b[0], dtype=torch.float32) for b in batch]
    ys = [torch.tensor(b[1], dtype=torch.int64) for b in batch]

    xs_pad = pad_sequence(xs, batch_first=True)
    ys_pad = pad_sequence(ys, batch_first=True)
    xs_len = [len(x) for x in xs]
    ys_len = [len(y) for y in ys]

    return xs_pad, ys_pad, xs_len, ys_len

# %%
import torch
d = TrialDataset(users=USERS, trials_dir="tunnickel/Suturing")

dl = torch.utils.data.DataLoader(d, batch_size=3, collate_fn=pad_collate)
# %%
itt = iter(dl)
# %%
x,y,xl,yl =next(itt)
x.shape

# %%

# %%
