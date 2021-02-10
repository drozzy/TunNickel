# %%
from sklearn import preprocessing
from torch.utils.data import Dataset
import numpy as np
from read_data import read_kinematic_data

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

        self.kinematic_data = read_kinematic_data()
        self.labels = {}
        self.trials_kinematic_data = self.kinematic_data[task]
        self.masks = {}
        self.trial_keys = []

        for line in open(f"dataset/Experimental_setup/{self.task}/unBalanced/GestureRecognition/UserOut/{self.user_out}/itr_1/{self.mode}"):
            # line is:  "Knot_Tying_C001_000037_001227.txt  Knot_Tying_C001.txt"
            _, trial_key = line.split()
            trial_kinematic_data = self.trials_kinematic_data[trial_key]
            labels, mask = self.labels_and_mask(trial_kinematic_data, trial_key) 

            self.labels[trial_key] = labels
            self.masks[trial_key] = mask
            self.trial_keys.append(trial_key)
            
        self.trial_keys = sorted(self.trial_keys)

    def __getitem__(self, index):
        trial_key = self.trial_keys[index]
        trial_kinematic_data = self.trials_kinematic_data[trial_key]
        labels = self.labels[trial_key]
        mask = self.masks[trial_key]

        return trial_kinematic_data, labels, mask

    def __len__(self):
        return len(self.trial_keys)

    def labels_and_mask(self, trial_kinematic_data, trial_name):
        length, _ = trial_kinematic_data.shape
        labels = np.zeros(length)
        # zeros where there is no signal, 1 where there is
        mask = np.zeros(length)
        
        path = f"dataset/{self.task}/transcriptions/{trial_name}"
        with open(path, 'r') as f:
            for line in f:
                from_step, to_step, gesture, _ = line.split(' ')
                from_step = int(from_step)
                to_step = int(to_step)

                i = self.gestures.index(gesture)
                label = self.targets[i]

                labels[from_step - 1: to_step] = label
                mask[from_step - 1: to_step] = 1

        return labels, mask



# task = "Knot_Tying"
# datasetdir = "dataset"
# path = os.path.join(datasetdir, "Experimental_setup", task, "Balanced", "GestureClassification", "UserOut", "1_Out", "itr_1")
# train_txt = os.path.join(path, "Train.txt")
ds = SegmentationDataset()
# trial_data, labels, mask = ds[0]
# print(trial_data.shape, labels.shape, mask.shape)
# print(labels[84:99])
# print(mask)
# %%
ds[0]
ds.trial_keys
# %%
