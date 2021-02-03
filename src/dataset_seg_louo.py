# %%
from sklearn import preprocessing
from torch.utils.data import Dataset
import numpy as np
from read_data import read_data

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
        self.trials_data = self.data[task]
        self.trials = sorted(self.trials_data.keys())

    def __getitem__(self, index):
        trial_name = self.trials[index]
        trial_data = self.trials_data[trial_name]
        labels, mask = self.labels_and_mask(trial_data, trial_name)

        return trial_data, labels, mask

    def __len__(self):
        return len(self.trials)

    def labels_and_mask(self, trial_data, trial_name):

        length, _ = trial_data.shape
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
# ds = SegmentationDataset()
# trial_data, labels, mask = ds[0]
# print(trial_data.shape, labels.shape, mask.shape)
# print(labels[84:99])
# print(mask)
# %%
