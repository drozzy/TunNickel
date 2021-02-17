from tunnickel.data import (trial_name, trial_names_for_users, 
    read_data_and_labels, TrialDataset, 
    USERS, pad_collate, TrialsDataModule, downsample)
from importlib import resources
import torch

def test_trial_name():
    assert trial_name("B", 1) == "Suturing_B001.txt"

def test_trial_names_for_users():
    assert trial_names_for_users(["C", "B"]) == [
        "Suturing_B001.txt",
        "Suturing_B002.txt",
        "Suturing_B003.txt",
        "Suturing_B004.txt",
        "Suturing_B005.txt",
        "Suturing_C001.txt",
        "Suturing_C002.txt",
        "Suturing_C003.txt",
        "Suturing_C004.txt",
        "Suturing_C005.txt"
    ]

def test_read_data_and_labels_return_correct_data_shape():
    with resources.path("tunnickel", f"Suturing") as trials_dir:
        dd, lbl = read_data_and_labels("Suturing_B001.txt", trials_dir)
        x = 5635 - 79
        assert min(lbl) == 0
        assert max(lbl) == 10
        assert dd.shape == (x, 76)
        assert lbl.shape == (x,)

def test_downsample():
    with resources.path("tunnickel", f"Suturing") as trials_dir:
        xs, ys = read_data_and_labels("Suturing_B001.txt", trials_dir)
        
        total = 5635 - 79 
        
        xd, yd = downsample(xs, ys, factor=6)

        assert xd.shape == (int(total/6), 76)
        assert yd.shape == (int(total/6),)


def test_trial_dataset_returns_correct_x_y_shapes():
    with resources.path("tunnickel", f"Suturing") as trials_dir:

        d = TrialDataset(USERS, trials_dir)
        for x,y in d:
            assert x.shape[1] == 76
            assert y.shape[0] > 0

def test_dataloader_with_dataset():
    with resources.path("tunnickel", f"Suturing") as trials_dir:
        batch_size = 3
        d = TrialDataset(users=USERS, trials_dir=trials_dir)
        dl = torch.utils.data.DataLoader(d, batch_size=batch_size, collate_fn=pad_collate)
        itt = iter(dl)
        x,y,lens =next(itt)
        assert x.shape[0] == y.shape[0] == batch_size
        assert len(lens) == batch_size

def test_datamodule():
    with resources.path("tunnickel", f"Suturing") as trials_dir:
        batch_size = 3
        dm = TrialsDataModule(trials_dir, test_users=[USERS[0]], train_batch_size=batch_size)
        dm.prepare_data()
        dm.setup(None)
    
        dl = dm.train_dataloader()
        itt = iter(dl)
        x,y,lens =next(itt)
        assert x.shape[0] == y.shape[0] == batch_size
        assert len(lens) == batch_size
