from tunnickel.data import trial_name, trial_names_for_users, read_data_and_labels

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

def test_read_data_and_labels():
    dd, lbl = read_data_and_labels("Suturing_B001.txt")
    x = 5635 - 79
    assert min(lbl) == 0
    assert max(lbl) == 10
    assert dd.shape == (x, 76)
    assert lbl.shape == (x,)
   