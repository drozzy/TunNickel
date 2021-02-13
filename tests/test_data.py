from data import trial_name, trial_names_for_users

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