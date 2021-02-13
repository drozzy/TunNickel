from data import trial_name

def test_trialname():
    assert trial_name("B", 1) == "Suturing_B001.txt"