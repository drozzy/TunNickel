# Turns out that Experimantal Setup Gesture recognition folder gives me 
# everything I need to setup the dataset from the beginnign.
# Key ideas:
# 1. Preload all data into dataset given train/test mode, task,
#  user-out setting right away (dont do it in getindex)
# 2. Can simply the dataset class significantly
# Pseudocode
self.labels = {}
self.trials_data = read_data() 
self.masks = {}
For line in GestureRecognition/UserOut/Train[Test]:
    # line is:  "Knot_Tying_C001_000037_001227.txt  Knot_Tying_C001.txt"
    from_step, to_step, trial_key <- line.split()
    # trial_key is thing on the right "Knot_Tying_C001.txt"   ^
    fname = trial_key
    
    trial_data = self.trials_data[trial_key]
    self.labels[trial_key] = get_labels(trial_data, trial_key) # have already

    mask = np.zeroes(length(trial-data))
    mask[from_step-1:to_step] = 1
    self.masks[trial_key] = mask

def getindex(i):
    trial_key = self.trial_keys_sorted[i]
    trial_data = self.trials_data[trial_key]
    mask = self.masks[trial_key]
    labels = self.labels[trial_key]
    return trial_data, labels, mask