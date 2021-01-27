# %%
from analysis import write_results, read_results, summary, write_summary
from run_experiment import run_experiment
from m01_lstm import LSTMModel
from m02_cnn import CnnModel
from m03_neuralode_cnn import NeuralODECnnModel

# %%
def main(model_specs, tasks, user_ids, patience, max_epochs, num_gestures):
    r = run_experiment(model_specs=model_specs, tasks=tasks, user_ids=user_ids, patience=patience, max_epochs=max_epochs, num_gestures=num_gestures)
    fname = write_results(r)
    r = read_results(fname)
    s = summary(r)
    fname = write_summary(s)

if __name__ == '__main__':
    num_gestures = 15
    num_classes = num_gestures
    
    tasks = ['Knot_Tying', 'Needle_Passing', 'Suturing']

    model_specs = [
        (CnnModel, num_classes, 0.001),
        (LSTMModel, num_classes, 0.0005),
        (NeuralODECnnModel, num_classes, 0.0005)
    ]
    user_ids=[1,2,3,4,5,6,7,8]

    main(model_specs=model_specs, tasks=tasks, user_ids=user_ids, patience=500, max_epochs=1000, num_gestures=num_gestures)
    

# %%
