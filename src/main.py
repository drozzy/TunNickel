# %%
from analysis import write_results, read_results, summary, write_summary
from run_experiment import run_experiment, get_num_classes
from m01_lstm import LSTMModel
from m02_cnn import CnnModel
from m03_neuralode_cnn import NeuralODECnnModel

# %%
def main(model_specs, user_ids, patience, max_epochs):
    r = run_experiment(model_specs=model_specs, user_ids=user_ids, patience=patience, max_epochs=max_epochs)
    fname = write_results(r)
    r = read_results(fname)
    s = summary(r)
    fname = write_summary(s)

if __name__ == '__main__':
    num_classes = get_num_classes()
    task_types = ['Knot_Tying', 'Needle_Passing', 'Suturing']

    model_specs = [
        (CnnModel, num_classes, 0.001),
        (LSTMModel, num_classes, 0.0005),
        (NeuralODECnnModel, num_classes, 0.0005)
    ]
    user_ids=[1,2,3,4,5,6,7,8]

    main(model_specs=model_specs, task_types=task_types, user_ids=user_ids, patience=500, max_epochs=1000)
    

# %%

# trials = [
#     (CnnModel, 0.001)
# ]
# user_ids = [1]
# r = run_experiment(trials=trials, user_ids=user_ids, num_classes=get_num_classes(), patience=1, max_epochs=2)

# fname = write_results(r)
# r = read_results("2021-01-08_02_17_14_Results.json")
# s = summary(r)
# fname = write_summary(s)
# %%
