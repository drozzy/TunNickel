# %% 
from pathlib import Path
from tunnickel.data import USERS, NUM_LABELS
from pytorch_lightning import Trainer
from importlib import resources
from tunnickel.train import train
import numpy as np
import argparse
from pytorch_lightning import seed_everything
# %%
#KINEMATICS_USECOLS = [c-1 for c in [39, 40, 41, 51, 52, 53, 57,
#                                    58, 59, 60, 70, 71, 72, 76]] # or None for all columns

def create_experiment_name(test_users):
    return ",".join(test_users)

def main(project_name : str, model_name : str, seed, gpus, repeat, max_epochs, enable_logging,
            min_params, max_params, batch_size, patience, dropout):
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(seed)

    KINEMATICS_USECOLS = None    
    NUM_WORKERS = 16
    DOWNSAMPLE_FACTOR = 6
    NUM_FEATURES = len(KINEMATICS_USECOLS) if KINEMATICS_USECOLS is not None else 76
    
    # Goal: Run Neural ODE with skip connection experiment and beat the Multi-Task RNN 85.5%
    with resources.path("tunnickel", f"Suturing") as trials_dir:
        accuracies = []
        for _ in range(repeat):
            for user_out in USERS:
                test_users = [user_out]

                experiment_name = create_experiment_name(test_users)

                results = train(project_name, experiment_name, model_name, test_users, max_epochs, 
                    trials_dir, batch_size, patience, gpus, NUM_WORKERS, DOWNSAMPLE_FACTOR, KINEMATICS_USECOLS,
                    NUM_FEATURES, NUM_LABELS, enable_logging, min_params, max_params, dropout, seed)

                acc = results[0]['test_acc_epoch']
                accuracies.append(acc)

    accuracy = 100*np.mean(accuracies)
    std      = 100*np.std(accuracies)

    summary = f"{model_name} Final Accuracy: {accuracy}, Std: {std}"
    print(summary)


    Path("results").mkdir(parents=True, exist_ok=True)
    with open(f'results/{model_name}_{NUM_FEATURES}_{repeat}.txt', 'w') as f:
        f.write(summary)
    # %%

def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed',
        default=42, 
        type=int,
        help="Random seed for reproducibility.")
    parser.add_argument('--model',
        default='S-ANODE-LSTM', 
        choices=['LSTM_NODE', 'LSTM2_NODE', 'ODE-RNN-Rubanova', 'CNN', 'NODE-CNN', 'S-ANODE-Linear', 'ANODE-Linear', 'NODE-Linear', 'ANODE-LSTM', 'S-ANODE-LSTM', 'LSTM', 'Linear'],
        help='Model to train'
    )
    parser.add_argument('--project-name',
        default='TunNickel',         
        help='Project name for things like logging.'
    )    
    parser.add_argument('--dropout',
        default=0.5,
        type=float,
        help='Dropout values for models that use it.')
    parser.add_argument('--max-epochs',
        default=500,
        type=int,
        help='Max epochs to train for.')
    parser.add_argument('--patience',
        default=50,
        type=int,
        help='How long to wait for metric to improve before giving up on training.')        
    parser.add_argument('--batch-size',
        default=5,
        type=int,
        help='Training batch size.')        
    parser.add_argument('--repeat',
        default=1,
        type=int,
        help='How many times to repeat the experiment')
    parser.add_argument('--disable-logging',
        default=False,
        action='store_true',
        help='Disable logging of the experiment.')
    parser.add_argument('--gpus',
        default=1,
        type=int,
        help='Number of GPUs to use. Setting to 0 disables GPU training.')
    parser.add_argument('--max-params',
        default=160_000,
        type=int,
        help='Verify the maximum number of parameters a model should have.')
    parser.add_argument('--min-params',
        default=140_000,
        type=int,
        help='Verify the minimum number of parameters a model should have.')
            
    return parser

if __name__ == '__main__':
    parser = create_parser()
    a = parser.parse_args()
    main(a.project_name, a.model, a.seed, a.gpus, a.repeat, a.max_epochs, not a.disable_logging,
        a.min_params, a.max_params, a.batch_size, a.patience, a.dropout)
