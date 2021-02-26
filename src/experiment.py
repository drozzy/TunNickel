# %% 
from pathlib import Path
from tunnickel.hybrid import HybridNeuralDE
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

def main(project_name : str, model_name : str, seed, deterministic, gpus, repeat, max_epochs, enable_logging,
            min_params, max_params):
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(seed)

    KINEMATICS_USECOLS = None
    BATCH_SIZE = 32
    PATIENCE = 500
    NUM_WORKERS = 16
    DOWNSAMPLE_FACTOR = 6
    NUM_FEATURES = len(KINEMATICS_USECOLS) if KINEMATICS_USECOLS is not None else 76
    # MODEL = ResNeuralOdeModel(num_features=NUM_FEATURES, num_classes=NUM_LABELS)
    # MODEL = LstmNeuralOdeModel(num_features=NUM_FEATURES, num_classes=NUM_LABELS, hidden_size=32)
    # MODEL = LstmResNeuralOdeModel(num_features=NUM_FEATURES, num_classes=NUM_LABELS, hidden_size=32)
    # MODEL = LstmResAugNeuralOdeModel(num_features=NUM_FEATURES, num_classes=NUM_LABELS, hidden_size=32)
    # MODEL = LstmModel(num_features=NUM_FEATURES, num_classes=NUM_LABELS, hidden_size=32)
    # MODEL = LstmAugNeuralOdeModel(num_features=NUM_FEATURES, num_classes=NUM_LABELS, hidden_size=32)
    # MODEL = LinearModel(num_features=NUM_FEATURES, num_classes=NUM_LABELS, hidden_size=32)
    # MODEL = LinearNeuralOdeModel(num_features=NUM_FEATURES, num_classes=NUM_LABELS, hidden_size=32)
    # More integration points along the way
    # MODEL = LstmNeuralOdeModel(num_features=NUM_FEATURES, num_classes=NUM_LABELS, hidden_size=32, s_span=torch.linspace(0, 1, 10))
    # MODEL = LinearModel(num_features=NUM_FEATURES, num_classes=NUM_LABELS, hidden_size=32)

    # aug_dims = 8 # 8 # 0
    # lstm_cell = torch.nn.LSTMCell(NUM_FEATURES, 64)
    # vec_f = torch.nn.Sequential(
    #    torch.nn.Linear(64+aug_dims, 64+aug_dims),
    #    nn.Tanh()
    # )
    # final  = torch.nn.Linear(64, NUM_LABELS)
    # # flow = NeuralODE(vec_f)

    # flow = torch.nn.Sequential(
    #            Augmenter(augment_dims=8, augment_idx=1),
    #            NeuralODE(vec_f),
    #            torch.nn.Linear(64+aug_dims, 64)
    # )

    # GPUS = 0
    #MODELS = [
    #   HybridNeuralDE(jump=lstm_cell, flow=flow, out=final, last_output=False)
    #]

    # Goal: Run Neural ODE with skip connection experiment and beat the Multi-Task RNN 85.5%
    with resources.path("tunnickel", f"Suturing") as trials_dir:
        accuracies = []
        for _ in range(repeat):
            for user_out in USERS:
                
                results = train(project_name=project_name, model_name=model_name, test_users=[user_out], max_epochs=max_epochs, 
                    trials_dir=trials_dir, batch_size=BATCH_SIZE, patience=PATIENCE, 
                    gpus=gpus, num_workers=NUM_WORKERS, downsample_factor=DOWNSAMPLE_FACTOR, usecols=KINEMATICS_USECOLS,
                    deterministic=deterministic,num_features=NUM_FEATURES, num_classes=NUM_LABELS, enable_logging=enable_logging,
                    min_params=min_params, max_params=max_params)
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
        choices=['LSTM2_NODE', 'ODE-RNN-Rubanova', 'CNN', 'NODE-CNN', 'S-ANODE-Linear', 'ANODE-Linear', 'NODE-Linear', 'Hybrid-NODE-LSTM', 'ANODE-LSTM', 'S-ANODE-LSTM', 'LSTM', 'Linear'],
        help='Model to train'
    )
    parser.add_argument('--project-name',
        default='TunNickel',         
        help='Project name for things like logging.'
    )
    parser.add_argument('--max-epochs',
        default=10_000,
        type=int,
        help='Max epochs to train for.')
    parser.add_argument('--repeat',
        default=1,
        type=int,
        help='How many times to repeat the experiment')
    parser.add_argument('--deterministic',
        default=False,
        action='store_true',
        help='Run training in determinstic mode for reproducibility.')
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
    main(a.project_name, a.model, a.seed, a.deterministic, a.gpus, a.repeat, a.max_epochs, not a.disable_logging,
        a.min_params, a.max_params)
