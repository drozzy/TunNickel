# %% 
from tunnickel.model import LinearNeuralOdeModel, LinearModel, LstmModel, Module, ResNeuralOdeModel, ResAugNeuralOdeModel, AugNeuralOdeModel, PlainNeuralOdeModel, LstmNeuralOdeModel, LstmResNeuralOdeModel, LstmResAugNeuralOdeModel, LstmAugNeuralOdeModel
from tunnickel.data import TrialsDataModule, USERS, NUM_LABELS
import torch
from pytorch_lightning import Trainer
import torch.nn.functional as F
from importlib import resources
import torch
from tunnickel.train import train
import numpy as np


MAX_EPOCHS = 10_000
BATCH_SIZE = 32
PATIENCE = 500
GPUS = 1
NUM_WORKERS = 16
DOWNSAMPLE_FACTOR = 6
NUM_FEATURES = 14 # 76
# MODEL = ResNeuralOdeModel(num_features=76, num_classes=NUM_LABELS)
# MODEL = LstmNeuralOdeModel(num_features=76, num_classes=NUM_LABELS, hidden_size=32)
# MODEL = LstmResNeuralOdeModel(num_features=76, num_classes=NUM_LABELS, hidden_size=32)
# MODEL = LstmResAugNeuralOdeModel(num_features=76, num_classes=NUM_LABELS, hidden_size=32)
MODEL = LstmModel(num_features=NUM_FEATURES, num_classes=NUM_LABELS, hidden_size=32)
# MODEL = LstmAugNeuralOdeModel(num_features=76, num_classes=NUM_LABELS, hidden_size=32)
# MODEL = LinearModel(num_features=76, num_classes=NUM_LABELS, hidden_size=32)
# MODEL = LinearNeuralOdeModel(num_features=76, num_classes=NUM_LABELS, hidden_size=32)
# More integration points along the way
# MODEL = LstmNeuralOdeModel(num_features=76, num_classes=NUM_LABELS, hidden_size=32, s_span=torch.linspace(0, 1, 10))
# MODEL = LinearModel(num_features=76, num_classes=NUM_LABELS, hidden_size=32)

# Goal: Run Neural ODE with skip connection experiment and beat the Multi-Task RNN 85.5%
with resources.path("tunnickel", f"Suturing") as trials_dir:
    accuracies = []
    for user_out in USERS:
        result = train(test_users=[user_out], model=MODEL, max_epochs=MAX_EPOCHS, 
            trials_dir=trials_dir, batch_size=BATCH_SIZE, patience=PATIENCE, 
            gpus=GPUS, num_workers=NUM_WORKERS, downsample_factor=DOWNSAMPLE_FACTOR)
        acc = result['test_acc_epoch']
        accuracies.append(acc)

accuracy = 100*np.mean(accuracies)
std      = 100*np.std(accuracies)

summary = f"Final Accuracy: {accuracy}, Std: {std}"
print(summary)

with open('results.txt', 'w') as f:
    f.write(summary)

# trainer = Trainer(max_epochs=max_epochs)
# mo = Module(num_features=76, num_classes=NUM_LABELS)

# dm = TrialsDataModule(trials_dir, test_users=test_users, train_batch_size=batch_size)

# result = trainer.fit(mo, datamodule=dm)

# trainer.test()


    # train_with_users(test_users=[USERS[0]], max_epochs=1, trials_dir=trials_dir)
# %%
