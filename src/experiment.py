# %% 
from tunnickel.model import Model, Module
from tunnickel.data import TrialsDataModule, USERS, NUM_LABELS
import torch
from pytorch_lightning import Trainer
import torch.nn.functional as F
from importlib import resources
import torch
from tunnickel.train import train

MAX_EPOCHS = 1_000
BATCH_SIZE = 2

# Goal: Run Neural ODE with skip connection experiment and beat the Multi-Task RNN 85.5%
with resources.path("tunnickel", f"Suturing") as trials_dir:
    accuracies = []
    for user_out in USERS:
        acc = train(test_users=[user_out], max_epochs=MAX_EPOCHS, trials_dir=trials_dir, batch_size=BATCH_SIZE)
        accuracies.append(acc)

print("Accuracy was:")


trainer = Trainer(max_epochs=max_epochs)
mo = Module(num_features=76, num_classes=NUM_LABELS)

dm = TrialsDataModule(trials_dir, test_users=test_users, train_batch_size=batch_size)

result = trainer.fit(mo, datamodule=dm)

trainer.test()


    # train_with_users(test_users=[USERS[0]], max_epochs=1, trials_dir=trials_dir)
# %%