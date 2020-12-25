# %% Imports
import torch.nn.functional as F
import pytorch_lightning as pl
from torchdyn.models import *
from torchdyn import *
import torch.nn as nn

# %% Baseline linear model
class NeuralODECnnModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.func = nn.Sequential(
            nn.Conv1d(in_channels=76+10, out_channels=76+10, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.neuralDE = NeuralDE(self.func)
        
        self.convs = nn.Sequential(Augmenter(augment_dims=10),
            self.neuralDE,
            nn.Conv1d(76+10, 1, 3, padding=1))

        self.final = nn.Linear(1, num_classes)

        self.accuracy_train = pl.metrics.Accuracy()
        self.accuracy_test = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.convs(x)
        x = x.mean(dim=2)
        x = self.final(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop, independent of forward
        x, y, xlens = batch
        y_pred_logits = self(x)
        loss = F.cross_entropy(y_pred_logits, y)
        
        self.log('train_loss', loss)
        self.log("train_acc_step", self.accuracy_train(y_pred_logits, y))
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.accuracy_train.compute())

    def validation_step(self, batch, batch_idx):
        x, y, xlens = batch
        y_pred_logits = self(x)
        loss = F.cross_entropy(y_pred_logits, y)
        
        self.log('val_loss', loss)
        self.log("val_acc_step", self.accuracy_val(y_pred_logits, y))
        return loss
    
    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.accuracy_val.compute())

    def test_step(self, batch, batch_idx):
        x, y, xlens = batch
        y_pred_logits = self(x)
        loss = F.cross_entropy(y_pred_logits, y)
        
        self.log('test_loss', loss)
        self.log("test_acc_step", self.accuracy_test(y_pred_logits, y))
        return loss
    
    def test_epoch_end(self, outs):
        self.log("test_acc_epoch", self.accuracy_test.compute())


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

# %%

# model = NeuralODECnnModel.load_from_checkpoint("SuperDuperChainsaw.jl-src/1x6l4af8/checkpoints/epoch=13.ckpt")

# # trainer = pl.Trainer(gpus=1, max_epochs=EPOCHS, 
# #         callbacks=[early_stop_callback],
# #         logger=WandbLogger(run_name(model)))
# #     trainer.fit(model, data)
# #     results = trainer.test(datamodule=data);
# # %% Graph stuff!
# import torch
# BATCH= 50
# from data_module import KinematicsDataModule

# data = KinematicsDataModule()
# data.prepare_data()
# data.setup()
# x_in, yn = next(iter(data.val_dataloader()))

# x_in = x_in[:BATCH, 0:1, :]
# yn = yn[:BATCH]

# x = x_in.permute(0, 2, 1)
# seq = model.model
# # %%
# x_out = seq[0](x)
# s_span = torch.linspace(0,1,100)

# trajectory = model.neuralDE.trajectory(x_out, s_span).detach().cpu()
# # %%
# trajectory_n = trajectory[:, :, :, 0]
# print(trajectory_n.size())
# # %% 
# trajectory_p = trajectory_n[:, :, 0:2] #<- select the two features here e.g. 10:12
# # Traj should be:
# # trajectory[:,i,0]
# # 100 x BATCH  x 2-features 
# # This translates to BATCH lines


# # %%
# from torchdyn.plot import plot_2D_depth_trajectory, plot_2D_state_space
# normalized_y = 8*yn/3 - 2  # No idea what's going here...

# # yn here should be the class of the line. It's binary.
# plot_2D_depth_trajectory(s_span, trajectory_p, normalized_y, BATCH)


# # %%
# # plot_2D_state_space(trajectory[:,:,-2:], yn[::10], len(X)//10)
# plot_2D_state_space(trajectory_p, normalized_y, BATCH)


# # %%
# import torch
# torch.linspace(0, 1, 2)
# %%
