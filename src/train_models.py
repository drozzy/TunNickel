# %% Imports
import csv
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_module import JigsawsDataModule
from pytorch_lightning.loggers import WandbLogger
import datetime
import numpy as np
from m01_lstm import LSTMModel
from m02_cnn import CnnModel
from m03_neuralode_cnn import NeuralODECnnModel
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb


def run_name(model_class):
    name = model_class.__name__.split("Model")[0]
    timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')[1]
    return f"{name} {timestamp}" 

def main(patience=500, max_epochs=1000, use_wandb=True):
    r = {}

    d = JigsawsDataModule(user_out=f"1_Out") 
    d.prepare_data()
    d.setup()
    num_classes = d.num_gestures

    model_classes = [
        CnnModel,
        LSTMModel,
        NeuralODECnnModel
    ]

    for model_class in model_classes:
        model_result = {}
        r[model_class.__name__] = model_result

        for user_out in [1,2,3,4,5,6,7,8]: # 8-way cross validation
            user_result = {}
            model_result[user_out] = user_result

            data = JigsawsDataModule(user_out=f"{user_out}_Out") 

            early_stop_callback = EarlyStopping(
                monitor='val_acc_epoch',
                patience=patience,
                mode='max'
            )
            model = model_class(num_classes)
            
            checkpoint_callback = ModelCheckpoint(
                save_top_k=1,
                monitor='val_acc_epoch',
                mode='max'
            )

            name = run_name(model_class)

            logger = WandbLogger(name) if use_wandb else True
            trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, 
                callbacks=[early_stop_callback, checkpoint_callback],
                logger=logger)

                
            trainer.fit(model, data)

            results = trainer.test(datamodule=data);

            num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
            if use_wandb:
                wandb.run.summary["num_parameters"] = num_parameters
                wandb.finish()

            user_result.update(*results)
        # %%
    s = summary(r)
    print(s)
    write_summary(s)

def summary(r):
    summary = ""
    for model_name, model_r in r.items():
        summary += f" --- {model_name} ---\n"
        user_accs = []
        for user_out, user_r in model_r.items():
            acc = user_r["test_acc_epoch"]
            summary += f"        UserOut_{user_out} Acc = {acc}\n"
            user_accs.append(acc)
        summary += f"Model Accuracy Mean: {np.mean(user_accs)}\n"
        summary += f"Model Accuracy Variance: {np.var(user_accs)}\n\n"
    return summary

def write_summary(s):
    timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')
    date = timestamp[0]
    time = timestamp[1]
    with open(f"Results {date} {time}.csv", 'w') as f:
        f.write(s)


def write_results(r):
    timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')[1]
    with open(f"Results_{timestamp}.csv", 'w') as f:
        for model_name, model_r in r.items():
            f.write(f" --- {model_name} ---\n")
            user_accs = []
            for user_out, user_r in model_r.items():
                acc = user_r["test_acc_epoch"]
                f.write(f"        UserOut_{user_out} Acc = {acc}\n")
                user_accs.append(acc)
            f.write(f"Model Accuracy Mean: {np.mean(user_accs)}\n\n")
            f.write(f"Model Accuracy Variance: {np.var(user_accs)}\n\n")


if __name__ == '__main__':
    main(max_epochs=1000, use_wandb=False)

