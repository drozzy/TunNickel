"""Rewrite of `train_models.py` to take into consideration task."""
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


def make_trial_name(model_class):
    name = model_class.__name__.split("Model")[0]
    timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')[1]
    return f"{name} {timestamp}" 

def run_experiment(patience=500, max_epochs=1000, use_wandb=True):
    r = {}

    d = JigsawsDataModule(user_out=f"1_Out") 
    d.prepare_data()
    d.setup()
    num_classes = d.num_gestures

    model_classes = [
        (CnnModel, 0.001),
        (LSTMModel, 0.0005),
        (NeuralODECnnModel, 0.0005)
    ]

    for (model_class, model_lr) in model_classes:
        model_result = run_model_trial(model_class=model_class, num_classes=num_classes, model_lr=model_lr, patience=patience, use_wandb=use_wandb, max_epochs=max_epochs)
        r[model_class.__name__] = model_result

    s = summary(r)
    print(s)
    write_summary(s)

def run_model_trial(model_class, num_classes, model_lr, patience, use_wandb, max_epochs):
    model_result = {}
    for task in ['Knot_Tying']:
        task_result = run_task_trial(task=task, model_class=model_class, num_classes=num_classes, model_lr=model_lr, patience=patience, use_wandb=use_wandb, max_epochs=max_epochs)
        model_result[task] = task_result
    return model_result

def run_task_trial(task,  model_class, num_classes, model_lr, patience, use_wandb, max_epochs):    
    task_result = {}
    for user_out in [1,2,3,4,5,6,7,8]: # 8-way cross validation
        user_result = run_user_trial(task=task, user_out=user_out, model_class=model_class, num_classes=num_classes, model_lr=model_lr, patience=patience, use_wandb=use_wandb, max_epochs=max_epochs)
        task_result[user_out] = user_result        
    return task_result

def run_user_trial(task, user_out, model_class, num_classes, model_lr, patience, use_wandb, max_epochs):
    user_result = {}    

    data = JigsawsDataModule(task=task, user_out=f"{user_out}_Out") 

    early_stop_callback = EarlyStopping(
        monitor='val_acc_epoch',
        patience=patience,
        mode='max'
    )
    model = model_class(num_classes, lr=model_lr)
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_acc_epoch',
        mode='max'
    )

    name = make_trial_name(model_class)

    logger = WandbLogger(name) if use_wandb else True
    trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, 
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger)

        
    trainer.fit(model, datamodule=data)

    results = trainer.test(datamodule=data);

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
    if use_wandb:
        wandb.run.summary["num_parameters"] = num_parameters
        wandb.finish()

    user_result.update(*results)
    return user_result
    
def summary(r):
    summary = ""
    for model_name, model_r in r.items():
        for task, task_r in model_r.items():
            summary += f"Task: {task} ---- {model_name} ---\n"
            user_accs = []
            for user_out, user_r in task_r.items():
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
    with open(f"Results_{date}_{time}.csv", 'w') as f:
        f.write(s)


if __name__ == '__main__':
    run_experiment(max_epochs=10, use_wandb=False)
