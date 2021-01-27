# %% Imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_module_louo import LouoDataModule
from data_module_loso import LosoDataModule
from pytorch_lightning.callbacks import ModelCheckpoint

def make_model(model_spec):
    (model_class, num_classes, lr) = model_spec
    return model_class(num_classes=num_classes, lr=lr)

def get_model_name(model_spec):
    (model_class, _, _) = model_spec
    return model_class.__name__

def run_experiment(model_specs, tasks, patience, max_epochs, tasks, user_ids, super_trial_ids, num_gestures):
    r = {}
    for model_spec in model_specs:        
        model = make_model(model_spec)
        model_name = get_model_name(model_spec)
        r[model_name] = model_trial(model=model, tasks=tasks, patience=patience, max_epochs=max_epochs, tasks=tasks, user_ids=user_ids, super_trial_ids=super_trial_ids, num_gestures=num_gestures)
    return r

def model_trial(model, tasks, patience, max_epochs, tasks, user_ids, super_trial_ids, num_gestures):
    r = {}
    for task in tasks:
        task_result = task_trial(task=task, model=model, patience=patience, max_epochs=max_epochs, user_ids=user_ids, super_trial_ids=super_trial_ids, num_gestures=num_gestures)
        r[task] = task_result
    return r

def task_trial(task,  model, patience, max_epochs, user_ids, super_trial_ids, num_gestures): 
    r = {}
    r["LOUO"] = louo_trials(task, model, patience, max_epochs, user_ids, num_gestures)     
    r["LOSO"] = loso_trials(task, model, patience, max_epochs, super_trial_ids, num_gestures)
    return r

def loso_trials(task, model, patience, max_epochs, super_trial_ids, num_gestures):
    r = {}
    for super_trial_out in super_trial_ids:
        super_trial_result = loso_trial(task=task, super_trial_out=super_trial_out, model=model, patience=patience, max_epochs=max_epochs, num_gestures=num_gestures)
        r[super_trial_out] = super_trial_result
    return r

def louo_trials(task, model, patience, max_epochs, user_ids, num_gestures):
    r = {}
    for user_out in user_ids: # n-way cross validation
        data_missing = (task == 'Needle_Passing' and user_out == 6)
        if data_missing:
            continue

        user_result = louo_trial(task=task, user_out=user_out, model=model, patience=patience, max_epochs=max_epochs, num_gestures=num_gestures)
        r[user_out] = user_result        
    return r

def loso_trial(task, super_trial_out, model, patience, max_epochs, num_gestures):
    data = LosoDataModule(task=task, super_trial_out=f"{super_trial_out}_Out", num_gestures=num_gestures)
    trainer = create_trainer(patience, max_epochs)
    trainer.fit(model, datamodule=data)
    results = trainer.test(datamodule=data)
    log_num_params(model)
    return results

def louo_trial(task, user_out, model, patience, max_epochs):
    data = LouoDataModule(task=task, user_out=f"{user_out}_Out", num_gestures=num_gestures) 
    trainer = create_trainer(patience, max_epochs)    
    trainer.fit(model, datamodule=data)
    results = trainer.test(datamodule=data)
    log_num_params(model)    
    return results

def create_trainer(patience, max_epochs):
    early_stop_callback = EarlyStopping(
        monitor='val_acc_epoch',
        patience=patience,
        mode='max'
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_acc_epoch',
        mode='max'
    )
    trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, 
        callbacks=[early_stop_callback, checkpoint_callback])
    return trainer

def log_num_params(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {num_parameters}")




# %%
