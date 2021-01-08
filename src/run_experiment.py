# %% Imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_module import JigsawsDataModule
from pytorch_lightning.callbacks import ModelCheckpoint

def run_experiment(trials, num_classes, patience, max_epochs, user_ids):
    r = {}
    for trial in trials:
        model_name, model_result = run_trial(trial, num_classes, patience, max_epochs, user_ids)
        r[model_name] = model_result
    return r

def run_trial(model_trial, num_classes, patience, max_epochs, user_ids):
    (model_class, model_lr) = model_trial
    n = model_class.__name__
    m = model_class(num_classes, lr=model_lr)
    r = run_model_trial(model=m, patience=patience, max_epochs=max_epochs, user_ids=user_ids)

    return n, r

def run_model_trial(model, patience, max_epochs, user_ids):
    model_result = {}
    for task in ['Knot_Tying', 'Needle_Passing', 'Suturing']:
        task_result = run_task_trial(task=task, model=model, patience=patience, max_epochs=max_epochs, user_ids=user_ids)
        model_result[task] = task_result
    return model_result

def run_task_trial(task,  model, patience, max_epochs, user_ids): 
    task_result = {}
    r = run_user_out_trials(task, model, patience, max_epochs, user_ids)
    task_result["LOUO"] = r
    
    return task_result

def run_user_out_trials(task, model, patience, max_epochs, user_ids):
    r = {}
    for user_out in user_ids: # n-way cross validation
        data_missing = (task == 'Needle_Passing' and user_out == 6)
        if data_missing:
            continue

        user_result = run_user_trial(task=task, user_out=user_out, model=model, patience=patience, max_epochs=max_epochs)
        r[user_out] = user_result        
    return r

def run_user_trial(task, user_out, model, patience, max_epochs):
    data = JigsawsDataModule(task=task, user_out=f"{user_out}_Out") 
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

def get_num_classes():
    d = JigsawsDataModule(user_out=f"1_Out")
    d.prepare_data()
    d.setup()
    return d.num_gestures 



# %%
