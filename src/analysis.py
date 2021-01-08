import datetime, json
import numpy as np

def write_results(r):
    prefix = get_prefix()
    fname = f"{prefix}_Results.json"
    with open(fname, "w") as f:
        json.dump(r, f)
    return fname

def write_summary(s):
    prefix = get_prefix()
    fname = f"{prefix}_Summary.txt"
    with open(fname, 'w') as f:
        f.write(s)
    return fname

def read_results(fname):
    with open(fname) as f:
        return json.load(f)

def summary(r):
    summary = ""
    for model_name, model_r in r.items():
        for task, task_r in model_r.items():
            louo = task_r["LOUO"]
            user_accs = get_user_acc(louo)
            summary += get_task_summary(task, model_name, user_accs)
            
    return summary

def get_task_summary(task, model_name, user_accs):
    s = f"Task: {task} ---- {model_name} ---\n"
    s += f"Mean: {np.mean(user_accs)}\n"
    s += f"Variance: {np.var(user_accs)}\n\n"
    s += f"Stdev: {np.std(user_accs)}\n\n"
    return s            
    
def get_user_acc(louo):
    user_accs = []
    for _user_out, user_rs in louo.items():        
        user_r = user_rs[0]
        acc = user_r["test_acc_epoch"]        
        user_accs.append(acc)
    return user_accs
    
def get_prefix():
    timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')
    date = timestamp[0]
    time = timestamp[1]
    return f"{date}_{time}".replace(":", "_")
    
