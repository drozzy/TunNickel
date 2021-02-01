# %%
import datetime, json
import numpy as np
import csv
import pandas as pd

def write_results(r):
    prefix = get_prefix()
    fname = f"{prefix}_Results.json"
    with open(fname, "w") as f:
        json.dump(r, f)
    return fname

def read_results(fname):
    with open(fname) as f:
        return json.load(f)

def collect_analysis(results):
    
    header=["EvalType", "Method", "Task", "AccMean", "AccStdev", "AccVar"]
    lines = []
    
    for model_name, model_r in results.items():
        for task_name, task_r in model_r.items():
            for eval_type, eval_r in task_r.items():
                # for out_id, out_r in eval_r.items():

                    accs = get_accs(eval_r)
                    mean = np.mean(accs)
                    var = np.var(accs)
                    stdev = np.std(accs)
                    
                    line = [eval_type, model_name, task_name, mean, stdev, var]
                    lines.append(line)

    lines.sort()

    r = [header]
    r.extend(lines)
    return r
    
def get_accs(eval_r):
    accs = []
    for _out, out_rs in eval_r.items():        
        our_r = out_rs[0]
        acc = our_r["test_acc_epoch"]        
        accs.append(acc)
    return accs

def get_prefix():
    timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')
    date = timestamp[0]
    time = timestamp[1]
    return f"{date}_{time}".replace(":", "_")

def get_analysis_fname():
    prefix = get_prefix()
    return f"{prefix}_Analysis.csv"

def write_analysis(r):
    l = collect_analysis(r)
    fname = get_analysis_fname()

    with open(fname, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for line in l:
            spamwriter.writerow(line)
            
    return fname


def analysis_df(fname):    
    df = pd.read_csv(fname)
    df = df.set_index(['EvalType','Method', 'Task'])
    df = df[:]*100
    return df

# # %% Test it out

# fname = "2021-01-27_00_29_02_Results.json"
# r = read_results(fname)
# write_analysis(r)
# # %%

# %%

