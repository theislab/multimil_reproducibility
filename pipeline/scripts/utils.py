import pandas as pd
import yaml
import hashlib
import os

# from https://github.com/HCA-integration/hca_integration_toolbox/blob/main/workflow/utils/misc.py#L129
def create_hash(string: str, digest_size: int = 5):
    string = string.encode('utf-8')
    return hashlib.blake2b(string, digest_size=digest_size).hexdigest()

def create_tasks_df(config, save=None):
    tasks_df = []
    with open(config, "r") as stream:
        params = yaml.safe_load(stream)
    for task in params['TASKS']:
        task_dict = params['TASKS'][task]
        params_list = []
        method_dfs = []
        for method in task_dict['methods']:
            method_params = task_dict['methods'][method]
            df_params = pd.read_csv(method_params, sep='\t', index_col=0)
            params_list = [str(row) for row in df_params.to_dict(orient='records')]
            method_df = {}
            method_df['params'] = params_list
            method_df['hash'] = [create_hash(row + method + task) for row in params_list]
            method_df['method'] = method
            method_dfs.append(pd.DataFrame(method_df))
        method_dfs = pd.concat(method_dfs)
        method_dfs['task'] = task
        for key in task_dict:
            if key != 'methods':
                method_dfs[key] = task_dict[key] 
        
        tasks_df.append(method_dfs)
    tasks_df = pd.concat(tasks_df)
    if save is not None:
        tasks_df.to_csv(save, sep='\t')
    return tasks_df

def get_existing_checkpoints(rootdir):

    checkpoints = []

    for _, _, files in os.walk(rootdir):
        for filename in files:
            if filename.endswith('.ckpt'):
                checkpoints.append(filename.strip('.ckpt'))

    return checkpoints
                