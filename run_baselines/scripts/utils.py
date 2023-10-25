import pandas as pd
import yaml
import hashlib

# from https://github.com/HCA-integration/hca_integration_toolbox/blob/main/workflow/utils/misc.py#L129
def create_hash(string: str, digest_size: int = 5):
    string = string.encode('utf-8')
    return hashlib.blake2b(string, digest_size=digest_size).hexdigest()

def create_tasks_df(config):
    tasks_df = {}
    with open(config, "r") as stream:
        params = yaml.safe_load(stream)
    for task in params['TASKS']:
        task_dict = params['TASKS'][task]
        params_list = []
        for method in task_dict['methods']:
            method_params = task_dict['methods'][method]
            df_params = pd.read_csv(method_params, sep='\t', index_col=0)
            params_list = [str(row) for row in df_params.to_dict(orient='records')]
            tasks_df['params'] = params_list
            tasks_df['hash'] = [create_hash(row) for row in params_list]
            tasks_df['method'] = method
        for key in task_dict:
            if key != 'methods':
                tasks_df[key] = task_dict[key]
    return pd.DataFrame(tasks_df)
            