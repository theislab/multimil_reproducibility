import time 
start_time = time.time()
import argparse
from pprint import pprint
import numpy as np
import pandas as pd
import scanpy as sc
import ast

from pb_rf import run_pb_rf
from gex_rf import run_gex_rf
from pb_nn import run_pb_nn
from gex_nn import run_gex_nn
from pb_mr import run_pb_mr
from gex_mr import run_gex_mr
from ct_pb_nn import run_ct_pb_nn
from ct_pb_rf import run_ct_pb_rf
from ct_pb_mr import run_ct_pb_mr
from freq_mr import run_freq_mr
from freq_rf import run_freq_rf
from freq_nn import run_freq_nn
from run_multigrate_full import run_multigrate
from run_multigrate_mil import run_multigrate_mil

print('--- %s seconds ---' % (time.time() - start_time))

METHOD_MAP = dict(
    pb_rf=dict(function=run_pb_rf, mode='rna'),
    gex_rf=dict(function=run_gex_rf, mode='rna'),
    pb_nn=dict(function=run_pb_nn, mode='rna'),
    gex_nn=dict(function=run_gex_nn, mode='rna'),
    pb_mr=dict(function=run_pb_mr, mode='rna'),
    gex_mr=dict(function=run_gex_mr, mode='rna'),
    ct_pb_nn=dict(function=run_ct_pb_nn, mode='rna'),
    ct_pb_rf=dict(function=run_ct_pb_rf, mode='rna'),
    ct_pb_mr=dict(function=run_ct_pb_mr, mode='rna'),
    freq_mr=dict(function=run_freq_mr, mode='rna'),
    freq_rf=dict(function=run_freq_rf, mode='rna'),
    freq_nn=dict(function=run_freq_nn, mode='rna'),
    multigrate=dict(function=run_multigrate, mode='paired'),
    multigrate_reg=dict(function=run_multigrate, mode='paired'),
    multigrate_mil=dict(function=run_multigrate_mil, mode='embed'),
    multigrate_mil_reg=dict(function=run_multigrate_mil, mode='embed'),
)

params = snakemake.params.params

method_params = ast.literal_eval(params['params']) # this is dict
input1 = params['input1']
input2 = params['input2']
label_key = params['label_key']
batch_key = params['batch_key']
condition_key = params['condition_key']
condition_regression_key = params.get('condition_regression_key', None)
sample_key = params['sample_key']
n_splits = params['n_splits']
h = params['hash']
method = params['method']
task = params['task']
output_file = snakemake.output.tsv

method_mode = METHOD_MAP[method]['mode']
method_function = METHOD_MAP[method]['function']

regression = False
if (method == 'multigrate_mil_reg') or (method == 'multigrate_reg'):
    regression = True

if method_mode == 'rna' or method_mode == 'embed':
    adata = sc.read_h5ad(input1)
    df = method_function(
        adata, 
        sample_key=sample_key, 
        condition_key=condition_key, 
        label_key=label_key,
        batch_key=batch_key,
        n_splits=n_splits, 
        output_file=output_file,
        params=method_params,
        hash=h,
        method=method,
        task=task,
        regression=regression,
        condition_regression_key=condition_regression_key,
    )
elif method_mode == 'paired':
    adata1 = sc.read_h5ad(input1)
    adata2 = None
    if input2 is not None:
        adata2 = sc.read_h5ad(input2)
    df = method_function(
        adata1=adata1,
        adata2=adata2, 
        sample_key=sample_key, 
        condition_key=condition_key, 
        label_key=label_key,
        batch_key=batch_key,
        n_splits=n_splits, 
        output_file=output_file,
        params=method_params,
        hash=h,
        method=method,
        task=task,
        regression=regression,
        condition_regression_key=condition_regression_key,
    )

df['hash'] = h
df['method_params'] = params['params']
df['task'] = task
df.to_csv(output_file, sep='\t')
