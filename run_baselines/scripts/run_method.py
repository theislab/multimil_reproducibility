import argparse
from pprint import pprint
import numpy as np
import pandas as pd
import scanpy as sc
import json

from pb_rf import run_pb_rf
from gex_rf import run_gex_rf
from pb_nn import run_pb_nn
from gex_nn import run_gex_nn

METHOD_MAP = dict(
    pb_rf=dict(function=run_pb_rf, mode='rna'),
    gex_rf=dict(function=run_gex_rf, mode='rna'),
    pb_nn=dict(function=run_pb_nn, mode='rna'),
    gex_nn=dict(function=run_gex_nn, mode='rna'),
)

params = snakemake.params.params
method_params = json.loads(params['params'].replace("\'", "\"")) # this is dict
input = params['input']
method = params['method']
label_key = params['label_key']
batch_key = params['batch_key']
condition_key = params['condition_key']
sample_key = params['sample_key']
n_splits = params['n_splits']
hash = params['hash']
output_file = snakemake.output.tsv

method_mode = METHOD_MAP[method]['mode']
method_function = METHOD_MAP[method]['function']

if method_mode == 'rna':
    adata = sc.read_h5ad(input)
    df = method_function(
        adata, 
        sample_key=sample_key, 
        condition_key=condition_key, 
        label_key=label_key,
        batch_key=batch_key,
        n_splits=n_splits, 
        output_file=output_file,
        params=method_params,
        hash=hash,
    )
    df['hash'] = hash
    df['method_params'] = params['params']
    df['task'] = params['task']
    df.to_csv(output_file, sep='\t')