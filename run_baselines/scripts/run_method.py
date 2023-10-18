import argparse
from pprint import pprint
import numpy as np
import pandas as pd
import scanpy as sc

from pb_rf import run_pb_rf
from gex_rf import run_gex_rf

METHOD_MAP = dict(
    pb_rf=dict(function=run_pb_rf, mode='rna'),
    gex_rf=dict(function=run_gex_rf, mode='rna'),
)

input_files = snakemake.input
output_file = snakemake.output.tsv
method = snakemake.wildcards.method
label_key = snakemake.params.label_key
batch_key = snakemake.params.batch_key
condition_key = snakemake.params.condition_key
sample_key = snakemake.params.sample_key
n_splits = snakemake.params.n_splits
params = snakemake.params.params

method_mode = METHOD_MAP[method]['mode']
method_function = METHOD_MAP[method]['function']

params_df = pd.read_table(params, index_col=0)

best_df = None
best_accuracy = -1

for i in range(len(params_df)):
    params = params_df.iloc[i].to_dict()

    if method_mode == 'rna':
        adata = sc.read_h5ad(input_files[0])
        accuracy, df = method_function(
            adata, 
            sample_key=sample_key, 
            condition_key=condition_key, 
            label_key=label_key,
            batch_key=batch_key,
            n_splits=n_splits, 
            output_file=output_file,
            params=params,
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_df = df
            best_df['params'] = str(params)

best_df.to_csv(output_file, sep='\t')