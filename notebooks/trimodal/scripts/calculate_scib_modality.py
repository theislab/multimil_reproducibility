import scanpy as sc
import pandas as pd
import numpy as np
import scib
import sys

filepath = sys.argv[1]
print(filepath)
print(f'/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/trimodal/metrics/{filepath[87:-8]}.csv')

adata = sc.read(filepath)

print('adata pre subsetting')
print(adata)
adata = adata[adata.obs['Site'].isin(['site1', 'site2'])].copy()
print('adata post subsetting')
print(adata)

cluster_key = 'cluster'
label_key = 'l2_cell_type'
batch_key = 'Modality'
embed = 'latent'

df = {}
sc.pp.neighbors(adata, use_rep=embed)

print('Isolated label score ASW')
df['Isolated label score ASW'] = scib.metrics.isolated_labels_asw(
    adata, 
    label_key = label_key, 
    batch_key = batch_key, 
    embed = embed,
)

print('Batch ASW')
df['Batch ASW'] = scib.metrics.silhouette_batch(
    adata, 
    batch_key = batch_key,
    label_key = label_key,
    embed = embed,
)

df['Overall Score'] = 0.6 * np.mean([df['Isolated label score ASW']]) + 0.4 * np.mean([df['Batch ASW']])

df = pd.DataFrame(df, index=[0])

df.to_csv(f'/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/trimodal/metrics/{filepath[87:-19]}_m.csv')
