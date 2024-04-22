import scanpy as sc
import pandas as pd
import numpy as np
import scib
import sys

filepath = sys.argv[1]
query_type = sys.argv[2]
print(filepath)
print(f'/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/trimodal/metrics/{filepath[87:-8]}.csv')

adata = sc.read(filepath)

print('adata pre subsetting:')
print(adata)
adata = adata[adata.obs['type_of_query'].isin(['reference', query_type])].copy()
print('adata post subsetting:')
print(adata)

cluster_key = 'cluster'
label_key = 'l2_cell_type'
batch_key = 'type_of_query'
embed = 'latent'

df = {}
sc.pp.neighbors(adata, use_rep=embed)

scib.metrics.cluster_optimal_resolution(
    adata,
    label_key=label_key,
    cluster_key=cluster_key,
    use_rep=embed,
)

print('ARI')
df['ARI'] = scib.metrics.ari(adata, cluster_key=cluster_key, label_key=label_key)

print('NMI')
df['NMI'] = scib.metrics.nmi(adata, cluster_key=cluster_key, label_key=label_key)

print('Isolated label score ASW')
df['Isolated label score ASW'] = scib.metrics.isolated_labels_asw(
    adata, 
    label_key = label_key, 
    batch_key = batch_key, 
    embed = embed,
)

print('Label ASW')
df['Label ASW'] = scib.metrics.silhouette(
    adata,
    label_key = label_key,
    embed = embed,
)

print('Batch ASW')
df['Batch ASW'] = scib.metrics.silhouette_batch(
    adata, 
    batch_key = batch_key,
    label_key = label_key,
    embed = embed,
)

print('Graph Connectivity')
df['Graph Connectivity'] = scib.metrics.graph_connectivity(
    adata,
    label_key = label_key
)

df['Overall Score'] = 0.6 * np.mean([df['ARI'], df['NMI'], df['Isolated label score ASW'], df['Label ASW']]) + 0.4 * np.mean([df['Batch ASW'], df['Graph Connectivity']])

df = pd.DataFrame(df, index=[0])

query_type = query_type.replace(' ', '_')
df.to_csv(f'/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/trimodal/metrics/{filepath[87:-19]}_{query_type}.csv')
