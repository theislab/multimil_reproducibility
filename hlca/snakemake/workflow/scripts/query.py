import scanpy as sc
import pandas as pd
import multigrate as mtg
import multimil as mtm
from pathlib import Path
import shutil

from sklearn.metrics import classification_report

import torch
from matplotlib import pyplot as plt
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import math
import os
import random
import scvi
import copy
# from sklearn.metrics import classification_report

print('============ Multigrate querying ============')

torch.set_float32_matmul_precision('medium')

input_files = snakemake.input
model_params = snakemake.params['model_params']
donor = snakemake.params['donor'],
condition = snakemake.params['condition'],
n_splits = snakemake.params['n_splits']
subset_umap = snakemake.params['subset_umap']
organize_params = snakemake.params['organize_params']
setup_params = snakemake.params['setup_params']
umap_colors = snakemake.params['umap_colors']

# train params
lr = model_params.pop('lr')
batch_size = model_params.pop('batch_size')
kl = model_params.pop('kl')

train_params = snakemake.params['train_params']

output_files = snakemake.output

# clean up
condition = condition[0]
donor = donor[0]

#model_params['n_layers_decoders'] = [model_params['n_layers_decoders']]
#model_params['n_layers_encoders'] = [model_params['n_layers_encoders']]

i = int(snakemake.wildcards.split)
checkpoint = snakemake.wildcards.checkpoint

adata = sc.read(input_files[0])

adata = mtg.data.organize_multiome_anndatas(
    adatas = [[adata]],
    **organize_params
    )

query = adata[adata.obs[f"split{i}"] == "val"].copy()
adata = adata[adata.obs[f"split{i}"] == "train"].copy()

idx = adata.obs[donor].sort_values().index
adata = adata[idx].copy()

mtm.model.MultiVAE_MIL.setup_anndata(adata, **setup_params)

mil = mtm.model.MultiVAE_MIL(
    adata,
    patient_label=donor,
    losses=["nb"],
    loss_coefs={
        'kl': kl,
        'integ': 0,
    },
    classification=[condition],
    **model_params,
)

state_dict = torch.load(input_files[1])['state_dict']
for key in list(state_dict.keys()):
    state_dict[key.replace('module.', '')] = state_dict.pop(key)

mil.module.load_state_dict(state_dict)
    
# mil.get_model_output(batch_size=batch_size)

# if subset_umap is not None:
#     # change to adata?
#     idx = random.sample(list(adata.obs_names), subset_umap)
#     adata = adata[idx].copy()

# sc.pp.neighbors(adata, use_rep="latent")
# sc.tl.umap(adata)
# sc.pl.umap(
#     adata,
#     color=umap_colors+["cell_attn"],
#     ncols=1,
#     show=False,
# )
# print(f'saving train umap as {output_files[i]}...')
# plt.savefig(output_files[i], bbox_inches="tight")
# plt.close()

# mil.plot_losses(save=output_files[n_splits+i])

# query
idx = query.obs[donor].sort_values().index
query = query[idx].copy()

# mtm.model.MultiVAE_MIL.setup_query(
#     adata,
#     query,
#     **setup_params,
# )

new_model = mtm.model.MultiVAE_MIL.load_query_data(query, use_prediction_labels=False, reference_model=mil)

# before finetuning
# new_model.is_trained_ = True
# new_model.get_model_output(batch_size=batch_size)

# report = classification_report(
#         query.obs[condition], query.obs[f"predicted_{condition}"], output_dict=True
#     )
# df = pd.DataFrame(report).T
# print('saving classification report without fine-tuning...')
# df.to_csv(output_files[i])

path_to_checkpoints = f'data/multigrate/{snakemake.params.paramspace_pattern}/{i}/query_checkpoints/{checkpoint}/'
dirpath=Path(path_to_checkpoints)
if dirpath.exists():
    shutil.rmtree(dirpath)

os.makedirs(path_to_checkpoints, exist_ok=True)

new_model.finetune_query(
    lr=lr, batch_size=batch_size, save_loss=output_files[0], path_to_checkpoints=path_to_checkpoints, **train_params
    #mil.train(lr=lr, batch_size=batch_size, path_to_checkpoints=path_to_checkpoints, **train_params)
)

    
    # new_model.get_model_output(batch_size=batch_size)




# print('=================')
# print('here we go again:')
# print(model_params)
# #print(type(condition))
# #print(condition[0])
# print(condition)
# print(donor)
# print(n_splits)
# for i in range(n_splits):
#     adata.obs[[condition]].to_csv(output_files[i])
