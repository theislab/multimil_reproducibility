import scanpy as sc
import pandas as pd
import multigrate as mtg
import multimil as mtm
from pathlib import Path
import shutil

from matplotlib import pyplot as plt
import scanpy as sc
import pandas as pd
import anndata as ad
import os
import random
import torch
from sklearn.metrics import classification_report

def get_existing_checkpoints(rootdir):

    checkpoints = []

    for root, _, files in os.walk(rootdir):
        for filename in files:
            if filename.endswith('.ckpt'):
                checkpoints.append(filename.strip('.ckpt'))

    return checkpoints




print('============ Multigrate training ============')

torch.set_float32_matmul_precision('medium')

input_files = snakemake.input
model_params = snakemake.params['model_params']
donor = snakemake.params['donor'],
n_splits = snakemake.params['n_splits']
subset_umap = snakemake.params['subset_umap']
organize_params = snakemake.params['organize_params']
setup_params = snakemake.params['setup_params']
umap_colors = snakemake.params['umap_colors']

# train params
lr = model_params.pop('lr')
batch_size = 256
kl = model_params.pop('kl')
integ = model_params.pop('integ')

train_params = snakemake.params['train_params']
query_train_params = snakemake.params['query_train_params']

output_files = snakemake.output

donor = donor[0]

for i in range(n_splits):
    print(f'Split {i}...')

    ########################
    ######## TRAIN #########
    ########################

    rna_multiome = sc.read(input_files[0])
    rna_cite = sc.read(input_files[1])
    atac = sc.read(input_files[2])
    adt = sc.read(input_files[3])

    print('Organizing multiome anndatas...')
    adata = mtg.data.organize_multiome_anndatas(
        adatas = [[rna_cite, rna_multiome], [None, atac], [adt, None]],
        **organize_params
        )
    
    del rna_cite, rna_multiome, atac, adt

    #query = adata[adata.obs[f"split{i}"] == "val"].copy()
    #adata = adata[adata.obs[f"split{i}"] == "train"].copy()

    #idx = adata.obs[donor].sort_values().index
    #adata = adata[idx].copy()

    print('Setting up anndata...')
    mtg.model.MultiVAE.setup_anndata(
        adata, 
        **setup_params
    )

    print('Initializing the model...')
    model = mtg.model.MultiVAE(
        adata,
        losses=['nb', 'mse', 'mse'],
        loss_coefs={
            'kl': kl,
            'integ': integ,
        },
        integrate_on='Modality',
        mmd='marginal',     # change MMD here
        **model_params,
    )

    path_to_train_checkpoints = f'data/multigrate/{snakemake.params.paramspace_pattern}/{i}/checkpoints/'
    dirpath=Path(path_to_train_checkpoints)
    if dirpath.exists():
        shutil.rmtree(dirpath)

    os.makedirs(path_to_train_checkpoints, exist_ok=True)
    print('Starting training...')
    model.train(
        lr=lr,
        batch_size=batch_size, 
        path_to_checkpoints=path_to_train_checkpoints, 
        **train_params
    )

#    model.is_trained_ = True

    print('Starting inference...')
    model.get_latent_representation(batch_size=batch_size)

    model.save(f'data/multigrate/{snakemake.params.paramspace_pattern}/{i}/model/', overwrite=True)

    if subset_umap > 0:
        print(f'Subsetting to {subset_umap}...')
        # change to adata?
        idx = random.sample(list(adata.obs_names), subset_umap)
        adata = adata[idx].copy()

    print('Calculating neighbors...')
    sc.pp.neighbors(adata, use_rep="latent")
    sc.tl.umap(adata)
    sc.pl.umap(
        adata,
        color=umap_colors,
        ncols=1,
        frameon=False,
        show=False,
    )
    print(f'Saving train umap as {output_files[i]}...')
    plt.savefig(output_files[i], bbox_inches="tight")
    plt.close()

    print(f'Saving train losses as {output_files[i+n_splits]}...')
    model.plot_losses(save=output_files[i+n_splits])

    ####################
    ###### QUERY #######
    ####################

    rna_multiome_query = sc.read(input_files[4])
    rna_cite_query = sc.read(input_files[5])
    atac_query = sc.read(input_files[6])
    adt_query = sc.read(input_files[7])

    query = mtg.data.organize_multiome_anndatas(
        adatas = [[rna_cite_query, rna_multiome_query], [None, atac_query], [adt_query, None]],
        layers = [['counts', 'counts'], [None, 'log-norm'], [None, None]],
    )

    mtg.model.MultiVAE.setup_anndata(
        query,
        categorical_covariate_keys=['Modality', 'Samplename'],
        rna_indices_end=4000,
    )

    idx_atac_query = query.obs['Samplename'] == 'site4_donor9_multiome'
    idx_scrna_query = query.obs['Samplename'] == 'site4_donor8_cite'
    idx_snrna_query = query.obs['Samplename'] == 'site4_donor8_multiome'

    idx_mutiome_query = query.obs['Samplename'] == 'site4_donor1_multiome'
    idx_cite_query = query.obs['Samplename'] == 'site4_donor1_cite'

    query[idx_atac_query, :4000].X = 0
    query[idx_scrna_query, 4000:].X = 0
    query[idx_snrna_query, 4000:].X = 0

    q_model = mtg.model.MultiVAE.load_query_data(query, model)

    path_to_query_checkpoints = f'data/multigrate/{snakemake.params.paramspace_pattern}/{i}/query_checkpoints/'

    q_model.train(
        lr=lr,
        batch_size=batch_size, 
        path_to_checkpoints=path_to_query_checkpoints, 
        weight_decay=0, 
        **query_train_params
    )

    q_model.save(f'data/multigrate/{snakemake.params.paramspace_pattern}/{i}/query_model/', overwrite=True)

    q_model.get_latent_representation(adata=query)
    q_model.get_latent_representation(adata=adata)

    adata.obs['reference'] = 'reference'
    query.obs['reference'] = 'query'

    adata.obs['type_of_query'] = 'reference'
    query.obs.loc[idx_atac_query, 'type_of_query'] = 'ATAC query'
    query.obs.loc[idx_scrna_query, 'type_of_query'] = 'scRNA query'
    query.obs.loc[idx_snrna_query, 'type_of_query'] = 'snRNA query'
    query.obs.loc[idx_mutiome_query, 'type_of_query'] = 'multiome query'
    query.obs.loc[idx_cite_query, 'type_of_query'] = 'CITE-seq query'

    adata_both = ad.concat([adata, query])

    sc.pp.neighbors(adata_both, use_rep='latent')
    sc.tl.umap(adata_both)

    sc.pl.umap(
        adata_both, 
        color=[
            'l1_cell_type',
            'l2_cell_type',
            'reference',
            'Modality',
            'Samplename'
        ], 
        ncols=1, 
        frameon=False,
        show=False,
    )

    print(f'Saving train umap as {output_files[i+2*n_splits]}...')
    plt.savefig(output_files[i+2*n_splits], bbox_inches="tight")
    plt.close()

    adata_both.write(f'data/multigrate/{snakemake.params.paramspace_pattern}/{i}/adata_both.h5ad')

    for qt in ['CITE-seq query', 'multiome query', 'scRNA query', 'snRNA query', 'ATAC query']:
        sc.pl.umap(
            adata_both,
            color='type_of_query',
            ncols=1,
            frameon=False,
            groups=[qt],
            show=False,
        )
        plt.savefig(f"{output_files[i+2*n_splits][:-4]}_{qt.replace(' ', '_')}.png", bbox_inches="tight")
        plt.close()


