import warnings
warnings.simplefilter(action='ignore')

import argparse
import numpy as np
import pandas as pd
import anndata as ad
import time
import os
import shutil
import os.path
from os import path
import json
from itertools import cycle
import scanpy as sc
import torch
from matplotlib import pyplot as plt
import multigrate as mtg
import random
import scib

def train(seed=None, split=None, run=None, config_name=None, **config):
    # load experiment configurations

    experiment_config = config["experiment"]
    train_params = config["model"]["train"]
    save_anndatas = experiment_config["save_anndatas"]
    map_query = experiment_config["query"]
    if map_query:
        query_train_params = config["model"]["query_train"]
    calc_train_metrics = experiment_config["metrics_train"]
    calc_query_reference_metrics = experiment_config["metrics_query_reference"]
    calc_separate_queries_metrics = experiment_config["metrics_separate_queries"]
    random_seed = experiment_config.get(
        "seed", seed
    )  # if specified in cofig use that one
    if random_seed is None:
        raise RuntimeError("seed is None")
    umap_colors = experiment_config["umap_colors"]

    from scvi._settings import settings
    settings.seed = random_seed
    config["experiment"]["seed"] = random_seed

    organize_params = config["model"]["organize_params"]
    setup_params = config["model"]["setup_params"]
    model_params = config["model"]["model_params"]
    
    # create huge name
    name = ""
    if config_name is not None:
        name += config_name.split('/')[-1].split('.')[0]

    if model_params.get('mmd', None) is not None:
       mmd = model_params.get('mmd')
       name += f'-{mmd}'
    
    # add split counter
    if split is not None:
        name = name + "-split" + str(split)
    if run is not None:
        name = name + "-" + str(run)
    for loss, value in model_params['loss_coefs'].items():
        name += f'-{loss}-{value}'
    
    output_dir = experiment_config["output_dir"] + name + "/"
    # configure output directory to save logs and results
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    json.dump(config, open(os.path.join(output_dir, "config.json"), "w"), indent=2)

    adatas = []
    for adata_set in organize_params["adatas"]:
        adatas.append([])
        for adata_path in adata_set:
            if adata_path is not None:
                adata = sc.read_h5ad(adata_path)
                adatas[-1].append(adata)
            else:
                adatas[-1].append(None)
    organize_params["adatas"] = adatas

    adata = mtg.data.organize_multiome_anndatas(**organize_params)

    # if map_query:
    #     if split is None:
    #         query = adata[adata.obs["split"] == "test"].copy()
    #         adata = adata[adata.obs["split"] == "train"].copy()
    #     else:
    #         split_name = "split" + str(split)
    #         query = adata[adata.obs[split_name] == "test"].copy()
    #         adata = adata[adata.obs[split_name] == "train"].copy()

    scrna_query = ['site4_donor1_cite']
    cite_query = ['site4_donor8_cite', 'site4_donor9_cite']
    snrna_query = ['site4_donor1_multiome']
    multiome_query = ['site4_donor8_multiome']
    atac_query = ['site4_donor9_multiome']

    idx_atac_query = adata.obs['Samplename'].isin(atac_query)
    idx_snrna_query = adata.obs['Samplename'].isin(snrna_query)
    idx_scrna_query = adata.obs['Samplename'].isin(scrna_query)
    idx_multiome_query = adata.obs['Samplename'].isin(multiome_query)
    idx_cite_query = adata.obs['Samplename'].isin(cite_query)

    adata[idx_atac_query, :4000].X = 0
    adata[idx_scrna_query, 4000:].X = 0
    adata[idx_snrna_query, 4000:].X = 0

    adata.obs['type_of_query'] = 'reference'
    adata.obs.loc[idx_cite_query, 'type_of_query'] = 'CITE-seq query'
    adata.obs.loc[idx_multiome_query, 'type_of_query'] = 'multiome query'
    adata.obs.loc[idx_atac_query, 'type_of_query'] = 'ATAC-seq query'
    adata.obs.loc[idx_scrna_query, 'type_of_query'] = 'scRNA-seq query'
    adata.obs.loc[idx_snrna_query, 'type_of_query'] = 'snRNA-seq query'

    query = adata[adata.obs['type_of_query'] != 'reference'].copy()
    adata = adata[adata.obs['type_of_query'] == 'reference'].copy()

    adata.obs['reference'] = 'reference'
    query.obs['reference'] = 'query'

    del adatas
    del organize_params

    mtg.model.MultiVAE.setup_anndata(
        adata, 
        **setup_params
    )

    model = mtg.model.MultiVAE(
        adata=adata, 
        **model_params
    )

    # weight_decay = 0 if not end2end else 1e-3
    model.train(
        **train_params
    )

    model.get_latent_representation()
    adata.obsm['latent_ref'] = adata.obsm['latent'].copy()

    sc.pp.neighbors(adata, use_rep='latent')
    sc.tl.umap(adata)
    sc.pl.umap(
            adata,
            color=umap_colors,
            ncols=1,
            frameon=False,
            show=False,
        )
    plt.savefig(output_dir + "umap_latent.png", bbox_inches="tight")
    plt.close()

    model.plot_losses(save=output_dir+'losses.png')
    model.save(output_dir + "model/", overwrite=True)

    if save_anndatas:
        tmp = sc.AnnData(adata.obsm["latent"])
        tmp.obs = adata.obs
        tmp.obsm = adata.obsm
        tmp.uns = adata.uns
        del tmp.uns['modality_lengths']
        tmp.write(output_dir + "adata.h5ad")

    if calc_train_metrics:
        print('Calculating reference scIB metrics...')
        batch_key = experiment_config.get('batch_key', None)
        label_key = experiment_config.get('label_key', None)
        metrics = scib.metrics.metrics(
            adata, 
            adata, 
            batch_key=batch_key, 
            label_key=label_key, 
            embed='latent',
            ari_=True,
            nmi_=True,
            silhouette_=True,
            graph_conn_=True,
            isolated_labels_asw_=True,
        )
        mean_integ_metrics = np.mean([metrics[0][i] for i in ['graph_conn', 'ASW_label/batch']])
        mean_bio_metrics = np.mean([metrics[0][i] for i in ['ASW_label', 'NMI_cluster/label', 'ARI_cluster/label', 'isolated_label_silhouette']])
        
        overall_score = 0.4*mean_integ_metrics + 0.6*mean_bio_metrics
    
        metrics.loc['overall', 0] = overall_score
        metrics.to_csv(output_dir + 'train_metrics.csv')

    if map_query:

        mtg.model.MultiVAE.setup_anndata(
            query,
            **setup_params
        )

        q_model = mtg.model.MultiVAE.load_query_data(query, model)

        q_model.train(**query_train_params, weight_decay=0)

        q_model.get_latent_representation(adata=query)
        q_model.get_latent_representation(adata=adata)

        q_model.plot_losses(save=output_dir+'losses_finetune.png')

        adata_both = ad.concat([adata, query])

        del adata

        sc.pp.neighbors(adata_both, use_rep='latent')
        sc.tl.umap(adata_both)

        if save_anndatas:
            tmp = sc.AnnData(query.obsm["latent"])
            tmp.obs = query.obs
            tmp.obsm = query.obsm
            tmp.uns = query.uns
            del tmp.uns['modality_lengths']
            tmp.write(output_dir + "query.h5ad")

            tmp = sc.AnnData(adata_both.obsm["latent"])
            tmp.obs = adata_both.obs
            tmp.obsm = adata_both.obsm
            tmp.uns = adata_both.uns
            # del tmp.uns['modality_lengths']
            tmp.write(output_dir + "adata_both.h5ad")

        del query

        sc.pl.umap(
            adata_both,
            color=umap_colors + ['reference'], 
            ncols=1,
            frameon=False,
            show=False,
        )
        plt.savefig(output_dir + "umap_latent_ref_query_after.png", bbox_inches="tight")
        plt.close()

        for group in ['reference', 'CITE-seq query', 'ATAC-seq query', 'scRNA-seq query', 'snRNA-seq query', 'multiome query']:
            sc.pl.umap(
                adata_both,
                color='type_of_query', 
                ncols=1,
                frameon=False,
                palette=sc.pl.palettes.default_20,
                groups=[group],
                show=False,
            )
            group = group.replace(' ', '_')
            plt.savefig(output_dir + f"umap_query_{group}.png", bbox_inches="tight")
            plt.close()

        q_model.save(output_dir + "model_updated/", overwrite=True)

        if calc_query_reference_metrics:
            print('Calculating reference/query scIB metrics...')
            batch_key = "reference"
            label_key = experiment_config.get('label_key', None)
            metrics = scib.metrics.metrics(
                adata_both, 
                adata_both, 
                batch_key=batch_key, 
                label_key=label_key, 
                embed='latent',
                ari_=True,
                nmi_=True,
                silhouette_=True,
                graph_conn_=True,
                isolated_labels_asw_=True,
            )
            mean_integ_metrics = np.mean([metrics[0][i] for i in ['graph_conn', 'ASW_label/batch']])
            mean_bio_metrics = np.mean([metrics[0][i] for i in ['ASW_label', 'NMI_cluster/label', 'ARI_cluster/label', 'isolated_label_silhouette']])
            
            overall_score = 0.4*mean_integ_metrics + 0.6*mean_bio_metrics
        
            metrics.loc['overall', 0] = overall_score
            metrics.to_csv(output_dir + 'reference_query_metrics.csv')

        if calc_separate_queries_metrics:
            print('Calculating separate query scIB metrics...')
            for query_type in ['CITE-seq query', 'ATAC-seq query', 'scRNA-seq query', 'snRNA-seq query', 'multiome query']:
                adata_tmp = adata_both[adata_both.obs['type_of_query'].isin(['reference', query_type])].copy()
                sc.pp.neighbors(adata_tmp, use_rep='latent')
                batch_key = "type_of_query"
                label_key = experiment_config.get('label_key', None)
                metrics = scib.metrics.metrics(
                    adata_tmp, 
                    adata_tmp, 
                    batch_key=batch_key, 
                    label_key=label_key, 
                    embed='latent',
                    ari_=True,
                    nmi_=True,
                    silhouette_=True,
                    graph_conn_=True,
                    isolated_labels_asw_=True,
                )
                mean_integ_metrics = np.mean([metrics[0][i] for i in ['graph_conn', 'ASW_label/batch']])
                mean_bio_metrics = np.mean([metrics[0][i] for i in ['ASW_label', 'NMI_cluster/label', 'ARI_cluster/label', 'isolated_label_silhouette']])
                
                overall_score = 0.4*mean_integ_metrics + 0.6*mean_bio_metrics
            
                metrics.loc['overall', 0] = overall_score
                query_type = query_type.replace(' ', '_')
                metrics.to_csv(output_dir + f'reference_{query_type}_metrics.csv')

def parse_args():
    parser = argparse.ArgumentParser(description="Perform model training.")
    parser.add_argument("--config-file", type=str)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--runs", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    split = args.split
    runs = int(args.runs)

    print(args.config_file)

    if runs is not None:
        for i in range(runs):

            with open(args.config_file) as json_file:
                config = json.load(json_file)

            if i == 0:
                seed = 0
            else:
                seed = random.randint(0, 9999)

            train(seed=seed, split=split, run=i, config_name=args.config_file, **config)
    else:
        with open(args.config_file) as json_file:
            config = json.load(json_file)
        seed = 0
        train(seed=seed, split=split, **config)