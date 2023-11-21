import time
start_time = time.time()

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
import scvi
from sklearn.metrics import classification_report
from utils import get_existing_checkpoints

import warnings
warnings.filterwarnings('ignore')


print('--- %s seconds ---' % (time.time()-start_time))

def run_multigrate(adata1, adata2, sample_key, condition_key, n_splits, params, hash, task, **kwargs):

    print('============ Multigrate training ============')

    torch.set_float32_matmul_precision('medium')

    donor = sample_key
    condition = condition_key
    n_splits = n_splits

    setup_params = {
        "rna_indices_end": params['rna_indices_end'],
        "categorical_covariate_keys": params['categorical_covariate_keys'].strip('][').replace('\'', '').replace('\"', '').split(', '),
    }
    model_params = {
        "z_dim": params['z_dim'],
        "attn_dim": params['attn_dim'],
        "class_loss_coef": params['class_loss_coef'],
        "cond_dim": params['cond_dim'],
    }
    subset_umap = params['subset_umap']
    umap_colors = params['umap_colors'].strip('][').replace('\'', '').replace('\"', '').split(', ')
    train_params = {
        "max_epochs": params['train_max_epochs'],
        "save_checkpoint_every_n_epochs": params['train_save_checkpoint_every_n_epochs'],
    }
    query_train_params = {
        "max_epochs": params['query_max_epochs'],
        "save_checkpoint_every_n_epochs": params['query_save_checkpoint_every_n_epochs'],
    }
    # layers = params['layers']
    # train params
    lr = params['lr']
    batch_size = params['batch_size']
    kl = params['kl']
    seed = params['seed']

    scvi.settings.seed = seed

    dfs = []

    for i in range(n_splits):
        print(f'Split {i}...')

        ########################
        ######## TRAIN #########
        ########################

        if adata2 is None:
            rna = adata1
            losses = ['nb']
            print('Organizing multiome anndatas...')
            adata = mtg.data.organize_multiome_anndatas(
                adatas = [[rna]],
                #layers = layers,
                )
            
            del rna
        else:
            rna = adata1
            losses = ['nb', 'mse']
            print('Organizing multiome anndatas...')
            adata = mtg.data.organize_multiome_anndatas(
                adatas = [[rna], [adata2]],
                #layers = layers,
                )
            
            del rna

        query = adata[adata.obs[f"split{i}"] == "val"].copy()
        adata = adata[adata.obs[f"split{i}"] == "train"].copy()

        idx = adata.obs[donor].sort_values().index
        adata = adata[idx].copy()

        print('Setting up anndata...')
        mtm.model.MultiVAE_MIL.setup_anndata(
            adata, 
            **setup_params
        )

        print('Initializing the model...')
        mil = mtm.model.MultiVAE_MIL(
            adata,
            patient_label=donor,
            losses=losses,
            loss_coefs={
                'kl': kl,
            },
            classification=[condition],
            **model_params,
        )

        path_to_train_checkpoints = f'data/multigrate/{task}/{hash}/{i}/checkpoints/'
        dirpath=Path(path_to_train_checkpoints)
        if dirpath.exists():
            shutil.rmtree(dirpath)

        os.makedirs(path_to_train_checkpoints, exist_ok=True)
        print('Starting training...')
        mil.train(
            lr=lr,
            batch_size=batch_size, 
            path_to_checkpoints=path_to_train_checkpoints,
            **train_params
        )

        mil.is_trained_ = True

        print('Starting inference...')
        mil.get_model_output(batch_size=batch_size)

        mil.save(f'data/multigrate/{task}/{hash}/{i}/model/', overwrite=True)

        if subset_umap is not None:
            print(f'Subsetting to {subset_umap}...')
            idx = random.sample(list(adata.obs_names), subset_umap)
            adata = adata[idx].copy()

        print('Calculating neighbors...')
        sc.pp.neighbors(adata, use_rep="latent")
        sc.tl.umap(adata)
        sc.pl.umap(
            adata,
            color=umap_colors+["cell_attn"],
            ncols=1,
            show=False,
        )

        plt.savefig(f'data/multigrate/{task}/{hash}/{i}/train_umap.png', bbox_inches="tight")
        plt.close()

        #print('Saving train losses...')
        mil.plot_losses(save=f'data/multigrate/{task}/{hash}/{i}/train_losses.png')

        checkpoints  = get_existing_checkpoints(path_to_train_checkpoints)

        print(f"Found {len(checkpoints)} in {path_to_train_checkpoints}.")

        for ckpt in checkpoints:

            #########################################
            ###### PREDICT WITHOUT FINETUNING #######
            #########################################

            train_state_dict = torch.load(path_to_train_checkpoints + f'{ckpt}.ckpt')['state_dict']
            for key in list(train_state_dict.keys()):
                train_state_dict[key.replace('module.', '')] = train_state_dict.pop(key)

            mil.module.load_state_dict(train_state_dict)

            idx = query.obs[donor].sort_values().index
            query = query[idx].copy()

            new_model = mtm.model.MultiVAE_MIL.load_query_data(query, use_prediction_labels=False, reference_model=mil)

            new_model.is_trained_ = True
            new_model.get_model_output(query, batch_size=batch_size)

            report = classification_report(
                query.obs[condition], query.obs[f"predicted_{condition}"], output_dict=True
            )
            df = pd.DataFrame(report).T
            
            df.to_csv(path_to_train_checkpoints + f'{ckpt}.csv')

            df['split'] = i
            df['method'] = 'multigrate'
            df['epoch'] = ckpt.split('-')[0].split('=')[-1]
            df['query_epoch'] = 0

            dfs.append(df)

            #######################
            ###### FINETUNE #######
            #######################

            path_to_query_checkpoints = f'data/multigrate/{task}/{hash}/{i}/query_checkpoints/{ckpt}/'
            dirpath=Path(path_to_query_checkpoints)
            if dirpath.exists():
                shutil.rmtree(dirpath)

            os.makedirs(path_to_query_checkpoints, exist_ok=True)

            new_model.finetune_query(
                lr=lr, 
                batch_size=batch_size, 
                save_loss=path_to_query_checkpoints+'finetune_losses.png', 
                path_to_checkpoints=path_to_query_checkpoints, 
                weight_decay=0,
                n_epochs_kl_warmup=0,
                **query_train_params
            )

            #######################################
            ###### PREDICT AFTER FINETUNING #######
            #######################################

            query_checkpoints  = get_existing_checkpoints(path_to_query_checkpoints)

            print(f"Found {len(query_checkpoints)} in {path_to_query_checkpoints}.")

            for query_ckpt in query_checkpoints:
                
                query_state_dict = torch.load(path_to_query_checkpoints + f'{query_ckpt}.ckpt')['state_dict']
                for key in list(query_state_dict.keys()):
                    query_state_dict[key.replace('module.', '')] = query_state_dict.pop(key)
                    key = key.replace('module.', '')
                    query_state_dict[f'vae.{key}'] = query_state_dict.pop(key)

                train_state_dict.update(query_state_dict)

                new_model.module.load_state_dict(train_state_dict)
                new_model.is_trained_ = True
                new_model.get_model_output(query, batch_size=batch_size)

                report = classification_report(
                    query.obs[condition], query.obs[f"predicted_{condition}"], output_dict=True
                )
                df = pd.DataFrame(report).T

                df.to_csv(path_to_query_checkpoints + f'{query_ckpt}.csv')

                new_model.get_model_output(adata, batch_size=batch_size)

                adata.obs['reference'] = 'reference'
                query.obs['reference'] = 'query'
                adata_both = ad.concat([adata, query])

                if subset_umap is not None:
                    print(f'Subsetting to {subset_umap}...')
                    idx = random.sample(list(adata_both.obs_names), subset_umap)
                    adata_both = adata_both[idx].copy()

                sc.pp.neighbors(adata_both, use_rep='latent')
                sc.tl.umap(adata_both)

                sc.pl.umap(
                    adata_both,
                    color=umap_colors+["cell_attn", "reference"],
                    ncols=1,
                    show=False,
                )
                plt.savefig(path_to_query_checkpoints + f'{query_ckpt}_umap.png', bbox_inches="tight")
                plt.close()

                df['split'] = i
                df['method'] = 'multigrate'
                df['epoch'] = ckpt.split('-')[0].split('=')[-1]
                df['query_epoch'] = query_ckpt.split('-')[0].split('=')[-1]
                dfs.append(df)

    df = pd.concat(dfs)
    return df
