import scanpy as sc
import pandas as pd
import numpy as np
import multigrate as mtg
import multimil as mtm
from pathlib import Path
import shutil
import ast

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


input_file = snakemake.input.tsv
output_file = snakemake.output.txt
config = snakemake.params['config']

df = pd.read_csv(input_file, sep='\t', index_col=0)

for task in config['TASKS']:
    df_task = df[df['task'] == task]

    input1 = config['TASKS'][task]['input1']
    input2 = config['TASKS'][task]['input2']

    condition_key = config['TASKS'][task]['condition_key']
    sample_key = config['TASKS'][task]['sample_key']
    n_splits = config['TASKS'][task]['n_splits']

    donor = sample_key
    condition = condition_key
    n_splits = n_splits


    for _, row in df_task.iterrows():
        h = row['hash']
        train_epoch = row['best_epoch']
        query_epoch = row['best_query_epoch']
        params = ast.literal_eval(row['best_params'])

        if row['Unnamed: 1'] == 'multigrate':
            continue
            # print(f'Multigrate with hash = {h}...')

            # adata1 = sc.read_h5ad(input1)
            # adata2 = None
            # if input2 is not None:
            #     adata2 = sc.read_h5ad(input2)

            # setup_params = {
            #     "rna_indices_end": params['rna_indices_end'],
            #     "categorical_covariate_keys": params['categorical_covariate_keys'].strip('][').replace('\'', '').replace('\"', '').split(', '),
            # }
            # model_params = {
            #     "z_dim": params['z_dim'],
            #     "attn_dim": params['attn_dim'],
            #     "class_loss_coef": params['class_loss_coef'],
            #     "cond_dim": params['cond_dim'],
            # }

            # lr = params['lr']
            # batch_size = params['batch_size']
            # kl = params['kl']
            # seed = params['seed']

            # scvi.settings.seed = seed

            # for i in range(n_splits):
            #     print(f'Split {i}...')


            #     if adata2 is None:
            #         rna = adata1

            #         print('Organizing multiome anndatas...')
            #         adata = mtg.data.organize_multiome_anndatas(
            #             adatas = [[rna]],
            #             )
                    
            #         del rna
            #     else:
            #         rna = adata1
            #         print('Organizing multiome anndatas...')
            #         adata = mtg.data.organize_multiome_anndatas(
            #             adatas = [[rna, adata2]],
            #             )
                    
            #         del rna

            #     query = adata[adata.obs[f"split{i}"] == "val"].copy()
            #     adata = adata[adata.obs[f"split{i}"] == "train"].copy()

            #     idx = adata.obs[donor].sort_values().index
            #     adata = adata[idx].copy()

            #     print('Setting up anndata...')
            #     mtm.model.MultiVAE_MIL.setup_anndata(
            #         adata, 
            #         **setup_params
            #     )

            #     print('Initializing the model...')
            #     mil = mtm.model.MultiVAE_MIL(
            #         adata,
            #         patient_label=donor,
            #         losses=[
            #             "nb",
            #         ],
            #         loss_coefs={
            #             'kl': kl,
            #         },
            #         classification=[condition],
            #         **model_params,
            #     )

            #     path_to_train_checkpoints = f'data/multigrate/{task}/{h}/{i}/checkpoints/'
            #     train_checkpoints = get_existing_checkpoints(path_to_train_checkpoints)
            #     best_ckpt = None
            #     for ckpt in train_checkpoints:
            #         print('--------')
            #         print(str(train_epoch))
            #         print(h)
            #         if str(int(train_epoch)) in ckpt:
            #             best_ckpt = ckpt
            #             break

            #     train_state_dict = torch.load(path_to_train_checkpoints + f'{best_ckpt}.ckpt')['state_dict']
            #     for key in list(train_state_dict.keys()):
            #         train_state_dict[key.replace('module.', '')] = train_state_dict.pop(key)

            #     mil.module.load_state_dict(train_state_dict)

            #     mil.is_trained_ = True
            #     mil.get_model_output(adata, batch_size=batch_size)

            #     idx = query.obs[donor].sort_values().index
            #     query = query[idx].copy()

            #     new_model = mtm.model.MultiVAE_MIL.load_query_data(query, use_prediction_labels=False, reference_model=mil)

            #     path_to_query_checkpoints = f'data/multigrate/{task}/{h}/{i}/query_checkpoints/{best_ckpt}/'
            #     query_checkpoints = get_existing_checkpoints(path_to_query_checkpoints)
            #     best_q_ckpt = None
            #     for ckpt in query_checkpoints:
            #         if str(int(query_epoch)) in ckpt:
            #             best_q_ckpt = ckpt
            #             break

            #     query_state_dict = torch.load(path_to_query_checkpoints + f'{best_q_ckpt}.ckpt')['state_dict']
            #     for key in list(query_state_dict.keys()):
            #         query_state_dict[key.replace('module.', '')] = query_state_dict.pop(key)
            #         key = key.replace('module.', '')
            #         query_state_dict[f'vae.{key}'] = query_state_dict.pop(key)

            #     train_state_dict.update(query_state_dict)

            #     new_model.is_trained_ = True
            #     new_model.get_model_output(query, batch_size=batch_size)

            #     adata.obs['reference'] = 'reference'
            #     query.obs['reference'] = 'query'
            #     adata_both = ad.concat([adata, query])

            #     sc.pp.neighbors(adata_both, use_rep='latent')
            #     sc.tl.umap(adata_both)

            #     adata_both.write(f'data/multigrate/{task}/{h}_adata_both.h5ad')


        elif row['Unnamed: 1'] == 'multigrate_mil':
            # continue
            print(f'Multigrate mil with hash = {h}...')    
            
            adata1 = sc.read_h5ad(input1)

            setup_params = {
                "rna_indices_end": params['rna_indices_end'],
                "categorical_covariate_keys": params['categorical_covariate_keys'].strip('][').replace('\'', '').replace('\"', '').split(', '),
            }
            model_params = {
                "class_loss_coef": params['class_loss_coef'],
                "z_dim": adata1.X.shape[1],
            }

            lr = params['lr']
            batch_size = params['batch_size']
            seed = params['seed']

            scvi.settings.seed = seed

            for i in range(n_splits):
                print(f'Split {i}...')

                rna = adata1

                print('Organizing multiome anndatas...')
                adata = mtg.data.organize_multiome_anndatas(
                    adatas = [[rna]],
                        #layers = layers,
                )
                    
                del rna
                query = adata[adata.obs[f"split{i}"] == "val"].copy()
                adata = adata[adata.obs[f"split{i}"] == "train"].copy()

                idx = adata.obs[donor].sort_values().index
                adata = adata[idx].copy()

                print('Setting up anndata...')
                mtm.model.MILClassifier.setup_anndata(
                    adata, 
                    **setup_params
                )

                print('Initializing the model...')
                mil = mtm.model.MILClassifier(
                    adata, 
                    classification=[
                        condition
                    ],
                    patient_label=donor,
                    **model_params,
                )

                path_to_train_checkpoints = f'data/multigrate_mil/{task}/{h}/{i}/checkpoints/'
                train_checkpoints = get_existing_checkpoints(path_to_train_checkpoints)
                best_ckpt = None
                for ckpt in train_checkpoints:
                    if str(int(train_epoch)) in ckpt:
                        best_ckpt = ckpt
                        break

                train_state_dict = torch.load(path_to_train_checkpoints + f'{best_ckpt}.ckpt')['state_dict']
                for key in list(train_state_dict.keys()):
                    train_state_dict[key.replace('module.', '')] = train_state_dict.pop(key)

                mil.module.load_state_dict(train_state_dict)

                mil.is_trained_ = True
                mil.get_model_output(adata, batch_size=batch_size)

                idx = query.obs[donor].sort_values().index
                query = query[idx].copy()

                new_model = mtm.model.MILClassifier.load_query_data(query, reference_model=mil)

                new_model.is_trained_ = True
                new_model.get_model_output(query, batch_size=batch_size)

                adata.obs['reference'] = 'reference'
                query.obs['reference'] = 'query'
                adata_both = ad.concat([adata, query])

                sc.pp.neighbors(adata_both, use_rep='latent')
                sc.tl.umap(adata_both)

                # adata_both.write(f'data/multigrate_mil/{task}/{h}_adata_both.h5ad')

                adata1.obsm[f'latent_{i}'] = adata_both.obsm['latent']
                adata1.obs[f'cell_attn_{i}'] = adata_both.obs['cell_attn']
            
            adata1.write(f'data/multigrate_mil/{task}/{h}_adata_both.h5ad')

        else:
            continue

with open(output_file, 'w') as f:
    f.write('done!')