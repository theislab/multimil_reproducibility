import warnings
warnings.simplefilter(action='ignore')

import argparse
import numpy as np
import pandas as pd
import time
import os
import shutil
import os.path
from os import path
import json
from itertools import cycle
import scanpy as sc
import scib

if __name__ == "__main__":
    rootdir = '/lustre/groups/ml01/workspace/anastasia.litinetskaya/experiments/integration/10x/'
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.h5ad'):
                adata = sc.read(os.path.join(subdir, file))
                name = subdir.split('/')[-1].split('-')[0]
                batch_key = "batch"
                label_key = "cell_type"
                if int(name) >= 14:
                    sc.pp.neighbors(adata, use_rep='latent')
                    # need to trick scIB a bit to calculate ASW cell type but not ASW batch, 
                    # so set batch_key to be sth random, will ignore later
                    adata.obs['batch'] = '0'
                    adata.obs['batch'][:3000] = '1'
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
                        verbose=True,
                    )
                    
                    print(name)
                    print(metrics)

                    mean_integ_metrics = np.mean([metrics[0][i] for i in ['graph_conn', 'ASW_label/batch']])
                    mean_bio_metrics = np.mean([metrics[0][i] for i in ['ASW_label', 'NMI_cluster/label', 'ARI_cluster/label', 'isolated_label_silhouette']])
                    
                    overall_score = 0.4*mean_integ_metrics + 0.6*mean_bio_metrics
                
                    metrics.loc['overall', 0] = overall_score
                    
                    metrics.to_csv(subdir + '/metrics.csv')