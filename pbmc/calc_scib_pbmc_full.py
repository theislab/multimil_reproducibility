import scanpy as sc
import scib
import numpy as np

for adata_path, method_name in zip(
    [
        '/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline/data/pp/totalvi_pbmc.h5ad',
        '/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline/data/pp/mtg_pretrain_pbmc_full/pbmc_full_mtg1.h5ad',
    ],
    [
        'totalvi',
        'multimil',
    ],
):

    adata = sc.read(adata_path)

    batch_key = "Site"
    label_key = "initial_clustering"

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

    mean_integ_metrics = np.mean([metrics[0][i] for i in ['graph_conn', 'ASW_label/batch']])
    mean_bio_metrics = np.mean([metrics[0][i] for i in ['ASW_label', 'NMI_cluster/label', 'ARI_cluster/label', 'isolated_label_silhouette']])

    overall_score = 0.4*mean_integ_metrics + 0.6*mean_bio_metrics

    metrics.loc['batch', 0] = mean_integ_metrics
    metrics.loc['bio', 0] = mean_bio_metrics
    metrics.loc['overall', 0] = overall_score

    metrics.to_csv(f'/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline/data/metrics/scib_full_pbmc/metrics_{method_name}.csv')