import os
import pandas as pd
import argparse
import scanpy as sc
import scib


def calc_metrics(file_path, batch_key, label_key):
    adata = sc.read(file_path)

    embed_key = 'embed'
    adata.obsm[embed_key] = adata.X.copy()

    metrics = scib.metrics.metrics(
        adata, 
        adata, 
        batch_key=batch_key, # Site
        label_key=label_key, # Initial clustering
        embed=embed_key,
        ari_=True,
        nmi_=True,
        silhouette_=True,
        graph_conn_=True,
        isolated_labels_asw_=True,
    )
    return metrics

def process_h5ad_files(file_path, batch_key, label_key):

    output_path = os.path.join(file_path, file_path.replace(".h5ad", ".csv"))
    
    result = calc_metrics(file_path, batch_key, label_key)
    
    result.to_csv(output_path)
    print(f"Processed {file_path} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .h5ad and save metrics as .csv.")
    parser.add_argument("file_path", type=str, help="Path to the .h5ad file.")
    parser.add_argument("batch_key", type=str, help="Batch key for calc_metrics function.")
    parser.add_argument("label_key", type=str, help="Label key for calc_metrics function.")
    args = parser.parse_args()
    
    process_h5ad_files(args.file_path, args.batch_key, args.label_key)
