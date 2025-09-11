
from helical import Geneformer, GeneformerConfig
import anndata as ad
import scanpy as sc

model_config = GeneformerConfig(model_name="gf-12L-95M-i4096", batch_size=10)

geneformer = Geneformer(configurer=model_config)

adata = ad.read_h5ad('/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline_mil/data/pp/pbmc_full_rna.h5ad')

# adata.X = adata.layers['counts'].copy()
adata.X = adata.X.astype(int)

# otherwise error in prosess_data
sc.pp.filter_cells(adata, min_genes=30)

data = geneformer.process_data(adata)
embeddings = geneformer.get_embeddings(data)
embed = sc.AnnData(embeddings, obs=adata.obs)

embed.write_h5ad("/lustre/groups/ml01/workspace/anastasia.litinetskaya/code/multimil_reproducibility/pbmc/data/geneformer_embeddings.h5ad")