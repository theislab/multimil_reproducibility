
from helical.models.scgpt.model import scGPT, scGPTConfig
import anndata as ad
import scanpy as sc

scgpt_config = scGPTConfig(batch_size=10)

scgpt = scGPT(configurer = scgpt_config)

adata = ad.read_h5ad('/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline_mil/data/pp/pbmc_full_rna.h5ad')

# adata.X = adata.layers['counts'].copy()

data = scgpt.process_data(adata)
embeddings = scgpt.get_embeddings(data)

embed = sc.AnnData(embeddings, obs=adata.obs)

embed.write_h5ad("/lustre/groups/ml01/workspace/anastasia.litinetskaya/code/multimil_reproducibility/pbmc/data/scgpt_embeddings.h5ad")