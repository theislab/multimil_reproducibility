import scanpy as sc
import scvi
import muon
import mudata as md
import torch
import numpy as np
import matplotlib.pyplot as plt

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

sc.set_figure_params(figsize=(4, 4))
torch.set_float32_matmul_precision("high")

rna = sc.read('/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline/data/pp/pbmc_full_rna.h5ad')
adt = sc.read('/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline/data/pp/pbmc_full_adt.h5ad')

isotype_controls = [
    'AB_Mouse IgG1_K_Iso',
     'AB_Mouse_IgG2a_K_Iso',
     'AB_Mouse_IgG2b_K_Iso',
     'AB_Rat_IgG2b_K_Iso'
]

proteins = list(set(adt.var_names).difference(set(isotype_controls)))
adt = adt[: ,proteins].copy()

rna.obsm['protein_expression'] = adt.layers['counts'].A.copy()
rna.X = rna.layers['counts'].A.copy()

scvi.model.TOTALVI.setup_anndata(
    rna,
    batch_key="sample_id",
    protein_expression_obsm_key="protein_expression",
    categorical_covariate_keys=["Site"],
)

model = scvi.model.TOTALVI(rna)

model.train(max_epochs=30)

plt.plot(model.history["elbo_train"], label="train")
plt.plot(model.history["elbo_validation"], label="val")
plt.title("Negative ELBO over training epochs")
plt.ylim(1200, 2000)
plt.legend()

plt.savefig('/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline/data/figures/totalvi_loss_pbmc.png', bbox_inches="tight")
plt.close()

rna.obsm['X_totalVI'] = model.get_latent_representation()
model.save('/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline/data/models/totalvi_pbmc/', overwrite=True)

sc.pp.neighbors(rna, use_rep='X_totalVI')
sc.tl.umap(rna)

sc.pl.umap(
    rna, 
    color=[
        'Status_on_day_collection_summary',
        'initial_clustering',
        'Site'
        ],
    frameon=False,
    ncols=1,
    show=False
)
plt.savefig('/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline/data/figures/totalvi_umap_pbmc.png', bbox_inches="tight")

rna.obsm['latent'] = rna.obsm['X_totalVI'].copy() # to make consistent with multimil
rna.write('/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline/data/pp/totalvi_pbmc.h5ad')