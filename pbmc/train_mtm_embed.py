import scanpy as sc
import multigrate as mtg
from matplotlib import pyplot as plt

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

adata = mtg.data.organize_multiome_anndatas(
     adatas = [[rna], [adt]],
)

mtg.model.MultiVAE.setup_anndata(
    adata,
    rna_indices_end=2000,
    categorical_covariate_keys=[
        "Site",
        "patient_id",
    ]
)

vae = mtg.model.MultiVAE(
    adata, 
    losses=[
        "nb", "mse"
    ],
    loss_coefs={
        "kl": 0.1,
    },
    cond_dim=16,
    z_dim=20, # input dim = dim of the pre-trained 
)

vae.train(lr=1e-4)

vae.plot_losses(save='/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline/data/figures/mtg_pretrain_full_pbmc/losses1.png')

vae.get_latent_representation()

sc.pp.neighbors(adata, use_rep='latent')
sc.tl.umap(adata)

sc.pl.umap(
    adata, 
    color=[
      "Status_on_day_collection_summary",
      "Site",
      "initial_clustering",
      "full_clustering",
    ],
    ncols=1,
    frameon=False,
    show=False,
)
plt.savefig(f'/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline/data/figures/mtg_pretrain_full_pbmc/train_umap1.png', bbox_inches="tight")
plt.close()

adata.write('/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline/data/pp/mtg_pretrain_pbmc_full/pbmc_full_mtg1.h5ad')
vae.save('/lustre/groups/ml01/projects/2022_multigrate_anastasia.litinetskaya/multimil_reproducibility/pipeline/data/models/mtg_pretrain_pbmc_full/model1/', overwrite=True)