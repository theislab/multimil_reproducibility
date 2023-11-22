install.packages("remotes")
remotes::install_github("davismcc/scater")
library(StabMap)
library(SingleCellMultiModal)
library(SingleCellExperiment)
library(scran)
library(Seurat)


rna_ref <- readRDS("~/Desktop/JN/stabmap/rna_ref.rds")
rna_sce <- as.SingleCellExperiment(rna_ref)
atac <- readRDS("~/Desktop/JN/stabmap/atac.rds")
atac_sce <- as.SingleCellExperiment(atac)
adt <- readRDS("~/Desktop/JN/stabmap/adt.rds")
adt_sce <- as.SingleCellExperiment(adt)

assay_list = list(rna_log = logcounts(rna_sce),
                  atac_log = logcounts(atac_sce),
                  adt_log = logcounts(adt_sce))

lapply(assay_list, dim)
lapply(assay_list, class)

mosaicDataUpSet(assay_list, plot = TRUE)

mdt = mosaicDataTopology(assay_list)
mdt
plot(mdt)

stab = stabMap(assay_list,
               reference_list = c("rna_log"),
               maxFeatures = Inf,
               plot = TRUE)
dim(stab)
stab[1:5,1:5]

stab_umap = calculateUMAP(t(stab))
dim(stab_umap)

write.csv(stab, file = "stabmap_ATAC_ADT_mapping.csv")