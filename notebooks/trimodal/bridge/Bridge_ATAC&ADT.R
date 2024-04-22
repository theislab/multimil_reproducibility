library("Seurat")
library(Signac)
library(ggplot2)
options(Seurat.object.assay.version = 'v5')
setwd("~/Desktop/JN/wnn-bridge")
#read the concatanated RNA-multiom and RNA CITE
bridge_ref <- readRDS("~/Desktop/JN/wnn-bridge/bridge_ref.rds")
#have a look at the UMAP and group by celltype
p1 <- DimPlot(bridge_ref, reduction = "umap", group.by = "celltype", label = TRUE, repel = TRUE) + NoLegend()
p1
#have a look at the UMAP and group by batch
p2 <- DimPlot(bridge_ref, reduction = "umap", group.by = "batch", label = TRUE, repel = TRUE) + NoLegend()
p2
#read the multiome bridge data, containing the RNA and ATAC for side3 donor 7
multiom_bridge <- readRDS("~/Desktop/JN/wnn-bridge/multiom_bridge.rds")
#read the ATAC query, ATAC for side4 donor9
atac_query <- readRDS("~/Desktop/JN/wnn-bridge/atac_query.rds")
atac_query <- RenameAssays(object = atac_query, originalexp = "ATAC", verbose=TRUE) 
#Preprocessing/normalization for all datasets
#note that RNA reference is already SCT transformed, and pca and umap reduction is already perdormed
#normalize multiome RNA
DefaultAssay(multiom_bridge) <- "RNA"
multiom_bridge <- SCTransform(multiom_bridge, verbose = TRUE)
#normalize multiome ATAC
DefaultAssay(multiom_bridge) <- "ATAC"
multiom_bridge <- RunTFIDF(multiom_bridge)
multiom_bridge <- FindTopFeatures(multiom_bridge, min.cutoff = "q0")
#normalize query
atac_query <- RunTFIDF(atac_query)
# Drop first dimension for ATAC reduction
dims.atac <- 2:50
dims.rna <- 1:50
DefaultAssay(multiom_bridge) <-  "RNA"
DefaultAssay(bridge_ref) <- "integrated"
#prepare the bridge ref
obj.rna.ext <- PrepareBridgeReference(
  reference = bridge_ref, bridge = multiom_bridge,
  reference.reduction = "pca", reference.dims = dims.rna,
  normalization.method = "LogNormalize")
# this function translates the query dataset using the same dictionary as was used to translate the reference, and then identifies anchors in this space
bridge.anchor <- FindBridgeTransferAnchors(
  extended.reference = obj.rna.ext, query = atac_query,
  reduction = "lsiproject", dims = dims.atac)
#Once we have identified anchors, we can map the query dataset onto the reference
obj.atac <- MapQuery(anchorset = bridge.anchor, reference = obj.rna.ext,
                     query = atac_query,reduction.model = "umap")
#Now we can visualize the results, plotting the scATAC-seq cells
obj.atac <- FindTopFeatures(obj.atac, min.cutoff = "q0")
obj.atac <- RunSVD(obj.atac)
obj.atac <- RunUMAP(obj.atac, reduction = "lsi", dims = 2:50)

DimPlot(obj.atac, group.by = "celltype", reduction = "umap", label = FALSE)

p10 <- DimPlot(obj.rna.ext, group.by = "celltype", label = TRUE) + NoLegend() + ggtitle("RNAref")
p11 <- DimPlot(obj.atac, group.by = "celltype", label = TRUE) + NoLegend() + ggtitle("ATAC")
p10 + p11

saveRDS(obj.rna.ext, file = "objrnaext.rds")
saveRDS(obj.atac, file = "objatac.rds")

#For ADT mapping, first start by reading the CITE bridge data, site3 donor 7
cite_bridge <- readRDS("~/Desktop/JN/wnn-bridge/cite_bridge.rds")
# and reading the ADT query 
adt_query <- readRDS("~/Desktop/JN/wnn-bridge/adt_query.rds")
adt_query <- RenameAssays(object = adt_query, originalexp = "ADT", verbose=TRUE) 
#Preprocess and normalize all datasets
DefaultAssay(cite_bridge) <- "RNA"
cite_bridge <- SCTransform(cite_bridge, verbose = TRUE)

DefaultAssay(cite_bridge) <- "ADT"
VariableFeatures(cite_bridge) <- rownames(cite_bridge)
cite_bridge <- NormalizeData(cite_bridge, normalization.method = 'CLR', margin = 2, verbose=TRUE)
cite_bridge <- ScaleData(cite_bridge, verbose=TRUE)

VariableFeatures(adt_query) <- rownames(adt_query)
adt_query <- NormalizeData(adt_query, normalization.method = 'CLR', margin = 2, verbose=TRUE)
adt_query <- ScaleData(adt_query, verbose = TRUE)
adt_query <- RunPCA(adt_query, verbose=TRUE, reduction.name = 'apca')

#Map ADT query onto RNAref
dims.adt <- 1:50
dims.rna <- 1:50
obj.rna.ext2 <- PrepareBridgeReference(reference = bridge_ref,
                                       bridge = cite_bridge,
                                       bridge.query.assay = "ADT",
                                       reference.reduction = "pca",
                                       reference.dims = dims.rna,
                                       normalization.method = 'LogNormalize',
                                       supervised.reduction = "spca",
                                       verbose=TRUE)
bridge.anchor2 <- FindBridgeTransferAnchors(extended.reference = obj.rna.ext2, 
                                            query = adt_query,
                                            reduction = "pcaproject",
                                            dims = dims.adt,
                                            scale = FALSE,
                                            verbose=TRUE)
obj.adt <- MapQuery(anchorset = bridge.anchor2, 
                      reference = obj.rna.ext2, 
                      query = adt_query, 
                      reduction.model = "umap",
                      verbose=TRUE)

obj.adt <- RunUMAP(obj.adt, reduction = "apca",verbose = TRUE, dims = 1:30)
p12 <- DimPlot(obj.adt, group.by = "celltype", label = TRUE) + NoLegend() + ggtitle("ADT")
p10 + p12

saveRDS(obj.rna.ext2, file = "objrnaext2.rds")
saveRDS(obj.adt, file = "objadt.rds")

# save as .csv to then read in with python to work with scanpy objects
bridge_ref <- readRDS("bridge_ref.rds")
objatac <- readRDS("objatac.rds")
objadt <- readRDS("objadt.rds")
obj.rna.ext <- readRDS("objrnaext.rds")
obj.rna.ext2 <- readRDS("objrnaext2.rds")

write.table(bridge_ref@reductions[["pca"]]@cell.embeddings, file="ref_pca.csv", sep = ';')
write.table(objatac@reductions[["ref.Bridge.reduc"]]@cell.embeddings, file="atac_ref_pca.csv", sep = ';')
write.table(objadt@reductions[["ref.Bridge.reduc"]]@cell.embeddings, file="adt_ref_pca.csv", sep = ';')

write.table(bridge_ref@reductions[["umap"]]@cell.embeddings, file="ref_umap.csv", sep = ';')
write.table(objatac@reductions[["ref.umap"]]@cell.embeddings, file="atac_ref_umap.csv", sep = ';')
write.table(objadt@reductions[["ref.umap"]]@cell.embeddings, file="adt_ref_umap.csv", sep = ';')

write.table(bridge_ref@meta.data, file='ref_meta.csv', sep = ';')
write.table(objatac@meta.data, file='atac_meta.csv', sep = ';')
write.table(objadt@meta.data, file='adt_meta.csv', sep = ';')

write.table(obj.rna.ext@assays[["Bridge"]]@data, file='objrnaext_bridge.csv', sep=';')
write.table(obj.rna.ext2@assays[["Bridge"]]@data, file='objrnaext_bridge2.csv', sep=';')





