library(edgeR)
library(MAST)

fit_model <- function(adata_){
    # create an edgeR object with counts and grouping factor
    y <- DGEList(assay(adata_, "X"), group = colData(adata_)$condition)
    # filter out genes with low counts
    print("Dimensions before subsetting:")
    print(dim(y))
    print("")
    keep <- filterByExpr(y)
    y <- y[keep, , keep.lib.sizes=FALSE]
    print("Dimensions after subsetting:")
    print(dim(y))
    print("")
    # normalize
    y <- calcNormFactors(y)
    # will add study covariate to the design matrix as there are still some batch effects
    group <- colData(adata_)$group
    replicate <- colData(adata_)$sample
    study <- colData(adata_)$study
    # create a design matrix: here we have multiple donors so also consider that in the design matrix
    design <- model.matrix(~ 0 + group + replicate)
    # design <- design[, colnames(design) != "replicateMH9179823"]
    print(design)
    # estimate dispersion
    y <- estimateDisp(y, design = design)
    # fit the model
    fit <- glmQLFit(y, design)

    png(file="mds.png")
    plotMDS(y, col=ifelse(y$samples$group == "all", "red", "blue"))
    dev.off()

    png(file="bcv.png")
    plotBCV(y)
    dev.off()

    myContrast <- makeContrasts("grouptop - groupall", levels = y$design)
    qlf <- glmQLFTest(fit, contrast=myContrast)
    # get all of the DE genes and calculate Benjamini-Hochberg adjusted FDR
    tt <- topTags(qlf, n = Inf)
    tt <- tt$table

    png(file="smear.png")
    plotSmear(qlf, de.tags = rownames(tt)[which(tt$FDR<0.01)])
    dev.off()

    return(tt)
    #return(list("fit"=fit, "design"=design, "y"=y))
}