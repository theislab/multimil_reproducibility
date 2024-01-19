import scanpy as sc
import pandas as pd
import numpy as np
import scipy

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


def run_gex_rf(adata, condition_key, n_splits, params, **kwargs):

    adata.obs[condition_key] = adata.obs[condition_key].astype('category')
    rename_dict = {name: number for number, name in enumerate(np.unique(adata.obs[condition_key]))}
    
    if params['norm'] is True:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    val_accuracies = []
    val_avg = []
    dfs = []

    for i in range(n_splits):
        print(f'Processing split = {i}...')
        # train data
        if scipy.sparse.issparse(adata.X):
            x = pd.DataFrame(adata[adata.obs[f'split{i}'] == 'train'].X.A).to_numpy()
        else:
            x = pd.DataFrame(adata[adata.obs[f'split{i}'] == 'train'].X).to_numpy()
        y = adata[adata.obs[f'split{i}'] == 'train'].obs[condition_key].cat.rename_categories(rename_dict)
        y = y.to_numpy()
        print("Train shapes:")
        print(f"x.shape = {x.shape}")
        print(f"y.shape = {y.shape}")
        # val data
        if scipy.sparse.issparse(adata.X):
            x_val = pd.DataFrame(adata[adata.obs[f'split{i}'] == 'val'].X.A).to_numpy()
        else:
            x_val = pd.DataFrame(adata[adata.obs[f'split{i}'] == 'val'].X).to_numpy()
        y_val = adata[adata.obs[f'split{i}'] == 'val'].obs[condition_key].cat.rename_categories(rename_dict)
        y_val = y_val.to_numpy()
        print("Val shapes:")
        print(f"x_val.shape = {x_val.shape}")
        print(f"y_val.shape = {y_val.shape}")
        # fit
        X = x
        Y = y
        clf = RandomForestClassifier()
        clf.fit(X, Y)
        print(f'Train accuracy = {np.sum(clf.predict(X) == Y)/len(Y)}.')
        y_pred = clf.predict(x_val)
        val_accuracy = np.sum(y_pred == y_val)/len(y_val)
        
        df = classification_report(y_val, y_pred, output_dict=True)
        df = pd.DataFrame(df).T
        df['split'] = i
        df['method'] = 'gex_rf'
        dfs.append(df)
        
        val_accuracies.append(df["f1-score"]["accuracy"])
        val_avg.append(df["f1-score"]["weighted avg"])
        
        val_accuracy = df["f1-score"]["accuracy"]
        
        print(f'Val accuracy = {val_accuracy}.')
        print('===========================')

    df = pd.concat(dfs)
    
    print(f"Mean validation accuracy across 5 CV splits for a random forest model = {np.mean(np.array(val_accuracies))}.")
    print(f"Mean validation weighted avg across 5 CV splits for a random forest model = {np.mean(np.array(val_avg))}.")
    return df
