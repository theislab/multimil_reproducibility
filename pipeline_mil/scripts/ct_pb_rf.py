import scanpy as sc
import pandas as pd
import decoupler as dc
import numpy as np

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


def run_ct_pb_rf(adata, sample_key, condition_key, n_splits, params, label_key, **kwargs):
    adata.obs[condition_key] = adata.obs[condition_key].astype('category')
    all_cts = set(adata.obs[label_key].cat.categories)
    all_cts_sorted = sorted(list(all_cts))

    adata.obs[sample_key] = adata.obs[sample_key].astype(str)
    adata_ = dc.get_pseudobulk(adata, sample_col=sample_key, groups_col=label_key, layer=None, min_prop=-1, min_smpls=0, min_cells=0, min_counts=0, skip_checks=True)
    
    rename_dict = {name: number for number, name in enumerate(np.unique(adata_.obs[condition_key]))}

    if params['norm'] is True:
        sc.pp.normalize_total(adata_, target_sum=1e4)
        sc.pp.log1p(adata_)
    adata_.obs[condition_key] = adata_.obs[condition_key].astype('category')
    adata_.obs[sample_key] = adata_.obs[sample_key].astype('category')
    adata_.obs[label_key] = adata_.obs[label_key].astype('category')

    df = {}
    for donor in adata_.obs[sample_key].cat.categories:
        df[donor] = {}
        for ct in adata_.obs[label_key].cat.categories:
            tmp = adata_[adata_.obs[sample_key] == donor].copy()
            tmp = tmp[tmp.obs[label_key] == ct].copy()
            if len(tmp) > 0:
                df[donor][ct] = tmp.X[0]
    missing_columns = {}
    for donor in df.keys():
        df[donor] = pd.DataFrame(df[donor])
        missing_columns[donor] = list(all_cts.difference(set(df[donor].columns)))
        df[donor][[missing_columns[donor]]] = 0.0
        df[donor] = df[donor][all_cts_sorted]
        df[donor] = df[donor].T.stack().values
    df = pd.DataFrame(df)
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.T
    adata_ = sc.AnnData(df)
    
    obs_to_keep = [sample_key, condition_key]
    obs_to_keep.extend([f'split{i}' for i in range(n_splits)])
    obs = adata.obs[obs_to_keep].drop_duplicates().sort_values(sample_key).set_index(sample_key)
    adata_.obs = adata_.obs.join(obs)
    adata_.obs[condition_key] = adata_.obs[condition_key].astype('category')

    val_accuracies = []
    val_avg = []
    dfs = []

    for i in range(n_splits):
        print(f'Processing split = {i}...')
        df = adata.obs[[f'split{i}', sample_key]].drop_duplicates()
        train = list(df[df[f'split{i}'] == 'train'][sample_key])
        val = list(df[df[f'split{i}'] == 'val'][sample_key])
        # train data
        x = pd.DataFrame(adata_[adata_.obs_names.isin(train)].X).to_numpy()
        y = adata_[adata_.obs_names.isin(train)].obs[condition_key].cat.rename_categories(rename_dict)
        y = y.to_numpy()
        print("Train shapes:")
        print(f"x.shape = {x.shape}")
        print(f"y.shape = {y.shape}")
        # val data
        x_val = pd.DataFrame(adata_[adata_.obs_names.isin(val)].X).to_numpy()
        y_val = adata_[adata_.obs_names.isin(val)].obs[condition_key].cat.rename_categories(rename_dict)
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
        
        df = classification_report(y_val, y_pred, output_dict=True)
        df = pd.DataFrame(df).T
        df['split'] = i
        df['method'] = 'ct_pb_rf'
        dfs.append(df)

        val_accuracy = df["f1-score"]["accuracy"]

        val_accuracies.append(val_accuracy)
        val_avg.append(df["f1-score"]["weighted avg"])
        
        
        print(f'Val accuracy = {val_accuracy}.')
        print('===========================')

    df = pd.concat(dfs)
    print(f"Mean validation accuracy across 5 CV splits for a random forest model = {np.mean(np.array(val_accuracies))}.")
    print(f"Mean validation weighted avg across 5 CV splits for a random forest model = {np.mean(np.array(val_avg))}.")
    return df
