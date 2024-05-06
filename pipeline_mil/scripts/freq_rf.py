import scanpy as sc
import pandas as pd
import decoupler as dc
import numpy as np

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


def create_frequency_dataset(adata, celltype, donor, condition, standartize, rename_dict, ct_to_keep):
    missing_ct = list(set(ct_to_keep) - set(adata.obs[celltype]))
    df = adata.obs[[celltype, donor]].groupby([celltype, donor]).size().reset_index(name='count')

    unique_samples = np.unique(adata.obs[donor])
    missing_df = {celltype: [], donor: [], 'count': []}
    for ct in missing_ct:
        for sample in unique_samples:
            missing_df[celltype].append(ct)
            missing_df[donor].append(sample)
            missing_df['count'].append(0)
    missing_df = dict(missing_df)
    missing_df = pd.DataFrame(missing_df)

    df = pd.concat([df, missing_df])
    df = df.reset_index()
    
    X = []
    y = []

    for sample in df[donor].unique():
        df_sample = df[df[donor] == sample]
        df_sample = df_sample.sort_values(celltype)
        X.append(df_sample['count'].values)
        y.append(rename_dict[adata[adata.obs[donor] == sample].obs[condition][0]])

    X = np.array(X)
    y = np.array(y)

    # drop donors with less than 100 cells in total
    idx = np.argwhere(np.sum(X, axis=1) <= 100)
    X = np.delete(X, idx, axis=0)
    y = np.delete(y, idx)
    
    if standartize is True:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    return X, y

def run_freq_rf(adata, sample_key, condition_key, n_splits, params, label_key, method, **kwargs):

    adata.obs[condition_key] = adata.obs[condition_key].astype('category')
    adata.obs[sample_key] = adata.obs[sample_key].astype('category')
    adata.obs[label_key] = adata.obs[label_key].astype('category')

    # all_cts = set(adata.obs[label_key].cat.categories)
    # all_cts_sorted = sorted(list(all_cts))

    adata.obs[sample_key] = adata.obs[sample_key].astype(str)

    rename_dict = {name: number for number, name in enumerate(np.unique(adata.obs[condition_key]))}
    ct_to_keep = list(np.unique(adata.obs[label_key]))
    standartize = params['norm']

    val_accuracies = []
    val_avg = []
    dfs = []
    
    for i in range(n_splits):
        print(f'Processing split = {i}...')
        train = adata[adata.obs[f'split{i}'] == 'train'].copy()
        val = adata[adata.obs[f'split{i}'] == 'val'].copy()
        # train data
        x, y = create_frequency_dataset(
            train,
            celltype=label_key,
            donor=sample_key,
            condition=condition_key,
            standartize=standartize,
            rename_dict=rename_dict,
            ct_to_keep=ct_to_keep,
        )
        print("Train shapes:")
        print(f"x.shape = {x.shape}")
        print(f"y.shape = {y.shape}")
        # val data
        x_val, y_val = create_frequency_dataset(
            val,
            celltype=label_key,
            donor=sample_key,
            condition=condition_key,
            standartize=standartize,
            rename_dict=rename_dict,
            ct_to_keep=ct_to_keep,
        )
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
        df['method'] = 'freq_rf'
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
