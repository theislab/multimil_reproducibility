import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
onehot_encoder = OneHotEncoder(sparse_output=False)
from scipy.special import logsumexp


def loss(X, Y, W):
    """
    Y: onehot encoded
    """
    Z = - X @ W
    N = X.shape[0]
    loss = 1/N * (np.trace(X @ W @ Y.T) + np.sum(logsumexp(Z, axis=1)))
    return loss

def gradient(X, Y, W, mu):
    """
    Y: onehot encoded 
    """
    Z = - X @ W
    P = softmax(Z, axis=1)
    N = X.shape[0]
    gd = 1/N * (X.T @ (Y - P)) + 2 * mu * W
    return gd

def gradient_descent(X, Y, max_iter=2000, eta=0.1, mu=0.01):
    """
    Very basic gradient descent algorithm with fixed eta and mu
    """
    Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
    W = np.zeros((X.shape[1], Y_onehot.shape[1]))
    step = 0
    step_lst = [] 
    loss_lst = []
    W_lst = []
 
    while step < max_iter:
        step += 1
        W -= eta * gradient(X, Y_onehot, W, mu)
        step_lst.append(step)
        W_lst.append(W)
        loss_lst.append(loss(X, Y_onehot, W))

    df = pd.DataFrame({
        'step': step_lst, 
        'loss': loss_lst
    })
    return df, W

class Multiclass:
    def fit(self, X, Y):
        self.loss_steps, self.W = gradient_descent(X, Y)

    def loss_plot(self):
        return self.loss_steps.plot(
            x='step', 
            y='loss',
            xlabel='step',
            ylabel='loss'
        )

    def predict(self, H):
        Z = - H @ self.W
        P = softmax(Z, axis=1)
        return np.argmax(P, axis=1)

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

def run_freq_mr(adata, sample_key, condition_key, n_splits, params, label_key, method, **kwargs):

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
        model = Multiclass()
        model.fit(X, Y)
        print(f'Train accuracy = {np.sum(model.predict(X) == Y)/len(Y)}.')
        y_pred = model.predict(x_val)
        val_accuracy = np.sum(y_pred == y_val)/len(y_val)

        df = classification_report(y_val, y_pred, output_dict=True)
        df = pd.DataFrame(df).T
        df['split'] = i
        df['method'] = 'freq_mr'
        dfs.append(df)
        
        val_accuracy = df["f1-score"]["accuracy"]

        val_accuracies.append(val_accuracy)
        val_avg.append(df["f1-score"]["weighted avg"])
        
        print(f'Val accuracy = {val_accuracy}.')
        print('===========================')

    df = pd.concat(dfs)

    print(f"Mean validation accuracy across 5 CV splits for a {method} model = {np.mean(np.array(val_accuracies))}.")
    print(f"Mean validation weighted avg across 5 CV splits for a {method} model = {np.mean(np.array(val_avg))}.")
    return df