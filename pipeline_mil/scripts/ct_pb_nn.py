import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from random import sample
import math
import numpy as np
import scanpy as sc
import decoupler as dc
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        if x.shape[0] > 1:
            x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
    
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc


def run_ct_pb_nn(adata, sample_key, condition_key, n_splits, params, hash, method, task, label_key, **kwargs):
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

    num_features = adata_.shape[1]
    num_classes = len(adata_.obs[condition_key].cat.categories)
    train_fraction = 0.8
    batch_size = params['batch_size']
    learning_rate = params['lr']
    epochs = params['epochs']

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
        # val data, later this is called test data because the val data is subset of train
        x_val = pd.DataFrame(adata_[adata_.obs_names.isin(val)].X).to_numpy()
        y_val = adata_[adata_.obs_names.isin(val)].obs[condition_key].cat.rename_categories(rename_dict)
        y_val = y_val.to_numpy()
        print("Val shapes:")
        print(f"x_val.shape = {x_val.shape}")
        print(f"y_val.shape = {y_val.shape}")
        # fit
        X = x
        Y = y
        n_of_train_samples = int(math.ceil(len(y) * train_fraction))
        train_samples = sample(range(len(y)), n_of_train_samples)
        val_samples = [i for i in range(len(y)) if i not in train_samples]
        X_test = x_val
        y_test = y_val
        X_train = x[train_samples]
        y_train = y[train_samples]
        X_val = x[val_samples]
        y_val = y[val_samples]
        
        # create datasets
        train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
        test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
        
        # create loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)
        
        # init model
        model = MulticlassClassification(num_feature = num_features, num_class=num_classes)
        # define loss
        criterion = nn.CrossEntropyLoss()
        # define optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # loss recoder
        accuracy_stats = {
            'train': [],
            "val": []
        }
        loss_stats = {
            'train': [],
            "val": []
        }
        
        # train
        print("Begin training.")
        for e in tqdm(range(1, epochs+1)):

            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0
            model.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch, y_train_batch
                optimizer.zero_grad()

                y_train_pred = model(X_train_batch)

                train_loss = criterion(y_train_pred, y_train_batch)
                train_acc = multi_acc(y_train_pred, y_train_batch)

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()


            # VALIDATION    
            with torch.no_grad():

                val_epoch_loss = 0
                val_epoch_acc = 0

                model.eval()
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch, y_val_batch

                    y_val_pred = model(X_val_batch)

                    val_loss = criterion(y_val_pred, y_val_batch)
                    val_acc = multi_acc(y_val_pred, y_val_batch)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            loss_stats['train'].append(train_epoch_loss/len(train_loader))
            loss_stats['val'].append(val_epoch_loss/len(val_loader))
            accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
            accuracy_stats['val'].append(val_epoch_acc/len(val_loader))


        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
        
        # losses
        # Create dataframes
        train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        # Plot the dataframes
        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
        fig_path = f'data/reports/{task}/{method}/{hash}/figures/'
        os.makedirs(fig_path, exist_ok = True)
        sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
        sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch').get_figure().savefig(fig_path + f'plot_loss_split_{i}.png')
        
        # predict
        y_pred_list = []
        with torch.no_grad():
            model.eval()
            for X_batch, _ in test_loader:
                X_batch = X_batch
                y_test_pred = model(X_batch)
                _, y_pred_tags = torch.max(y_test_pred, dim = 1)
                y_pred_list.append(y_pred_tags.cpu().numpy())
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        
        df = classification_report(y_test, y_pred_list, output_dict=True)
        df = pd.DataFrame(df).T
        df['split'] = i
        df['method'] = 'ct_pb_nn'
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
