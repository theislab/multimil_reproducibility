from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


input_file = snakemake.input.tsv
df = pd.read_csv(input_file, sep='\t', index_col=None)

for i, task in enumerate(np.unique(df['task'])):
    runs = {}
    yerr = {}
    df_task = df[df['task'] == task]
    for method in np.unique(df_task['method']):
        runs[method] = eval(df_task.loc[df_task['method'] == method, 'accuracies'].values[0])
        yerr[method] = np.array(runs[method]).std()
    
    df_tmp = df_task[['method', 'accuracy']]
    df_tmp.index = df_tmp['method']
    df_tmp = df_tmp[['accuracy']].sort_values('accuracy', ascending=False)
    df_tmp = df_tmp.T
    
    yerr_sorted = [yerr[method] for method in df_tmp.columns]

    ax = df_tmp.T.plot(kind='bar', zorder=3, figsize=(5, 3), color=["#00a8cc"], rot=90, yerr=yerr_sorted)
    ax.get_legend().remove()
    ax.grid(zorder=0, linewidth=0.5)
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title(task)
    plt.savefig(snakemake.output[i], bbox_inches='tight')