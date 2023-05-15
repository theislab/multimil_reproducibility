
import os


def get_existing_checkpoints(rootdir):

    checkpoints = []
    splits = []
    patterns = []

    for root, _, files in os.walk(rootdir):
        for filename in files:
            if filename.endswith('ckpt') and 'query' not in root:
                patterns.append(root.split('/')[-3])
                splits.append(root.split('/')[-2])
                checkpoints.append(filename.strip('.ckpt'))

    return checkpoints, splits, patterns



def get_query_checkpoints(rootdir):

    checkpoints = []
    splits = []
    patterns = []
    q_checkpoints = []

    for root, _, files in os.walk(rootdir):
        for filename in files:
            if filename.endswith('ckpt') and 'query' in root:
                checkpoints.append(root.split('/')[-1])
                patterns.append(root.split('/')[-4])
                splits.append(root.split('/')[-3])
                q_checkpoints.append(filename.strip('.ckpt'))

    return checkpoints, splits, patterns, q_checkpoints