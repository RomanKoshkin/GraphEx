import matplotlib
import matplotlib.pyplot as plt
import torch, time, sys, os

sys.path.append('../')
import numpy as np
import pandas as pd
from tqdm import trange

import scipy.io as sio
from tqdm import tqdm, trange

from utils.constants import bar_format
from utils.utils import load_config
from utils.synthetic import get_background

params = load_config('../configs/config.yaml')

DATASET_NAME = sys.argv[1]
assert DATASET_NAME in ['SINGLE', 'DOUBLE', 'QUAD'], "DATASET_NAME must be one of: 'SINGLE', 'DOUBLE' or 'QUAD'."


def double_spikes_in_background(X):
    h, w = X.shape
    # for each neuron, add the same spikes at permuted times
    for i in range(h):
        perm_idx = np.arange(w)
        np.random.shuffle(perm_idx)
        X[i, :] = np.logical_or(X[i, :], X[i, perm_idx])

    # for each timestep, permute the neuron ids
    for j in range(w):
        perm_idx = np.arange(h)
        np.random.shuffle(perm_idx)
        X[:, j] = X[perm_idx, j]

    return X


# load CA1 data
dat1 = sio.loadmat(f'{params.path_to_data}/position_per_frame.mat')['position_per_frame'].flatten().astype('float')
dat1 = dat1 / dat1.max() * 96  # make sure the position ranges from 0 to 96 cm (as in the paper)
X = sio.loadmat(f'{params.path_to_data}/neuronal_activity_mat.mat')['neuronal_activity_mat']
print('shape: ', X.shape)

# NOTE: leave 76 or 152 neurons
if DATASET_NAME == 'QUAD':
    X = X[:76 * 2, :]
else:
    X = X[:76, :]

# make background activity (with same FR as some input data)
X_ = get_background(np.copy(X))
print(f'Number of spikes: {X_.sum()}')
print(f'Spike density:: {X_.sum()/np.prod(X.shape)}')

if DATASET_NAME in ['DOUBLE', 'QUAD']:
    # Double the number of spikes in the background activity
    X_ = double_spikes_in_background(np.copy(X_))
print(f'Number of spikes: {X_.sum()}')
print(f'Spike density:: {X_.sum()/np.prod(X_.shape)}')

# create a long background activity mimicking a real CA1 recording.
A = []
for i in trange(30, bar_format=bar_format):
    permutation_indices = np.random.permutation(X_.shape[1])
    A.append(X_[:, permutation_indices])
X_ = np.hstack(A)
print(X_.shape)
print(f'Spike density:: {X_.sum()/np.prod(X_.shape)}')

# update parameters with values calculated based on data size
end = X_.shape[1]
Ts = X_[:, :end].shape[1]
T = Ts * params.dt
t = np.arange(0, T, params.dt)
params.N = X_[:, :end].shape[0]
params.kernel_size = (params.N, params.W)
params.padding = tuple(params.padding)
params.Ts = Ts

# embed sequences at set intervals, of set duration on set neurons:
p_drop = 0.2
gap_ts = 200
seqlen = 40
jitter_std = 10

second_sequence_starts_at_neuron = seqlen + 10
second_sequence_starts_at_t = seqlen + 80

seqA_gt = []
for st in trange(0, params.Ts - gap_ts, gap_ts, bar_format=bar_format):
    try:
        for i, disp in enumerate(range(st, st + seqlen)):
            jitter = np.round(np.random.randn() * jitter_std).astype(int)
            j = disp + jitter
            if j < params.Ts:
                pass
                X_[i, j] = np.random.choice([0, 1], p=[p_drop, 1 - p_drop])
        seqA_gt.append(np.round(np.mean([st, j])))
    except Exception as e:
        print(e)

GT = [seqA_gt]

# for seqNMF
sio.savemat(f"../baselines/seqNMF/matlab_matrix_{DATASET_NAME}.mat", {'X_': X_})

# for PP-Seq and Ours
np.save(f'../baselines/PPSeq.jl/demo/data/seqA_gt_{DATASET_NAME}', seqA_gt)

for leng in tqdm([4441, 8882, 13323, 17764, 22205, 26646, 100000, 500000], bar_format=bar_format):

    XX_subset = X_[:, :leng]

    # for Ours
    np.save(f'../baselines/PPSeq.jl/demo/data/X_{DATASET_NAME}_{leng}', XX_subset)

    # for PP-Seq
    with open(f'../baselines/PPSeq.jl/demo/data/one_seq_{DATASET_NAME}_{leng}.txt', 'w') as f:
        for i in range(XX_subset.shape[0]):
            for j in range(XX_subset.shape[1]):
                if XX_subset[i, j] == 1:
                    f.write(f"{float(i+1)}\t{float(j)}\n")