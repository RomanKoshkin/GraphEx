import sys

sys.path.append('../../../..')
from utils.constants import bar_format
import numpy as np
import pandas as pd
from tqdm import trange

import scipy.io as sio
from tqdm import tqdm, trange
from termcolor import cprint

from utils.utils import load_config, saveForPPSeq
from utils.synthetic import get_background, get_background2

# load the parameters from YAML file
params = load_config('../../../../configs/config.yaml')
params.fs = 1 / params.dt  # sampling rate

# load data
dat1 = sio.loadmat(f'../../../../data/position_per_frame.mat')['position_per_frame'].flatten().astype('float')
dat1 = dat1 / dat1.max() * 96  # make sure the position ranges from 0 to 96 cm (as in the paper)
X = sio.loadmat(f'../../../../data/neuronal_activity_mat.mat')['neuronal_activity_mat']

# update parameters with values calculated based on data size

end = X.shape[1]
Ts = X[:, :end].shape[1]
T = Ts * params.dt
t = np.arange(0, T, params.dt)
params.N = X[:, :end].shape[0]
params.kernel_size = (params.N, params.W)
params.padding = tuple(params.padding)
params.Ts = Ts

grid = pd.read_csv('grid.csv')

for k in trange(grid.shape[0], bar_format=bar_format):

    # make background activity (with same FR as some input data)
    # X_ = get_background(X)
    X_ = get_background2(X, pruning_factor=0.5)

    p_drop = float(grid.iloc[k].p_drop)
    gap_ts = int(grid.iloc[k].isi)
    seqlen = int(grid.iloc[k].seqlen)
    jitter_std = float(grid.iloc[k].jitter)
    cellid = int(grid.iloc[k].cellid)

    # embed sequences at set intervals, of set duration on set neurons:
    seqA_gt = []
    for st in range(100, params.Ts - gap_ts, gap_ts):
        for i, disp in enumerate(range(st, st + seqlen)):
            jitter = np.round(np.random.randn() * jitter_std).astype(int)
            j = disp + jitter
            if j < params.Ts:
                pass
                X_[i, j] = np.random.choice([0, 1], p=[p_drop, 1 - p_drop])
        seqA_gt.append(np.round(np.mean([st, j])))

    GT = [seqA_gt]
    sio.savemat(f"matlab_dataset_{p_drop}_{jitter_std}.mat", {'X_': X_})