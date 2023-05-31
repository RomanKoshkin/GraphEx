import time, sys, os

sys.path.append('../')

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from utils.constants import *
import numpy as np
from tqdm import trange, tqdm

import torch
import torch.optim as optim
import scipy.io as sio
from tqdm import trange
from termcolor import cprint
from utils.utils import load_config
from utils.utils import tonp
from utils.graph_conversion import convertToGraphs
from utils.synthetic import get_background2
from utils.synthetic import embed_one_seq, embed_three_seq
from utils.GNNTrainer import Trainer
from utils.metrics import Metrics
from models.models import WeightedGCN
from models.Kmeans import Kmeans
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.distance import cdist
from sklearn.metrics import auc

# import resource

# torch.multiprocessing.set_sharing_strategy('file_system')
# print(torch.multiprocessing.get_sharing_strategy())
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# load the parameters from YAML file
params = load_config('../configs/config.yaml')
params.fs = 1 / params.dt  # sampling rate
params.N = 452

# load CA1 data

dat1 = sio.loadmat(f'{params.path_to_data}/position_per_frame.mat')['position_per_frame'].flatten().astype('float')
dat1 = dat1 / dat1.max() * 96  # make sure the position ranges from 0 to 96 cm (as in the paper)
X = sio.loadmat(f'{params.path_to_data}/neuronal_activity_mat.mat')['neuronal_activity_mat']
params.Ts = X.shape[1]

# embed one sequence
runid = int(sys.argv[1])
p_drop = float(sys.argv[2])
gap_ts = 450
seqlen = 100
jitter_std = int(sys.argv[3])

X_ = get_background2(X, pruning_factor=0.5)
X_, GT = embed_one_seq(X_, params, p_drop, gap_ts, seqlen, jitter_std)
GT[0] = [i - seqlen // 2 for i in GT[0]]

# convert to graphs
winsize = 100
step_sz = 4
tau = 25

# make data and move data to device
DATA = convertToGraphs(X_, winsize, step_sz, tau, num_workers=20)
for i in trange(len(DATA), bar_format=bar_format):
    DATA[i] = DATA[i].to(device=params.device)

loader = DataLoader(DATA, batch_size=len(DATA), shuffle=False)
params.z_dim = 2
params.K = 2
deterministic = False

try:
    fnames = [i for i in os.listdir("../data/") if i.startswith('proj')]
    for fn in fnames:
        os.remove(f"../data/{fn}")

    model = WeightedGCN(params).to(params.device)
    kmeans = Kmeans(params, debug=False)
    writer = SummaryWriter(log_dir='../runs/')

    optimizer = optim.AdamW(model.parameters(), lr=0.05)  # tell the optimizer which var we want optimized
    # optimizer = optim.AdamW(model.parameters()) # tell the optimizer which var we want optimized
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=100)
    # scheduler = None

    trainer = Trainer(DATA, X_, model, optimizer, scheduler, kmeans, params, step_sz)
    trainer.MAX_EPOCHS = 150

    trainer.train()

    params.W = 100
    num_thresholds = 100
    x = np.arange(0, len(trainer.embeds_np) * step_sz, step_sz)
    proximities = np.reciprocal(cdist(trainer.embeds_np, tonp(kmeans.centroids), metric='euclidean'))
    proximities /= proximities.max(axis=0)
    proximities *= params.K - 1

    AUC = []
    for i in range(2):

        proximities_interp = np.interp(
            np.arange(proximities.shape[0] * step_sz),
            np.arange(0, proximities.shape[0] * step_sz, step_sz),
            proximities[:, i],
        )

        TPR, FPR = [], []

        for alpha in tqdm(np.linspace(0, proximities_interp.max(), num_thresholds), bar_format=bar_format):

            metrics = Metrics(params, GT, alpha)
            metrics_dict = metrics.get(proximities_interp)

            TPR.append(metrics_dict['tpr'])
            FPR.append(metrics_dict['fpr'])
        AUC.append(auc(FPR, TPR))

    np.save(
        f'../experiments/AUC_{p_drop}_{jitter_std}_{runid}',
        {
            'auc0': AUC[0],
            'auc1': AUC[1],
            'p_drop': p_drop,
            'runid': runid,
            'jitter_std': jitter_std,
        },
    )

except Exception as e:
    print(e)
    cprint(f'ENDING run {p_drop}', color='yellow')