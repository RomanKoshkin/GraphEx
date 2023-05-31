import time, sys, os

sys.path.append('../')

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from utils.constants import *
import numpy as np
from tqdm import trange

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
from models.models import WeightedGCN
from models.Kmeans import Kmeans
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import resource

torch.multiprocessing.set_sharing_strategy('file_system')
print(torch.multiprocessing.get_sharing_strategy())
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

assert len(sys.argv) > 1, "Dataset type argument missing: must be 0, 1 for NULL, NOT NULL."
Null = bool(int(sys.argv[1]))
assert Null in [True, False], "Wrong argument for dataset type (must be 0, 1 for NULL, NOT NULL)."

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
p_drop = 0.2
gap_ts = 800
seqlen = 100
jitter_std = 15

# convert to graphs
winsize = 100
step_sz = 4
tau = 25

X_ = get_background2(X, pruning_factor=0.0)
X_, GT = embed_three_seq(X_, params, p_drop, gap_ts, seqlen, jitter_std)

if Null:
    cprint('Using null data', color='cyan')
    X_ = get_background2(X_, pruning_factor=0.0)

# make data and move data to device
DATA = convertToGraphs(X_, winsize, step_sz, tau, num_workers=20)
for i in trange(len(DATA), bar_format=bar_format):
    DATA[i] = DATA[i].to(device=params.device)

loader = DataLoader(DATA, batch_size=len(DATA), shuffle=False)
params.z_dim = 6
params.K = 6
deterministic = False

runid = 0
while runid < 20:
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

        for batch in loader:
            pass
        ems, logits = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch, deterministic=deterministic)
        gram = tonp(ems.matmul(ems.T))
        np.save(
            f'../experiments/iter_{"null" if Null else "real"}_{runid}',
            {
                'sm': gram.sum(axis=0),
                'simmat': gram,
                'ems': tonp(ems),
            },
        )
        runid += 1
    except Exception as e:
        print(e)
        cprint(f'RESTARTING run {runid}', color='yellow')