import sys

sys.path.append('../../../..')
import numpy as np
import pandas as pd
from utils.constants import bar_format
from utils.metrics import PPSeqMetrics
from tqdm import trange

cellid = 0
runid = 0
Ts = 18137
W = 200

grid = pd.read_csv('grid.csv')

ACC = []
for runid in range(12):
    for cellid in trange(81, bar_format=bar_format):
        try:
            d = grid.iloc[cellid].to_dict()
            # print(d)
            _, cellid_, p_drop, isi, seqlen, jitter = d.values()
            assert (cellid == cellid_)

            pred_events = np.load(f'artifacts/events_{cellid}_{runid}.npy')
            GT = np.load(f'datasets/gt_{cellid}.npy')

            metrics = PPSeqMetrics(GT, Ts, W)
            acc = metrics.get(pred_events)

            acc['runid'] = runid
            acc['cellid'] = cellid

            acc['p_drop'] = p_drop
            acc['gap_ts'] = isi
            acc['seqlen'] = seqlen
            acc['jitter_std'] = jitter

            ACC.append(acc)
        except Exception as e:
            print(f'Exception in cellid: {cellid}, runid: {runid}')

pd.DataFrame(ACC).to_csv('accuracy_performance.csv')
