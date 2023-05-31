import matplotlib
import matplotlib.pyplot as plt
import torch, time, sys, os, pickle

sys.path.append('../../../../')

import pandas as pd

#++++++++++++++++++++
from datetime import datetime
from this import d
import time, os
import subprocess, datetime

started = []
done = []

TIMES = dict()


def check_procs():
    global PROC, TIMES
    running_procs = 0
    for runid, proc in enumerate(PROC):
        if proc.poll() != 0:
            running_procs += 1
        else:
            pass
    return running_procs


PROC = []

runid = int(sys.argv[1])
grid = pd.read_csv('grid.csv')

for k in range(grid.shape[0]):
    p_drop = float(grid.iloc[k].p_drop)
    gap_ts = int(grid.iloc[k].isi)
    seqlen = int(grid.iloc[k].seqlen)
    jitter_std = float(grid.iloc[k].jitter)
    cellid = int(grid.iloc[k].cellid)

    time.sleep(2)

    PROC.append(subprocess.Popen([
        'julia',
        "--project=../../.",
        '../accuracy_bench.jl',
        f'{cellid}',
        f'{runid}',
    ]))

    print(f'{cellid} started at {datetime.datetime.now()}')

    # pause if more then 5 processes are still running
    while check_procs() > 5:
        time.sleep(3)
