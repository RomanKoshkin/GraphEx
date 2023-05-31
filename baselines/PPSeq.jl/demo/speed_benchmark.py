import time, sys

sys.path.append('../../../')

import pandas as pd
from datetime import datetime
from this import d
import subprocess, datetime

TIMES = []

PROC = []
for dataset_len in [4441, 8882, 13323, 17764, 22205, 26646, 100000, 500000]:

    time.sleep(2)

    PROC.append(subprocess.Popen([
        'julia',
        "--project=../.",
        'speed_benchmark.jl',
        f'{dataset_len}',
    ]))

    start_ts = time.time()
    time.sleep(3)
    print(f'{dataset_len} started at {datetime.datetime.now()}')

    # wait for the process to finish
    while PROC[-1].poll() != 0:
        time.sleep(3)

    finish_ts = time.time()
    print(f'{dataset_len} finished at {datetime.datetime.now()}')
    TIMES.append({
        'dataset_len': dataset_len,
        'start': start_ts,
        'end': finish_ts,
        'runtime': finish_ts - start_ts,
    })

    pd.DataFrame(TIMES).to_csv('PPseq_runtimes.csv')

print('exited')