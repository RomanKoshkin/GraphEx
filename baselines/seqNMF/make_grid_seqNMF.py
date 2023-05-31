from datetime import datetime
import time, os, pickle
import subprocess, datetime
import pandas as pd

cellid = 0
d = []
for p_drop in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
    for isi in [450]:
        for seqlen in [100]:
            for jitter in [5, 10, 20, 30, 40, 50]:
                d.append({
                    'cellid': cellid,
                    'p_drop': p_drop,
                    'isi': isi,
                    'seqlen': seqlen,
                    'jitter': jitter,
                })
                cellid += 1

pd.DataFrame(d).to_csv('grid.csv')