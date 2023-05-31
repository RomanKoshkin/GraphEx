from datetime import datetime
import time, os, pickle
import subprocess, datetime
import pandas as pd

cellid = 0
d = []
for p_drop in [0.2, 0.3, 0.4]:
    for isi in [400, 600, 800]:
        for seqlen in [60, 80, 100]:
            for jitter in [10, 20, 30]:
                d.append({
                    'cellid': cellid,
                    'p_drop': p_drop,
                    'isi': isi,
                    'seqlen': seqlen,
                    'jitter': jitter,
                })
                cellid += 1

pd.DataFrame(d).to_csv('grid.csv')