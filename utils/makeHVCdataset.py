import pandas as pd
import numpy as np
import scipy.io as sio

df = pd.read_csv('../baselines/PPSeq.jl/demo/data/songbird_spikes.txt', sep='\t', header=None)
df.columns = ['neuronid', 'spiketime']
df.neuronid = df.neuronid.astype(int)
df.spiketime = (df.spiketime * 1000).astype(int)

X = np.zeros((df.neuronid.max() + 1, df.spiketime.max() + 1))
for nid, spt in zip(df.neuronid, df.spiketime):
    X[nid, spt] = 1

np.save('../datasets/HVC', X)
sio.savemat(f"../baselines/seqNMF/HVC.mat", {'X_': X_})
