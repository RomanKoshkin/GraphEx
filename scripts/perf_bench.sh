#!/bin/bash

cd ../notebooks
# for p_drop in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65
p_drop=0.2
# for jitter_std in 5 10 15 20 25 30
for jitter_std in 35 40 45 50
do
    for runid in 0 1 2 3 4 5 6 7 8 9
    do
        python AUC_bench.py $runid $p_drop $jitter_std
    done
done
