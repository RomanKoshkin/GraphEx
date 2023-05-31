# Unsupervised Detection of Cell Assemblies with Graph Neural Networks

- [paper]()

# Reproducibility

To get started, run notebooks in `notebooks`.

# Miscellaneous

### AUC curves

`sh perf_bench.sh` to generate data needed to plot the AUC curves from Fig. 1.

### Generate data for Fig 3 and 4

```bash
cd scripts
python boot_one_seq.py
```

### Prepare HVC data for _seqNMF_ and our method

```bash
cd utils
python makeHVCdataset.py
```

### Videos

The `Trainer` object saves frames to `data`. If you want to see the optimization progress, make a video from the frames:

```bash
cd scripts
sh make_vids2.sh
```

### Baselines

All you need is in `cd baselines`.

Make `grid.csv` and datasets with:

- _seqNMF_

```bash
cd baselines/seqNMF
python make_grid_seqNMF.py
python make_datasets_seqNMF.py
```

- _PP-Seq_

```bash
cd baselines/PPSeq.jl/demo/acc_bench
python make_grid.py
python generate_datasets.py
python run_grid.py
python evaluate.py
```
