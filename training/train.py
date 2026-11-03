"""Train BrainRoiGNN with nested cross-validation (5 outer folds x 3 repetitions).

Writes to <out_dir>:
  oof_predictions.npz  P(living) for every epoch from each repetition (p_living, shape 3 x N),
                       outer fold of each epoch (fold), decision threshold of that fold
                       (threshold), labels (y; 0 = deceased, 1 = living), subject_ids
  fold_info.json       per outer fold: chosen hyperparameters, epoch count E_k,
                       inner-holdout AUC, decision threshold

Run:
    python -m training.train --features <features.npz> --out-dir <out_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from training.engine import Dataset, TrainConfig, run_fold, subject_stratified_kfold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="feature file written by features.build_features")
    ap.add_argument("--out-dir", required=True, help="directory for the out-of-fold predictions")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg_all = yaml.safe_load(open(a.config, encoding="utf-8"))
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required (the training step is captured in a CUDA graph).")
    device = torch.device("cuda")
    cfg = TrainConfig(**cfg_all["training"])
    ds = Dataset(a.features)
    out_dir = a.out_dir
    os.makedirs(out_dir, exist_ok=True)

    n = len(ds.y)
    p = np.full((len(cfg.seeds), n), np.nan, np.float32)
    fold = np.full((len(cfg.seeds), n), -1, np.int8)
    thr = np.full((len(cfg.seeds), n), 0.5, np.float32)
    infos, t0 = [], time.time()
    for si, seed in enumerate(cfg.seeds):
        for fi, (tr_idx, te_idx, tr_subs) in enumerate(subject_stratified_kfold(ds.subj, ds.y, cfg.n_folds, seed), 1):
            p_te, info = run_fold(cfg, ds, seed, fi, tr_idx, te_idx, tr_subs, device)
            p[si, te_idx], fold[si, te_idx], thr[si, te_idx] = p_te, fi, info["threshold"]
            infos.append(info)
            print(f"seed {seed} fold {fi}: {info}  ({time.time() - t0:.0f} s)", flush=True)
            torch.cuda.empty_cache()

    np.savez_compressed(os.path.join(out_dir, "oof_predictions.npz"), p_living=p, fold=fold,
                        threshold=thr, y=ds.y, subject_ids=ds.subj, seeds=np.array(cfg.seeds))
    with open(os.path.join(out_dir, "fold_info.json"), "w") as fh:
        json.dump(infos, fh, indent=1)


if __name__ == "__main__":
    main()
