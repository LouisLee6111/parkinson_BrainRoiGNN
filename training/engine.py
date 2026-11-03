"""Nested cross-validation and training for BrainRoiGNN.

Outer loop: subject-wise stratified 5-fold cross-validation, repeated with seeds 0, 1, 2.
Per outer fold (the outer test fold is used only for the final prediction):
  1. split the outer-training subjects into a training core and an inner holdout
     (subject-wise, stratified, 18 %);
  2. for every hyperparameter combination in the grid: fit normalisation on the core,
     train with early stopping on the holdout subject-level balanced accuracy
     (patience 20, max 60 epochs); score the best state by holdout subject-level AUC;
  3. keep the best combination (ties -> earlier grid entry) and its best epoch E_k;
  4. refit normalisation on the full outer-training set and train a fresh model for
     exactly E_k epochs; predict the outer test fold;
  5. decision threshold: 4-fold cross-validation over the outer-training subjects with
     the chosen configuration and E_k; the threshold maximising balanced accuracy on
     the pooled out-of-fold subject probabilities (ties -> closest to 0.5).

Training: AdamW, class-weighted cross-entropy, class-balanced weighted random sampling
of epochs, cosine annealing over max_epochs, gradient-norm clipping at 1.0.
Every complete 60-node graph shares one edge index, so batches are assembled by
indexing GPU tensors; the optimisation step is captured in a CUDA graph.
All random seeds are set explicitly (see train_model and run_fold).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features.transforms import center_edges, edge_zscore_stats, node_zscore
from model.brainroignn import BrainRoiGNN, channel_to_roi


@dataclass
class TrainConfig:
    seeds: List[int]
    n_folds: int
    inner_holdout_fraction: float
    threshold_inner_folds: int
    hidden: int
    gine_layers: int
    heads: int
    dropout: float
    weight_decay: float
    max_epochs: int
    patience: int
    batch_size: int
    grid: List[Dict]
    lr: float = 3e-4
    std_node: bool = False


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
class Dataset:
    """Features of all retained epochs (CPU), edges subject-wise mean-centred."""

    def __init__(self, features_file: str):
        d = np.load(features_file, allow_pickle=True)
        self.subj = d["subject_ids"].astype(str)
        self.y = d["y"].astype(np.int64)
        self.node = d["node"].astype(np.float32)                     # (N,60,10)
        self.edge_up = center_edges(d["edge"], self.subj)            # (N,1770,10)
        self.ch_names = d["ch_names"].astype(str).tolist()
        self.iu = np.triu_indices(len(self.ch_names), k=1)


class GPUData:
    """Normalised tensors for one normalisation fit, resident on the GPU."""

    def __init__(self, ds: Dataset, fit_idx: np.ndarray, device, std_node: bool = False):
        mu, sd = edge_zscore_stats(ds.edge_up, fit_idx)
        self.E = torch.from_numpy((ds.edge_up - mu) / sd).to(device)
        node = node_zscore(ds.node, fit_idx) if std_node else ds.node
        self.X = torch.from_numpy(node.astype(np.float32)).to(device)
        self.Y = torch.from_numpy(ds.y).to(device)
        iu = torch.as_tensor(np.stack(ds.iu), device=device)
        self.base_ei = torch.cat([iu, iu.flip(0)], dim=1)            # both directions
        self.device = device


class Batch:
    __slots__ = ("x", "edge_index", "edge_attr", "batch", "num_graphs", "static_num_nodes")


def make_batch(g: GPUData, x: torch.Tensor, e_up: torch.Tensor) -> Batch:
    """x (B,60,F), e_up (B,1770,Fe) -> batched complete graphs."""
    B, n = x.shape[0], x.shape[1]
    off = (torch.arange(B, device=x.device) * n).view(B, 1, 1)
    b = Batch()
    b.x = x.reshape(-1, x.shape[-1])
    b.edge_index = (g.base_ei.unsqueeze(0) + off).permute(1, 0, 2).reshape(2, -1)
    b.edge_attr = torch.cat([e_up, e_up], dim=1).reshape(-1, e_up.shape[-1])
    b.batch = torch.arange(B, device=x.device).repeat_interleave(n)
    b.num_graphs = B
    b.static_num_nodes = n
    return b


def subject_stratified_kfold(subj: np.ndarray, y: np.ndarray, n_folds: int, seed: int):
    """Stratified k-fold over subjects -> list of (train_idx, test_idx, train_subjects)."""
    subjects = np.array(sorted(np.unique(subj)))
    labels = np.array([int(y[subj == s][0]) for s in subjects])
    folds = []
    for tr, te in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(subjects, labels):
        folds.append((np.flatnonzero(np.isin(subj, subjects[tr])), np.flatnonzero(np.isin(subj, subjects[te])),
                      subjects[tr].tolist()))
    return folds


def subject_stratified_holdout(subjects: np.ndarray, labels: np.ndarray, frac: float, rng):
    """Split subjects into (core, holdout), stratified by label; the holdout keeps at
    least one subject per class (two when the class has >= 8 subjects)."""
    holdout = []
    for cls in np.unique(labels):
        cls_subs = subjects[labels == cls]
        n_hold = max(1, int(round(frac * len(cls_subs))))
        if len(cls_subs) >= 8:
            n_hold = max(2, n_hold)
        n_hold = min(n_hold, len(cls_subs) - 1)
        holdout.extend(rng.permutation(cls_subs)[:n_hold].tolist())
    holdout = set(holdout)
    return np.array([s for s in subjects.tolist() if s not in holdout]), np.array(sorted(holdout))


# ----------------------------------------------------------------------------
# model / training / prediction
# ----------------------------------------------------------------------------
def build_model(cfg: TrainConfig, ds: Dataset, device) -> BrainRoiGNN:
    return BrainRoiGNN(ds.node.shape[2], ds.edge_up.shape[2], ch_to_roi=channel_to_roi(ds.ch_names),
                       hidden_dim=cfg.hidden, num_blocks=cfg.gine_layers, num_heads=cfg.heads,
                       dropout=cfg.dropout).to(device)


@torch.no_grad()
def predict(model, g: GPUData, idx: np.ndarray, bs: int = 512) -> np.ndarray:
    """P(living) per epoch."""
    model.eval()
    out = []
    it = torch.as_tensor(idx, device=g.device)
    for i in range(0, len(it), bs):
        j = it[i:i + bs]
        out.append(torch.softmax(model(make_batch(g, g.X[j], g.E[j])), 1)[:, 1].float().cpu().numpy())
    return np.concatenate(out)


def subject_probs(p_ep: np.ndarray, subj: np.ndarray):
    subs = np.array(sorted(np.unique(subj)))
    return subs, np.array([p_ep[subj == s].mean() for s in subs])


def subject_labels(ds: Dataset, subs) -> np.ndarray:
    return np.array([int(ds.y[ds.subj == s][0]) for s in subs])


def balanced_accuracy(p_liv: np.ndarray, y: np.ndarray, t: float = 0.5) -> float:
    """Deceased (y = 0) is predicted when P(living) < t."""
    pred_d = p_liv < t
    sens = (pred_d & (y == 0)).sum() / max((y == 0).sum(), 1)
    spec = (~pred_d & (y == 1)).sum() / max((y == 1).sum(), 1)
    return float(0.5 * (sens + spec))


def train_model(cfg: TrainConfig, ds: Dataset, g: GPUData, tr_idx: np.ndarray, n_epochs: int,
                device, torch_seed: int, monitor_idx: Optional[np.ndarray] = None):
    """Train a fresh model for n_epochs. With monitor_idx, early stopping on its
    subject-level balanced accuracy and the best state is restored."""
    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)
    model = build_model(cfg, ds, device)
    y_tr = ds.y[tr_idx]
    counts = np.maximum(np.bincount(y_tr, minlength=2).astype(np.float64), 1.0)
    samp_w = torch.as_tensor((1.0 / counts)[y_tr], dtype=torch.double)   # class-balanced sampling
    cw = 1.0 / counts
    cw = cw * (len(counts) / cw.sum())
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=torch.tensor(cfg.lr, device=device),
                            weight_decay=cfg.weight_decay, fused=True, capturable=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    gen = torch.Generator().manual_seed(torch_seed)
    tr_t = torch.as_tensor(tr_idx, device=device)
    bs = cfg.batch_size
    static_j = torch.zeros(bs, dtype=torch.long, device=device)

    def step(j_local):
        opt.zero_grad(set_to_none=True)
        j = tr_t[j_local]
        loss = crit(model(make_batch(g, g.X[j], g.E[j])), g.Y[j])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, foreach=True)
        opt.step()

    graph = None
    best = dict(bacc=-1.0, state=None, epoch=-1)
    no_imp, last = 0, 0
    for ep in range(n_epochs):
        last = ep + 1
        model.train()
        # weighted random sampling with replacement, one pass = len(tr_idx) epochs
        order = torch.multinomial(samp_w, len(tr_idx), replacement=True, generator=gen).to(device)
        n_full = len(order) // bs
        for b in range(n_full):
            static_j.copy_(order[b * bs:(b + 1) * bs])
            if graph is None and b < 3:
                # three warm-up steps (real optimisation steps) on a side stream, then capture
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    step(static_j)
                torch.cuda.current_stream().wait_stream(s)
                if b == 2:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        step(static_j)
                continue
            graph.replay()
        rem = order[n_full * bs:]
        if len(rem) >= 2:               # BatchNorm needs more than one graph
            step(rem)
        sched.step()
        if monitor_idx is None:
            continue
        subs, ps = subject_probs(predict(model, g, monitor_idx), ds.subj[monitor_idx])
        bacc = balanced_accuracy(ps, subject_labels(ds, subs))
        if bacc > best["bacc"]:
            best.update(bacc=bacc, epoch=ep + 1,
                        state={k: v.detach().clone() for k, v in model.state_dict().items()})
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= cfg.patience:
            break
    info = dict(epochs_run=last)
    if monitor_idx is not None:
        model.load_state_dict(best["state"])
        info.update(best_epoch=int(best["epoch"]), monitor_bacc=float(best["bacc"]))
    model.eval()
    return model, info


def holdout_auc(model, g: GPUData, ds: Dataset, idx: np.ndarray) -> float:
    subs, ps = subject_probs(predict(model, g, idx), ds.subj[idx])
    ys = subject_labels(ds, subs)
    if len(np.unique(ys)) < 2:
        return float("nan")
    return float(roc_auc_score((ys == 0).astype(int), 1 - ps))


def best_threshold(p_liv: np.ndarray, y: np.ndarray) -> float:
    """Threshold on P(living) maximising balanced accuracy; candidates are midpoints of
    the sorted unique probabilities; ties -> the candidate closest to 0.5."""
    u = np.unique(p_liv)
    cands = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])
    sc = np.array([balanced_accuracy(p_liv, y, t) for t in cands])
    best = np.flatnonzero(sc >= sc.max() - 1e-12)
    return float(cands[best[np.argmin(np.abs(cands[best] - 0.5))]])


def inner_cv_threshold(cfg: TrainConfig, ds: Dataset, tr_idx, tr_subs, n_epochs: int, device,
                       tseed: int) -> float:
    tr_subs = np.asarray(tr_subs)
    skf = StratifiedKFold(cfg.threshold_inner_folds, shuffle=True, random_state=tseed % (2 ** 31))
    ps, ys = [], []
    for k, (a, b) in enumerate(skf.split(tr_subs, subject_labels(ds, tr_subs))):
        ia = tr_idx[np.isin(ds.subj[tr_idx], tr_subs[a])]
        ib = tr_idx[np.isin(ds.subj[tr_idx], tr_subs[b])]
        g = GPUData(ds, ia, device, std_node=cfg.std_node)
        m, _ = train_model(cfg, ds, g, ia, n_epochs, device, tseed + 10 * k)
        s_, p_ = subject_probs(predict(m, g, ib), ds.subj[ib])
        ps.append(p_)
        ys.append(subject_labels(ds, s_))
        del m, g
    return best_threshold(np.concatenate(ps), np.concatenate(ys))


def run_fold(cfg: TrainConfig, ds: Dataset, seed: int, fi: int, tr_idx: np.ndarray, te_idx: np.ndarray,
             tr_subs, device):
    """One outer fold (fi starts at 1). Returns (P(living) for te_idx, info)."""
    tseed = 7919 * seed + 104729 * fi
    tr_subs_arr = np.asarray(tr_subs)
    rng = np.random.default_rng(100000 + seed * 1000 + fi)
    core_subs, es_subs = subject_stratified_holdout(tr_subs_arr, subject_labels(ds, tr_subs_arr),
                                                    cfg.inner_holdout_fraction, rng)
    core_idx = tr_idx[np.isin(ds.subj[tr_idx], core_subs)]
    es_idx = tr_idx[np.isin(ds.subj[tr_idx], es_subs)]
    cand = []
    for ci, combo in enumerate(cfg.grid):
        ccfg = replace(cfg, **combo)
        g_core = GPUData(ds, core_idx, device, std_node=ccfg.std_node)
        m_es, inf_es = train_model(ccfg, ds, g_core, core_idx, ccfg.max_epochs, device, tseed,
                                   monitor_idx=es_idx)
        cand.append((holdout_auc(m_es, g_core, ds, es_idx), -ci, ccfg, inf_es))
        del m_es, g_core
    auc, _, ccfg, inf_es = max(cand, key=lambda c: (np.nan_to_num(c[0], nan=-1.0), c[1]))
    e_k = int(inf_es["best_epoch"])
    g = GPUData(ds, tr_idx, device, std_node=ccfg.std_node)
    model, _ = train_model(ccfg, ds, g, tr_idx, e_k, device, tseed + 1)
    p_test = predict(model, g, te_idx)
    del model, g
    thr = inner_cv_threshold(ccfg, ds, tr_idx, tr_subs, e_k, device, tseed + 2)
    info = dict(seed=seed, fold=fi, chosen=dict(lr=ccfg.lr, std_node=ccfg.std_node), final_epochs=e_k,
                inner_holdout_auc=float(auc), threshold=thr)
    return p_test, info
