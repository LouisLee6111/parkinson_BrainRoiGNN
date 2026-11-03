"""Graph features for every retained epoch.

Node features: log10 relative band power per channel. Welch PSD (Hann window,
500 samples, 50 % overlap -> 1-Hz bins); band power = sum of PSD bins in [lo, hi)
times the bin width (the last band includes its upper edge); relative to the sum
over the ten bands.

Edge features: weighted Phase Lag Index per channel pair and band. Each 2-s epoch is
band-pass filtered (order-4 zero-phase Butterworth), Hilbert-transformed, and
    wPLI = |mean_t Im(a_i a_j*)| / mean_t |Im(a_i a_j*)|
over the 1,000 samples (denominator guarded against exact zero only).
Stored as the upper triangle, shape (epochs, 1770, bands).

Output: <out_file> (.npz) with node (E,60,10), edge (E,1770,10), y, subject_ids,
ch_names, in sorted subject order and the epoch order of the processed files.

Run:
    python -m features.build_features --processed-dir <processed_dir> --out-file <features.npz> \
        --exclude <IDs from Supplementary Table S1>
"""
from __future__ import annotations

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import yaml
from scipy.signal import butter, hilbert, sosfiltfilt, welch


def rel_logpower(X: np.ndarray, sfreq: float, bands, nperseg: int) -> np.ndarray:
    """X (E,C,T) -> (E,C,B) log10 relative band power."""
    f, P = welch(X.astype(np.float64), fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    df = f[1] - f[0]
    bp = []
    for i, (lo, hi) in enumerate(bands):
        m = (f >= lo) & ((f <= hi) if i == len(bands) - 1 else (f < hi))
        bp.append(P[..., m].sum(-1) * df)
    bp = np.stack(bp, -1)
    return np.log10(np.maximum(bp / bp.sum(-1, keepdims=True), 1e-30)).astype(np.float32)


def wpli_epoch(x: np.ndarray, sfreq: float, bands, order: int) -> np.ndarray:
    """x (C,T) -> (B,C,C) wPLI in [0, 1], zero diagonal."""
    out = np.zeros((len(bands), x.shape[0], x.shape[0]), np.float32)
    nyq = sfreq / 2
    for b, (lo, hi) in enumerate(bands):
        sos = butter(order, [lo / nyq, min(hi, nyq - 1e-6) / nyq], btype="bandpass", output="sos")
        a = hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)
        a = a / np.abs(a).max()                      # global rescale; wPLI is scale invariant
        im = (a[:, None, :] * np.conj(a[None, :, :])).imag
        num = np.abs(im.mean(-1))
        den = np.abs(im).mean(-1)
        w = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        np.fill_diagonal(w, 0.0)
        out[b] = w
    return out


def subject_features(path: str, bands, nperseg: int, order: int):
    d = np.load(path, allow_pickle=True)
    sf = float(np.asarray(d["sfreq"]).reshape(-1)[0])
    X = d["X"].astype(np.float64)
    iu = np.triu_indices(X.shape[1], k=1)
    edge = np.stack([wpli_epoch(e, sf, bands, order) for e in X])          # (E,B,C,C)
    edge_up = np.ascontiguousarray(edge[:, :, iu[0], iu[1]].transpose(0, 2, 1))
    return (rel_logpower(X, sf, bands, nperseg), edge_up, d["y"].astype(np.int64),
            d["subject_ids"].astype(str), d["ch_names"].astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, help="output directory of preprocessing")
    ap.add_argument("--out-file", required=True, help="feature file to write (.npz)")
    ap.add_argument("--exclude", nargs="*", default=[], help="participant IDs to leave out (Supplementary Table S1)")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    fc, excl = yaml.safe_load(open(a.config, encoding="utf-8"))["features"], set(a.exclude)
    files = sorted(glob.glob(os.path.join(a.processed_dir, "sub-*_processed.npz")))
    files = [f for f in files if os.path.basename(f)[4:-14] not in excl]
    bands = [tuple(b) for b in fc["bands"]]
    fn = partial(subject_features, bands=bands, nperseg=fc["welch_nperseg"], order=fc["butter_order"])
    with ProcessPoolExecutor(max_workers=fc["n_workers"]) as ex:
        res = list(ex.map(fn, files))
    out = a.out_file
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, node=np.concatenate([r[0] for r in res]), edge=np.concatenate([r[1] for r in res]),
             y=np.concatenate([r[2] for r in res]), subject_ids=np.concatenate([r[3] for r in res]),
             ch_names=res[0][4], bands=np.array(bands))
    print(f"{len(res)} subjects, {sum(len(r[2]) for r in res)} epochs -> {out}")


if __name__ == "__main__":
    main()
