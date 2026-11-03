"""Preprocessing of the raw BrainVision recordings (OpenNeuro ds007020).

Per subject:
  1. keep the EEG channels common to all recordings (sorted by name; no interpolation),
  2. band-pass 1-45 Hz with a zero-phase FIR filter (MNE 'firwin' defaults),
  3. re-reference to the common average,
  4. crop the first and last 10 % of the recording,
  5. cut fixed-length events every 2.1 s and keep the first 2.0 s of each
     (non-overlapping 2-s epochs, 1,000 samples at 500 Hz),
  6. reject epochs whose peak-to-peak amplitude exceeds 150 uV on any channel.

Output: one file per subject, <out_dir>/sub-<ID>_processed.npz with
X (epochs, channels, samples) in volts, y (0 = deceased, 1 = living), sfreq, ch_names.

Run:
    python -m preprocessing.preprocess --bids-root <ds007020> --out-dir <processed_dir>
"""
from __future__ import annotations

import argparse
import os

import mne
import numpy as np
import pandas as pd
import yaml
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids

LABELS = {"deceased": 0, "living": 1}


def bids_path(bids_root: str, subject: str):
    paths = BIDSPath(subject=subject, datatype="eeg", suffix="eeg", extension=".vhdr", root=bids_root).match()
    if not paths:
        paths = BIDSPath(subject=subject, datatype="eeg", suffix="eeg", root=bids_root).match()
    return paths[0] if paths else None


def common_channels(bids_root: str, subjects) -> list:
    common = None
    for s in subjects:
        p = bids_path(bids_root, s)
        if p is None:
            continue
        names = set(read_raw_bids(bids_path=p, verbose=False).ch_names)
        common = names if common is None else common & names
    return sorted(common)


def preprocess_subject(bids_root: str, subject: str, channels: list, pc: dict):
    raw = read_raw_bids(bids_path=bids_path(bids_root, subject), verbose=False)
    raw.pick_channels(channels)
    raw.reorder_channels(channels)
    raw.load_data(verbose=False)
    sfreq = raw.info["sfreq"]
    raw.filter(l_freq=pc["l_freq"], h_freq=pc["h_freq"], fir_design="firwin", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    t_end = raw.times[-1]
    raw.crop(tmin=t_end * pc["crop_fraction"], tmax=t_end * (1 - pc["crop_fraction"]))
    events = mne.make_fixed_length_events(raw, duration=pc["event_spacing_s"], overlap=0.0)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=pc["event_spacing_s"], baseline=None,
                        preload=True, verbose=False)
    data = epochs.get_data()[:, :, :int(pc["epoch_length_s"] * sfreq)]
    ptp = data.max(axis=2) - data.min(axis=2)
    data = data[ptp.max(axis=1) <= pc["ptp_reject_uV"] * 1e-6]
    return data, sfreq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", required=True, help="root of OpenNeuro ds007020 (contains participants.tsv)")
    ap.add_argument("--out-dir", required=True, help="directory for the preprocessed files")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    root, out_dir = a.bids_root, a.out_dir
    pc = yaml.safe_load(open(a.config, encoding="utf-8"))["preprocessing"]
    os.makedirs(out_dir, exist_ok=True)

    parts = pd.read_csv(os.path.join(root, "participants.tsv"), sep="\t")
    labels = {str(r.participant_id).replace("sub-", ""): LABELS[str(r.survival_status).strip().lower()]
              for r in parts.itertuples()}
    subjects = get_entity_vals(root, "subject")
    channels = common_channels(root, subjects)
    print(f"{len(channels)} common channels")

    for s in subjects:
        if s not in labels or bids_path(root, s) is None:
            continue
        data, sfreq = preprocess_subject(root, s, channels, pc)
        print(f"sub-{s}: {data.shape[0]} epochs kept")
        if data.shape[0] == 0:
            continue
        np.savez_compressed(os.path.join(out_dir, f"sub-{s}_processed.npz"), X=data,
                            y=np.full(data.shape[0], labels[s]), subject_ids=np.full(data.shape[0], s),
                            sfreq=sfreq, ch_names=channels)


if __name__ == "__main__":
    main()
