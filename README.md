# BrainRoiGNN

Code for *BrainRoiGNN: An Anatomy-aware Explainable Graph Neural Network for
Resting-state EEG in Parkinson's Disease* (Scientific Reports): preprocessing, feature
construction, the model and nested cross-validation training.

Not included: trained weights, baseline models, participant-exclusion logic (excluded
participants are listed in Supplementary Table S1), evaluation and
statistics code, attribution code and saved predictions.

## Data

OpenNeuro **ds007020, version 1.0.0** (not included).

## Installation

```
conda env create -f environment.yml
conda activate brainroignn
```

Training requires an NVIDIA GPU with CUDA.

## Usage

Run from the repository root; settings are in `config.yaml`. `--exclude` takes the
participant IDs excluded in Supplementary Table S1.

```
python -m preprocessing.preprocess --bids-root <ds007020> --out-dir <processed_dir>
python -m features.build_features --processed-dir <processed_dir> --out-file <features.npz> --exclude <IDs>
python -m training.train --features <features.npz> --out-dir <out_dir>
```

Trained weights are not distributed; the model is obtained by retraining. Random seeds
are set in the code.
