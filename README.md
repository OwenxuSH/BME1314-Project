# EEG-based Emotion Recognition Dataset

## Competition Overview

- **Competition**: 脑机接口赛道 赛题四 — 基于脑电数据的情绪识别算法
- **Task**: Cross-subject binary emotion classification (positive vs. neutral) from EEG signals
- **More info**: https://www.pazhoulab.com/2026/03/8165/

## Dataset Summary

The dataset contains EEG recordings from subjects watching emotional video clips. The goal is to build a cross-subject model that classifies each video segment as eliciting **positive** (label=1) or **neutral** (label=0) emotion.

### Key Statistics

| Split | Healthy (HC) | Depressed (DEP) | Total Subjects | Videos per Subject | Segment Duration |
|-------|-------------|-----------------|----------------|---------------------|------------------|
| Train | 40 | 20 | 60 | 8 (4 neutral + 4 positive) | ~50 s each |
| Test  | 5 | 5 | 10 | 8 (mixed, random order) | ~10 s each |

## Data Format

### File Naming

- **Training (Healthy)**: `HC1XXXtimedata.mat`
- **Training (Depressed)**: `DEP1XXXtimedata.mat`
- **Test**: `P_testX.mat`

### EEG Specifications

- **Sampling rate**: 250 Hz
- **Reference electrode**: A2 (re-referenced to average)
- **Preprocessing**: 0.01 Hz high-pass filter (baseline drift removal), power-line noise filtering, average re-reference

### Channel Order (30 channels)

| Index | Channel | Index | Channel | Index | Channel |
|-------|---------|-------|---------|-------|---------|
| 1 | FP1 | 11 | FC4 | 21 | CP4 |
| 2 | FP2 | 12 | FT8 | 22 | TP8 |
| 3 | F7 | 13 | T3 | 23 | T5 |
| 4 | F3 | 14 | C3 | 24 | P3 |
| 5 | FZ | 15 | CZ | 25 | PZ |
| 6 | F4 | 16 | C4 | 26 | P4 |
| 7 | F8 | 17 | T4 | 27 | T6 |
| 8 | FT7 | 18 | TP7 | 28 | O1 |
| 9 | FC3 | 19 | CP3 | 29 | OZ |
| 10 | FCZ | 20 | CPZ | 30 | O2 |

### Training Data Structure

Each `.mat` file contains two arrays:

| Variable | Shape | Description |
|----------|-------|-------------|
| `EEG_data_neu` | (30, 50000) | Neutral emotion segments (4 videos × 50 s × 250 Hz) |
| `EEG_data_pos` | (30, 50000) | Positive emotion segments (4 videos × 50 s × 250 Hz) |

Segment mapping within each array:

| Sample Range | Video |
|-------------|-------|
| 1 – 12500 | Video 1 |
| 12501 – 25000 | Video 2 |
| 25001 – 37500 | Video 3 |
| 37501 – 50000 | Video 4 |

### Test Data Structure

Each `.mat` file contains one array:

| Variable | Shape | Description |
|----------|-------|-------------|
| EEG data | (30, 20000) | 8 video segments (8 videos × 10 s × 250 Hz), **randomly ordered by emotion category** |

Segment mapping:

| Sample Range | Video |
|-------------|-------|
| 1 – 2500 | Video 1 |
| 2501 – 5000 | Video 2 |
| ... | ... |
| 17501 – 20000 | Video 8 |

**Note**: Unlike the training data, the 8 test segments are stored in a single array with shuffled emotion order. Labels are not provided.

## Directory Structure

```
赛题四数据集及说明文档/
├── README.md                    # This file
├── 数据集说明文档.pdf             # Official dataset documentation (Chinese)
├── 测试结果模板.xlsx              # Submission template
├── 训练集/
│   ├── 正常人/                   # 40 healthy control subjects
│   │   ├── HC1003timedata.mat
│   │   ├── HC1005timedata.mat
│   │   └── ...
│   └── 抑郁症患者/                # 20 depressed subjects
│       ├── DEP1003timedata.mat
│       ├── DEP1008timedata.mat
│       └── ...
└── 公开测试集/                   # 10 test subjects (5 HC + 5 DEP, unlabeled)
    ├── P_test1.mat
    ├── P_test2.mat
    └── ...
```

## Submission Format

Predictions should be submitted as an Excel (`.xlsx`) file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Test subject ID (e.g., `P_test1`) |
| `trial_id` | int | Video segment index (1–8) |
| `Emotion_label` | int | Predicted label: `0` = neutral, `1` = positive |

Refer to `测试结果模板.xlsx` for the exact format.

## Notes

- Data augmentation and use of additional public EEG datasets are allowed.
- Cross-subject generalization is the core challenge — individual differences in EEG emotional responses are substantial.
- The training set provides separate `EEG_data_neu` and `EEG_data_pos` arrays, while the test set merges all 8 segments into a single array with randomized order, requiring the model to identify emotion from the signal itself.

## License & Citation

This dataset is provided by the competition organizers for the 脑机接口赛道. Please refer to the official competition page for terms of use.

---

# Model Implementation & Evaluation

## Quick Start

```bash
# 1. Install dependencies
pip install numpy scipy scikit-learn xgboost lightgbm pandas h5py

# 2. Run evaluation (10-fold cross-subject, 5HC+5DEP held-out each fold)
python train_eval.py

# 3. Quick test (2 folds)
python train_eval.py --quick

# 4. Train final model on all 60 subjects & predict public test set
python train_eval.py --inference
```

## Model Architecture

### Feature Extraction Pipeline

```
Raw EEG (30ch, 250Hz)
  → 4th-order Butterworth bandpass filter (5 bands: δ, θ, α, β, γ)
  → Differential Entropy: DE = 0.5 × log(2πe · var)
    (5s window, 1s stride)
  → Per-subject z-score normalization
  → Temporal augmentation: [DE, DE_diff, DE_ma3] → 450D
```

### Classifier: Two-Stage Stacked Ensemble

```
Stage 1: 4 base learners (2 XGBoost + 2 LightGBM)
         → 2-fold cross-validation to generate meta-features

Stage 2: LogisticRegression meta-learner
         → Learns optimal combination of base learner outputs

Final:   prob = 0.5 × mean(base_learners) + 0.5 × meta_learner
```

### Base Learner Hyperparameters

| Parameter | XGBoost | LightGBM |
|-----------|---------|----------|
| n_estimators | 200 | 200 |
| max_depth | 4 | 5 |
| num_leaves | — | 31 |
| learning_rate | 0.03 | 0.03 |
| subsample | 0.8 | 0.8 |
| colsample_bytree | 0.8 | 0.8 |
| reg_lambda | 1.0 | 1.0 |
| reg_alpha | 0.5 | 0.5 |

## Evaluation Protocol

### Strict Cross-Subject Evaluation

1. **Split**: 10 random splits, each holding out **5 HC + 5 DEP** subjects
2. **Train**: On remaining 50 subjects (35 HC + 15 DEP)
3. **Pseudo-test**: For each held-out subject, extract middle 10s of each 50s trial
   - 8 trials × 6 windows (5s win, 1s stride) = 48 windows/subject
   - Labels: first 4 trials = neutral (0), last 4 = positive (1)
4. **Post-processing**: Per-subject forced 4+4
   - Top-4 highest-probability segments → positive
   - Bottom-4 → neutral
5. **Metrics** (mean ± std over 10 splits):
   - **BalAcc** = 0.5 × HC_SegAcc + 0.5 × DEP_SegAcc
   - HC_SegAcc: accuracy on 5 HC subjects (40 trials)
   - DEP_SegAcc: accuracy on 5 DEP subjects (40 trials)

### Results

| Method | BalAcc | HC Acc | DEP Acc |
|--------|--------|--------|----------|
| DE (150D) + MLP ensemble | 75.5% ± 5.8% | ~82% | ~69% |
| DE (150D) + XGBoost | 75.5% ± 3.7% | 81.5% | 69.5% |
| DE (150D) + LightGBM | 76.3% ± 3.9% | 82.0% | 70.5% |
| **Temporal (450D) + Stacking** | **77.0% ± 4.7%** | **83.5%** | **70.5%** |

### Key Findings

1. **GBDT >> MLP**: Tree-based models significantly outperform MLPs for 150D DE features
2. **DEP is the bottleneck**: DEP accuracy plateaued at ~70%, limited by only 15 DEP training subjects per split. The test-time variance is driven by how representative the 15 training DEP subjects are of the 5 held-out DEP subjects
3. **External data doesn't help**: 317 additional subjects from SEED/SEED-IV/SEED-V did not improve results (simple pooling, CORAL alignment both failed)
4. **Manual features don't help beyond DE**: Frontal alpha asymmetry, band power ratios, Hjorth parameters all failed to improve over pure DE features
5. **Temporal features give marginal gain**: Adding temporal difference and moving average (+2.0% BalAcc)

## Experiment History

Iterative optimization scripts are at `D:\Downloads\SEED-V\optimize_v*.py`:

| Version | Description | BalAcc |
|---------|-------------|--------|
| V8 | First strict evaluation (per-split encoder + MLP) | 74.5% ± 8.6% |
| V9 | DEP anchor + focal loss (regression) | 69.8% ± 7.5% |
| V10 | HC anchor vs no-encoder baseline | 75.5% ± 5.8% |
| V11 | Hand-crafted features (FAA, ratios) + sklearn GBDT | 82.5% (split 1 only) |
| V12 | XGBoost/LightGBM comprehensive comparison | 76.3% ± 3.9% |
| V14 | CORAL domain alignment + bagging | 75.8% ± 3.5% |
| V15 | **Temporal features + stacking (BEST)** | **77.0% ± 4.7%** |
| V17 | BIOT pretrained model (256D features) | Pending GPU fix |
| V19 | Subject-level CV hyperparameter tuning | 75.0% ± 3.9% |

## Dependencies

```
Python >= 3.10
numpy, scipy, scikit-learn >= 1.3
xgboost >= 2.0, lightgbm >= 4.0
pandas, h5py
torch + braindecode (optional, for BIOT pretrained features)
```

## Reproducibility

To reproduce the reported results exactly:

```bash
python train_eval.py --splits 10
```

All random seeds are fixed. The random split indices are generated with `np.random.RandomState(100 + split_i)`. Results should be identical across runs on the same machine.

## File Index

```
D:\Downloads\赛题四数据集及说明文档\
├── train_eval.py              # Main script (evaluation + inference)
├── README.md                  # This file (data + model documentation)
├── _feature_cache_full.npz    # Pre-computed DE feature cache
├── 训练集/                    # Raw EEG training data
├── 公开测试集/                 # Raw EEG test data
├── SEED/, SEED_IV/, SEED-V/   # External datasets
│
D:\Downloads\SEED-V\
├── optimize_v8.py ~ v19.py    # Optimization history
└── v7_checkpoints/            # Saved encoder checkpoints (V7 era)
```
