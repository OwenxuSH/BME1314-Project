"""
EEG Emotion Recognition — Best Model (BalAcc=77.0% ± 4.7%)
=============================================================
Architecture:
  1. DE (Differential Entropy) features: 30 channels × 5 frequency bands = 150D
  2. Temporal augmentation: diff + moving average → 450D
  3. XGBoost + LightGBM base learners with 2-fold stacking
  4. LogisticRegression meta-learner
  5. Strict cross-subject evaluation: 10 splits, leave out 5 HC + 5 DEP each

Usage:
  python train_eval.py                    # Run full evaluation (10 splits)
  python train_eval.py --quick            # Quick test (2 splits)
  python train_eval.py --inference        # Train on all 60 subjects, predict test set
"""

import os, sys, time, gc, argparse, numpy as np
from scipy import signal as sig
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(DATA_DIR, "_feature_cache_full.npz")
TEST_DIR = os.path.join(DATA_DIR, "公开测试集")
FS = 250                      # Sampling rate (Hz)
WIN_SEC, STR_SEC = 5.0, 1.0   # Window length and stride (seconds)
WIN_SAMP = int(FS * WIN_SEC)   # 1250 samples
STR_SAMP = int(FS * STR_SEC)   # 250 samples
BANDS = [
    ("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 14),
    ("beta", 14, 31), ("gamma", 31, 50),
]
# 30-channel 10-20 montage
CHANNELS = [
    "FP1","FP2","F7","F3","FZ","F4","F8","FT7","FC3","FCZ","FC4","FT8",
    "T3","C3","CZ","C4","T4","TP7","CP3","CPZ","CP4","TP8",
    "T5","P3","PZ","P4","T6","O1","OZ","O2",
]

# Butterworth filter cache
_butter_cache = {}

# =============================================================================
# Feature Extraction
# =============================================================================
def _bandpass_filter(data, lo, hi):
    """4th-order Butterworth bandpass filter (cached)."""
    k = (lo, hi)
    if k not in _butter_cache:
        nyq = FS / 2
        _butter_cache[k] = sig.butter(4, [lo / nyq, hi / nyq], btype="band")
    b, a = _butter_cache[k]
    return sig.filtfilt(b, a, data, axis=-1)

def extract_de(eeg_segment, segment_length):
    """
    Extract Differential Entropy features from an EEG segment.

    DE = 0.5 * log(2 * pi * e * variance) per channel per frequency band.

    Parameters
    ----------
    eeg_segment : ndarray (30, N) — raw EEG for one trial/segment
    segment_length : int — number of samples in the segment

    Returns
    -------
    features : ndarray (n_windows, 150) — DE features
    """
    # Bandpass filter into 5 frequency bands
    filtered = [_bandpass_filter(eeg_segment, lo, hi) for _, lo, hi in BANDS]

    n_windows = (segment_length - WIN_SAMP) // STR_SAMP + 1
    features = np.zeros((n_windows, 150), dtype=np.float32)

    for i in range(n_windows):
        start = i * STR_SAMP
        col = 0
        for bi in range(len(BANDS)):
            window = filtered[bi][:, start:start + WIN_SAMP]
            variance = np.var(window, axis=1) + 1e-12
            features[i, col:col + 30] = 0.5 * np.log(2 * np.pi * np.e * variance)
            col += 30

    return features

def per_subject_norm(X):
    """Per-subject z-score normalization."""
    mu = X.mean(0, keepdims=True)
    sigma = X.std(0, keepdims=True) + 1e-8
    return (X - mu) / sigma

def add_temporal_features(de_features):
    """
    Augment DE features with temporal difference and moving average.

    Input:  (N, 150) DE features for one subject
    Output: (N, 450) = [DE, DE_diff, DE_ma3]
    """
    N, D = de_features.shape
    features = [de_features]

    # Temporal difference: delta[t] = x[t] - x[t-1]
    diff = np.zeros_like(de_features)
    diff[1:] = de_features[1:] - de_features[:-1]
    features.append(diff)

    # 3-point moving average
    ma3 = np.zeros_like(de_features)
    for i in range(N):
        lo, hi = max(0, i - 1), min(N, i + 2)
        ma3[i] = de_features[lo:hi].mean(0)
    features.append(ma3)

    return np.concatenate(features, 1).astype(np.float32)

# =============================================================================
# Data Loading
# =============================================================================
def build_cache():
    """
    Build DE feature cache from raw training data.
    Called automatically if _feature_cache_full.npz doesn't exist.
    """
    print("  Building feature cache from raw data (one-time, ~2 min)...")
    import h5py as _h5py

    comp_X, comp_y = [], []
    for subdir, n_subjects in [("正常人", 40), ("抑郁症患者", 20)]:
        d = os.path.join(DATA_DIR, "训练集", subdir)
        files = sorted([f for f in os.listdir(d) if f.endswith(".mat")])
        for fi, fn in enumerate(files):
            fp = os.path.join(d, fn)
            try:
                m = loadmat(fp)
                neu, pos = m["EEG_data_neu"], m["EEG_data_pos"]
            except NotImplementedError:
                with _h5py.File(fp, "r") as f:
                    neu = np.array(f["EEG_data_neu"])
                    pos = np.array(f["EEG_data_pos"])
            if neu.shape[0] != 30:
                neu = neu.T
            if pos.shape[0] != 30:
                pos = pos.T
            # Extract DE per trial (46 windows each), then concatenate: 8×46 = 368
            TRIAL_SAMPS = 12500  # 50s at 250Hz
            de_trials, labels_list = [], []
            for ti in range(8):
                if ti < 4:
                    seg = neu[:, ti * TRIAL_SAMPS:(ti + 1) * TRIAL_SAMPS]
                    labels_list.append(np.zeros(46, dtype=np.int64))
                else:
                    seg = pos[:, (ti - 4) * TRIAL_SAMPS:(ti - 3) * TRIAL_SAMPS]
                    labels_list.append(np.ones(46, dtype=np.int64))
                de_trials.append(extract_de(seg, TRIAL_SAMPS))
            de = np.concatenate(de_trials, axis=0)  # (368, 150) total
            labels = np.concatenate(labels_list)
            comp_X.append(de.astype(np.float32))
            comp_y.append(labels)
            print(f"    [{fi+1}/{n_subjects}] {subdir}: {fn}  {de.shape}")

    np.savez_compressed(CACHE, comp_X=np.array(comp_X, dtype=object),
                        comp_y=np.array(comp_y, dtype=object),
                        extra_X=np.array([], dtype=object),
                        extra_y=np.array([], dtype=object),
                        extra_n=np.array([], dtype=object))
    print(f"  Cache saved to {CACHE}")
    return comp_X, comp_y


def load_competition_data():
    """
    Load DE features for all 60 competition subjects.
    Builds cache from raw data if not already cached.
    """
    if not os.path.exists(CACHE):
        comp_X, comp_y = build_cache()
    else:
        cache = np.load(CACHE, allow_pickle=True)
        comp_X, comp_y = cache["comp_X"], cache["comp_y"]

    subjects = []
    for i in range(len(comp_X)):
        X = comp_X[i].astype(np.float32)
        X = per_subject_norm(X).astype(np.float32)
        y = comp_y[i].astype(np.int64)
        subjects.append((X, y))
    return subjects

def load_temporal_data(subjects):
    """Add temporal features to pre-loaded subjects."""
    return [(add_temporal_features(X), y) for X, y in subjects]

def make_pseudo_test(raw_eeg):
    """
    Build pseudo-test from raw EEG (for held-out evaluation).

    Each subject's raw EEG has 8 trials of 50s each.
    We extract the middle 10s of each trial, producing 6 windows per trial.

    Parameters
    ----------
    raw_eeg : ndarray (30, 100000) — 400s of raw EEG

    Returns
    -------
    de_features : ndarray (48, 150) — DE features
    labels : ndarray (48,) — ground truth (0=neutral, 1=positive)
    """
    TRIAL_LEN = 12500      # 50s at 250Hz
    EXTRACT_LEN = 2500     # 10s at 250Hz
    margin = (TRIAL_LEN - EXTRACT_LEN) // 2

    features, labels = [], []
    for vi in range(8):
        start = vi * TRIAL_LEN + margin
        eeg_seg = raw_eeg[:, start:start + EXTRACT_LEN]
        de = extract_de(eeg_seg, EXTRACT_LEN)
        features.append(de)
        labels.append(0 if vi < 4 else 1)

    return np.concatenate(features, 0), np.array(labels)

# =============================================================================
# Evaluation
# =============================================================================
def evaluate_predictions(probs_per_subject, true_labels):
    """
    Evaluate with forced 4+4 post-processing.

    For each subject:
      1. Average window probabilities within each of the 8 segments
      2. Top-4 segments → predicted positive (1), bottom-4 → neutral (0)
      3. Compare with ground truth

    Parameters
    ----------
    probs_per_subject : list of ndarray
        Predicted probabilities for each of 10 held-out subjects
    true_labels : list of ndarray
        Ground truth labels (each length 8, four 0s and four 1s)

    Returns
    -------
    dict with seg_acc, hc_acc, dep_acc, bal_acc, n_positive
    """
    seg_ok = hc_ok = dep_ok = 0
    all_probs = []

    for si in range(10):
        prob = probs_per_subject[si]
        # Aggregate 6 windows → 1 segment score
        seg_probs = [prob[vi * 6:(vi + 1) * 6].mean() for vi in range(8)]
        # Forced 4+4: top 4 segments → positive
        order = np.argsort(seg_probs)[::-1]
        forced = np.zeros(8, dtype=int)
        forced[order[:4]] = 1
        ok = (forced == true_labels[si]).sum()
        seg_ok += ok
        if si < 5:
            hc_ok += ok
        else:
            dep_ok += ok
        all_probs.append(seg_probs)

    n_pos = sum(sum(1 for p in sp if p >= 0.5) for sp in all_probs)
    return {
        "seg_acc": seg_ok / 80,
        "hc_acc": hc_ok / 40,
        "dep_acc": dep_ok / 40,
        "bal_acc": 0.5 * hc_ok / 40 + 0.5 * dep_ok / 40,
        "n_positive": n_pos,
    }

# =============================================================================
# Classifier: Stacked XGBoost + LightGBM
# =============================================================================
class StackedClassifier:
    """
    Two-stage stacking classifier.

    Stage 1: 4 base learners (2 XGBoost + 2 LightGBM) with 2-fold cross-validation
    Stage 2: LogisticRegression meta-learner trained on CV predictions

    Final prediction = 0.5 * average(base_learners) + 0.5 * meta_learner
    """

    def __init__(self, random_seed=42):
        self.seed = random_seed

    def _make_xgb(self, seed):
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, reg_alpha=0.5, min_child_weight=5,
            objective="binary:logistic", eval_metric="logloss",
            random_state=seed, n_jobs=-1,
        )

    def _make_lgb(self, seed):
        return lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, num_leaves=31,
            learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, reg_alpha=0.5, min_child_samples=20,
            objective="binary", metric="binary_logloss",
            random_state=seed, n_jobs=-1, verbose=-1,
        )

    def fit(self, X, y):
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        self.scaler = sc
        n = len(Xs)
        half = n // 2
        idx1, idx2 = np.arange(half), np.arange(half, n)

        # --- Stage 1: Cross-validated base learners ---
        meta_X = np.zeros((n, 2))

        # Fold 1 trains → predict fold 2
        m1 = self._make_xgb(self.seed)
        m1.fit(Xs[idx1], y[idx1])
        meta_X[idx2, 0] = m1.predict_proba(Xs[idx2])[:, 1]

        m2 = self._make_lgb(self.seed)
        m2.fit(Xs[idx1], y[idx1])
        meta_X[idx2, 1] = m2.predict_proba(Xs[idx2])[:, 1]

        # Fold 2 trains → predict fold 1
        m3 = self._make_xgb(self.seed + 1)
        m3.fit(Xs[idx2], y[idx2])
        meta_X[idx1, 0] = m3.predict_proba(Xs[idx1])[:, 1]

        m4 = self._make_lgb(self.seed + 1)
        m4.fit(Xs[idx2], y[idx2])
        meta_X[idx1, 1] = m4.predict_proba(Xs[idx1])[:, 1]

        # --- Stage 2: Meta-learner ---
        self.meta = LogisticRegression(C=1.0, random_state=self.seed)
        self.meta.fit(meta_X, y)

        # --- Final base learners on all data ---
        self.base_models = [
            self._make_xgb(self.seed),
            self._make_lgb(self.seed),
            self._make_xgb(self.seed + 1),
            self._make_lgb(self.seed + 1),
        ]
        for m in self.base_models:
            m.fit(Xs, y)

        return self

    def predict_proba(self, X):
        """Return per-window probability predictions."""
        Xs = self.scaler.transform(X)
        p = np.zeros((len(Xs), 4))
        for i, m in enumerate(self.base_models):
            p[:, i] = m.predict_proba(Xs)[:, 1]
        # Blend average ensemble + stacking meta
        p_avg = p.mean(1)
        p_meta = self.meta.predict_proba(np.column_stack([
            (p[:, 0] + p[:, 2]) / 2,
            (p[:, 1] + p[:, 3]) / 2,
        ]))[:, 1]
        return 0.5 * p_avg + 0.5 * p_meta

# =============================================================================
# Main Evaluation
# =============================================================================
def run_evaluation(n_splits=10, seed_start=100):
    """
    Run strict cross-subject evaluation.

    For each split:
      1. Leave out 5 HC + 5 DEP subjects
      2. Train stacking classifier on 50 remaining subjects
      3. Evaluate on held-out subjects using pseudo-test protocol
    """
    print("=" * 65)
    print("EEG Emotion Recognition — Cross-Subject Evaluation")
    print("=" * 65)

    # Load data
    print("\n[1/3] Loading data...")
    subjects_de = load_competition_data()
    subjects_temporal = load_temporal_data(subjects_de)
    print(f"  Loaded 60 subjects: 40 HC + 20 DEP")
    print(f"  DE features: 150D, Temporal features: {subjects_temporal[0][0].shape[1]}D")

    # Raw EEG for pseudo-test
    print("  Loading raw EEG for held-out evaluation...")
    from scipy.io import loadmat as _loadmat
    import h5py as _h5py

    def _load_raw(idx):
        if idx < 40:
            subdir, fi = "正常人", idx
        else:
            subdir, fi = "抑郁症患者", idx - 40
        d = os.path.join(DATA_DIR, "训练集", subdir)
        files = sorted([f for f in os.listdir(d) if f.endswith(".mat")])
        fp = os.path.join(d, files[fi])
        try:
            m = _loadmat(fp)
            neu, pos = m["EEG_data_neu"], m["EEG_data_pos"]
        except NotImplementedError:
            with _h5py.File(fp, "r") as f:
                neu = np.array(f["EEG_data_neu"])
                pos = np.array(f["EEG_data_pos"])
        if neu.shape[0] != 30:
            neu = neu.T
        if pos.shape[0] != 30:
            pos = pos.T
        return np.concatenate([neu, pos], axis=1).astype(np.float64)

    all_raw = [_load_raw(i) for i in range(60)]

    # Run splits
    print(f"\n[2/3] Running {n_splits}-split evaluation...")
    results = []

    for split_i in range(n_splits):
        t0 = time.time()
        rng = np.random.RandomState(seed_start + split_i)

        # Split subjects
        hc_held = sorted(rng.choice(40, 5, replace=False))
        dep_held = sorted(rng.choice(20, 5, replace=False))
        dep_held_g = [40 + i for i in dep_held]
        tr_hc = [i for i in range(40) if i not in hc_held]
        tr_dep = [i for i in range(20) if i not in dep_held]
        tr_ids = tr_hc + [40 + i for i in tr_dep]

        # Training data
        tr_X = np.concatenate(
            [subjects_temporal[i][0] for i in tr_ids], 0).astype(np.float32)
        tr_y = np.concatenate(
            [subjects_temporal[i][1] for i in tr_ids], 0).astype(np.int64)

        # Train
        clf = StackedClassifier(random_seed=42 + split_i * 10)
        clf.fit(tr_X, tr_y)

        # Pseudo-test
        pt_probs, pt_labels = [], []
        for hidx in hc_held + dep_held_g:
            fx, fy = make_pseudo_test(all_raw[hidx])
            # Critical: per-subject normalization BEFORE temporal features
            fx_norm = per_subject_norm(fx)
            fx_temp = add_temporal_features(fx_norm)
            pt_probs.append(clf.predict_proba(fx_temp))
            pt_labels.append(fy)

        # Evaluate
        metrics = evaluate_predictions(pt_probs, pt_labels)
        results.append(metrics)

        elapsed = time.time() - t0
        print(f"  Split {split_i + 1:2d}/{n_splits}: "
              f"BalAcc={metrics['bal_acc']:.4f}  "
              f"HC={metrics['hc_acc']:.4f}  "
              f"DEP={metrics['dep_acc']:.4f}  "
              f"({elapsed:.0f}s)")

        gc.collect()

    # Summary
    print(f"\n[3/3] Results ({n_splits} splits)")
    print("-" * 65)
    arr = np.array([[r["bal_acc"], r["hc_acc"], r["dep_acc"]] for r in results])
    print(f"  BalAcc  = {arr[:, 0].mean():.4f} ± {arr[:, 0].std():.4f}")
    print(f"  HC_Acc  = {arr[:, 1].mean():.4f} ± {arr[:, 1].std():.4f}")
    print(f"  DEP_Acc = {arr[:, 2].mean():.4f} ± {arr[:, 2].std():.4f}")
    per_split = "  ".join([f"{r['bal_acc']:.3f}" for r in results])
    print(f"  Per-split: [{per_split}]")
    print("-" * 65)
    return results

# =============================================================================
# Inference on Public Test Set
# =============================================================================
def run_inference():
    """
    Train on all 60 competition subjects, predict on 10 public test subjects.
    Generates submission file.
    """
    import pandas as pd

    print("=" * 65)
    print("EEG Emotion Recognition — Test Set Inference")
    print("=" * 65)

    # Load training data
    print("\n[1/4] Loading training data...")
    subjects_de = load_competition_data()
    subjects_temporal = load_temporal_data(subjects_de)
    tr_X = np.concatenate([s[0] for s in subjects_temporal], 0).astype(np.float32)
    tr_y = np.concatenate([s[1] for s in subjects_de], 0).astype(np.int64)
    print(f"  Training: {len(subjects_de)} subjects, {len(tr_X)} windows")

    # Train classifier
    print("\n[2/4] Training stacked classifier on all 60 subjects...")
    clf = StackedClassifier(random_seed=42)
    clf.fit(tr_X, tr_y)
    print("  Done.")

    # Load & process test data
    print("\n[3/4] Processing test set...")
    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".mat")])
    rows_before = []   # raw threshold predictions (before forced 4+4)
    rows_after = []    # forced 4+4 predictions

    for fn in test_files:
        fp = os.path.join(TEST_DIR, fn)
        m = loadmat(fp)
        key = [k for k in m.keys() if not k.startswith("__")][0]
        eeg = m[key].astype(np.float64)
        if eeg.shape[0] != 30:
            eeg = eeg.T

        user_id = fn.replace(".mat", "")

        # Extract DE for all 8 segments, normalize per-subject (48 windows)
        de_segments = [extract_de(eeg[:, vi * 2500:(vi + 1) * 2500], 2500)
                       for vi in range(8)]
        de_all = np.concatenate(de_segments, axis=0)       # (48, 150)
        de_norm = per_subject_norm(de_all)
        de_temp = add_temporal_features(de_norm)            # (48, 450)

        # Window-level predictions → segment-level probabilities
        all_probs = clf.predict_proba(de_temp)               # (48,)
        seg_probs = [all_probs[vi * 6:(vi + 1) * 6].mean() for vi in range(8)]

        # Before forced 4+4: raw threshold at 0.5
        raw_labels = [1 if p >= 0.5 else 0 for p in seg_probs]

        # After forced 4+4: top-4 highest → positive
        order = np.argsort(seg_probs)[::-1]
        forced_labels = np.zeros(8, dtype=int)
        forced_labels[order[:4]] = 1

        for vi in range(8):
            rows_before.append({
                "user_id": user_id,
                "trial_id": vi + 1,
                "Emotion_label": int(raw_labels[vi]),
            })
            rows_after.append({
                "user_id": user_id,
                "trial_id": vi + 1,
                "Emotion_label": int(forced_labels[vi]),
            })
        print(f"  {fn}: raw={raw_labels}  forced={forced_labels.tolist()}  "
              f"(probs: {[f'{p:.3f}' for p in seg_probs]})")

    # Save submissions
    print(f"\n[4/4] Saving submissions...")
    out_before = os.path.join(DATA_DIR, "submission_result_v3_before_forced.xlsx")
    out_after = os.path.join(DATA_DIR, "submission_result_v3.xlsx")
    pd.DataFrame(rows_before).to_excel(out_before, index=False)
    pd.DataFrame(rows_after).to_excel(out_after, index=False)
    print(f"  Before forced 4+4: {out_before}")
    print(f"  After forced 4+4:  {out_after}")
    print("=" * 65)

# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG Emotion Recognition — Cross-Subject Evaluation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with only 2 splits")
    parser.add_argument("--inference", action="store_true",
                        help="Train on all subjects and predict test set")
    parser.add_argument("--splits", type=int, default=10,
                        help="Number of evaluation splits (default: 10)")
    args = parser.parse_args()

    n_splits = 2 if args.quick else args.splits

    if args.inference:
        run_inference()
    else:
        run_evaluation(n_splits=n_splits)
