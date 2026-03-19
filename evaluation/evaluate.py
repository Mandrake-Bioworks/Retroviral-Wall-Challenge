#!/usr/bin/env python3
"""
Official evaluation script for the RT Prime Editing Activity Prediction Challenge.

Metric: CLS (Composite Leaderboard Score)
=========================================

    CLS = 2 * PR_AUC * WSpearman / (PR_AUC + WSpearman)

CLS is the harmonic mean of two components:

1. PR-AUC (Precision-Recall Area Under Curve)
   - Measures how well predicted_score separates active from inactive RTs.
   - Chosen over ROC-AUC because the dataset is imbalanced (21 active, 36 inactive).
   - A trivial "predict all inactive" baseline gets PR-AUC ~ 0.37 (the base rate).

2. Weighted Spearman (WSpearman)
   - Measures how well predicted_score ranks RTs by PE efficiency.
   - Each RT is weighted by (pe_efficiency_pct + 0.1).
   - MMLV-RT (41% efficiency) has weight 41.1; inactive RTs have weight 0.1.
   - This means correctly ranking the best RTs matters far more than ordering
     the inactive ones.
   - Computed as the weighted Pearson correlation of ranks.

Why harmonic mean?
   - You cannot compensate for terrible ranking with great classification,
     or vice versa. Both problems must be solved simultaneously.
   - If either component is zero, CLS is zero.

Usage:
    python evaluate.py --predictions path/to/predictions.csv

    The predictions CSV must have columns:
        rt_name          - RT identifier (must match data/rt_sequences.csv)
        predicted_score  - Continuous score (higher = more likely active/efficient)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata


DATA_DIR = Path(__file__).parent.parent / "data"
EPSILON = 0.1  # small constant so inactive RTs still have nonzero weight


def load_ground_truth():
    """Load ground truth labels and efficiency values."""
    gt = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    return gt[["rt_name", "active", "pe_efficiency_pct", "rt_family"]]


def weighted_spearman(predicted_scores, true_efficiency, weights):
    """
    Weighted Spearman correlation.

    1. Rank both predicted_scores and true_efficiency.
    2. Compute weighted Pearson correlation of those ranks.

    Parameters
    ----------
    predicted_scores : array-like, shape (n,)
    true_efficiency  : array-like, shape (n,)
    weights          : array-like, shape (n,)
        Per-sample weights. Higher weight = more important to rank correctly.

    Returns
    -------
    float : weighted Spearman correlation in [-1, 1]
    """
    pred_ranks = rankdata(predicted_scores)
    true_ranks = rankdata(true_efficiency)
    w = np.asarray(weights, dtype=float)

    # Weighted means
    w_sum = w.sum()
    mu_p = np.dot(w, pred_ranks) / w_sum
    mu_t = np.dot(w, true_ranks) / w_sum

    # Weighted covariance and variances
    dp = pred_ranks - mu_p
    dt = true_ranks - mu_t
    cov = np.dot(w, dp * dt) / w_sum
    var_p = np.dot(w, dp ** 2) / w_sum
    var_t = np.dot(w, dt ** 2) / w_sum

    denom = np.sqrt(var_p * var_t)
    if denom < 1e-12:
        return 0.0

    return cov / denom


def compute_cls(y_true, y_score, pe_efficiency):
    """
    Compute CLS = harmonic_mean(PR-AUC, Weighted Spearman).

    Parameters
    ----------
    y_true        : array-like, shape (n,) — binary active labels (0 or 1)
    y_score       : array-like, shape (n,) — continuous predicted scores
    pe_efficiency : array-like, shape (n,) — true PE efficiency percentages

    Returns
    -------
    dict with keys: cls, pr_auc, w_spearman
    """
    # 1. PR-AUC
    pr_auc = average_precision_score(y_true, y_score)

    # 2. Weighted Spearman
    weights = pe_efficiency + EPSILON
    w_spearman = weighted_spearman(y_score, pe_efficiency, weights)
    w_spearman = max(w_spearman, 0.0)  # floor at 0 — negative correlation is no better than random

    # 3. CLS (harmonic mean)
    if pr_auc + w_spearman < 1e-12:
        cls = 0.0
    else:
        cls = 2.0 * pr_auc * w_spearman / (pr_auc + w_spearman)

    return {"cls": cls, "pr_auc": pr_auc, "w_spearman": w_spearman}


def evaluate(predictions_path):
    """Run full evaluation and print results."""
    gt = load_ground_truth()
    pred = pd.read_csv(predictions_path)

    # Validate columns
    if "predicted_score" not in pred.columns:
        raise ValueError("Predictions CSV must have a 'predicted_score' column.")
    if "rt_name" not in pred.columns:
        raise ValueError("Predictions CSV must have an 'rt_name' column.")

    # Merge
    merged = gt.merge(pred[["rt_name", "predicted_score"]], on="rt_name", how="left")
    missing_preds = merged["predicted_score"].isna().sum()
    if missing_preds > 0:
        print(f"WARNING: {missing_preds} RTs have no prediction. Filling with 0.0.")
        merged["predicted_score"] = merged["predicted_score"].fillna(0.0)

    y_true = merged["active"].values
    y_score = merged["predicted_score"].values.astype(float)
    pe_eff = merged["pe_efficiency_pct"].values.astype(float)

    # Compute CLS
    result = compute_cls(y_true, y_score, pe_eff)

    print("=" * 60)
    print("RT PRIME EDITING ACTIVITY PREDICTION — EVALUATION")
    print("=" * 60)
    print()
    print(f"  PR-AUC:             {result['pr_auc']:.4f}")
    print(f"  Weighted Spearman:  {result['w_spearman']:.4f}")
    print(f"  ─────────────────────────────")
    print(f"  CLS:                {result['cls']:.4f}")
    print()

    # Per-family breakdown
    families = merged["rt_family"].values
    print("Per-family PR-AUC breakdown:")
    print(f"  {'Family':<25s} {'n':>3s} {'Active':>6s} {'PR-AUC':>8s}")
    print("  " + "-" * 45)
    for fam in sorted(set(families)):
        mask = families == fam
        n = mask.sum()
        na = int(y_true[mask].sum())
        if na > 0 and na < n:
            fam_prauc = average_precision_score(y_true[mask], y_score[mask])
            print(f"  {fam:<25s} {n:3d} {na:6d} {fam_prauc:8.4f}")
        else:
            print(f"  {fam:<25s} {n:3d} {na:6d}      N/A")

    print()
    print(f"Baseline reference:  CLS = 0.318 (Handcrafted + RF)")
    print(f"Trivial baseline:    CLS = 0.000 (predict all inactive)")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions for the RT Prime Editing Activity Prediction Challenge (CLS metric)"
    )
    parser.add_argument(
        "--predictions", required=True, help="Path to predictions CSV"
    )
    args = parser.parse_args()
    evaluate(args.predictions)
