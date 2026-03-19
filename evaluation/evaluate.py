#!/usr/bin/env python3
"""
Official evaluation script for the RT Prime Editing Activity Prediction Challenge.

Metric: CLS (Cross-Lineage Score)
=========================================

    CLS = 2 * PR_AUC * WSpearman / (PR_AUC + WSpearman)

Evaluation scheme: Global Pooled Out-of-Fold (OOF)
===================================================

Submissions are scored on LOFO (Leave-One-Family-Out) cross-validated predictions:

    1. Train 7 models, each holding out one of 7 evolutionary families.
    2. For each fold, predict on the held-out family.
    3. Concatenate all 7 sets into a single array of 57 predictions.
       Every prediction was made by a model that never saw that RT's
       evolutionary family during training.
    4. Compute CLS once on this global 57-prediction array.

Why global pooling instead of per-fold averaging?
    - 3 families (CRISPR-associated, Other, Unclassified) have 0 active RTs,
      making per-fold PR-AUC undefined.
    - Small folds (n=1 to n=5) produce noisy per-fold metrics.
    - Global pooling computes on N=57 with 21 positives — stable and clean.

Components:

    PR-AUC (Precision-Recall Area Under Curve)
        Measures separation of active vs inactive RTs.
        Imbalance-aware (21 active, 36 inactive). Random baseline ~ 0.37.

    Weighted Spearman
        Weighted rank correlation between predicted_score and pe_efficiency_pct.
        Weights = pe_efficiency_pct + epsilon (epsilon=0.01).
        MMLV-RT (41%) has weight ~41; inactive RTs have weight ~0.01.
        Correctly ranking the best RTs matters far more than ordering inactives.
        Floored at 0 — negative correlation scores zero, not negative.

    Harmonic Mean
        Forces both components to be good. PR-AUC=0.9 + WSpearman=0.3 -> CLS=0.45.

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


DATA_DIR = Path(__file__).parent.parent / "data"
EPSILON = 0.01  # small constant so inactive RTs still have nonzero weight


def load_ground_truth():
    """Load ground truth labels and efficiency values."""
    gt = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    return gt[["rt_name", "active", "pe_efficiency_pct", "rt_family"]]


def weighted_spearman(predicted_scores, true_efficiency, weights):
    """
    Weighted Spearman correlation.

    1. Rank both predicted_scores and true_efficiency.
    2. Compute weighted Pearson correlation of those ranks.

    Ranking uses argsort(argsort(...)) which assigns unique integer ranks.

    Parameters
    ----------
    predicted_scores : array-like, shape (n,)
    true_efficiency  : array-like, shape (n,)
    weights          : array-like, shape (n,)

    Returns
    -------
    float : weighted Spearman correlation in [-1, 1]
    """
    pred_ranks = np.argsort(np.argsort(predicted_scores)).astype(float)
    true_ranks = np.argsort(np.argsort(true_efficiency)).astype(float)

    w = np.asarray(weights, dtype=float)
    w = w / w.sum()  # normalize to sum to 1

    # Weighted means
    mu_p = np.dot(w, pred_ranks)
    mu_t = np.dot(w, true_ranks)

    # Weighted covariance and standard deviations
    dp = pred_ranks - mu_p
    dt = true_ranks - mu_t
    cov = np.sum(w * dp * dt)
    std_p = np.sqrt(np.sum(w * dp ** 2))
    std_t = np.sqrt(np.sum(w * dt ** 2))

    if std_p < 1e-12 or std_t < 1e-12:
        return 0.0

    return cov / (std_p * std_t)


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
    w_spearman = max(w_spearman, 0.0)  # floor at 0

    # 3. CLS (harmonic mean)
    if pr_auc <= 0 or w_spearman <= 0:
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

    if len(merged) != 57:
        print(f"WARNING: Expected 57 RTs, got {len(merged)}.")

    y_true = merged["active"].values
    y_score = merged["predicted_score"].values.astype(float)
    pe_eff = merged["pe_efficiency_pct"].values.astype(float)

    # Compute CLS
    result = compute_cls(y_true, y_score, pe_eff)

    print("=" * 60)
    print("RT PRIME EDITING ACTIVITY PREDICTION — EVALUATION")
    print("=" * 60)
    print()
    print("  Global Pooled OOF scoring on 57 RTs")
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
        "--predictions", required=True, help="Path to predictions CSV (global pooled OOF)"
    )
    args = parser.parse_args()
    evaluate(args.predictions)
