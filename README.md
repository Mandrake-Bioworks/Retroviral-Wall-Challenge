# Mandrake Open Problems #1: The Retroviral Wall

**Can you predict which reverse transcriptases will work for prime editing — without memorising their evolutionary family?**

A machine learning challenge by [Mandrake Bio](https://www.mandrakebio.com) — an AI-first gene editing company building the next generation of gene editing tools.

📅 **Submissions close:** March 31, 2026
🏆 **Details & prizes:** https://www.mandrakebio.com/retroviral-wall-challenge/
📋 **Submit here:** https://www.mandrakebio.com/retroviral-wall-challenge/#register-interest

---

## The Problem

Prime editing is arguably the most precise genome editing technology ever built. It can write any of the 12 possible point mutations, make small insertions and deletions, and do it all without double-strand breaks or donor DNA templates. Where earlier CRISPR tools cut and hope, prime editing writes exactly what you want. It has the potential to correct the majority of known pathogenic mutations in the human genome.

At the heart of every prime editor is a **reverse transcriptase (RT)** enzyme fused to a Cas9 nickase. The RT reads an RNA template and writes the desired edit directly into DNA. It determines whether the edit works, how efficiently it works, and in which cell types it works. Today's best prime editors use MMLV-derived RTs — large enzymes (~700 amino acids) that are difficult to package into viral delivery vectors and challenging to deliver therapeutically. Smaller, more efficient RTs would unlock prime editing for a far wider range of applications.

Nature has evolved thousands of RTs across retroviruses, bacteria, mobile genetic elements, and more. Some will work for prime editing. Most won't. Experimentally screening thousands of candidates is expensive and slow — each round takes months of cloning, expression, and activity measurement. This is the fundamental bottleneck in enzyme engineering: the dry-lab to wet-lab loop. Lab testing is rate-limiting, and the datasets that come back are inevitably small and sparse. **Every candidate you can eliminate computationally is a month saved at the bench.**

Unlike antibody engineering, where binding affinity provides a single, well-defined metric to optimise against, enzyme function has no universal objective. An RT that works for prime editing must satisfy multiple constraints simultaneously — catalytic efficiency, processivity, fidelity, thermostability, structural compatibility with the Cas9 fusion, and more. There is no single number to hill-climb. Prediction requires learning from an ensemble of biophysical signals, many of which interact in ways that are poorly understood.

**This benchmark asks:** given a reverse transcriptase protein sequence and its computed biophysical/structural properties, can you predict whether it will be active for prime editing, and how efficient it will be?

The dataset comes from Doman et al. (2023), who experimentally tested 57 diverse RT enzymes for prime editing activity.

---

## Why This Is Hard

This looks like a simple binary classification problem (57 samples, 66 features). It isn't.

Yes, 57 samples with 66 features is a small-data problem. Standard ML approaches will overfit. That's the point — we want to see who can extract real signal from limited, high-dimensional biological data. This mirrors the reality of experimental biology, where datasets are expensive and small. **Cracking small-data biological prediction is the actual challenge.**

The dataset has pathologies that will eat naive approaches alive:

### Failure Mode 1: Family Memorisation

The 57 RTs come from 7 evolutionary families. The active/inactive ratio is heavily confounded with family membership. Retroviral RTs are mostly active (12/18). Non-retroviral families are mostly inactive. A model that learns "Retroviral → active" gets ~75% accuracy but may have learned family identity rather than the underlying biophysics of what makes an RT work.

The goal is to ensure your model's performance isn't purely explained by family distribution. We use leave-one-family-out cross-validation as a sanity check alongside standard evaluation — not as the only metric, but as evidence that the model has learned something real.

### Failure Mode 2: Class Imbalance Masking

21 active, 36 inactive. A model that predicts "inactive" for everything gets 63% accuracy. **Do not report accuracy alone.** F1 on the positive class is what matters.

### Failure Mode 3: The Retroviral Wall

The Retroviral family contains 12 of 21 active RTs (57%). When held out during cross-validation, your model must predict Retroviral activity from patterns learned on non-retroviral families — which have only 9 active RTs across 3 families. **Breaking through this wall is the central challenge.**

### Failure Mode 4: AUC on Single-Class Folds

Three families (CRISPR-associated, Other, Unclassified) have zero active members. AUC is undefined for these folds. A model predicting "inactive" for all gets Acc=1.0 on these folds. Don't count these folds as successes.

---

## Dataset

This dataset goes beyond raw activity labels. We've enriched every RT using our computational platform to maximise the signal available from 57 experimentally tested enzymes.

```
data/
├── rt_sequences.csv          # 57 RTs: name, sequence, active, PE efficiency, family
├── handcrafted_features.csv  # 66 biophysical features per RT
├── esm2_embeddings.npz       # ESM-2 1280-dim mean-pooled embeddings
├── family_splits.csv         # Family membership and class balance
├── feature_dictionary.csv    # What each of the 66 features means
└── structures/               # ESMFold predicted 3D structures (57 PDB files)
```

You are free to use any additional tools, models, or databases — protein language models, structural predictors, external RT datasets, novel featurisation approaches, anything. Document what you use in your writeup.

### rt_sequences.csv

| Column | Description |
|--------|-------------|
| `rt_name` | Unique identifier |
| `sequence` | Full amino acid sequence |
| `active` | 1 = active for prime editing, 0 = inactive |
| `pe_efficiency_pct` | Prime editing efficiency (%) — 0 for inactive, 1.5–41% for active |
| `rt_family` | Evolutionary family (7 families) |
| `protein_length_aa` | Sequence length |

### Family Breakdown

| Family | n | Active | Inactive | Notes |
|--------|---|--------|----------|-------|
| Retroviral | 18 | 12 | 6 | Includes MMLV (41%), the gold standard |
| Retron | 12 | 5 | 7 | Bacterial RTs |
| LTR_Retrotransposon | 11 | 2 | 9 | Mobile genetic elements |
| CRISPR-associated | 5 | 0 | 5 | All inactive |
| Group_II_Intron | 5 | 2 | 3 | Self-splicing elements |
| Other | 5 | 0 | 5 | All inactive |
| Unclassified | 1 | 0 | 1 | Single sample |

### handcrafted_features.csv

66 biophysical features computed from predicted structures and sequences, grouped into 13 categories:

| Group | # Features | Description |
|-------|-----------|-------------|
| ESM-IF Scores | 2 | Inverse folding perplexity and log-likelihood |
| Physicochemical | 3 | Instability index, hydropathy (GRAVY), helix-beta ratio |
| Charge & Solubility | 2 | Net charge at pH 7, CamSol intrinsic solubility |
| Thermostability | 10 | Predicted fraction retaining structure at 40–80°C, thermophilicity class |
| Catalytic Triad | 4 | Triad detection, inter-residue distances, RMSD to HIV-RT |
| Structural Contacts | 5 | Per-residue hydrophobic, salt bridge, and H-bond contacts (whole protein + pocket) |
| Hydrophobicity | 5 | Mean, std, and fraction hydrophobic residues (whole protein + YXDD region) |
| Active Site Geometry | 7 | YXDD motif secondary structure, beta-hairpin detection, motif sequence |
| SASA | 5 | Per-residue SASA, apolar ratio, relative accessibility (whole + pocket + N-terminal) |
| Ramachandran Quality | 2 | Favoured and outlier residue percentages |
| Electrostatic Potentials | 8 | Mean surface potential across subdomains (overall, fingers, palm, thumb, connection, motifs) |
| FoldSeek Structural Alignment | 10 | TM-scores to reference RT crystal structures across families |
| Metadata | 3 | RT family, sequence length, resolved residue count |

See `feature_dictionary.csv` for per-feature descriptions.

### structures/

ESMFold predicted 3D structures for all 57 RT enzymes in PDB format.

**Missing values:** Some structural features have NaN for RTs where the structural analysis pipeline could not resolve certain elements (e.g., no thumb domain detected, no YXDD motif found). Missing ≠ zero — handle accordingly.

**FoldSeek warning:** TM-scores to reference families (e.g., `foldseek_TM_MMLV`) encode structural similarity to known RT families. These are biophysically meaningful but are correlated with family membership. Use with care in leave-one-family-out evaluation.

### esm2_embeddings.npz

Pre-computed ESM-2 (650M parameter) embeddings, 1280 dimensions, mean-pooled across residue positions.

```python
data = np.load("data/esm2_embeddings.npz", allow_pickle=True)
names = data['names']       # (57,) RT names
embeddings = data['embeddings']  # (57, 1280) float32
```

**Warning:** ESM-2 embeddings encode evolutionary family membership extremely well. Models trained on raw embeddings will achieve high LOO-CV accuracy but fail on leave-one-family-out. They memorise family, not function.

---

## Evaluation

### Primary Metric: CLS (Cross-Lineage Score)

All submissions are ranked by a single metric — **CLS** — that forces your model to solve two problems at once: classification and ranking.

```
CLS = 2 × PR-AUC × WSpearman / (PR-AUC + WSpearman)
```

CLS is the harmonic mean of PR-AUC and Weighted Spearman. If either component is near zero, CLS collapses. You cannot compensate for terrible ranking with great classification, or vice versa. **Both problems must be solved.**

#### PR-AUC (Classification)

Precision-Recall Area Under Curve. How well do your predicted scores separate active RTs from inactive ones?

- **Precision:** Of the RTs you called active, how many actually are? (bench time efficiency)
- **Recall:** Of all truly active RTs, how many did you catch? (discovery rate)
- **Why PR-AUC?** The dataset is imbalanced — 21 active vs 36 inactive. A model predicting "everything inactive" gets 63% accuracy for free. PR-AUC focuses on finding the active RTs without false alarms.
- Random baseline: ~0.37 (the base rate of active RTs)

#### Weighted Spearman (Ranking)

Among the RTs you score highly, are the best-performing ones actually ranked highest?

- Weight per RT: `weight_i = pe_efficiency_i + ε` (ε = 0.1)
- **MMLV (41% efficiency)** has weight 41.1 — getting its rank wrong is heavily penalized.
- **Inactive RTs (0%)** have weight ~0.1 — their relative ranking barely matters.
- A negative correlation (worse than random) is floored at 0.
- Random baseline: ~0.000

### Leave-One-Family-Out (LOFO) Cross-Validation

All predictions must be generated using LOFO cross-validation:

```
For each of the 7 families:
    1. Hold out all RTs from that family as the test set
    2. Train on the remaining 6 families
    3. Predict on the held-out family
    4. Record predictions

The 7 folds produce 57 out-of-fold predictions (one per RT).
PR-AUC and WSpearman are computed on these pooled predictions — not averaged per family.
```

LOFO tests whether your model can generalise to an entirely unseen evolutionary lineage. When predicting Retroviral RTs, the model has never seen a Retroviral RT during training.

### Stage 1: Cross-Validation (March 2026)

Submissions are ranked by CLS on the 57 pooled LOFO predictions. Top submissions are selected based on quantitative performance and methodological novelty.

### Stage 2: Proprietary Validation (Q2 2026)

Your Stage 1 model stays on the leaderboard. We re-evaluate it on ~40 new RT candidates with real wet-lab PE efficiency data from Mandrake's lab. The leaderboard is updated and winners are announced. You don't submit anything new — your model carries forward.

---

## Baselines

Here are the approaches we've tested on leave-one-family-out:

| Approach | PR-AUC | WSpearman | CLS | Notes |
|----------|--------|-----------|-----|-------|
| Predict all inactive | ~0.37 | 0.000 | 0.000 | Trivial baseline |
| Random scores | ~0.37 | ~0.000 | ~0.000 | No signal |
| ESM-2 + Ridge (α=1000) | 0.548 | 0.000 | 0.000 | Embeddings memorise family |
| ESM-2 + RF (d=10) | 0.485 | 0.000 | 0.000 | Same problem |
| HandCrafted + RF (d=10) | 0.596 | 0.217 | 0.318 | Current best |
| HandCrafted + LogReg (C=0.01) | 0.460 | 0.000 | 0.000 | Classification only |
| PLS + LightGBM LambdaRank | 0.659 | 0.000 | 0.000 | Best PR-AUC, but no ranking |

---

## Submission

Submit through the **[official submission form](https://www.mandrakebio.com/retroviral-wall-challenge/#register-interest)** by **March 31, 2026** with:

1. **Code** — Full runnable pipeline (notebook or scripts) with a `requirements.txt`. We must be able to reproduce your results.
2. **Predictions CSV** — `rt_name`, `predicted_active`, `predicted_score`
3. **Writeup** — 1-2 pages. What you did, why, what worked, what didn't.

See `submissions/example_submission.csv` for the expected format:

```csv
rt_name,predicted_active,predicted_score
MMLV-RT,1,0.95
BLV-RT,0,0.12
...
```

- `predicted_active`: binary 0/1 prediction
- `predicted_score`: continuous score (higher = more likely active), used for AUC and ranking

---

## Rules

1. **All predictions must use LOFO cross-validation** — 7 folds, 57 out-of-fold predictions. CLS is computed on the pooled predictions.
2. **External data allowed** — protein databases, language model embeddings, structural predictions, etc. Describe everything you used.
3. **Reproducibility required** — the pipeline must be fully automated. We will run your code independently.
4. **No manual curation of predictions.**

---

## File Structure

```
Retroviral-Wall-Challenge/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── rt_sequences.csv
│   ├── handcrafted_features.csv
│   ├── esm2_embeddings.npz
│   ├── family_splits.csv
│   ├── feature_dictionary.csv
│   └── structures/
└── submissions/
    └── example_submission.csv
```

---

## Compute Credits

Have a compelling idea but need GPU resources to execute? Reach out with your proposed approach through the [official challenge page](https://www.mandrakebio.com/retroviral-wall-challenge/#register-interest). We provide compute credits on a case-by-case basis for ideas we find promising.

---

## License

**Code:** MIT License. See [LICENSE](LICENSE).

**Data:** Derived from Doman et al. (2023). Features and embeddings provided under CC-BY-4.0.

---

*A challenge by [Mandrake Bio](https://www.mandrakebio.com) — making life programmable.*
