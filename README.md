# Hedging Language Detection in S&P 500 Earnings Calls

Sentence-level binary classification of hedging language in financial earnings
call transcripts. A systematic empirical study of frozen sentence embedding
classifiers and LLM-generated synthetic data augmentation strategies for a
severely class-imbalanced NLP task.

A sentence is labeled **positive** if it expresses epistemic uncertainty,
conditionality, or caution about a forward-looking claim — typically via modals
(*may*, *might*, *could*), cognitive verbs (*we believe*, *we expect*), or
conditional framing (*subject to*, *depending on*).

---

## Problem

**Why it is hard:**
- 100:1 class imbalance (674 positive training examples vs ~68K negatives)
- No clean decision boundary — positive examples are geometrically diffused
  throughout the negative manifold in embedding space (confirmed by UMAP)
- Linguistically subtle — hedging is defined by epistemic stance, not surface
  markers alone

**Dataset:** `hfmlsoc/sp500_dataset_earnings_calls` — 962 manually validated
positive examples drawn from S&P 500 earnings call transcripts.

**Splits (locked throughout all experiments):**

| Split | Size | Positives |
|---|---|---|
| Train | 69,510 | 674 |
| Calibration | 9,931 | 96 |
| Test | 19,861 | 192 |

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

## Approach

All experiments use **frozen `all-MiniLM-L6-v2` embeddings** (384-dim,
SentenceTransformers). The encoder is never fine-tuned — the embedding space
is treated as a fixed representation and the classifier head is the only
trained component.

**Decision threshold** is optimized on the calibration set for each
experiment and applied to the test set without modification. F1 is the
primary metric; Precision is the hardest to move and most relevant to
compliance applications.

---

## Classifier Heads

Five classifier heads evaluated on frozen embeddings:

| Head | Architecture | Imbalance handling |
|---|---|---|
| LR | Linear (384→2) | `class_weight='balanced'` |
| MLP-1 | 384→256→2 | Inverse class frequency weights |
| MLP-2 | 384→256→128→2 | Inverse class frequency weights |
| Tree ensemble (RF, ET, XGB, LGBM) | Various | See below |
| SVM RBF | RBF kernel | `class_weight='balanced'` |

MLP classifiers use AdamW optimizer and early stopping on calibration F1.
Tree ensembles use `class_weight='balanced'` (RF, ET) and `scale_pos_weight`
(XGB, LGBM). SVM uses `gamma='scale'` and was evaluated across
C ∈ {0.1, 1.0, 10.0}.

---

## Augmentation Strategies

Two LLM-generated synthetic augmentation strategies evaluated:

### Positive Augmentation (CoT)
Chain-of-thought prompted `Llama-3.1-8B-Instant` and `Llama-3.3-70B-Versatile`
to generate genuine hedged variants from real positive seeds. Model reasons
explicitly about hedging markers before generating.

**Filter:** cosine similarity to nearest real positive ≥ τ (one-sided lower
bound — retains candidates geometrically close to real positives).

### Hard Contrastive Negative Augmentation
Prompted `Llama-3.1-8B-Instant` and `Llama-3.3-70B-Versatile` to generate
sentences that use hedging surface markers (epistemic verbs, modals) but
express confirmed past facts rather than genuine epistemic uncertainty.
Targets the decision boundary directly — hard negatives near the positive
manifold.

**Filter:** bounded cosine similarity 0.4 ≤ τ ≤ 0.9 (excludes near-duplicate
positives at the upper end and distant noise at the lower end — more
principled than a one-sided threshold for contrastive negatives specifically).

---

## Results

### Baseline — Real Data Only

| Classifier | Threshold | P | R | F1 |
|---|---|---|---|---|
| LR | 0.69 | 0.085 | 0.729 | 0.152 |
| MLP-1 | 0.69 | 0.190 | 0.464 | 0.270 |
| MLP-2 | 0.69 | 0.134 | 0.448 | 0.207 |

LR establishes the floor. MLP-1 peaks — MLP-2 drops, suggesting insufficient
data to support the extra capacity. The capacity-data interaction motivates
synthetic augmentation.

### Tree-Based Classifiers — A Negative Result

Four ensemble methods evaluated as alternatives to the MLP head, with
appropriate imbalance handling for each:

| Classifier | Threshold | P | R | F1 |
|---|---|---|---|---|
| Random Forest | 0.06 | 0.117 | 0.115 | 0.116 |
| Extra Trees | 0.05 | 0.137 | 0.089 | 0.108 |
| XGBoost | 0.64 | 0.253 | 0.099 | 0.142 |
| LightGBM | 0.61 | 0.167 | 0.109 | 0.132 |

**All four underperform the LR baseline (F1=0.152).** Tree-based classifiers
partition the feature space with axis-aligned splits — a fundamentally wrong
inductive bias for 384-dimensional dense sentence embeddings where the signal
is distributed across all dimensions simultaneously. XGBoost's DET curve shows
competitive ranking ability at relaxed operating points, but the operating point
under threshold optimization falls well below any MLP variant. Augmentation
experiments not pursued given the weak baseline.

### 8B Augmentation

| Condition | Classifier | Threshold | P | R | F1 |
|---|---|---|---|---|---|
| Vanilla CoT (τ≥0.8) | MLP-2 | 0.68 | 0.183 | 0.354 | 0.242 |
| CoT (τ≥0.8) | MLP-1 | 0.69 | 0.171 | 0.521 | 0.257 |
| Contrastive (τ≥0.7) | MLP-1 | 0.69 | 0.205 | 0.417 | 0.275 |

Key findings:
- Positive augmentation benefits MLP-2 (data-hungry) but not MLP-1
  (already capacity-matched to real data)
- Contrastive negatives benefit MLP-1 via precision gains — boundary
  sharpening complements a near-capacity classifier
- The augmentation strategy interacts with classifier capacity in a
  predictable and reproducible way across conditions

### 70B Augmentation

Upgrading the generator from 8B to 70B produces measurably higher quality
candidates — mean cosine similarity to real positives increases from 0.700
(vanilla 8B) to 0.735 (CoT 70B), and survival rates at strict thresholds
are higher across the board.

| Condition | Classifier | Threshold | P | R | F1 |
|---|---|---|---|---|---|
| CoT 70B (τ≥0.6) | MLP-1 | 0.59 | 0.223 | 0.422 | 0.292 |
| Contrastive 70B (0.4≤τ≤0.9) | MLP-1 | 0.60 | 0.240 | 0.370 | 0.291 |

The 70B contrastive experiment confirmed the bounded filter rationale: τ≥0.8
produced zero survivors (70B generates more geometrically distant hard
negatives than 8B), while the bounded filter 0.4≤τ≤0.9 retained 651/673
candidates at a principled similarity range.

### SVM RBF — C Sweep

C=0.1 (F1=0.235) < C=1.0 (F1=0.289) < C=10.0 (F1=0.301) — harder margin
consistently improves performance. All subsequent SVM experiments use C=10.

### SVM RBF (C=10) — Full Results

| Condition | P | R | F1 |
|---|---|---|---|
| Real data only | 0.267 | 0.344 | **0.301** |
| + 70B CoT positives | 0.290 | 0.234 | 0.259 |
| + 70B Contrastive negatives | 0.251 | 0.328 | 0.284 |

**SVM RBF C=10 on real data alone is the best result of the project.**
Synthetic augmentation consistently hurts SVM regardless of strategy:
- CoT positives expand the positive manifold, disrupting the tight
  max-margin boundary — precision rises but recall collapses
- Contrastive negatives shift support vectors away from the optimal
  position — partial recovery but still below real-data baseline

The augmentation pattern for SVM inverts the MLP pattern: the same
strategies that help MLP classifiers hurt SVM, and vice versa. This
reflects a fundamental difference in how the two classifier families
exploit the embedding space.

---

## Summary

| Classifier | Augmentation | P | R | F1 |
|---|---|---|---|---|
| LR | None | 0.085 | 0.729 | 0.152 |
| XGBoost | None | 0.253 | 0.099 | 0.142 |
| MLP-2 | None | 0.134 | 0.448 | 0.207 |
| MLP-1 | None | 0.190 | 0.464 | 0.270 |
| MLP-1 | Contrastive 8B | 0.205 | 0.417 | 0.275 |
| MLP-1 | Contrastive 70B | 0.240 | 0.370 | 0.291 |
| MLP-1 | CoT 70B | 0.223 | 0.422 | 0.292 |
| **SVM RBF C=10** | **None** | **0.267** | **0.344** | **0.301** |

**Best result:** SVM RBF (C=10, gamma='scale', class_weight='balanced') on
frozen `all-MiniLM-L6-v2` embeddings, real training data only.
F1=0.301 | Precision=0.267 | Recall=0.344

---

## Key Findings

**1. Capacity-augmentation interaction**
MLP-1 is capacity-matched to the real data — augmentation adds noise.
MLP-2 is data-hungry — positive augmentation helps. Contrastive negatives
sharpen boundaries for near-capacity classifiers. These interactions are
consistent and reproducible across 8B and 70B generation conditions.

**2. Generator quality matters for contrastive negatives**
Upgrading from 8B to 70B improves contrastive negative quality measurably
(geometric coherence, domain specificity) and pushes MLP-1 precision from
0.205 to 0.240. The bounded filter (0.4≤τ≤0.9) is more principled than a
one-sided threshold for contrastive negatives — excludes both potential
mislabeled positives (τ>0.9) and distant noise (τ<0.4).

**3. Tree-based classifiers are incompatible with dense embeddings**
Axis-aligned splits cannot exploit the distributed signal in 384-dimensional
sentence embeddings. All four tree ensemble methods fail to match the LR
baseline despite appropriate imbalance handling — a clean negative result
that motivates kernel and neural approaches.

**4. Inductive bias substitutes for augmentation**
SVM RBF with a hard margin (C=10) achieves F1=0.301 on real data alone —
surpassing the best augmented MLP result (F1=0.292) without any synthetic
data. The right inductive bias for dense continuous embedding spaces
(max-margin, kernel trick) is more valuable than augmentation volume or
quality for this classifier family.

**5. Augmentation effects are classifier-specific**
The same augmentation strategy that helps one classifier hurts another.
There is no universally beneficial augmentation regime — the interaction
between augmentation strategy, classifier capacity, and embedding geometry
must be evaluated empirically for each classifier head.

---

## Repository Structure

```
hedging-txtclf-experiments/
├── src/
│   └── generation/
│       ├── synthesize_cot.py               — 8B CoT positive augmentation
│       ├── synthesize_cot_70b.py           — 70B CoT positive augmentation
│       ├── synthesize_hard_contrastive.py  — 8B hard contrastive negatives
│       └── synthesize_hard_contrastive_70b.py — 70B hard contrastive negatives
├── notebooks/
│   ├── 09_mlp_baseline.ipynb              — LR + MLP-1 + MLP-2 on real data
│   ├── 10_mlp_positive_aug.ipynb          — positive augmentation experiments
│   ├── 11_mlp_contrastive_aug.ipynb       — contrastive negative experiments
│   ├── 13_mlp_combined_70b.ipynb          — combined 70B augmentation
│   └── 13_tree_baseline.ipynb             — tree ensemble experiments
├── data/
│   ├── processed/    — train/cal/test splits and frozen embeddings (not tracked)
│   ├── synthetic/    — LLM-generated augmentation parquets (not tracked)
│   └── results/      — metrics JSONs, DET curves, UMAP plots
└── requirements.txt
```

---

## Experimental Setup

- **Encoder:** `sentence-transformers/all-MiniLM-L6-v2` (frozen throughout)
- **Generation models:** `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`
  via Groq API
- **Threshold optimization:** calibration set F1 sweep, never test set
- **MLP training:** AdamW, inverse class frequency weights, early stopping
  on calibration F1 (patience=5)
- **SVM:** scikit-learn `SVC`, C sweep ∈ {0.1, 1.0, 10.0}, `gamma='scale'`
- **Seed:** 42 throughout

---

## Citation

If you use this code or findings, please cite:

```
@misc{kamat2025hedging,
  author = {Kamat, Vikram},
  title  = {LLM Synthetic Data Strategies for Imbalanced Text Classification:
             An Empirical Study},
  note   = {Hedging Language Detection in S\&P 500 Earnings Calls},
  year   = {2025},
}
```