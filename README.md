# Hedging Language Classifier

An end-to-end pipeline studying the effect of synthetic data augmentation on
imbalanced text classification. The task is sentence-level binary classification
of hedging language in S&P 500 earnings call transcripts.

A sentence is labeled **positive** if it expresses epistemic uncertainty,
conditionality, or caution about a forward-looking claim — typically via modals
(*may*, *might*, *could*), cognitive verbs (*we believe*, *we expect*), or
conditional framing (*subject to*, *depending on*).

## Project Status

Work in progress. Core experiments completed through encoder fine-tuning;
non-linear classifier experiments planned.

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Dataset

- **Source:** `hfmlsoc/sp500_dataset_earnings_calls` (HuggingFace)
- **962 manually validated positives**, ~98,000 negatives — roughly **100:1 class imbalance**
- Three-way stratified split: Train 69,510 / Calibration 9,931 / Test 19,861
- Calibration and test sets permanently locked — real data only

## Structure

```
hedging-txtclf-experiments/
    src/
        generation/         — LLM-based synthetic data generation scripts
    notebooks/
        04_baseline.ipynb               — frozen embeddings + LR baseline
        05_positive_augmentation.ipynb  — filtered positive augmentation
        06_contrastive_augmentation.ipynb — hard contrastive negatives
        07_error_driven_augmentation.ipynb — false positive cluster-based augmentation
        08_finetuning_embedding.ipynb   — staged unfreezing of all-MiniLM-L6-v2
    data/
        processed/          — train/cal/test splits and embeddings (not tracked)
        synthetic/          — LLM-generated candidates (not tracked)
        results/            — metrics JSONs, calibrated score artifacts
```

## Methodology

### Evaluation Framework
All experiments use **Venn-Abers inductive conformal calibration** throughout —
converting raw classifier scores into calibrated probability intervals $[p_0, p_1]$
and a point estimate. The decision threshold is optimized on the calibration set
and applied to the test set without modification. This exposes score-scaling
artifacts that raw F1 misses under extreme class imbalance.

Primary metrics: calibrated precision, recall, and F1. DET curves and UMAP
visualizations used as diagnostic tools across experiments.

### Baseline
Frozen `all-MiniLM-L6-v2` (384-dim) embeddings + Logistic Regression
(`class_weight='balanced'`). Embeddings cached as `.npy` files and reused
across all conditions.

| | Precision | Recall | F1 |
|---|---|---|---|
| Raw ($\tau=0.5$) | 0.059 | 0.839 | 0.110 |
| Calibrated ($\tau=0.07$) | 0.136 | 0.260 | **0.179** |

## Experimental Results

### Synthetic Data Augmentation (Frozen Encoder)

All augmentation experiments use `Llama-3-8B-Instruct` for generation and
Venn-Abers calibrated scores for filtering. Results in the calibrated regime.

**Key findings:**
- Naive augmentation hurts calibrated metrics — filtering is essential
- Hard contrastive negatives show more promising embedding geometry than filtered
  positives despite identical metrics — distributed throughout the positive manifold
  rather than concentrated in one region
- Error-driven augmentation (false positives → k-means clustering → contrastive
  generation) is the most principled strategy — targets the classifier's own
  failure modes directly
- Frozen encoder is the primary bottleneck — not data quality, not augmentation
  strategy

### Chain-of-Thought Prompting (Diagnostic)

CoT reasoning added to the generation prompt — model explicitly identifies
hedging markers before generating variants. Generation quality improved visibly.
However, filter survival rate collapsed under the frozen encoder:

- Standard threshold ($\tau \geq 0.2$): **0 survivors** from 992 candidates
- Relaxed threshold ($\tau \geq 0.1$): **38 survivors** (3.8% vs 12.8% non-CoT)

CoT sentences are more linguistically diverse and less similar to training
positives in embedding space — the frozen encoder penalizes this diversity.
CoT candidates are retained for re-filtering after encoder fine-tuning.

### Encoder Fine-Tuning: Staged Unfreezing (Notebook 08)

`all-MiniLM-L6-v2` fine-tuned with staged unfreezing (1, 2, 3, and 6 layers),
cross-entropy loss with class weights, early stopping on calibrated F1.

| Condition | Precision | Recall | F1 |
|---|---|---|---|
| Frozen baseline | 0.136 | 0.260 | 0.179 |
| Top 1 layer | 0.109 | 0.266 | 0.155 |
| Top 2 layers | 0.144 | 0.146 | 0.145 |
| Top 3 layers | 0.093 | 0.318 | 0.144 |
| Full fine-tune | 0.101 | 0.286 | 0.150 |

**Key findings:**
- No unfreezing condition improved over the frozen baseline on calibrated F1
- Recall improved consistently with unfreezing depth (up to 0.318) — the encoder
  responds to fine-tuning signal and becomes more sensitive to hedging patterns
- Calibrated F1 during training peaked at ~0.44 across conditions but was
  unstable — training signal exists but a linear classifier cannot stably exploit it
- DET curves show regional improvements over the frozen baseline in the
  mid-range FPR region, suggesting the fine-tuned representations carry
  genuine discriminative signal


## Planned Experiments

### MLP Head on Frozen Embeddings
Replace logistic regression with a small MLP (2 layers, hidden dim 256,
ReLU, dropout 0.1) trained on frozen embeddings. Tests whether a non-linear
classifier resolves the geometric overlap problem independently of encoder
quality. Will be evaluated across augmentation regimes for direct comparison
with LR results.

### MLP Head with Fine-Tuned Encoder (Notebook 09)
End-to-end training of encoder + MLP jointly — both for training signal
and evaluation, with no train/evaluate inconsistency. Venn-Abers calibration
applied on top of MLP outputs. Best unfreezing depth from Notebook 08 used
as starting point. XGBoost on fine-tuned embeddings also evaluated as a
non-differentiable non-linear alternative.

### CoT Candidate Re-filtering
Re-embed all synthetic candidates (CoT positives, filtered positives,
error-driven) using the best fine-tuned encoder and re-run Venn-Abers
filter. Expected to substantially improve CoT survival rate.

### DistilBERT + Error-Driven Augmentation
Revisit DistilBERT fine-tuning with error-driven contrastive augmentation
seeded from DistilBERT's own false positives — not the frozen LR's.
Natural next step once MLP experiments establish a stable evaluation baseline.

## Notes

- All calibrated metrics use Venn-Abers inductive conformal calibration
- Threshold always optimized on calibration set, never test set
- Baseline artifacts frozen as `.npy` files for reproducibility across notebooks
- Generator: `Llama-3-8B-Instruct` via Groq API (free tier, 500K TPD limit)
- Fine-tuning runs on Google Colab T4 GPU via VS Code Colab extension
