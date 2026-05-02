# Debiasing Word Embeddings — Bolukbasi et al. (2016)

This module implements **hard debiasing** of word embeddings as introduced by
Bolukbasi et al. at NeurIPS 2016. It serves as a practical implementation
example in Chapter 4 of the accompanying master's thesis.

---

## Theoretical Background

> **Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016).**
> Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings.
> *Advances in Neural Information Processing Systems (NeurIPS) 29*, 4349–4357.
> https://arxiv.org/abs/1607.06520

The algorithm removes gender bias from word embeddings in three steps:

1. **Identify bias subspace** — PCA on definitional word pairs (she/he, woman/man, …)
   extracts the 1-D gender direction *g*.
2. **Neutralize** — Project gender-neutral words (e.g., occupations) off *g*:
   `w_new = normalize(w − (w·g)·g)`
3. **Equalize** — Force gendered pairs (e.g., *grandmother/grandfather*) to be
   equidistant from all neutralized words.

**Bias metrics used:**

| Metric | Formula | Reference |
|--------|---------|-----------|
| DirectBias | `(1/|N|) Σ |cos(w, g)|` | Bolukbasi et al. 2016, Def. 1 |
| WEAT d | `(mean_X s − mean_Y s) / std s` | Caliskan et al. 2017 |

---

## Module Structure

```
debiasing/
├── debias_embeddings.py     # Core algorithm: identify_bias_subspace / neutralize / equalize
├── demo_debiasing.py        # Synthetic verification (deterministic, ~1 s)
├── plot_debiasing.py        # Visualisation of synthetic demo → results/debiasing_synthetic.png
├── debiasing_word2vec.py    # Real word2vec-google-news-300 application
├── requirements.txt
├── results/
│   ├── debiasing_synthetic.png
│   ├── debiasing_word2vec.png
│   ├── debiasing_word2vec_results.csv
│   └── top_shifted_occupations.md
└── README.md
```

---

## Installation

```bash
pip install -r debiasing/requirements.txt
```

---

## Reproducing the Synthetic Demo (~1 second)

```bash
cd debiasing
python demo_debiasing.py
python plot_debiasing.py
```

**Expected output (deterministic):**

```
DirectBias   0.1954  →  0.0000  (100.0 % reduction)
WEAT d       2.0000  →  0.0000  (100.0 % reduction)
```

The synthetic demo uses analytically designed 150-D vectors where all word
dimensions are orthogonal except the controlled gender axis. This idealized
setting achieves mathematically exact debiasing; the real-embedding demo shows
the realistic residuals.

---

## Reproducing the word2vec Variant (~30 seconds + first-run download)

```bash
cd debiasing
python debiasing_word2vec.py
```

**First run:** downloads `word2vec-google-news-300` (~1.6 GB) into `~/gensim-data/`
and caches it permanently. Subsequent runs are fast.

**Measured results (word2vec-google-news-300, 3 M vocabulary):**

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| DirectBias | 0.1596 | 0.0000 | 100.0 % |
| WEAT effect size | 0.6406 | 0.4939 | 22.9 % |

DirectBias drops to zero because all profession vectors are analytically
projected off the gender direction. WEAT shows a partial reduction (22.9 %)
because the WEAT attribute words (career / family) are not in the neutralisation
set, so residual gender associations through the target words (male / female)
remain.

---

## Output Files

| File | Description |
|------|-------------|
| `results/debiasing_synthetic.png` | Three-panel figure: DirectBias bars, WEAT bars, PCA scatter with before/after arrows |
| `results/debiasing_word2vec.png` | Two-panel figure: DirectBias and WEAT bars for real embeddings |
| `results/debiasing_word2vec_results.csv` | Numeric results: metric, before, after, reduction_pct |
| `results/top_shifted_occupations.md` | Top-10 professions by absolute bias shift |

---

## Relation to the Master's Thesis

This module provides the **Chapter 4 implementation reference** for the thesis
*"Ethik und Bias in KI-Sprachmodellen"* (Kenan Husic, FH Campus Wien, 2026).

The synthetic demo illustrates that Bolukbasi's algorithm achieves mathematically
perfect debiasing under idealised conditions. The word2vec application replicates
the empirical setting of the original paper and confirms that stereotypically
female-coded professions (*homemaker*, *nurse*, *socialite*) carried the largest
gender bias before debiasing.
