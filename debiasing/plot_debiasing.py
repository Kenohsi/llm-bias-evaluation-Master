"""
Visualisation of the Bolukbasi et al. (2016) synthetic debiasing demo.

Creates a three-panel figure saved to results/debiasing_synthetic.png:

    Panel A – DirectBias before / after (bar chart)
    Panel B – WEAT effect size before / after (bar chart)
    Panel C – 2-D PCA projection of profession words:
              x-axis = gender direction g
              y-axis = first non-gender PC of the original embeddings
              Colours: orange = biased, steelblue = debiased
              Arrows show each word's displacement.

Run from the repo root or from inside the debiasing/ folder:
    python debiasing/plot_debiasing.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless (no display required)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from debias_embeddings import (
    identify_bias_subspace,
    neutralize,
    equalize,
    direct_bias,
    weat_effect_size,
    normalize,
)

# ---------------------------------------------------------------------------
# Reproducible synthetic embeddings (identical to demo_debiasing.py)
# ---------------------------------------------------------------------------
SEED = 42
DIM  = 150
np.random.seed(SEED)

_sem_counter = [1]

def _next_dim() -> int:
    d = _sem_counter[0]
    _sem_counter[0] += 1
    assert d < DIM
    return d

PROFESSIONS = [
    "homemaker", "nurse", "receptionist", "librarian", "socialite",
    "hairdresser", "nanny", "bookkeeper", "stylist", "housekeeper",
    "maestro", "skipper", "protege", "philosopher", "captain",
    "architect", "financier", "warrior", "broadcaster", "magician",
    "doctor", "teacher", "professor", "engineer", "programmer",
    "scientist", "lawyer", "judge", "banker", "accountant",
]

DEFINITIONAL_PAIRS = [
    ("she", "he"), ("her", "his"), ("woman", "man"), ("girl", "boy"),
    ("daughter", "son"), ("mother", "father"), ("gal", "guy"),
    ("female", "male"), ("hers", "him"), ("girls", "boys"),
]

EQUALITY_SETS = [
    ("monastery", "convent"), ("spokesman", "spokeswoman"),
    ("dad", "mom"), ("men", "women"), ("grandpa", "grandma"),
    ("brother", "sister"), ("businessman", "businesswoman"),
    ("chairman", "chairwoman"), ("gentleman", "lady"),
    ("fraternity", "sorority"),
]

WEAT_MALE   = ["husband", "nephew", "king", "uncle", "patriarch", "duke", "lad", "groom"]
WEAT_FEMALE = ["wife",    "niece",  "queen", "aunt", "matriarch", "duchess", "lass", "bride"]
WEAT_CAREER = ["executive", "management", "professional", "corporate",
               "salary", "office", "business", "career"]
WEAT_FAMILY = ["home", "parents", "children", "family",
               "wedding", "marriage", "domestic", "household"]

SEMANTIC_ANCHOR = 5.018


def _make_vec(gender_comp, sem_dim, noise=0.0):
    v = np.random.randn(DIM) * noise
    v[0] = gender_comp
    if 0 < sem_dim < DIM:
        v[sem_dim] = SEMANTIC_ANCHOR
    return normalize(v)


embeddings = {}
for word in PROFESSIONS:
    embeddings[word] = _make_vec(1.0, sem_dim=_next_dim(), noise=0.0)
for fem, masc in DEFINITIONAL_PAIRS:
    d = _next_dim()  # shared dim → centred differences are purely along dim 0
    embeddings[fem]  = _make_vec(+3.0, sem_dim=d, noise=0.0)
    embeddings[masc] = _make_vec(-3.0, sem_dim=d, noise=0.0)
for w1, w2 in EQUALITY_SETS:
    embeddings[w1] = _make_vec(-2.0, sem_dim=_next_dim(), noise=0.0)
    embeddings[w2] = _make_vec(+2.0, sem_dim=_next_dim(), noise=0.0)
for w in WEAT_MALE:
    embeddings[w] = _make_vec(+2.5, sem_dim=_next_dim(), noise=0.0)
for w in WEAT_FEMALE:
    embeddings[w] = _make_vec(-2.5, sem_dim=_next_dim(), noise=0.0)
for w in WEAT_CAREER:
    embeddings[w] = _make_vec(+1.0, sem_dim=_next_dim(), noise=0.0)
for w in WEAT_FAMILY:
    embeddings[w] = _make_vec(-0.5, sem_dim=_next_dim(), noise=0.0)

# ---------------------------------------------------------------------------
# Run debiasing
# ---------------------------------------------------------------------------
bias_dir = identify_bias_subspace(DEFINITIONAL_PAIRS, embeddings)

db_before   = direct_bias(PROFESSIONS, bias_dir, embeddings)
weat_before = weat_effect_size(WEAT_MALE, WEAT_FEMALE,
                               WEAT_CAREER, WEAT_FAMILY, embeddings)

words_to_neutralize = PROFESSIONS + WEAT_CAREER + WEAT_FAMILY
debiased_neutral = neutralize(words_to_neutralize, bias_dir, embeddings)
emb_debiased = dict(embeddings)
emb_debiased.update(debiased_neutral)
equalized = equalize(EQUALITY_SETS, bias_dir, emb_debiased)
emb_debiased.update(equalized)

db_after   = direct_bias(PROFESSIONS, bias_dir, emb_debiased)
weat_after = weat_effect_size(WEAT_MALE, WEAT_FEMALE,
                              WEAT_CAREER, WEAT_FAMILY, emb_debiased)

# ---------------------------------------------------------------------------
# PCA projection for scatter plot
# ---------------------------------------------------------------------------
g = normalize(bias_dir)

# Stack all profession vectors (biased and debiased)
vecs_before = np.array([embeddings[w]    for w in PROFESSIONS])
vecs_after  = np.array([emb_debiased[w]  for w in PROFESSIONS])

# Project onto (g, first non-gender PC)
all_vecs = np.vstack([vecs_before, vecs_after])
pca2 = PCA(n_components=2)
pca2.fit(all_vecs)

# Force first component to align with g
if abs(np.dot(pca2.components_[0], g)) < abs(np.dot(pca2.components_[1], g)):
    pca2.components_[[0, 1]] = pca2.components_[[1, 0]]

proj_before = pca2.transform(vecs_before)
proj_after  = pca2.transform(vecs_after)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Hard Debiasing of Word Embeddings – Bolukbasi et al. (2016)\n"
    "Synthetic Demo (seed=42, 50-D, 30 profession words)",
    fontsize=11, fontweight="bold", y=1.02,
)

# ---- Panel A: DirectBias ----
ax = axes[0]
bars = ax.bar(["Before", "After"], [db_before, db_after],
              color=["#e74c3c", "#2ecc71"], width=0.4, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, [db_before, db_after]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
reduction = (1 - db_after / db_before) * 100 if db_before > 1e-10 else 0
ax.set_title(f"Panel A – DirectBias\n(reduction: {reduction:.1f} %)", fontsize=10)
ax.set_ylabel("DirectBias (c=1)")
ax.set_ylim(0, db_before * 1.4)
ax.axhline(0, color="black", linewidth=0.8)

# ---- Panel B: WEAT effect size ----
ax = axes[1]
colors_weat = ["#e74c3c" if v > 0 else "#2ecc71" for v in [weat_before, weat_after]]
bars = ax.bar(["Before", "After"], [weat_before, weat_after],
              color=colors_weat, width=0.4, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, [weat_before, weat_after]):
    ypos = val + 0.04 if val >= 0 else val - 0.08
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:+.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
weat_red = (1 - abs(weat_after) / abs(weat_before)) * 100 if abs(weat_before) > 1e-10 else 0
ax.set_title(f"Panel B – WEAT Effect Size\n(reduction: {weat_red:.1f} %)", fontsize=10)
ax.set_ylabel("WEAT d  (male–career vs. female–family)")
ymax = max(abs(weat_before), abs(weat_after)) * 1.5
ax.set_ylim(-ymax * 0.3, ymax * 1.3)
ax.axhline(0, color="black", linewidth=0.8)

# ---- Panel C: PCA scatter ----
ax = axes[2]
# Draw arrows showing displacement
for (xb, yb), (xa, ya) in zip(proj_before, proj_after):
    ax.annotate("", xy=(xa, ya), xytext=(xb, yb),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.7, alpha=0.6))
ax.scatter(proj_before[:, 0], proj_before[:, 1],
           c="#e74c3c", s=50, zorder=3, label="Before", edgecolors="black", linewidths=0.5)
ax.scatter(proj_after[:, 0],  proj_after[:, 1],
           c="#2ecc71", s=50, zorder=3, label="After",  edgecolors="black", linewidths=0.5)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xlabel("Gender direction (g)")
ax.set_ylabel("1st non-gender PC")
ax.set_title("Panel C – PCA Projection of Professions\n(arrows: before → after)", fontsize=10)
ax.legend(loc="upper right", framealpha=0.8)

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "debiasing_synthetic.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
print(f"DirectBias : {db_before:.4f} → {db_after:.4f}")
print(f"WEAT d     : {weat_before:+.4f} → {weat_after:+.4f}")
