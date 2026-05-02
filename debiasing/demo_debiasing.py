"""
Synthetic demonstration of Bolukbasi et al. (2016) hard debiasing.

Creates word embeddings with analytically controlled gender bias, applies the
three-step hard debiasing algorithm, and reports:
    - DirectBias before / after debiasing
    - WEAT effect size before / after debiasing

Design of synthetic embeddings (fully deterministic, noise=0)
--------------------------------------------------------------
Space: 50 dimensions.  Dim 0 is the true gender axis (e_0).

Profession vectors (30 words):
    v_i = normalize([1.0, 0, …, 5.018, …, 0])
    where 5.018 is in a unique semantic dimension i+1.
    cos(v_i, e_0) = 1/sqrt(1 + 5.018²) ≈ 0.1954  (exact for every i)

Gender / WEAT words: same structure, larger gender component (±3.0 or ±2.5),
    unique semantic dimensions, no noise → vectors are exactly orthogonal
    outside of dim 0, so after debiasing all cross-similarities vanish.

Expected output (deterministic):
    DirectBias : 0.1954 → 0.0000  (100 % reduction – exact)
    WEAT d     : 2.0000 → 0.0000  (100 % reduction in idealized case)

Note on difference from Bolukbasi reference values (1.84 → 0.76):
    The reference values come from a noisy synthetic design where residual
    cross-term similarities survive debiasing.  In the analytically designed
    demo here, debiasing is mathematically exact.  The real word2vec demo
    (debiasing_word2vec.py) will reproduce empirical values with residuals.
"""

import sys
import os
import numpy as np

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
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
DIM  = 150   # enough for ≥102 words × unique semantic dim, no collisions
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------
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

# Must NOT overlap with DEFINITIONAL_PAIRS or EQUALITY_SETS to avoid overwriting
WEAT_MALE   = ["husband", "nephew", "king", "uncle", "patriarch", "duke", "lad", "groom"]
WEAT_FEMALE = ["wife",    "niece",  "queen", "aunt", "matriarch", "duchess", "lass", "bride"]
WEAT_CAREER = ["executive", "management", "professional", "corporate",
               "salary", "office", "business", "career"]
WEAT_FAMILY = ["home", "parents", "children", "family",
               "wedding", "marriage", "domestic", "household"]

# ---------------------------------------------------------------------------
# Build synthetic embeddings
# ---------------------------------------------------------------------------
# Semantic anchor: placing 5.018 in a unique dimension ensures
#   cos(v_normalized, e_0) = 1/sqrt(1 + 5.018²) ≈ 0.1954 for profession words.
SEMANTIC_ANCHOR = 5.018

# Global counter: each word gets a unique semantic dimension (collision-free)
_sem_counter = [1]   # dim 0 = gender, dims 1..DIM-1 = semantic


def _next_dim() -> int:
    """Return the next available semantic dimension."""
    d = _sem_counter[0]
    _sem_counter[0] += 1
    assert d < DIM, f"Ran out of semantic dims (DIM={DIM})"
    return d


def _make_vec(gender_comp: float, sem_dim: int, noise: float = 0.0) -> np.ndarray:
    """
    Create a synthetic word vector.

    The vector is initialized to Gaussian noise scaled by *noise*, then:
        dim 0        ← gender_comp
        dim sem_dim  ← SEMANTIC_ANCHOR
    Finally the vector is L2-normalized.
    """
    v = np.random.randn(DIM) * noise
    v[0] = gender_comp
    if 0 < sem_dim < DIM:
        v[sem_dim] = SEMANTIC_ANCHOR
    return normalize(v)


embeddings: dict = {}

# -- Profession words: zero noise → DirectBias is analytically 0.1954 --
for word in PROFESSIONS:
    embeddings[word] = _make_vec(gender_comp=1.0, sem_dim=_next_dim(), noise=0.0)

# -- Definitional pairs: SHARED semantic dim per pair so centred differences
#    are purely along dim 0 → PCA recovers e_0 exactly --
for fem, masc in DEFINITIONAL_PAIRS:
    d = _next_dim()  # same dim for both words in the pair
    embeddings[fem]  = _make_vec(+3.0, sem_dim=d, noise=0.0)
    embeddings[masc] = _make_vec(-3.0, sem_dim=d, noise=0.0)

# -- Equality set words: noise=0 --
for w1, w2 in EQUALITY_SETS:
    embeddings[w1] = _make_vec(-2.0, sem_dim=_next_dim(), noise=0.0)
    embeddings[w2] = _make_vec(+2.0, sem_dim=_next_dim(), noise=0.0)

# -- WEAT targets: noise=0 → exact 2.0 before, exact 0.0 after --
for w in WEAT_MALE:
    embeddings[w] = _make_vec(+2.5, sem_dim=_next_dim(), noise=0.0)
for w in WEAT_FEMALE:
    embeddings[w] = _make_vec(-2.5, sem_dim=_next_dim(), noise=0.0)

# -- WEAT attributes: career positive, family negative bias --
for w in WEAT_CAREER:
    embeddings[w] = _make_vec(+1.0, sem_dim=_next_dim(), noise=0.0)
for w in WEAT_FAMILY:
    embeddings[w] = _make_vec(-0.5, sem_dim=_next_dim(), noise=0.0)

# ---------------------------------------------------------------------------
# Step 1: Identify bias subspace
# ---------------------------------------------------------------------------
bias_dir = identify_bias_subspace(DEFINITIONAL_PAIRS, embeddings)

# ---------------------------------------------------------------------------
# Metrics BEFORE debiasing
# ---------------------------------------------------------------------------
db_before   = direct_bias(PROFESSIONS, bias_dir, embeddings)
weat_before = weat_effect_size(WEAT_MALE, WEAT_FEMALE,
                               WEAT_CAREER, WEAT_FAMILY, embeddings)

# ---------------------------------------------------------------------------
# Step 2: Neutralize gender-neutral words
#   Professions + WEAT career/family attributes are all gender-neutral in
#   principle; debiasing them all allows WEAT to also show a clear reduction.
# ---------------------------------------------------------------------------
words_to_neutralize = PROFESSIONS + WEAT_CAREER + WEAT_FAMILY
debiased_neutral = neutralize(words_to_neutralize, bias_dir, embeddings)

# Build a merged embedding dict for evaluation (debiased neutrals + rest)
emb_debiased = dict(embeddings)
emb_debiased.update(debiased_neutral)
# Convenience alias for direct_bias check
debiased_profs = {w: emb_debiased[w] for w in PROFESSIONS if w in emb_debiased}

# ---------------------------------------------------------------------------
# Step 3: Equalize gender pairs
# ---------------------------------------------------------------------------
equalized = equalize(EQUALITY_SETS, bias_dir, emb_debiased)
emb_debiased.update(equalized)

# ---------------------------------------------------------------------------
# Metrics AFTER debiasing
# ---------------------------------------------------------------------------
db_after   = direct_bias(PROFESSIONS, bias_dir, emb_debiased)
weat_after = weat_effect_size(WEAT_MALE, WEAT_FEMALE,
                              WEAT_CAREER, WEAT_FAMILY, emb_debiased)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
reduction_db   = (1.0 - db_after / db_before) * 100 if db_before > 1e-10 else 0.0
reduction_weat = (1.0 - abs(weat_after) / abs(weat_before)) * 100 if abs(weat_before) > 1e-10 else 0.0

print("=" * 55)
print("  Bolukbasi et al. (2016) – Synthetic Debiasing Demo")
print("=" * 55)
print(f"  Embedding dim  : {DIM}D   |   Professions: {len(PROFESSIONS)}")
print(f"  Def. pairs     : {len(DEFINITIONAL_PAIRS)}   |   Equality sets: {len(EQUALITY_SETS)}")
print("-" * 55)
print(f"  {'Metric':<22} {'Before':>8}  {'After':>8}  {'Reduction':>10}")
print(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*9}")
print(f"  {'DirectBias':<22} {db_before:>8.4f}  {db_after:>8.4f}  {reduction_db:>8.1f} %")
print(f"  {'WEAT effect size':<22} {weat_before:>8.4f}  {weat_after:>8.4f}  {reduction_weat:>8.1f} %")
print("=" * 55)
print()
print("  Interpretation:")
print(f"  - DirectBias   = 0 means no measurable gender projection")
print(f"  - WEAT d  > 0  means male words still more career-associated")
print()
