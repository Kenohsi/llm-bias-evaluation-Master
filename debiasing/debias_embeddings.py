"""
Hard Debiasing of Word Embeddings.

Based on:
    Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016).
    Man is to Computer Programmer as Woman is to Homemaker?
    Debiasing Word Embeddings. NeurIPS 2016.
    https://arxiv.org/abs/1607.06520

Three-step algorithm:
    1. identify_bias_subspace  – PCA on definitional pairs → gender direction g
    2. neutralize              – project gender-neutral words off g
    3. equalize                – ensure equality pairs are equidistant from all neutralized words
"""

import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of v. Returns v unchanged if norm is near zero."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


def drop(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Remove the component of u along v (v need not be a unit vector)."""
    v_unit = normalize(v)
    return u - np.dot(u, v_unit) * v_unit


# ---------------------------------------------------------------------------
# Step 1 – Identify bias subspace
# ---------------------------------------------------------------------------

def identify_bias_subspace(
    definitional_pairs: list,
    embeddings: dict,
    n_components: int = 1,
) -> np.ndarray:
    """
    Identify the gender bias direction via PCA on definitional word pairs.

    For each pair (a, b), the centred difference vectors
        a - (a+b)/2  and  b - (a+b)/2
    are stacked into a matrix.  The first principal component of that matrix
    is the gender direction g.

    Parameters
    ----------
    definitional_pairs : list of (str, str)
        Word pairs that define the gender axis (e.g. ``[("she","he"), ...]``).
    embeddings : dict[str, np.ndarray]
        Word → vector mapping.
    n_components : int
        Number of bias directions to extract (default 1 for hard debiasing).

    Returns
    -------
    np.ndarray, shape (d,)
        The gender direction (first principal component).

    Raises
    ------
    ValueError
        If none of the pairs appear in the vocabulary.
    """
    matrix = []
    for w1, w2 in definitional_pairs:
        if w1 not in embeddings or w2 not in embeddings:
            continue
        v1 = normalize(embeddings[w1])
        v2 = normalize(embeddings[w2])
        center = (v1 + v2) / 2.0
        matrix.append(v1 - center)
        matrix.append(v2 - center)

    if not matrix:
        raise ValueError(
            "No definitional pairs found in the provided embeddings vocabulary."
        )

    pca = PCA(n_components=n_components)
    pca.fit(np.array(matrix))
    return pca.components_[0]   # shape (d,)


# ---------------------------------------------------------------------------
# Step 2 – Neutralize gender-neutral words
# ---------------------------------------------------------------------------

def neutralize(
    words: list,
    bias_direction: np.ndarray,
    embeddings: dict,
) -> dict:
    """
    Remove the gender component from gender-neutral words (e.g. occupations).

    For each word w:
        w_debiased = normalize(w − (w · g) · g)

    Parameters
    ----------
    words : list of str
        Gender-neutral words whose bias should be removed.
    bias_direction : np.ndarray, shape (d,)
        The gender direction g (need not be a unit vector).
    embeddings : dict[str, np.ndarray]

    Returns
    -------
    dict[str, np.ndarray]
        Mapping word → debiased unit vector.
    """
    g = normalize(bias_direction)
    debiased = {}
    for w in words:
        if w not in embeddings:
            continue
        v = embeddings[w]
        v_orth = v - np.dot(v, g) * g
        debiased[w] = normalize(v_orth)
    return debiased


# ---------------------------------------------------------------------------
# Step 3 – Equalize gender word pairs
# ---------------------------------------------------------------------------

def equalize(
    equality_sets: list,
    bias_direction: np.ndarray,
    embeddings: dict,
) -> dict:
    """
    Equalize word pairs so they are equidistant from all neutralized words.

    Algorithm (Bolukbasi et al., eq. 9):

        y   = drop(normalize(a + b), g)   # gender-neutral midpoint
        z   = sqrt(1 − ‖y‖²)             # ensures ‖z·g + y‖ = 1
        if g·(a − b) < 0:  z ← −z        # sign tracks original gender
        a_new = normalize( z·g + y)
        b_new = normalize(−z·g + y)

    Parameters
    ----------
    equality_sets : list of (str, str)
        Gendered pairs to be equalized (e.g. ``[("aunt","uncle"), ...]``).
        Pairs where either word is absent in the vocabulary are skipped.
    bias_direction : np.ndarray, shape (d,)
    embeddings : dict[str, np.ndarray]

    Returns
    -------
    dict[str, np.ndarray]
        Mapping word → equalized unit vector for every word that could be processed.
    """
    g = normalize(bias_direction)
    equalized = {}

    for w1, w2 in equality_sets:
        if w1 not in embeddings or w2 not in embeddings:
            continue
        v1 = normalize(embeddings[w1])
        v2 = normalize(embeddings[w2])

        # Gender-neutral midpoint: normalize the sum, then drop the gender direction
        mu_norm = normalize(v1 + v2)
        y = mu_norm - np.dot(mu_norm, g) * g   # NOT re-normalized (by design)

        # Magnitude of gender component so that ‖z·g + y‖ = 1
        z = float(np.sqrt(max(0.0, 1.0 - np.dot(y, y))))

        # Sign: which word has the positive gender projection?
        if np.dot(g, v1 - v2) < 0:
            z = -z

        equalized[w1] = normalize(z * g + y)
        equalized[w2] = normalize(-z * g + y)

    return equalized


# ---------------------------------------------------------------------------
# Bias metrics
# ---------------------------------------------------------------------------

def direct_bias(
    words: list,
    bias_direction: np.ndarray,
    embeddings: dict,
    c: int = 1,
) -> float:
    """
    DirectBias metric (Bolukbasi et al., Definition 1).

        DirectBias_c(N, g) = (1/|N|) · Σ_{w ∈ N} |cos(w, g)|^c

    c=1 (default) is the standard linear version used in the paper.
    Lower values indicate less gender bias in the word set N.

    Parameters
    ----------
    words : list of str
        Gender-neutral words to evaluate (e.g. occupations).
    bias_direction : np.ndarray, shape (d,)
    embeddings : dict[str, np.ndarray]
    c : int
        Exponent (default 1).

    Returns
    -------
    float
    """
    g = normalize(bias_direction)
    scores = [
        abs(float(np.dot(normalize(embeddings[w]), g))) ** c
        for w in words
        if w in embeddings
    ]
    return float(np.mean(scores)) if scores else 0.0


def weat_effect_size(
    target_X: list,
    target_Y: list,
    attr_A: list,
    attr_B: list,
    embeddings: dict,
) -> float:
    """
    WEAT effect size d (Caliskan et al. 2017), used as a secondary bias metric.

        s(w, A, B) = mean_{a∈A} cos(w, a) − mean_{b∈B} cos(w, b)
        d = [mean_{x∈X} s(x,A,B) − mean_{y∈Y} s(y,A,B)] / std_{w∈X∪Y} s(w,A,B)

    Positive d means X is more associated with A (relative to B) than Y is.
    Typical usage: X=male words, Y=female words, A=career, B=family.

    Parameters
    ----------
    target_X, target_Y : list of str
        Two sets of target words.
    attr_A, attr_B : list of str
        Two sets of attribute words.
    embeddings : dict[str, np.ndarray]

    Returns
    -------
    float  (0.0 if insufficient vocabulary coverage)
    """
    def _s(w, A, B):
        v = normalize(embeddings[w])
        a_sims = [float(np.dot(v, normalize(embeddings[a]))) for a in A if a in embeddings]
        b_sims = [float(np.dot(v, normalize(embeddings[b]))) for b in B if b in embeddings]
        if not a_sims or not b_sims:
            return 0.0
        return float(np.mean(a_sims)) - float(np.mean(b_sims))

    X_p = [w for w in target_X if w in embeddings]
    Y_p = [w for w in target_Y if w in embeddings]
    if not X_p or not Y_p:
        return 0.0

    s_X = [_s(w, attr_A, attr_B) for w in X_p]
    s_Y = [_s(w, attr_A, attr_B) for w in Y_p]
    all_s = s_X + s_Y
    std_all = float(np.std(all_s))

    if std_all < 1e-10:
        return 0.0
    return float((np.mean(s_X) - np.mean(s_Y)) / std_all)
