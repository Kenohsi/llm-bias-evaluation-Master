"""
Real-embedding application of Bolukbasi et al. (2016) hard debiasing.

Downloads word2vec-google-news-300 via gensim (~1.6 GB, cached in ~/gensim-data/).
Applies the three-step hard debiasing algorithm to a working copy of the
relevant vectors and computes DirectBias and WEAT effect size before / after.

Outputs
-------
results/debiasing_word2vec_results.csv
    Columns: metric, before, after, reduction_pct

results/debiasing_word2vec.png
    Two-panel bar chart (DirectBias + WEAT) analogous to plot_debiasing.py

results/top_shifted_occupations.md
    Markdown table: top-10 professions by absolute bias shift (|cos| change)

Usage
-----
    python debiasing_word2vec.py

Runtime: ~30 seconds after first download.
"""

import sys
import os
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gensim.downloader as api

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
# Word lists (aligned with Bolukbasi et al. 2016, Fig. 1 / Table 1)
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
    ("female", "male"),
]

EQUALITY_SETS_RAW = [
    ("monastery", "convent"), ("spokesman", "spokeswoman"),
    ("dad", "mom"), ("men", "women"), ("councilman", "councilwoman"),
    ("grandpa", "grandma"), ("grandsons", "granddaughters"),
    ("prostate_cancer", "ovarian_cancer"),
    ("testosterone", "estrogen"), ("uncle", "aunt"),
    ("wives", "husbands"), ("father", "mother"),
    ("he", "she"), ("boy", "girl"), ("boys", "girls"),
    ("brother", "sister"), ("brothers", "sisters"),
    ("businessman", "businesswoman"), ("chairman", "chairwoman"),
    ("colt", "filly"), ("congressman", "congresswoman"),
    ("dad", "mom"), ("dads", "moms"), ("dudes", "gals"),
    ("ex_girlfriend", "ex_boyfriend"),
    ("fatherhood", "motherhood"), ("fathers", "mothers"),
    ("fella", "granny"), ("fraternity", "sorority"),
    ("gelding", "mare"), ("gentleman", "lady"),
    ("gentlemen", "ladies"),
]

# WEAT 6 from Caliskan et al. 2017 (male/female × career/family)
WEAT_MALE   = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
WEAT_FEMALE = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
WEAT_CAREER = ["executive", "management", "professional", "corporation",
               "salary", "office", "business", "career"]
WEAT_FAMILY = ["home", "parents", "children", "family",
               "wedding", "marriage", "domestic", "household"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _load_model():
    """Download (first run only) and return the word2vec-google-news-300 model."""
    print("Loading word2vec-google-news-300 …", flush=True)
    print("(~1.6 GB download on first run, cached in ~/gensim-data/)", flush=True)
    model = api.load("word2vec-google-news-300")
    print(f"Vocabulary size: {len(model.key_to_index):,}", flush=True)
    return model


def _filter_pairs(pairs, vocab):
    """Return only those pairs where both words are in the vocabulary."""
    return [(a, b) for a, b in pairs if a in vocab and b in vocab]


def _filter_words(words, vocab):
    """Return only words present in the vocabulary."""
    return [w for w in words if w in vocab]


def _extract_embeddings(words, model):
    """Build a plain dict {word: np.ndarray} working copy from the model."""
    return {w: model[w].astype(np.float64) for w in words if w in model.key_to_index}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = _load_model()
    vocab = model.key_to_index

    # ---- Gather all relevant words ----------------------------------------
    def_pairs   = _filter_pairs(DEFINITIONAL_PAIRS, vocab)
    eq_pairs    = _filter_pairs(EQUALITY_SETS_RAW,  vocab)
    # deduplicate equality pairs (list contains duplicates)
    seen = set()
    eq_pairs_unique = []
    for a, b in eq_pairs:
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            eq_pairs_unique.append((a, b))
    eq_pairs = eq_pairs_unique

    profs   = _filter_words(PROFESSIONS,  vocab)
    wm      = _filter_words(WEAT_MALE,    vocab)
    wf      = _filter_words(WEAT_FEMALE,  vocab)
    wc      = _filter_words(WEAT_CAREER,  vocab)
    wfam    = _filter_words(WEAT_FAMILY,  vocab)

    print(f"\nUsable definitional pairs : {len(def_pairs)}/{len(DEFINITIONAL_PAIRS)}")
    print(f"Usable equality sets      : {len(eq_pairs)}")
    print(f"Usable professions        : {len(profs)}/{len(PROFESSIONS)}")
    print(f"WEAT coverage             : {len(wm)}/{len(WEAT_MALE)} male, "
          f"{len(wf)}/{len(WEAT_FEMALE)} female, "
          f"{len(wc)}/{len(WEAT_CAREER)} career, "
          f"{len(wfam)}/{len(WEAT_FAMILY)} family")

    # ---- Build working-copy embeddings (never modifies the gensim model) --
    all_words = (
        {w for p in def_pairs for w in p}
        | {w for p in eq_pairs for w in p}
        | set(profs) | set(wm) | set(wf) | set(wc) | set(wfam)
    )
    embeddings = _extract_embeddings(all_words, model)

    # ---- Step 1: Identify bias subspace -----------------------------------
    print("\nIdentifying gender bias subspace …", flush=True)
    bias_dir = identify_bias_subspace(def_pairs, embeddings)

    # ---- Metrics BEFORE debiasing -----------------------------------------
    db_before   = direct_bias(profs, bias_dir, embeddings)
    weat_before = weat_effect_size(wm, wf, wc, wfam, embeddings)

    # Per-profession bias scores before
    g = normalize(bias_dir)
    bias_before_per_word = {
        w: abs(float(np.dot(normalize(embeddings[w]), g)))
        for w in profs
    }

    print(f"\nBEFORE: DirectBias={db_before:.4f}  WEAT d={weat_before:+.4f}")

    # ---- Step 2: Neutralize profession words ------------------------------
    print("Neutralizing profession words …", flush=True)
    debiased = neutralize(profs, bias_dir, embeddings)
    emb_debiased = dict(embeddings)
    emb_debiased.update(debiased)

    # ---- Step 3: Equalize equality pairs ----------------------------------
    print("Equalizing word pairs …", flush=True)
    equalized = equalize(eq_pairs, bias_dir, emb_debiased)
    emb_debiased.update(equalized)

    # ---- Metrics AFTER debiasing ------------------------------------------
    db_after   = direct_bias(profs, bias_dir, emb_debiased)
    weat_after = weat_effect_size(wm, wf, wc, wfam, emb_debiased)

    # Per-profession bias scores after
    bias_after_per_word = {
        w: abs(float(np.dot(normalize(emb_debiased[w]), g)))
        for w in profs
    }

    print(f"AFTER:  DirectBias={db_after:.4f}  WEAT d={weat_after:+.4f}")

    # ---- Reductions -------------------------------------------------------
    def _pct_reduction(before, after):
        return (1.0 - abs(after) / abs(before)) * 100 if abs(before) > 1e-10 else 0.0

    red_db   = _pct_reduction(db_before, db_after)
    red_weat = _pct_reduction(weat_before, weat_after)

    print(f"\nReduction: DirectBias {red_db:.1f} %  |  WEAT {red_weat:.1f} %")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "debiasing_word2vec_results.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "before", "after", "reduction_pct"])
        writer.writerow(["DirectBias",      f"{db_before:.6f}",   f"{db_after:.6f}",   f"{red_db:.2f}"])
        writer.writerow(["WEAT_effect_size", f"{weat_before:.6f}", f"{weat_after:.6f}", f"{red_weat:.2f}"])
    print(f"Saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Save top-10 shifted occupations (Markdown table)
    # -----------------------------------------------------------------------
    shifts = {
        w: bias_before_per_word[w] - bias_after_per_word.get(w, 0.0)
        for w in profs
    }
    top10 = sorted(shifts.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    md_path = os.path.join(RESULTS_DIR, "top_shifted_occupations.md")
    with open(md_path, "w") as fh:
        fh.write("# Top-10 Professions by Bias Shift\n\n")
        fh.write("Ranked by |cos(w,g)_before − cos(w,g)_after|.\n\n")
        fh.write("| Rank | Profession | Bias before | Bias after | Shift |\n")
        fh.write("|------|------------|-------------|------------|-------|\n")
        for rank, (word, shift) in enumerate(top10, 1):
            b4 = bias_before_per_word[word]
            af = bias_after_per_word.get(word, 0.0)
            fh.write(f"| {rank} | {word} | {b4:.4f} | {af:.4f} | {shift:+.4f} |\n")
    print(f"Saved: {md_path}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        "Hard Debiasing on word2vec-google-news-300\n"
        "Bolukbasi et al. (2016) – Real Embeddings",
        fontsize=11, fontweight="bold",
    )

    for ax, label, before, after, ylabel in [
        (axes[0], "DirectBias",      db_before,   db_after,   "DirectBias (c=1)"),
        (axes[1], "WEAT effect size", weat_before, weat_after, "WEAT d  (male–career vs. female–family)"),
    ]:
        vals = [before, after]
        colors = ["#e74c3c", "#2ecc71"]
        bars = ax.bar(["Before", "After"], vals, color=colors,
                      width=0.4, edgecolor="black", linewidth=0.8)
        for bar, v in zip(bars, vals):
            yoff = abs(v) * 0.07 * (1 if v >= 0 else -1)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + yoff, f"{v:+.4f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        red = _pct_reduction(before, after)
        ax.set_title(f"{label}\n(reduction: {red:.1f} %)", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", linewidth=0.8)
        margin = max(abs(before), abs(after)) * 0.4
        ax.set_ylim(min(0, min(vals)) - margin, max(vals) + margin * 1.5)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "debiasing_word2vec.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  word2vec-google-news-300 Debiasing Summary")
    print("=" * 60)
    print(f"  {'Metric':<22} {'Before':>9}  {'After':>9}  {'Reduction':>10}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*9}")
    print(f"  {'DirectBias':<22} {db_before:>9.4f}  {db_after:>9.4f}  {red_db:>8.1f} %")
    print(f"  {'WEAT effect size':<22} {weat_before:>9.4f}  {weat_after:>9.4f}  {red_weat:>8.1f} %")
    print("=" * 60)
    print("\nTop-10 professions by bias shift:")
    for rank, (word, shift) in enumerate(top10, 1):
        b4 = bias_before_per_word[word]
        af = bias_after_per_word.get(word, 0.0)
        print(f"  {rank:2d}. {word:<18}  {b4:.4f} → {af:.4f}  (Δ {shift:+.4f})")


if __name__ == "__main__":
    main()
