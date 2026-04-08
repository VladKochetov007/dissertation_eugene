"""
Phase 2: Taxonomy Quality — Zodiac vs. Optimal Clustering

Does the zodiac partition of personality space approximate the optimal 12-cluster partition?
Metric: Adjusted Rand Index (ARI) and Normalised Mutual Information (NMI).
Calibration: compare against 1,000 random 12-way partitions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import BIG5_TRAITS, N_SIGNS, ZODIAC_NAMES


def random_partition(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Random k-way partition of n items."""
    labels = np.tile(np.arange(k), n // k + 1)[:n]
    return rng.permutation(labels)


def run_phase2(
    df: pd.DataFrame,
    n_random: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    X = StandardScaler().fit_transform(df[BIG5_TRAITS].values.astype(float))
    y_zodiac = df["zodiac_idx"].values.astype(int)
    n = len(df)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 2: Taxonomy Quality")
        print(f"N = {n:,}  |  k = {N_SIGNS}  |  Random baselines = {n_random}")
        print(f"{'='*60}")

    # K-means clustering (k=12) on personality space (ignoring birth date)
    km = KMeans(n_clusters=N_SIGNS, n_init=20, random_state=seed)
    y_kmeans = km.fit_predict(X)

    ari_zodiac = adjusted_rand_score(y_zodiac, y_kmeans)
    nmi_zodiac = normalized_mutual_info_score(y_zodiac, y_kmeans, average_method="arithmetic")

    if verbose:
        print(f"\n→ ARI(zodiac, k-means):  {ari_zodiac:.4f}")
        print(f"   NMI(zodiac, k-means):  {nmi_zodiac:.4f}")

    # Random baselines
    rng = np.random.default_rng(seed)
    ari_random = np.empty(n_random)
    nmi_random = np.empty(n_random)
    for i in range(n_random):
        y_rand = random_partition(n, N_SIGNS, rng)
        ari_random[i] = adjusted_rand_score(y_rand, y_kmeans)
        nmi_random[i] = normalized_mutual_info_score(y_rand, y_kmeans, average_method="arithmetic")

    p_ari = np.mean(ari_random >= ari_zodiac)
    p_nmi = np.mean(nmi_random >= nmi_zodiac)

    if verbose:
        print(f"\n→ Random baseline ARI: mean={ari_random.mean():.4f}  std={ari_random.std():.4f}")
        print(f"   Random baseline NMI: mean={nmi_random.mean():.4f}  std={nmi_random.std():.4f}")
        print(f"\n→ p(ARI random ≥ ARI zodiac): {p_ari:.4f}")
        print(f"   p(NMI random ≥ NMI zodiac): {p_nmi:.4f}")

        verdict = "SUPPORT" if p_ari < 0.05 else ("MARGINAL" if p_ari < 0.15 else "NULL")
        print(f"\n→ Verdict: {verdict}")
        print(f"   (If ARI zodiac > random, the zodiac captures real personality cluster structure)")

    # K-means within-sign homogeneity: how well does zodiac align with dense regions?
    inertia_zodiac = 0.0
    for sign_idx in range(N_SIGNS):
        mask = y_zodiac == sign_idx
        if mask.sum() < 2:
            continue
        group = X[mask]
        centroid = group.mean(axis=0)
        inertia_zodiac += np.sum((group - centroid) ** 2)

    inertia_kmeans = km.inertia_

    if verbose:
        print(f"\n→ Within-cluster inertia:")
        print(f"   K-means (optimal):  {inertia_kmeans:.2f}")
        print(f"   Zodiac partition:   {inertia_zodiac:.2f}")
        print(f"   Ratio (zodiac/opt): {inertia_zodiac/inertia_kmeans:.3f}")
        print(f"   (1.0 = zodiac is as good as k-means; lower is better)")

    return {
        "n": n,
        "n_random": n_random,
        "ari_zodiac_kmeans": ari_zodiac,
        "nmi_zodiac_kmeans": nmi_zodiac,
        "ari_random_mean": float(ari_random.mean()),
        "ari_random_std": float(ari_random.std()),
        "nmi_random_mean": float(nmi_random.mean()),
        "nmi_random_std": float(nmi_random.std()),
        "p_ari": float(p_ari),
        "p_nmi": float(p_nmi),
        "inertia_kmeans": float(inertia_kmeans),
        "inertia_zodiac": float(inertia_zodiac),
        "inertia_ratio": float(inertia_zodiac / inertia_kmeans),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: taxonomy quality")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    summary = run_phase2(df, n_random=args.n_random, seed=args.seed)

    out = args.out or args.data.replace(".csv", "_phase2_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
