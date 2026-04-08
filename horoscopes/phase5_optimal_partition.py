"""
Phase 5: Optimal Partition Discovery — The Centrepiece Analysis

Let the data discover the optimal 12-segment partition of the calendar year
and measure its convergence with the traditional zodiac boundaries.

Algorithm:
  1. For each day-of-year d, compute mean trait vector (kernel-smoothed over ±bandwidth days)
  2. Find the 11 cut-points that minimise total within-segment variance (dynamic programming)
  3. Compute Mean Absolute Deviation (MAD) between data-driven cuts and zodiac boundaries
  4. Permutation test: compare MAD against 10,000 random sets of 11 cut-points
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import BIG5_TRAITS, N_SIGNS, ZODIAC_BOUNDARIES


# --- Kernel-smoothed trajectory ---

def smooth_trajectory(
    doy: np.ndarray,
    traits: np.ndarray,
    bandwidth: int = 7,
    min_count: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each day d in [1, 365], compute the kernel-weighted mean trait vector
    using a Gaussian kernel with the given bandwidth (in days), treating the
    year as circular.

    Returns:
      days: array of days with enough data (>= min_count)
      means: (len(days), n_traits) smoothed mean trait vectors
    """
    n_traits = traits.shape[1]
    days = np.arange(1, 366)
    smoothed = np.full((365, n_traits), np.nan)

    for d in days:
        # Circular distance
        diff = np.abs(doy - d)
        diff = np.minimum(diff, 365 - diff)
        weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
        total_w = weights.sum()
        if total_w < min_count:
            continue
        smoothed[d - 1] = (weights[:, None] * traits).sum(axis=0) / total_w

    valid = ~np.isnan(smoothed[:, 0])
    return days[valid], smoothed[valid]


# --- Dynamic programming optimal segmentation ---

def optimal_segmentation(
    trajectory: np.ndarray,
    k: int = N_SIGNS,
) -> tuple[list[int], float]:
    """
    Find the k-1 cut-points (in trajectory indices) that minimise total
    within-segment variance via dynamic programming (1D optimal segmentation).

    trajectory: (T, d) array of smoothed mean vectors for T time points
    k: number of segments

    Returns:
      cut_indices: list of k-1 cut point indices (0-indexed into trajectory)
      total_cost: minimum within-segment variance
    """
    T = len(trajectory)

    # Precompute within-segment cost for all (i, j) pairs using prefix sums
    # cost(i, j) = sum of squared deviations of trajectory[i:j] from its mean
    # = sum_{t=i}^{j-1} ||x_t - mean(x_i..x_{j-1})||^2
    # Efficiently: = sum ||x_t||^2 - (sum x_t)^2 / (j-i)

    sq_sum = (trajectory ** 2).sum(axis=1)          # (T,)
    cum_sq = np.concatenate([[0], np.cumsum(sq_sum)])  # (T+1,)
    cum_x  = np.vstack([np.zeros(trajectory.shape[1]),
                         np.cumsum(trajectory, axis=0)])  # (T+1, d)

    def seg_cost(i, j):
        if j <= i:
            return 0.0
        length = j - i
        sum_x = cum_x[j] - cum_x[i]           # (d,)
        sum_sq_x = cum_sq[j] - cum_sq[i]
        return float(sum_sq_x - np.dot(sum_x, sum_x) / length)

    # DP table: dp[s][t] = min cost to segment trajectory[0:t] into s segments
    INF = float("inf")
    dp = [[INF] * (T + 1) for _ in range(k + 1)]
    split = [[0] * (T + 1) for _ in range(k + 1)]

    dp[0][0] = 0.0
    for s in range(1, k + 1):
        for t in range(s, T + 1):
            for m in range(s - 1, t):
                cost = dp[s - 1][m] + seg_cost(m, t)
                if cost < dp[s][t]:
                    dp[s][t] = cost
                    split[s][t] = m

    # Backtrack to find cut points
    cuts = []
    t = T
    for s in range(k, 0, -1):
        m = split[s][t]
        if s > 1:
            cuts.append(m)
        t = m
    cuts = sorted(cuts)

    return cuts, dp[k][T]


def cuts_to_doy(cut_indices: list[int], valid_days: np.ndarray) -> list[int]:
    """Convert trajectory cut indices back to day-of-year values."""
    return [int(valid_days[i]) for i in cut_indices]


def mad_to_zodiac(data_cuts_doy: list[int]) -> float:
    """
    Mean Absolute Deviation between data-driven cut-points and
    nearest zodiac boundary (in days, circular distance on [1,365]).
    """
    if len(data_cuts_doy) != len(ZODIAC_BOUNDARIES):
        raise ValueError(f"Expected {len(ZODIAC_BOUNDARIES)} cuts, got {len(data_cuts_doy)}")
    deviations = []
    for dc in data_cuts_doy:
        dists = [abs(dc - zb) for zb in ZODIAC_BOUNDARIES]
        dists_circ = [min(d, 365 - d) for d in dists]
        deviations.append(min(dists_circ))
    return float(np.mean(deviations))


def run_phase5(
    df: pd.DataFrame,
    bandwidth: int = 7,
    k: int = N_SIGNS,
    n_permutations: int = 10_000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    traits = df[BIG5_TRAITS].values.astype(float)
    doy = df["doy"].values.astype(int)
    n = len(df)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 5: Optimal Partition Discovery")
        print(f"N = {n:,}  |  bandwidth = {bandwidth} days  |  k = {k}")
        print(f"Zodiac boundaries: {ZODIAC_BOUNDARIES}")
        print(f"{'='*60}")

    # Step 1: Smooth trajectory
    valid_days, smoothed = smooth_trajectory(doy, traits, bandwidth=bandwidth)
    if verbose:
        print(f"\n→ Smoothed trajectory: {len(valid_days)} valid days out of 365")

    # Step 2: Dynamic programming optimal segmentation
    if verbose:
        print(f"→ Running DP segmentation (k={k})...")
    cut_indices, total_cost = optimal_segmentation(smoothed, k=k)
    data_cuts_doy = cuts_to_doy(cut_indices, valid_days)

    if verbose:
        print(f"\n→ Data-driven cut-points (day of year):")
        print(f"   {data_cuts_doy}")
        print(f"→ Zodiac boundaries:")
        print(f"   {ZODIAC_BOUNDARIES}")

    # Step 3: MAD to zodiac
    observed_mad = mad_to_zodiac(data_cuts_doy)
    if verbose:
        print(f"\n→ Mean Absolute Deviation to zodiac: {observed_mad:.1f} days")

    # Step 4: Permutation test — null distribution of MAD under random cut-points
    if verbose:
        print(f"→ Permutation test ({n_permutations:,} random sets)...")
    rng = np.random.default_rng(seed)
    null_mads = np.empty(n_permutations)
    for i in range(n_permutations):
        random_cuts = sorted(rng.integers(1, 365, size=k - 1).tolist())
        null_mads[i] = mad_to_zodiac(random_cuts)

    p_value = float(np.mean(null_mads <= observed_mad))  # one-sided: is observed MAD unusually SMALL?

    if verbose:
        p50 = np.percentile(null_mads, 50)
        p5 = np.percentile(null_mads, 5)
        print(f"\n→ Null MAD: median={p50:.1f}  5th pctile={p5:.1f}")
        print(f"   Observed MAD: {observed_mad:.1f}")
        print(f"   Percentile of observed: {np.mean(null_mads <= observed_mad)*100:.1f}th")
        print(f"   p (one-sided, convergence): {p_value:.4f}")

        if p_value < 0.05:
            print(f"\n→ RESULT: Data-driven boundaries CONVERGE on zodiac (p={p_value:.4f})")
        else:
            print(f"\n→ RESULT: NULL — data-driven boundaries do not cluster near zodiac (p={p_value:.4f})")

        # Per-cut alignment
        print(f"\n→ Per-cut deviation from nearest zodiac boundary:")
        for i, (dc, zb) in enumerate(zip(sorted(data_cuts_doy), ZODIAC_BOUNDARIES)):
            dev = min(abs(dc - zb), 365 - abs(dc - zb))
            print(f"   Cut {i+1}: day {dc:3d}  →  nearest zodiac: day {zb:3d}  (±{dev} days)")

    return {
        "n": n,
        "bandwidth": bandwidth,
        "k": k,
        "valid_days": int(len(valid_days)),
        "data_cuts_doy": data_cuts_doy,
        "zodiac_boundaries": ZODIAC_BOUNDARIES,
        "observed_mad": observed_mad,
        "null_mad_median": float(np.median(null_mads)),
        "null_mad_p5": float(np.percentile(null_mads, 5)),
        "null_mad_p95": float(np.percentile(null_mads, 95)),
        "p_convergence": p_value,
        "observed_percentile": float(np.mean(null_mads <= observed_mad) * 100),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 5: optimal partition discovery")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--bandwidth", type=int, default=7,
                        help="Kernel smoothing bandwidth in days")
    parser.add_argument("--k", type=int, default=N_SIGNS)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    summary = run_phase5(
        df,
        bandwidth=args.bandwidth,
        k=args.k,
        n_permutations=args.permutations,
        seed=args.seed,
    )

    out = args.out or args.data.replace(".csv", "_phase5_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
