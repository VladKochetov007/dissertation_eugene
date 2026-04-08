"""
Phase 4: Continuity Analysis — Smooth Gradient vs. Discontinuities

Two questions:
  1. Is personality variation over birth-day-of-year smooth or discontinuous?
  2. If discontinuous, do the breakpoints align with traditional zodiac boundaries?

Method:
  - Kernel-smooth each trait over day-of-year (circular)
  - Run PELT changepoint detection on the multivariate smoothed trajectory
  - At multiple scales: k=4 (elements/triplicities), k=12 (signs), k=36 (decans)
  - For each k: compute MAD between detected breakpoints and nearest zodiac boundary
  - Permutation test to assess whether alignment exceeds chance

Zodiac reference boundaries at each scale:
  k=4  (elements — Fire/Earth/Air/Water):
       Aries(80), Cancer(172), Libra(266), Capricorn(1→356)
       → start-days: [1, 80, 172, 266]  (4 segments, 3 internal boundaries)
  k=12 (signs): ZODIAC_BOUNDARIES (11 boundaries)
  k=36 (decans, ~10-day sub-signs): every ~10 days within each sign (35 boundaries)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).parent))
from utils import BIG5_TRAITS, N_SIGNS, ZODIAC_BOUNDARIES


# --- Zodiac reference boundaries at multiple scales ---

# k=4: four classical elements (Fire, Earth, Air, Water triplicities)
# Fire:  Aries(80), Leo(204), Sagittarius(326)
# Earth: Taurus(110), Virgo(235), Capricorn(1/356)
# Air:   Gemini(141), Libra(266), Aquarius(20)
# Water: Cancer(172), Scorpio(296), Pisces(50)
# Element-group boundaries (start of each triplicity group of 3 signs):
ELEMENT_BOUNDARIES = [80, 172, 266]   # Aries, Cancer, Libra start days (3 internal cuts → 4 segments)

# k=36: decans (each sign split into 3 ~10-day periods)
# Approximate: 365/36 ≈ 10.1 days each, starting from day 1
DECAN_BOUNDARIES = [int(round(1 + i * 365 / 36)) for i in range(1, 36)]  # 35 boundaries

SCALE_BOUNDARIES = {
    4:  ELEMENT_BOUNDARIES,
    12: ZODIAC_BOUNDARIES,
    36: DECAN_BOUNDARIES,
}


# --- Smooth trajectory (circular kernel) ---

def smooth_circular(
    doy: np.ndarray,
    traits: np.ndarray,
    bandwidth: int = 7,
) -> np.ndarray:
    """
    Returns (365, n_traits) kernel-smoothed mean trajectory for days 1..365.
    Uses Gaussian kernel with circular wrapping.
    """
    n_traits = traits.shape[1]
    smoothed = np.zeros((365, n_traits))

    for d in range(1, 366):
        diff = np.abs(doy - d)
        diff = np.minimum(diff, 365 - diff)
        weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
        total_w = weights.sum()
        if total_w == 0:
            continue
        smoothed[d - 1] = (weights[:, None] * traits).sum(axis=0) / total_w

    return smoothed  # shape (365, n_traits)


# --- Smoothness test: variance of second derivative ---

def smoothness_ratio(trajectory: np.ndarray) -> float:
    """
    Ratio of second-derivative variance to first-derivative variance.
    High ratio = more abrupt changes (discontinuous); low = smoother.
    Returns mean across traits.
    """
    ratios = []
    for col in range(trajectory.shape[1]):
        y = trajectory[:, col]
        d1 = np.diff(y)
        d2 = np.diff(d1)
        if d1.var() == 0:
            continue
        ratios.append(d2.var() / d1.var())
    return float(np.mean(ratios))


def smoothness_permutation_test(
    trajectory: np.ndarray,
    observed_ratio: float,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """
    Null: shuffle trajectory rows → destroy temporal structure.
    Test whether observed smoothness ratio is lower than null (i.e. data IS smoother).
    """
    rng = np.random.default_rng(seed)
    null_ratios = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(trajectory)
        null_ratios[i] = smoothness_ratio(perm)
    p_smooth = float(np.mean(null_ratios <= observed_ratio))
    return p_smooth, null_ratios


# --- PELT changepoint detection ---

def detect_changepoints(
    trajectory: np.ndarray,
    n_breakpoints: int,
    model: str = "rbf",
) -> list[int]:
    """
    Detect n_breakpoints in the multivariate trajectory using PELT.
    Returns list of breakpoint indices (0-indexed into trajectory rows, i.e. days 1..365).
    """
    # ruptures expects shape (n_samples, n_features)
    algo = rpt.Pelt(model=model, min_size=5).fit(trajectory)
    # PELT finds optimal number; force exactly n_breakpoints using Binseg
    algo_binseg = rpt.Binseg(model=model, min_size=5).fit(trajectory)
    result = algo_binseg.predict(n_bkps=n_breakpoints)
    # result is list of end-indices (1-indexed); convert to start-of-new-segment (day numbers)
    breakpoints_idx = [r - 1 for r in result[:-1]]  # drop the final end marker
    return breakpoints_idx


def idx_to_doy(indices: list[int]) -> list[int]:
    """Convert 0-indexed trajectory positions to day-of-year (1-indexed)."""
    return [i + 1 for i in indices]


# --- MAD alignment to reference boundaries ---

def mad_alignment(
    detected_doy: list[int],
    reference_boundaries: list[int],
    scale: int,
) -> float:
    """
    For each detected breakpoint, find the nearest reference boundary
    (with circular wrapping on [1, 365]).
    Returns mean absolute deviation.
    """
    if len(detected_doy) != len(reference_boundaries):
        # Align: for each detected point, nearest reference
        deviations = []
        for d in detected_doy:
            dists = [min(abs(d - r), 365 - abs(d - r)) for r in reference_boundaries]
            deviations.append(min(dists))
        return float(np.mean(deviations))
    else:
        deviations = []
        for d, r in zip(sorted(detected_doy), sorted(reference_boundaries)):
            deviations.append(min(abs(d - r), 365 - abs(d - r)))
        return float(np.mean(deviations))


def permutation_mad_test(
    detected_doy: list[int],
    reference_boundaries: list[int],
    observed_mad: float,
    n_permutations: int = 5000,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """
    Null: k random breakpoints uniformly on [1, 365].
    p-value = fraction of null MADs ≤ observed (one-sided: convergence).
    """
    k = len(detected_doy)
    rng = np.random.default_rng(seed)
    null_mads = np.array([
        mad_alignment(
            sorted(rng.integers(1, 366, size=k).tolist()),
            reference_boundaries,
            k,
        )
        for _ in range(n_permutations)
    ])
    p_value = float(np.mean(null_mads <= observed_mad))
    return p_value, null_mads


# --- Main phase runner ---

def run_phase4(
    df: pd.DataFrame,
    bandwidth: int = 7,
    n_permutations: int = 5000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    traits = df[BIG5_TRAITS].values.astype(float)
    doy = df["doy"].values.astype(int)
    n = len(df)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 4: Continuity Analysis")
        print(f"N = {n:,}  |  bandwidth = {bandwidth}  |  permutations = {n_permutations:,}")
        print(f"{'='*60}")

    # Step 1: Smooth trajectory
    trajectory = smooth_circular(doy, traits, bandwidth=bandwidth)

    # Step 2: Smoothness test
    obs_ratio = smoothness_ratio(trajectory)
    p_smooth, null_ratios = smoothness_permutation_test(
        trajectory, obs_ratio, n_permutations=min(n_permutations, 1000), seed=seed
    )
    if verbose:
        print(f"\n→ Smoothness test (2nd-derivative variance ratio)")
        print(f"   Observed ratio: {obs_ratio:.4f}")
        print(f"   Null median:    {np.median(null_ratios):.4f}")
        print(f"   p (data is smooth): {p_smooth:.4f}")
        if p_smooth < 0.05:
            verdict = "SMOOTH — trajectory varies continuously, favours seasonal model"
        elif p_smooth > 0.95:
            verdict = "DISCONTINUOUS — abrupt changes present, favours categorical sign model"
        else:
            verdict = "MIXED — moderate smoothness"
        print(f"   Verdict: {verdict}")

    # Step 3: Multi-scale changepoint detection
    scale_results = {}
    scales = [4, 12, 36]

    for k in scales:
        n_bkps = k - 1
        ref_boundaries = SCALE_BOUNDARIES[k]

        try:
            bkp_indices = detect_changepoints(trajectory, n_bkps)
            bkp_doy = idx_to_doy(bkp_indices)
        except Exception as e:
            if verbose:
                print(f"\n→ Scale k={k}: changepoint detection failed ({e})")
            scale_results[k] = {"error": str(e)}
            continue

        observed_mad = mad_alignment(bkp_doy, ref_boundaries, k)
        p_val, null_mads = permutation_mad_test(
            bkp_doy, ref_boundaries, observed_mad,
            n_permutations=n_permutations, seed=seed
        )

        scale_results[k] = {
            "n_breakpoints": n_bkps,
            "detected_doy": sorted(bkp_doy),
            "reference_boundaries": sorted(ref_boundaries),
            "observed_mad": observed_mad,
            "null_mad_median": float(np.median(null_mads)),
            "null_mad_p5": float(np.percentile(null_mads, 5)),
            "p_convergence": p_val,
            "observed_percentile": float(np.mean(null_mads <= observed_mad) * 100),
        }

        if verbose:
            scale_label = {4: "elements (k=4)", 12: "signs (k=12)", 36: "decans (k=36)"}[k]
            print(f"\n→ Scale: {scale_label}")
            print(f"   Detected breakpoints (day of year): {sorted(bkp_doy)}")
            print(f"   Reference boundaries:               {sorted(ref_boundaries)}")
            print(f"   MAD = {observed_mad:.1f} days  "
                  f"(null median = {np.median(null_mads):.1f},  5th pctile = {np.percentile(null_mads, 5):.1f})")
            print(f"   p (convergence) = {p_val:.4f}  "
                  f"[{np.mean(null_mads <= observed_mad)*100:.1f}th percentile of null]")

            if p_val < 0.05:
                print(f"   ✓ Breakpoints CONVERGE on {scale_label} boundaries")
            else:
                print(f"   ✗ No significant convergence at {scale_label} scale")

            # Per-cut detail for k=12
            if k == 12:
                print(f"\n   Per-cut alignment (k=12):")
                for d, r in zip(sorted(bkp_doy), sorted(ref_boundaries)):
                    dev = min(abs(d - r), 365 - abs(d - r))
                    print(f"     Day {d:3d}  →  nearest zodiac boundary: {r:3d}  (±{dev} days)")

    summary = {
        "n": n,
        "bandwidth": bandwidth,
        "smoothness_ratio": obs_ratio,
        "p_smooth": p_smooth,
        "null_smooth_median": float(np.median(null_ratios)),
        "scales": scale_results,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 4 Summary")
        print(f"{'='*60}")
        print(f"  Smoothness p={p_smooth:.4f}  (ratio={obs_ratio:.4f})")
        for k, res in scale_results.items():
            if "error" in res:
                continue
            sig = "✓" if res["p_convergence"] < 0.05 else "✗"
            print(f"  k={k:2d}: MAD={res['observed_mad']:.1f}d  p={res['p_convergence']:.4f}  {sig}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 4: continuity analysis")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--bandwidth", type=int, default=7)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    summary = run_phase4(df, bandwidth=args.bandwidth,
                         n_permutations=args.permutations, seed=args.seed)

    out = args.out or args.data.replace(".csv", "_phase4_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
