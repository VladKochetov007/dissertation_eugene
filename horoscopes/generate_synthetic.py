"""
Synthetic data generator for pipeline validation.

Two models:
  - null:   personality traits are iid N(0,1), no birth-timing signal
  - signal: traits have a smooth seasonal component (sinusoidal) plus noise,
            with weak discontinuities near zodiac boundaries
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from utils import BIG5_TRAITS, ZODIAC_BOUNDARIES, doy_to_zodiac, N_SIGNS


def generate(
    n: int = 10_000,
    model: str = "null",       # "null" | "signal" | "boundary"
    seed: int = 42,
    signal_strength: float = 0.15,  # fraction of total variance explained by birth timing
) -> pd.DataFrame:
    """
    Generate synthetic respondents.

    Returns DataFrame with columns:
      doy              day of year (1–365, uniform)
      E, A, C, N, O   Big Five domain scores (mean 0, std 1 scale)
      zodiac_idx       zodiac sign index (0–11)
    """
    rng = np.random.default_rng(seed)

    # Day of year: uniform on 1–365
    doy = rng.integers(1, 366, size=n)
    theta = 2 * np.pi * (doy - 1) / 365  # angular position in year

    # Pure noise component
    noise = rng.standard_normal((n, len(BIG5_TRAITS)))

    if model == "null":
        traits = noise

    elif model == "signal":
        # Smooth sinusoidal seasonal component (different phase per trait)
        phases = np.array([0.0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3])
        seasonal = np.column_stack([
            np.sin(theta + phase) for phase in phases
        ])
        # Mix: signal_strength controls how much seasonal variance contributes
        traits = np.sqrt(signal_strength) * seasonal + np.sqrt(1 - signal_strength) * noise

    elif model == "boundary":
        # Same as signal but add discontinuous jumps at true zodiac boundaries
        phases = np.array([0.0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3])
        seasonal = np.column_stack([
            np.sin(theta + phase) for phase in phases
        ])
        # Add jump at each zodiac boundary
        zodiac_idx = doy_to_zodiac(doy)
        # Each sign gets a small fixed offset drawn from N(0, jump_sd)
        jump_sd = np.sqrt(signal_strength / 2)
        sign_offsets = rng.normal(0, jump_sd, (N_SIGNS, len(BIG5_TRAITS)))
        jumps = sign_offsets[zodiac_idx]
        traits = (
            np.sqrt(signal_strength / 2) * seasonal
            + jumps
            + np.sqrt(1 - signal_strength) * noise
        )
    else:
        raise ValueError(f"Unknown model: {model!r}")

    df = pd.DataFrame(traits, columns=BIG5_TRAITS)
    df["doy"] = doy
    df["zodiac_idx"] = doy_to_zodiac(doy)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic personality + birth-date data")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--model", choices=["null", "signal", "boundary"], default="null")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--signal-strength", type=float, default=0.15)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = generate(n=args.n, model=args.model, seed=args.seed, signal_strength=args.signal_strength)

    out = args.out or f"data/synthetic_{args.model}_n{args.n}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows → {out}")
    print(df[BIG5_TRAITS + ["doy", "zodiac_idx"]].describe().round(3))


if __name__ == "__main__":
    main()
