"""
Full pipeline runner: generate synthetic data → Phase 1 → 2 → 3 → 5.
Useful for validating the pipeline before real data arrives.

Usage:
  python run_pipeline.py --model null        # null hypothesis (no signal)
  python run_pipeline.py --model signal      # smooth seasonal signal
  python run_pipeline.py --model boundary    # signal + zodiac discontinuities
  python run_pipeline.py --data path/to/real.csv  # run on real data
"""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic import generate
from phase1_classifier import run_phase1
from phase2_clustering import run_phase2
from phase3_boundary import run_phase3
from phase4_continuity import run_phase4
from phase5_optimal_partition import run_phase5


def run_all(df: pd.DataFrame, args) -> dict:
    t0 = perf_counter()
    results = {}
    fast = getattr(args, "fast", False)

    print(f"\n{'#'*60}")
    print(f"# ZODIAC EMPIRICAL TAXONOMY PIPELINE")
    print(f"# N={len(df):,}  seed={args.seed}{'  [FAST MODE]' if fast else ''}")
    print(f"{'#'*60}")

    results["phase1"] = run_phase1(df, cv_folds=args.cv, n_permutations=args.permutations, seed=args.seed, fast=fast)
    results["phase2"] = run_phase2(df, n_random=args.random, seed=args.seed)
    results["phase3"] = run_phase3(df, n_random=args.random, cv=args.cv, seed=args.seed)
    results["phase4"] = run_phase4(df, bandwidth=args.bandwidth, n_permutations=args.permutations, seed=args.seed)
    results["phase5"] = run_phase5(df, bandwidth=args.bandwidth, n_permutations=args.permutations, seed=args.seed)

    elapsed = perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    p1 = results["phase1"]
    best = p1["best_classifier"]
    best_acc = p1["classifiers"][best]["accuracy"]
    print(f"Phase 1 accuracy ({best}): {best_acc:.4f}  ({best_acc/p1['chance_baseline']:.2f}x chance)"
          f"  η²={p1['eta_squared']:.5f}  p_perm={p1['permutation_p']:.4f}")
    p2 = results["phase2"]
    print(f"Phase 2 ARI(zodiac, k-means): {p2['ari_zodiac_kmeans']:.4f}"
          f"  p={p2['p_ari']:.4f}  inertia_ratio={p2['inertia_ratio']:.3f}")
    p3 = results["phase3"]
    print(f"Phase 3 accuracy (zodiac={p3['named_accuracies']['zodiac']:.4f}"
          f"  random_mean={p3['random_acc_mean']:.4f})  p={p3['p_zodiac_vs_random']:.4f}")
    p4 = results["phase4"]
    p4_summary = "  ".join(
        f"k={k}: MAD={v['observed_mad']:.1f}d p={v['p_convergence']:.4f}"
        for k, v in p4["scales"].items() if "error" not in v
    )
    print(f"Phase 4 smooth_p={p4['p_smooth']:.4f}  [{p4_summary}]")
    p5 = results["phase5"]
    print(f"Phase 5 MAD={p5['observed_mad']:.1f} days  p_convergence={p5['p_convergence']:.4f}"
          f"  (null median={p5['null_mad_median']:.1f})")
    print(f"\nTotal time: {elapsed:.0f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Full zodiac pipeline")
    parser.add_argument("--model", choices=["null", "signal", "boundary"],
                        default="null", help="Synthetic data model (ignored if --data provided)")
    parser.add_argument("--n", type=int, default=10_000, help="Synthetic N")
    parser.add_argument("--signal-strength", type=float, default=0.15)
    parser.add_argument("--data", type=str, default=None, help="Path to real CSV")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--random", type=int, default=500)
    parser.add_argument("--bandwidth", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: LR only, fewer permutations (for smoke testing)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data)
        label = Path(args.data).stem
    else:
        print(f"\nGenerating synthetic data: model={args.model}, N={args.n:,}")
        df = generate(n=args.n, model=args.model, seed=args.seed,
                      signal_strength=args.signal_strength)
        label = f"synthetic_{args.model}_n{args.n}"

    results = run_all(df, args)

    out = args.out or f"data/{label}_pipeline_results.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON
    def to_json(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Not serializable: {type(obj)}")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=to_json)
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    main()
