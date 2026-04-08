"""
Phase 1: Signal Detection — Classifier Benchmark

Train a multiclass classifier to predict zodiac sign from Big Five trait scores.
Primary metric: accuracy vs. 8.33% chance baseline (1/12).
Reports effect size (Cohen's d equivalent, η²) alongside p-values.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from utils import BIG5_TRAITS, CHANCE_BASELINE, N_SIGNS, ZODIAC_NAMES


def eta_squared_manova(X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute η² (eta-squared) for multivariate ANOVA:
    proportion of total variance in trait space explained by zodiac sign group.
    Uses trace(SS_between) / trace(SS_total).
    """
    grand_mean = X.mean(axis=0)
    ss_total = np.sum((X - grand_mean) ** 2)
    ss_between = 0.0
    for sign_idx in np.unique(y):
        mask = y == sign_idx
        n_k = mask.sum()
        group_mean = X[mask].mean(axis=0)
        ss_between += n_k * np.sum((group_mean - grand_mean) ** 2)
    return ss_between / ss_total


def permutation_test_accuracy(
    clf_pipeline,
    X: np.ndarray,
    y: np.ndarray,
    observed_acc: float,
    n_permutations: int = 1000,
    cv: int = 5,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """
    Permutation test: shuffle labels and refit CV, build null distribution of accuracy.
    Returns (p_value, null_distribution).
    """
    rng = np.random.default_rng(seed)
    null_accs = np.empty(n_permutations)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        fold_accs = []
        for train_idx, test_idx in skf.split(X, y_perm):
            clf_pipeline.fit(X[train_idx], y_perm[train_idx])
            fold_accs.append(accuracy_score(y_perm[test_idx], clf_pipeline.predict(X[test_idx])))
        null_accs[i] = np.mean(fold_accs)

    p_value = np.mean(null_accs >= observed_acc)
    return p_value, null_accs


def run_phase1(
    df: pd.DataFrame,
    cv_folds: int = 5,
    n_permutations: int = 500,
    seed: int = 42,
    verbose: bool = True,
    fast: bool = False,
) -> dict:
    X = df[BIG5_TRAITS].values.astype(float)
    y = df["zodiac_idx"].values.astype(int)

    n = len(df)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 1: Signal Detection")
        print(f"N = {n:,}  |  Traits: {BIG5_TRAITS}  |  Classes: {N_SIGNS}")
        print(f"Chance baseline: {CHANCE_BASELINE:.4f} ({CHANCE_BASELINE*100:.2f}%)")
        print(f"{'='*60}")

    classifiers = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=seed,
                                       solver="lbfgs")),
        ]),
    }
    if not fast:
        classifiers["RandomForest"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
        ])
        classifiers["GradientBoosting"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                               max_depth=3, random_state=seed)),
        ])

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    results = {}

    for name, clf in classifiers.items():
        if verbose:
            print(f"\n→ {name}")

        y_pred = cross_val_predict(clf, X, y, cv=skf, n_jobs=-1)
        acc = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)
        lift = acc / CHANCE_BASELINE

        # Confidence interval for accuracy (Wilson)
        z = 1.96
        p_hat = acc
        ci_lo = (p_hat + z**2 / (2*n) - z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))) / (1 + z**2/n)
        ci_hi = (p_hat + z**2 / (2*n) + z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))) / (1 + z**2/n)

        # Binomial test: is accuracy > chance?
        binom_result = stats.binomtest(
            int(round(acc * n)), n, p=CHANCE_BASELINE, alternative="greater"
        )

        results[name] = {
            "accuracy": acc,
            "ci_95": (ci_lo, ci_hi),
            "lift": lift,
            "f1_macro": f1_macro,
            "p_binom": binom_result.pvalue,
            "y_pred": y_pred,
        }

        if verbose:
            print(f"  Accuracy:  {acc:.4f}  ({acc*100:.2f}%)  [95% CI: {ci_lo:.4f}–{ci_hi:.4f}]")
            print(f"  Lift:      {lift:.3f}x over chance ({CHANCE_BASELINE*100:.2f}%)")
            print(f"  F1 macro:  {f1_macro:.4f}")
            print(f"  p (binom): {binom_result.pvalue:.4e}")

    # Effect size: η² from MANOVA
    eta2 = eta_squared_manova(X, y)
    if verbose:
        print(f"\n→ Effect size (η²): {eta2:.6f}")
        print(f"   Interpretation: birth timing explains {eta2*100:.3f}% of personality variance")

    # Per-sign breakdown using best classifier
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    y_pred_best = results[best_name]["y_pred"]
    per_sign = {}
    for i, sign in enumerate(ZODIAC_NAMES):
        mask = y == i
        if mask.sum() == 0:
            continue
        sign_acc = accuracy_score(y[mask], y_pred_best[mask])
        per_sign[sign] = {
            "n": int(mask.sum()),
            "accuracy": sign_acc,
            "lift": sign_acc / CHANCE_BASELINE,
        }

    if verbose:
        print(f"\n→ Per-sign accuracy (best classifier: {best_name})")
        for sign, s in sorted(per_sign.items(), key=lambda x: -x[1]["accuracy"]):
            bar = "█" * int(s["lift"] * 5)
            print(f"  {sign:<14} {s['accuracy']:.3f} ({s['lift']:.2f}x)  {bar}")

    # Permutation test on best classifier
    if verbose:
        print(f"\n→ Permutation test ({n_permutations} iterations)...")
    best_clf = classifiers[best_name]
    p_perm, null_dist = permutation_test_accuracy(
        best_clf, X, y,
        observed_acc=results[best_name]["accuracy"],
        n_permutations=n_permutations,
        cv=cv_folds,
        seed=seed,
    )
    if verbose:
        print(f"  Null distribution: mean={null_dist.mean():.4f}, std={null_dist.std():.4f}")
        print(f"  Observed accuracy: {results[best_name]['accuracy']:.4f}")
        print(f"  p (permutation):   {p_perm:.4f}")

    summary = {
        "n": n,
        "chance_baseline": CHANCE_BASELINE,
        "classifiers": {k: {kk: vv for kk, vv in v.items() if kk != "y_pred"}
                        for k, v in results.items()},
        "best_classifier": best_name,
        "eta_squared": eta2,
        "per_sign": per_sign,
        "permutation_p": p_perm,
        "null_dist_mean": float(null_dist.mean()),
        "null_dist_std": float(null_dist.std()),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 1: classifier benchmark")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with doy, E, A, C, N, O, zodiac_idx")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    summary = run_phase1(df, cv_folds=args.cv, n_permutations=args.permutations, seed=args.seed)

    out = args.out or args.data.replace(".csv", "_phase1_results.json")
    # Convert tuples to lists for JSON serialisation
    for clf_res in summary["classifiers"].values():
        if "ci_95" in clf_res:
            clf_res["ci_95"] = list(clf_res["ci_95"])
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
