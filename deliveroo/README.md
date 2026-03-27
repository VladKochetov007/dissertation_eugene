# Deliveroo MLE Take-Home

## Contents

| File | Part | Description |
|---|---|---|
| `deliveroo-prep-time-plan.md` | **Part 1** | ML project plan for restaurant prep-time estimation. Covers feature engineering, iterative modelling (baselines → GBMs → mixed-effects → survival analysis), human expert knowledge incorporation, and evaluation strategy. |
| `evaluate_model.py` | **Part 2** | Implementation of `evaluate_model` — computes binary log-loss using only Python builtins and raises `ModelPerformanceError` if the loss exceeds `MODEL_PERF_THRESHOLD`. |
| `test_evaluate_model.py` | **Part 2** | 18 unit tests covering correctness (known log-loss values), edge cases (zero/one predictions, empty inputs, length mismatches), and threshold enforcement. |
| `process_review.tex` | **Part 3** | LaTeX source for the code review of `process.py`. |
| `process_review.pdf` | **Part 3** | Compiled review: 5 blocking bugs, 9 issues, 5 suggestions, test-data verification, and a full clean rewrite. |
| `process_clean.py` | **Part 3** | Clean rewrite of `process.py` addressing all findings. |

## Running the tests

```bash
cd deliveroo
python -m unittest test_evaluate_model -v
```

All 18 tests should pass with no external dependencies.

## Running the clean process.py

```bash
cd deliveroo
python process_clean.py input.txt output.txt
```
