# Deliveroo MLE Take-Home

## Contents

| File | Task | Description |
|---|---|---|
| `task1_project_plan.md` | **Task 1** | ML project plan for restaurant prep-time estimation. Covers feature engineering, iterative modelling (baselines → GBMs → mixed-effects → survival analysis), human expert knowledge incorporation, explainability, and evaluation strategy. |
| `task2_evaluate_model.py` | **Task 2** | Implementation of `evaluate_model` — computes binary log-loss using only Python builtins and raises `ModelPerformanceError` if the loss exceeds `MODEL_PERF_THRESHOLD`. |
| `task2_test_evaluate_model.py` | **Task 2** | 18 unit tests covering correctness (known log-loss values), edge cases (zero/one predictions, empty inputs, length mismatches), and threshold enforcement. |
| `task3_code_review.tex` | **Task 3** | LaTeX source for the code review of `process.py`. |
| `task3_code_review.pdf` | **Task 3** | Compiled review: 5 blocking bugs, 9 issues, 5 suggestions, test-data verification, and a full clean rewrite. |
| `task3_process_clean.py` | **Task 3** | Clean rewrite of `process.py` addressing all findings. |

## Running the tests (Task 2)

```bash
cd deliveroo
python -m unittest task2_test_evaluate_model -v
```

All 18 tests pass with no external dependencies.

## Running the clean process.py (Task 3)

```bash
cd deliveroo
python task3_process_clean.py input.txt output.txt
```
