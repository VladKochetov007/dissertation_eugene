# Dissertation Simulations

Implementations of multi-agent policy gradient methods for the dissertation
"Meta-Learning Multi-Agent Policy Gradients with Applications to LLM Steering".

## Structure

- `meta_mapg.py` — Core implementations: Independent PG, LOLA, Meta-MAPG
- `games.py` — Game environments (Matching Pennies, IPD, General-Sum)
- `run_experiments.py` — Run all experiments and generate figures
- `figures/` — Output directory for plots

## Quick Start

```bash
pip install numpy matplotlib
python run_experiments.py
```

## Experiments

1. **Matching Pennies** (zero-sum): Demonstrates Term 3 necessity — Meta-MAPG converges to Nash, Meta-PG diverges
2. **Iterated Prisoner's Dilemma**: LOLA and Meta-MAPG learn cooperation, Independent PG defects
3. **Convergence comparison**: Learning curves for all three methods across games
