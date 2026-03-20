"""
Run all dissertation experiments and generate figures.

Experiments:
1. Matching Pennies (zero-sum): Nash convergence comparison
2. Prisoner's Dilemma: Cooperation emergence
3. Parameter trajectory phase plots

Usage:
    python run_experiments.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from games import matching_pennies, prisoners_dilemma, coordination_game, sigmoid
from meta_mapg import run_independent_pg, run_lola, run_meta_mapg, run_meta_pg

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Plotting style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'lines.linewidth': 1.5,
})

COLORS = {
    'Independent PG': '#d62728',
    'LOLA': '#2ca02c',
    'Meta-PG': '#ff7f0e',
    'Meta-MAPG': '#1f77b4',
}


def experiment_matching_pennies():
    """
    Matching Pennies: Zero-sum game.
    Nash equilibrium: p1 = p2 = 0.5 (φ = 0).
    
    Key result: Meta-MAPG converges to Nash; Meta-PG diverges; 
    Independent PG oscillates; LOLA converges but may overshoot.
    """
    print("Running Experiment 1: Matching Pennies...")
    game = matching_pennies()
    steps = 300
    phi_init = 0.5  # Start away from Nash

    results = {
        'Independent PG': run_independent_pg(game, phi_init, -phi_init, lr=0.3, steps=steps),
        'LOLA': run_lola(game, phi_init, -phi_init, lr=0.1, lr_opponent=0.1, steps=steps),
        'Meta-PG': run_meta_pg(game, phi_init, -phi_init, lr_inner=0.1, lr_outer=0.05, lookahead=3, steps=steps),
        'Meta-MAPG': run_meta_mapg(game, phi_init, -phi_init, lr_inner=0.1, lr_outer=0.05, lookahead=3, steps=steps),
    }

    # Plot 1: Policy probabilities over time
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, hist in results.items():
        axes[0].plot(hist['p1'], label=f'{name} (Agent 1)', color=COLORS[name])
        axes[1].plot(hist['p2'], label=f'{name} (Agent 2)', color=COLORS[name])

    for ax in axes:
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Nash')
        ax.set_xlabel('Step')
        ax.set_ylabel('P(action=0)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best', fontsize=8)

    axes[0].set_title('Matching Pennies — Agent 1')
    axes[1].set_title('Matching Pennies — Agent 2')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'matching_pennies_convergence.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'matching_pennies_convergence.png'))
    plt.close()

    # Plot 2: Phase portrait (p1 vs p2)
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, hist in results.items():
        ax.plot(hist['p1'], hist['p2'], label=name, color=COLORS[name], alpha=0.8)
        ax.scatter(hist['p1'][0], hist['p2'][0], color=COLORS[name], marker='o', s=60, zorder=5)
        ax.scatter(hist['p1'][-1], hist['p2'][-1], color=COLORS[name], marker='*', s=100, zorder=5)

    ax.plot(0.5, 0.5, 'kx', markersize=15, markeredgewidth=3, label='Nash Equilibrium')
    ax.set_xlabel('Agent 1: P(action=0)')
    ax.set_ylabel('Agent 2: P(action=0)')
    ax.set_title('Matching Pennies — Phase Portrait')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'matching_pennies_phase.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'matching_pennies_phase.png'))
    plt.close()

    print("  → Saved matching_pennies_convergence.{pdf,png}")
    print("  → Saved matching_pennies_phase.{pdf,png}")


def experiment_prisoners_dilemma():
    """
    Prisoner's Dilemma.
    Nash: mutual defection (p1=0, p2=0, i.e. φ→-∞).
    Pareto: mutual cooperation (p1=1, p2=1).
    
    Key result: LOLA and Meta-MAPG learn to cooperate;
    Independent PG and Meta-PG converge to defection.
    """
    print("Running Experiment 2: Prisoner's Dilemma...")
    game = prisoners_dilemma()
    steps = 500

    # Start with slight cooperation bias; LOLA needs higher opponent lr
    # to capture the opponent-shaping effect that drives cooperation
    results = {
        'Independent PG': run_independent_pg(game, 0.5, 0.5, lr=0.5, steps=steps),
        'LOLA': run_lola(game, 0.5, 0.5, lr=0.05, lr_opponent=1.0, steps=steps),
        'Meta-PG': run_meta_pg(game, 0.5, 0.5, lr_inner=0.5, lr_outer=0.1, lookahead=5, steps=steps),
        'Meta-MAPG': run_meta_mapg(game, 0.5, 0.5, lr_inner=0.5, lr_outer=0.1, lookahead=5, steps=steps),
    }

    # Plot: Cooperation probability over time
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, hist in results.items():
        axes[0].plot(hist['p1'], label=name, color=COLORS[name])
        axes[1].plot(hist['V1'], label=name, color=COLORS[name])

    axes[0].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Full Cooperation')
    axes[0].axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, label='Full Defection')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('P(Cooperate)')
    axes[0].set_title("Prisoner's Dilemma — Cooperation Rate (Agent 1)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc='best', fontsize=8)

    axes[1].axhline(y=3.0, color='gray', linestyle=':', alpha=0.5, label='Mutual Cooperation (3)')
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Mutual Defection (1)')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Expected Return')
    axes[1].set_title("Prisoner's Dilemma — Agent 1 Return")
    axes[1].legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'prisoners_dilemma.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'prisoners_dilemma.png'))
    plt.close()

    print("  → Saved prisoners_dilemma.{pdf,png}")


def experiment_sensitivity():
    """
    Sensitivity analysis: How does lookahead depth affect Meta-MAPG convergence?
    """
    print("Running Experiment 3: Lookahead Sensitivity...")
    game = matching_pennies()
    steps = 300
    phi_init = 0.8

    fig, ax = plt.subplots(figsize=(8, 5))

    for L in [1, 2, 3, 5, 10]:
        hist = run_meta_mapg(game, phi_init, -phi_init,
                             lr_inner=0.1, lr_outer=0.05,
                             lookahead=L, steps=steps)
        # Distance from Nash
        dist = [np.sqrt((p1 - 0.5)**2 + (p2 - 0.5)**2)
                for p1, p2 in zip(hist['p1'], hist['p2'])]
        ax.plot(dist, label=f'L={L}', alpha=0.8)

    ax.set_xlabel('Step')
    ax.set_ylabel('Distance from Nash Equilibrium')
    ax.set_title('Meta-MAPG: Effect of Lookahead Depth')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'lookahead_sensitivity.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'lookahead_sensitivity.png'))
    plt.close()

    print("  → Saved lookahead_sensitivity.{pdf,png}")


def experiment_initial_conditions():
    """
    Robustness: Run Meta-MAPG vs Independent PG from multiple initial conditions
    in Matching Pennies.
    """
    print("Running Experiment 4: Initial Condition Robustness...")
    game = matching_pennies()
    steps = 200

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    np.random.seed(42)
    inits = [(np.random.randn() * 1.5, np.random.randn() * 1.5) for _ in range(8)]

    for phi1_0, phi2_0 in inits:
        # Independent PG
        h_ind = run_independent_pg(game, phi1_0, phi2_0, lr=0.3, steps=steps)
        axes[0].plot(h_ind['p1'], h_ind['p2'], color=COLORS['Independent PG'], alpha=0.4)
        axes[0].scatter(h_ind['p1'][0], h_ind['p2'][0], color=COLORS['Independent PG'],
                        marker='o', s=30, zorder=5, alpha=0.6)

        # Meta-MAPG
        h_meta = run_meta_mapg(game, phi1_0, phi2_0,
                               lr_inner=0.1, lr_outer=0.05, lookahead=3, steps=steps)
        axes[1].plot(h_meta['p1'], h_meta['p2'], color=COLORS['Meta-MAPG'], alpha=0.4)
        axes[1].scatter(h_meta['p1'][0], h_meta['p2'][0], color=COLORS['Meta-MAPG'],
                        marker='o', s=30, zorder=5, alpha=0.6)

    for ax, title in zip(axes, ['Independent PG', 'Meta-MAPG']):
        ax.plot(0.5, 0.5, 'kx', markersize=15, markeredgewidth=3)
        ax.set_xlabel('Agent 1: P(action=0)')
        ax.set_ylabel('Agent 2: P(action=0)')
        ax.set_title(f'Matching Pennies — {title} (8 initial conditions)')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'initial_conditions.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'initial_conditions.png'))
    plt.close()

    print("  → Saved initial_conditions.{pdf,png}")


def print_summary():
    """Print numerical summary of final states."""
    print("\n" + "="*70)
    print("NUMERICAL SUMMARY")
    print("="*70)

    for game_fn, game_name in [(matching_pennies, "Matching Pennies"),
                                (prisoners_dilemma, "Prisoner's Dilemma")]:
        game = game_fn()
        steps = 500
        print(f"\n{game_name}:")
        print(f"  {'Method':<20} {'p1 final':>10} {'p2 final':>10} {'V1 final':>10} {'V2 final':>10}")
        print(f"  {'-'*60}")

        for name, runner, kwargs in [
            ('Independent PG', run_independent_pg, dict(lr=0.3, steps=steps)),
            ('LOLA', run_lola, dict(lr=0.1, lr_opponent=0.2, steps=steps)),
            ('Meta-PG', run_meta_pg, dict(lr_inner=0.2, lr_outer=0.05, lookahead=3, steps=steps)),
            ('Meta-MAPG', run_meta_mapg, dict(lr_inner=0.2, lr_outer=0.05, lookahead=3, steps=steps)),
        ]:
            h = runner(game, 0.3, -0.3, **kwargs)
            print(f"  {name:<20} {h['p1'][-1]:>10.4f} {h['p2'][-1]:>10.4f} "
                  f"{h['V1'][-1]:>10.4f} {h['V2'][-1]:>10.4f}")


if __name__ == "__main__":
    print("="*70)
    print("DISSERTATION SIMULATIONS")
    print("Meta-Learning Multi-Agent Policy Gradients")
    print("="*70 + "\n")

    experiment_matching_pennies()
    experiment_prisoners_dilemma()
    experiment_sensitivity()
    experiment_initial_conditions()
    print_summary()

    print(f"\nAll figures saved to {FIGURE_DIR}/")
