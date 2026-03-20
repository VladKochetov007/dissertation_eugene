"""
Extended dissertation experiments — Chapter 10.

Experiments:
5.  Gradient term decomposition (Terms 1, 2, 3) across all games
6.  Extended game suite (Stag Hunt, Chicken, Battle of Sexes)
7.  N-agent public goods game (3, 5, 7 agents)
8.  N-agent Stag Hunt with varying thresholds
9.  Stochastic vs exact gradient comparison
10. Learning rate landscape (heatmaps)
11. Convergence rate quantification (formal metrics)
12. Gradient term dynamics over training (time series)

Usage:
    python3 run_extended_experiments.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm

from games import matching_pennies, prisoners_dilemma, coordination_game, sigmoid
from games_extended import (
    stag_hunt, chicken, battle_of_sexes, deadlock,
    n_player_public_goods, n_player_stag_hunt,
    MatrixGame,
)
from meta_mapg import run_independent_pg, run_lola, run_meta_mapg, run_meta_pg
from meta_mapg_extended import (
    run_meta_mapg_decomposed,
    run_n_agent_independent_pg, run_n_agent_lola, run_n_agent_meta_mapg,
    run_stochastic_independent_pg, run_stochastic_meta_mapg,
)

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'lines.linewidth': 1.5,
})

COLORS = {
    'Independent PG': '#d62728',
    'LOLA': '#2ca02c',
    'Meta-PG': '#ff7f0e',
    'Meta-MAPG': '#1f77b4',
}

def _nash_mixed(game):
    """Compute mixed-strategy Nash for any 2x2 MatrixGame."""
    a = game.R1[0, 0] - game.R1[1, 0] - game.R1[0, 1] + game.R1[1, 1]
    if abs(a) < 1e-10:
        return None
    p2_star = (game.R1[1, 1] - game.R1[0, 1]) / a
    b = game.R2[0, 0] - game.R2[0, 1] - game.R2[1, 0] + game.R2[1, 1]
    if abs(b) < 1e-10:
        return None
    p1_star = (game.R2[1, 1] - game.R2[1, 0]) / b
    if 0 <= p1_star <= 1 and 0 <= p2_star <= 1:
        return (p1_star, p2_star)
    return None


TERM_COLORS = {
    'Term 1 (direct)': '#1f77b4',
    'Term 2 (own-learning)': '#ff7f0e',
    'Term 3 (peer-learning)': '#2ca02c',
}


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Gradient Term Decomposition
# ═════════════════════════════════════════════════════════════════════

def experiment_gradient_decomposition():
    """
    THE key experiment. Shows relative magnitudes of Terms 1, 2, 3
    in Meta-MAPG across different games. Demonstrates:
    - In zero-sum (Matching Pennies): Term 3 is essential
    - In cooperative (Stag Hunt): Term 2 dominates
    - In mixed (Chicken): all terms contribute
    """
    print("Running Experiment 5: Gradient Term Decomposition...")

    games = [
        (matching_pennies(), 0.8, -0.5),
        (prisoners_dilemma(), 0.5, 0.5),
        (stag_hunt(), 0.5, 0.5),
        (chicken(), 0.3, -0.3),
        (battle_of_sexes(), 0.0, 0.0),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for idx, (game, phi1_0, phi2_0) in enumerate(games):
        h = run_meta_mapg_decomposed(
            game, phi1_0, phi2_0,
            lr_inner=0.1, lr_outer=0.05, lookahead=3, steps=300,
        )

        ax = axes_flat[idx]

        # Smoothed term magnitudes (rolling mean for clarity)
        window = 10
        for key, label, color in [
            ('term1_mag', 'Term 1 (direct)', TERM_COLORS['Term 1 (direct)']),
            ('term2_mag', 'Term 2 (own-learning)', TERM_COLORS['Term 2 (own-learning)']),
            ('term3_mag', 'Term 3 (peer-learning)', TERM_COLORS['Term 3 (peer-learning)']),
        ]:
            vals = np.array(h[key])
            smoothed = np.convolve(vals, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=label, color=color, alpha=0.85)

        ax.set_title(game.name, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('|Gradient Term|')
        ax.set_yscale('symlog', linthresh=1e-4)
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

    # Summary bar chart: average term proportions
    ax_summary = axes_flat[5]
    game_names = []
    term_props = {1: [], 2: [], 3: []}

    for game, phi1_0, phi2_0 in games:
        h = run_meta_mapg_decomposed(
            game, phi1_0, phi2_0,
            lr_inner=0.1, lr_outer=0.05, lookahead=3, steps=300,
        )
        t1 = np.mean(h['term1_mag'])
        t2 = np.mean(h['term2_mag'])
        t3 = np.mean(h['term3_mag'])
        total = t1 + t2 + t3 + 1e-10
        term_props[1].append(t1 / total)
        term_props[2].append(t2 / total)
        term_props[3].append(t3 / total)
        game_names.append(game.name.replace(" ", "\n"))

    x = np.arange(len(game_names))
    width = 0.25
    ax_summary.bar(x - width, term_props[1], width, label='Term 1', color=TERM_COLORS['Term 1 (direct)'])
    ax_summary.bar(x, term_props[2], width, label='Term 2', color=TERM_COLORS['Term 2 (own-learning)'])
    ax_summary.bar(x + width, term_props[3], width, label='Term 3', color=TERM_COLORS['Term 3 (peer-learning)'])
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(game_names, fontsize=8)
    ax_summary.set_ylabel('Proportion of Total Gradient')
    ax_summary.set_title('Term Proportions (Averaged)', fontweight='bold')
    ax_summary.legend(fontsize=8)
    ax_summary.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Meta-MAPG Gradient Decomposition: Terms 1, 2, 3', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURE_DIR, 'gradient_decomposition.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'gradient_decomposition.png'))
    plt.close()
    print("  → Saved gradient_decomposition.{pdf,png}")


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: Extended Game Suite
# ═════════════════════════════════════════════════════════════════════

def experiment_extended_games():
    """
    All four methods across Stag Hunt, Chicken, Battle of Sexes.
    Shows how game structure determines which method succeeds.
    """
    print("Running Experiment 6: Extended Game Suite...")

    game_configs = [
        (stag_hunt(), 0.5, 0.5, 500, "P(Hunt Stag)"),
        (chicken(), 0.3, -0.3, 500, "P(Dare)"),
        (battle_of_sexes(), 0.0, 0.0, 500, "P(action=0)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col, (game, phi1_0, phi2_0, steps, ylabel) in enumerate(game_configs):
        results = {
            'Independent PG': run_independent_pg(game, phi1_0, phi2_0, lr=0.3, steps=steps),
            'LOLA': run_lola(game, phi1_0, phi2_0, lr=0.1, lr_opponent=0.3, steps=steps),
            'Meta-PG': run_meta_pg(game, phi1_0, phi2_0, lr_inner=0.2, lr_outer=0.05, lookahead=3, steps=steps),
            'Meta-MAPG': run_meta_mapg(game, phi1_0, phi2_0, lr_inner=0.2, lr_outer=0.05, lookahead=3, steps=steps),
        }

        # Top row: policy probabilities
        ax = axes[0, col]
        for name, hist in results.items():
            ax.plot(hist['p1'], label=f'{name}', color=COLORS[name], alpha=0.85)
        nash = _nash_mixed(game)
        if nash:
            ax.axhline(y=nash[0], color='black', linestyle='--', alpha=0.4, label=f'Nash ({nash[0]:.2f})')
        ax.set_title(f'{game.name} — Agent 1', fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

        # Bottom row: expected returns
        ax = axes[1, col]
        for name, hist in results.items():
            ax.plot(hist['V1'], label=name, color=COLORS[name], alpha=0.85)
        ax.set_title(f'{game.name} — Agent 1 Return', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('E[Return]')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Extended Game Suite: Policy & Return Convergence', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURE_DIR, 'extended_games.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'extended_games.png'))
    plt.close()
    print("  → Saved extended_games.{pdf,png}")


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: N-Agent Public Goods Game
# ═════════════════════════════════════════════════════════════════════

def experiment_n_agent_public_goods():
    """
    Scale beyond 2 players. Public goods game with 3, 5, 7 agents.
    Tests whether Meta-MAPG can sustain cooperation at scale.
    Directly relevant to Agents of Chaos (6 LLM agents).
    """
    print("Running Experiment 7: N-Agent Public Goods...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col, n in enumerate([3, 5, 7]):
        game = n_player_public_goods(n, multiplier=1.8, cost=1.0)
        phi_init = np.ones(n) * 0.3
        steps = 300

        results = {}

        # Independent PG
        results['Independent PG'] = run_n_agent_independent_pg(
            game, phi_init.copy(), lr=0.3, steps=steps)

        # LOLA (slower for large N due to cross-derivatives)
        if n <= 5:
            results['LOLA'] = run_n_agent_lola(
                game, phi_init.copy(), lr=0.1, lr_opponent=0.2, steps=steps)

        # Meta-MAPG
        results['Meta-MAPG'] = run_n_agent_meta_mapg(
            game, phi_init.copy(), lr_inner=0.2, lr_outer=0.05,
            lookahead=2, steps=steps)

        # Top: mean cooperation probability
        ax = axes[0, col]
        for name, hist in results.items():
            mean_coop = [np.mean(p) for p in hist['probs']]
            color = COLORS.get(name, '#9467bd')
            ax.plot(mean_coop, label=name, color=color, alpha=0.85)
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4)
        ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.4)
        ax.set_title(f'{n}-Agent Public Goods — Mean P(Cooperate)', fontweight='bold')
        ax.set_ylabel('Mean P(Cooperate)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom: mean expected return
        ax = axes[1, col]
        for name, hist in results.items():
            mean_ret = [np.mean(r) for r in hist['returns']]
            color = COLORS.get(name, '#9467bd')
            ax.plot(mean_ret, label=name, color=color, alpha=0.85)
        ax.set_title(f'{n}-Agent Public Goods — Mean Return', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean E[Return]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('N-Agent Public Goods: Scaling Multi-Agent Learning',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURE_DIR, 'n_agent_public_goods.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'n_agent_public_goods.png'))
    plt.close()
    print("  → Saved n_agent_public_goods.{pdf,png}")


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 8: N-Agent Stag Hunt with Varying Thresholds
# ═════════════════════════════════════════════════════════════════════

def experiment_n_agent_stag_hunt():
    """
    N-player Stag Hunt where cooperation requires threshold participants.
    Tests the tipping point behaviour: Meta-MAPG should coordinate
    better when threshold < N (partial cooperation sufficient).
    """
    print("Running Experiment 8: N-Agent Stag Hunt...")

    n = 5
    thresholds = [2, 3, 4, 5]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for col, threshold in enumerate(thresholds):
        game = n_player_stag_hunt(n, threshold=threshold)
        phi_init = np.zeros(n)
        steps = 300

        h_ind = run_n_agent_independent_pg(
            game, phi_init.copy(), lr=0.3, steps=steps)
        h_meta = run_n_agent_meta_mapg(
            game, phi_init.copy(), lr_inner=0.2, lr_outer=0.05,
            lookahead=2, steps=steps)

        # Top: individual agent cooperation probs (Meta-MAPG)
        ax = axes[0, col]
        for i in range(n):
            probs_i = [p[i] for p in h_meta['probs']]
            ax.plot(probs_i, alpha=0.6, label=f'Agent {i+1}')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4)
        ax.set_title(f'Threshold={threshold}/{n}\nMeta-MAPG Agents', fontweight='bold', fontsize=10)
        ax.set_ylabel('P(Hunt Stag)')
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # Bottom: mean cooperation comparison
        ax = axes[1, col]
        mean_ind = [np.mean(p) for p in h_ind['probs']]
        mean_meta = [np.mean(p) for p in h_meta['probs']]
        ax.plot(mean_ind, label='Ind. PG', color=COLORS['Independent PG'])
        ax.plot(mean_meta, label='Meta-MAPG', color=COLORS['Meta-MAPG'])
        ax.set_title(f'Mean P(Cooperate)', fontweight='bold', fontsize=10)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean P(Hunt Stag)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{n}-Agent Stag Hunt: Effect of Cooperation Threshold',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURE_DIR, 'n_agent_stag_hunt.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'n_agent_stag_hunt.png'))
    plt.close()
    print("  → Saved n_agent_stag_hunt.{pdf,png}")


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 9: Stochastic vs Exact Gradient Comparison
# ═════════════════════════════════════════════════════════════════════

def experiment_stochastic_comparison():
    """
    Compare exact (analytic) vs stochastic (sample-based) policy gradients.
    Shows that the theoretical results hold under realistic sampling noise.
    Critical for practical relevance: real systems don't have exact gradients.
    """
    print("Running Experiment 9: Stochastic vs Exact Comparison...")

    games_list = [
        (matching_pennies(), 0.5, -0.5),
        (prisoners_dilemma(), 0.5, 0.5),
        (stag_hunt(), 0.3, 0.3),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for col, (game, phi1_0, phi2_0) in enumerate(games_list):
        # Exact methods
        h_exact_ind = run_independent_pg(game, phi1_0, phi2_0, lr=0.3, steps=500)
        h_exact_meta = run_meta_mapg(game, phi1_0, phi2_0,
                                      lr_inner=0.1, lr_outer=0.05,
                                      lookahead=3, steps=500)

        # Stochastic methods (more steps, smaller lr)
        batch_sizes = [16, 64, 256]
        stoch_results = {}
        for bs in batch_sizes:
            h = run_stochastic_independent_pg(
                game, phi1_0, phi2_0, lr=0.05, steps=2000,
                batch_size=bs, seed=42)
            stoch_results[f'Stoch. PG (B={bs})'] = h

        # Top: Agent 1 policy over steps
        ax = axes[0, col]
        ax.plot(h_exact_ind['p1'], label='Exact Ind. PG', color='#d62728', linewidth=2)
        ax.plot(h_exact_meta['p1'], label='Exact Meta-MAPG', color='#1f77b4', linewidth=2)
        for bs, color_alpha in zip(batch_sizes, [0.3, 0.5, 0.7]):
            key = f'Stoch. PG (B={bs})'
            # Subsample to match step count
            p1 = stoch_results[key]['p1']
            steps_sub = np.linspace(0, len(p1)-1, 500).astype(int)
            ax.plot(steps_sub * 500 / len(p1), [p1[s] for s in steps_sub],
                    label=key, alpha=color_alpha, color='#9467bd')

        ax.set_title(f'{game.name} — P(action=0)', fontweight='bold')
        ax.set_ylabel('P(action=0)')
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)

        # Bottom: Returns
        ax = axes[1, col]
        ax.plot(h_exact_ind['V1'], label='Exact Ind. PG', color='#d62728', linewidth=2)
        ax.plot(h_exact_meta['V1'], label='Exact Meta-MAPG', color='#1f77b4', linewidth=2)
        for bs, color_alpha in zip(batch_sizes, [0.3, 0.5, 0.7]):
            key = f'Stoch. PG (B={bs})'
            V1 = stoch_results[key]['V1']
            steps_sub = np.linspace(0, len(V1)-1, 500).astype(int)
            ax.plot(steps_sub * 500 / len(V1), [V1[s] for s in steps_sub],
                    label=key, alpha=color_alpha, color='#9467bd')

        ax.set_title(f'{game.name} — Return', fontweight='bold')
        ax.set_xlabel('Step (normalized)')
        ax.set_ylabel('E[Return]')
        if col == 0:
            ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Exact vs Stochastic Policy Gradients', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURE_DIR, 'stochastic_comparison.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'stochastic_comparison.png'))
    plt.close()
    print("  → Saved stochastic_comparison.{pdf,png}")


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 10: Learning Rate Landscape
# ═════════════════════════════════════════════════════════════════════

def experiment_lr_landscape():
    """
    Heatmap: final distance from Nash/optimal as function of (lr_inner, lr_outer)
    for Meta-MAPG. Identifies the stable region of hyperparameter space.
    """
    print("Running Experiment 10: Learning Rate Landscape...")

    game = matching_pennies()
    lr_inner_range = np.linspace(0.01, 0.5, 20)
    lr_outer_range = np.linspace(0.005, 0.2, 20)

    # Metric: final distance from Nash (0.5, 0.5)
    dist_matrix = np.zeros((len(lr_outer_range), len(lr_inner_range)))

    for i, lr_o in enumerate(lr_outer_range):
        for j, lr_i in enumerate(lr_inner_range):
            try:
                h = run_meta_mapg(game, 0.8, -0.5,
                                   lr_inner=lr_i, lr_outer=lr_o,
                                   lookahead=3, steps=300)
                final_p1, final_p2 = h['p1'][-1], h['p2'][-1]
                if np.isnan(final_p1) or np.isnan(final_p2):
                    dist_matrix[i, j] = 1.0
                else:
                    dist_matrix[i, j] = np.sqrt((final_p1 - 0.5)**2 + (final_p2 - 0.5)**2)
            except (OverflowError, FloatingPointError):
                dist_matrix[i, j] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap for Matching Pennies
    im = axes[0].imshow(dist_matrix, origin='lower', aspect='auto',
                         extent=[lr_inner_range[0], lr_inner_range[-1],
                                 lr_outer_range[0], lr_outer_range[-1]],
                         cmap='RdYlGn_r')
    plt.colorbar(im, ax=axes[0], label='Distance from Nash')
    axes[0].set_xlabel(r'$\alpha_{inner}$')
    axes[0].set_ylabel(r'$\alpha_{outer}$')
    axes[0].set_title('Matching Pennies: Meta-MAPG\nDistance from Nash at t=300', fontweight='bold')
    # Mark the stable region
    axes[0].contour(lr_inner_range, lr_outer_range, dist_matrix,
                     levels=[0.05], colors='white', linewidths=2, linestyles='--')

    # Same for Prisoner's Dilemma (metric: cooperation level)
    game2 = prisoners_dilemma()
    coop_matrix = np.zeros((len(lr_outer_range), len(lr_inner_range)))

    for i, lr_o in enumerate(lr_outer_range):
        for j, lr_i in enumerate(lr_inner_range):
            try:
                h = run_meta_mapg(game2, 0.5, 0.5,
                                   lr_inner=lr_i, lr_outer=lr_o,
                                   lookahead=5, steps=500)
                final_p1, final_p2 = h['p1'][-1], h['p2'][-1]
                if np.isnan(final_p1) or np.isnan(final_p2):
                    coop_matrix[i, j] = 0.0
                else:
                    coop_matrix[i, j] = (final_p1 + final_p2) / 2
            except (OverflowError, FloatingPointError):
                coop_matrix[i, j] = 0.0

    im2 = axes[1].imshow(coop_matrix, origin='lower', aspect='auto',
                          extent=[lr_inner_range[0], lr_inner_range[-1],
                                  lr_outer_range[0], lr_outer_range[-1]],
                          cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im2, ax=axes[1], label='Mean P(Cooperate)')
    axes[1].set_xlabel(r'$\alpha_{inner}$')
    axes[1].set_ylabel(r'$\alpha_{outer}$')
    axes[1].set_title("Prisoner's Dilemma: Meta-MAPG\nCooperation at t=500", fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'lr_landscape.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'lr_landscape.png'))
    plt.close()
    print("  → Saved lr_landscape.{pdf,png}")


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 11: Convergence Rate Quantification
# ═════════════════════════════════════════════════════════════════════

def experiment_convergence_rates():
    """
    Formal convergence metrics: time to reach ε-ball around equilibrium,
    and asymptotic convergence rate (slope on log-distance plot).

    Table output suitable for dissertation Chapter 10.
    """
    print("Running Experiment 11: Convergence Rate Quantification...")

    game = matching_pennies()
    steps = 500
    eps_thresholds = [0.1, 0.05, 0.01]
    nash = (0.5, 0.5)

    methods = {
        'Independent PG': lambda: run_independent_pg(game, 0.5, -0.5, lr=0.3, steps=steps),
        'LOLA': lambda: run_lola(game, 0.5, -0.5, lr=0.1, lr_opponent=0.2, steps=steps),
        'Meta-PG': lambda: run_meta_pg(game, 0.5, -0.5, lr_inner=0.1, lr_outer=0.05,
                                         lookahead=3, steps=steps),
        'Meta-MAPG': lambda: run_meta_mapg(game, 0.5, -0.5, lr_inner=0.1, lr_outer=0.05,
                                             lookahead=3, steps=steps),
    }

    # Compute metrics
    table_data = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, runner in methods.items():
        h = runner()
        dists = [np.sqrt((p1 - nash[0])**2 + (p2 - nash[1])**2)
                 for p1, p2 in zip(h['p1'], h['p2'])]

        ax.plot(dists, label=name, color=COLORS[name])

        # Time to reach each ε-ball
        times = {}
        for eps in eps_thresholds:
            reached = [t for t, d in enumerate(dists) if d < eps]
            times[eps] = reached[0] if reached else float('inf')

        # Convergence rate: fit log(dist) ~ -rate * t for last 50% of steps
        half = len(dists) // 2
        dists_tail = np.array(dists[half:])
        dists_tail_pos = np.maximum(dists_tail, 1e-10)
        log_dists = np.log(dists_tail_pos)
        t_vals = np.arange(half, len(dists))

        if np.std(log_dists) > 0.01:
            slope, _ = np.polyfit(t_vals, log_dists, 1)
            rate = -slope
        else:
            rate = 0.0

        final_dist = dists[-1]
        table_data.append((name, times, rate, final_dist))

    for eps in eps_thresholds:
        ax.axhline(y=eps, color='gray', linestyle=':', alpha=0.3)
        ax.text(steps - 5, eps * 1.1, f'ε={eps}', fontsize=8, ha='right', color='gray')

    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance from Nash Equilibrium')
    ax.set_title('Convergence to Nash: Matching Pennies', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'convergence_rates.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'convergence_rates.png'))
    plt.close()

    # Print table
    print("\n  ┌─────────────────────┬────────┬────────┬────────┬──────────┬───────────┐")
    print("  │ Method              │ T(0.1) │ T(0.05)│ T(0.01)│ Rate     │ Final d   │")
    print("  ├─────────────────────┼────────┼────────┼────────┼──────────┼───────────┤")
    for name, times, rate, final_d in table_data:
        t_str = {eps: (f"{times[eps]:>4d}" if times[eps] < float('inf') else " DNR")
                 for eps in eps_thresholds}
        print(f"  │ {name:<19} │ {t_str[0.1]:>6} │ {t_str[0.05]:>6} │ {t_str[0.01]:>6} │ "
              f"{rate:>8.5f} │ {final_d:>9.6f} │")
    print("  └─────────────────────┴────────┴────────┴────────┴──────────┴───────────┘")
    print("  (T(ε) = first step where distance < ε; DNR = did not reach; Rate = exp. decay)")

    print("  → Saved convergence_rates.{pdf,png}")


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 12: Gradient Term Dynamics (Stacked Area)
# ═════════════════════════════════════════════════════════════════════

def experiment_term_dynamics():
    """
    Stacked area chart showing how the relative contribution of
    Terms 1, 2, 3 evolves over training. Key insight: early training
    is dominated by different terms than late training.
    """
    print("Running Experiment 12: Gradient Term Dynamics...")

    games_list = [
        (matching_pennies(), 0.8, -0.5, "zero-sum"),
        (stag_hunt(), 0.5, 0.5, "coordination"),
        (chicken(), 0.3, -0.3, "anti-coordination"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for col, (game, phi1_0, phi2_0, game_type) in enumerate(games_list):
        h = run_meta_mapg_decomposed(
            game, phi1_0, phi2_0,
            lr_inner=0.1, lr_outer=0.05, lookahead=3, steps=300,
        )

        # Normalize to proportions at each step
        t1 = np.array(h['term1_mag'])
        t2 = np.array(h['term2_mag'])
        t3 = np.array(h['term3_mag'])
        total = t1 + t2 + t3 + 1e-10
        t1_prop = t1 / total
        t2_prop = t2 / total
        t3_prop = t3 / total

        # Smooth
        window = 15
        kernel = np.ones(window) / window
        t1_s = np.convolve(t1_prop, kernel, mode='valid')
        t2_s = np.convolve(t2_prop, kernel, mode='valid')
        t3_s = np.convolve(t3_prop, kernel, mode='valid')

        ax = axes[col]
        x = np.arange(len(t1_s))
        ax.fill_between(x, 0, t1_s, alpha=0.7,
                         color=TERM_COLORS['Term 1 (direct)'], label='Term 1 (direct)')
        ax.fill_between(x, t1_s, t1_s + t2_s, alpha=0.7,
                         color=TERM_COLORS['Term 2 (own-learning)'], label='Term 2 (own-learning)')
        ax.fill_between(x, t1_s + t2_s, t1_s + t2_s + t3_s, alpha=0.7,
                         color=TERM_COLORS['Term 3 (peer-learning)'], label='Term 3 (peer-learning)')

        ax.set_title(f'{game.name}\n({game_type})', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Proportion of Total Gradient')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Gradient Term Dynamics Over Training', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(FIGURE_DIR, 'term_dynamics.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'term_dynamics.png'))
    plt.close()
    print("  → Saved term_dynamics.{pdf,png}")


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 13: Jacobian Norm Evolution
# ═════════════════════════════════════════════════════════════════════

def experiment_jacobian_evolution():
    """
    Track how the Jacobians J11 (own-learning) and J21 (peer-learning)
    evolve over training. Shows the 'learning about learning' dynamics.
    """
    print("Running Experiment 13: Jacobian Evolution...")

    games_list = [
        (matching_pennies(), 0.8, -0.5),
        (prisoners_dilemma(), 0.5, 0.5),
        (stag_hunt(), 0.5, 0.5),
        (chicken(), 0.3, -0.3),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for col, (game, phi1_0, phi2_0) in enumerate(games_list):
        h = run_meta_mapg_decomposed(
            game, phi1_0, phi2_0,
            lr_inner=0.1, lr_outer=0.05, lookahead=3, steps=300,
        )

        # Top: Jacobian values
        ax = axes[0, col]
        ax.plot(h['J11'], label='J₁₁ (own)', color='#1f77b4', linewidth=2)
        ax.plot(h['J21'], label='J₂₁ (peer)', color='#2ca02c', linewidth=2)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
        ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.4)
        ax.set_title(f'{game.name}', fontweight='bold')
        ax.set_ylabel('Jacobian Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom: Jacobian ratio |J21/J11|
        ax = axes[1, col]
        ratio = np.array([abs(j21 / (j11 + 1e-10)) for j11, j21 in zip(h['J11'], h['J21'])])
        ax.plot(ratio, color='#9467bd', linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('|J₂₁ / J₁₁|')
        ax.set_title('Peer/Own Influence Ratio', fontweight='bold', fontsize=10)
        ax.set_yscale('symlog', linthresh=0.01)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Jacobian Evolution: How Agents Learn About Learning',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURE_DIR, 'jacobian_evolution.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'jacobian_evolution.png'))
    plt.close()
    print("  → Saved jacobian_evolution.{pdf,png}")


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 14: Ablation Study — Term Removal
# ═════════════════════════════════════════════════════════════════════

def experiment_ablation():
    """
    Ablation: compare full Meta-MAPG (T1+T2+T3), Meta-PG (T1+T2),
    LOLA (T1+T3), and Independent PG (T1 only).

    This is the definitive comparison showing each term's contribution.
    """
    print("Running Experiment 14: Ablation Study...")

    games_list = [
        (matching_pennies(), 0.8, -0.5, 300),
        (prisoners_dilemma(), 0.5, 0.5, 500),
        (stag_hunt(), 0.5, 0.5, 400),
        (chicken(), 0.3, -0.3, 400),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    method_labels = {
        'Independent PG': 'T1 only',
        'LOLA': 'T1 + T3',
        'Meta-PG': 'T1 + T2',
        'Meta-MAPG': 'T1 + T2 + T3',
    }

    for col, (game, phi1_0, phi2_0, steps) in enumerate(games_list):
        results = {
            'Independent PG': run_independent_pg(game, phi1_0, phi2_0, lr=0.3, steps=steps),
            'LOLA': run_lola(game, phi1_0, phi2_0, lr=0.1, lr_opponent=0.2, steps=steps),
            'Meta-PG': run_meta_pg(game, phi1_0, phi2_0, lr_inner=0.1, lr_outer=0.05,
                                    lookahead=3, steps=steps),
            'Meta-MAPG': run_meta_mapg(game, phi1_0, phi2_0, lr_inner=0.1, lr_outer=0.05,
                                        lookahead=3, steps=steps),
        }

        # Top: phase portrait
        ax = axes[0, col]
        for name, hist in results.items():
            ax.plot(hist['p1'], hist['p2'], label=f'{method_labels[name]}',
                    color=COLORS[name], alpha=0.8)
            ax.scatter(hist['p1'][-1], hist['p2'][-1], color=COLORS[name],
                       marker='*', s=100, zorder=5)

        nash = _nash_mixed(game)
        if nash:
            ax.plot(nash[0], nash[1], 'kx', markersize=12, markeredgewidth=3, label='Nash')
        ax.set_title(f'{game.name}', fontweight='bold')
        ax.set_xlabel('Agent 1: P(a=0)')
        ax.set_ylabel('Agent 2: P(a=0)')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        if col == 0:
            ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

        # Bottom: social welfare (V1 + V2)
        ax = axes[1, col]
        for name, hist in results.items():
            welfare = [v1 + v2 for v1, v2 in zip(hist['V1'], hist['V2'])]
            ax.plot(welfare, label=method_labels[name], color=COLORS[name], alpha=0.85)
        ax.set_title(f'Social Welfare (V₁+V₂)', fontweight='bold', fontsize=10)
        ax.set_xlabel('Step')
        ax.set_ylabel('V₁ + V₂')
        if col == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Ablation: Contribution of Each Gradient Term',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURE_DIR, 'ablation.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'ablation.png'))
    plt.close()
    print("  → Saved ablation.{pdf,png}")


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 15: N-Agent Term Decomposition
# ═════════════════════════════════════════════════════════════════════

def experiment_n_agent_term_decomposition():
    """
    Gradient term decomposition for N-agent games.
    Shows how Terms 1, 2, 3 scale with number of agents.
    Directly relevant to Agents of Chaos interpretation.
    """
    print("Running Experiment 15: N-Agent Term Decomposition...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for col, n in enumerate([3, 5, 7]):
        game = n_player_public_goods(n, multiplier=1.8, cost=1.0)
        phi_init = np.ones(n) * 0.3
        steps = 200

        h = run_n_agent_meta_mapg(
            game, phi_init.copy(), lr_inner=0.2, lr_outer=0.05,
            lookahead=2, steps=steps)

        ax = axes[col]
        window = 10
        for key, label, color in [
            ('term1_norms', 'Term 1 (direct)', TERM_COLORS['Term 1 (direct)']),
            ('term2_norms', 'Term 2 (own-learning)', TERM_COLORS['Term 2 (own-learning)']),
            ('term3_norms', 'Term 3 (peer-learning)', TERM_COLORS['Term 3 (peer-learning)']),
        ]:
            vals = np.array(h[key])
            smoothed = np.convolve(vals, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=label, color=color)

        ax.set_title(f'N={n} Public Goods', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('||Gradient Term||')
        ax.set_yscale('symlog', linthresh=1e-4)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Gradient Term Scaling with Agent Count', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(FIGURE_DIR, 'n_agent_terms.pdf'))
    plt.savefig(os.path.join(FIGURE_DIR, 'n_agent_terms.png'))
    plt.close()
    print("  → Saved n_agent_terms.{pdf,png}")


# ═════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════

def print_extended_summary():
    """Comprehensive summary table across all games and methods."""
    print("\n" + "=" * 90)
    print("EXTENDED NUMERICAL SUMMARY")
    print("=" * 90)

    all_games = [
        (matching_pennies, "Matching Pennies", 0.5, -0.5),
        (prisoners_dilemma, "Prisoner's Dilemma", 0.5, 0.5),
        (stag_hunt, "Stag Hunt", 0.5, 0.5),
        (chicken, "Chicken", 0.3, -0.3),
        (battle_of_sexes, "Battle of Sexes", 0.0, 0.0),
    ]

    for game_fn, game_name, phi1_0, phi2_0 in all_games:
        game = game_fn()
        steps = 500
        nash = _nash_mixed(game)
        nash_str = f"({nash[0]:.2f}, {nash[1]:.2f})" if nash else "pure only"

        print(f"\n{game_name} [Mixed Nash: {nash_str}]:")
        print(f"  {'Method':<20} {'p1':>8} {'p2':>8} {'V1':>8} {'V2':>8} {'V1+V2':>8} {'d(Nash)':>8}")
        print(f"  {'-' * 68}")

        for name, runner, kwargs in [
            ('Independent PG', run_independent_pg, dict(lr=0.3, steps=steps)),
            ('LOLA', run_lola, dict(lr=0.1, lr_opponent=0.2, steps=steps)),
            ('Meta-PG', run_meta_pg, dict(lr_inner=0.1, lr_outer=0.05, lookahead=3, steps=steps)),
            ('Meta-MAPG', run_meta_mapg, dict(lr_inner=0.1, lr_outer=0.05, lookahead=3, steps=steps)),
        ]:
            h = runner(game, phi1_0, phi2_0, **kwargs)
            p1_f, p2_f = h['p1'][-1], h['p2'][-1]
            V1_f, V2_f = h['V1'][-1], h['V2'][-1]
            welfare = V1_f + V2_f
            if nash:
                d_nash = np.sqrt((p1_f - nash[0])**2 + (p2_f - nash[1])**2)
            else:
                d_nash = float('nan')
            print(f"  {name:<20} {p1_f:>8.4f} {p2_f:>8.4f} {V1_f:>8.4f} {V2_f:>8.4f} "
                  f"{welfare:>8.4f} {d_nash:>8.4f}")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("EXTENDED DISSERTATION EXPERIMENTS")
    print("Meta-Learning Multi-Agent Policy Gradients")
    print("=" * 70 + "\n")

    experiment_gradient_decomposition()    # Exp 5
    experiment_extended_games()            # Exp 6
    experiment_n_agent_public_goods()      # Exp 7
    experiment_n_agent_stag_hunt()         # Exp 8
    experiment_stochastic_comparison()     # Exp 9
    experiment_lr_landscape()              # Exp 10
    experiment_convergence_rates()         # Exp 11
    experiment_term_dynamics()             # Exp 12
    experiment_jacobian_evolution()        # Exp 13
    experiment_ablation()                  # Exp 14
    experiment_n_agent_term_decomposition()  # Exp 15
    print_extended_summary()

    print(f"\n{'=' * 70}")
    print(f"All extended figures saved to {FIGURE_DIR}/")
    print(f"{'=' * 70}")
