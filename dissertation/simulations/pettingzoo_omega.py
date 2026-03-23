"""
PettingZoo × Ω-Framework Integration

Wraps PettingZoo environments for the Ω-gradient system:
  - Evidence-weighted updates (EW-PG)
  - Opponent shaping (LOLA-PG)
  - Cooperative communication (Coop-PG)
  - Fixed-point NE search (FP-NE) as meta-algorithm

The key addition: FP-NE search as a SIXTH contribution to the Ω-framework.
Not a modification of the gradient rule itself, but a meta-algorithm that
uses Ω-PG as the inner loop while explicitly searching for fixed points
of the best-response map with Bayesian stopping.

Games from PettingZoo "classic" module:
  - Rock-Paper-Scissors (unique mixed NE, tests cycling)
  - Connect Four (perfect information, deep tree)
  - Tic-Tac-Toe (small, solvable)

Plus custom matrix games wrapped in PettingZoo AEC API:
  - Stag Hunt, BoS, Chicken (multiple NE, tests selection)
  - N-player Public Goods (N > 2, tests scaling)

Author: Eugene Shcherbinin
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable
import warnings
warnings.filterwarnings('ignore')

# PettingZoo
from pettingzoo.classic import rps_v2
from pettingzoo.utils import wrappers

# Our modules
from fixed_point_ne import (
    Game, project_simplex, bayesian_fp_search,
    BayesianNECounter, find_fixed_point, are_same_ne
)

FIGURES_DIR = Path(__file__).parent / "figures" / "pettingzoo"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


# ============================================================
# PettingZoo ↔ Matrix Game Bridge
# ============================================================

def pettingzoo_to_payoff_matrix(env_fn, n_episodes: int = 10000) -> Game:
    """
    Estimate payoff matrices from a PettingZoo environment via Monte Carlo.

    For environments with discrete action spaces, we can estimate
    R[a1, a2] = E[reward | actions = (a1, a2)] by sampling.
    """
    env = env_fn()
    env.reset()

    agents = env.possible_agents
    assert len(agents) == 2, "Only 2-player games supported"

    n1 = env.action_space(agents[0]).n
    n2 = env.action_space(agents[1]).n

    R1_sum = np.zeros((n1, n2))
    R2_sum = np.zeros((n1, n2))
    counts = np.zeros((n1, n2))

    for _ in range(n_episodes):
        env.reset()
        a1 = np.random.randint(n1)
        a2 = np.random.randint(n2)

        actions = {}
        rewards = {}

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                env.step(None)
            else:
                if agent == agents[0]:
                    env.step(a1)
                else:
                    env.step(a2)

        # Collect rewards from the environment
        # PettingZoo accumulates rewards during agent_iter
        # We need to re-run and track rewards
        env.reset()
        reward_dict = {agents[0]: 0, agents[1]: 0}
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            reward_dict[agent] += reward
            if term or trunc:
                env.step(None)
            else:
                if agent == agents[0]:
                    env.step(a1)
                else:
                    env.step(a2)

        R1_sum[a1, a2] += reward_dict[agents[0]]
        R2_sum[a1, a2] += reward_dict[agents[1]]
        counts[a1, a2] += 1

    counts = np.maximum(counts, 1)
    R1 = R1_sum / counts
    R2 = R2_sum / counts

    name = env.metadata.get('name', 'PettingZoo Game')
    env.close()

    return Game(name, R1, R2)


# ============================================================
# Ω-PG: Full Omega Gradient for PettingZoo envs
# ============================================================

@dataclass
class OmegaConfig:
    """Configuration for the Ω-gradient."""
    lr: float = 0.1              # base learning rate
    evidence_weights: Optional[np.ndarray] = None  # V_i for EW-PG
    lola_lambda: float = 0.1     # opponent shaping strength
    lola_anneal: bool = True     # anneal LOLA over time
    coop_beta: float = 0.0       # cooperation term strength
    use_fp_search: bool = True   # use FP-NE meta-search
    fp_search_budget: int = 50   # max searches in FP phase
    fp_confidence: float = 0.05  # Bayesian stopping threshold


class OmegaAgent:
    """
    A single agent in the Ω-framework.

    Maintains:
      - policy (softmax over logits)
      - evidence weight V_i
      - opponent model (for LOLA)
      - self-knowledge estimate (for Coop)
    """

    def __init__(self, n_actions: int, agent_id: int, evidence: float = 1.0):
        self.n_actions = n_actions
        self.agent_id = agent_id
        self.logits = np.random.randn(n_actions) * 0.1
        self.evidence = evidence
        self.opponent_model = None  # estimated opponent policy
        self.history = []
        self._update_policy()

    def _update_policy(self):
        logits = self.logits - self.logits.max()
        self.policy = np.exp(logits) / np.exp(logits).sum()

    def act(self, explore_eps: float = 0.0) -> int:
        if np.random.rand() < explore_eps:
            return np.random.randint(self.n_actions)
        return np.random.choice(self.n_actions, p=self.policy)

    def reinforce_gradient(self, action: int, reward: float) -> np.ndarray:
        """REINFORCE score function gradient."""
        grad = -self.policy.copy()
        grad[action] += 1
        return reward * grad

    def update(self, grad: np.ndarray, lr: float, evidence_weight: float = 1.0):
        """Ω-update: evidence-weighted gradient step."""
        self.logits += lr * evidence_weight * grad
        self._update_policy()
        self.history.append(self.policy.copy())


def run_omega_pg(game: Game, config: OmegaConfig, n_episodes: int = 3000,
                 n_runs: int = 30) -> dict:
    """
    Run the full Ω-PG on a matrix game.

    Components:
      1. EW: scale gradient by w_i = V_min / V_i
      2. LOLA: add opponent-shaping correction
      3. Coop: (simplified) share policy info weighted by self-knowledge
      4. FP-NE: optionally pre-search for good NE to initialize from
    """
    V = config.evidence_weights if config.evidence_weights is not None else np.array([1.0, 1.0])
    w = np.min(V) / V  # EW weights

    all_welfares = []
    all_distances_to_ne = []

    # Phase 0: FP-NE pre-search (if enabled)
    init_strategy = None
    if config.use_fp_search:
        search = bayesian_fp_search(game, max_searches=config.fp_search_budget,
                                     confidence_threshold=config.fp_confidence,
                                     verbose=False)
        if search['counter'].n_discovered > 0:
            best = max(search['counter'].discovered_ne, key=lambda x: x[2] + x[3])
            init_strategy = (best[0], best[1])

    # Get true NE for distance tracking
    true_ne = game.compute_all_ne()

    for run in range(n_runs):
        np.random.seed(run * 41)

        agent1 = OmegaAgent(game.n1, 0, evidence=V[0])
        agent2 = OmegaAgent(game.n2, 1, evidence=V[1])

        # Initialize from FP-NE search result
        if init_strategy is not None:
            # Convert policy to logits (inverse softmax)
            p1, p2 = init_strategy
            agent1.logits = np.log(np.maximum(p1, 1e-8))
            agent2.logits = np.log(np.maximum(p2, 1e-8))
            agent1._update_policy()
            agent2._update_policy()

        welfare_traj = []

        for ep in range(n_episodes):
            lr = config.lr / (1 + ep / 500)
            lola_strength = config.lola_lambda / (1 + ep / 200) if config.lola_anneal else config.lola_lambda

            p1 = agent1.policy
            p2 = agent2.policy

            # Sample actions
            a1 = agent1.act()
            a2 = agent2.act()

            r1 = game.R1[a1, a2]
            r2 = game.R2[a1, a2]

            # REINFORCE gradients
            g1 = agent1.reinforce_gradient(a1, r1)
            g2 = agent2.reinforce_gradient(a2, r2)

            # LOLA correction: anticipate opponent's gradient step
            if config.lola_lambda > 0:
                # Approximate: how does opponent's update affect my payoff?
                # d(V1)/d(phi2) via finite difference on expected payoffs
                eps = 0.01
                p2_plus = project_simplex(p2 + eps * g2)
                p2_minus = project_simplex(p2 - eps * g2)
                v1_plus = p1 @ game.R1 @ p2_plus
                v1_minus = p1 @ game.R1 @ p2_minus
                lola_1 = (v1_plus - v1_minus) / (2 * eps) * g2[:game.n1] if game.n1 == game.n2 else np.zeros(game.n1)

                p1_plus = project_simplex(p1 + eps * g1)
                p1_minus = project_simplex(p1 - eps * g1)
                v2_plus = p1_plus @ game.R2 @ p2
                v2_minus = p1_minus @ game.R2 @ p2
                lola_2 = (v2_plus - v2_minus) / (2 * eps) * g1[:game.n2] if game.n1 == game.n2 else np.zeros(game.n2)

                g1 = g1 + lola_strength * lola_1
                g2 = g2 + lola_strength * lola_2

            # Cooperative term (simplified): share policy info
            if config.coop_beta > 0:
                # Each agent nudges toward revealed policy of other
                coop_1 = config.coop_beta * (p2[:game.n1] - p1) if game.n1 == game.n2 else np.zeros(game.n1)
                coop_2 = config.coop_beta * (p1[:game.n2] - p2) if game.n1 == game.n2 else np.zeros(game.n2)
                g1 = g1 + coop_1 * min(V[1] / V.max(), 1.0)  # weighted by other's evidence
                g2 = g2 + coop_2 * min(V[0] / V.max(), 1.0)

            # Ω-update: evidence-weighted
            agent1.update(g1, lr, w[0])
            agent2.update(g2, lr, w[1])

            # Track welfare
            v1, v2 = game.payoffs(agent1.policy, agent2.policy)
            welfare_traj.append(v1 + v2)

        all_welfares.append(welfare_traj)

        # Distance to nearest NE
        if true_ne:
            min_dist = min(
                np.linalg.norm(agent1.policy - ne[0]) + np.linalg.norm(agent2.policy - ne[1])
                for ne in true_ne
            )
            all_distances_to_ne.append(min_dist)

    return {
        'welfares': np.array(all_welfares),
        'final_dist_to_ne': np.array(all_distances_to_ne) if all_distances_to_ne else None,
        'init_strategy': init_strategy,
        'game': game,
    }


# ============================================================
# PettingZoo Experiments
# ============================================================

def experiment_rps_omega():
    """
    Rock-Paper-Scissors via PettingZoo.
    Unique NE = (1/3, 1/3, 1/3). Tests whether Ω-PG + FP-NE finds it.
    """
    print("\n" + "="*70)
    print("PettingZoo Experiment 1: Rock-Paper-Scissors")
    print("="*70)

    # Extract payoff matrix from PettingZoo env
    game = pettingzoo_to_payoff_matrix(rps_v2.env, n_episodes=5000)
    print(f"  Estimated R1:\n{game.R1.round(2)}")
    print(f"  Estimated R2:\n{game.R2.round(2)}")

    configs = {
        'Standard PG': OmegaConfig(lr=0.1, lola_lambda=0, coop_beta=0, use_fp_search=False),
        'EW-PG': OmegaConfig(lr=0.1, evidence_weights=np.array([5.0, 1.0]),
                              lola_lambda=0, coop_beta=0, use_fp_search=False),
        'LOLA-PG': OmegaConfig(lr=0.1, lola_lambda=0.3, coop_beta=0, use_fp_search=False),
        'Ω-PG': OmegaConfig(lr=0.1, evidence_weights=np.array([5.0, 1.0]),
                              lola_lambda=0.1, coop_beta=0.05, use_fp_search=False),
        'Ω-PG + FP-NE': OmegaConfig(lr=0.1, evidence_weights=np.array([5.0, 1.0]),
                                      lola_lambda=0.1, coop_beta=0.05, use_fp_search=True),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']

    results = {}
    for i, (name, config) in enumerate(configs.items()):
        result = run_omega_pg(game, config, n_episodes=2000, n_runs=30)
        results[name] = result

        welfare = result['welfares'].mean(axis=0)
        # Smooth
        w = 50
        smooth = np.convolve(welfare, np.ones(w)/w, mode='valid')

        axes[0].plot(range(w-1, len(welfare)), smooth, label=name,
                     color=colors[i], linewidth=1.5)

        if result['final_dist_to_ne'] is not None:
            print(f"  {name:20s}: final dist to NE = "
                  f"{result['final_dist_to_ne'].mean():.4f} ± "
                  f"{result['final_dist_to_ne'].std():.4f}")

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Social Welfare')
    axes[0].set_title('RPS: Convergence to Nash')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Bar chart of final distance to NE
    names = list(results.keys())
    dists = [results[n]['final_dist_to_ne'].mean() if results[n]['final_dist_to_ne'] is not None else np.nan
             for n in names]
    errs = [results[n]['final_dist_to_ne'].std() if results[n]['final_dist_to_ne'] is not None else 0
            for n in names]

    axes[1].bar(range(len(names)), dists, yerr=errs, capsize=5,
                color=colors[:len(names)], alpha=0.8)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('Distance to NE')
    axes[1].set_title('RPS: Final Distance to Nash Equilibrium')
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle('PettingZoo RPS × Ω-Framework', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'rps_omega.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'rps_omega.png'}")


def experiment_multi_ne_selection():
    """
    Games with multiple NE: Stag Hunt, BoS, Chicken.
    Tests whether FP-NE pre-search + Ω-PG finds Pareto-better equilibria
    than Ω-PG alone.
    """
    print("\n" + "="*70)
    print("PettingZoo Experiment 2: Multi-NE Selection")
    print("="*70)

    from fixed_point_ne import stag_hunt, battle_of_sexes, chicken

    games = [stag_hunt(), battle_of_sexes(), chicken()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, game in enumerate(games):
        ax = axes[idx]
        true_ne = game.compute_all_ne()

        configs = {
            'Ω-PG (random init)': OmegaConfig(
                lr=0.1, evidence_weights=np.array([3.0, 1.0]),
                lola_lambda=0.1, coop_beta=0.05, use_fp_search=False),
            'Ω-PG + FP-NE': OmegaConfig(
                lr=0.1, evidence_weights=np.array([3.0, 1.0]),
                lola_lambda=0.1, coop_beta=0.05, use_fp_search=True,
                fp_search_budget=80),
        }

        for i, (name, config) in enumerate(configs.items()):
            result = run_omega_pg(game, config, n_episodes=1500, n_runs=40)
            welfare = result['welfares'].mean(axis=0)
            w = 30
            smooth = np.convolve(welfare, np.ones(w)/w, mode='valid')
            color = '#e74c3c' if 'random' in name else '#2ecc71'
            ax.plot(range(w-1, len(welfare)), smooth, label=name,
                    color=color, linewidth=1.5)

        # Mark NE welfare levels
        if true_ne:
            for j, ne in enumerate(true_ne):
                v1, v2 = game.payoffs(ne[0], ne[1])
                ax.axhline(y=v1+v2, color='gray', linestyle=':', alpha=0.4)
                ax.text(10, v1+v2+0.05, f'NE{j+1}: {v1+v2:.1f}', fontsize=8, color='gray')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Social Welfare')
        ax.set_title(f'{game.name}\n({len(true_ne)} NE)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('FP-NE Search Selects Better Equilibria', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'multi_ne_selection.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'multi_ne_selection.png'}")


def experiment_omega_ablation():
    """
    Ablation study: contribution of each Ω-component.
    On Stag Hunt (multiple NE, cooperation matters).
    """
    print("\n" + "="*70)
    print("PettingZoo Experiment 3: Ω-Component Ablation")
    print("="*70)

    from fixed_point_ne import stag_hunt
    game = stag_hunt()

    V = np.array([5.0, 1.0])

    configs = {
        'Vanilla PG':        OmegaConfig(lr=0.1, lola_lambda=0, coop_beta=0, use_fp_search=False),
        '+ EW':              OmegaConfig(lr=0.1, evidence_weights=V, lola_lambda=0, coop_beta=0, use_fp_search=False),
        '+ EW + LOLA':       OmegaConfig(lr=0.1, evidence_weights=V, lola_lambda=0.15, coop_beta=0, use_fp_search=False),
        '+ EW + LOLA + Coop': OmegaConfig(lr=0.1, evidence_weights=V, lola_lambda=0.15, coop_beta=0.05, use_fp_search=False),
        'Full Ω + FP-NE':   OmegaConfig(lr=0.1, evidence_weights=V, lola_lambda=0.15, coop_beta=0.05, use_fp_search=True, fp_search_budget=60),
    }

    colors = ['#bdc3c7', '#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    final_welfares = {}
    for i, (name, config) in enumerate(configs.items()):
        result = run_omega_pg(game, config, n_episodes=2000, n_runs=40)
        welfare = result['welfares'].mean(axis=0)
        w = 40
        smooth = np.convolve(welfare, np.ones(w)/w, mode='valid')

        axes[0].plot(range(w-1, len(welfare)), smooth, label=name,
                     color=colors[i], linewidth=1.5 if 'Full' not in name else 2.5)

        final_welfares[name] = result['welfares'][:, -200:].mean()

        print(f"  {name:25s}: final welfare = {final_welfares[name]:.3f}")

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Social Welfare')
    axes[0].set_title('Stag Hunt: Ω-Component Ablation')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Best NE welfare
    true_ne = game.compute_all_ne()
    if true_ne:
        best = max(game.payoffs(ne[0], ne[1])[0] + game.payoffs(ne[0], ne[1])[1] for ne in true_ne)
        axes[0].axhline(y=best, color='green', linestyle=':', alpha=0.5, label=f'Best NE = {best:.1f}')

    # Waterfall chart showing incremental improvement
    names = list(final_welfares.keys())
    vals = list(final_welfares.values())
    increments = [vals[0]] + [vals[i] - vals[i-1] for i in range(1, len(vals))]

    bars = axes[1].bar(range(len(names)), increments, color=colors, alpha=0.8)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    axes[1].set_ylabel('Incremental Welfare Improvement')
    axes[1].set_title('Contribution of Each Ω-Component')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=0, color='black', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'omega_ablation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'omega_ablation.png'}")


def experiment_scaling():
    """
    Experiment 4: How does FP-NE search scale with game size?
    Test on 3×3, 5×5, 10×10 random games.
    """
    print("\n" + "="*70)
    print("PettingZoo Experiment 4: Scaling with Game Size")
    print("="*70)

    sizes = [2, 3, 5, 7, 10]
    n_random_games = 10

    fp_times = []
    fp_ne_found = []
    pg_welfares = []
    fp_welfares = []

    import time

    for d in sizes:
        times = []
        ne_counts = []
        pg_w = []
        fp_w = []

        for trial in range(n_random_games):
            np.random.seed(trial * 100 + d)
            R1 = np.random.randn(d, d)
            R2 = np.random.randn(d, d)
            game = Game(f"Random {d}×{d}", R1, R2)

            # FP-NE search
            t0 = time.time()
            search = bayesian_fp_search(game, max_searches=100,
                                         confidence_threshold=0.05, verbose=False)
            t1 = time.time()
            times.append(t1 - t0)
            ne_counts.append(search['counter'].n_discovered)

            if search['counter'].n_discovered > 0:
                best = max(search['counter'].discovered_ne, key=lambda x: x[2] + x[3])
                fp_w.append(best[2] + best[3])
            else:
                fp_w.append(np.nan)

            # PG baseline (just run vanilla PG)
            config = OmegaConfig(lr=0.1, lola_lambda=0, coop_beta=0, use_fp_search=False)
            result = run_omega_pg(game, config, n_episodes=1000, n_runs=5)
            pg_w.append(result['welfares'][:, -100:].mean())

        fp_times.append(np.mean(times))
        fp_ne_found.append(np.mean(ne_counts))
        pg_welfares.append(np.nanmean(pg_w))
        fp_welfares.append(np.nanmean(fp_w))

        print(f"  d={d:2d}: FP search {np.mean(times):.2f}s, "
              f"found {np.mean(ne_counts):.1f} NE, "
              f"FP welfare={np.nanmean(fp_w):.3f}, "
              f"PG welfare={np.nanmean(pg_w):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(sizes, fp_times, 'ko-', linewidth=2)
    axes[0].set_xlabel('Game size d')
    axes[0].set_ylabel('Search time (seconds)')
    axes[0].set_title('FP-NE Search Time')
    axes[0].grid(alpha=0.3)

    axes[1].plot(sizes, fp_ne_found, 's-', color='#2ecc71', linewidth=2)
    axes[1].set_xlabel('Game size d')
    axes[1].set_ylabel('NE found (avg)')
    axes[1].set_title('Number of NE Discovered')
    axes[1].grid(alpha=0.3)

    axes[2].plot(sizes, fp_welfares, 'o-', color='#2ecc71', linewidth=2, label='FP-NE + Ω')
    axes[2].plot(sizes, pg_welfares, 's--', color='#e74c3c', linewidth=2, label='Vanilla PG')
    axes[2].set_xlabel('Game size d')
    axes[2].set_ylabel('Social Welfare')
    axes[2].set_title('Welfare: FP-NE vs PG')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle('Scaling: FP-NE Search on Random d×d Games', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'scaling.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'scaling.png'}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("PettingZoo × Ω-Framework Experiments")
    print("=" * 70)
    print("Testing FP-NE search as meta-algorithm for the Ω-gradient")
    print("=" * 70)

    experiment_rps_omega()
    experiment_multi_ne_selection()
    experiment_omega_ablation()
    experiment_scaling()

    print("\n" + "="*70)
    print("All PettingZoo experiments complete.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*70)
