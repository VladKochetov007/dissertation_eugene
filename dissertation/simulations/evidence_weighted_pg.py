"""
Evidence-Weighted Policy Gradient: Simulation Experiments

Validates Theorem 5.1 (AM-HM variance improvement) from the dissertation.
Compares standard PG vs EWPG on 2-player matrix games with heterogeneous
evidence quality (noise levels) across agents.

Author: Eugene Shcherbinin
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

FIGURES_DIR = Path(__file__).parent / "figures" / "ewpg"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


# ============================================================
# Game Definitions
# ============================================================

class MatrixGame:
    """A 2-player matrix game with payoff matrices R1, R2."""
    def __init__(self, name, R1, R2):
        self.name = name
        self.R1 = np.array(R1, dtype=float)
        self.R2 = np.array(R2, dtype=float)
        self.n_actions = [self.R1.shape[0], self.R1.shape[1]]

    def expected_payoffs(self, p1, p2):
        """Expected payoff given mixed strategies p1, p2."""
        v1 = p1 @ self.R1 @ p2
        v2 = p1 @ self.R2 @ p2
        return v1, v2

    def policy_gradient(self, p1, p2):
        """Exact policy gradient for each player.
        For player 1: dV1/dp1 = R1 @ p2
        For player 2: dV2/dp2 = R2.T @ p1
        """
        g1 = self.R1 @ p2
        g2 = self.R2.T @ p1
        return g1, g2


# Standard games
MATCHING_PENNIES = MatrixGame(
    "Matching Pennies",
    R1=[[1, -1], [-1, 1]],
    R2=[[-1, 1], [1, -1]]
)

PRISONERS_DILEMMA = MatrixGame(
    "Prisoner's Dilemma",
    R1=[[-1, -3], [0, -2]],
    R2=[[-1, 0], [-3, -2]]
)

ROCK_PAPER_SCISSORS = MatrixGame(
    "Rock-Paper-Scissors",
    R1=[[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
    R2=[[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
)


def nash_equilibrium(game):
    """Return Nash equilibrium for known games."""
    if game.name == "Matching Pennies":
        return np.array([0.5, 0.5]), np.array([0.5, 0.5])
    elif game.name == "Rock-Paper-Scissors":
        return np.array([1/3, 1/3, 1/3]), np.array([1/3, 1/3, 1/3])
    elif game.name == "Prisoner's Dilemma":
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])  # Defect-Defect
    else:
        raise ValueError(f"Unknown game: {game.name}")


def project_simplex(x):
    """Project onto the probability simplex."""
    # Clip and renormalize
    x = np.maximum(x, 1e-8)
    return x / x.sum()


def reinforce_estimate(game, p1, p2, player, n_samples=1):
    """REINFORCE gradient estimate with n_samples episodes."""
    grads = []
    for _ in range(n_samples):
        # Sample actions
        a1 = np.random.choice(len(p1), p=p1)
        a2 = np.random.choice(len(p2), p=p2)

        if player == 0:
            reward = game.R1[a1, a2]
            log_grad = np.zeros_like(p1)
            log_grad[a1] = 1.0 / p1[a1]
            # Score function: reward * nabla log pi
            grad = reward * log_grad
        else:
            reward = game.R2[a1, a2]
            log_grad = np.zeros_like(p2)
            log_grad[a2] = 1.0 / p2[a2]
            grad = reward * log_grad
        grads.append(grad)

    return np.mean(grads, axis=0)


# ============================================================
# PG Algorithms
# ============================================================

def run_standard_pg(game, n_episodes, gamma_base, p_exp=1.0,
                    noise_scales=None, n_samples=1, init=None):
    """Standard PG with optional heterogeneous noise."""
    p1_star, p2_star = nash_equilibrium(game)
    n1, n2 = game.n_actions

    p1 = init[0].copy() if init else project_simplex(np.random.dirichlet(np.ones(n1)))
    p2 = init[1].copy() if init else project_simplex(np.random.dirichlet(np.ones(n2)))

    distances = []

    for n in range(1, n_episodes + 1):
        gamma_n = gamma_base / (n + 10) ** p_exp

        # Gradient estimates (REINFORCE)
        g1 = reinforce_estimate(game, p1, p2, player=0, n_samples=n_samples)
        g2 = reinforce_estimate(game, p1, p2, player=1, n_samples=n_samples)

        # Add extra noise to simulate heterogeneous evidence
        if noise_scales is not None:
            g1 += np.random.randn(*g1.shape) * noise_scales[0]
            g2 += np.random.randn(*g2.shape) * noise_scales[1]

        # Update
        p1 = project_simplex(p1 + gamma_n * g1)
        p2 = project_simplex(p2 + gamma_n * g2)

        # Track distance to Nash
        dist = np.linalg.norm(p1 - p1_star)**2 + np.linalg.norm(p2 - p2_star)**2
        distances.append(dist)

    return np.array(distances)


def run_evidence_weighted_pg(game, n_episodes, gamma_base, p_exp=1.0,
                              evidence_weights=None, noise_scales=None,
                              n_samples=1, init=None):
    """Evidence-weighted PG: each agent scales gradient by w_i = V_i / max(V)."""
    p1_star, p2_star = nash_equilibrium(game)
    n1, n2 = game.n_actions

    p1 = init[0].copy() if init else project_simplex(np.random.dirichlet(np.ones(n1)))
    p2 = init[1].copy() if init else project_simplex(np.random.dirichlet(np.ones(n2)))

    V = evidence_weights if evidence_weights is not None else np.array([1.0, 1.0])
    w = V / V.max()  # Normalize to [0, 1]

    distances = []

    for n in range(1, n_episodes + 1):
        gamma_n = gamma_base / (n + 10) ** p_exp

        # Gradient estimates
        g1 = reinforce_estimate(game, p1, p2, player=0, n_samples=n_samples)
        g2 = reinforce_estimate(game, p1, p2, player=1, n_samples=n_samples)

        # Add heterogeneous noise (inversely proportional to evidence)
        if noise_scales is not None:
            g1 += np.random.randn(*g1.shape) * noise_scales[0]
            g2 += np.random.randn(*g2.shape) * noise_scales[1]

        # Evidence-weighted update
        p1 = project_simplex(p1 + gamma_n * w[0] * g1)
        p2 = project_simplex(p2 + gamma_n * w[1] * g2)

        dist = np.linalg.norm(p1 - p1_star)**2 + np.linalg.norm(p2 - p2_star)**2
        distances.append(dist)

    return np.array(distances)


# ============================================================
# Experiment 1: Convergence comparison
# ============================================================

def experiment_convergence(game, n_episodes=5000, n_runs=100):
    """Compare convergence of Standard PG vs EWPG."""
    print(f"\n{'='*60}")
    print(f"Experiment 1: Convergence on {game.name}")
    print(f"{'='*60}")

    # Evidence weights: agent 1 has high evidence, agent 2 has low
    V = np.array([10.0, 1.0])
    # Noise inversely proportional to evidence: sigma_i = C / sqrt(V_i)
    C_noise = 2.0
    noise_scales = np.array([C_noise / np.sqrt(V[0]), C_noise / np.sqrt(V[1])])

    print(f"Evidence weights: V = {V}")
    print(f"Noise scales: sigma = {noise_scales}")
    print(f"HM/AM ratio: {2*V[0]*V[1]/(V[0]+V[1]) / ((V[0]+V[1])/2):.4f}")

    # Fixed initial condition
    n1, n2 = game.n_actions
    init = [np.ones(n1)/n1 + 0.1*np.random.randn(n1),
            np.ones(n2)/n2 + 0.1*np.random.randn(n2)]
    init = [project_simplex(p) for p in init]

    std_distances = np.zeros((n_runs, n_episodes))
    ewpg_distances = np.zeros((n_runs, n_episodes))

    for run in range(n_runs):
        np.random.seed(run)
        std_distances[run] = run_standard_pg(
            game, n_episodes, gamma_base=0.5, p_exp=0.75,
            noise_scales=noise_scales, n_samples=3, init=init
        )
        np.random.seed(run)  # Same seed for fair comparison
        ewpg_distances[run] = run_evidence_weighted_pg(
            game, n_episodes, gamma_base=0.5, p_exp=0.75,
            evidence_weights=V, noise_scales=noise_scales,
            n_samples=3, init=init
        )

    # Average over runs
    std_mean = std_distances.mean(axis=0)
    ewpg_mean = ewpg_distances.mean(axis=0)
    std_std = std_distances.std(axis=0) / np.sqrt(n_runs)
    ewpg_std = ewpg_distances.std(axis=0) / np.sqrt(n_runs)

    # Smoothing for plotting
    window = 50
    std_smooth = np.convolve(std_mean, np.ones(window)/window, mode='valid')
    ewpg_smooth = np.convolve(ewpg_mean, np.ones(window)/window, mode='valid')

    # Empirical variance ratio (last 1000 episodes)
    empirical_ratio = ewpg_distances[:, -1000:].var() / std_distances[:, -1000:].var()
    theoretical_ratio = 2*V[0]*V[1]/(V[0]+V[1]) / ((V[0]+V[1])/2)
    print(f"\nEmpirical variance ratio (last 1000 eps): {empirical_ratio:.4f}")
    print(f"Theoretical HM/AM ratio: {theoretical_ratio:.4f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.arange(window-1, n_episodes)
    ax.semilogy(x, std_smooth, label='Standard PG', color='#e74c3c', alpha=0.9)
    ax.semilogy(x, ewpg_smooth, label='Evidence-Weighted PG', color='#2ecc71', alpha=0.9)
    ax.fill_between(range(n_episodes), std_mean - 2*std_std, std_mean + 2*std_std,
                     alpha=0.1, color='#e74c3c')
    ax.fill_between(range(n_episodes), ewpg_mean - 2*ewpg_std, ewpg_mean + 2*ewpg_std,
                     alpha=0.1, color='#2ecc71')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(r'$\|\pi_n - \pi^*\|^2$', fontsize=12)
    ax.set_title(f'Convergence: Standard PG vs EWPG ({game.name})\n'
                 f'V = {V}, HM/AM = {theoretical_ratio:.3f}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f'convergence_{game.name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / f'convergence_{game.name.lower().replace(chr(32), chr(95))}.png'}")

    return std_mean, ewpg_mean


# ============================================================
# Experiment 2: Variance improvement vs evidence ratio
# ============================================================

def experiment_variance_ratio(game, n_episodes=3000, n_runs=80):
    """Measure empirical variance ratio vs theoretical HM/AM for varying V1/V2."""
    print(f"\n{'='*60}")
    print(f"Experiment 2: Variance ratio vs evidence heterogeneity ({game.name})")
    print(f"{'='*60}")

    ratios = [1, 2, 3, 5, 8, 10, 15, 20]
    empirical_ratios = []
    theoretical_ratios = []

    n1, n2 = game.n_actions
    init = [np.ones(n1)/n1, np.ones(n2)/n2]

    for r in ratios:
        V = np.array([float(r), 1.0])
        C_noise = 2.0
        noise_scales = np.array([C_noise / np.sqrt(V[0]), C_noise / np.sqrt(V[1])])

        std_final = []
        ewpg_final = []

        for run in range(n_runs):
            np.random.seed(run * 1000 + r)
            d_std = run_standard_pg(game, n_episodes, gamma_base=0.5, p_exp=0.75,
                                     noise_scales=noise_scales, n_samples=3, init=init)
            np.random.seed(run * 1000 + r)
            d_ewpg = run_evidence_weighted_pg(game, n_episodes, gamma_base=0.5, p_exp=0.75,
                                               evidence_weights=V, noise_scales=noise_scales,
                                               n_samples=3, init=init)
            std_final.append(d_std[-500:].mean())
            ewpg_final.append(d_ewpg[-500:].mean())

        emp_ratio = np.mean(ewpg_final) / np.mean(std_final)
        theo_ratio = 2*r / (1+r)**2 * (1+r)/1  # HM/AM for [r, 1]
        # HM = 2*r*1/(r+1), AM = (r+1)/2, ratio = 4r/(r+1)^2
        theo_ratio = 4*r / (1+r)**2

        empirical_ratios.append(emp_ratio)
        theoretical_ratios.append(theo_ratio)
        print(f"  r = {r:3d}: empirical = {emp_ratio:.4f}, theoretical = {theo_ratio:.4f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(ratios, theoretical_ratios, 'k-o', label='Theoretical HM/AM', linewidth=2, markersize=6)
    ax.plot(ratios, empirical_ratios, 's--', color='#3498db', label='Empirical ratio',
            linewidth=1.5, markersize=6)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'Evidence ratio $r = V_1/V_2$', fontsize=12)
    ax.set_ylabel(r'Variance ratio $\sigma_w^2 / \sigma_{\mathrm{std}}^2$', fontsize=12)
    ax.set_title(f'AM-HM Variance Improvement ({game.name})', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f'variance_ratio_{game.name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: variance_ratio plot")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Evidence-Weighted Policy Gradient Experiments")
    print("=" * 60)

    for game in [MATCHING_PENNIES, ROCK_PAPER_SCISSORS]:
        experiment_convergence(game, n_episodes=3000, n_runs=50)
        experiment_variance_ratio(game, n_episodes=2000, n_runs=40)

    print("\nAll experiments complete. Figures saved to:", FIGURES_DIR)
