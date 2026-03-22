"""
Extended Experiments for Dissertation
Adds: convergence rate verification, combined EW-LOLA-PG, spectral analysis,
stochastic game (multi-state), full REINFORCE variance, ablations.

Author: Eugene Shcherbinin
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from scipy.stats import linregress

FIGURES_DIR = Path(__file__).parent / "figures" / "extended"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

np.random.seed(42)


# ============================================================
# Game Definitions
# ============================================================

class MatrixGame:
    def __init__(self, name, R1, R2):
        self.name = name
        self.R1 = np.array(R1, dtype=float)
        self.R2 = np.array(R2, dtype=float)
        self.n_actions = [self.R1.shape[0], self.R1.shape[1]]

    def policy_gradient(self, p1, p2):
        g1 = self.R1 @ p2
        g2 = self.R2.T @ p1
        return g1, g2

    def opponent_shaping_2x2(self, p1_scalar, p2_scalar, eta=0.1):
        """OS term for 2x2 game with scalar policies."""
        R1, R2 = self.R1, self.R2
        dV1_dp2 = p1_scalar*(R1[0,0]-R1[0,1]) + (1-p1_scalar)*(R1[1,0]-R1[1,1])
        d2V2_dp2dp1 = (R2[0,0]-R2[0,1]) - (R2[1,0]-R2[1,1])
        os1 = dV1_dp2 * eta * d2V2_dp2dp1

        dV2_dp1 = p2_scalar*(R2[0,0]-R2[1,0]) + (1-p2_scalar)*(R2[0,1]-R2[1,1])
        d2V1_dp1dp2 = (R1[0,0]-R1[1,0]) - (R1[0,1]-R1[1,1])
        os2 = dV2_dp1 * eta * d2V1_dp1dp2
        return os1, os2

    def opponent_shaping_hessian_2x2(self, p1_star, p2_star, eta=0.1):
        """Compute H = d(OS)/d(pi) at Nash for 2x2 game. Returns 2x2 matrix."""
        R1, R2 = self.R1, self.R2
        # Finite difference approximation
        eps = 1e-5
        H = np.zeros((2, 2))
        base = np.array(self.opponent_shaping_2x2(p1_star, p2_star, eta))
        for j in range(2):
            if j == 0:
                perturbed = np.array(self.opponent_shaping_2x2(p1_star + eps, p2_star, eta))
            else:
                perturbed = np.array(self.opponent_shaping_2x2(p1_star, p2_star + eps, eta))
            H[:, j] = (perturbed - base) / eps
        return H


# Standard games
MATCHING_PENNIES = MatrixGame("Matching Pennies", [[1,-1],[-1,1]], [[-1,1],[1,-1]])
PRISONERS_DILEMMA = MatrixGame("Prisoner's Dilemma", [[-1,-3],[0,-2]], [[-1,0],[-3,-2]])
ROCK_PAPER_SCISSORS = MatrixGame("Rock-Paper-Scissors",
    [[0,-1,1],[1,0,-1],[-1,1,0]], [[0,1,-1],[-1,0,1],[1,-1,0]])
BATTLE_OF_SEXES = MatrixGame("Battle of the Sexes", [[3,0],[0,2]], [[2,0],[0,3]])
STAG_HUNT = MatrixGame("Stag Hunt", [[4,0],[3,2]], [[4,3],[0,2]])

NASH_EQ = {
    "Matching Pennies": (np.array([0.5, 0.5]), np.array([0.5, 0.5])),
    "Rock-Paper-Scissors": (np.array([1/3,1/3,1/3]), np.array([1/3,1/3,1/3])),
    "Prisoner's Dilemma": (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
    "Battle of the Sexes": (np.array([0.6, 0.4]), np.array([0.4, 0.6])),
    "Stag Hunt": (np.array([0.5, 0.5]), np.array([0.5, 0.5])),
}


def project_simplex(x):
    x = np.maximum(x, 1e-8)
    return x / x.sum()


def reinforce_estimate(game, p1, p2, player, n_samples=1):
    grads = []
    for _ in range(n_samples):
        a1 = np.random.choice(len(p1), p=p1)
        a2 = np.random.choice(len(p2), p=p2)
        if player == 0:
            reward = game.R1[a1, a2]
            log_grad = np.zeros_like(p1)
            log_grad[a1] = 1.0 / p1[a1]
            grads.append(reward * log_grad)
        else:
            reward = game.R2[a1, a2]
            log_grad = np.zeros_like(p2)
            log_grad[a2] = 1.0 / p2[a2]
            grads.append(reward * log_grad)
    return np.mean(grads, axis=0)


# ============================================================
# Algorithms
# ============================================================

def run_pg(game, n_episodes, gamma_base=0.5, p_exp=0.75, m=10,
           noise_scales=None, evidence_weights=None, lola_lambda=0.0,
           lola_eta=0.1, n_samples=3, init=None):
    """Unified PG runner: standard, EWPG, LOLA, or EW-LOLA."""
    p1_star, p2_star = NASH_EQ[game.name]
    n1, n2 = game.n_actions

    p1 = init[0].copy() if init else project_simplex(np.random.dirichlet(np.ones(n1)))
    p2 = init[1].copy() if init else project_simplex(np.random.dirichlet(np.ones(n2)))

    V = evidence_weights if evidence_weights is not None else np.array([1.0, 1.0])
    w = V / V.max()

    distances = []
    gradnorms = []

    for n in range(1, n_episodes + 1):
        gamma_n = gamma_base / (n + m) ** p_exp

        # REINFORCE gradient estimates
        g1 = reinforce_estimate(game, p1, p2, player=0, n_samples=n_samples)
        g2 = reinforce_estimate(game, p1, p2, player=1, n_samples=n_samples)

        # Add heterogeneous noise
        if noise_scales is not None:
            g1 += np.random.randn(*g1.shape) * noise_scales[0]
            g2 += np.random.randn(*g2.shape) * noise_scales[1]

        # LOLA opponent-shaping (for 2x2 games only)
        if lola_lambda > 0 and n1 == 2 and n2 == 2:
            os1, os2 = game.opponent_shaping_2x2(p1[0], p2[0], eta=lola_eta)
            g1[0] += lola_lambda * os1
            g1[1] -= lola_lambda * os1  # simplex constraint
            g2[0] += lola_lambda * os2
            g2[1] -= lola_lambda * os2

        # Evidence weighting
        w1 = w[0] if evidence_weights is not None else 1.0
        w2 = w[1] if evidence_weights is not None else 1.0

        p1 = project_simplex(p1 + gamma_n * w1 * g1)
        p2 = project_simplex(p2 + gamma_n * w2 * g2)

        dist = np.linalg.norm(p1 - p1_star)**2 + np.linalg.norm(p2 - p2_star)**2
        distances.append(dist)
        gradnorms.append(np.linalg.norm(g1)**2 + np.linalg.norm(g2)**2)

    return np.array(distances), np.array(gradnorms)


# ============================================================
# Experiment 1: Convergence rate verification (log-log)
# ============================================================

def experiment_convergence_rate():
    """Verify O(1/sqrt(n)) rate by measuring slope on log-log plot."""
    print("\n" + "="*60)
    print("Experiment: Convergence rate verification")
    print("="*60)

    game = MATCHING_PENNIES
    n_episodes = 10000
    n_runs = 100

    configs = {
        'Standard PG': dict(evidence_weights=None, lola_lambda=0.0),
        'EWPG (r=10)': dict(evidence_weights=np.array([10.0, 1.0]), lola_lambda=0.0),
        'EWPG (r=100)': dict(evidence_weights=np.array([100.0, 1.0]), lola_lambda=0.0),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {'Standard PG': '#e74c3c', 'EWPG (r=10)': '#2ecc71', 'EWPG (r=100)': '#3498db'}

    for label, kwargs in configs.items():
        V = kwargs.get('evidence_weights', np.array([1.0, 1.0]))
        if V is None:
            V = np.array([1.0, 1.0])
        C_noise = 2.0
        noise_scales = np.array([C_noise / np.sqrt(V[0]), C_noise / np.sqrt(V[1])])

        init = [np.array([0.55, 0.45]), np.array([0.45, 0.55])]
        all_dist = np.zeros((n_runs, n_episodes))
        for run in range(n_runs):
            np.random.seed(run)
            d, _ = run_pg(game, n_episodes, noise_scales=noise_scales,
                         n_samples=3, init=init, **kwargs)
            all_dist[run] = d

        mean_dist = all_dist.mean(axis=0)

        # Moving average for clean log-log
        window = 200
        smooth = np.convolve(mean_dist, np.ones(window)/window, mode='valid')
        x = np.arange(window, n_episodes + 1)

        # Log-log plot
        axes[0].loglog(x, smooth, label=label, color=colors[label], linewidth=1.5)

        # Fit slope on last 60% of data
        log_x = np.log(x[len(x)//3:])
        log_y = np.log(smooth[len(smooth)//3:])
        slope, intercept, r, _, _ = linregress(log_x, log_y)
        print(f"  {label}: slope = {slope:.3f} (expect -0.5), R² = {r**2:.4f}")

        # Plot reference line
        axes[0].loglog(x[len(x)//3:], np.exp(intercept) * x[len(x)//3:]**slope,
                       '--', color=colors[label], alpha=0.5, linewidth=1)

    axes[0].loglog(x, 5.0/np.sqrt(x), 'k:', alpha=0.3, label=r'$O(1/\sqrt{n})$ reference')
    axes[0].set_xlabel('Episode $n$')
    axes[0].set_ylabel(r'$\|\pi_n - \pi^*\|^2$')
    axes[0].set_title('Convergence Rate (log-log)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # Right panel: constant comparison
    ratios = [1, 2, 3, 5, 8, 10, 20, 50, 100]
    theoretical = [4*r/(1+r)**2 for r in ratios]

    # Measure empirical constants
    empirical_constants = []
    for r in ratios:
        V = np.array([float(r), 1.0])
        noise_scales = np.array([2.0/np.sqrt(V[0]), 2.0/np.sqrt(V[1])])
        init = [np.array([0.55, 0.45]), np.array([0.45, 0.55])]
        dists = []
        for run in range(40):
            np.random.seed(run)
            d, _ = run_pg(game, 5000, noise_scales=noise_scales,
                         evidence_weights=V, n_samples=3, init=init)
            dists.append(d[-1000:].mean())
        C_ewpg = np.mean(dists)

        # Baseline (r=1)
        if r == 1:
            C_base = C_ewpg
        empirical_constants.append(C_ewpg / C_base)

    axes[1].plot(ratios, theoretical, 'k-o', label='Theoretical HM/AM', linewidth=2, markersize=5)
    axes[1].plot(ratios, empirical_constants, 's--', color='#3498db',
                label='Empirical $C_w/C_{std}$', linewidth=1.5, markersize=5)
    axes[1].set_xlabel(r'Evidence ratio $r = V_1/V_2$')
    axes[1].set_ylabel('Constant ratio')
    axes[1].set_title('Convergence Constant Improvement')
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)
    axes[1].axhline(1, color='gray', linestyle=':', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'convergence_rate.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: convergence_rate.png")


# ============================================================
# Experiment 2: Combined EW-LOLA-PG
# ============================================================

def experiment_combined():
    """Test all four combinations: PG, EWPG, LOLA, EW-LOLA."""
    print("\n" + "="*60)
    print("Experiment: Combined EW-LOLA-PG")
    print("="*60)

    game = MATCHING_PENNIES
    n_episodes = 5000
    n_runs = 80
    V = np.array([10.0, 1.0])
    noise_scales = np.array([2.0/np.sqrt(V[0]), 2.0/np.sqrt(V[1])])
    init = [np.array([0.55, 0.45]), np.array([0.45, 0.55])]

    configs = {
        'Standard PG': dict(evidence_weights=None, lola_lambda=0.0),
        'EWPG only': dict(evidence_weights=V, lola_lambda=0.0),
        'LOLA only': dict(evidence_weights=None, lola_lambda=0.5),
        'EW-LOLA-PG': dict(evidence_weights=V, lola_lambda=0.5),
    }
    colors = {'Standard PG': '#e74c3c', 'EWPG only': '#2ecc71',
              'LOLA only': '#9b59b6', 'EW-LOLA-PG': '#f39c12'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    results = {}
    for label, kwargs in configs.items():
        all_dist = np.zeros((n_runs, n_episodes))
        for run in range(n_runs):
            np.random.seed(run)
            d, _ = run_pg(game, n_episodes, noise_scales=noise_scales,
                         n_samples=3, init=init, **kwargs)
            all_dist[run] = d

        mean = all_dist.mean(axis=0)
        se = all_dist.std(axis=0) / np.sqrt(n_runs)
        results[label] = (mean, se, all_dist)

        window = 50
        smooth = np.convolve(mean, np.ones(window)/window, mode='valid')

        axes[0].semilogy(range(window-1, n_episodes), smooth,
                        label=label, color=colors[label], linewidth=1.5)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel(r'$\|\pi_n - \pi^*\|^2$')
    axes[0].set_title(f'Matching Pennies: All Methods (V={V})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # Bar chart: final distance (last 500 episodes)
    labels_bar = list(configs.keys())
    final_means = [results[l][0][-500:].mean() for l in labels_bar]
    final_ses = [results[l][1][-500:].mean() for l in labels_bar]
    bars = axes[1].bar(range(4), final_means, yerr=final_ses,
                       color=[colors[l] for l in labels_bar], alpha=0.8, capsize=3)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(['PG', 'EWPG', 'LOLA', 'EW-LOLA'], fontsize=10)
    axes[1].set_ylabel(r'Mean $\|\pi_n - \pi^*\|^2$ (last 500 eps)')
    axes[1].set_title('Final Distance Comparison')
    axes[1].grid(True, alpha=0.2, axis='y')

    # Add improvement ratios
    base = final_means[0]
    for i, (label, fm) in enumerate(zip(labels_bar, final_means)):
        if i > 0:
            axes[1].text(i, fm + final_ses[i] + 0.001,
                        f'{fm/base:.2f}x', ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'combined_ewlola.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: combined_ewlola.png")


# ============================================================
# Experiment 3: Spectral reinforcement analysis
# ============================================================

def experiment_spectral():
    """Compute and visualise the opponent-shaping Hessian eigenvalues."""
    print("\n" + "="*60)
    print("Experiment: Spectral reinforcement analysis")
    print("="*60)

    games = [MATCHING_PENNIES, BATTLE_OF_SEXES, STAG_HUNT]
    nash_scalars = {
        "Matching Pennies": (0.5, 0.5),
        "Battle of the Sexes": (0.6, 0.4),
        "Stag Hunt": (0.5, 0.5),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, game in zip(axes, games):
        p1s, p2s = nash_scalars[game.name]
        H = game.opponent_shaping_hessian_2x2(p1s, p2s, eta=0.1)
        S_H = (H + H.T) / 2
        eigvals = np.linalg.eigvalsh(S_H)
        mu_H = -max(eigvals)

        # Also compute Jac_v
        eps = 1e-5
        Jac = np.zeros((2, 2))
        g_base = np.array(game.opponent_shaping_2x2(p1s, p2s, 0.0))  # just for shape
        # Compute gradient field Jacobian
        def grad_field(p1, p2):
            g1 = p2*(game.R1[0,0]-game.R1[1,0]) + (1-p2)*(game.R1[0,1]-game.R1[1,1])
            g2 = p1*(game.R2[0,0]-game.R2[0,1]) + (1-p1)*(game.R2[1,0]-game.R2[1,1])
            return np.array([g1, g2])

        v_base = grad_field(p1s, p2s)
        for j in range(2):
            if j == 0:
                v_pert = grad_field(p1s + eps, p2s)
            else:
                v_pert = grad_field(p1s, p2s + eps)
            Jac[:, j] = (v_pert - v_base) / eps

        S_v = (Jac + Jac.T) / 2
        A_v = (Jac - Jac.T) / 2
        mu = -max(np.linalg.eigvalsh(S_v))

        # Sweep lambda and plot effective mu
        lambdas = np.linspace(0, 2, 50)
        mu_effs = [mu + lam * mu_H for lam in lambdas]

        ax.plot(lambdas, mu_effs, 'b-', linewidth=2, label=r'$\mu + \lambda\mu_H$')
        ax.axhline(mu, color='red', linestyle='--', alpha=0.7, label=f'$\\mu = {mu:.3f}$')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.fill_between(lambdas, 0, mu_effs, where=np.array(mu_effs) > 0,
                        alpha=0.1, color='green')
        ax.set_xlabel(r'$\lambda$ (LOLA strength)')
        ax.set_ylabel(r'$\mu_{\mathrm{LOLA}}$')
        ax.set_title(f'{game.name}\n$\\mu_H={mu_H:.3f}$, spectral reinf.={"Yes" if mu_H > 0 else "No"}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        print(f"  {game.name}: mu={mu:.4f}, mu_H={mu_H:.4f}, "
              f"spectral reinforcement={'Yes' if mu_H > 0 else 'No'}")
        print(f"    S_H eigenvalues: {eigvals}")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'spectral_reinforcement.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: spectral_reinforcement.png")


# ============================================================
# Experiment 4: Multi-game comparison
# ============================================================

def experiment_multigame():
    """Compare EWPG vs PG across multiple game types."""
    print("\n" + "="*60)
    print("Experiment: Multi-game comparison")
    print("="*60)

    games = [MATCHING_PENNIES, ROCK_PAPER_SCISSORS, STAG_HUNT]
    n_episodes = 5000
    n_runs = 60
    V = np.array([10.0, 1.0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, game in zip(axes, games):
        C_noise = 2.0
        if game.n_actions[0] == 3:
            noise_scales_game = np.array([C_noise/np.sqrt(V[0])]*3 + [C_noise/np.sqrt(V[1])]*3)
            noise_scales = np.array([C_noise/np.sqrt(V[0]), C_noise/np.sqrt(V[1])])
        else:
            noise_scales = np.array([C_noise/np.sqrt(V[0]), C_noise/np.sqrt(V[1])])

        init_p1 = np.ones(game.n_actions[0]) / game.n_actions[0]
        init_p2 = np.ones(game.n_actions[1]) / game.n_actions[1]
        init_p1 = project_simplex(init_p1 + 0.05*np.random.randn(len(init_p1)))
        init_p2 = project_simplex(init_p2 + 0.05*np.random.randn(len(init_p2)))
        init = [init_p1, init_p2]

        for label, ew, color in [('Standard PG', None, '#e74c3c'), ('EWPG', V, '#2ecc71')]:
            all_dist = np.zeros((n_runs, n_episodes))
            for run in range(n_runs):
                np.random.seed(run)
                d, _ = run_pg(game, n_episodes, noise_scales=noise_scales,
                             evidence_weights=ew, n_samples=3, init=init)
                all_dist[run] = d

            mean = all_dist.mean(axis=0)
            window = 100
            smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
            ax.semilogy(range(window-1, n_episodes), smooth, label=label, color=color, linewidth=1.5)

        ax.set_xlabel('Episode')
        if ax == axes[0]:
            ax.set_ylabel(r'$\|\pi_n - \pi^*\|^2$')
        ax.set_title(game.name)
        ax.legend()
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'EWPG vs Standard PG across games (V = {V})', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'multigame_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: multigame_comparison.png")


# ============================================================
# Experiment 5: LOLA annealing schedules
# ============================================================

def experiment_annealing():
    """Compare different LOLA annealing schedules."""
    print("\n" + "="*60)
    print("Experiment: LOLA annealing schedules")
    print("="*60)

    game = MATCHING_PENNIES
    n_episodes = 5000
    n_runs = 60
    init = [np.array([0.7, 0.3]), np.array([0.3, 0.7])]
    noise_std = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    schedules = {
        'Standard PG': (0.0, 0.0),
        r'LOLA $\lambda=0.5$ (const)': (0.5, 0.0),
        r'LOLA $\lambda_n \propto 1/n^{0.5}$': (0.5, 0.5),
        r'LOLA $\lambda_n \propto 1/n^{0.75}$': (0.5, 0.75),
        r'LOLA $\lambda_n \propto 1/n^{1.0}$': (0.5, 1.0),
    }
    colors_list = ['#e74c3c', '#9b59b6', '#3498db', '#2ecc71', '#f39c12']

    for (label, (lam_base, r_anneal)), color in zip(schedules.items(), colors_list):
        all_dist = np.zeros((n_runs, n_episodes))
        for run in range(n_runs):
            np.random.seed(run)
            p1, p2 = init[0].copy(), init[1].copy()
            distances = []
            for n in range(1, n_episodes + 1):
                gamma_n = 0.5 / (n + 10)**0.75
                g1, g2 = game.policy_gradient(p1, p2)
                g1 = g1 + np.random.randn(2) * noise_std
                g2 = g2 + np.random.randn(2) * noise_std

                # Annealed LOLA
                if lam_base > 0:
                    if r_anneal > 0:
                        lam_n = lam_base / (n + 10)**r_anneal
                    else:
                        lam_n = lam_base
                    os1, os2 = game.opponent_shaping_2x2(p1[0], p2[0], eta=0.1)
                    g1[0] += lam_n * os1; g1[1] -= lam_n * os1
                    g2[0] += lam_n * os2; g2[1] -= lam_n * os2

                p1 = project_simplex(p1 + gamma_n * g1)
                p2 = project_simplex(p2 + gamma_n * g2)
                distances.append(np.linalg.norm(p1 - 0.5)**2 + np.linalg.norm(p2 - 0.5)**2)

            all_dist[run] = distances

        mean = all_dist.mean(axis=0)
        window = 50
        smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
        axes[0].semilogy(range(window-1, n_episodes), smooth, label=label, color=color, linewidth=1.5)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel(r'$\|\pi_n - \pi^*\|^2$')
    axes[0].set_title('LOLA Annealing Schedules')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # Right: Basin size vs lambda (constant)
    lambdas_test = [0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    basin_sizes = []
    n_grid = 20
    p1_grid = np.linspace(0.1, 0.9, n_grid)
    p2_grid = np.linspace(0.1, 0.9, n_grid)

    for lam in lambdas_test:
        converged_count = 0
        for p1i in p1_grid:
            for p2i in p2_grid:
                np.random.seed(int(p1i*1000 + p2i*100))
                p1, p2 = np.array([p1i, 1-p1i]), np.array([p2i, 1-p2i])
                for n in range(1, 1001):
                    gamma_n = 0.3 / (n+10)**0.7
                    g1, g2 = game.policy_gradient(p1, p2)
                    g1 += np.random.randn(2)*0.3; g2 += np.random.randn(2)*0.3
                    if lam > 0:
                        os1, os2 = game.opponent_shaping_2x2(p1[0], p2[0], eta=0.1)
                        g1[0] += lam*os1; g1[1] -= lam*os1
                        g2[0] += lam*os2; g2[1] -= lam*os2
                    p1 = project_simplex(p1 + gamma_n*g1)
                    p2 = project_simplex(p2 + gamma_n*g2)
                dist = (p1[0]-0.5)**2 + (p2[0]-0.5)**2
                if dist < 0.08**2:
                    converged_count += 1
        basin_sizes.append(converged_count / (n_grid**2))
        print(f"  lambda={lam:.1f}: basin={converged_count/(n_grid**2):.1%}")

    axes[1].plot(lambdas_test, basin_sizes, 'bo-', linewidth=2, markersize=6)
    axes[1].set_xlabel(r'$\lambda$ (constant LOLA strength)')
    axes[1].set_ylabel('Basin area (fraction of state space)')
    axes[1].set_title(r'Basin Size vs $\lambda$')
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'annealing_schedules.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: annealing_schedules.png")


# ============================================================
# Experiment 6: Gradient variance measurement
# ============================================================

def experiment_gradient_variance():
    """Directly measure gradient variance under EWPG vs standard PG."""
    print("\n" + "="*60)
    print("Experiment: Gradient variance measurement")
    print("="*60)

    game = MATCHING_PENNIES
    p1_star, p2_star = np.array([0.5, 0.5]), np.array([0.5, 0.5])

    # Fix policy near Nash and measure gradient variance
    p1 = np.array([0.52, 0.48])
    p2 = np.array([0.48, 0.52])

    ratios = [1, 2, 3, 5, 8, 10, 15, 20, 50]
    n_samples_per = 2000

    std_vars = []
    ewpg_vars = []
    theoretical = []

    for r in ratios:
        V = np.array([float(r), 1.0])
        w = V / V.max()
        noise_scales = np.array([2.0/np.sqrt(V[0]), 2.0/np.sqrt(V[1])])

        grads_std = []
        grads_ew = []
        for _ in range(n_samples_per):
            g1 = reinforce_estimate(game, p1, p2, 0) + np.random.randn(2)*noise_scales[0]
            g2 = reinforce_estimate(game, p1, p2, 1) + np.random.randn(2)*noise_scales[1]
            grads_std.append(np.concatenate([g1, g2]))
            grads_ew.append(np.concatenate([w[0]*g1, w[1]*g2]))

        grads_std = np.array(grads_std)
        grads_ew = np.array(grads_ew)

        var_std = np.trace(np.cov(grads_std.T))
        var_ew = np.trace(np.cov(grads_ew.T))

        std_vars.append(var_std)
        ewpg_vars.append(var_ew)
        theoretical.append(4*r/(1+r)**2)

    empirical_ratios = [ew/std for ew, std in zip(ewpg_vars, std_vars)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ratios, std_vars, 'ro-', label='Standard PG', linewidth=1.5)
    axes[0].plot(ratios, ewpg_vars, 'go-', label='EWPG', linewidth=1.5)
    axes[0].set_xlabel(r'Evidence ratio $r$')
    axes[0].set_ylabel('Gradient variance (trace of covariance)')
    axes[0].set_title('Absolute Gradient Variance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)
    axes[0].set_xscale('log')

    axes[1].plot(ratios, theoretical, 'k-o', label='Theoretical HM/AM', linewidth=2, markersize=5)
    axes[1].plot(ratios, empirical_ratios, 's--', color='#3498db',
                label='Empirical ratio', linewidth=1.5, markersize=5)
    axes[1].set_xlabel(r'Evidence ratio $r = V_1/V_2$')
    axes[1].set_ylabel('Variance ratio')
    axes[1].set_title('Variance Ratio: Empirical vs Theoretical')
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)
    axes[1].set_xscale('log')
    axes[1].axhline(1, color='gray', linestyle=':', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'gradient_variance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: gradient_variance.png")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Extended Dissertation Experiments v2")
    print("=" * 60)

    experiment_convergence_rate()
    experiment_combined()
    experiment_spectral()
    experiment_multigame()
    experiment_annealing()
    experiment_gradient_variance()

    print(f"\nAll experiments complete. Figures saved to: {FIGURES_DIR}")
