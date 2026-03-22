"""
Roadmap Experiments: Addressing Reviewer Critiques

1. Real REINFORCE variance (no artificial noise injection)
2. Multi-state stochastic game (2-state Markov game)
3. Adaptive evidence weight estimation (learned V_i)

Author: Eugene Shcherbinin
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures" / "roadmap"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'figure.dpi': 150,
})

np.random.seed(42)


# ============================================================
# 1. REAL REINFORCE on Matrix Games (no artificial noise)
# ============================================================

class MatrixGame:
    def __init__(self, name, R1, R2):
        self.name = name
        self.R1 = np.array(R1, dtype=float)
        self.R2 = np.array(R2, dtype=float)

def real_reinforce_gradient(game, p1, p2, player, n_episodes_sample=1):
    """True REINFORCE estimator: sample actions, compute score-weighted return.
    No artificial noise — variance comes entirely from sampling."""
    grads = []
    for _ in range(n_episodes_sample):
        a1 = np.random.choice(len(p1), p=p1)
        a2 = np.random.choice(len(p2), p=p2)
        if player == 0:
            reward = game.R1[a1, a2]
            score = np.zeros_like(p1)
            score[a1] = 1.0 / p1[a1]
            # Subtract baseline (mean reward)
            baseline = p1 @ game.R1 @ p2
            grads.append((reward - baseline) * score)
        else:
            reward = game.R2[a1, a2]
            score = np.zeros_like(p2)
            score[a2] = 1.0 / p2[a2]
            baseline = p1 @ game.R2 @ p2
            grads.append((reward - baseline) * score)
    return np.mean(grads, axis=0)

def project_simplex(x):
    x = np.maximum(x, 1e-8)
    return x / x.sum()

def experiment_real_reinforce():
    """Compare Standard PG vs EWPG using ONLY real REINFORCE variance."""
    print("\n" + "="*60)
    print("Experiment: Real REINFORCE (no artificial noise)")
    print("="*60)

    game = MatrixGame("Matching Pennies", [[1,-1],[-1,1]], [[-1,1],[1,-1]])
    p1_star, p2_star = np.array([0.5, 0.5]), np.array([0.5, 0.5])

    n_episodes = 8000
    n_runs = 80

    # Heterogeneous sample budgets: agent 1 gets more samples per episode
    sample_configs = {
        'Equal (K=1,1)': (1, 1),
        'Equal (K=5,5)': (5, 5),
        'Heterogeneous (K=10,1)': (10, 1),
        'Heterogeneous (K=50,1)': (50, 1),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

    # LEFT: Standard PG with different sample budgets
    for (label, (K1, K2)), color in zip(sample_configs.items(), colors):
        all_dist = np.zeros((n_runs, n_episodes))
        for run in range(n_runs):
            np.random.seed(run * 100)
            p1 = np.array([0.6, 0.4])
            p2 = np.array([0.4, 0.6])
            for n in range(1, n_episodes + 1):
                gamma_n = 0.3 / (n + 10)**0.75
                g1 = real_reinforce_gradient(game, p1, p2, 0, n_episodes_sample=K1)
                g2 = real_reinforce_gradient(game, p1, p2, 1, n_episodes_sample=K2)
                p1 = project_simplex(p1 + gamma_n * g1)
                p2 = project_simplex(p2 + gamma_n * g2)
                all_dist[run, n-1] = np.linalg.norm(p1-p1_star)**2 + np.linalg.norm(p2-p2_star)**2

        mean = all_dist.mean(axis=0)
        window = 100
        smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
        axes[0].semilogy(range(window-1, n_episodes), smooth, label=f'PG {label}',
                        color=color, linewidth=1.5)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel(r'$\|\pi_n - \pi^*\|^2$')
    axes[0].set_title('Standard PG: Effect of Sample Budget')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # RIGHT: EWPG vs Standard PG with heterogeneous samples
    K1, K2 = 10, 1
    for label, use_ew, color in [('Standard PG', False, '#e74c3c'), ('EWPG', True, '#2ecc71')]:
        all_dist = np.zeros((n_runs, n_episodes))
        for run in range(n_runs):
            np.random.seed(run * 100)
            p1 = np.array([0.6, 0.4])
            p2 = np.array([0.4, 0.6])
            for n in range(1, n_episodes + 1):
                gamma_n = 0.3 / (n + 10)**0.75
                g1 = real_reinforce_gradient(game, p1, p2, 0, n_episodes_sample=K1)
                g2 = real_reinforce_gradient(game, p1, p2, 1, n_episodes_sample=K2)
                if use_ew:
                    # Evidence weight proportional to sample count
                    V = np.array([float(K1), float(K2)])
                    w = V / V.max()
                    p1 = project_simplex(p1 + gamma_n * w[0] * g1)
                    p2 = project_simplex(p2 + gamma_n * w[1] * g2)
                else:
                    p1 = project_simplex(p1 + gamma_n * g1)
                    p2 = project_simplex(p2 + gamma_n * g2)
                all_dist[run, n-1] = np.linalg.norm(p1-p1_star)**2 + np.linalg.norm(p2-p2_star)**2

        mean = all_dist.mean(axis=0)
        se = all_dist.std(axis=0) / np.sqrt(n_runs)
        window = 100
        smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
        axes[1].semilogy(range(window-1, n_episodes), smooth, label=label,
                        color=color, linewidth=1.5)
        axes[1].fill_between(range(n_episodes), mean-2*se, mean+2*se, alpha=0.08, color=color)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel(r'$\|\pi_n - \pi^*\|^2$')
    axes[1].set_title(f'EWPG vs PG with Real REINFORCE (K=({K1},{K2}))')
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'real_reinforce.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: real_reinforce.png")


# ============================================================
# 2. MULTI-STATE STOCHASTIC GAME
# ============================================================

class TwoStateGame:
    """A 2-player, 2-state, 2-action stochastic game.
    State 0: zero-sum coordination challenge
    State 1: prisoner's dilemma variant
    Transitions depend on joint actions.
    """
    def __init__(self):
        self.name = "Two-State Markov Game"
        self.n_states = 2
        self.n_actions = 2

        # Reward matrices per state
        self.R1 = [
            np.array([[2, -1], [-1, 1]]),   # State 0: modified matching pennies
            np.array([[-1, -3], [0, -2]]),  # State 1: prisoner's dilemma
        ]
        self.R2 = [
            np.array([[-2, 1], [1, -1]]),   # State 0: zero-sum
            np.array([[-1, 0], [-3, -2]]),  # State 1: PD
        ]

        # Transition probabilities P[s][a1][a2] -> distribution over next states
        self.P = [
            [  # From state 0
                [[0.7, 0.3], [0.4, 0.6]],  # a1=0: [a2=0, a2=1]
                [[0.4, 0.6], [0.2, 0.8]],  # a1=1: [a2=0, a2=1]
            ],
            [  # From state 1
                [[0.5, 0.5], [0.3, 0.7]],
                [[0.6, 0.4], [0.5, 0.5]],
            ],
        ]
        self.P = np.array(self.P)  # shape: (2, 2, 2, 2)
        self.zeta = 0.1  # termination probability

    def sample_episode(self, pi1, pi2, s0=None):
        """Sample one episode. pi1, pi2: (n_states, n_actions) policy arrays."""
        if s0 is None:
            s = np.random.choice(self.n_states)
        else:
            s = s0
        trajectory = []
        while True:
            a1 = np.random.choice(self.n_actions, p=pi1[s])
            a2 = np.random.choice(self.n_actions, p=pi2[s])
            r1 = self.R1[s][a1, a2]
            r2 = self.R2[s][a1, a2]
            trajectory.append((s, a1, a2, r1, r2))
            if np.random.rand() < self.zeta:
                break
            s_next = np.random.choice(self.n_states, p=self.P[s, a1, a2])
            s = s_next
        return trajectory

    def reinforce_gradient(self, pi1, pi2, player, n_episodes=1):
        """REINFORCE gradient for stochastic game."""
        grad = np.zeros_like(pi1 if player == 0 else pi2)
        for _ in range(n_episodes):
            traj = self.sample_episode(pi1, pi2)
            total_return = sum(t[3] if player == 0 else t[4] for t in traj)
            for s, a1, a2, r1, r2 in traj:
                if player == 0:
                    score = np.zeros(self.n_actions)
                    score[a1] = 1.0 / pi1[s, a1]
                    grad[s] += total_return * score
                else:
                    score = np.zeros(self.n_actions)
                    score[a2] = 1.0 / pi2[s, a2]
                    grad[s] += total_return * score
        return grad / max(n_episodes, 1)


def experiment_stochastic_game():
    """Run PG and EWPG on a multi-state stochastic game."""
    print("\n" + "="*60)
    print("Experiment: Multi-state stochastic game")
    print("="*60)

    game = TwoStateGame()
    n_episodes = 6000
    n_runs = 40

    # Find approximate Nash by running many episodes of standard PG
    # (we don't know the analytical Nash for this game)
    # Use a long run as reference
    np.random.seed(999)
    pi1_ref = np.ones((2, 2)) / 2
    pi2_ref = np.ones((2, 2)) / 2
    for n in range(1, 20001):
        gamma_n = 0.1 / (n + 100)**0.75
        g1 = game.reinforce_gradient(pi1_ref, pi2_ref, 0, n_episodes=5)
        g2 = game.reinforce_gradient(pi1_ref, pi2_ref, 1, n_episodes=5)
        for s in range(2):
            pi1_ref[s] = project_simplex(pi1_ref[s] + gamma_n * g1[s])
            pi2_ref[s] = project_simplex(pi2_ref[s] + gamma_n * g2[s])
    print(f"  Reference Nash approx: pi1={pi1_ref}, pi2={pi2_ref}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Heterogeneous sample budgets
    K1, K2 = 8, 1

    for label, use_ew, color in [('Standard PG', False, '#e74c3c'), ('EWPG', True, '#2ecc71')]:
        all_dist = np.zeros((n_runs, n_episodes))
        for run in range(n_runs):
            np.random.seed(run)
            pi1 = np.ones((2, 2)) / 2
            pi2 = np.ones((2, 2)) / 2
            # Small perturbation
            pi1[0] = project_simplex(pi1[0] + 0.1*np.random.randn(2))
            pi1[1] = project_simplex(pi1[1] + 0.1*np.random.randn(2))

            for n in range(1, n_episodes + 1):
                gamma_n = 0.1 / (n + 50)**0.75
                g1 = game.reinforce_gradient(pi1, pi2, 0, n_episodes=K1)
                g2 = game.reinforce_gradient(pi1, pi2, 1, n_episodes=K2)

                if use_ew:
                    V = np.array([float(K1), float(K2)])
                    w = V / V.max()
                else:
                    w = np.array([1.0, 1.0])

                for s in range(2):
                    pi1[s] = project_simplex(pi1[s] + gamma_n * w[0] * g1[s])
                    pi2[s] = project_simplex(pi2[s] + gamma_n * w[1] * g2[s])

                dist = np.sum((pi1 - pi1_ref)**2) + np.sum((pi2 - pi2_ref)**2)
                all_dist[run, n-1] = dist

            if run % 10 == 0:
                print(f"    {label} run {run}/{n_runs}")

        mean = all_dist.mean(axis=0)
        se = all_dist.std(axis=0) / np.sqrt(n_runs)
        window = 100
        smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
        axes[0].semilogy(range(window-1, n_episodes), smooth, label=label,
                        color=color, linewidth=1.5)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel(r'$\|\pi_n - \pi^*\|^2$')
    axes[0].set_title(f'Multi-State Stochastic Game (K=({K1},{K2}))')
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # Right: per-state convergence
    # Re-run one example to get per-state distances
    np.random.seed(0)
    pi1 = np.ones((2, 2)) / 2
    pi2 = np.ones((2, 2)) / 2
    dist_s0, dist_s1 = [], []
    for n in range(1, n_episodes + 1):
        gamma_n = 0.1 / (n + 50)**0.75
        g1 = game.reinforce_gradient(pi1, pi2, 0, n_episodes=K1)
        g2 = game.reinforce_gradient(pi1, pi2, 1, n_episodes=K2)
        V = np.array([float(K1), float(K2)])
        w = V / V.max()
        for s in range(2):
            pi1[s] = project_simplex(pi1[s] + gamma_n * w[0] * g1[s])
            pi2[s] = project_simplex(pi2[s] + gamma_n * w[1] * g2[s])
        dist_s0.append(np.sum((pi1[0]-pi1_ref[0])**2) + np.sum((pi2[0]-pi2_ref[0])**2))
        dist_s1.append(np.sum((pi1[1]-pi1_ref[1])**2) + np.sum((pi2[1]-pi2_ref[1])**2))

    window = 100
    axes[1].semilogy(range(window-1, n_episodes),
                     np.convolve(dist_s0, np.ones(window)/window, mode='valid'),
                     label='State 0 (zero-sum)', color='#3498db', linewidth=1.5)
    axes[1].semilogy(range(window-1, n_episodes),
                     np.convolve(dist_s1, np.ones(window)/window, mode='valid'),
                     label="State 1 (PD)", color='#e67e22', linewidth=1.5)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel(r'Per-state $\|\pi_n(s) - \pi^*(s)\|^2$')
    axes[1].set_title('EWPG: Per-State Convergence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'stochastic_game.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: stochastic_game.png")


# ============================================================
# 3. ADAPTIVE EVIDENCE WEIGHT ESTIMATION
# ============================================================

def experiment_adaptive_evidence():
    """Agents estimate their own V_i from recent gradient variance."""
    print("\n" + "="*60)
    print("Experiment: Adaptive evidence weight estimation")
    print("="*60)

    game = MatrixGame("Matching Pennies", [[1,-1],[-1,1]], [[-1,1],[1,-1]])
    p1_star, p2_star = np.array([0.5, 0.5]), np.array([0.5, 0.5])

    n_episodes = 8000
    n_runs = 60

    # Agent 1 gets K1 samples, agent 2 gets K2
    K1, K2 = 10, 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = {
        'Standard PG': 'fixed_equal',
        'EWPG (oracle V)': 'fixed_oracle',
        'EWPG (adaptive V)': 'adaptive',
    }
    colors_m = {'Standard PG': '#e74c3c', 'EWPG (oracle V)': '#2ecc71', 'EWPG (adaptive V)': '#3498db'}

    for label, method in methods.items():
        all_dist = np.zeros((n_runs, n_episodes))
        all_weights = np.zeros((n_runs, n_episodes, 2))

        for run in range(n_runs):
            np.random.seed(run * 100)
            p1 = np.array([0.6, 0.4])
            p2 = np.array([0.4, 0.6])

            # Running variance estimator for adaptive weights
            grad_history_1 = []
            grad_history_2 = []
            window_size = 50

            for n in range(1, n_episodes + 1):
                gamma_n = 0.3 / (n + 10)**0.75

                g1 = real_reinforce_gradient(game, p1, p2, 0, n_episodes_sample=K1)
                g2 = real_reinforce_gradient(game, p1, p2, 1, n_episodes_sample=K2)

                if method == 'fixed_equal':
                    w1, w2 = 1.0, 1.0
                elif method == 'fixed_oracle':
                    V = np.array([float(K1), float(K2)])
                    w = V / V.max()
                    w1, w2 = w[0], w[1]
                elif method == 'adaptive':
                    grad_history_1.append(g1.copy())
                    grad_history_2.append(g2.copy())
                    if len(grad_history_1) > window_size:
                        grad_history_1.pop(0)
                        grad_history_2.pop(0)

                    if len(grad_history_1) >= 10:
                        var1 = np.var(grad_history_1, axis=0).sum() + 1e-6
                        var2 = np.var(grad_history_2, axis=0).sum() + 1e-6
                        # V_i estimated as inverse variance
                        V_est = np.array([1.0/var1, 1.0/var2])
                        w_est = V_est / V_est.max()
                        w1, w2 = w_est[0], w_est[1]
                    else:
                        w1, w2 = 1.0, 1.0

                p1 = project_simplex(p1 + gamma_n * w1 * g1)
                p2 = project_simplex(p2 + gamma_n * w2 * g2)

                dist = np.linalg.norm(p1 - p1_star)**2 + np.linalg.norm(p2 - p2_star)**2
                all_dist[run, n-1] = dist
                all_weights[run, n-1] = [w1, w2]

        mean = all_dist.mean(axis=0)
        window = 100
        smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
        axes[0].semilogy(range(window-1, n_episodes), smooth, label=label,
                        color=colors_m[label], linewidth=1.5)

        # Track weights for adaptive
        if method == 'adaptive':
            w_mean = all_weights.mean(axis=0)
            axes[1].plot(w_mean[:, 0], label=r'$w_1$ (K=10)', color='#3498db', linewidth=1)
            axes[1].plot(w_mean[:, 1], label=r'$w_2$ (K=1)', color='#e74c3c', linewidth=1)
            axes[1].axhline(K1/(K1+K2)*2, color='#3498db', linestyle='--', alpha=0.5,
                           label=f'Oracle $w_1$={K1/max(K1,K2):.1f}')
            axes[1].axhline(K2/(K1+K2)*2, color='#e74c3c', linestyle='--', alpha=0.5,
                           label=f'Oracle $w_2$={K2/max(K1,K2):.1f}')

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel(r'$\|\pi_n - \pi^*\|^2$')
    axes[0].set_title(f'Convergence: PG vs Oracle vs Adaptive EWPG')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Evidence weight $w_i$')
    axes[1].set_title('Adaptive Weight Estimation')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)
    axes[1].set_ylim(-0.1, 1.5)

    # Right: bar chart comparing final distances
    final_labels = list(methods.keys())
    final_dists = []
    for label, method in methods.items():
        # Re-extract from all_dist of last run (hack: re-run quickly)
        np.random.seed(0)
        p1 = np.array([0.6, 0.4])
        p2 = np.array([0.4, 0.6])
        gh1, gh2 = [], []
        for n in range(1, 4001):
            gamma_n = 0.3 / (n+10)**0.75
            g1 = real_reinforce_gradient(game, p1, p2, 0, n_episodes_sample=K1)
            g2 = real_reinforce_gradient(game, p1, p2, 1, n_episodes_sample=K2)
            if method == 'fixed_equal':
                w1, w2 = 1.0, 1.0
            elif method == 'fixed_oracle':
                w1, w2 = 1.0, K2/K1
            else:
                gh1.append(g1.copy()); gh2.append(g2.copy())
                if len(gh1)>50: gh1.pop(0); gh2.pop(0)
                if len(gh1)>=10:
                    v1 = np.var(gh1,axis=0).sum()+1e-6; v2 = np.var(gh2,axis=0).sum()+1e-6
                    Ve = np.array([1/v1,1/v2]); we = Ve/Ve.max(); w1,w2 = we
                else:
                    w1,w2 = 1.0,1.0
            p1 = project_simplex(p1+gamma_n*w1*g1)
            p2 = project_simplex(p2+gamma_n*w2*g2)
        final_dists.append(np.linalg.norm(p1-p1_star)**2 + np.linalg.norm(p2-p2_star)**2)

    bars = axes[2].bar(range(3), final_dists,
                       color=[colors_m[l] for l in final_labels], alpha=0.8)
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(['PG', 'Oracle\nEWPG', 'Adaptive\nEWPG'], fontsize=9)
    axes[2].set_ylabel(r'Final $\|\pi - \pi^*\|^2$')
    axes[2].set_title('Final Distance (single run)')
    axes[2].grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'adaptive_evidence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: adaptive_evidence.png")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Roadmap Experiments: Addressing Reviewer Critiques")
    print("=" * 60)

    experiment_real_reinforce()
    experiment_stochastic_game()
    experiment_adaptive_evidence()

    print(f"\nAll roadmap experiments complete. Figures saved to: {FIGURES_DIR}")
