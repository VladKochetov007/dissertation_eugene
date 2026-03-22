"""
Full Experimental Suite for the Ω-Framework Dissertation

Five experiments validating the theoretical results:
  1. EW-PG variance reduction (existing, refined)
  2. Blessing of dimensionality — logarithmic scaling in action space
  3. Coalition formation and communication bottleneck (Coop-PG)
  4. Combined Ω-PG: LOLA + Cooperation + Evidence weighting
  5. Sparsity curriculum with L1 regularization

Author: Eugene Shcherbinin
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from collections import defaultdict
import time

# ============================================================
# Setup
# ============================================================

FIGURES_DIR = Path(__file__).parent / "figures" / "full"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# Plotting defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

COLORS = {
    'standard': '#e74c3c',
    'ewpg': '#2ecc71',
    'lola': '#3498db',
    'coop': '#9b59b6',
    'omega': '#f39c12',
    'l1': '#1abc9c',
    'entropy': '#e67e22',
    'sparse_ew': '#2c3e50',
}


# ============================================================
# Core: N-player General-Sum Matrix Game
# ============================================================

class NPlayerGame:
    """N-player matrix game with arbitrary action spaces.

    payoffs[i] is an N-dimensional array: payoffs[i][a1, a2, ..., aN] = R_i(a)
    """
    def __init__(self, name, payoffs, nash=None):
        self.name = name
        self.N = len(payoffs)
        self.payoffs = [np.array(R, dtype=float) for R in payoffs]
        self.n_actions = [R.shape[i] for i, R in enumerate(payoffs)]
        self.nash = nash  # list of Nash equilibria (mixed strategies)

    def expected_payoff(self, policies):
        """Expected payoff for all agents given mixed strategy profiles."""
        # Compute joint probability tensor
        joint = policies[0]
        for pi in policies[1:]:
            joint = np.tensordot(joint, pi, axes=0)
        return [np.sum(joint * R) for R in self.payoffs]

    def policy_gradient(self, policies, player):
        """Exact policy gradient for player i.
        dV_i/dpi_i = sum_{a_{-i}} pi_{-i}(a_{-i}) * R_i(a_i, a_{-i})
        """
        n_a = self.n_actions[player]
        R = self.payoffs[player]

        # Marginalize over all other players
        other_joint = np.ones(1)
        for j in range(self.N):
            if j != player:
                other_joint = np.tensordot(other_joint, policies[j], axes=0)

        # Contract with payoff matrix
        grad = np.zeros(n_a)
        for a_i in range(n_a):
            # Select the slice for this action
            idx = [slice(None)] * self.N
            idx[player] = a_i
            grad[a_i] = np.sum(other_joint.flatten() * R[tuple(idx)].flatten())

        return grad

    def reinforce_estimate(self, policies, player, n_samples=1):
        """REINFORCE gradient estimate via sampling."""
        n_a = self.n_actions[player]
        grads = []
        for _ in range(n_samples):
            actions = [np.random.choice(len(p), p=p) for p in policies]
            reward = self.payoffs[player][tuple(actions)]
            log_grad = np.zeros(n_a)
            a_i = actions[player]
            log_grad[a_i] = 1.0 / max(policies[player][a_i], 1e-10)
            grads.append(reward * log_grad)
        return np.mean(grads, axis=0)


def project_simplex(x, min_prob=1e-8):
    """Project onto the probability simplex."""
    x = np.maximum(x, min_prob)
    return x / x.sum()


def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()


# ============================================================
# Standard Games
# ============================================================

def make_matching_pennies(d=2):
    """Matching pennies with d actions. First 2 are the real game,
    rest are dummy actions with zero payoff (for blessing experiment)."""
    R1 = np.zeros((d, d))
    R2 = np.zeros((d, d))
    # Core 2x2 game
    R1[0, 0] = 1;  R1[0, 1] = -1
    R1[1, 0] = -1; R1[1, 1] = 1
    R2[0, 0] = -1; R2[0, 1] = 1
    R2[1, 0] = 1;  R2[1, 1] = -1
    nash = [np.ones(d) / d, np.ones(d) / d]  # Uniform over first 2 is NE
    # Actually Nash: 0.5 on action 0, 0.5 on action 1, 0 on rest
    nash = [np.zeros(d), np.zeros(d)]
    nash[0][:2] = 0.5; nash[1][:2] = 0.5
    return NPlayerGame(f"MP-{d}", [R1, R2], nash)


def make_team_game(n_players=4, n_actions=3):
    """General-sum team game: agents split into 2 teams.
    Within team: coordination bonus. Between teams: zero-sum."""
    team_size = n_players // 2
    shape = tuple([n_actions] * n_players)
    payoffs = []
    for i in range(n_players):
        R = np.random.randn(*shape) * 0.1  # small base noise
        team = 0 if i < team_size else 1
        # Coordination bonus: agents in same team matching actions
        for idx in np.ndindex(*shape):
            team_actions = [idx[j] for j in range(n_players) if (j < team_size) == (i < team_size)]
            if len(set(team_actions)) == 1:  # all same action
                R[idx] += 1.0
            # Penalty if opposing team coordinates
            opp_actions = [idx[j] for j in range(n_players) if (j < team_size) != (i < team_size)]
            if len(set(opp_actions)) == 1:
                R[idx] -= 0.5
        payoffs.append(R)
    return NPlayerGame(f"Team-{n_players}p-{n_actions}a", payoffs)


# ============================================================
# Experiment 1: EW-PG Variance Reduction (refined)
# ============================================================

def experiment_1_ewpg_variance(n_episodes=3000, n_runs=50):
    """Validate HM/AM variance improvement across multiple games."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Evidence-Weighted PG Variance Reduction")
    print("=" * 60)

    games = [make_matching_pennies(2), make_matching_pennies(3)]
    evidence_configs = [
        {"V": np.array([1.0, 1.0]), "label": "Equal (V=[1,1])"},
        {"V": np.array([5.0, 1.0]), "label": "Moderate (V=[5,1])"},
        {"V": np.array([20.0, 1.0]), "label": "Extreme (V=[20,1])"},
    ]

    fig, axes = plt.subplots(len(games), len(evidence_configs),
                              figsize=(5*len(evidence_configs), 4*len(games)))
    if len(games) == 1:
        axes = axes[np.newaxis, :]

    for gi, game in enumerate(games):
        for ei, cfg in enumerate(evidence_configs):
            ax = axes[gi, ei]
            V = cfg["V"]
            noise_scales = 2.0 / np.sqrt(V)
            w = V.min() / V  # evidence weights

            # Theoretical HM/AM
            N = len(V)
            AM = V.mean()
            HM = N / np.sum(1.0 / V)
            ratio = HM / AM

            std_all = np.zeros((n_runs, n_episodes))
            ew_all = np.zeros((n_runs, n_episodes))

            for run in range(n_runs):
                init = [project_simplex(np.ones(n) / n + 0.05 * np.random.randn(n))
                        for n in game.n_actions]

                # Standard PG
                policies = [p.copy() for p in init]
                for n in range(n_episodes):
                    gamma = 0.5 / (n + 10) ** 0.75
                    for i in range(game.N):
                        g = game.reinforce_estimate(policies, i, n_samples=3)
                        g += np.random.randn(*g.shape) * noise_scales[i]
                        policies[i] = project_simplex(policies[i] + gamma * g)
                    dist = sum(np.linalg.norm(policies[i] - game.nash[i])**2
                               for i in range(game.N))
                    std_all[run, n] = dist

                # EW-PG
                policies = [p.copy() for p in init]
                for n in range(n_episodes):
                    gamma = 0.5 / (n + 10) ** 0.75
                    for i in range(game.N):
                        g = game.reinforce_estimate(policies, i, n_samples=3)
                        g += np.random.randn(*g.shape) * noise_scales[i]
                        policies[i] = project_simplex(policies[i] + gamma * w[i] * g)
                    dist = sum(np.linalg.norm(policies[i] - game.nash[i])**2
                               for i in range(game.N))
                    ew_all[run, n] = dist

            # Smooth and plot
            window = 50
            std_mean = np.convolve(std_all.mean(0), np.ones(window)/window, 'valid')
            ew_mean = np.convolve(ew_all.mean(0), np.ones(window)/window, 'valid')
            x = np.arange(len(std_mean))

            ax.semilogy(x, std_mean, color=COLORS['standard'], label='Standard PG')
            ax.semilogy(x, ew_mean, color=COLORS['ewpg'], label='EW-PG')
            ax.set_title(f'{game.name}, {cfg["label"]}\nHM/AM={ratio:.3f}')
            ax.set_xlabel('Episode')
            ax.set_ylabel(r'$\|\pi - \pi^*\|^2$')
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 1: EW-PG Variance Reduction', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp1_ewpg_variance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved exp1_ewpg_variance.png")


# ============================================================
# Experiment 2: Blessing of Dimensionality
# ============================================================

def experiment_2_blessing(n_runs=30, n_episodes=2000):
    """Action space expansion: measure GRADIENT VARIANCE vs dimension.

    Instead of convergence time (which depends on many hyperparameters),
    we measure the key theoretical prediction directly:
    - Standard PG: gradient variance scales as O(d)
    - L1 sparse PG: gradient variance scales as O(k log d)

    We use exact gradients + calibrated noise to isolate the
    dimensionality effect from REINFORCE sampling difficulties.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Blessing of Dimensionality")
    print("=" * 60)

    dimensions = [2, 5, 10, 20, 50, 100, 200, 500]

    methods = {
        'Standard PG': {'l1': 0.0, 'ew': False, 'entropy': 0.0},
        'PG + Entropy': {'l1': 0.0, 'ew': False, 'entropy': 0.01},
        'PG + L1': {'l1': 0.1, 'ew': False, 'entropy': 0.0},
        'Sparse-EW-PG': {'l1': 0.1, 'ew': True, 'entropy': 0.0},
    }

    # Track: final distance to Nash, gradient variance at end
    results_dist = {name: [] for name in methods}
    results_var = {name: [] for name in methods}
    results_dist_std = {name: [] for name in methods}
    results_var_std = {name: [] for name in methods}

    for d in dimensions:
        game = make_matching_pennies(d)
        print(f"\n  d = {d} (k=2 active actions)")

        for method_name, params in methods.items():
            final_dists = []
            final_vars = []

            for run in range(n_runs):
                np.random.seed(run * 10000 + d)
                # Start with slight bias toward correct actions
                init = np.ones(d) / d
                policies = [project_simplex(init + 0.02 * np.random.randn(d))
                            for _ in range(2)]

                V_est = np.ones(game.N)
                recent_var = []

                for n in range(1, n_episodes + 1):
                    gamma = 1.0 / (n + 10) ** 0.6

                    for i in range(game.N):
                        # Use exact gradient + scaled noise (simulates REINFORCE
                        # with known variance structure)
                        g_exact = game.policy_gradient(policies, i)
                        # Noise scales with sqrt(d) as in REINFORCE
                        noise_scale = 0.5 * np.sqrt(d) / (n + 10)**0.25
                        g = g_exact + np.random.randn(d) * noise_scale

                        # Track gradient variance
                        if n > n_episodes - 200:
                            recent_var.append(np.linalg.norm(g - g_exact)**2)

                        V_est[i] = 0.9 * V_est[i] + 0.1 * np.linalg.norm(g)**2

                        w = 1.0
                        if params['ew'] and n > 50:
                            V_min = max(V_est.min(), 1e-6)
                            w = V_min / max(V_est[i], 1e-6)

                        new_p = policies[i] + gamma * w * g

                        if params['entropy'] > 0:
                            ent_grad = -(np.log(np.maximum(policies[i], 1e-10)) + 1)
                            new_p += gamma * params['entropy'] * ent_grad

                        if params['l1'] > 0:
                            mu_n = params['l1'] / (1 + n / 300)
                            new_p = np.sign(new_p) * np.maximum(
                                np.abs(new_p) - gamma * mu_n, 0)

                        policies[i] = project_simplex(new_p)

                dist = sum(np.linalg.norm(policies[i] - game.nash[i])**2
                           for i in range(game.N))
                final_dists.append(dist)
                final_vars.append(np.mean(recent_var) if recent_var else 0)

            results_dist[method_name].append(np.mean(final_dists))
            results_var[method_name].append(np.mean(final_vars))
            results_dist_std[method_name].append(np.std(final_dists) / np.sqrt(n_runs))
            results_var_std[method_name].append(np.std(final_vars) / np.sqrt(n_runs))

            print(f"    {method_name:20s}: dist={np.mean(final_dists):.4f}, "
                  f"var={np.mean(final_vars):.2f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, color_key in [('Standard PG', 'standard'), ('PG + Entropy', 'entropy'),
                             ('PG + L1', 'l1'), ('Sparse-EW-PG', 'sparse_ew')]:
        ax1.loglog(dimensions, results_dist[name], label=name, color=COLORS[color_key],
                   marker='o', linewidth=2)
    ax1.set_xlabel('Action space dimension $d$')
    ax1.set_ylabel(r'Final $\|\pi - \pi^*\|^2$')
    ax1.set_title('Distance to Nash vs Action Space')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for name, color_key in [('Standard PG', 'standard'), ('PG + Entropy', 'entropy'),
                             ('PG + L1', 'l1'), ('Sparse-EW-PG', 'sparse_ew')]:
        ax2.loglog(dimensions, results_var[name], label=name, color=COLORS[color_key],
                   marker='o', linewidth=2)
    # Reference lines
    ds = np.array(dimensions, dtype=float)
    ax2.loglog(ds, ds * 0.25, 'k:', alpha=0.4, label=r'$O(d)$ reference')
    ax2.loglog(ds, np.log(ds + 1) * 5, 'k--', alpha=0.4, label=r'$O(\log d)$ reference')
    ax2.set_xlabel('Action space dimension $d$')
    ax2.set_ylabel('Gradient variance')
    ax2.set_title('Gradient Variance vs Action Space (key prediction)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Experiment 2: Blessing of Dimensionality\n'
                 '2-action game embedded in d-dimensional space (k=2 sparse)',
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp2_blessing.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n  Saved exp2_blessing.png")


# ============================================================
# Experiment 3: Coalition Formation and Communication
# ============================================================

def experiment_3_coalition(n_episodes=3000, n_runs=20):
    """4-player team game with coalition discovery.
    Agents must discover that 2v2 coordination improves payoff.
    Test: communication quality vs evidence level.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Coalition Formation & Communication Bottleneck")
    print("=" * 60)

    game = make_team_game(n_players=4, n_actions=3)

    # Three conditions:
    # (a) All high evidence (low noise)
    # (b) Mixed: team1 high evidence, team2 "vibing" (high noise)
    # (c) All vibing (high noise)
    conditions = [
        {"name": "All articulate", "noise": [0.1, 0.1, 0.1, 0.1]},
        {"name": "Mixed (team1 art., team2 vibe)", "noise": [0.1, 0.1, 2.0, 2.0]},
        {"name": "All vibing", "noise": [2.0, 2.0, 2.0, 2.0]},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ci, cond in enumerate(conditions):
        ax = axes[ci]
        noise = np.array(cond["noise"])
        print(f"\n  Condition: {cond['name']}")

        # Run three methods: no cooperation, with cooperation, full (coop + EW)
        for method, color_key, label in [
            ('standard', 'standard', 'Standard PG'),
            ('coop', 'coop', 'Coop-PG'),
            ('ew_coop', 'omega', 'EW-Coop-PG'),
        ]:
            all_payoffs = np.zeros((n_runs, n_episodes))

            for run in range(n_runs):
                np.random.seed(run * 100 + ci)
                policies = [project_simplex(np.ones(na) / na)
                            for na in game.n_actions]

                # Communication buffers (last communicated policy)
                comm_policies = [p.copy() for p in policies]
                V_est = np.ones(game.N)

                for n in range(n_episodes):
                    gamma = 0.3 / (n + 10) ** 0.7
                    beta = 0.5 / (n + 10) ** 0.5  # cooperation schedule

                    for i in range(game.N):
                        g = game.reinforce_estimate(policies, i, n_samples=3)
                        g += np.random.randn(*g.shape) * noise[i]

                        # Evidence weight
                        V_est[i] = 0.95 * V_est[i] + 0.05 * np.linalg.norm(g)**2
                        w = 1.0
                        if method == 'ew_coop':
                            V_min = max(np.min(V_est), 1e-6)
                            w = V_min / max(V_est[i], 1e-6)

                        # Cooperative term: use communicated policies of teammates
                        coop_grad = np.zeros_like(g)
                        if method in ('coop', 'ew_coop'):
                            team = [j for j in range(game.N)
                                    if (j < 2) == (i < 2) and j != i]
                            for j in team:
                                # Compute gradient of team reward wrt pi_i
                                # using j's communicated policy
                                test_policies = [comm_policies[k].copy()
                                                 for k in range(game.N)]
                                test_policies[i] = policies[i]
                                coop_grad += game.policy_gradient(test_policies, i)
                            coop_grad /= max(len(team), 1)

                        new_p = policies[i] + gamma * w * (g + beta * coop_grad)
                        policies[i] = project_simplex(new_p)

                    # Update communication (with noise = self-knowledge loss)
                    for i in range(game.N):
                        comm_noise = noise[i] * 0.5  # communication channel noise
                        comm_policies[i] = project_simplex(
                            policies[i] + np.random.randn(*policies[i].shape) * comm_noise
                        )

                    # Track team payoffs
                    payoffs = game.expected_payoff(policies)
                    all_payoffs[run, n] = sum(payoffs)

            # Smooth and plot
            mean_pay = np.convolve(all_payoffs.mean(0), np.ones(50)/50, 'valid')
            ax.plot(range(len(mean_pay)), mean_pay, color=COLORS[color_key],
                    label=label, linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Total payoff')
        ax.set_title(cond['name'])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 3: Coalition Formation & Communication\n'
                 '4-player team game (2v2), testing self-knowledge bottleneck',
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp3_coalition.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n  Saved exp3_coalition.png")


# ============================================================
# Experiment 4: Full Ω-PG (EW + LOLA + Coop)
# ============================================================

def experiment_4_full_omega(n_episodes=3000, n_runs=20):
    """6-player game: 2 teams of 3. Within-team cooperation,
    between-team opponent shaping. Compare all method combinations."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Full Ω-PG (EW + LOLA + Coop)")
    print("=" * 60)

    game = make_team_game(n_players=4, n_actions=3)
    noise = np.array([0.5, 0.5, 0.5, 0.5])

    methods = {
        'Standard PG': dict(ew=False, lola=False, coop=False),
        'EW-PG': dict(ew=True, lola=False, coop=False),
        'LOLA-PG': dict(ew=False, lola=True, coop=False),
        'Coop-PG': dict(ew=False, lola=False, coop=True),
        'Full Ω-PG': dict(ew=True, lola=True, coop=True),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    all_results = {}

    for method_name, flags in methods.items():
        color = {'Standard PG': COLORS['standard'], 'EW-PG': COLORS['ewpg'],
                 'LOLA-PG': COLORS['lola'], 'Coop-PG': COLORS['coop'],
                 'Full Ω-PG': COLORS['omega']}[method_name]

        payoff_runs = np.zeros((n_runs, n_episodes))
        variance_runs = np.zeros((n_runs, n_episodes))

        for run in range(n_runs):
            np.random.seed(run * 777)
            policies = [project_simplex(np.ones(na) / na) for na in game.n_actions]
            comm_policies = [p.copy() for p in policies]
            V_est = np.ones(game.N)
            grad_var_history = []

            for n in range(n_episodes):
                gamma = 0.3 / (n + 10) ** 0.7
                lam = 0.3 / (n + 10) ** 0.5 if flags['lola'] else 0.0
                beta = 0.3 / (n + 10) ** 0.5 if flags['coop'] else 0.0

                step_grads = []
                for i in range(game.N):
                    g = game.reinforce_estimate(policies, i, n_samples=3)
                    g += np.random.randn(*g.shape) * noise[i]
                    step_grads.append(np.linalg.norm(g)**2)

                    # Evidence
                    V_est[i] = 0.95 * V_est[i] + 0.05 * np.linalg.norm(g)**2
                    w = 1.0
                    if flags['ew']:
                        V_min = max(np.min(V_est), 1e-6)
                        w = V_min / max(V_est[i], 1e-6)

                    # LOLA: opponent shaping (simplified — use gradient of
                    # opponent's anticipated response)
                    os_grad = np.zeros_like(g)
                    if flags['lola']:
                        opponents = [j for j in range(game.N)
                                     if (j < 2) != (i < 2)]
                        for j in opponents:
                            # Approximate: cross-derivative via finite difference
                            eps = 0.01
                            p_plus = [p.copy() for p in policies]
                            p_plus[j] = project_simplex(p_plus[j] + eps * np.ones_like(p_plus[j]))
                            g_plus = game.policy_gradient(p_plus, i)
                            os_grad += (g_plus - game.policy_gradient(policies, i)) / eps
                        os_grad /= max(len(opponents), 1)

                    # Cooperation
                    coop_grad = np.zeros_like(g)
                    if flags['coop']:
                        team = [j for j in range(game.N)
                                if (j < 2) == (i < 2) and j != i]
                        for j in team:
                            test_p = [comm_policies[k].copy() for k in range(game.N)]
                            test_p[i] = policies[i]
                            coop_grad += game.policy_gradient(test_p, i)
                        coop_grad /= max(len(team), 1)

                    # Full update
                    full_g = g + lam * os_grad + beta * coop_grad
                    policies[i] = project_simplex(policies[i] + gamma * w * full_g)

                # Update communication
                for i in range(game.N):
                    comm_policies[i] = project_simplex(
                        policies[i] + np.random.randn(*policies[i].shape) * noise[i] * 0.3
                    )

                payoffs = game.expected_payoff(policies)
                payoff_runs[run, n] = sum(payoffs)
                variance_runs[run, n] = np.mean(step_grads)

        # Plot payoffs
        mean_pay = np.convolve(payoff_runs.mean(0), np.ones(50)/50, 'valid')
        ax1.plot(range(len(mean_pay)), mean_pay, color=color,
                 label=method_name, linewidth=2)

        # Plot gradient variance
        mean_var = np.convolve(variance_runs.mean(0), np.ones(100)/100, 'valid')
        ax2.semilogy(range(len(mean_var)), mean_var, color=color,
                     label=method_name, linewidth=2)

        all_results[method_name] = payoff_runs.mean(0)[-500:].mean()
        print(f"  {method_name:15s}: mean payoff (last 500) = {all_results[method_name]:.4f}")

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total payoff')
    ax1.set_title('Total Team Payoff')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Gradient variance')
    ax2.set_title('Gradient Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Experiment 4: Full Ω-PG Comparison\n'
                 '4-player team game (2v2), all method combinations',
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp4_full_omega.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n  Saved exp4_full_omega.png")


# ============================================================
# Experiment 5: Sparsity Curriculum
# ============================================================

def experiment_5_sparsity(n_episodes=3000, n_runs=20):
    """Large action space with sparse optimum. Track d_eff, sparsity,
    and the curriculum phases over training."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Sparsity Curriculum (L1 + Evidence Weighting)")
    print("=" * 60)

    d = 50  # 50 actions, only 3 are optimal
    game = make_matching_pennies(d)  # 2 active, 48 dummy

    # Track metrics over training
    metrics = {name: defaultdict(list) for name in
               ['Standard PG', 'PG + Entropy', 'Sparse-EW-PG']}

    for method in metrics:
        print(f"\n  {method}:")
        for run in range(n_runs):
            np.random.seed(run * 555)
            policies = [project_simplex(np.ones(d) / d + 0.001 * np.random.randn(d))
                        for _ in range(2)]
            V_est = np.ones(2)
            grad_history = [[] for _ in range(2)]

            d_eff_trace = []
            sparsity_trace = []
            dist_trace = []

            for n in range(1, n_episodes + 1):
                gamma = 0.5 / (n + 10) ** 0.75

                for i in range(2):
                    g = game.reinforce_estimate(policies, i, n_samples=5)

                    # Track gradient covariance for d_eff
                    grad_history[i].append(g)
                    if len(grad_history[i]) > 30:
                        grad_history[i] = grad_history[i][-50:]

                    V_est[i] = 0.95 * V_est[i] + 0.05 * np.linalg.norm(g)**2
                    w = 1.0

                    if method == 'Sparse-EW-PG':
                        V_min = max(V_est.min(), 1e-6)
                        w = V_min / max(V_est[i], 1e-6)

                    new_p = policies[i] + gamma * w * g

                    if method == 'PG + Entropy':
                        ent_grad = -(np.log(policies[i] + 1e-10) + 1)
                        new_p += gamma * 0.01 * ent_grad

                    if method == 'Sparse-EW-PG':
                        mu_n = 0.05 / (1 + n / 500)
                        new_p = np.sign(new_p) * np.maximum(np.abs(new_p) - gamma * mu_n, 0)

                    policies[i] = project_simplex(new_p)

                # Compute metrics
                # Effective dimension: use gradient history
                if len(grad_history[0]) >= 20:
                    G = np.array(grad_history[0][-20:])
                    G = G - G.mean(0)
                    if G.shape[0] > 1:
                        cov = G.T @ G / G.shape[0]
                        eigs = np.linalg.eigvalsh(cov)
                        eigs = np.maximum(eigs, 0)
                        if eigs.max() > 1e-10:
                            d_eff = eigs.sum() / eigs.max()
                        else:
                            d_eff = 1.0
                    else:
                        d_eff = 1.0
                else:
                    d_eff = d

                # Sparsity: number of actions with pi > 0.01
                n_active = sum(1 for p in policies for a in p if a > 0.01)
                sparsity = n_active / (2 * d)

                dist = sum(np.linalg.norm(policies[i] - game.nash[i])**2
                           for i in range(2))

                d_eff_trace.append(d_eff)
                sparsity_trace.append(sparsity)
                dist_trace.append(dist)

            metrics[method]['d_eff'].append(d_eff_trace)
            metrics[method]['sparsity'].append(sparsity_trace)
            metrics[method]['dist'].append(dist_trace)

        print(f"    Final d_eff = {np.mean([t[-1] for t in metrics[method]['d_eff']]):.1f}")
        print(f"    Final sparsity = {np.mean([t[-1] for t in metrics[method]['sparsity']]):.3f}")
        print(f"    Final dist = {np.mean([t[-1] for t in metrics[method]['dist']]):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for method, color_key in [('Standard PG', 'standard'),
                               ('PG + Entropy', 'entropy'),
                               ('Sparse-EW-PG', 'sparse_ew')]:
        # d_eff
        mean_deff = np.mean(metrics[method]['d_eff'], axis=0)
        smooth_deff = np.convolve(mean_deff, np.ones(50)/50, 'valid')
        axes[0].plot(range(len(smooth_deff)), smooth_deff,
                     color=COLORS[color_key], label=method, linewidth=2)

        # Sparsity
        mean_sp = np.mean(metrics[method]['sparsity'], axis=0)
        smooth_sp = np.convolve(mean_sp, np.ones(50)/50, 'valid')
        axes[1].plot(range(len(smooth_sp)), smooth_sp,
                     color=COLORS[color_key], label=method, linewidth=2)

        # Distance to Nash
        mean_dist = np.mean(metrics[method]['dist'], axis=0)
        smooth_dist = np.convolve(mean_dist, np.ones(50)/50, 'valid')
        axes[2].semilogy(range(len(smooth_dist)), smooth_dist,
                         color=COLORS[color_key], label=method, linewidth=2)

    axes[0].set_xlabel('Episode'); axes[0].set_ylabel(r'$d_{\mathrm{eff}}$')
    axes[0].set_title(f'Effective Dimension (ambient d={d})')
    axes[0].axhline(y=2, color='gray', linestyle=':', alpha=0.5, label='k=2 (true)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Fraction active (>0.01)')
    axes[1].set_title('Policy Sparsity')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Episode'); axes[2].set_ylabel(r'$\|\pi - \pi^*\|^2$')
    axes[2].set_title('Distance to Nash')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'Experiment 5: Sparsity Curriculum (d={d}, k=2)\n'
                 'L1 regularization discovers sparse support, then refines',
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp5_sparsity.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n  Saved exp5_sparsity.png")


# ============================================================
# Experiment 2b: Effective dimension tracking
# ============================================================

def experiment_2b_deff_scaling(n_runs=20):
    """Measure d_eff vs ambient d for sparse games.

    Key fix: use MANY gradient samples (200+) so covariance estimation
    is well-conditioned even in high d. We measure d_eff of the
    TRUE gradient noise, not the policy iterates.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2b: Effective Dimension Scaling")
    print("=" * 60)

    dimensions = [5, 10, 20, 50, 100, 200]
    n_grad_samples = 300  # enough to estimate covariance in all dims

    deff_by_method = {'Standard PG': [], 'Sparse-EW-PG': []}
    deff_std_by_method = {'Standard PG': [], 'Sparse-EW-PG': []}

    for d in dimensions:
        game = make_matching_pennies(d)

        for method in deff_by_method:
            deffs = []
            for run in range(n_runs):
                np.random.seed(run * 333 + d)

                # Run PG for a bit to get a representative policy
                policies = [project_simplex(np.ones(d)/d) for _ in range(2)]
                for n in range(1, 501):
                    gamma = 1.0 / (n + 10) ** 0.6
                    for i in range(2):
                        g = game.policy_gradient(policies, i)
                        noise = np.random.randn(d) * 0.5 * np.sqrt(d) / (n+10)**0.25
                        new_p = policies[i] + gamma * (g + noise)
                        if method == 'Sparse-EW-PG':
                            mu_n = 0.1 / (1 + n / 300)
                            new_p = np.sign(new_p) * np.maximum(
                                np.abs(new_p) - gamma * mu_n, 0)
                        policies[i] = project_simplex(new_p)

                # Collect many gradient samples at current policy
                grads = []
                for _ in range(n_grad_samples):
                    g = game.reinforce_estimate(policies, 0, n_samples=1)
                    grads.append(g)
                G = np.array(grads)
                G = G - G.mean(0)

                # d_eff = tr(Sigma) / lambda_max(Sigma)
                cov = G.T @ G / (G.shape[0] - 1)
                eigs = np.linalg.eigvalsh(cov)
                eigs = np.maximum(eigs, 0)
                if eigs.max() > 1e-12:
                    d_eff = eigs.sum() / eigs.max()
                else:
                    d_eff = 1.0
                deffs.append(d_eff)

            deff_by_method[method].append(np.mean(deffs))
            deff_std_by_method[method].append(np.std(deffs) / np.sqrt(n_runs))

        print(f"  d={d:4d}: Std PG d_eff={deff_by_method['Standard PG'][-1]:.1f}, "
              f"Sparse-EW d_eff={deff_by_method['Sparse-EW-PG'][-1]:.1f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, color_key in [('Standard PG', 'standard'), ('Sparse-EW-PG', 'sparse_ew')]:
        ax.errorbar(dimensions, deff_by_method[method], yerr=deff_std_by_method[method],
                    label=method, color=COLORS[color_key], marker='o', linewidth=2, capsize=3)
    ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5, label='k=2 (true sparsity)')
    ax.plot(dimensions, dimensions, 'k--', alpha=0.3, label=r'$d_{\mathrm{eff}} = d$')
    ax.set_xlabel('Ambient dimension $d$')
    ax.set_ylabel(r'Effective dimension $d_{\mathrm{eff}}$')
    ax.set_title('Effective Dimension vs Ambient Dimension\n'
                 r'Blessing: $d_{\mathrm{eff}} \approx k$ regardless of $d$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp2b_deff_scaling.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n  Saved exp2b_deff_scaling.png")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FULL EXPERIMENTAL SUITE — Ω-Framework Dissertation")
    print("=" * 60)

    t0 = time.time()

    experiment_1_ewpg_variance(n_episodes=2000, n_runs=30)
    experiment_2_blessing(n_runs=20, n_episodes=1500)
    experiment_2b_deff_scaling(n_runs=15)
    experiment_3_coalition(n_episodes=2000, n_runs=15)
    experiment_4_full_omega(n_episodes=2000, n_runs=15)
    experiment_5_sparsity(n_episodes=2000, n_runs=15)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed:.0f}s")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"{'=' * 60}")
