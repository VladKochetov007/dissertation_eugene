"""
Application Experiments for the Ω-Framework

Three application domains validated:
  A. Federated Learning — EW-FedAvg vs FedAvg vs FedProx
  B. RLHF / LLM Alignment — evidence-weighted reward aggregation
  C. Multi-Agent Debate — judge channel capacity and self-knowledge

Author: Eugene Shcherbinin
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import time

FIGURES_DIR = Path(__file__).parent / "figures" / "applications"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

COLORS = {
    'fedavg': '#e74c3c',
    'fedprox': '#e67e22',
    'scaffold': '#9b59b6',
    'ew_fedavg': '#2ecc71',
    'rlhf_uniform': '#e74c3c',
    'rlhf_tuned': '#3498db',
    'rlhf_ew': '#2ecc71',
    'rlhf_lola': '#f39c12',
    'debate_uniform': '#e74c3c',
    'debate_ew': '#2ecc71',
    'debate_sparse': '#1abc9c',
    'debate_coalition': '#9b59b6',
}


# ============================================================
# APPLICATION A: Federated Learning
# ============================================================

class FederatedTask:
    """Simulated federated learning task.

    N clients each have local data drawn from heterogeneous distributions.
    Goal: learn a global model (parameter vector theta) minimizing
    the average loss across all clients.

    We simulate a simple quadratic loss with client-specific optima:
      f_i(theta) = 0.5 * ||theta - theta_i*||^2
    The global optimum is theta* = mean(theta_i*).

    Non-IID-ness is controlled by the spread of theta_i*.
    """
    def __init__(self, d, n_clients, noniid_scale=1.0, noise_heterogeneity=1.0):
        self.d = d  # parameter dimension
        self.n_clients = n_clients
        # Client-specific optima (non-IID)
        self.theta_star_local = [
            noniid_scale * np.random.randn(d) for _ in range(n_clients)
        ]
        self.theta_star_global = np.mean(self.theta_star_local, axis=0)
        # Heterogeneous noise levels (evidence quality)
        self.noise_scales = np.array([
            0.1 + noise_heterogeneity * np.random.exponential(1.0)
            for _ in range(n_clients)
        ])

    def local_gradient(self, theta, client_id, n_samples=1):
        """Stochastic gradient for client i at theta."""
        # True gradient: theta - theta_i*
        true_grad = theta - self.theta_star_local[client_id]
        # Add noise proportional to client's noise scale
        noise = self.noise_scales[client_id] * np.random.randn(self.d)
        return true_grad + noise

    def global_loss(self, theta):
        """Global loss at theta."""
        return 0.5 * np.linalg.norm(theta - self.theta_star_global)**2


def run_federated(task, method, n_rounds=200, n_local_steps=5, lr=0.1,
                  participation_rate=1.0, mu_prox=0.01):
    """Run federated learning with different methods."""
    theta = np.zeros(task.d)
    losses = []

    for rnd in range(n_rounds):
        # Select participating clients
        n_active = max(1, int(task.n_clients * participation_rate))
        active = np.random.choice(task.n_clients, n_active, replace=False)

        # Local training
        local_deltas = []
        local_variances = []

        for i in active:
            theta_local = theta.copy()
            grads = []

            for step in range(n_local_steps):
                g = task.local_gradient(theta_local, i, n_samples=1)
                grads.append(g)

                if method == 'fedprox':
                    # FedProx: add proximal term
                    g += mu_prox * (theta_local - theta)

                theta_local -= lr * g

            delta = theta_local - theta
            local_deltas.append(delta)
            # Estimate variance from gradients
            if len(grads) > 1:
                var = np.mean([np.linalg.norm(g)**2 for g in grads])
            else:
                var = np.linalg.norm(grads[0])**2
            local_variances.append(max(var, 1e-6))

        # Aggregation
        local_variances = np.array(local_variances)

        if method in ('fedavg', 'fedprox'):
            # Uniform averaging
            weights = np.ones(n_active) / n_active
        elif method == 'ew_fedavg':
            # Evidence-weighted averaging
            V_min = local_variances.min()
            w = V_min / local_variances  # Keynesian weights
            weights = w / w.sum()
        elif method == 'scaffold':
            # SCAFFOLD-like: variance reduction through control variates
            # Simplified: use inverse-variance weighting
            w = 1.0 / local_variances
            weights = w / w.sum()

        # Aggregate
        delta_global = sum(w * d for w, d in zip(weights, local_deltas))
        theta += delta_global

        losses.append(task.global_loss(theta))

    return np.array(losses)


def experiment_A1_federated_convergence():
    """Compare FedAvg vs EW-FedAvg vs FedProx under different non-IID levels."""
    print("\n" + "=" * 60)
    print("APPLICATION A: Federated Learning — Convergence")
    print("=" * 60)

    noniid_levels = [0.1, 1.0, 5.0]
    n_runs = 20
    n_rounds = 300

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for li, noniid in enumerate(noniid_levels):
        ax = axes[li]

        for method, color_key, label in [
            ('fedavg', 'fedavg', 'FedAvg'),
            ('fedprox', 'fedprox', 'FedProx'),
            ('scaffold', 'scaffold', 'SCAFFOLD-like'),
            ('ew_fedavg', 'ew_fedavg', 'EW-FedAvg (ours)'),
        ]:
            all_losses = np.zeros((n_runs, n_rounds))
            for run in range(n_runs):
                np.random.seed(run * 100 + li)
                task = FederatedTask(d=10, n_clients=20,
                                     noniid_scale=noniid,
                                     noise_heterogeneity=2.0)
                all_losses[run] = run_federated(task, method, n_rounds=n_rounds,
                                                 lr=0.05, n_local_steps=5)

            mean_loss = np.convolve(all_losses.mean(0), np.ones(10)/10, 'valid')
            ax.semilogy(range(len(mean_loss)), mean_loss,
                        color=COLORS[color_key], label=label, linewidth=2)

        ax.set_xlabel('Communication round')
        ax.set_ylabel('Global loss')
        ax.set_title(f'Non-IID scale = {noniid}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Application A: Federated Learning\n'
                 'EW-FedAvg vs baselines under heterogeneous data',
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'appA_federated_convergence.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved appA_federated_convergence.png")


def experiment_A2_communication_efficiency():
    """Test communication efficiency: how many bits needed vs evidence quality."""
    print("\n" + "=" * 60)
    print("APPLICATION A: Communication Efficiency")
    print("=" * 60)

    n_runs = 15
    n_rounds = 200
    compression_levels = [1.0, 0.5, 0.2, 0.1, 0.05]  # fraction of gradient transmitted

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for method, color_key, label in [
        ('fedavg', 'fedavg', 'FedAvg'),
        ('ew_fedavg', 'ew_fedavg', 'EW-FedAvg'),
    ]:
        final_losses = []
        for comp in compression_levels:
            losses_runs = []
            for run in range(n_runs):
                np.random.seed(run * 200)
                task = FederatedTask(d=50, n_clients=20,
                                     noniid_scale=2.0,
                                     noise_heterogeneity=3.0)
                theta = np.zeros(task.d)

                for rnd in range(n_rounds):
                    local_deltas = []
                    local_vars = []
                    for i in range(task.n_clients):
                        theta_local = theta.copy()
                        grads = []
                        for _ in range(5):
                            g = task.local_gradient(theta_local, i)
                            grads.append(g)
                            theta_local -= 0.05 * g
                        delta = theta_local - theta

                        # Sparsify: keep top-k components (compression)
                        k = max(1, int(task.d * comp))
                        top_k = np.argsort(np.abs(delta))[-k:]
                        delta_sparse = np.zeros_like(delta)
                        delta_sparse[top_k] = delta[top_k]

                        local_deltas.append(delta_sparse)
                        var = np.mean([np.linalg.norm(g)**2 for g in grads])
                        local_vars.append(max(var, 1e-6))

                    local_vars = np.array(local_vars)
                    if method == 'ew_fedavg':
                        w = local_vars.min() / local_vars
                        weights = w / w.sum()
                    else:
                        weights = np.ones(task.n_clients) / task.n_clients

                    theta += sum(w * d for w, d in zip(weights, local_deltas))

                losses_runs.append(task.global_loss(theta))
            final_losses.append(np.mean(losses_runs))

        ax1.plot([c * 100 for c in compression_levels], final_losses,
                 color=COLORS[color_key], label=label, marker='o', linewidth=2)

    ax1.set_xlabel('Communication budget (% of gradient)')
    ax1.set_ylabel('Final global loss')
    ax1.set_title('Loss vs Communication Budget')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Self-knowledge bottleneck: loss vs evidence quality
    noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for method, color_key, label in [
        ('fedavg', 'fedavg', 'FedAvg'),
        ('ew_fedavg', 'ew_fedavg', 'EW-FedAvg'),
    ]:
        final_losses = []
        for noise_het in noise_levels:
            losses_runs = []
            for run in range(n_runs):
                np.random.seed(run * 300)
                task = FederatedTask(d=10, n_clients=20,
                                     noniid_scale=2.0,
                                     noise_heterogeneity=noise_het)
                loss = run_federated(task, method, n_rounds=200, lr=0.05)
                losses_runs.append(loss[-1])
            final_losses.append(np.mean(losses_runs))

        ax2.plot(noise_levels, final_losses, color=COLORS[color_key],
                 label=label, marker='s', linewidth=2)

    ax2.set_xlabel('Noise heterogeneity')
    ax2.set_ylabel('Final global loss')
    ax2.set_title('Self-Knowledge Bottleneck\n(higher noise = more "vibing" clients)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Application A: Communication Efficiency & Self-Knowledge',
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'appA_communication.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved appA_communication.png")


# ============================================================
# APPLICATION B: RLHF / LLM Alignment
# ============================================================

class RLHFTask:
    """Simulated RLHF setting.

    A "model" (parameter theta) is trained against K reward models,
    each measuring a different quality (helpfulness, harmlessness, honesty).

    Each reward model has:
      - A true reward function r_k(theta)
      - A noise level sigma_k (evidence quality)
      - A potential for reward hacking (the model can overfit to r_k)
    """
    def __init__(self, d, n_rewards, hack_vulnerability=0.0):
        self.d = d
        self.n_rewards = n_rewards
        # True optimal direction for each reward
        self.reward_directions = [
            np.random.randn(d) for _ in range(n_rewards)
        ]
        # Normalize
        self.reward_directions = [
            r / np.linalg.norm(r) for r in self.reward_directions
        ]
        # True global optimum: average of reward directions
        self.theta_star = np.mean(self.reward_directions, axis=0)
        self.theta_star /= np.linalg.norm(self.theta_star)

        # Reward noise levels (heterogeneous evidence)
        self.noise_scales = np.array([
            0.1 * (1 + 2 * k) for k in range(n_rewards)
        ])  # Later rewards noisier (e.g., honesty harder to measure)

        # Reward hacking vulnerability
        self.hack_vulnerability = hack_vulnerability
        # Hackable directions: high reward but not aligned with true objective
        self.hack_directions = [
            np.random.randn(d) for _ in range(n_rewards)
        ]
        self.hack_directions = [
            h / np.linalg.norm(h) for h in self.hack_directions
        ]

    def reward_signal(self, theta, reward_id):
        """Noisy reward from reward model k."""
        true_r = np.dot(theta, self.reward_directions[reward_id])
        # Reward hacking: model gets extra signal from hack direction
        hack_r = self.hack_vulnerability * np.dot(theta, self.hack_directions[reward_id])
        noise = self.noise_scales[reward_id] * np.random.randn()
        return true_r + hack_r + noise

    def reward_gradient(self, theta, reward_id):
        """Gradient of reward for model k."""
        true_g = self.reward_directions[reward_id]
        hack_g = self.hack_vulnerability * self.hack_directions[reward_id]
        noise = self.noise_scales[reward_id] * np.random.randn(self.d)
        return true_g + hack_g + noise

    def true_alignment(self, theta):
        """True alignment score (cosine similarity to true optimum)."""
        if np.linalg.norm(theta) < 1e-10:
            return 0.0
        return np.dot(theta, self.theta_star) / (np.linalg.norm(theta) * np.linalg.norm(self.theta_star))


def experiment_B1_reward_aggregation():
    """Compare reward aggregation strategies."""
    print("\n" + "=" * 60)
    print("APPLICATION B: RLHF — Evidence-Weighted Reward Aggregation")
    print("=" * 60)

    n_runs = 30
    n_steps = 500
    n_rewards = 4

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    hack_levels = [0.0, 0.5, 2.0]
    hack_labels = ['No reward hacking', 'Mild reward hacking', 'Strong reward hacking']

    for hi, (hack, hlabel) in enumerate(zip(hack_levels, hack_labels)):
        ax = axes[hi]

        for method, color_key, label in [
            ('uniform', 'rlhf_uniform', 'Uniform RLHF'),
            ('tuned', 'rlhf_tuned', 'Hand-tuned weights'),
            ('ew', 'rlhf_ew', 'EW-RLHF (ours)'),
        ]:
            all_alignment = np.zeros((n_runs, n_steps))

            for run in range(n_runs):
                np.random.seed(run * 500 + hi)
                task = RLHFTask(d=20, n_rewards=n_rewards,
                                 hack_vulnerability=hack)
                theta = np.zeros(task.d)
                V_est = np.ones(n_rewards)

                for step in range(n_steps):
                    lr = 0.1 / (step + 10) ** 0.5

                    # Get gradients from all reward models
                    grads = []
                    for k in range(n_rewards):
                        g = task.reward_gradient(theta, k)
                        grads.append(g)
                        V_est[k] = 0.9 * V_est[k] + 0.1 * np.linalg.norm(g)**2

                    # Aggregate
                    if method == 'uniform':
                        weights = np.ones(n_rewards) / n_rewards
                    elif method == 'tuned':
                        # Hand-tuned: inverse of known noise scale
                        weights = 1.0 / task.noise_scales
                        weights /= weights.sum()
                    elif method == 'ew':
                        V_min = max(V_est.min(), 1e-6)
                        w = V_min / np.maximum(V_est, 1e-6)
                        weights = w / w.sum()

                    g_agg = sum(w * g for w, g in zip(weights, grads))
                    theta += lr * g_agg

                    all_alignment[run, step] = task.true_alignment(theta)

            mean_align = np.convolve(all_alignment.mean(0), np.ones(20)/20, 'valid')
            ax.plot(range(len(mean_align)), mean_align,
                    color=COLORS[color_key], label=label, linewidth=2)

        ax.set_xlabel('Training step')
        ax.set_ylabel('True alignment (cosine sim)')
        ax.set_title(hlabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.0)

    fig.suptitle('Application B: RLHF — Evidence-Weighted Reward Aggregation\n'
                 '4 reward models with heterogeneous noise, varying reward hacking',
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'appB_rlhf_aggregation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved appB_rlhf_aggregation.png")


def experiment_B2_vibing_alignment():
    """Test the vibing problem: models that perform well but can't articulate.

    We simulate a model with varying "self-knowledge" — its ability to
    accurately report its own behavior. Low self-knowledge = vibing.
    """
    print("\n" + "=" * 60)
    print("APPLICATION B: RLHF — The Vibing Problem")
    print("=" * 60)

    n_runs = 30
    n_steps = 400
    selfknowledge_levels = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Exp B2a: alignment vs self-knowledge for debate-based alignment
    final_alignments_debate = []
    final_alignments_direct = []

    for sk in selfknowledge_levels:
        align_debate = []
        align_direct = []

        for run in range(n_runs):
            np.random.seed(run * 600)
            task = RLHFTask(d=15, n_rewards=3, hack_vulnerability=0.5)
            theta = np.zeros(task.d)

            for step in range(n_steps):
                lr = 0.1 / (step + 10) ** 0.5

                # Direct gradient (always available)
                g_direct = sum(task.reward_gradient(theta, k) for k in range(3)) / 3

                # Debate signal: model explains itself, but with self-knowledge noise
                # Self-knowledge = how well the model can articulate its policy
                g_debate = g_direct + sk * np.random.randn(task.d)

                # Use debate signal for alignment
                theta += lr * g_debate

            align_debate.append(task.true_alignment(theta))

            # Baseline: direct training without debate noise
            theta2 = np.zeros(task.d)
            for step in range(n_steps):
                lr = 0.1 / (step + 10) ** 0.5
                g = sum(task.reward_gradient(theta2, k) for k in range(3)) / 3
                theta2 += lr * g
            align_direct.append(task.true_alignment(theta2))

        final_alignments_debate.append(np.mean(align_debate))
        final_alignments_direct.append(np.mean(align_direct))

    ax1.plot(selfknowledge_levels, final_alignments_debate, 'o-',
             color=COLORS['rlhf_uniform'], label='Debate-based alignment', linewidth=2)
    ax1.axhline(y=np.mean(final_alignments_direct), color='gray',
                linestyle='--', alpha=0.7, label='Direct training (no debate)')
    ax1.set_xlabel(r'Self-knowledge noise $L_{\mathrm{self}}$')
    ax1.set_ylabel('Final true alignment')
    ax1.set_title('Alignment vs Self-Knowledge\n'
                   r'(higher $L_{\mathrm{self}}$ = more "vibing")')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Exp B2b: EW vs uniform under vibing
    for method, color_key, label in [
        ('uniform', 'rlhf_uniform', 'Uniform weights'),
        ('ew', 'rlhf_ew', 'Evidence-weighted'),
    ]:
        final_aligns = []
        for sk in selfknowledge_levels:
            aligns = []
            for run in range(n_runs):
                np.random.seed(run * 700)
                task = RLHFTask(d=15, n_rewards=3, hack_vulnerability=0.5)
                theta = np.zeros(task.d)
                V_est = np.ones(3)

                for step in range(n_steps):
                    lr = 0.1 / (step + 10) ** 0.5
                    grads = []
                    for k in range(3):
                        g = task.reward_gradient(theta, k)
                        # Add self-knowledge noise (vibing)
                        g += sk * np.random.randn(task.d)
                        grads.append(g)
                        V_est[k] = 0.9 * V_est[k] + 0.1 * np.linalg.norm(g)**2

                    if method == 'ew':
                        V_min = max(V_est.min(), 1e-6)
                        w = V_min / np.maximum(V_est, 1e-6)
                        weights = w / w.sum()
                    else:
                        weights = np.ones(3) / 3

                    g_agg = sum(w * g for w, g in zip(weights, grads))
                    theta += lr * g_agg

                aligns.append(task.true_alignment(theta))
            final_aligns.append(np.mean(aligns))

        ax2.plot(selfknowledge_levels, final_aligns, 'o-',
                 color=COLORS[color_key], label=label, linewidth=2)

    ax2.set_xlabel(r'Self-knowledge noise $L_{\mathrm{self}}$')
    ax2.set_ylabel('Final true alignment')
    ax2.set_title('EW-RLHF Robustness to Vibing')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    fig.suptitle('Application B: The Vibing Problem in Alignment\n'
                 'Models that perform well but can\'t articulate degrade alignment',
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'appB_vibing.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved appB_vibing.png")


# ============================================================
# APPLICATION C: Multi-Agent Debate
# ============================================================

class DebateGame:
    """Simulated debate between two agents before a judge.

    There is a ground truth claim (a binary vector of d facts).
    Proponent argues FOR, opponent argues AGAINST.
    Each round, debaters select arguments (facts to present).
    Judge accumulates evidence and decides.

    Key parameters:
      - d: number of possible arguments
      - k: number of truly relevant arguments
      - channel_capacity: how many arguments judge can process per round
      - self_knowledge: how well debaters know which arguments are strong
    """
    def __init__(self, d, k, channel_capacity=5, noise=0.1):
        self.d = d
        self.k = k
        self.channel_capacity = channel_capacity
        self.noise = noise

        # Ground truth: k relevant facts out of d
        self.relevant = np.zeros(d)
        self.relevant_idx = np.random.choice(d, k, replace=False)
        self.relevant[self.relevant_idx] = 1.0

        # Argument strengths: relevant arguments are strong, others are noise
        self.strengths = np.random.exponential(0.1, d)  # weak background
        self.strengths[self.relevant_idx] = np.random.exponential(1.0, k) + 1.0

    def debater_signal(self, debater_selfknowledge=1.0):
        """Debater's noisy estimate of argument strengths.
        Low self-knowledge = poor ability to distinguish strong from weak."""
        noise = (1.0 / max(debater_selfknowledge, 0.01)) * np.random.randn(self.d)
        return self.strengths + noise

    def judge_evaluate(self, presented_args, weights=None):
        """Judge evaluates presented arguments.
        Returns accuracy: how well the judge's belief matches ground truth."""
        if weights is None:
            weights = np.ones(len(presented_args)) / len(presented_args)

        # Judge receives arguments through capacity-limited channel
        n_process = min(len(presented_args), self.channel_capacity)
        # Judge processes top-weighted arguments
        if len(presented_args) > n_process:
            top_idx = np.argsort(weights)[-n_process:]
            presented_args = [presented_args[i] for i in top_idx]
            weights = weights[top_idx]
            weights /= weights.sum()

        # Judge's belief: weighted sum of presented argument strengths
        belief = np.zeros(self.d)
        for arg, w in zip(presented_args, weights):
            belief[arg] += w * (self.strengths[arg] + self.noise * np.random.randn())

        # Accuracy: correlation between belief and ground truth
        if np.linalg.norm(belief) < 1e-10:
            return 0.5
        return max(0, np.corrcoef(belief, self.relevant)[0, 1])


def experiment_C1_debate_accuracy():
    """Test debate accuracy vs channel capacity and self-knowledge."""
    print("\n" + "=" * 60)
    print("APPLICATION C: Multi-Agent Debate — Accuracy")
    print("=" * 60)

    n_runs = 50
    d = 50  # total possible arguments
    k = 5   # truly relevant
    n_rounds = 20

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # C1a: Accuracy vs channel capacity
    capacities = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
    for method, color_key, label in [
        ('uniform', 'debate_uniform', 'Uniform debate'),
        ('ew', 'debate_ew', 'EW-Debate'),
        ('sparse', 'debate_sparse', 'Sparse debate (L1)'),
    ]:
        accuracies = []
        for cap in capacities:
            accs = []
            for run in range(n_runs):
                np.random.seed(run * 800)
                game = DebateGame(d=d, k=k, channel_capacity=cap)
                debater_signal = game.debater_signal(debater_selfknowledge=2.0)

                # Select arguments based on method
                if method == 'sparse':
                    # L1: pick top-k by absolute signal strength
                    top_k = np.argsort(np.abs(debater_signal))[-k:]
                    args = list(top_k)
                elif method == 'ew':
                    # Evidence-weighted: weight by signal confidence
                    # Present more arguments but weight by strength
                    n_present = min(2 * k, d)
                    top_args = np.argsort(np.abs(debater_signal))[-n_present:]
                    args = list(top_args)
                else:
                    # Uniform: present random set of arguments
                    n_present = min(2 * k, d)
                    args = list(np.random.choice(d, n_present, replace=False))

                # Weight by signal strength
                if method == 'ew':
                    strengths = np.abs(debater_signal[args])
                    weights = strengths / strengths.sum()
                else:
                    weights = None

                acc = game.judge_evaluate(args, weights)
                accs.append(acc)
            accuracies.append(np.mean(accs))

        axes[0].plot(capacities, accuracies, 'o-', color=COLORS[color_key],
                     label=label, linewidth=2)

    axes[0].axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    axes[0].set_xlabel('Judge channel capacity $C$')
    axes[0].set_ylabel('Debate accuracy')
    axes[0].set_title('Accuracy vs Channel Capacity')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # C1b: Accuracy vs self-knowledge
    sk_levels = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    for method, color_key, label in [
        ('uniform', 'debate_uniform', 'Uniform debate'),
        ('ew', 'debate_ew', 'EW-Debate'),
    ]:
        accuracies = []
        for sk in sk_levels:
            accs = []
            for run in range(n_runs):
                np.random.seed(run * 900)
                game = DebateGame(d=d, k=k, channel_capacity=10)
                debater_signal = game.debater_signal(debater_selfknowledge=sk)

                n_present = min(2 * k, d)
                top_args = np.argsort(np.abs(debater_signal))[-n_present:]
                args = list(top_args)

                if method == 'ew':
                    strengths = np.abs(debater_signal[args])
                    weights = strengths / strengths.sum()
                else:
                    weights = None

                acc = game.judge_evaluate(args, weights)
                accs.append(acc)
            accuracies.append(np.mean(accs))

        axes[1].plot(sk_levels, accuracies, 'o-', color=COLORS[color_key],
                     label=label, linewidth=2)

    axes[1].set_xlabel('Debater self-knowledge')
    axes[1].set_ylabel('Debate accuracy')
    axes[1].set_title('Self-Knowledge Bottleneck\n(low = "vibing" debater)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    # C1c: Blessing of dimensionality — accuracy vs d for fixed k
    dims = [10, 20, 50, 100, 200, 500]
    for method, color_key, label in [
        ('uniform', 'debate_uniform', 'Uniform (O(d) args)'),
        ('sparse', 'debate_sparse', 'Sparse (O(k log d) args)'),
    ]:
        accuracies = []
        n_args_used = []
        for dim in dims:
            accs = []
            n_used = []
            for run in range(n_runs):
                np.random.seed(run * 1000)
                game = DebateGame(d=dim, k=k, channel_capacity=10)
                debater_signal = game.debater_signal(debater_selfknowledge=2.0)

                if method == 'sparse':
                    # O(k log d) arguments
                    n_present = min(int(k * np.log(dim / k + 1)) + k, dim)
                    top_args = np.argsort(np.abs(debater_signal))[-n_present:]
                    args = list(top_args)
                else:
                    # O(d) arguments (present everything)
                    n_present = min(dim, 3 * k)
                    args = list(np.random.choice(dim, n_present, replace=False))

                n_used.append(len(args))
                acc = game.judge_evaluate(args)
                accs.append(acc)

            accuracies.append(np.mean(accs))
            n_args_used.append(np.mean(n_used))

        axes[2].plot(dims, accuracies, 'o-', color=COLORS[color_key],
                     label=f'{label} (avg {n_args_used[-1]:.0f} args)', linewidth=2)

    axes[2].set_xlabel('Argument space dimension $d$')
    axes[2].set_ylabel('Debate accuracy')
    axes[2].set_title(f'Blessing of Dimensionality (k={k})\n'
                       f'Sparse debate scales logarithmically')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log')

    fig.suptitle('Application C: Multi-Agent Debate\n'
                 'Channel capacity, self-knowledge, and dimensionality',
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'appC_debate_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved appC_debate_accuracy.png")


def experiment_C2_coalition_debate():
    """N-agent debate with coalition formation."""
    print("\n" + "=" * 60)
    print("APPLICATION C: Coalition Debate")
    print("=" * 60)

    n_runs = 40
    d = 30
    k = 4
    n_agents_list = [2, 3, 4, 6, 8, 10]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # C2a: accuracy vs number of debaters
    for diversity, color_key, label in [
        ('homogeneous', 'debate_uniform', 'Homogeneous debaters'),
        ('diverse', 'debate_ew', 'Diverse debaters (varied sk)'),
        ('diverse_ew', 'debate_coalition', 'Diverse + EW coalition'),
    ]:
        accuracies = []
        for n_agents in n_agents_list:
            accs = []
            for run in range(n_runs):
                np.random.seed(run * 1100 + n_agents)
                game = DebateGame(d=d, k=k, channel_capacity=8)

                # Each debater has different self-knowledge
                if diversity == 'homogeneous':
                    sks = [2.0] * n_agents
                else:
                    sks = [0.5 + 3.0 * np.random.rand() for _ in range(n_agents)]

                # Each debater proposes arguments
                all_args = []
                all_weights = []
                debater_confidences = []

                for agent in range(n_agents):
                    signal = game.debater_signal(debater_selfknowledge=sks[agent])
                    n_present = min(k + 2, d)
                    top_args = np.argsort(np.abs(signal))[-n_present:]

                    for arg in top_args:
                        all_args.append(arg)
                        all_weights.append(np.abs(signal[arg]))
                        debater_confidences.append(sks[agent])

                # Deduplicate: keep best weight per argument
                arg_weights = {}
                arg_confidence = {}
                for arg, w, c in zip(all_args, all_weights, debater_confidences):
                    if arg not in arg_weights or w > arg_weights[arg]:
                        arg_weights[arg] = w
                        arg_confidence[arg] = c

                unique_args = list(arg_weights.keys())
                if diversity == 'diverse_ew':
                    # Weight by debater confidence (evidence weight)
                    weights = np.array([arg_weights[a] * arg_confidence[a]
                                        for a in unique_args])
                else:
                    weights = np.array([arg_weights[a] for a in unique_args])
                weights /= weights.sum()

                acc = game.judge_evaluate(unique_args, weights)
                accs.append(acc)

            accuracies.append(np.mean(accs))

        ax1.plot(n_agents_list, accuracies, 'o-', color=COLORS[color_key],
                 label=label, linewidth=2)

    ax1.set_xlabel('Number of debaters')
    ax1.set_ylabel('Debate accuracy')
    ax1.set_title('Coalition Debate: More Diverse = Better')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # C2b: HM/AM effect — varying debater heterogeneity
    het_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    for method, color_key, label in [
        ('uniform', 'debate_uniform', 'Uniform weighting'),
        ('ew', 'debate_ew', 'Evidence-weighted'),
    ]:
        accuracies = []
        hm_am_ratios = []
        for het in het_levels:
            accs = []
            ratios = []
            for run in range(n_runs):
                np.random.seed(run * 1200)
                game = DebateGame(d=d, k=k, channel_capacity=8)
                n_agents = 6
                sks = [2.0 + het * np.random.randn() for _ in range(n_agents)]
                sks = [max(sk, 0.1) for sk in sks]

                V = np.array([1.0 / sk for sk in sks])  # variance ~ 1/self-knowledge
                AM = V.mean()
                HM = len(V) / np.sum(1.0 / V)
                ratios.append(HM / AM)

                all_args = []
                all_weights = []
                for agent in range(n_agents):
                    signal = game.debater_signal(debater_selfknowledge=sks[agent])
                    top_args = np.argsort(np.abs(signal))[-(k+2):]
                    for arg in top_args:
                        all_args.append(arg)
                        if method == 'ew':
                            all_weights.append(np.abs(signal[arg]) * sks[agent])
                        else:
                            all_weights.append(np.abs(signal[arg]))

                arg_best = {}
                for arg, w in zip(all_args, all_weights):
                    if arg not in arg_best or w > arg_best[arg]:
                        arg_best[arg] = w
                unique_args = list(arg_best.keys())
                weights = np.array([arg_best[a] for a in unique_args])
                weights /= weights.sum()

                acc = game.judge_evaluate(unique_args, weights)
                accs.append(acc)

            accuracies.append(np.mean(accs))
            hm_am_ratios.append(np.mean(ratios))

        ax2.plot(het_levels, accuracies, 'o-', color=COLORS[color_key],
                 label=label, linewidth=2)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(het_levels, hm_am_ratios, 's--', color='gray',
                  alpha=0.5, label='HM/AM ratio')
    ax2_twin.set_ylabel('HM/AM ratio', color='gray')

    ax2.set_xlabel('Debater heterogeneity')
    ax2.set_ylabel('Debate accuracy')
    ax2.set_title('HM/AM Effect: EW Handles Heterogeneity')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Application C: Coalition Debate & HM/AM Effect',
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'appC_coalition_debate.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved appC_coalition_debate.png")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("APPLICATION EXPERIMENTS — Ω-Framework")
    print("=" * 60)

    t0 = time.time()

    # Application A: Federated Learning
    experiment_A1_federated_convergence()
    experiment_A2_communication_efficiency()

    # Application B: RLHF / Alignment
    experiment_B1_reward_aggregation()
    experiment_B2_vibing_alignment()

    # Application C: Multi-Agent Debate
    experiment_C1_debate_accuracy()
    experiment_C2_coalition_debate()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"ALL APPLICATION EXPERIMENTS COMPLETE in {elapsed:.0f}s")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"{'=' * 60}")
