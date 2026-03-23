"""
Homotopy Method + Spectral Analysis of the Best-Response Operator

Three experiments:

1. HOMOTOPY: Track QRE as τ decreases from ∞ to 0.
   At high τ: unique fixed point (Banach contraction).
   As τ decreases: bifurcation points where one QRE splits into multiple NE.
   Connects to catastrophe theory.

2. SPECTRAL ANALYSIS: Eigenvalues of the Jacobian of BR_τ at fixed points.
   - |λ_max| < 1 → stable (attracting) fixed point
   - |λ_max| = 1 → bifurcation (marginal stability)
   - |λ_max| > 1 → unstable (repelling) fixed point
   The spectral radius vs τ curve shows when stability is lost.

3. META-LEARNING FOR IPD COOPERATION: Phase -1 that optimizes over
   the trajectory of Ω-PG updates (simplified Meta-MAPG).
   Instead of optimizing per-episode reward, optimize the CUMULATIVE
   reward over K adaptation episodes. This enables TFT-like strategies.

Author: Eugene Shcherbinin
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from fixed_point_ne import Game, project_simplex, find_fixed_point
from iterated_games import IteratedGame, reinforce_gradient

FIGURES_DIR = Path(__file__).parent / "figures" / "advanced"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 12, 'legend.fontsize': 10, 'figure.dpi': 150,
})


# ============================================================
# Experiment 1: HOMOTOPY — Tracking QRE Bifurcations
# ============================================================

def softmax_br(game, p2, tau):
    """Softmax best response for player 1."""
    payoffs = game.R1 @ p2
    logits = payoffs / max(tau, 1e-8)
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()

def softmax_br2(game, p1, tau):
    """Softmax best response for player 2."""
    payoffs = game.R2.T @ p1
    logits = payoffs / max(tau, 1e-8)
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()

def find_qre(game, tau, n_restarts=20, max_iter=500, tol=1e-8):
    """Find all QRE at temperature tau via random restarts."""
    found = []
    for _ in range(n_restarts):
        p1 = np.random.dirichlet(np.ones(game.n1))
        p2 = np.random.dirichlet(np.ones(game.n2))
        alpha = 0.3
        for _ in range(max_iter):
            br1 = softmax_br(game, p2, tau)
            br2 = softmax_br2(game, p1, tau)
            p1_new = (1 - alpha) * p1 + alpha * br1
            p2_new = (1 - alpha) * p2 + alpha * br2
            if np.max(np.abs(p1_new - p1)) + np.max(np.abs(p2_new - p2)) < tol:
                p1, p2 = p1_new, p2_new
                break
            p1, p2 = p1_new, p2_new

        # Check if genuinely a QRE
        residual = np.sum((softmax_br(game, p2, tau) - p1)**2) + \
                   np.sum((softmax_br2(game, p1, tau) - p2)**2)
        if residual < 1e-6:
            # Check if novel
            is_new = True
            for f1, f2 in found:
                if np.max(np.abs(f1 - p1)) + np.max(np.abs(f2 - p2)) < 0.02:
                    is_new = False
                    break
            if is_new:
                found.append((p1.copy(), p2.copy()))
    return found


def experiment_homotopy():
    """
    Track the QRE correspondence as τ → 0.
    At bifurcation points, one QRE splits into multiple.
    """
    print("\n" + "="*70)
    print("ADVANCED 1: QRE Homotopy — Bifurcation Tracking")
    print("="*70)

    games = [
        Game("Stag Hunt", np.array([[4, 0], [3, 3]]), np.array([[4, 3], [0, 3]])),
        Game("Battle of Sexes", np.array([[3, 0], [0, 1]]), np.array([[1, 0], [0, 3]])),
        Game("Chicken", np.array([[0, 4], [1, 2]]), np.array([[0, 1], [4, 2]])),
    ]

    tau_values = np.concatenate([
        np.linspace(3.0, 0.5, 30),
        np.linspace(0.5, 0.05, 40),
        np.linspace(0.05, 0.005, 20),
    ])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for g_idx, game in enumerate(games):
        ax_top = axes[0, g_idx]
        ax_bot = axes[1, g_idx]

        all_p1 = []  # (tau, p1[0]) pairs
        all_welfare = []  # (tau, welfare) pairs

        for tau in tau_values:
            qres = find_qre(game, tau, n_restarts=15)
            for p1, p2 in qres:
                all_p1.append((tau, p1[0]))
                v1, v2 = game.payoffs(p1, p2)
                all_welfare.append((tau, v1 + v2))

        # Plot bifurcation diagram
        if all_p1:
            taus, p1s = zip(*all_p1)
            ax_top.scatter(taus, p1s, s=3, c='#2c3e50', alpha=0.7)
        ax_top.set_xlabel('Temperature τ')
        ax_top.set_ylabel('P(action 0) — Player 1')
        ax_top.set_title(f'{game.name}: Bifurcation Diagram')
        ax_top.set_xlim(max(tau_values), min(tau_values))
        ax_top.axvline(x=np.max(np.abs(game.R1)), color='red', linestyle='--',
                       alpha=0.3, label=f'τ = M = {np.max(np.abs(game.R1)):.1f}')
        ax_top.legend(fontsize=8)
        ax_top.grid(alpha=0.3)

        # Welfare diagram
        if all_welfare:
            taus_w, ws = zip(*all_welfare)
            ax_bot.scatter(taus_w, ws, s=3, c='#2ecc71', alpha=0.7)
        ax_bot.set_xlabel('Temperature τ')
        ax_bot.set_ylabel('Social Welfare')
        ax_bot.set_title(f'{game.name}: Welfare vs τ')
        ax_bot.set_xlim(max(tau_values), min(tau_values))
        ax_bot.grid(alpha=0.3)

        # Count NE at different τ
        n_qre_high = len(find_qre(game, 2.0, n_restarts=10))
        n_qre_mid = len(find_qre(game, 0.1, n_restarts=15))
        n_qre_low = len(find_qre(game, 0.01, n_restarts=20))
        print(f"  {game.name}: QRE count τ=2.0:{n_qre_high}, τ=0.1:{n_qre_mid}, τ=0.01:{n_qre_low}")

    fig.suptitle('QRE Homotopy: Tracking Equilibrium Bifurcations as τ → 0\n'
                 '(unique QRE at high τ splits into multiple NE at low τ)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'homotopy_bifurcation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'homotopy_bifurcation.png'}")


# ============================================================
# Experiment 2: SPECTRAL ANALYSIS of BR Jacobian
# ============================================================

def br_jacobian(game, p1, p2, tau, eps=1e-5):
    """
    Numerical Jacobian of the joint BR map at (p1, p2).
    BR: R^(n1+n2) → R^(n1+n2)
    """
    n1, n2 = game.n1, game.n2
    n = n1 + n2
    J = np.zeros((n, n))

    x0 = np.concatenate([p1, p2])
    br0_1 = softmax_br(game, p2, tau)
    br0_2 = softmax_br2(game, p1, tau)
    f0 = np.concatenate([br0_1, br0_2])

    for j in range(n):
        x_plus = x0.copy()
        x_plus[j] += eps
        # Re-normalize to simplex
        if j < n1:
            p1_plus = project_simplex(x_plus[:n1])
            p2_plus = x_plus[n1:]
        else:
            p1_plus = x_plus[:n1]
            p2_plus = project_simplex(x_plus[n1:])

        br1_plus = softmax_br(game, p2_plus, tau)
        br2_plus = softmax_br2(game, p1_plus, tau)
        f_plus = np.concatenate([br1_plus, br2_plus])

        J[:, j] = (f_plus - f0) / eps

    return J


def experiment_spectral():
    """
    Spectral analysis of the BR Jacobian at QRE/NE.
    Track eigenvalues as τ → 0.
    """
    print("\n" + "="*70)
    print("ADVANCED 2: Spectral Analysis of BR Jacobian")
    print("="*70)

    games = [
        Game("Matching Pennies", np.array([[1, -1], [-1, 1]]), np.array([[-1, 1], [1, -1]])),
        Game("Stag Hunt", np.array([[4, 0], [3, 3]]), np.array([[4, 3], [0, 3]])),
        Game("Battle of Sexes", np.array([[3, 0], [0, 1]]), np.array([[1, 0], [0, 3]])),
    ]

    tau_values = np.linspace(0.05, 3.0, 50)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for g_idx, game in enumerate(games):
        ax_top = axes[0, g_idx]
        ax_bot = axes[1, g_idx]

        spectral_radii = []
        max_eigenvalues = []
        all_eig_taus = []
        all_eig_vals = []

        for tau in tau_values:
            qres = find_qre(game, tau, n_restarts=10)
            if not qres:
                spectral_radii.append(np.nan)
                max_eigenvalues.append(np.nan)
                continue

            # Use first QRE
            p1, p2 = qres[0]
            J = br_jacobian(game, p1, p2, tau)
            eigvals = np.linalg.eigvals(J)
            sr = np.max(np.abs(eigvals))
            spectral_radii.append(sr)
            max_eigenvalues.append(np.max(eigvals.real))

            for ev in eigvals:
                all_eig_taus.append(tau)
                all_eig_vals.append(np.abs(ev))

        # Top: spectral radius vs tau
        ax_top.plot(tau_values, spectral_radii, 'ko-', markersize=3, linewidth=1.5)
        ax_top.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='ρ = 1 (stability boundary)')
        ax_top.set_xlabel('Temperature τ')
        ax_top.set_ylabel('Spectral Radius ρ(J)')
        ax_top.set_title(f'{game.name}')
        ax_top.legend(fontsize=8)
        ax_top.grid(alpha=0.3)

        # Bottom: all eigenvalue magnitudes
        if all_eig_taus:
            ax_bot.scatter(all_eig_taus, all_eig_vals, s=5, alpha=0.5, c='#3498db')
            ax_bot.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax_bot.set_xlabel('Temperature τ')
        ax_bot.set_ylabel('|λ| (eigenvalue magnitude)')
        ax_bot.set_title(f'Eigenvalue Spectrum')
        ax_bot.grid(alpha=0.3)

        print(f"  {game.name}: ρ at τ=0.1: {spectral_radii[np.argmin(np.abs(tau_values-0.1))]:.3f}, "
              f"ρ at τ=2.0: {spectral_radii[np.argmin(np.abs(tau_values-2.0))]:.3f}")

    fig.suptitle('Spectral Analysis of BR Jacobian: Stability vs Temperature\n'
                 '(ρ < 1 = stable/attracting, ρ > 1 = unstable/repelling)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'spectral_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'spectral_analysis.png'}")


# ============================================================
# Experiment 3: META-LEARNING for IPD Cooperation
# ============================================================

def experiment_meta_learning_ipd():
    """
    Meta-learning (Phase -1) for the Iterated Prisoner's Dilemma.

    Instead of optimizing per-episode reward (which converges to defection),
    optimize the CUMULATIVE reward over K adaptation episodes.

    Simplified Meta-MAPG: the agent's initial policy θ_0 is optimized so
    that the K-step adaptation trajectory (θ_0 → θ_1 → ... → θ_K)
    produces high total reward.

    The key insight: if I start near TFT and my opponent adapts to cooperate,
    my K-step trajectory reward is higher than if I start at defection.
    """
    print("\n" + "="*70)
    print("ADVANCED 3: Meta-Learning for IPD Cooperation")
    print("="*70)

    game = IteratedGame(
        "IPD",
        R1=np.array([[3, 0], [4, 1]]),  # CC=3, CD=0, DC=4, DD=1
        R2=np.array([[3, 4], [0, 1]]),
        gamma=0.96
    )

    n_meta_steps = 50       # meta-optimization steps
    K = 10                   # inner adaptation episodes per meta-step
    n_inner_episodes = 30    # episodes per inner adaptation step
    meta_lr = 0.05
    inner_lr = 0.2
    n_runs = 8

    # Opponents to meta-train against
    from iterated_games import (make_tit_for_tat, make_always_cooperate,
                                 make_always_defect, make_pavlov)

    opponents = [
        ('TFT', make_tit_for_tat(game)),
        ('Always-C', make_always_cooperate(game)),
        ('Pavlov', make_pavlov(game)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Compare: per-episode PG vs meta-PG
    methods = {
        'Per-Episode PG': False,
        'Meta-PG (K=10)': True,
    }

    for opp_idx, (opp_name, opp_logits) in enumerate(opponents):
        for m_name, use_meta in methods.items():
            all_coop_trajectories = []
            all_payoff_trajectories = []

            for run in range(n_runs):
                np.random.seed(run * 333 + opp_idx)

                # Initialize agent policy
                logits = np.random.randn(game.n_states, game.n_actions_1) * 0.1

                coop_history = []
                payoff_history = []

                if use_meta:
                    # Meta-learning: optimize θ_0 for K-step trajectory reward
                    for meta_step in range(n_meta_steps):
                        # Inner loop: K steps of adaptation
                        inner_logits = logits.copy()
                        trajectory_reward = 0.0
                        trajectory_coop = 0.0

                        for k in range(K):
                            # Play n_inner_episodes
                            total_r = 0.0
                            total_coop = 0
                            total_actions = 0
                            grad = np.zeros_like(inner_logits)

                            for _ in range(3):
                                r1, r2, traj = game.play_episode(inner_logits, opp_logits)
                                total_r += r1
                                grad += reinforce_gradient(game, inner_logits, 0, traj)
                                for s, a1, a2, _, _ in traj:
                                    total_coop += (a1 == 0)
                                    total_actions += 1

                            grad /= 3
                            total_r /= 3
                            trajectory_reward += total_r
                            trajectory_coop += total_coop / max(total_actions, 1)

                            # Inner update
                            inner_logits = inner_logits + inner_lr * grad / (1 + k / 5)

                        trajectory_coop /= K
                        coop_history.append(trajectory_coop)
                        payoff_history.append(trajectory_reward / K)

                        # Meta-gradient: finite difference on θ_0
                        meta_grad = np.zeros_like(logits)
                        eps = 0.1
                        for i in range(game.n_states):
                            for j in range(game.n_actions_1):
                                logits_plus = logits.copy()
                                logits_plus[i, j] += eps

                                inner_plus = logits_plus.copy()
                                reward_plus = 0.0
                                for k in range(K):
                                    r_sum = 0.0
                                    g = np.zeros_like(inner_plus)
                                    for _ in range(2):
                                        r1, _, traj = game.play_episode(inner_plus, opp_logits)
                                        r_sum += r1
                                        g += reinforce_gradient(game, inner_plus, 0, traj)
                                    g /= 2
                                    r_sum /= 2
                                    reward_plus += r_sum
                                    inner_plus = inner_plus + inner_lr * g / (1 + k / 5)

                                meta_grad[i, j] = (reward_plus - trajectory_reward) / eps

                        logits = logits + meta_lr * meta_grad / (1 + meta_step / 20)

                else:
                    # Standard per-episode PG
                    for ep in range(n_meta_steps * K):
                        lr = inner_lr / (1 + ep / 100)
                        total_r = 0.0
                        total_coop = 0
                        total_actions = 0
                        grad = np.zeros_like(logits)

                        for _ in range(3):
                            r1, r2, traj = game.play_episode(logits, opp_logits)
                            total_r += r1
                            grad += reinforce_gradient(game, logits, 0, traj)
                            for s, a1, a2, _, _ in traj:
                                total_coop += (a1 == 0)
                                total_actions += 1

                        grad /= 3
                        total_r /= 3
                        logits = logits + lr * grad

                        if ep % K == 0:
                            coop_history.append(total_coop / max(total_actions, 1))
                            payoff_history.append(total_r)

                all_coop_trajectories.append(coop_history)
                all_payoff_trajectories.append(payoff_history)

            # Plot cooperation
            ax = axes[0, opp_idx]
            coop = np.array(all_coop_trajectories).mean(axis=0)
            color = '#e74c3c' if not use_meta else '#2ecc71'
            n_pts = min(len(coop), n_meta_steps)
            ax.plot(range(n_pts), coop[:n_pts], label=m_name,
                    color=color, linewidth=1.5)

            # Plot payoff
            ax = axes[1, opp_idx]
            payoff = np.array(all_payoff_trajectories).mean(axis=0)
            ax.plot(range(n_pts), payoff[:n_pts], label=m_name,
                    color=color, linewidth=1.5)

        axes[0, opp_idx].set_title(f'vs {opp_name}')
        axes[0, opp_idx].set_ylabel('Cooperation Rate')
        axes[0, opp_idx].set_ylim(-0.05, 1.05)
        axes[0, opp_idx].legend(fontsize=9)
        axes[0, opp_idx].grid(alpha=0.3)

        axes[1, opp_idx].set_xlabel('Meta-Step')
        axes[1, opp_idx].set_ylabel('Avg Payoff')
        axes[1, opp_idx].legend(fontsize=9)
        axes[1, opp_idx].grid(alpha=0.3)

    # Final policy analysis
    print("\n  Final cooperation rates (Meta-PG vs Per-Episode PG):")
    for opp_idx, (opp_name, opp_logits) in enumerate(opponents):
        for use_meta in [False, True]:
            np.random.seed(42 + opp_idx)
            logits = np.random.randn(game.n_states, game.n_actions_1) * 0.1
            # Quick run
            for step in range(20 if use_meta else 200):
                grad = np.zeros_like(logits)
                for _ in range(3):
                    _, _, traj = game.play_episode(logits, opp_logits)
                    grad += reinforce_gradient(game, logits, 0, traj)
                grad /= 3
                logits += 0.2 * grad
            pi = game.softmax(logits)
            coop_rate = pi[:, 0].mean()
            name = "Meta-PG" if use_meta else "PG"
            print(f"    {opp_name} ({name}): coop_rate = {coop_rate:.2%}")

    fig.suptitle('Meta-Learning for IPD: Can Trajectory Optimization\n'
                 'Sustain Cooperation?',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'meta_learning_ipd.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'meta_learning_ipd.png'}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Advanced Experiments: Homotopy, Spectral, Meta-Learning")
    print("=" * 70)

    experiment_homotopy()
    experiment_spectral()
    experiment_meta_learning_ipd()

    print("\n" + "="*70)
    print("All advanced experiments complete.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*70)
