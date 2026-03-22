"""
LOLA Basin of Attraction Experiment

Validates Theorem 6.3 (basin enlargement) from the dissertation.
Compares the basin of attraction of Standard PG vs LOLA-PG on
2-player matrix games by sweeping over initial conditions.

Author: Eugene Shcherbinin
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures" / "lola"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


# ============================================================
# Game Definitions (2x2 for simplex visualization)
# ============================================================

class MatrixGame2x2:
    """A 2-player 2-action matrix game."""
    def __init__(self, name, R1, R2, nash_p1, nash_p2):
        self.name = name
        self.R1 = np.array(R1, dtype=float)
        self.R2 = np.array(R2, dtype=float)
        self.nash_p1 = nash_p1  # Nash eq probability on action 0
        self.nash_p2 = nash_p2

    def policy_gradient(self, p1, p2):
        """Exact policy gradient.
        p1, p2 are scalars (probability of action 0).
        Returns scalar gradients dV1/dp1, dV2/dp2.
        """
        # V1(p1, p2) = p1*p2*R1[0,0] + p1*(1-p2)*R1[0,1] + (1-p1)*p2*R1[1,0] + (1-p1)*(1-p2)*R1[1,1]
        # dV1/dp1 = p2*(R1[0,0]-R1[1,0]) + (1-p2)*(R1[0,1]-R1[1,1])
        g1 = p2 * (self.R1[0,0] - self.R1[1,0]) + (1-p2) * (self.R1[0,1] - self.R1[1,1])
        g2 = p1 * (self.R2[0,0] - self.R2[0,1]) + (1-p1) * (self.R2[1,0] - self.R2[1,1])
        return g1, g2

    def opponent_shaping(self, p1, p2, eta=0.1):
        """LOLA opponent-shaping term for each player.
        OS_1 = dV1/dp2 * dp2'/dp1
        where p2' = p2 + eta * dV2/dp2 is player 2's anticipated update.
        dp2'/dp1 = eta * d(dV2/dp2)/dp1
        """
        # dV1/dp2 = p1*(R1[0,0]-R1[0,1]) + (1-p1)*(R1[1,0]-R1[1,1])
        dV1_dp2 = p1*(self.R1[0,0]-self.R1[0,1]) + (1-p1)*(self.R1[1,0]-self.R1[1,1])

        # dV2/dp1 = p2*(R2[0,0]-R2[0,1]) + ... wait, need to be careful
        # V2 = p1*p2*R2[0,0] + p1*(1-p2)*R2[0,1] + (1-p1)*p2*R2[1,0] + (1-p1)*(1-p2)*R2[1,1]
        # dV2/dp2 = p1*(R2[0,0]-R2[0,1]) + (1-p1)*(R2[1,0]-R2[1,1])
        dV2_dp2 = p1*(self.R2[0,0]-self.R2[0,1]) + (1-p1)*(self.R2[1,0]-self.R2[1,1])

        # d(dV2/dp2)/dp1 = (R2[0,0]-R2[0,1]) - (R2[1,0]-R2[1,1])
        d2V2_dp2dp1 = (self.R2[0,0]-self.R2[0,1]) - (self.R2[1,0]-self.R2[1,1])

        # dp2'/dp1 = eta * d2V2/dp2dp1
        dp2_dp1 = eta * d2V2_dp2dp1

        # OS_1 = dV1/dp2 * dp2'/dp1
        os1 = dV1_dp2 * dp2_dp1

        # Symmetrically for player 2
        dV2_dp1 = p2*(self.R2[0,0]-self.R2[1,0]) + (1-p2)*(self.R2[0,1]-self.R2[1,1])
        d2V1_dp1dp2 = (self.R1[0,0]-self.R1[1,0]) - (self.R1[0,1]-self.R1[1,1])
        dp1_dp2 = eta * d2V1_dp1dp2
        os2 = dV2_dp1 * dp1_dp2

        return os1, os2


MATCHING_PENNIES = MatrixGame2x2(
    "Matching Pennies",
    R1=[[1, -1], [-1, 1]],
    R2=[[-1, 1], [1, -1]],
    nash_p1=0.5, nash_p2=0.5
)

# A coordination game with interior Nash
BATTLE_OF_SEXES = MatrixGame2x2(
    "Battle of the Sexes",
    R1=[[3, 0], [0, 2]],
    R2=[[2, 0], [0, 3]],
    nash_p1=0.6, nash_p2=0.4
)


def clip_policy(p, eps=1e-4):
    """Clip scalar policy to (eps, 1-eps)."""
    return np.clip(p, eps, 1 - eps)


# ============================================================
# PG Algorithms (scalar policies for 2x2 games)
# ============================================================

def run_pg_scalar(game, p1_init, p2_init, n_episodes, gamma_base,
                  p_exp=0.75, lola_lambda=0.0, lola_eta=0.1,
                  noise_std=0.0):
    """Run PG or LOLA-PG for a 2x2 game with scalar policies."""
    p1, p2 = p1_init, p2_init
    trajectory = [(p1, p2)]

    for n in range(1, n_episodes + 1):
        gamma_n = gamma_base / (n + 10) ** p_exp

        g1, g2 = game.policy_gradient(p1, p2)

        # Add LOLA term
        if lola_lambda > 0:
            os1, os2 = game.opponent_shaping(p1, p2, eta=lola_eta)
            g1 += lola_lambda * os1
            g2 += lola_lambda * os2

        # Add noise
        if noise_std > 0:
            g1 += np.random.randn() * noise_std
            g2 += np.random.randn() * noise_std

        p1 = clip_policy(p1 + gamma_n * g1)
        p2 = clip_policy(p2 + gamma_n * g2)
        trajectory.append((p1, p2))

    return np.array(trajectory)


def check_convergence(trajectory, nash_p1, nash_p2, threshold=0.05):
    """Check if trajectory converged to Nash."""
    p1_final, p2_final = trajectory[-1]
    dist = (p1_final - nash_p1)**2 + (p2_final - nash_p2)**2
    return dist < threshold**2


# ============================================================
# Experiment: Basin of attraction mapping
# ============================================================

def experiment_basin(game, n_grid=30, n_episodes=2000, n_runs=10):
    """Map basin of attraction for Standard PG vs LOLA-PG."""
    print(f"\n{'='*60}")
    print(f"Basin of Attraction: {game.name}")
    print(f"Nash equilibrium: ({game.nash_p1}, {game.nash_p2})")
    print(f"{'='*60}")

    p1_grid = np.linspace(0.05, 0.95, n_grid)
    p2_grid = np.linspace(0.05, 0.95, n_grid)

    std_basin = np.zeros((n_grid, n_grid))
    lola_basin = np.zeros((n_grid, n_grid))

    total = n_grid * n_grid
    count = 0

    for i, p1_init in enumerate(p1_grid):
        for j, p2_init in enumerate(p2_grid):
            std_converged = 0
            lola_converged = 0

            for run in range(n_runs):
                np.random.seed(run * 10000 + i * 100 + j)

                # Standard PG
                traj_std = run_pg_scalar(
                    game, p1_init, p2_init, n_episodes,
                    gamma_base=0.3, p_exp=0.7, noise_std=0.3
                )
                if check_convergence(traj_std, game.nash_p1, game.nash_p2, threshold=0.08):
                    std_converged += 1

                np.random.seed(run * 10000 + i * 100 + j)

                # LOLA-PG
                traj_lola = run_pg_scalar(
                    game, p1_init, p2_init, n_episodes,
                    gamma_base=0.3, p_exp=0.7, lola_lambda=0.5,
                    lola_eta=0.1, noise_std=0.3
                )
                if check_convergence(traj_lola, game.nash_p1, game.nash_p2, threshold=0.08):
                    lola_converged += 1

            std_basin[j, i] = std_converged / n_runs
            lola_basin[j, i] = lola_converged / n_runs

            count += 1
            if count % 100 == 0:
                print(f"  Progress: {count}/{total}")

    # Statistics
    std_area = (std_basin > 0.5).mean()
    lola_area = (lola_basin > 0.5).mean()
    print(f"\nBasin area (>50% convergence):")
    print(f"  Standard PG: {std_area:.1%}")
    print(f"  LOLA-PG:     {lola_area:.1%}")
    print(f"  Enlargement: {lola_area/max(std_area, 1e-6):.2f}x")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cmap = plt.cm.RdYlGn

    im1 = axes[0].imshow(std_basin, extent=[0.05, 0.95, 0.05, 0.95],
                          origin='lower', cmap=cmap, vmin=0, vmax=1, aspect='auto')
    axes[0].plot(game.nash_p1, game.nash_p2, 'w*', markersize=15, markeredgecolor='k')
    axes[0].set_title('Standard PG', fontsize=13)
    axes[0].set_xlabel(r'$p_1$ (Player 1)', fontsize=11)
    axes[0].set_ylabel(r'$p_2$ (Player 2)', fontsize=11)

    im2 = axes[1].imshow(lola_basin, extent=[0.05, 0.95, 0.05, 0.95],
                          origin='lower', cmap=cmap, vmin=0, vmax=1, aspect='auto')
    axes[1].plot(game.nash_p1, game.nash_p2, 'w*', markersize=15, markeredgecolor='k')
    axes[1].set_title('LOLA-PG', fontsize=13)
    axes[1].set_xlabel(r'$p_1$ (Player 1)', fontsize=11)

    # Difference: LOLA - Standard
    diff = lola_basin - std_basin
    im3 = axes[2].imshow(diff, extent=[0.05, 0.95, 0.05, 0.95],
                          origin='lower', cmap='RdBu', vmin=-0.5, vmax=0.5, aspect='auto')
    axes[2].plot(game.nash_p1, game.nash_p2, 'k*', markersize=15)
    axes[2].set_title('LOLA - Standard (green = LOLA better)', fontsize=13)
    axes[2].set_xlabel(r'$p_1$ (Player 1)', fontsize=11)

    fig.colorbar(im1, ax=axes[0], label='Convergence rate')
    fig.colorbar(im2, ax=axes[1], label='Convergence rate')
    fig.colorbar(im3, ax=axes[2], label='Difference')

    fig.suptitle(f'Basin of Attraction: {game.name}', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f'basin_{game.name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: basin plot")

    return std_basin, lola_basin


# ============================================================
# Experiment: Trajectory visualization
# ============================================================

def experiment_trajectories(game, n_trajectories=8, n_episodes=500):
    """Visualize sample trajectories for Standard PG vs LOLA-PG."""
    print(f"\n{'='*60}")
    print(f"Trajectory Visualization: {game.name}")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (method, lola_lam, title) in zip(
        axes,
        [(0, 0.0, 'Standard PG'), (1, 0.5, 'LOLA-PG')]
    ):
        for t in range(n_trajectories):
            np.random.seed(t * 777)
            p1_init = np.random.uniform(0.1, 0.9)
            p2_init = np.random.uniform(0.1, 0.9)

            traj = run_pg_scalar(
                game, p1_init, p2_init, n_episodes,
                gamma_base=0.3, p_exp=0.7,
                lola_lambda=lola_lam, lola_eta=0.1, noise_std=0.2
            )

            converged = check_convergence(traj, game.nash_p1, game.nash_p2, threshold=0.08)
            color = '#2ecc71' if converged else '#e74c3c'
            alpha = 0.7 if converged else 0.4

            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=0.8)
            ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=4)

        ax.plot(game.nash_p1, game.nash_p2, 'k*', markersize=15, zorder=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r'$p_1$', fontsize=11)
        ax.set_ylabel(r'$p_2$', fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'Trajectories: {game.name} (green=converged, red=diverged)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f'trajectories_{game.name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: trajectories plot")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("LOLA Basin of Attraction Experiments")
    print("=" * 60)

    for game in [MATCHING_PENNIES, BATTLE_OF_SEXES]:
        experiment_basin(game, n_grid=25, n_episodes=1500, n_runs=8)
        experiment_trajectories(game, n_trajectories=12, n_episodes=800)

    print("\nAll experiments complete. Figures saved to:", FIGURES_DIR)
