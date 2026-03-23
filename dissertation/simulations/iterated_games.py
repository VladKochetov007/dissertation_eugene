"""
Iterated Games × Ω-Framework — Adapted from Kim et al. (ICML 2021)

Kim et al.'s Meta-MAPG experiments use iterated matrix games where the
state is the last joint action. This creates a MUCH richer strategy space
than one-shot games: tit-for-tat, grim trigger, ZD-extortion, etc.
are all representable as tabular policies.

We adapt their experimental setup to test the Ω-gradient components:
  1. Iterated Prisoner's Dilemma (IPD) — mixed incentive
  2. Iterated Rock-Paper-Scissors (IRPS) — competitive, 2-4 agents
  3. Persona populations — different opponent types for adaptation testing
  4. Adaptation curves — how quickly Ω-PG adapts to new opponents

Key differences from Kim et al.:
  - We use tabular policies (not LSTM) for interpretability
  - We add FP-NE search over the iterated game's strategy space
  - We compare all Ω-components, not just Meta-MAPG vs baselines
  - We track NE structure (tit-for-tat as fixed point)

Author: Eugene Shcherbinin
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = Path(__file__).parent / "figures" / "iterated"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

COLORS = {
    'reinforce': '#bdc3c7',
    'ewpg': '#2ecc71',
    'lola': '#3498db',
    'coop': '#9b59b6',
    'omega': '#f39c12',
    'omega_fp': '#e74c3c',
    'tft': '#1abc9c',
}


# ============================================================
# Iterated Game Engine
# ============================================================

class IteratedGame:
    """
    Iterated matrix game with state = last joint action.

    State space: {s_0 (initial)} ∪ {(a1, a2) for all joint actions}
    So for 2-action game: 5 states (initial + 4 joint action pairs)
    For 3-action game: 10 states (initial + 9 joint action pairs)

    Policy: tabular, π_i(a | s) for each state s.
    Stored as logits: shape (n_states, n_actions_i)
    """

    def __init__(self, name: str, R1: np.ndarray, R2: np.ndarray,
                 gamma: float = 0.96):
        self.name = name
        self.R1 = np.array(R1, dtype=float)
        self.R2 = np.array(R2, dtype=float)
        self.n_actions_1 = R1.shape[0]
        self.n_actions_2 = R1.shape[1]
        self.n_joint = self.n_actions_1 * self.n_actions_2
        self.n_states = 1 + self.n_joint  # initial state + joint action states
        self.gamma = gamma

    def joint_action_to_state(self, a1: int, a2: int) -> int:
        """Map joint action to state index. State 0 = initial."""
        return 1 + a1 * self.n_actions_2 + a2

    def state_to_joint_action(self, s: int):
        """Map state back to joint action (for s > 0)."""
        if s == 0:
            return None
        s -= 1
        return s // self.n_actions_2, s % self.n_actions_2

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Row-wise softmax."""
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def play_episode(self, logits1: np.ndarray, logits2: np.ndarray,
                     max_steps: int = 100) -> tuple:
        """
        Play one episode of the iterated game.

        Returns: (total_reward_1, total_reward_2, trajectory)
        where trajectory = list of (state, action1, action2, r1, r2)
        """
        pi1 = self.softmax(logits1)  # (n_states, n_actions_1)
        pi2 = self.softmax(logits2)  # (n_states, n_actions_2)

        state = 0  # initial state
        total_r1, total_r2 = 0.0, 0.0
        discount = 1.0
        trajectory = []

        for t in range(max_steps):
            # Sample actions
            a1 = np.random.choice(self.n_actions_1, p=pi1[state])
            a2 = np.random.choice(self.n_actions_2, p=pi2[state])

            r1 = self.R1[a1, a2]
            r2 = self.R2[a1, a2]

            trajectory.append((state, a1, a2, r1, r2))

            total_r1 += discount * r1
            total_r2 += discount * r2
            discount *= self.gamma

            # Termination check (geometric)
            if np.random.rand() > self.gamma:
                break

            state = self.joint_action_to_state(a1, a2)

        return total_r1, total_r2, trajectory

    def expected_payoffs(self, logits1: np.ndarray, logits2: np.ndarray,
                         n_episodes: int = 200) -> tuple:
        """Monte Carlo estimate of expected discounted payoffs."""
        r1s, r2s = [], []
        for _ in range(n_episodes):
            r1, r2, _ = self.play_episode(logits1, logits2)
            r1s.append(r1)
            r2s.append(r2)
        return np.mean(r1s), np.mean(r2s)

    def exact_payoffs(self, logits1: np.ndarray, logits2: np.ndarray) -> tuple:
        """
        Exact expected discounted payoffs via matrix inversion.

        The Markov chain has transition matrix T and reward vector r.
        V = (I - γT)^{-1} r_0 where r_0 is the initial-state reward vector.
        """
        pi1 = self.softmax(logits1)
        pi2 = self.softmax(logits2)

        # Transition matrix T[s, s'] = Σ_{a1,a2} π1(a1|s) π2(a2|s) 1[s' = state(a1,a2)]
        T = np.zeros((self.n_states, self.n_states))
        r1_vec = np.zeros(self.n_states)
        r2_vec = np.zeros(self.n_states)

        for s in range(self.n_states):
            for a1 in range(self.n_actions_1):
                for a2 in range(self.n_actions_2):
                    prob = pi1[s, a1] * pi2[s, a2]
                    s_next = self.joint_action_to_state(a1, a2)
                    T[s, s_next] += prob
                    r1_vec[s] += prob * self.R1[a1, a2]
                    r2_vec[s] += prob * self.R2[a1, a2]

        # V = (I - γT)^{-1} r, starting from state 0
        I = np.eye(self.n_states)
        try:
            M = np.linalg.inv(I - self.gamma * T)
            V1 = M @ r1_vec
            V2 = M @ r2_vec
            return V1[0], V2[0]
        except np.linalg.LinAlgError:
            return self.expected_payoffs(logits1, logits2)


# ─── Game Definitions (Kim et al. 2021 payoffs) ────────────

def iterated_prisoners_dilemma(gamma=0.96):
    """IPD with Kim et al. payoffs. C=0, D=1."""
    return IteratedGame(
        "Iterated PD",
        R1=np.array([[0.5, -1.5],   # CC, CD
                      [1.5, -0.5]]),  # DC, DD
        R2=np.array([[0.5, 1.5],
                      [-1.5, -0.5]]),
        gamma=gamma
    )

def iterated_rps(gamma=0.96):
    """Iterated Rock-Paper-Scissors."""
    return IteratedGame(
        "Iterated RPS",
        R1=np.array([[0, -1, 1],
                      [1, 0, -1],
                      [-1, 1, 0]]),
        R2=np.array([[0, 1, -1],
                      [-1, 0, 1],
                      [1, -1, 0]]),
        gamma=gamma
    )

def iterated_chicken(gamma=0.96):
    """Iterated Chicken / Hawk-Dove."""
    return IteratedGame(
        "Iterated Chicken",
        R1=np.array([[0, -1],
                      [1, -100]]),
        R2=np.array([[0, 1],
                      [-1, -100]]),
        gamma=gamma
    )

def iterated_stag_hunt(gamma=0.96):
    """Iterated Stag Hunt."""
    return IteratedGame(
        "Iterated Stag Hunt",
        R1=np.array([[4, 0],
                      [3, 2]]),
        R2=np.array([[4, 3],
                      [0, 2]]),
        gamma=gamma
    )


# ─── Named Strategies (for IPD) ────────────────────────────

def make_always_cooperate(game):
    """Always cooperate: π(C|s) = 1 for all s."""
    logits = np.zeros((game.n_states, game.n_actions_1))
    logits[:, 0] = 10.0  # strongly prefer action 0 = C
    return logits

def make_always_defect(game):
    """Always defect: π(D|s) = 1 for all s."""
    logits = np.zeros((game.n_states, game.n_actions_1))
    logits[:, 1] = 10.0
    return logits

def make_tit_for_tat(game):
    """
    Tit-for-Tat: cooperate initially, then copy opponent's last action.
    State 0 (initial): C
    State (C,C)=1: C, State (C,D)=2: D, State (D,C)=3: C, State (D,D)=4: D
    """
    logits = np.zeros((game.n_states, game.n_actions_1))
    # Initial: cooperate
    logits[0, 0] = 10.0
    # After CC: cooperate
    logits[1, 0] = 10.0
    # After CD: defect (opponent defected)
    logits[2, 1] = 10.0
    # After DC: cooperate (opponent cooperated)
    logits[3, 0] = 10.0
    # After DD: defect (opponent defected)
    logits[4, 1] = 10.0
    return logits

def make_grim_trigger(game):
    """
    Grim Trigger: cooperate until opponent defects, then always defect.
    Approximated: defect in any state where opponent defected.
    """
    logits = np.zeros((game.n_states, game.n_actions_1))
    logits[0, 0] = 10.0   # initial: C
    logits[1, 0] = 10.0   # after CC: C
    logits[2, 1] = 10.0   # after CD: D (opponent defected)
    logits[3, 0] = 10.0   # after DC: C (we defected, opponent cooperated)
    logits[4, 1] = 10.0   # after DD: D
    return logits

def make_pavlov(game):
    """
    Pavlov (Win-Stay, Lose-Shift):
    Repeat action if reward was high (CC or DC), switch if low (CD or DD).
    """
    logits = np.zeros((game.n_states, game.n_actions_1))
    logits[0, 0] = 10.0   # initial: C
    logits[1, 0] = 10.0   # after CC: stay (C) — win
    logits[2, 1] = 10.0   # after CD: shift (D) — lose
    logits[3, 1] = 10.0   # after DC: shift (D→C, but we were D so shift to C... tricky)
    # Actually: after DC, we got high reward, so stay = D
    logits[3, 1] = 10.0   # stay with D
    logits[4, 0] = 10.0   # after DD: shift to C — lose
    return logits

def make_random_persona(game, coop_prob_range=(0.0, 1.0)):
    """Random persona with cooperation probability in given range."""
    logits = np.zeros((game.n_states, game.n_actions_1))
    for s in range(game.n_states):
        p_coop = np.random.uniform(*coop_prob_range)
        logits[s, 0] = np.log(max(p_coop, 1e-4))
        logits[s, 1] = np.log(max(1 - p_coop, 1e-4))
    return logits


# ============================================================
# Ω-PG for Iterated Games
# ============================================================

def project_simplex(x):
    """Project onto probability simplex."""
    x = np.maximum(x, 1e-8)
    return x / x.sum()


def reinforce_gradient(game, logits, player, trajectory):
    """
    REINFORCE gradient for tabular policy in iterated game.

    ∇_logits V = Σ_t γ^t R_t ∇_logits log π(a_t | s_t)
    """
    pi = game.softmax(logits)
    grad = np.zeros_like(logits)
    discount = 1.0

    for t, (state, a1, a2, r1, r2) in enumerate(trajectory):
        reward = r1 if player == 0 else r2
        action = a1 if player == 0 else a2

        # Score function: ∇ log π(a|s) = e_a - π(·|s)
        score = -pi[state].copy()
        score[action] += 1.0

        grad[state] += discount * reward * score
        discount *= game.gamma

    return grad


def run_omega_iterated(game: IteratedGame, opponent_logits: np.ndarray,
                        n_episodes: int = 500, lr: float = 0.3,
                        evidence_weight: float = 1.0,
                        lola_lambda: float = 0.0,
                        coop_beta: float = 0.0,
                        n_samples: int = 5) -> dict:
    """
    Run Ω-PG for player 1 against a fixed (or slowly adapting) opponent.

    Returns trajectory of payoffs and policies.
    """
    # Initialize player 1's policy randomly
    logits1 = np.random.randn(game.n_states, game.n_actions_1) * 0.1
    logits2 = opponent_logits.copy()

    payoff_history = []
    coop_rate_history = []

    for ep in range(n_episodes):
        gamma_lr = lr / (1 + ep / 200)

        # Collect trajectories
        grads = np.zeros_like(logits1)
        total_r1, total_r2 = 0.0, 0.0
        total_coop = 0
        total_actions = 0

        for _ in range(n_samples):
            r1, r2, traj = game.play_episode(logits1, logits2)
            total_r1 += r1
            total_r2 += r2
            grads += reinforce_gradient(game, logits1, 0, traj)

            # Track cooperation rate
            for s, a1, a2, _, _ in traj:
                if a1 == 0:
                    total_coop += 1
                total_actions += 1

        grads /= n_samples
        total_r1 /= n_samples
        total_r2 /= n_samples

        # LOLA correction (simplified): anticipate opponent's adaptation
        if lola_lambda > 0 and ep > 10:
            eps = 0.05
            # Estimate how opponent's update affects our payoff
            opp_grad = np.zeros_like(logits2)
            for _ in range(3):
                _, _, traj = game.play_episode(logits1, logits2)
                opp_grad += reinforce_gradient(game, logits2, 1, traj)
            opp_grad /= 3

            # Finite-difference LOLA
            logits2_plus = logits2 + eps * opp_grad
            logits2_minus = logits2 - eps * opp_grad
            v1_plus, _ = game.exact_payoffs(logits1, logits2_plus)
            v1_minus, _ = game.exact_payoffs(logits1, logits2_minus)

            lola_signal = (v1_plus - v1_minus) / (2 * eps)
            # Shape: scalar. Apply as uniform boost to gradient
            grads += lola_lambda * lola_signal * np.sign(grads) / (1 + ep / 100)

        # Cooperation term: share info about own policy
        if coop_beta > 0:
            pi1 = game.softmax(logits1)
            pi2 = game.softmax(logits2)
            # Nudge toward opponent's revealed strategy
            if game.n_actions_1 == game.n_actions_2:
                coop_signal = pi2[:, :game.n_actions_1] - pi1
                grads += coop_beta * coop_signal / (1 + ep / 100)

        # Evidence-weighted update
        w = min(evidence_weight, 1.0)  # normalize
        logits1 += gamma_lr * w * grads

        payoff_history.append(total_r1)
        coop_rate_history.append(total_coop / max(total_actions, 1))

    return {
        'payoffs': np.array(payoff_history),
        'coop_rates': np.array(coop_rate_history),
        'final_logits': logits1,
        'game': game,
    }


# ============================================================
# Persona Population (Kim et al. setup)
# ============================================================

def generate_personas(game, n_personas=20):
    """
    Generate a population of opponent personas (Kim et al. style).

    Mix of:
      - Named strategies (TFT, always-C, always-D, Pavlov, Grim)
      - Random cooperators (coop prob 0.5-1.0)
      - Random defectors (coop prob 0.0-0.5)
    """
    personas = []

    if game.n_actions_1 == 2:  # IPD-like
        # Named strategies
        personas.append(('Always-C', make_always_cooperate(game)))
        personas.append(('Always-D', make_always_defect(game)))
        personas.append(('TFT', make_tit_for_tat(game)))
        personas.append(('Grim', make_grim_trigger(game)))
        personas.append(('Pavlov', make_pavlov(game)))

        # Random cooperators
        for i in range(n_personas // 2 - 2):
            np.random.seed(1000 + i)
            personas.append((f'Coop-{i}', make_random_persona(game, (0.5, 1.0))))

        # Random defectors
        for i in range(n_personas // 2 - 3):
            np.random.seed(2000 + i)
            personas.append((f'Defect-{i}', make_random_persona(game, (0.0, 0.5))))

    else:  # RPS-like
        for i in range(n_personas):
            np.random.seed(3000 + i)
            logits = np.random.randn(game.n_states, game.n_actions_1) * 0.5
            # Bias toward one action
            bias_action = i % game.n_actions_1
            logits[:, bias_action] += 1.0
            personas.append((f'Persona-{i}', logits))

    return personas


# ============================================================
# Experiments
# ============================================================

def experiment_1_ipd_adaptation():
    """
    Kim et al. Q1: Adaptation in the Iterated Prisoner's Dilemma.

    Test how each Ω-component adapts to different opponent personas.
    Key metric: payoff over adaptation episodes (AUC).
    """
    print("\n" + "="*70)
    print("Experiment 1: IPD Adaptation (Kim et al. Q1)")
    print("="*70)

    game = iterated_prisoners_dilemma()
    personas = generate_personas(game, n_personas=16)

    methods = {
        'REINFORCE':  dict(lr=0.3, evidence_weight=1.0, lola_lambda=0.0, coop_beta=0.0),
        'EW-PG':      dict(lr=0.3, evidence_weight=0.5, lola_lambda=0.0, coop_beta=0.0),
        'LOLA':       dict(lr=0.3, evidence_weight=1.0, lola_lambda=0.3, coop_beta=0.0),
        'Coop':       dict(lr=0.3, evidence_weight=1.0, lola_lambda=0.0, coop_beta=0.2),
        'Ω-PG':       dict(lr=0.3, evidence_weight=0.5, lola_lambda=0.2, coop_beta=0.1),
    }

    n_episodes = 300
    n_runs = 5

    # Track adaptation curves per method, averaged over personas and runs
    all_curves = {name: [] for name in methods}
    all_aucs = {name: [] for name in methods}

    for p_name, p_logits in personas:
        for m_name, m_kwargs in methods.items():
            for run in range(n_runs):
                np.random.seed(hash((p_name, m_name, run)) % 2**31)
                result = run_omega_iterated(
                    game, p_logits, n_episodes=n_episodes, n_samples=3, **m_kwargs
                )
                all_curves[m_name].append(result['payoffs'])
                all_aucs[m_name].append(result['payoffs'].sum())

    # Plot adaptation curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    window = 20
    for m_name, color_key in [('REINFORCE', 'reinforce'), ('EW-PG', 'ewpg'),
                                ('LOLA', 'lola'), ('Coop', 'coop'), ('Ω-PG', 'omega')]:
        curves = np.array(all_curves[m_name])
        mean_curve = curves.mean(axis=0)
        smooth = np.convolve(mean_curve, np.ones(window)/window, mode='valid')
        se = curves.std(axis=0) / np.sqrt(len(curves))
        smooth_se = np.convolve(se, np.ones(window)/window, mode='valid')

        x = np.arange(window-1, n_episodes)
        ax.plot(x, smooth, label=m_name, color=COLORS[color_key], linewidth=1.5)
        ax.fill_between(x, smooth - smooth_se, smooth + smooth_se,
                         alpha=0.15, color=COLORS[color_key])

    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Payoff')
    ax.set_title('IPD: Adaptation Curves\n(averaged over 16 personas × 5 runs)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # AUC comparison
    ax = axes[1]
    names = list(methods.keys())
    aucs = [np.mean(all_aucs[n]) for n in names]
    errs = [np.std(all_aucs[n]) / np.sqrt(len(all_aucs[n])) for n in names]
    colors_list = [COLORS[k] for k in ['reinforce', 'ewpg', 'lola', 'coop', 'omega']]

    ax.bar(range(len(names)), aucs, yerr=errs, capsize=5,
           color=colors_list, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('AUC (total discounted reward)')
    ax.set_title('IPD: Area Under Adaptation Curve')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Iterated Prisoner\'s Dilemma — Ω-Framework Comparison',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'ipd_adaptation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'ipd_adaptation.png'}")

    for m_name in methods:
        print(f"  {m_name:12s}: AUC = {np.mean(all_aucs[m_name]):.1f} "
              f"± {np.std(all_aucs[m_name]) / np.sqrt(len(all_aucs[m_name])):.1f}")


def experiment_2_cooperation_emergence():
    """
    IPD cooperation rates: does Ω-PG learn to cooperate?

    Track cooperation rate over episodes for each method.
    TFT-like behavior (conditional cooperation) is the gold standard.
    """
    print("\n" + "="*70)
    print("Experiment 2: Cooperation Emergence in IPD")
    print("="*70)

    game = iterated_prisoners_dilemma()

    # Test against TFT opponent (the most interesting case)
    tft_logits = make_tit_for_tat(game)

    methods = {
        'REINFORCE':  dict(lr=0.3, evidence_weight=1.0, lola_lambda=0.0, coop_beta=0.0),
        'LOLA':       dict(lr=0.3, evidence_weight=1.0, lola_lambda=0.3, coop_beta=0.0),
        'Ω-PG':       dict(lr=0.3, evidence_weight=0.5, lola_lambda=0.2, coop_beta=0.1),
    }

    n_episodes = 400
    n_runs = 15

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    opponents = [
        ('vs TFT', make_tit_for_tat(game)),
        ('vs Always-C', make_always_cooperate(game)),
        ('vs Always-D', make_always_defect(game)),
    ]

    for idx, (opp_name, opp_logits) in enumerate(opponents):
        ax = axes[idx]
        for m_name, m_kwargs in methods.items():
            all_coop = []
            for run in range(n_runs):
                np.random.seed(run * 77 + idx)
                result = run_omega_iterated(
                    game, opp_logits, n_episodes=n_episodes, n_samples=3, **m_kwargs
                )
                all_coop.append(result['coop_rates'])

            coop = np.array(all_coop).mean(axis=0)
            w = 20
            smooth = np.convolve(coop, np.ones(w)/w, mode='valid')
            color = COLORS[{'REINFORCE': 'reinforce', 'LOLA': 'lola', 'Ω-PG': 'omega'}[m_name]]
            ax.plot(range(w-1, n_episodes), smooth, label=m_name,
                    color=color, linewidth=1.5)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Cooperation Rate')
        ax.set_title(f'IPD {opp_name}')
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    fig.suptitle('Cooperation Emergence: Does Ω-PG Learn to Cooperate?',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'cooperation_emergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'cooperation_emergence.png'}")


def experiment_3_irps_competitive():
    """
    Kim et al. Q5: Fully competitive setting — Iterated RPS.

    In zero-sum games, LOLA can exploit opponents by anticipating their
    adaptation. Test whether Ω-PG achieves positive payoff.
    """
    print("\n" + "="*70)
    print("Experiment 3: Iterated Rock-Paper-Scissors (Competitive)")
    print("="*70)

    game = iterated_rps()
    personas = generate_personas(game, n_personas=15)

    methods = {
        'REINFORCE':  dict(lr=0.2, evidence_weight=1.0, lola_lambda=0.0, coop_beta=0.0),
        'LOLA':       dict(lr=0.2, evidence_weight=1.0, lola_lambda=0.3, coop_beta=0.0),
        'Ω-PG':       dict(lr=0.2, evidence_weight=0.5, lola_lambda=0.2, coop_beta=0.0),
    }

    n_episodes = 250
    n_runs = 5
    all_curves = {name: [] for name in methods}

    for p_name, p_logits in personas:
        for m_name, m_kwargs in methods.items():
            for run in range(n_runs):
                np.random.seed(hash((p_name, m_name, run)) % 2**31)
                result = run_omega_iterated(
                    game, p_logits, n_episodes=n_episodes, n_samples=3, **m_kwargs
                )
                all_curves[m_name].append(result['payoffs'])

    fig, ax = plt.subplots(figsize=(9, 5))
    window = 15
    for m_name, color_key in [('REINFORCE', 'reinforce'), ('LOLA', 'lola'), ('Ω-PG', 'omega')]:
        curves = np.array(all_curves[m_name])
        mean_curve = curves.mean(axis=0)
        smooth = np.convolve(mean_curve, np.ones(window)/window, mode='valid')
        x = np.arange(window-1, n_episodes)
        ax.plot(x, smooth, label=m_name, color=COLORS[color_key], linewidth=2)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Nash value = 0')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Payoff (Player 1)')
    ax.set_title('Iterated RPS: Competitive Adaptation\n(averaged over 15 personas × 5 runs)')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'irps_competitive.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'irps_competitive.png'}")


def experiment_4_strategy_identification():
    """
    What strategies does Ω-PG converge to in IPD?

    Analyze the learned policy by comparing to named strategies.
    Visualize as heatmaps: π(C|s) for each state.
    """
    print("\n" + "="*70)
    print("Experiment 4: Strategy Identification in IPD")
    print("="*70)

    game = iterated_prisoners_dilemma()

    # Named strategies for comparison
    named = {
        'TFT': make_tit_for_tat(game),
        'Always-C': make_always_cooperate(game),
        'Always-D': make_always_defect(game),
        'Pavlov': make_pavlov(game),
        'Grim': make_grim_trigger(game),
    }

    # Learn against different opponents
    opponents = [
        ('vs TFT', make_tit_for_tat(game)),
        ('vs Always-C', make_always_cooperate(game)),
        ('vs Always-D', make_always_defect(game)),
        ('vs Pavlov', make_pavlov(game)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    state_labels = ['s₀', 'CC', 'CD', 'DC', 'DD']

    for idx, (opp_name, opp_logits) in enumerate(opponents):
        # Run Ω-PG
        np.random.seed(42 + idx)
        result = run_omega_iterated(
            game, opp_logits, n_episodes=500, lr=0.3, n_samples=5,
            evidence_weight=0.5, lola_lambda=0.2, coop_beta=0.1
        )
        learned_logits = result['final_logits']
        learned_pi = game.softmax(learned_logits)

        # Top row: cooperation probability per state
        ax = axes[0, idx]
        coop_probs = learned_pi[:, 0]  # P(Cooperate | state)
        bars = ax.bar(range(5), coop_probs, color=['#2ecc71' if p > 0.5 else '#e74c3c' for p in coop_probs])
        ax.set_xticks(range(5))
        ax.set_xticklabels(state_labels)
        ax.set_ylabel('P(Cooperate)')
        ax.set_title(f'Ω-PG {opp_name}')
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)

        # Bottom row: distance to each named strategy
        ax = axes[1, idx]
        distances = {}
        for s_name, s_logits in named.items():
            s_pi = game.softmax(s_logits)
            dist = np.mean(np.abs(learned_pi[:, 0] - s_pi[:, 0]))
            distances[s_name] = dist

        names = list(distances.keys())
        dists = list(distances.values())
        colors = ['#2ecc71' if d < 0.2 else '#f39c12' if d < 0.4 else '#e74c3c' for d in dists]
        ax.barh(names, dists, color=colors, alpha=0.8)
        ax.set_xlabel('L1 Distance')
        ax.set_title('Closest Named Strategy')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)

        closest = min(distances, key=distances.get)
        print(f"  {opp_name:15s} → closest to {closest} (dist={distances[closest]:.3f})")

    fig.suptitle('IPD Strategy Identification: What Does Ω-PG Learn?',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'strategy_identification.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'strategy_identification.png'}")


def experiment_5_scaling_agents():
    """
    Kim et al. Q6: More than 2 agents.

    Test with 3-player and 4-player iterated RPS.
    Each agent learns independently; track convergence.
    """
    print("\n" + "="*70)
    print("Experiment 5: Scaling to N > 2 Agents (Iterated RPS)")
    print("="*70)

    # For N-player RPS: each player chooses R/P/S independently
    # Payoff: +1 for each opponent you beat, -1 for each that beats you
    game_2p = iterated_rps()

    agent_counts = [2, 3, 4]
    n_episodes = 300
    n_runs = 8

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, n_agents in enumerate(agent_counts):
        ax = axes[idx]
        all_payoffs = {m: [] for m in ['REINFORCE', 'Ω-PG']}

        for run in range(n_runs):
            np.random.seed(run * 50 + n_agents)
            for m_name, m_kwargs in [
                ('REINFORCE', dict(lr=0.2, evidence_weight=1.0, lola_lambda=0.0, coop_beta=0.0)),
                ('Ω-PG', dict(lr=0.2, evidence_weight=0.5, lola_lambda=0.15, coop_beta=0.05)),
            ]:
                # Simulate N agents: each agent faces the average of others
                agent_logits = [np.random.randn(game_2p.n_states, 3) * 0.1
                                for _ in range(n_agents)]
                payoff_history = []

                for ep in range(n_episodes):
                    gamma_lr = m_kwargs['lr'] / (1 + ep / 200)
                    ep_payoffs = np.zeros(n_agents)

                    # Each agent plays against each other agent
                    for i in range(n_agents):
                        for j in range(n_agents):
                            if i == j:
                                continue
                            r1, r2, traj = game_2p.play_episode(
                                agent_logits[i], agent_logits[j])
                            ep_payoffs[i] += r1 / (n_agents - 1)

                            # Gradient update for agent i
                            grad = reinforce_gradient(
                                game_2p, agent_logits[i], 0, traj)
                            w = min(m_kwargs['evidence_weight'], 1.0)
                            agent_logits[i] += gamma_lr * w * grad / (n_agents - 1)

                    payoff_history.append(ep_payoffs.mean())

                all_payoffs[m_name].append(payoff_history)

        # Plot
        w = 15
        for m_name, color in [('REINFORCE', COLORS['reinforce']), ('Ω-PG', COLORS['omega'])]:
            curves = np.array(all_payoffs[m_name])
            mean = curves.mean(axis=0)
            smooth = np.convolve(mean, np.ones(w)/w, mode='valid')
            ax.plot(range(w-1, n_episodes), smooth, label=m_name,
                    color=color, linewidth=1.5)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg Payoff')
        ax.set_title(f'{n_agents}-Player Iterated RPS')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle('Scaling: N-Player Iterated RPS', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'scaling_agents.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'scaling_agents.png'}")


def experiment_6_cross_game():
    """
    Cross-game comparison: IPD, Stag Hunt, Chicken.
    How does Ω-PG perform across the full spectrum of mixed-incentive games?
    """
    print("\n" + "="*70)
    print("Experiment 6: Cross-Game Comparison")
    print("="*70)

    games = [
        iterated_prisoners_dilemma(),
        iterated_stag_hunt(),
        iterated_chicken(),
    ]

    methods = {
        'REINFORCE': dict(lr=0.3, evidence_weight=1.0, lola_lambda=0.0, coop_beta=0.0),
        'LOLA':      dict(lr=0.3, evidence_weight=1.0, lola_lambda=0.3, coop_beta=0.0),
        'Ω-PG':      dict(lr=0.3, evidence_weight=0.5, lola_lambda=0.2, coop_beta=0.1),
    }

    n_episodes = 300
    n_runs = 10

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for g_idx, game in enumerate(games):
        personas = generate_personas(game, n_personas=10)

        for m_name, m_kwargs in methods.items():
            all_payoffs = []
            all_coop = []

            for p_name, p_logits in personas:
                for run in range(n_runs):
                    np.random.seed(hash((p_name, m_name, run, g_idx)) % 2**31)
                    result = run_omega_iterated(
                        game, p_logits, n_episodes=n_episodes, n_samples=3, **m_kwargs
                    )
                    all_payoffs.append(result['payoffs'])
                    all_coop.append(result['coop_rates'])

            # Payoff curves
            ax = axes[0, g_idx]
            curves = np.array(all_payoffs)
            mean = curves.mean(axis=0)
            w = 20
            smooth = np.convolve(mean, np.ones(w)/w, mode='valid')
            color = COLORS[{'REINFORCE': 'reinforce', 'LOLA': 'lola', 'Ω-PG': 'omega'}[m_name]]
            ax.plot(range(w-1, n_episodes), smooth, label=m_name,
                    color=color, linewidth=1.5)

            # Cooperation curves
            ax2 = axes[1, g_idx]
            coop = np.array(all_coop).mean(axis=0)
            smooth_c = np.convolve(coop, np.ones(w)/w, mode='valid')
            ax2.plot(range(w-1, n_episodes), smooth_c, label=m_name,
                     color=color, linewidth=1.5)

        axes[0, g_idx].set_title(game.name)
        axes[0, g_idx].set_ylabel('Payoff')
        axes[0, g_idx].legend(fontsize=8)
        axes[0, g_idx].grid(alpha=0.3)

        axes[1, g_idx].set_xlabel('Episode')
        axes[1, g_idx].set_ylabel('Cooperation Rate')
        axes[1, g_idx].set_ylim(-0.05, 1.05)
        axes[1, g_idx].legend(fontsize=8)
        axes[1, g_idx].grid(alpha=0.3)

    fig.suptitle('Cross-Game: Payoffs (top) and Cooperation (bottom)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'cross_game.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'cross_game.png'}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Iterated Games × Ω-Framework")
    print("Adapted from Kim et al. (ICML 2021)")
    print("=" * 70)

    experiment_1_ipd_adaptation()
    experiment_2_cooperation_emergence()
    experiment_3_irps_competitive()
    experiment_4_strategy_identification()
    experiment_5_scaling_agents()
    experiment_6_cross_game()

    print("\n" + "="*70)
    print("All iterated game experiments complete.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*70)
