"""
Extended game environments for dissertation experiments.

Adds: Stag Hunt, Chicken (Hawk-Dove), Battle of the Sexes,
      Rock-Paper-Scissors (3-action), and N-agent games.
"""

import numpy as np
from games import sigmoid, sigmoid_grad


# ─── Additional 2x2 Games ────────────────────────────────────────────

class MatrixGame:
    """Duplicated here to avoid circular import issues with extensions."""

    def __init__(self, R1: np.ndarray, R2: np.ndarray, name: str = ""):
        assert R1.shape == R2.shape == (2, 2)
        self.R1 = R1.astype(float)
        self.R2 = R2.astype(float)
        self.name = name

    def expected_returns(self, phi1: float, phi2: float):
        p1, p2 = sigmoid(phi1), sigmoid(phi2)
        P = np.array([[p1 * p2, p1 * (1 - p2)],
                       [(1 - p1) * p2, (1 - p1) * (1 - p2)]])
        return np.sum(P * self.R1), np.sum(P * self.R2)

    def gradients(self, phi1: float, phi2: float):
        p1, p2 = sigmoid(phi1), sigmoid(phi2)
        dp1, dp2 = sigmoid_grad(phi1), sigmoid_grad(phi2)

        dV1_dp1 = p2 * (self.R1[0, 0] - self.R1[1, 0]) + (1 - p2) * (self.R1[0, 1] - self.R1[1, 1])
        dV1_dp2 = p1 * (self.R1[0, 0] - self.R1[0, 1]) + (1 - p1) * (self.R1[1, 0] - self.R1[1, 1])
        dV2_dp1 = p2 * (self.R2[0, 0] - self.R2[1, 0]) + (1 - p2) * (self.R2[0, 1] - self.R2[1, 1])
        dV2_dp2 = p1 * (self.R2[0, 0] - self.R2[0, 1]) + (1 - p1) * (self.R2[1, 0] - self.R2[1, 1])

        return dV1_dp1 * dp1, dV1_dp2 * dp2, dV2_dp1 * dp1, dV2_dp2 * dp2

    def hessians(self, phi1: float, phi2: float):
        dp1, dp2 = sigmoid_grad(phi1), sigmoid_grad(phi2)
        d2V1_dp1dp2 = self.R1[0, 0] - self.R1[0, 1] - self.R1[1, 0] + self.R1[1, 1]
        d2V2_dp1dp2 = self.R2[0, 0] - self.R2[0, 1] - self.R2[1, 0] + self.R2[1, 1]
        return d2V1_dp1dp2 * dp1 * dp2, d2V2_dp1dp2 * dp1 * dp2

    def nash_mixed(self):
        """Compute mixed-strategy Nash equilibrium for 2x2 game (if it exists).
        Returns (p1*, p2*) or None if only pure NE exist."""
        # Agent 2's mixing probability p2* makes agent 1 indifferent:
        # p2*(R1[0,0]-R1[1,0]) + (1-p2*)*(R1[0,1]-R1[1,1]) = 0
        a = self.R1[0, 0] - self.R1[1, 0] - self.R1[0, 1] + self.R1[1, 1]
        if abs(a) < 1e-10:
            return None
        p2_star = (self.R1[1, 1] - self.R1[0, 1]) / a

        b = self.R2[0, 0] - self.R2[0, 1] - self.R2[1, 0] + self.R2[1, 1]
        if abs(b) < 1e-10:
            return None
        p1_star = (self.R2[1, 1] - self.R2[1, 0]) / b

        if 0 <= p1_star <= 1 and 0 <= p2_star <= 1:
            return (p1_star, p2_star)
        return None


def stag_hunt() -> MatrixGame:
    """
    Stag Hunt: Cooperation vs safety.
    Action 0 = Stag (cooperate), Action 1 = Hare (safe).
    Two NE: (Stag,Stag) payoff-dominant, (Hare,Hare) risk-dominant.
    Tests whether methods find payoff-dominant equilibrium.
    """
    R1 = np.array([[4, 0], [3, 3]])
    R2 = np.array([[4, 3], [0, 3]])
    return MatrixGame(R1, R2, name="Stag Hunt")


def chicken() -> MatrixGame:
    """
    Chicken (Hawk-Dove): Anti-coordination game.
    Action 0 = Dare, Action 1 = Swerve.
    Mixed NE at p=0.5. Two pure NE: (Dare,Swerve) and (Swerve,Dare).
    Tests which equilibrium different methods select.
    """
    R1 = np.array([[0, 4], [1, 2]])
    R2 = np.array([[0, 1], [4, 2]])
    return MatrixGame(R1, R2, name="Chicken")


def battle_of_sexes() -> MatrixGame:
    """
    Battle of the Sexes: Asymmetric coordination.
    Agent 1 prefers (0,0), Agent 2 prefers (1,1). Both prefer coordinating.
    Mixed NE exists. Tests fairness properties of learning algorithms.
    """
    R1 = np.array([[3, 0], [0, 1]])
    R2 = np.array([[1, 0], [0, 3]])
    return MatrixGame(R1, R2, name="Battle of the Sexes")


def deadlock() -> MatrixGame:
    """
    Deadlock: Both agents prefer mutual defection.
    Dominant strategy = Defect. No cooperation dilemma.
    Serves as a CONTROL — all methods should converge identically.
    """
    R1 = np.array([[1, 0], [3, 2]])
    R2 = np.array([[1, 3], [0, 2]])
    return MatrixGame(R1, R2, name="Deadlock")


# ─── N-Agent Extension ───────────────────────────────────────────────

class NPlayerMatrixGame:
    """
    N-player symmetric game with binary actions.

    Each agent i plays action a^i in {0, 1} with P(a^i=0) = sigmoid(phi^i).
    Payoff to agent i depends on own action and NUMBER of cooperators among others.

    reward_fn(own_action, n_cooperators_others, n_total) -> float
    """

    def __init__(self, n_agents: int, reward_fn, name: str = ""):
        self.n = n_agents
        self.reward_fn = reward_fn
        self.name = name

    def expected_return(self, phis: np.ndarray, agent_idx: int) -> float:
        """Compute E[G^i] for agent i given all policy parameters."""
        n = self.n
        probs = np.array([sigmoid(phi) for phi in phis])  # P(action=0) for each

        total = 0.0
        # Sum over all joint action profiles
        for mask in range(2 ** n):
            actions = [(mask >> j) & 1 for j in range(n)]  # 0 or 1 for each agent
            # Compute probability of this joint action
            p = 1.0
            for j in range(n):
                if actions[j] == 0:
                    p *= probs[j]
                else:
                    p *= (1 - probs[j])

            # Agent i's reward
            own_action = actions[agent_idx]
            n_coop_others = sum(1 for j in range(n) if j != agent_idx and actions[j] == 0)
            r = self.reward_fn(own_action, n_coop_others, n - 1)
            total += p * r

        return total

    def all_expected_returns(self, phis: np.ndarray) -> np.ndarray:
        """Compute expected returns for all agents."""
        return np.array([self.expected_return(phis, i) for i in range(self.n)])

    def gradient(self, phis: np.ndarray, agent_idx: int) -> np.ndarray:
        """Compute dV^i/dphi_j for all j, using finite differences."""
        eps = 1e-5
        grad = np.zeros(self.n)
        V0 = self.expected_return(phis, agent_idx)
        for j in range(self.n):
            phis_plus = phis.copy()
            phis_plus[j] += eps
            grad[j] = (self.expected_return(phis_plus, agent_idx) - V0) / eps
        return grad

    def all_gradients(self, phis: np.ndarray) -> np.ndarray:
        """Compute gradient matrix: G[i, j] = dV^i/dphi_j."""
        return np.array([self.gradient(phis, i) for i in range(self.n)])


def n_player_public_goods(n: int, multiplier: float = 1.5, cost: float = 1.0) -> NPlayerMatrixGame:
    """
    N-player public goods game.
    Action 0 = Contribute (cooperate), Action 1 = Free-ride (defect).
    Total contributions are multiplied and shared equally.

    Cooperating costs `cost`, total is multiplied by `multiplier/n`.
    """
    def reward_fn(own_action, n_coop_others, n_others):
        n_total = n_others + 1
        n_cooperators = n_coop_others + (1 if own_action == 0 else 0)
        shared_benefit = (n_cooperators * cost * multiplier) / n_total
        own_cost = cost if own_action == 0 else 0
        return shared_benefit - own_cost

    return NPlayerMatrixGame(n, reward_fn, name=f"{n}-Player Public Goods")


def n_player_stag_hunt(n: int, threshold: int = None) -> NPlayerMatrixGame:
    """
    N-player Stag Hunt with threshold.
    Need at least `threshold` cooperators (including self) for stag success.
    Action 0 = Hunt stag (cooperate), Action 1 = Hunt hare (safe).
    """
    if threshold is None:
        threshold = n  # All must cooperate

    def reward_fn(own_action, n_coop_others, n_others):
        if own_action == 0:  # Hunt stag
            n_coop_total = n_coop_others + 1
            if n_coop_total >= threshold:
                return 4.0  # Stag caught
            else:
                return 0.0  # Stag failed
        else:  # Hunt hare
            return 2.0  # Safe payoff

    return NPlayerMatrixGame(n, reward_fn, name=f"{n}-Player Stag Hunt (threshold={threshold})")
