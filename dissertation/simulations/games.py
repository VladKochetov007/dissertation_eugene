"""
Game environments for multi-agent policy gradient experiments.

Each game is a stateless (matrix) game defined by reward matrices R1, R2.
Agent i plays action a^i in {0, 1} with probability sigma(phi^i).
"""

import numpy as np


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


def sigmoid_grad(x: float) -> float:
    """Derivative of sigmoid: σ(x)(1 - σ(x))."""
    s = sigmoid(x)
    return s * (1.0 - s)


class MatrixGame:
    """
    Two-player stateless game with sigmoid-parameterised policies.

    Agent 1: P(action=0) = σ(φ₁)
    Agent 2: P(action=0) = σ(φ₂)

    R1[i,j] = reward to agent 1 when agent 1 plays i, agent 2 plays j.
    """

    def __init__(self, R1: np.ndarray, R2: np.ndarray, name: str = ""):
        assert R1.shape == R2.shape == (2, 2)
        self.R1 = R1.astype(float)
        self.R2 = R2.astype(float)
        self.name = name

    def expected_returns(self, phi1: float, phi2: float):
        """Compute E[G^1], E[G^2] given policy parameters."""
        p1 = sigmoid(phi1)  # P(a1=0)
        p2 = sigmoid(phi2)  # P(a2=0)

        # Joint probability matrix
        P = np.array([
            [p1 * p2, p1 * (1 - p2)],
            [(1 - p1) * p2, (1 - p1) * (1 - p2)]
        ])

        V1 = np.sum(P * self.R1)
        V2 = np.sum(P * self.R2)
        return V1, V2

    def gradients(self, phi1: float, phi2: float):
        """
        Exact policy gradients for both agents.

        Returns:
            dV1_dphi1, dV1_dphi2, dV2_dphi1, dV2_dphi2
        """
        p1 = sigmoid(phi1)
        p2 = sigmoid(phi2)
        dp1 = sigmoid_grad(phi1)
        dp2 = sigmoid_grad(phi2)

        # dV1/dphi1: differentiate p1 terms
        # V1 = p1*p2*R[0,0] + p1*(1-p2)*R[0,1] + (1-p1)*p2*R[1,0] + (1-p1)*(1-p2)*R[1,1]
        # dV1/dp1 = p2*(R[0,0]-R[1,0]) + (1-p2)*(R[0,1]-R[1,1])
        dV1_dp1 = p2 * (self.R1[0, 0] - self.R1[1, 0]) + (1 - p2) * (self.R1[0, 1] - self.R1[1, 1])
        dV1_dp2 = p1 * (self.R1[0, 0] - self.R1[0, 1]) + (1 - p1) * (self.R1[1, 0] - self.R1[1, 1])

        dV2_dp1 = p2 * (self.R2[0, 0] - self.R2[1, 0]) + (1 - p2) * (self.R2[0, 1] - self.R2[1, 1])
        dV2_dp2 = p1 * (self.R2[0, 0] - self.R2[0, 1]) + (1 - p1) * (self.R2[1, 0] - self.R2[1, 1])

        return dV1_dp1 * dp1, dV1_dp2 * dp2, dV2_dp1 * dp1, dV2_dp2 * dp2

    def hessians(self, phi1: float, phi2: float):
        """
        Cross-derivatives needed for LOLA: d²V^i / (dphi_j dphi_i).

        Returns:
            d2V1_dphi2_dphi1, d2V2_dphi1_dphi2
        """
        dp1 = sigmoid_grad(phi1)
        dp2 = sigmoid_grad(phi2)

        # d²V1/(dp1 dp2) = R1[0,0] - R1[0,1] - R1[1,0] + R1[1,1]
        d2V1_dp1dp2 = self.R1[0, 0] - self.R1[0, 1] - self.R1[1, 0] + self.R1[1, 1]
        d2V2_dp1dp2 = self.R2[0, 0] - self.R2[0, 1] - self.R2[1, 0] + self.R2[1, 1]

        return d2V1_dp1dp2 * dp1 * dp2, d2V2_dp1dp2 * dp1 * dp2


# ─── Standard Games ───────────────────────────────────────────────────

def matching_pennies() -> MatrixGame:
    """Zero-sum matching pennies. Nash: (0.5, 0.5), Value: 0."""
    R1 = np.array([[1, -1], [-1, 1]])
    R2 = -R1
    return MatrixGame(R1, R2, name="Matching Pennies")


def prisoners_dilemma() -> MatrixGame:
    """
    Prisoner's Dilemma. Action 0 = Cooperate, Action 1 = Defect.
    Nash: (D, D) with payoff (-1, -1). Pareto optimal: (C, C) with (3, 3).
    """
    R1 = np.array([[3, 0], [4, 1]])  # Rows: C, D for agent 1
    R2 = np.array([[3, 4], [0, 1]])  # Cols: C, D for agent 2
    return MatrixGame(R1, R2, name="Prisoner's Dilemma")


def coordination_game() -> MatrixGame:
    """Pure coordination game. Two Nash equilibria: (0,0) and (1,1)."""
    R1 = np.array([[2, 0], [0, 1]])
    R2 = R1.copy()
    return MatrixGame(R1, R2, name="Coordination Game")
