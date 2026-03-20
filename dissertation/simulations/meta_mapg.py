"""
Multi-agent policy gradient methods for stateless matrix games.

Implements three methods from the dissertation:
1. Independent PG (Naive) — Term 1 only
2. LOLA — Terms 1 + 3 (approximate)
3. Meta-MAPG — Terms 1 + 2 + 3

For stateless games with one-step episodes, Term 2 (own learning) manifests
as anticipating how your current gradient step affects your future policy.
In the multi-step chain setting of Kim et al. (2021), this becomes the
full inner-loop differentiation.
"""

import numpy as np
from games import MatrixGame, sigmoid


def run_independent_pg(
    game: MatrixGame,
    phi1_init: float = 0.0,
    phi2_init: float = 0.0,
    lr: float = 0.1,
    steps: int = 200,
) -> dict:
    """
    Independent (naive) policy gradient. Each agent treats the other as
    part of a stationary environment.

    Update: φᵢ ← φᵢ + α ∇_{φᵢ} Vⁱ(φ₁, φ₂)  [Term 1 only]
    """
    phi1, phi2 = phi1_init, phi2_init
    history = {"phi1": [], "phi2": [], "p1": [], "p2": [], "V1": [], "V2": []}

    for _ in range(steps):
        V1, V2 = game.expected_returns(phi1, phi2)
        dV1_dphi1, _, _, dV2_dphi2 = game.gradients(phi1, phi2)

        history["phi1"].append(phi1)
        history["phi2"].append(phi2)
        history["p1"].append(sigmoid(phi1))
        history["p2"].append(sigmoid(phi2))
        history["V1"].append(V1)
        history["V2"].append(V2)

        phi1 += lr * dV1_dphi1
        phi2 += lr * dV2_dphi2

    return history


def run_lola(
    game: MatrixGame,
    phi1_init: float = 0.0,
    phi2_init: float = 0.0,
    lr: float = 0.1,
    lr_opponent: float = 0.1,
    steps: int = 200,
) -> dict:
    """
    Learning with Opponent-Learning Awareness (Foerster et al., 2018).

    Agent i anticipates agent -i's gradient step and optimises accordingly:
    ∇_{φᵢ} Vⁱ(φᵢ, φ₋ᵢ + Δφ₋ᵢ) ≈ ∇_{φᵢ} Vⁱ + (∇_{φ₋ᵢ} Vⁱ)(∂Δφ₋ᵢ/∂φᵢ)

    This captures Term 1 + approximate Term 3 (peer learning).
    """
    phi1, phi2 = phi1_init, phi2_init
    history = {"phi1": [], "phi2": [], "p1": [], "p2": [], "V1": [], "V2": []}

    for _ in range(steps):
        V1, V2 = game.expected_returns(phi1, phi2)
        dV1_dphi1, dV1_dphi2, dV2_dphi1, dV2_dphi2 = game.gradients(phi1, phi2)
        d2V1_cross, d2V2_cross = game.hessians(phi1, phi2)

        history["phi1"].append(phi1)
        history["phi2"].append(phi2)
        history["p1"].append(sigmoid(phi1))
        history["p2"].append(sigmoid(phi2))
        history["V1"].append(V1)
        history["V2"].append(V2)

        # LOLA gradient for agent 1:
        # Standard gradient + opponent-anticipation term
        # The opponent-anticipation: dV1/dphi2 * d(phi2')/dphi1
        # where phi2' = phi2 + lr_opp * dV2/dphi2
        # so d(phi2')/dphi1 = lr_opp * d²V2/(dphi1 dphi2)
        # But we need d/dphi1 of [dV2/dphi2], which involves second derivatives.
        # For the matrix game, d(phi2')/dphi1 = lr_opp * d2V2_cross
        lola_correction_1 = dV1_dphi2 * lr_opponent * d2V2_cross
        lola_correction_2 = dV2_dphi1 * lr_opponent * d2V1_cross

        phi1 += lr * (dV1_dphi1 + lola_correction_1)
        phi2 += lr * (dV2_dphi2 + lola_correction_2)

    return history


def run_meta_mapg(
    game: MatrixGame,
    phi1_init: float = 0.0,
    phi2_init: float = 0.0,
    lr_inner: float = 0.1,
    lr_outer: float = 0.05,
    lookahead: int = 1,
    steps: int = 200,
) -> dict:
    """
    Meta-MAPG (Kim et al., 2021). Full three-term gradient.

    Outer loop: choose φ₀ to maximise V after `lookahead` inner-loop steps.
    Inner loop: both agents take standard PG steps.

    The outer gradient includes:
    - Term 1: Current policy gradient
    - Term 2: Own learning (how φ₀ affects own future φ)
    - Term 3: Peer learning (how φ₀ affects opponent's future φ)
    """
    phi1, phi2 = phi1_init, phi2_init
    history = {"phi1": [], "phi2": [], "p1": [], "p2": [], "V1": [], "V2": []}

    for _ in range(steps):
        V1, V2 = game.expected_returns(phi1, phi2)

        history["phi1"].append(phi1)
        history["phi2"].append(phi2)
        history["p1"].append(sigmoid(phi1))
        history["p2"].append(sigmoid(phi2))
        history["V1"].append(V1)
        history["V2"].append(V2)

        # Inner loop simulation: run `lookahead` steps of joint PG
        p1_inner, p2_inner = phi1, phi2

        # Track Jacobians: d(phi_inner)/d(phi_0)
        # For agent 1: dphi1_inner/dphi1_0, dphi1_inner/dphi2_0 (= 0 for own)
        # For agent 2: dphi2_inner/dphi1_0 (peer learning), dphi2_inner/dphi2_0
        J11 = 1.0  # dphi1_inner / dphi1_init
        J22 = 1.0  # dphi2_inner / dphi2_init
        J21 = 0.0  # dphi2_inner / dphi1_init (peer learning Jacobian)
        J12 = 0.0  # dphi1_inner / dphi2_init (peer learning Jacobian)

        for _ in range(lookahead):
            g = game.gradients(p1_inner, p2_inner)
            dV1_dp1, dV1_dp2, dV2_dp1, dV2_dp2 = g
            h = game.hessians(p1_inner, p2_inner)
            d2V1_cross, d2V2_cross = h

            # How the inner gradient step changes w.r.t. initial params
            # phi1_new = phi1_inner + lr * dV1/dphi1
            # dphi1_new/dphi1_0 = dphi1_inner/dphi1_0 + lr * d(dV1/dphi1)/dphi1_0
            #   = J11 + lr * [d2V1/dphi1² * J11 + d2V1/(dphi1 dphi2) * J21]
            # Simplified for matrix game:
            # d²V1/dphi1² involves third-order terms; approximate with finite diff
            eps = 1e-5
            g_plus = game.gradients(p1_inner + eps, p2_inner)
            g_minus = game.gradients(p1_inner - eps, p2_inner)
            d2V1_dp1dp1 = (g_plus[0] - g_minus[0]) / (2 * eps)

            g_plus2 = game.gradients(p1_inner, p2_inner + eps)
            g_minus2 = game.gradients(p1_inner, p2_inner - eps)
            d2V2_dp2dp2 = (g_plus2[3] - g_minus2[3]) / (2 * eps)

            # Also need d(dV1/dphi1)/dphi2 = d2V1_cross (already have)
            # And d(dV2/dphi2)/dphi1 = d2V2_cross (already have)

            # Update Jacobians
            J11_new = J11 + lr_inner * (d2V1_dp1dp1 * J11 + d2V1_cross * J21)
            J21_new = J21 + lr_inner * (d2V2_cross * J11 + d2V2_dp2dp2 * J21)
            J22_new = J22 + lr_inner * (d2V2_dp2dp2 * J22 + d2V2_cross * J12)
            J12_new = J12 + lr_inner * (d2V1_cross * J22 + d2V1_dp1dp1 * J12)

            J11, J21, J22, J12 = J11_new, J21_new, J22_new, J12_new

            # Inner loop step
            p1_inner += lr_inner * dV1_dp1
            p2_inner += lr_inner * dV2_dp2

        # Compute gradient of V_after w.r.t. phi_0 using chain rule
        g_after = game.gradients(p1_inner, p2_inner)
        dVf1_dp1, dVf1_dp2, dVf2_dp1, dVf2_dp2 = g_after

        # Full Meta-MAPG gradient for agent 1:
        # dV1_after/dphi1_0 = dV1_after/dphi1_inner * J11 + dV1_after/dphi2_inner * J21
        # Term 1 + Term 2: via J11 (own params chain)
        # Term 3: via J21 (peer params chain)
        meta_grad_1 = dVf1_dp1 * J11 + dVf1_dp2 * J21
        meta_grad_2 = dVf2_dp2 * J22 + dVf2_dp1 * J12

        phi1 += lr_outer * meta_grad_1
        phi2 += lr_outer * meta_grad_2

    return history


def run_meta_pg(
    game: MatrixGame,
    phi1_init: float = 0.0,
    phi2_init: float = 0.0,
    lr_inner: float = 0.1,
    lr_outer: float = 0.05,
    lookahead: int = 1,
    steps: int = 200,
) -> dict:
    """
    Meta-PG (Al-Shedivat et al., 2018). Terms 1 + 2 only (no peer learning).

    Same as Meta-MAPG but sets J21 = J12 = 0 (ignores cross-agent coupling).
    """
    phi1, phi2 = phi1_init, phi2_init
    history = {"phi1": [], "phi2": [], "p1": [], "p2": [], "V1": [], "V2": []}

    for _ in range(steps):
        V1, V2 = game.expected_returns(phi1, phi2)

        history["phi1"].append(phi1)
        history["phi2"].append(phi2)
        history["p1"].append(sigmoid(phi1))
        history["p2"].append(sigmoid(phi2))
        history["V1"].append(V1)
        history["V2"].append(V2)

        # Inner loop
        p1_inner, p2_inner = phi1, phi2
        J11 = 1.0
        J22 = 1.0

        for _ in range(lookahead):
            g = game.gradients(p1_inner, p2_inner)
            dV1_dp1, _, _, dV2_dp2 = g

            eps = 1e-5
            g_plus = game.gradients(p1_inner + eps, p2_inner)
            g_minus = game.gradients(p1_inner - eps, p2_inner)
            d2V1_dp1dp1 = (g_plus[0] - g_minus[0]) / (2 * eps)

            g_plus2 = game.gradients(p1_inner, p2_inner + eps)
            g_minus2 = game.gradients(p1_inner, p2_inner - eps)
            d2V2_dp2dp2 = (g_plus2[3] - g_minus2[3]) / (2 * eps)

            J11 = J11 + lr_inner * d2V1_dp1dp1 * J11
            J22 = J22 + lr_inner * d2V2_dp2dp2 * J22

            p1_inner += lr_inner * dV1_dp1
            p2_inner += lr_inner * dV2_dp2

        # Meta-PG gradient: only own-learning chain, no peer learning
        g_after = game.gradients(p1_inner, p2_inner)
        dVf1_dp1, _, _, dVf2_dp2 = g_after

        meta_pg_grad_1 = dVf1_dp1 * J11  # No J21 term
        meta_pg_grad_2 = dVf2_dp2 * J22  # No J12 term

        phi1 += lr_outer * meta_pg_grad_1
        phi2 += lr_outer * meta_pg_grad_2

    return history
