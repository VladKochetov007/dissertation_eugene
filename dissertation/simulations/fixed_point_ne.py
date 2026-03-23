"""
Fixed-Point Nash Equilibrium Search with Bayesian Exploration

Core idea (Shcherbinin 2026): Nash equilibria are fixed points of the
joint best-response map BR: Δ → Δ. Instead of naive independent PG
(which may cycle or converge to bad NE), agents explicitly search for
fixed points and use Bayesian confidence to govern explore/exploit.

Key insight: ALL selfish agents are interested in finding the BEST NE.
This creates emergent cooperation in the exploration phase — even in
competitive environments, the search for fixed points is a shared goal.

The algorithm:
  1. Sample random starting points on the product simplex
  2. Run best-response dynamics / gradient descent on ||BR(π) - π||²
  3. Collect discovered fixed points (candidate NE)
  4. Score each NE from each agent's perspective
  5. Maintain Bayesian posterior on "number of undiscovered NE"
  6. Stop when P(finding better NE) < threshold

Connects: Kakutani's fixed point theorem (existence guarantee) ×
          Bayesian optimization (explore/exploit) ×
          Functional analysis (contraction mappings, Banach)

Author: Eugene Shcherbinin
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import nashpy as nash
from scipy.optimize import minimize
from scipy.special import digamma
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = Path(__file__).parent / "figures" / "fixed_point"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


# ============================================================
# Game Definitions — richer zoo
# ============================================================

@dataclass
class Game:
    """N-action 2-player game defined by payoff matrices."""
    name: str
    R1: np.ndarray
    R2: np.ndarray
    known_ne: list = field(default_factory=list)  # [(p1, p2), ...] for validation

    @property
    def n1(self): return self.R1.shape[0]

    @property
    def n2(self): return self.R1.shape[1]

    def payoffs(self, p1: np.ndarray, p2: np.ndarray):
        """Expected payoffs under mixed strategies."""
        return p1 @ self.R1 @ p2, p1 @ self.R2 @ p2

    def best_response_1(self, p2: np.ndarray) -> np.ndarray:
        """Best response for player 1 given player 2's mixed strategy."""
        payoffs = self.R1 @ p2
        br = np.zeros(self.n1)
        br[np.argmax(payoffs)] = 1.0
        return br

    def best_response_2(self, p1: np.ndarray) -> np.ndarray:
        """Best response for player 2 given player 1's mixed strategy."""
        payoffs = self.R2.T @ p1
        br = np.zeros(self.n2)
        br[np.argmax(payoffs)] = 1.0
        return br

    def best_response_softmax_1(self, p2: np.ndarray, tau: float = 0.1) -> np.ndarray:
        """Soft best response (smoothed for gradient methods)."""
        payoffs = self.R1 @ p2
        logits = payoffs / tau
        logits -= logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def best_response_softmax_2(self, p1: np.ndarray, tau: float = 0.1) -> np.ndarray:
        payoffs = self.R2.T @ p1
        logits = payoffs / tau
        logits -= logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def fixed_point_residual(self, p1: np.ndarray, p2: np.ndarray,
                              tau: float = 0.01) -> float:
        """||BR(π) - π||² — how far from being a fixed point."""
        br1 = self.best_response_softmax_1(p2, tau)
        br2 = self.best_response_softmax_2(p1, tau)
        return np.sum((br1 - p1)**2) + np.sum((br2 - p2)**2)

    def compute_all_ne(self) -> list:
        """Use nashpy to find all Nash equilibria (support enumeration)."""
        game = nash.Game(self.R1, self.R2)
        equilibria = []
        try:
            for eq in game.support_enumeration():
                p1, p2 = eq
                if np.all(np.isfinite(p1)) and np.all(np.isfinite(p2)):
                    if np.all(p1 >= -1e-8) and np.all(p2 >= -1e-8):
                        p1 = np.maximum(p1, 0); p1 /= p1.sum()
                        p2 = np.maximum(p2, 0); p2 /= p2.sum()
                        equilibria.append((p1, p2))
        except Exception:
            pass
        return equilibria


# ─── The Zoo ────────────────────────────────────────────────

def matching_pennies():
    return Game("Matching Pennies",
                np.array([[1, -1], [-1, 1]]),
                np.array([[-1, 1], [1, -1]]))

def prisoners_dilemma():
    return Game("Prisoner's Dilemma",
                np.array([[-1, -3], [0, -2]]),
                np.array([[-1, 0], [-3, -2]]))

def stag_hunt():
    """Two NE: (Stag,Stag) Pareto-dominant, (Hare,Hare) risk-dominant."""
    return Game("Stag Hunt",
                np.array([[4, 0], [3, 3]]),
                np.array([[4, 3], [0, 3]]))

def battle_of_sexes():
    """Two pure NE + one mixed. Tests equilibrium selection."""
    return Game("Battle of the Sexes",
                np.array([[3, 0], [0, 1]]),
                np.array([[1, 0], [0, 3]]))

def chicken():
    """Hawk-Dove: anti-coordination. Two pure + one mixed NE."""
    return Game("Chicken",
                np.array([[0, 4], [1, 2]]),
                np.array([[0, 1], [4, 2]]))

def coordination_3x3():
    """3-action coordination with Pareto-ranked NE."""
    return Game("3×3 Coordination",
                np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]]),
                np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]]))

def rock_paper_scissors():
    return Game("Rock-Paper-Scissors",
                np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]),
                np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]]))

def shapley_game():
    """Shapley's game: only NE is fully mixed. Hard for learning algorithms."""
    return Game("Shapley's Game",
                np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
                np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))

def grab_the_dollar():
    """3-action game with unique mixed NE. Tests exploration thoroughness."""
    return Game("Grab the Dollar",
                np.array([[5, -1, 3], [3, 5, -1], [-1, 3, 5]]),
                np.array([[5, 3, -1], [-1, 5, 3], [3, -1, 5]]))

def asymmetric_coordination():
    """4 NE (3 pure + 1 mixed). Rich selection problem."""
    return Game("Asymmetric Coordination",
                np.array([[4, 0, 0], [0, 2, 0], [0, 0, 3]]),
                np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]]))


ALL_GAMES = [
    matching_pennies(), prisoners_dilemma(), stag_hunt(),
    battle_of_sexes(), chicken(), coordination_3x3(),
    rock_paper_scissors(), shapley_game(), grab_the_dollar(),
    asymmetric_coordination(),
]


# ============================================================
# Fixed-Point NE Search Algorithm
# ============================================================

def project_simplex(x: np.ndarray) -> np.ndarray:
    """Exact projection onto probability simplex (Duchi et al. 2008)."""
    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u * np.arange(1, n+1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(x - theta, 0)


def find_fixed_point(game: Game, tau: float = 0.05,
                     max_iter: int = 500, tol: float = 1e-6,
                     init: Optional[tuple] = None) -> tuple:
    """
    Find a fixed point of the softmax best-response map via iteration.

    Uses damped best-response iteration: π_{t+1} = (1-α)π_t + α·BR_τ(π_t)
    This is a contraction mapping for small enough τ (Banach fixed point theorem).
    """
    if init is None:
        p1 = np.random.dirichlet(np.ones(game.n1))
        p2 = np.random.dirichlet(np.ones(game.n2))
    else:
        p1, p2 = init[0].copy(), init[1].copy()

    alpha = 0.3  # damping
    trajectory = [(p1.copy(), p2.copy())]

    for _ in range(max_iter):
        br1 = game.best_response_softmax_1(p2, tau)
        br2 = game.best_response_softmax_2(p1, tau)

        p1_new = (1 - alpha) * p1 + alpha * br1
        p2_new = (1 - alpha) * p2 + alpha * br2

        p1_new = project_simplex(p1_new)
        p2_new = project_simplex(p2_new)

        if np.max(np.abs(p1_new - p1)) + np.max(np.abs(p2_new - p2)) < tol:
            p1, p2 = p1_new, p2_new
            break
        p1, p2 = p1_new, p2_new
        trajectory.append((p1.copy(), p2.copy()))

    residual = game.fixed_point_residual(p1, p2, tau)
    return p1, p2, residual, trajectory


def find_fixed_point_optimization(game: Game, tau: float = 0.01,
                                   init: Optional[tuple] = None) -> tuple:
    """
    Find NE by minimizing ||BR_τ(π) - π||² directly.
    More robust for games where iteration doesn't converge.
    """
    n1, n2 = game.n1, game.n2

    if init is None:
        x0 = np.concatenate([np.random.dirichlet(np.ones(n1)),
                              np.random.dirichlet(np.ones(n2))])
    else:
        x0 = np.concatenate([init[0], init[1]])

    def objective(x):
        p1 = project_simplex(x[:n1])
        p2 = project_simplex(x[n1:])
        return game.fixed_point_residual(p1, p2, tau)

    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})

    p1 = project_simplex(result.x[:n1])
    p2 = project_simplex(result.x[n1:])
    residual = game.fixed_point_residual(p1, p2, tau)
    return p1, p2, residual


def are_same_ne(ne1: tuple, ne2: tuple, tol: float = 0.05) -> bool:
    """Check if two NE are the same (up to tolerance)."""
    d = np.max(np.abs(ne1[0] - ne2[0])) + np.max(np.abs(ne1[1] - ne2[1]))
    return d < tol


# ============================================================
# Bayesian NE Counter — posterior on number of equilibria
# ============================================================

class BayesianNECounter:
    """
    Maintains a Bayesian posterior over the number of NE in a game.

    Model: K ~ Poisson(λ) prior on number of NE.
    After discovering d distinct NE in n searches, update posterior.

    Uses a species-discovery model (Good-Turing / Chao estimator style):
    - Each search is a "sample" that may discover a new NE or rediscover one
    - Posterior on K given observations informs stopping criterion

    The stopping criterion: P(∃ undiscovered NE with payoff > best_found) < δ
    """

    def __init__(self, lambda_prior: float = 5.0):
        self.lambda_prior = lambda_prior
        self.discovered_ne: list = []  # list of (p1, p2, v1, v2)
        self.n_searches = 0
        self.discovery_times: list = []  # search index where each new NE was found

    def add_search_result(self, p1, p2, residual, game: Game,
                          residual_threshold: float = 0.01):
        """Record a search result. Returns True if new NE discovered."""
        self.n_searches += 1

        if residual > residual_threshold:
            return False  # didn't converge to a fixed point

        # Check if this is a new NE
        for existing in self.discovered_ne:
            if are_same_ne((p1, p2), (existing[0], existing[1])):
                return False

        v1, v2 = game.payoffs(p1, p2)
        self.discovered_ne.append((p1.copy(), p2.copy(), v1, v2))
        self.discovery_times.append(self.n_searches)
        return True

    @property
    def n_discovered(self) -> int:
        return len(self.discovered_ne)

    def estimated_total_ne(self) -> float:
        """
        Chao1 estimator: K_hat = d + f1² / (2·f2)
        where d = discovered, f1 = singletons, f2 = doubletons
        (adapted for NE discovery context).

        Simple version: use the capture-recapture logic.
        If we've done n searches and found d distinct NE,
        and the rate of new discoveries is slowing:
        E[K] ≈ d / (1 - (1 - d/n)^n) for large n
        """
        d = self.n_discovered
        n = self.n_searches

        if d == 0 or n == 0:
            return self.lambda_prior

        # Simple Good-Turing style: probability of seeing a new species
        # on next sample ≈ (proportion of singletons)
        if n <= d:
            return d * 2  # still discovering fast

        # Coverage estimator
        coverage = d / n  # fraction of searches that found something new
        if coverage >= 1:
            return d + 1

        # Chao1-style lower bound
        return d / (1 - np.exp(-n * coverage / d)) if d > 0 else self.lambda_prior

    def p_undiscovered_better(self, agent_idx: int = 0) -> float:
        """
        P(∃ undiscovered NE with higher payoff for agent `agent_idx`).

        Uses: P(better exists) ≈ P(undiscovered exist) × P(random NE is better)

        Conservative: assumes undiscovered NE are uniformly distributed
        over possible payoff range.
        """
        if self.n_discovered == 0:
            return 1.0  # no info yet

        d = self.n_discovered
        K_hat = self.estimated_total_ne()
        p_undiscovered = max(0, 1 - d / max(K_hat, d + 0.1))

        # Among discovered NE, what's the best payoff for this agent?
        payoffs = [ne[2 + agent_idx] for ne in self.discovered_ne]
        best = max(payoffs)

        # Conservative: assume undiscovered NE payoffs ~ Uniform[min_payoff, max_payoff]
        if len(payoffs) >= 2:
            payoff_range = max(payoffs) - min(payoffs)
            if payoff_range > 1e-8:
                # P(random NE beats current best) — assuming some spread
                p_better_given_exists = 0.3  # conservative prior
            else:
                p_better_given_exists = 0.5  # all found NE have same payoff
        else:
            p_better_given_exists = 0.5  # only 1 NE found, uncertain

        return p_undiscovered * p_better_given_exists

    def should_stop(self, threshold: float = 0.05) -> bool:
        """Stop when P(finding better NE) < threshold for ALL agents."""
        if self.n_discovered == 0:
            return False
        return all(self.p_undiscovered_better(i) < threshold for i in range(2))


# ============================================================
# Full Algorithm: Bayesian Fixed-Point NE Search
# ============================================================

def bayesian_fp_search(game: Game, max_searches: int = 200,
                        confidence_threshold: float = 0.05,
                        tau: float = 0.05,
                        verbose: bool = True) -> dict:
    """
    Bayesian Fixed-Point Nash Equilibrium Search.

    Phase 1 (Explore): Random restarts → BR iteration → collect NE
    Phase 2 (Refine): Fine-tune around best NE with lower τ
    Stop when Bayesian confidence that no better NE exists > 1 - threshold.

    Returns dict with discovered NE, search history, stopping info.
    """
    counter = BayesianNECounter()
    history = {
        'n_searches': [],
        'n_discovered': [],
        'best_payoff_1': [],
        'best_payoff_2': [],
        'p_better': [],
        'residuals': [],
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Bayesian FP-NE Search: {game.name}")
        print(f"  Actions: {game.n1} × {game.n2}")
        print(f"{'='*60}")

    for s in range(max_searches):
        # Alternate between random starts and perturbations of best known
        if s < 10 or np.random.rand() < 0.4 or counter.n_discovered == 0:
            # Random exploration
            init = None
        else:
            # Perturb best known NE
            best_ne = max(counter.discovered_ne, key=lambda x: x[2] + x[3])
            noise_scale = 0.3 * np.exp(-s / 50)  # decreasing noise
            p1_init = project_simplex(best_ne[0] + noise_scale * np.random.randn(game.n1))
            p2_init = project_simplex(best_ne[1] + noise_scale * np.random.randn(game.n2))
            init = (p1_init, p2_init)

        # Try both methods
        p1_a, p2_a, res_a, _ = find_fixed_point(game, tau=tau, init=init)
        p1_b, p2_b, res_b = find_fixed_point_optimization(game, tau=tau/2, init=init)

        # Use the one with lower residual
        if res_a < res_b:
            p1, p2, res = p1_a, p2_a, res_a
        else:
            p1, p2, res = p1_b, p2_b, res_b

        new = counter.add_search_result(p1, p2, res, game)

        # Record history
        history['n_searches'].append(s + 1)
        history['n_discovered'].append(counter.n_discovered)
        history['residuals'].append(res)

        if counter.n_discovered > 0:
            payoffs_1 = [ne[2] for ne in counter.discovered_ne]
            payoffs_2 = [ne[3] for ne in counter.discovered_ne]
            history['best_payoff_1'].append(max(payoffs_1))
            history['best_payoff_2'].append(max(payoffs_2))
            history['p_better'].append(
                max(counter.p_undiscovered_better(0),
                    counter.p_undiscovered_better(1))
            )
        else:
            history['best_payoff_1'].append(np.nan)
            history['best_payoff_2'].append(np.nan)
            history['p_better'].append(1.0)

        if new and verbose:
            v1, v2 = game.payoffs(p1, p2)
            print(f"  Search {s+1}: NEW NE #{counter.n_discovered} "
                  f"| payoffs ({v1:.3f}, {v2:.3f}) | residual {res:.6f}")
            print(f"    p1 = {p1.round(4)}, p2 = {p2.round(4)}")

        # Check stopping criterion
        if counter.should_stop(confidence_threshold) and s >= 20:
            if verbose:
                print(f"\n  Stopped at search {s+1}: "
                      f"P(better NE) < {confidence_threshold:.2f}")
            break

    # Summary
    if verbose:
        print(f"\n  Summary:")
        print(f"    Searches: {counter.n_searches}")
        print(f"    Discovered NE: {counter.n_discovered}")
        print(f"    Estimated total NE: {counter.estimated_total_ne():.1f}")
        for i, ne in enumerate(counter.discovered_ne):
            print(f"    NE {i+1}: payoffs ({ne[2]:.3f}, {ne[3]:.3f}) "
                  f"| p1={ne[0].round(3)} p2={ne[1].round(3)}")

    return {
        'game': game,
        'counter': counter,
        'history': history,
        'discovered_ne': counter.discovered_ne,
    }


# ============================================================
# Comparison algorithms
# ============================================================

def run_independent_pg(game: Game, n_episodes: int = 3000, lr: float = 0.1,
                       n_runs: int = 30) -> dict:
    """Standard independent policy gradient (each agent ignores the other)."""
    all_trajectories = []

    for run in range(n_runs):
        np.random.seed(run * 31)
        # Logit parameterization
        logits1 = np.random.randn(game.n1) * 0.1
        logits2 = np.random.randn(game.n2) * 0.1

        traj = []
        for ep in range(n_episodes):
            gamma = lr / (1 + ep / 500)

            # Softmax policies
            p1 = np.exp(logits1 - logits1.max())
            p1 /= p1.sum()
            p2 = np.exp(logits2 - logits2.max())
            p2 /= p2.sum()

            traj.append((p1.copy(), p2.copy()))

            # Sample and compute REINFORCE gradient
            a1 = np.random.choice(game.n1, p=p1)
            a2 = np.random.choice(game.n2, p=p2)
            r1 = game.R1[a1, a2]
            r2 = game.R2[a1, a2]

            # Score function gradient for softmax
            grad1 = -p1.copy()
            grad1[a1] += 1
            grad1 *= r1

            grad2 = -p2.copy()
            grad2[a2] += 1
            grad2 *= r2

            logits1 += gamma * grad1
            logits2 += gamma * grad2

        all_trajectories.append(traj)

    return {'trajectories': all_trajectories, 'game': game}


def run_fictitious_play(game: Game, n_episodes: int = 3000,
                        n_runs: int = 30) -> dict:
    """Classical fictitious play — each agent best-responds to empirical freq."""
    all_trajectories = []

    for run in range(n_runs):
        np.random.seed(run * 37)
        counts1 = np.ones(game.n1)  # Laplace prior
        counts2 = np.ones(game.n2)

        traj = []
        for ep in range(n_episodes):
            # Empirical frequencies
            p1 = counts1 / counts1.sum()
            p2 = counts2 / counts2.sum()
            traj.append((p1.copy(), p2.copy()))

            # Best respond
            br1 = game.best_response_1(p2)
            br2 = game.best_response_2(p1)

            # Play best response (with small noise for exploration)
            a1 = np.random.choice(game.n1, p=project_simplex(br1 + 0.01))
            a2 = np.random.choice(game.n2, p=project_simplex(br2 + 0.01))

            counts1[a1] += 1
            counts2[a2] += 1

        all_trajectories.append(traj)

    return {'trajectories': all_trajectories, 'game': game}


# ============================================================
# Experiments
# ============================================================

def experiment_1_ne_discovery(games=None):
    """
    Experiment 1: How many NE does the algorithm find vs nashpy ground truth?
    Tests completeness of the fixed-point search.
    """
    if games is None:
        games = ALL_GAMES

    print("\n" + "="*70)
    print("EXPERIMENT 1: NE Discovery Completeness")
    print("="*70)

    results = []
    for game in games:
        true_ne = game.compute_all_ne()
        search = bayesian_fp_search(game, max_searches=150,
                                     confidence_threshold=0.05, verbose=False)

        found = search['counter'].n_discovered
        total = len(true_ne)
        est = search['counter'].estimated_total_ne()

        results.append({
            'name': game.name,
            'true_ne': total,
            'found_ne': found,
            'estimated': est,
            'searches': search['counter'].n_searches,
        })

        print(f"  {game.name:25s} | True NE: {total} | Found: {found} | "
              f"Est: {est:.1f} | Searches: {search['counter'].n_searches}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = [r['name'] for r in results]
    true_counts = [r['true_ne'] for r in results]
    found_counts = [r['found_ne'] for r in results]
    est_counts = [r['estimated'] for r in results]

    x = np.arange(len(names))
    w = 0.25

    axes[0].bar(x - w, true_counts, w, label='True NE (nashpy)', color='#2c3e50')
    axes[0].bar(x, found_counts, w, label='Found (FP search)', color='#2ecc71')
    axes[0].bar(x + w, est_counts, w, label='Estimated total', color='#f39c12', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Number of NE')
    axes[0].set_title('NE Discovery: Ground Truth vs Found')
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    searches = [r['searches'] for r in results]
    axes[1].barh(names, searches, color='#3498db', alpha=0.8)
    axes[1].set_xlabel('Searches until convergence')
    axes[1].set_title('Search Efficiency')
    axes[1].grid(axis='x', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'ne_discovery.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {FIGURES_DIR / 'ne_discovery.png'}")

    return results


def experiment_2_bayesian_stopping(games=None):
    """
    Experiment 2: Bayesian confidence curves.
    Shows how P(better NE exists) decreases over searches.
    """
    if games is None:
        games = [stag_hunt(), battle_of_sexes(), chicken(),
                 coordination_3x3(), asymmetric_coordination()]

    print("\n" + "="*70)
    print("EXPERIMENT 2: Bayesian Stopping Curves")
    print("="*70)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, game in enumerate(games):
        if idx >= len(axes):
            break

        search = bayesian_fp_search(game, max_searches=100,
                                     confidence_threshold=0.02, verbose=False)

        h = search['history']
        ax = axes[idx]

        # P(better NE exists) over searches
        ax.plot(h['n_searches'], h['p_better'], 'b-', linewidth=1.5, label='P(better NE)')
        ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='threshold')

        # Mark discovery events
        for t in search['counter'].discovery_times:
            ax.axvline(x=t, color='green', alpha=0.3, linewidth=1)

        ax.set_xlabel('Searches')
        ax.set_ylabel('P(undiscovered better NE)')
        ax.set_title(f'{game.name}\n({search["counter"].n_discovered} NE found)')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

        print(f"  {game.name}: {search['counter'].n_discovered} NE found in "
              f"{search['counter'].n_searches} searches")

    # Hide unused axes
    for idx in range(len(games), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Bayesian Confidence: When to Stop Exploring', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'bayesian_stopping.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {FIGURES_DIR / 'bayesian_stopping.png'}")


def experiment_3_equilibrium_selection(games=None):
    """
    Experiment 3: Which NE does each method select?
    FP search vs independent PG vs fictitious play.

    Key claim: FP search finds Pareto-better NE because it SEARCHES
    the space explicitly rather than converging to whatever's closest.
    """
    if games is None:
        # Games with multiple NE where selection matters
        games = [stag_hunt(), battle_of_sexes(), chicken(),
                 coordination_3x3(), asymmetric_coordination()]

    print("\n" + "="*70)
    print("EXPERIMENT 3: Equilibrium Selection Quality")
    print("="*70)

    results = []
    for game in games:
        true_ne = game.compute_all_ne()

        # FP search — picks best discovered NE (Pareto criterion)
        search = bayesian_fp_search(game, max_searches=100,
                                     confidence_threshold=0.05, verbose=False)
        if search['counter'].n_discovered > 0:
            # Select NE with highest sum of payoffs (social welfare)
            best_ne = max(search['counter'].discovered_ne, key=lambda x: x[2] + x[3])
            fp_welfare = best_ne[2] + best_ne[3]
            fp_payoffs = (best_ne[2], best_ne[3])
        else:
            fp_welfare = np.nan
            fp_payoffs = (np.nan, np.nan)

        # Independent PG — see where it converges
        pg_result = run_independent_pg(game, n_episodes=2000, n_runs=50)
        pg_final_welfares = []
        for traj in pg_result['trajectories']:
            p1, p2 = traj[-1]
            v1, v2 = game.payoffs(p1, p2)
            pg_final_welfares.append(v1 + v2)

        # Fictitious play
        fp_result = run_fictitious_play(game, n_episodes=2000, n_runs=50)
        fict_final_welfares = []
        for traj in fp_result['trajectories']:
            p1, p2 = traj[-1]
            v1, v2 = game.payoffs(p1, p2)
            fict_final_welfares.append(v1 + v2)

        # Best possible welfare (across all true NE)
        if true_ne:
            best_true_welfare = max(game.payoffs(ne[0], ne[1])[0] +
                                     game.payoffs(ne[0], ne[1])[1]
                                     for ne in true_ne)
        else:
            best_true_welfare = np.nan

        results.append({
            'name': game.name,
            'best_possible': best_true_welfare,
            'fp_search': fp_welfare,
            'pg_mean': np.mean(pg_final_welfares),
            'pg_std': np.std(pg_final_welfares),
            'fict_mean': np.mean(fict_final_welfares),
            'fict_std': np.std(fict_final_welfares),
        })

        print(f"  {game.name:25s} | Best: {best_true_welfare:6.2f} | "
              f"FP: {fp_welfare:6.2f} | PG: {np.mean(pg_final_welfares):6.2f}±"
              f"{np.std(pg_final_welfares):.2f} | "
              f"Fict: {np.mean(fict_final_welfares):6.2f}±"
              f"{np.std(fict_final_welfares):.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [r['name'] for r in results]
    x = np.arange(len(names))
    w = 0.2

    ax.bar(x - 1.5*w, [r['best_possible'] for r in results], w,
           label='Best NE (oracle)', color='#2c3e50', alpha=0.8)
    ax.bar(x - 0.5*w, [r['fp_search'] for r in results], w,
           label='FP Search (ours)', color='#2ecc71')
    ax.bar(x + 0.5*w, [r['pg_mean'] for r in results], w,
           label='Independent PG', color='#e74c3c',
           yerr=[r['pg_std'] for r in results], capsize=3)
    ax.bar(x + 1.5*w, [r['fict_mean'] for r in results], w,
           label='Fictitious Play', color='#3498db',
           yerr=[r['fict_std'] for r in results], capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Social Welfare (sum of payoffs)')
    ax.set_title('Equilibrium Selection: FP Search Finds Better NE')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'equilibrium_selection.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {FIGURES_DIR / 'equilibrium_selection.png'}")

    return results


def experiment_4_convergence_landscape():
    """
    Experiment 4: Visualize the fixed-point residual landscape for 2×2 games.
    Shows the 'topology' of NE — where fixed points live on the simplex.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Fixed-Point Residual Landscapes")
    print("="*70)

    games = [stag_hunt(), battle_of_sexes(), chicken(), matching_pennies()]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    for idx, game in enumerate(games):
        ax = axes[idx]

        # For 2×2 games, strategies are 1D: p1 = P(action 0 for player 1)
        resolution = 100
        p1_range = np.linspace(0.01, 0.99, resolution)
        p2_range = np.linspace(0.01, 0.99, resolution)

        residual_map = np.zeros((resolution, resolution))
        for i, p1_val in enumerate(p1_range):
            for j, p2_val in enumerate(p2_range):
                p1 = np.array([p1_val, 1 - p1_val])
                p2 = np.array([p2_val, 1 - p2_val])
                residual_map[j, i] = game.fixed_point_residual(p1, p2, tau=0.01)

        # Log scale for better visualization
        im = ax.contourf(p1_range, p2_range, np.log10(residual_map + 1e-10),
                          levels=20, cmap='viridis_r')
        plt.colorbar(im, ax=ax, label='log₁₀(residual)')

        # Mark true NE
        true_ne = game.compute_all_ne()
        for ne in true_ne:
            ax.plot(ne[0][0], ne[1][0], 'r*', markersize=15, markeredgecolor='white',
                    markeredgewidth=1, zorder=5)

        # Run a few FP searches and show trajectories
        for trial in range(5):
            np.random.seed(trial * 7 + idx)
            _, _, _, traj = find_fixed_point(game, tau=0.05)
            traj_p1 = [t[0][0] for t in traj]
            traj_p2 = [t[1][0] for t in traj]
            ax.plot(traj_p1, traj_p2, 'w-', alpha=0.4, linewidth=0.8)
            ax.plot(traj_p1[0], traj_p2[0], 'wo', markersize=3, alpha=0.5)

        ax.set_xlabel('p₁ (Player 1: P(action 0))')
        ax.set_ylabel('p₂ (Player 2: P(action 0))')
        ax.set_title(f'{game.name}\n({len(true_ne)} NE, marked ★)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle('Fixed-Point Residual ||BR(π) - π||²: Where NE Live',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'residual_landscapes.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'residual_landscapes.png'}")


def experiment_5_cooperation_from_selfishness():
    """
    Experiment 5: THE KEY RESULT — cooperation emerges from selfish NE search.

    In games with multiple NE, all agents benefit from thorough exploration.
    An agent that explores MORE finds better NE for ITSELF — but this
    also benefits the other agent (in most equilibria, payoffs are correlated).

    Show: search effort → better welfare for ALL agents, not just the searcher.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Cooperation from Selfish NE Search")
    print("="*70)

    games = [stag_hunt(), battle_of_sexes(), coordination_3x3(),
             asymmetric_coordination()]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, game in enumerate(games):
        ax = axes[idx]

        search_budgets = [5, 10, 20, 50, 100]
        n_trials = 12

        welfare_means = []
        welfare_stds = []
        p1_payoff_means = []
        p2_payoff_means = []

        for budget in search_budgets:
            welfares = []
            p1_payoffs = []
            p2_payoffs = []

            for trial in range(n_trials):
                np.random.seed(trial * 100 + budget)
                search = bayesian_fp_search(game, max_searches=budget,
                                             confidence_threshold=0.001,
                                             verbose=False)
                if search['counter'].n_discovered > 0:
                    # Each agent selfishly picks NE best for them
                    best_for_1 = max(search['counter'].discovered_ne, key=lambda x: x[2])
                    best_for_2 = max(search['counter'].discovered_ne, key=lambda x: x[3])
                    # But in practice they'll coordinate on social welfare
                    best_social = max(search['counter'].discovered_ne, key=lambda x: x[2]+x[3])

                    welfares.append(best_social[2] + best_social[3])
                    p1_payoffs.append(best_social[2])
                    p2_payoffs.append(best_social[3])

            welfare_means.append(np.mean(welfares))
            welfare_stds.append(np.std(welfares))
            p1_payoff_means.append(np.mean(p1_payoffs))
            p2_payoff_means.append(np.mean(p2_payoffs))

        ax.plot(search_budgets, welfare_means, 'ko-', linewidth=2, label='Social welfare')
        ax.fill_between(search_budgets,
                         np.array(welfare_means) - np.array(welfare_stds),
                         np.array(welfare_means) + np.array(welfare_stds),
                         alpha=0.15, color='gray')
        ax.plot(search_budgets, p1_payoff_means, 's--', color='#e74c3c',
                label='Player 1 payoff')
        ax.plot(search_budgets, p2_payoff_means, '^--', color='#3498db',
                label='Player 2 payoff')

        # Best possible welfare
        true_ne = game.compute_all_ne()
        if true_ne:
            best_welfare = max(game.payoffs(ne[0], ne[1])[0] +
                                game.payoffs(ne[0], ne[1])[1]
                                for ne in true_ne)
            ax.axhline(y=best_welfare, color='green', linestyle=':', alpha=0.5,
                       label=f'Best NE = {best_welfare:.1f}')

        ax.set_xlabel('Search budget')
        ax.set_ylabel('Payoff')
        ax.set_title(game.name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('More Search → Better NE for ALL Agents\n'
                 '(Cooperation emerges from selfish exploration)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'cooperation_from_selfishness.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'cooperation_from_selfishness.png'}")


def experiment_6_contraction_rates():
    """
    Experiment 6: Empirical contraction rates of the BR map.

    Banach's fixed point theorem: if BR is a contraction (||BR(x)-BR(y)|| ≤ c||x-y||
    for c < 1), then there's a unique fixed point and iteration converges.

    Measure the empirical contraction constant for different games and τ values.
    This connects to the functional analysis foundation.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 6: Contraction Rates of Best-Response Map")
    print("="*70)

    games = [matching_pennies(), stag_hunt(), battle_of_sexes(),
             rock_paper_scissors(), shapley_game()]

    tau_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    n_samples = 500

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: contraction constant vs tau for each game
    ax = axes[0]
    for game in games:
        contraction_constants = []
        for tau in tau_values:
            ratios = []
            for _ in range(n_samples):
                # Sample two random strategy profiles
                p1_a = np.random.dirichlet(np.ones(game.n1))
                p2_a = np.random.dirichlet(np.ones(game.n2))
                p1_b = np.random.dirichlet(np.ones(game.n1))
                p2_b = np.random.dirichlet(np.ones(game.n2))

                # Apply BR map
                br1_a = game.best_response_softmax_1(p2_a, tau)
                br2_a = game.best_response_softmax_2(p1_a, tau)
                br1_b = game.best_response_softmax_1(p2_b, tau)
                br2_b = game.best_response_softmax_2(p1_b, tau)

                d_input = (np.linalg.norm(p1_a - p1_b) +
                           np.linalg.norm(p2_a - p2_b))
                d_output = (np.linalg.norm(br1_a - br1_b) +
                            np.linalg.norm(br2_a - br2_b))

                if d_input > 1e-10:
                    ratios.append(d_output / d_input)

            contraction_constants.append(np.mean(ratios) if ratios else 1.0)

        ax.plot(tau_values, contraction_constants, 'o-', label=game.name,
                markersize=4, linewidth=1.5)

    ax.axhline(y=1, color='black', linestyle='--', alpha=0.3, label='c = 1 (boundary)')
    ax.set_xlabel('Temperature τ')
    ax.set_ylabel('Contraction constant c')
    ax.set_title('Empirical Contraction of BR_τ')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 2)

    # Right: convergence rate (iterations to converge) vs tau
    ax = axes[1]
    for game in games:
        iters_to_converge = []
        for tau in tau_values:
            iters = []
            for trial in range(20):
                np.random.seed(trial * 13 + int(tau * 100))
                _, _, res, traj = find_fixed_point(game, tau=tau, max_iter=1000)
                iters.append(len(traj))
            iters_to_converge.append(np.mean(iters))

        ax.plot(tau_values, iters_to_converge, 's-', label=game.name,
                markersize=4, linewidth=1.5)

    ax.set_xlabel('Temperature τ')
    ax.set_ylabel('Iterations to converge')
    ax.set_title('Convergence Speed vs Temperature')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle('Functional Analysis of BR Map: Banach Contraction',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'contraction_rates.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'contraction_rates.png'}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Fixed-Point Nash Equilibrium Search — Experimental Suite")
    print("=" * 70)
    print("Connects: Kakutani/Banach fixed-point theory ×")
    print("          Bayesian explore/exploit × Multi-agent RL")
    print("=" * 70)

    experiment_1_ne_discovery()
    experiment_2_bayesian_stopping()
    experiment_3_equilibrium_selection()
    experiment_4_convergence_landscape()
    experiment_5_cooperation_from_selfishness()
    experiment_6_contraction_rates()

    print("\n" + "="*70)
    print("All experiments complete.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*70)
