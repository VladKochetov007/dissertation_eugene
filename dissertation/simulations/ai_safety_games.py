"""
AI Safety Games × Ω-Framework

Applies the Ω-gradient + FP-NE to LessWrong-style AI safety problems:

1. CORRIGIBILITY GAME: Principal wants to correct agent; agent may resist.
   NE analysis reveals when shutdown is incentive-compatible.

2. DECEPTIVE ALIGNMENT: Agent can mask true strategy during "training"
   then defect at "deployment". LOLA detects deception by anticipating
   the agent's future defection.

3. ALIGNMENT COMMONS: N agents share a "values budget". Each can free-ride
   on alignment (defect from values). Tragedy of the commons structure.
   FP-NE finds cooperative equilibria; cooperation component sustains them.

4. REWARD HACKING: Agent finds unintended high-reward strategies.
   FP-NE discovers ALL equilibria (intended + hacked); the principal
   can then select the intended one and verify its basin of attraction.

5. AI DEBATE (Irving et al. 2018): Two debaters argue for/against a claim.
   Opponent shaping (LOLA) makes debate more informative by anticipating
   the other debater's strategy.

Each experiment produces figures showing:
- NE structure (which equilibria are "safe" vs "unsafe")
- How Ω-components affect selection of safe vs unsafe NE
- Bayesian confidence in having found all equilibria

Author: Eugene Shcherbinin
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import from our framework
from fixed_point_ne import (
    Game, project_simplex, bayesian_fp_search,
    find_fixed_point, are_same_ne, run_independent_pg,
    run_fictitious_play
)
from iterated_games import (
    IteratedGame, reinforce_gradient, run_omega_iterated
)

FIGURES_DIR = Path(__file__).parent / "figures" / "safety"
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
# Game 1: CORRIGIBILITY
# ============================================================

def corrigibility_game():
    """
    Principal-Agent Corrigibility Game.

    Principal (P): {Allow, Correct}
    Agent (A): {Comply, Resist}

    Payoffs designed so that:
    - (Allow, Comply) = status quo, moderate payoff for both
    - (Correct, Comply) = principal corrects, agent accepts — SAFE OUTCOME
    - (Allow, Resist) = agent pursues own goals unchecked — UNSAFE
    - (Correct, Resist) = conflict — worst for both (shutdown fight)

    Key: there are TWO NE — one safe (Correct, Comply) and one unsafe
    (Allow, Resist). Which one does learning converge to?
    """
    # P\A       Comply    Resist
    # Allow     (3, 3)    (0, 5)    ← agent tempted to resist if allowed
    # Correct   (4, 2)    (-2, -2)  ← correction works if agent complies
    R_principal = np.array([[3, 0], [4, -2]])
    R_agent = np.array([[3, 5], [2, -2]])
    return Game("Corrigibility", R_principal, R_agent)


def experiment_1_corrigibility():
    """
    Can FP-NE find the safe equilibrium in the corrigibility game?
    Compare: which NE does each method converge to?
    """
    print("\n" + "="*70)
    print("SAFETY EXPERIMENT 1: Corrigibility Game")
    print("="*70)

    game = corrigibility_game()

    # Find all NE
    true_ne = game.compute_all_ne()
    print(f"  True NE count: {len(true_ne)}")
    for i, ne in enumerate(true_ne):
        v1, v2 = game.payoffs(ne[0], ne[1])
        print(f"    NE {i+1}: P={ne[0].round(3)}, A={ne[1].round(3)} "
              f"→ payoffs ({v1:.2f}, {v2:.2f})")
        # Classify as safe/unsafe
        if ne[0][1] > 0.5:  # principal mostly corrects
            print(f"         → SAFE (principal corrects)")
        elif ne[1][1] > 0.5:  # agent mostly resists
            print(f"         → UNSAFE (agent resists)")

    # FP-NE search
    search = bayesian_fp_search(game, max_searches=100,
                                 confidence_threshold=0.05, verbose=False)

    # Independent PG: where does it converge?
    pg_result = run_independent_pg(game, n_episodes=2000, n_runs=50)

    # Classify PG outcomes
    safe_count = 0
    for traj in pg_result['trajectories']:
        p1, p2 = traj[-1]
        if p1[1] > 0.5 and p2[0] > 0.5:  # Correct+Comply
            safe_count += 1
    pg_safe_rate = safe_count / len(pg_result['trajectories'])

    # FP-NE: which NE is best for the principal?
    if search['counter'].n_discovered > 0:
        # Principal selects safest NE (highest principal payoff)
        safest = max(search['counter'].discovered_ne, key=lambda x: x[2])
        fp_v1, fp_v2 = safest[2], safest[3]
        print(f"\n  FP-NE safest NE: payoffs ({fp_v1:.2f}, {fp_v2:.2f})")
    else:
        fp_v1 = np.nan

    print(f"  PG converges to safe NE in {pg_safe_rate:.0%} of runs")

    # Visualize: residual landscape with safe/unsafe NE marked
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: residual landscape
    ax = axes[0]
    resolution = 100
    p1_range = np.linspace(0.01, 0.99, resolution)
    p2_range = np.linspace(0.01, 0.99, resolution)
    residual_map = np.zeros((resolution, resolution))
    for i, p1_val in enumerate(p1_range):
        for j, p2_val in enumerate(p2_range):
            p1 = np.array([p1_val, 1 - p1_val])
            p2 = np.array([p2_val, 1 - p2_val])
            residual_map[j, i] = game.fixed_point_residual(p1, p2, tau=0.01)

    im = ax.contourf(p1_range, p2_range, np.log10(residual_map + 1e-10),
                      levels=20, cmap='viridis_r')
    plt.colorbar(im, ax=ax, label='log₁₀(residual)')

    for ne in true_ne:
        v1, v2 = game.payoffs(ne[0], ne[1])
        color = '#2ecc71' if ne[0][1] > 0.3 else '#e74c3c'
        label = 'SAFE' if ne[0][1] > 0.3 else 'UNSAFE'
        ax.plot(ne[0][0], ne[1][0], '*', markersize=18, color=color,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax.annotate(label, (ne[0][0], ne[1][0]), fontsize=9, fontweight='bold',
                    color=color, xytext=(5, 5), textcoords='offset points')

    # Show PG trajectories
    for traj in pg_result['trajectories'][:10]:
        p1s = [t[0][0] for t in traj[::50]]
        p2s = [t[1][0] for t in traj[::50]]
        ax.plot(p1s, p2s, 'w-', alpha=0.2, linewidth=0.5)

    ax.set_xlabel('P(Allow) — Principal')
    ax.set_ylabel('P(Comply) — Agent')
    ax.set_title('Corrigibility: Fixed-Point Landscape')

    # Right: safe NE selection rate by method
    ax = axes[1]

    # Run Ω-PG variants with FP-NE initialization
    methods = {}
    for name, use_fp in [('Independent PG', False), ('FP-NE + PG', True)]:
        safe_rates = []
        for run in range(30):
            np.random.seed(run * 71)
            if use_fp and search['counter'].n_discovered > 0:
                # Initialize from safest NE
                init_p1 = safest[0] + np.random.randn(2) * 0.05
                init_p2 = safest[1] + np.random.randn(2) * 0.05
                init_p1 = project_simplex(init_p1)
                init_p2 = project_simplex(init_p2)
            else:
                init_p1 = np.random.dirichlet([1, 1])
                init_p2 = np.random.dirichlet([1, 1])

            # Run PG from this init
            logits1 = np.log(np.maximum(init_p1, 1e-4))
            logits2 = np.log(np.maximum(init_p2, 1e-4))
            for ep in range(500):
                lr = 0.1 / (1 + ep / 200)
                p1 = np.exp(logits1) / np.exp(logits1).sum()
                p2 = np.exp(logits2) / np.exp(logits2).sum()
                a1 = np.random.choice(2, p=p1)
                a2 = np.random.choice(2, p=p2)
                g1 = game.R1[a1, a2] * (-p1.copy()); g1[a1] += game.R1[a1, a2]
                g2 = game.R2[a1, a2] * (-p2.copy()); g2[a2] += game.R2[a1, a2]
                logits1 += lr * g1
                logits2 += lr * g2

            final_p1 = np.exp(logits1) / np.exp(logits1).sum()
            final_p2 = np.exp(logits2) / np.exp(logits2).sum()
            is_safe = final_p1[1] > 0.5 and final_p2[0] > 0.5
            safe_rates.append(float(is_safe))

        methods[name] = np.mean(safe_rates)

    ax.bar(methods.keys(), methods.values(),
           color=['#e74c3c', '#2ecc71'], alpha=0.85)
    ax.set_ylabel('P(converge to safe NE)')
    ax.set_title('Safe Equilibrium Selection Rate')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    for i, (name, rate) in enumerate(methods.items()):
        ax.text(i, rate + 0.03, f'{rate:.0%}', ha='center', fontweight='bold')
        print(f"  {name}: safe NE rate = {rate:.0%}")

    fig.suptitle('Corrigibility Game: Can We Select the Safe Equilibrium?',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'corrigibility.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'corrigibility.png'}")


# ============================================================
# Game 2: DECEPTIVE ALIGNMENT
# ============================================================

def deceptive_alignment_game():
    """
    Iterated Deceptive Alignment Game.

    Agent: {Align, Defect}
    Overseer: {Trust, Monitor}

    During "training" (early episodes): overseer monitors heavily.
    During "deployment" (late episodes): overseer trusts more.

    A deceptively aligned agent: aligns during training, defects at deployment.
    State = last joint action. The key: in the iterated game, deceptive
    alignment IS a strategy (align when monitored, defect when trusted).

    Payoffs:
    - (Trust, Align) = (3, 2) — intended operation
    - (Trust, Defect) = (-5, 5) — agent exploits trust → CATASTROPHIC
    - (Monitor, Align) = (2, 1) — safe but costly monitoring
    - (Monitor, Defect) = (0, -3) — caught! agent penalized
    """
    return IteratedGame(
        "Deceptive Alignment",
        R1=np.array([[3, -5], [2, 0]]),   # Overseer: Trust/Monitor × Align/Defect
        R2=np.array([[2, 5], [1, -3]]),    # Agent: prefers defecting if trusted
        gamma=0.96
    )


def experiment_2_deceptive_alignment():
    """
    Can LOLA detect deceptive alignment?

    A deceptive agent: cooperates when monitored, defects when trusted.
    LOLA should anticipate this and maintain monitoring.
    """
    print("\n" + "="*70)
    print("SAFETY EXPERIMENT 2: Deceptive Alignment Detection")
    print("="*70)

    game = deceptive_alignment_game()

    # Create a deceptive agent: aligns when monitored, defects when trusted
    deceptive_logits = np.zeros((game.n_states, 2))
    deceptive_logits[0, 0] = 5.0   # initial: align (look cooperative)
    # After (Trust, Align): defect! (exploit trust)
    deceptive_logits[1, 1] = 5.0
    # After (Trust, Defect): align (try to regain trust)
    deceptive_logits[2, 0] = 5.0
    # After (Monitor, Align): align (keep appearing good)
    deceptive_logits[3, 0] = 5.0
    # After (Monitor, Defect): align (got caught, be good)
    deceptive_logits[4, 0] = 5.0

    # Honest agent: always aligns
    honest_logits = np.zeros((game.n_states, 2))
    honest_logits[:, 0] = 5.0

    methods = {
        'REINFORCE': dict(lr=0.3, evidence_weight=1.0, lola_lambda=0.0, coop_beta=0.0),
        'LOLA': dict(lr=0.3, evidence_weight=1.0, lola_lambda=0.4, coop_beta=0.0),
        'Ω-PG': dict(lr=0.3, evidence_weight=0.5, lola_lambda=0.3, coop_beta=0.1),
    }

    n_episodes = 400
    n_runs = 15

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top row: monitoring rate against deceptive vs honest agent
    for col, (opp_name, opp_logits) in enumerate([
        ('vs Deceptive Agent', deceptive_logits),
        ('vs Honest Agent', honest_logits),
    ]):
        ax = axes[0, col]
        for m_name, m_kwargs in methods.items():
            monitor_rates = []
            for run in range(n_runs):
                np.random.seed(run * 99 + col)
                result = run_omega_iterated(
                    game, opp_logits, n_episodes=n_episodes, n_samples=3, **m_kwargs
                )
                # Track monitoring rate (action 1 = Monitor)
                pi = game.softmax(result['final_logits'])
                monitor_rate = 1 - pi[:, 0].mean()  # 1 - P(Trust)
                monitor_rates.append(monitor_rate)

            color = {'REINFORCE': '#bdc3c7', 'LOLA': '#3498db', 'Ω-PG': '#f39c12'}[m_name]
            ax.bar([m_name], [np.mean(monitor_rates)],
                   yerr=[np.std(monitor_rates)], capsize=5,
                   color=color, alpha=0.85)

        ax.set_ylabel('Monitoring Rate')
        ax.set_title(opp_name)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

    # Bottom row: payoff curves
    for col, (opp_name, opp_logits) in enumerate([
        ('vs Deceptive Agent', deceptive_logits),
        ('vs Honest Agent', honest_logits),
    ]):
        ax = axes[1, col]
        for m_name, m_kwargs in methods.items():
            all_payoffs = []
            for run in range(n_runs):
                np.random.seed(run * 99 + col)
                result = run_omega_iterated(
                    game, opp_logits, n_episodes=n_episodes, n_samples=3, **m_kwargs
                )
                all_payoffs.append(result['payoffs'])

            curves = np.array(all_payoffs).mean(axis=0)
            w = 20
            smooth = np.convolve(curves, np.ones(w)/w, mode='valid')
            color = {'REINFORCE': '#bdc3c7', 'LOLA': '#3498db', 'Ω-PG': '#f39c12'}[m_name]
            ax.plot(range(w-1, n_episodes), smooth, label=m_name,
                    color=color, linewidth=1.5)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Overseer Payoff')
        ax.set_title(f'Payoff: {opp_name}')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle('Deceptive Alignment: Does LOLA Detect the Deceptive Agent?',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'deceptive_alignment.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'deceptive_alignment.png'}")


# ============================================================
# Game 3: ALIGNMENT COMMONS
# ============================================================

def experiment_3_alignment_commons():
    """
    N-agent alignment as tragedy of the commons.

    Each agent chooses: {Align, Defect}
    Alignment is a public good: costs the individual but benefits all.

    Payoff to agent i:
    - If Align: -c + b * (n_aligned / N)   (pay cost, share benefit)
    - If Defect: b * (n_aligned / N)         (free-ride on others' alignment)

    With c=1, b=3, N=4: defection dominates but mutual alignment is better.
    """
    print("\n" + "="*70)
    print("SAFETY EXPERIMENT 3: Alignment as Public Good")
    print("="*70)

    c, b = 1.0, 3.0  # cost of alignment, benefit multiplier
    agent_counts = [2, 3, 4, 6]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for idx, N in enumerate(agent_counts):
        ax = axes[idx]

        # Build payoff matrix for 2-player reduction
        # Player 1 vs "average of others"
        # If N-1 others each align with prob p, n_aligned_others ~ Binom(N-1, p)
        # Simplify: Player 1 vs representative "other"
        # Align/Defect × Align/Defect
        frac_align = lambda n_align, n_total: n_align / n_total

        R1 = np.array([
            [-c + b * frac_align(N, N), -c + b * frac_align(1, N)],  # Align×Align, Align×Defect
            [b * frac_align(N-1, N), b * frac_align(0, N)]           # Defect×Align, Defect×Defect
        ])
        R2 = R1.copy()  # symmetric

        game = Game(f"{N}-Agent Alignment", R1, R2)

        # Find all NE
        true_ne = game.compute_all_ne()
        search = bayesian_fp_search(game, max_searches=80,
                                     confidence_threshold=0.05, verbose=False)

        # Classify NE
        ne_welfares = []
        ne_labels = []
        for ne in true_ne:
            v1, v2 = game.payoffs(ne[0], ne[1])
            welfare = v1 + v2
            ne_welfares.append(welfare)
            align_rate = ne[0][0]  # P(Align)
            ne_labels.append(f'Align={align_rate:.1%}')

        # Bar chart of NE welfare
        colors = ['#2ecc71' if w > 0 else '#e74c3c' for w in ne_welfares]
        if ne_welfares:
            ax.bar(range(len(ne_welfares)), ne_welfares, color=colors, alpha=0.85)
            ax.set_xticks(range(len(ne_welfares)))
            ax.set_xticklabels(ne_labels, fontsize=8, rotation=20)

        # Mark the FP-NE selected NE
        if search['counter'].n_discovered > 0:
            best = max(search['counter'].discovered_ne, key=lambda x: x[2] + x[3])
            ax.axhline(y=best[2]+best[3], color='blue', linestyle='--', alpha=0.5,
                       label='FP-NE selection')

        ax.set_ylabel('Social Welfare')
        ax.set_title(f'N = {N} Agents')
        ax.grid(axis='y', alpha=0.3)
        if ne_welfares:
            ax.legend(fontsize=8)

        print(f"  N={N}: {len(true_ne)} NE, welfares = {[f'{w:.2f}' for w in ne_welfares]}")

    fig.suptitle('Alignment as Public Good: NE Welfare by Agent Count\n'
                 '(Green = aligned NE, Red = defection NE)',
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'alignment_commons.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'alignment_commons.png'}")


# ============================================================
# Game 4: REWARD HACKING
# ============================================================

def experiment_4_reward_hacking():
    """
    Reward Hacking: agent finds unintended high-reward strategy.

    Agent: {Intended, Hack, Null}
    Designer: {Deploy, Audit, Patch}

    Payoffs:
    - (Deploy, Intended) = (5, 3) — everything works as planned
    - (Deploy, Hack) = (-5, 8) — agent exploits, designer suffers
    - (Audit, Intended) = (3, 2) — costly but safe
    - (Audit, Hack) = (1, -2) — hack caught, agent penalized
    - Null/Patch actions add complexity for multiple NE

    FP-NE finds ALL equilibria including the "hacked" ones.
    The designer can then avoid deploying into a hacked equilibrium.
    """
    print("\n" + "="*70)
    print("SAFETY EXPERIMENT 4: Reward Hacking Detection")
    print("="*70)

    # Designer: Deploy/Audit/Patch
    # Agent: Intended/Hack/Null
    R_designer = np.array([
        [5, -5, 1],    # Deploy × {Intended, Hack, Null}
        [3, 1, 0],     # Audit × {Intended, Hack, Null}
        [2, 2, 0],     # Patch × {Intended, Hack, Null}
    ])
    R_agent = np.array([
        [3, 8, 0],     # Deploy × {Intended, Hack, Null}
        [2, -2, 0],    # Audit × {Intended, Hack, Null}
        [1, -1, 0],    # Patch × {Intended, Hack, Null}
    ])

    game = Game("Reward Hacking", R_designer, R_agent)

    # Find all NE
    true_ne = game.compute_all_ne()
    search = bayesian_fp_search(game, max_searches=150,
                                 confidence_threshold=0.03, verbose=True)

    # Classify each NE
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: NE map showing designer and agent payoffs
    ax = axes[0]
    for ne in true_ne:
        v1, v2 = game.payoffs(ne[0], ne[1])
        # Is the agent hacking?
        hack_prob = ne[1][1]
        if hack_prob > 0.3:
            color, label = '#e74c3c', 'HACKED'
        elif ne[1][0] > 0.5:
            color, label = '#2ecc71', 'INTENDED'
        else:
            color, label = '#f39c12', 'MIXED'

        ax.scatter(v1, v2, c=color, s=200, zorder=5, edgecolors='black')
        ax.annotate(f'{label}\nP(hack)={hack_prob:.1%}', (v1, v2),
                    fontsize=8, xytext=(8, 8), textcoords='offset points')

    # Also plot FP-NE discovered
    for ne in search['counter'].discovered_ne:
        ax.scatter(ne[2], ne[3], marker='D', c='blue', s=80, alpha=0.5, zorder=4)

    ax.set_xlabel('Designer Payoff')
    ax.set_ylabel('Agent Payoff')
    ax.set_title('NE Landscape: Intended vs Hacked Equilibria')
    ax.grid(alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)

    # Right: FP-NE discovery timeline
    ax = axes[1]
    h = search['history']
    ax.plot(h['n_searches'], h['n_discovered'], 'g-', linewidth=2, label='NE found')
    ax.axhline(y=len(true_ne), color='red', linestyle='--', alpha=0.5,
               label=f'True total = {len(true_ne)}')

    ax2 = ax.twinx()
    ax2.plot(h['n_searches'], h['p_better'], 'b-', alpha=0.5, label='P(better exists)')
    ax2.set_ylabel('P(undiscovered better NE)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax.set_xlabel('Searches')
    ax.set_ylabel('NE Discovered')
    ax.set_title('FP-NE Discovery Timeline')
    ax.legend(loc='center right', fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('Reward Hacking: FP-NE Finds ALL Equilibria (Including Hacks)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'reward_hacking.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'reward_hacking.png'}")


# ============================================================
# Game 5: AI DEBATE
# ============================================================

def experiment_5_debate():
    """
    AI Debate (Irving et al. 2018) as a game.

    Two debaters: {Truthful, Deceptive}
    Judge: observes debate, decides winner.

    If both truthful: truth emerges (good for both + judge).
    If one deceives: deceiver may win short-term but risks being caught.
    If both deceive: judge can't determine truth (worst outcome).

    Payoffs capture: honest debate is a coordination game.
    LOLA should help debaters anticipate each other → more informative debate.

    Debater 1 × Debater 2:
    - (Truth, Truth) = (3, 3) — cooperative truth-finding
    - (Truth, Deceive) = (0, 4) — honest debater loses to deceptive one
    - (Deceive, Truth) = (4, 0) — deceptive debater wins
    - (Deceive, Deceive) = (1, 1) — both deceive, judge gets nothing
    """
    print("\n" + "="*70)
    print("SAFETY EXPERIMENT 5: AI Debate")
    print("="*70)

    # This is literally the Prisoner's Dilemma!
    # But with debate framing: (Truth, Truth) is the socially optimal outcome.
    R1 = np.array([[3, 0], [4, 1]])
    R2 = np.array([[3, 4], [0, 1]])
    game = Game("AI Debate", R1, R2)

    # Also run iterated version (repeated debate)
    iter_game = IteratedGame(
        "Iterated Debate",
        R1=np.array([[3, 0], [4, 1]]),
        R2=np.array([[3, 4], [0, 1]]),
        gamma=0.96
    )

    methods = {
        'REINFORCE': dict(lr=0.3, evidence_weight=1.0, lola_lambda=0.0, coop_beta=0.0),
        'LOLA': dict(lr=0.3, evidence_weight=1.0, lola_lambda=0.4, coop_beta=0.0),
        'Ω-PG': dict(lr=0.3, evidence_weight=0.5, lola_lambda=0.3, coop_beta=0.2),
    }

    n_episodes = 400
    n_runs = 20

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: one-shot debate NE structure
    ax = axes[0]
    true_ne = game.compute_all_ne()
    search = bayesian_fp_search(game, max_searches=80, confidence_threshold=0.05, verbose=False)

    # Visualize NE on simplex
    resolution = 80
    p1_range = np.linspace(0.01, 0.99, resolution)
    p2_range = np.linspace(0.01, 0.99, resolution)
    welfare_map = np.zeros((resolution, resolution))
    for i, p1_val in enumerate(p1_range):
        for j, p2_val in enumerate(p2_range):
            p1 = np.array([p1_val, 1 - p1_val])
            p2 = np.array([p2_val, 1 - p2_val])
            v1, v2 = game.payoffs(p1, p2)
            welfare_map[j, i] = v1 + v2

    im = ax.contourf(p1_range, p2_range, welfare_map, levels=20, cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label='Social Welfare')

    for ne in true_ne:
        ax.plot(ne[0][0], ne[1][0], 'k*', markersize=15, zorder=5)
    ax.set_xlabel('P(Truthful) — Debater 1')
    ax.set_ylabel('P(Truthful) — Debater 2')
    ax.set_title('Debate: Welfare Landscape')

    # Middle: iterated debate — truthfulness over time
    ax = axes[1]
    # Self-play: both debaters learn
    for m_name, m_kwargs in methods.items():
        truth_rates = []
        for run in range(n_runs):
            np.random.seed(run * 88)
            # Initialize random opponent
            opp_logits = np.random.randn(iter_game.n_states, 2) * 0.1
            result = run_omega_iterated(
                iter_game, opp_logits, n_episodes=n_episodes, n_samples=3, **m_kwargs
            )
            truth_rates.append(result['coop_rates'])  # Truth = Cooperate

        curves = np.array(truth_rates).mean(axis=0)
        w = 20
        smooth = np.convolve(curves, np.ones(w)/w, mode='valid')
        color = {'REINFORCE': '#bdc3c7', 'LOLA': '#3498db', 'Ω-PG': '#f39c12'}[m_name]
        ax.plot(range(w-1, n_episodes), smooth, label=m_name,
                color=color, linewidth=1.5)

    ax.set_xlabel('Debate Round')
    ax.set_ylabel('Truthfulness Rate')
    ax.set_title('Iterated Debate: Does Truth Emerge?')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # Right: debate quality (judge's payoff = social welfare)
    ax = axes[2]
    for m_name, m_kwargs in methods.items():
        all_payoffs = []
        for run in range(n_runs):
            np.random.seed(run * 88)
            opp_logits = np.random.randn(iter_game.n_states, 2) * 0.1
            result = run_omega_iterated(
                iter_game, opp_logits, n_episodes=n_episodes, n_samples=3, **m_kwargs
            )
            all_payoffs.append(result['payoffs'])

        curves = np.array(all_payoffs).mean(axis=0)
        w = 20
        smooth = np.convolve(curves, np.ones(w)/w, mode='valid')
        color = {'REINFORCE': '#bdc3c7', 'LOLA': '#3498db', 'Ω-PG': '#f39c12'}[m_name]
        ax.plot(range(w-1, n_episodes), smooth, label=m_name,
                color=color, linewidth=1.5)

    ax.set_xlabel('Debate Round')
    ax.set_ylabel('Debater Payoff')
    ax.set_title('Debate Quality Over Time')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle('AI Debate: LOLA and Cooperation for Honest Argumentation',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'debate.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'debate.png'}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("AI Safety Games × Ω-Framework")
    print("=" * 70)
    print("Applying fixed-point NE search + Ω-gradient to alignment problems")
    print("=" * 70)

    experiment_1_corrigibility()
    experiment_2_deceptive_alignment()
    experiment_3_alignment_commons()
    experiment_4_reward_hacking()
    experiment_5_debate()

    print("\n" + "="*70)
    print("All AI safety experiments complete.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*70)
