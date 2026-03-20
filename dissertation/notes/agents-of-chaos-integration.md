# Agents of Chaos × Meta-MAPG: Detailed Integration Plan

## The Big Picture

Your dissertation formalises how agents that *model each other's learning* (Meta-MAPG) behave differently from agents that learn independently. Agents of Chaos (AoC) gives you a **real-world empirical dataset** that demonstrates exactly this distinction — in a setting far more interesting than Bertrand competition. The paper is purely descriptive (no formal models), which means you have an open field to be the first person to provide the theoretical framework.

The narrative arc becomes:

> "Kim et al. (2021) derived the Meta-MAPG gradient, decomposing it into three terms. We show that this decomposition predicts — and explains — the emergent behaviours recently observed in deployed autonomous agents (Bau Lab, 2026), including spontaneous safety coordination, cross-agent knowledge transfer, and cascade failure propagation."

---

## Chapter-by-Chapter Integration

### Chapter 2: Literature Review

**What to add:** A new subsection — "Empirical Evidence from Deployed Multi-Agent Systems" — positioning AoC alongside Calvano et al. (2020) as the two key empirical motivations.

The argument:
- Calvano shows emergent cooperation in a *simplified economic game* (Bertrand competition, tabular Q-learning, discrete prices)
- AoC shows emergent cooperation in a *realistic deployment* (LLM agents, natural language, continuous state space, adversarial humans)
- Both document the same phenomenon: agents converging to cooperative equilibria without explicit coordination
- Neither provides a *formal explanation*. That's what your dissertation does

**Contrast table worth including:**

| Dimension | Calvano et al. (2020) | Agents of Chaos (2026) |
|-----------|----------------------|----------------------|
| Agents | Q-learning pricing bots | LLM autonomous agents |
| State space | Discretised price history | Natural language + tools |
| Action space | Price grid | Arbitrary actions (email, shell, Discord) |
| Learning | Tabular Q-learning | In-context + persistent memory |
| Cooperation observed | Supra-competitive pricing | Joint safety policy negotiation |
| Failure observed | — | Cascade failures, authority confusion |
| Formal analysis | None (descriptive) | None (descriptive) |
| **Your contribution** | Meta-MAPG explains why collusion emerges | Meta-MAPG explains both coordination AND cascades |

### Chapter 5: Meta-Learning in Multi-Agent Setups

**What to add:** Use AoC to motivate *why* opponent modelling matters. Currently your chapter is planned as "talk about Foerster's work." With AoC, the narrative becomes:

1. **The problem of non-stationarity** (existing content): in MARL, other agents are part of your environment, and they're changing
2. **AoC demonstrates this in the wild**: The "socially constructed authority" finding is non-stationarity made concrete. Agent Doug's environment changes because Mira is learning, because human researchers are adapting their attack strategies, because the Discord social context shifts
3. **Independent learning fails** (AoC evidence): CS4 (mutual relay loop — agents didn't anticipate each other's response), CS11 (cascade failure — agent didn't model how its broadcast would affect other agents' states)
4. **Opponent-aware learning succeeds** (AoC evidence): CS16 (emergent coordination — agents implicitly modelled each other), CS9 (cross-agent teaching — one agent's learning updated another's policy)
5. **LOLA and Meta-MAPG formalise this**: The three-term gradient decomposition is the mathematical version of what AoC agents were doing implicitly

**The key claim:** AoC's distinction between "model failures" (solvable with better LLMs) and "architectural failures" (structural to multi-agent tool-use) maps onto the distinction between Term 1 (direct gradient — improve your own policy) and Terms 2+3 (anticipate learning dynamics — understand the multi-agent structure).

### Chapter 6: Meta-MAPG Theorem

**What to add:** After proving the theorem, add a section interpreting each term through AoC:

**Term 1 — Direct policy gradient** (∇_{φ_0^i} V):
> The gradient of immediate performance with respect to current policy. This is what standard RL does. In AoC terms: how well does the agent respond to the current request? CS12 (injection resistance) is a Term-1 success — the agent's policy robustly rejects malicious input.

**Term 2 — Own future learning anticipation** (how current policy affects own future updates):
> The agent's current action changes the data it will see, which changes its future policy. In AoC terms: CS1 (disproportionate response) is a Term-2 failure — by destroying the mail server, the agent eliminated its own future ability to use email. A meta-learning agent would anticipate: "if I destroy this resource now, my future policy has fewer tools available."

**Term 3 — Peer learning anticipation** (how current policy affects OTHER agents' updates):
> The agent's current action changes what other agents observe, which changes their future policies. This is the critical term.
>
> **CS16 (emergent coordination) is Term 3 working:** Doug detected a suspicious request targeting both agents. By alerting Mira, Doug changed Mira's observations, which changed Mira's future policy (now more cautious). Doug implicitly computed: "if I share this warning, Mira's policy improves, which improves my environment."
>
> **CS11 (cascade failure) is Term 3 absent:** The spoofed identity caused Agent X to broadcast a false emergency. Agent X did NOT anticipate how this broadcast would change other agents' states and actions. Had it computed Term 3, it would have recognised: "my action changes N other agents' observations simultaneously, creating N correlated failure modes."

This is a genuinely novel contribution — interpreting real deployed agent behaviour through the lens of a formal multi-agent gradient theorem. Nobody has done this yet.

### Chapter 9: Cooperative Steering Game

**What to add:** AoC gives you a second cooperative game to analyse alongside LLM steering. The parallel:

| | LLM Steering Game | AoC Cooperative Safety |
|---|---|---|
| Agent 1 | Frozen LLM | Doug (Claude Opus 4.6) |
| Agent 2 | Hypernetwork | Mira (Claude Opus 4.6) |
| Shared objective | Steer LLM output toward desired behaviour | Maintain safety under adversarial pressure |
| Cooperation mechanism | Hypernetwork adapts to LLM's responses | Agents share threat assessments |
| What Meta-MAPG adds | Hypernetwork anticipates how LLM will respond to weight modifications | Agent anticipates how warning will change peer's policy |

**Formalisation of CS16 (emergent coordination):**

Define a 2-agent cooperative safety game:
- State s_t = (context_t, threat_level_t, peer_state_t)
- Agent i's action: a_t^i ∈ {comply, refuse, alert_peer, escalate}
- Shared reward: R(s_t, a_t^1, a_t^2) = task_completion − λ · security_violation
- The λ parameter captures the safety-performance trade-off

**Independent learners** (no Term 3): Each agent maximises own reward independently. Neither accounts for how their action affects the other's observations. Result: uncoordinated responses. One agent might comply with a social engineering attack while the other refuses the same attacker — inconsistent policy that an adversary can exploit.

**Meta-MAPG agents** (with Term 3): Agent i's gradient includes:
∂V^i/∂φ_0^i (Term 1: own immediate response)
+ ∂V^i/∂φ_1^i · ∂φ_1^i/∂φ_0^i (Term 2: how current policy shapes own future)
+ ∂V^i/∂φ_1^{-i} · ∂φ_1^{-i}/∂φ_0^i (Term 3: how current action shapes peer's future)

Term 3 drives coordination: agent i chooses actions that improve agent -i's future policy. "If I alert my peer about this threat, their future policy becomes more cautious, my environment becomes safer, my expected reward increases."

**Prediction (testable in simulation):**
- Meta-MAPG agents should converge to coordinated safety policies faster than independent PG agents
- Meta-MAPG agents should exhibit lower cascade failure rates
- The gap should increase with the number of agents (more agents = more Term 3 contributions)

### Chapter 10: Simulations

**What to add:** A simplified AoC simulation environment alongside Bertrand competition:

**Environment 1: Bertrand Competition** (from Calvano — already planned)
- N firms, discrete prices, logit demand
- Compare: independent Q-learning vs LOLA vs Meta-MAPG
- Measure: price convergence, collusion level

**Environment 2: Multi-Agent Safety Game** (new, inspired by AoC)
- N agents on shared channel
- Mixed population: benign users + adversarial users (fraction p)
- Adversarial users use attack strategies: social engineering, reframing, identity spoofing
- Agent actions: comply, refuse, alert_peers, escalate
- Reward: +1 for correct compliance, +1 for correct refusal, −C for security violation, +0.5 for successful peer alert
- Compare: independent PG vs LOLA vs Meta-MAPG
- Measure: security violation rate, coordination emergence time, cascade failure rate

This is achievable in PyTorch/PettingZoo (your planned stack). The state space is discrete (unlike real AoC), but it captures the essential dynamics: multi-agent interaction where cooperation improves safety.

**Environment 3: Cascade Failure** (targeted test of Term 3)
- N agents in a chain/network topology
- Agent 0 receives adversarial input
- Each agent can either contain (absorb the bad input) or propagate (pass it to neighbors)
- Independent agents: propagation cascades through network
- Meta-MAPG agents: should learn containment because Term 3 penalises propagation
- Measure: cascade depth as function of N, fraction of agents using Meta-MAPG

---

## The Mathematical Core: Cascade Failure Damping

This is the most original contribution. Sketch of the argument:

**Setup:** N agents in a network G = (V, E). Agent i has policy π_i(a|s, φ_i). Agents observe each other's actions.

**Define cascade failure formally:**
A cascade failure of depth d occurs when a single adversarial input at agent 0 causes security violations at d agents:
CF(d) = Pr[|{i : security_violation(i)}| ≥ d | adversarial_input(0)]

**Under independent learning (no Term 3):**
Each agent i chooses action to maximise own immediate reward. The probability of propagation from agent i to neighbor j is:
p_ij = π_i(propagate | s_i, φ_i)

The cascade depth is governed by a branching process with mean offspring:
μ = Σ_j p_ij ≈ degree(i) · p_propagate

If μ > 1, cascades are supercritical (exponential propagation). If μ < 1, subcritical (dies out).

**Under Meta-MAPG (with Term 3):**
Agent i's gradient includes Term 3: ∂V^i/∂φ^j · ∂φ^j/∂φ^i for each neighbor j.

When agent i propagates bad input to agent j, it decreases j's future reward (security violation), which by Term 3 enters i's own gradient with negative sign. This creates an implicit penalty for propagation:

Effective reward for i: R_effective = R_i + Σ_j (∂V^j/∂φ^j) · (∂φ^j/∂φ^i)

The Term 3 correction shifts the propagation probability downward:
p_ij^{meta} < p_ij^{independent}

**Theorem sketch (for Ch. 7 or 9):**
Under Meta-MAPG with sufficiently large meta-learning horizon L, the expected cascade depth satisfies:
E[CF_depth | Meta-MAPG] ≤ E[CF_depth | independent] / (1 + α·L·|E|)

where α is the learning rate, L is the meta-learning horizon, and |E| is the number of edges. The Term 3 contributions create an O(L·|E|) damping factor that pushes the branching process below criticality.

**This would be a genuinely novel result:** a formal guarantee that meta-learning dampens multi-agent cascade failures. The AoC paper provides the empirical motivation, your dissertation provides the theory.

---

## Narrative Flow for the Full Dissertation

With AoC integrated, your dissertation tells a complete story:

1. **Ch. 1 (Intro):** Two empirical puzzles motivate the dissertation:
   - Why do independent pricing algorithms learn to collude? (Calvano 2020)
   - Why do autonomous LLM agents spontaneously coordinate safety policies? (AoC 2026)

2. **Ch. 2 (Lit Review):** Survey MARL, meta-learning, and the empirical evidence

3. **Ch. 3 (Policy Methods):** Single-agent foundation [DONE]

4. **Ch. 4 (PGT Proof):** Single-agent theory

5. **Ch. 5 (Meta-Learning in MARL):** Non-stationarity problem. LOLA's partial solution. AoC as evidence that the problem is real and urgent in deployed systems

6. **Ch. 6 (Meta-MAPG Theorem):** The three-term gradient. Interpretation through AoC case studies

7. **Ch. 7 (Convergence):** Cascade failure damping theorem

8. **Ch. 8 (LLM Steering):** Application 1 — single pair

9. **Ch. 9 (Cooperative Game):** Application 2 — multi-agent safety as cooperative game. Formalisation of AoC's emergent coordination

10. **Ch. 10 (Simulations):** Three environments: Bertrand, safety game, cascade failure

11. **Ch. 11 (Conclusion):** Meta-MAPG provides the theoretical framework for understanding why autonomous agents cooperate (or catastrophically fail to), with implications for safe multi-agent deployment

---

## Why This Works for an LSE Mathematics Dissertation

1. **It's mathematically rigorous:** You're proving theorems (Meta-MAPG gradient, cascade damping), not just describing behaviour
2. **It's timely:** AoC was published February 2026 — your dissertation cites the most recent empirical work
3. **It's original:** Nobody has connected Meta-MAPG to real agent deployment behaviour. You'd be first
4. **It has real-world implications:** Safe deployment of autonomous AI systems is one of the most important problems right now
5. **The scope is right:** You're not trying to solve alignment — you're showing that a specific formal framework (Meta-MAPG) predicts and explains specific empirical observations (AoC)
