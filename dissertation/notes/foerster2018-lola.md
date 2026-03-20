# Reading Notes: Foerster et al. (2018) — LOLA

**Paper:** Learning with Opponent-Learning Awareness (LOLA)
**Authors:** Jakob Foerster, Richard Y. Chen, Maruan Al-Shedivat, Shimon Whiteson, Pieter Abbeel, Igor Mordatch
**URL:** https://arxiv.org/abs/1709.04326

## Key Claims

- Standard independent learners in MARL ignore the fact that other agents are also learning — they treat opponents as part of a stationary environment
- LOLA agents account for the anticipated parameter update of the OTHER agent when computing their own gradient
- This "opponent-learning awareness" leads to emergent cooperation in settings like the Iterated Prisoner's Dilemma (IPD) where naive learners defect
- LOLA is the direct predecessor to Kim et al.'s Meta-MAPG — it introduces the idea of differentiating through the opponent's learning step
- Key limitation: LOLA uses a first-order Taylor expansion (linear approximation) of the opponent's learning step

## Mathematical Setup

- Two-agent setting (i and -i)
- Each agent has policy parameters θ_i
- Standard policy gradient: each agent maximises own return J^i(θ_i, θ_{-i})
- Naive learner gradient: ∇_{θ_i} J^i(θ_i, θ_{-i})

### LOLA Gradient
Instead of treating θ_{-i} as fixed, LOLA anticipates that agent -i will take a gradient step:
θ_{-i}' = θ_{-i} + Δθ_{-i}

where Δθ_{-i} = α ∇_{θ_{-i}} J^{-i}(θ_i, θ_{-i})

LOLA agent i then optimises:
∇_{θ_i} J^i(θ_i, θ_{-i} + Δθ_{-i})

Expanding via chain rule:
∇_{θ_i} J^i + (∇_{θ_{-i}} J^i) · (∂Δθ_{-i}/∂θ_i)

This gives TWO terms:
1. Direct gradient: ∇_{θ_i} J^i — standard policy gradient
2. Opponent-anticipation term: accounts for how i's parameters affect -i's update, which then affects i's return

### Higher-Order LOLA
- Can recurse: what if -i also does LOLA? → LOLA with higher-order awareness
- Leads to infinite regress problem
- Kim et al.'s Meta-MAPG provides the principled resolution via meta-learning

## Connection to Meta-MAPG

- LOLA's two-term gradient is a SPECIAL CASE of Kim et al.'s three-term decomposition
- Kim et al. Term 1 (direct gradient) = LOLA Term 1
- Kim et al. Terms 2+3 (own future learning + peer learning anticipation) generalise LOLA's Term 2
- Kim et al. provides the THEOREM (formal gradient derivation) that LOLA motivated but didn't fully derive
- LOLA assumes one-step lookahead; Meta-MAPG handles multi-step learning dynamics via meta-value functions

## Key Results

- IPD: LOLA agents learn tit-for-tat-like cooperation, while naive learners converge to mutual defection
- This demonstrates that opponent-modelling can shift equilibria toward cooperation
- Relevant to algorithmic collusion: if firms (agents) model each other's learning, they can converge to supra-competitive prices

## Connections to Dissertation

- **Ch.5**: Full exposition of LOLA as bridge between single-agent PG and Meta-MAPG
- **Ch.6**: Meta-MAPG theorem generalises LOLA's gradient to N agents with formal proof
- **Ch.9**: Cooperative steering game — the "opponent-learning awareness" is exactly what makes LLM + hypernetwork cooperate effectively
- **Ch.10**: Simulations should compare naive PG vs LOLA vs Meta-MAPG in same environments

## Questions / Gaps

- [ ] Exact relationship between LOLA's first-order approximation and Kim et al.'s exact gradient
- [ ] Does LOLA converge to the same fixed points as Meta-MAPG in the two-agent case?
- [ ] How does the DiCE estimator (Foerster et al. 2018, separate paper) relate to the gradient computation?
- [ ] Can LOLA be extended to continuous action spaces naturally?
- [ ] What happens when N > 2 agents all do LOLA? Does the theory extend cleanly?

## Relevant Equations

Naive independent learning:
θ_i ← θ_i + α ∇_{θ_i} J^i(θ_i, θ_{-i})

LOLA update (first order):
θ_i ← θ_i + α [∇_{θ_i} J^i + (∇_{θ_{-i}} J^i)(∇²_{θ_i θ_{-i}} J^{-i})]

The second term requires SECOND-ORDER derivatives — differentiating through the opponent's gradient step.
