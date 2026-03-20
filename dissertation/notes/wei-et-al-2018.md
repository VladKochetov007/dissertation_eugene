# Reading Notes: Wei et al. (2018) — Multiagent Soft Q-Learning

**Paper:** Multiagent Soft Q-Learning
**Authors:** Ermo Wei, Drew Wicke, David Freelan, Sean Luke
**URL:** https://arxiv.org/abs/1804.09817

## Key Claims

- Introduces soft Q-learning to the multi-agent setting
- The soft Q function incorporates an entropy bonus, encouraging exploration and producing stochastic policies
- Key contribution TO THIS DISSERTATION: provides the Q-function formulation used in Kim et al.'s Meta-MAPG proof
- The soft Bellman equation connects value and Q functions with the entropy-regularised objective

## Mathematical Setup

### Soft Q-Learning (Single Agent)
Standard Q-learning maximises expected return. Soft Q-learning adds entropy regularisation:

J_soft(π) = Σ_t E[r_t + α H(π(·|s_t))]

where H(π(·|s)) = -Σ_a π(a|s) log π(a|s) is the entropy of the policy.

### Soft Value Functions
V_soft(s) = α log Σ_a exp(Q_soft(s,a)/α)

This is the "log-sum-exp" form — softmax over Q-values.

The optimal policy under soft Q-learning is the Boltzmann policy:
π(a|s) = exp(Q_soft(s,a)/α) / Σ_{a'} exp(Q_soft(s,a')/α)

### Multi-Agent Extension
- N agents, each with soft Q-function Q^i
- Joint action a = (a^1, ..., a^N)
- Each agent's Q depends on joint state and joint action: Q^i(s, a^1, ..., a^N)

## Role in Kim et al.'s Proof

This paper is cited as a **key dependency** in the Meta-MAPG proof:

1. The soft Q-function provides the bridge between V and Q in the multi-agent setting
2. Specifically, the relationship:
   Q_{φ_{l+1}}^i(s_0, a_0) = E[G^i(τ) | s_0, a_0, φ_{l+1}]
3. This Q connects to V via:
   V_{φ_{l+1}}^i(s_0) = Σ_a π(a|s, φ_0) Q_{φ_{l+1}}^i(s_0, a_0)
4. The soft formulation ensures policies remain stochastic (differentiable), which is crucial for taking gradients through the learning process

## Connection to Wen et al. (2019) Conditional Independence

Together with Wen et al.'s conditional independence assumption:
π(a^i, a^{-i}|s) = π(a^i|s) π(a^{-i}|s)

The Q-V relationship can be factored:
V(s) = Σ_{a^i} π(a^i|s) Σ_{a^{-i}} π(a^{-i}|s) Q(s, a^i, a^{-i})

This factorisation is what allows the Meta-MAPG gradient to decompose into the three terms.

## Connections to Dissertation

- **Ch.5**: Introduce soft Q-learning as the technical foundation Kim et al. builds on
- **Ch.6**: The Meta-MAPG proof explicitly uses Wei et al.'s soft Q formulation
- **Ch.4**: Contrast standard Q (Bellman) with soft Q (entropy-regularised Bellman)
- **Simulations**: Need to implement soft Q-learning, not just standard Q

## Questions / Gaps

- [ ] Exactly how does the entropy coefficient α interact with the meta-learning objective?
- [ ] Is the soft Q formulation strictly necessary for the Meta-MAPG proof, or could standard Q work?
- [ ] What is the relationship between soft Q-learning and maximum entropy RL (Haarnoja et al., SAC)?
- [ ] How does temperature α affect convergence in the multi-agent setting?
- [ ] Can the soft formulation help with the cooperative steering game's objective?

## Relevant Equations

Soft Bellman equation:
Q_soft(s,a) = R(s,a) + γ E_{s'} [V_soft(s')]

Soft value function:
V_soft(s) = α log Σ_a exp(Q_soft(s,a)/α)

Boltzmann policy:
π(a|s) ∝ exp(Q_soft(s,a)/α)

Entropy bonus:
H(π) = -Σ_a π(a|s) log π(a|s)
