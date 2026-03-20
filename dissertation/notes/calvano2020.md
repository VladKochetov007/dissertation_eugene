# Reading Notes: Calvano et al. (2020) — Artificial Intelligence, Algorithmic Pricing, and Collusion

**Paper:** Artificial Intelligence, Algorithmic Pricing, and Collusion
**Authors:** Emilio Calvano, Giacomo Calzolari, Vincenzo Denicolò, Sergio Pastorello
**Published:** American Economic Review, 2020
**URL:** https://doi.org/10.1257/aer.20190623

## Key Claims

- Q-learning pricing algorithms can learn to charge supra-competitive prices (collude) WITHOUT being explicitly programmed to do so
- This "algorithmic collusion" emerges from independent Q-learning agents in a repeated Bertrand competition
- The collusive strategies exhibit key features of human collusion: punishment phases following deviations
- Raises major antitrust/competition policy concerns — collusion can emerge from standard ML algorithms

## Mathematical Setup

### Bertrand Competition
- N firms (typically N=2) selling differentiated products
- Each firm i sets price p_i simultaneously
- Demand for firm i: q_i(p_i, p_{-i}) — depends on own and competitor prices
- Profit: π_i = (p_i - c_i) · q_i(p_i, p_{-i}) where c_i is marginal cost

### Nash Equilibrium vs Collusion
- **Nash (competitive):** Each firm maximises own profit taking competitor price as given → lower prices
- **Collusive (monopoly):** Firms jointly maximise total profit → higher prices, lower consumer welfare
- **Supra-competitive:** Prices above Nash but possibly below full monopoly

### Q-Learning Setup
- State: discretised price history (e.g., last period's prices)
- Action: choose price from discrete grid
- Reward: period profit π_i
- Each agent runs tabular Q-learning independently
- Exploration: ε-greedy with decaying ε

## Key Results

- Agents converge to supra-competitive prices significantly above Nash equilibrium
- The learned strategies include PUNISHMENT MECHANISMS: if one agent deviates to a lower price, the other retaliates with lower prices for several periods before returning to the collusive price
- This punishment-and-forgiveness pattern resembles human cartel behaviour
- Collusion emerges across a range of parameter settings (discount factors, learning rates, exploration rates)
- More agents → less collusion (harder to sustain), consistent with economic theory

## Connection to Meta-MAPG / Dissertation

### Why This Paper Matters
- Demonstrates that multi-agent learning dynamics can produce EMERGENT cooperation/collusion
- The agents don't explicitly model each other (independent Q-learners) — yet coordination emerges
- Meta-MAPG (Kim et al.) provides the theoretical framework for UNDERSTANDING this:
  - Term 2 (own future learning anticipation) = each agent learning to set prices that lead to good future Q-values
  - Term 3 (peer learning anticipation) = if agents could model each other's learning, collusion could be even more targeted

### Bertrand Competition as Test Environment
- From prelim presentation: Bertrand competition is a candidate simulation environment
- Can compare: independent Q-learning (Calvano) vs LOLA vs Meta-MAPG in same setting
- Prediction: Meta-MAPG agents should collude MORE EFFECTIVELY (explicitly model peer learning)
- This has implications for antitrust: more sophisticated algorithms → more robust collusion

### LLM Steering Connection
- Algorithmic collusion in pricing ↔ cooperative strategies in LLM steering
- If the LLM and hypernetwork are modelled as cooperative agents (Ch.9), Meta-MAPG should produce better cooperation than independent learning
- The "collusion" in steering is DESIRABLE — we WANT the agents to cooperate

## Connections to Dissertation

- **Ch.2**: Literature review — algorithmic collusion as motivation for studying MARL
- **Ch.5**: Calvano's independent Q-learners as baseline that LOLA and Meta-MAPG improve upon
- **Ch.9**: Cooperative steering game uses similar game structure (two agents, repeated interaction)
- **Ch.10**: Reproduce Calvano's results as baseline, then compare with Meta-MAPG agents
- **Prelim**: Featured prominently in prelim presentation as motivating application

## Questions / Gaps

- [ ] Exact Bertrand competition parameterisation used by Calvano — need to replicate
- [ ] How does their Q-learning discretisation work? Grid resolution effects?
- [ ] Can we replace Q-learning with policy gradient methods and get similar collusion?
- [ ] What happens with Meta-MAPG agents in the same setting? More/less collusion?
- [ ] Connection to competition law — is algorithmic collusion legally actionable?
- [ ] Demand function specification: logit demand model details

## Relevant Equations

Logit demand (standard IO model):
q_i = exp((a_i - p_i) / μ) / [Σ_j exp((a_j - p_j) / μ) + exp(a_0 / μ)]

where a_i is product quality, μ is differentiation parameter, a_0 is outside option.

Firm profit:
π_i = (p_i - c) · q_i(p_i, p_{-i})

Nash equilibrium condition:
∂π_i/∂p_i = 0 for all i (simultaneously)

Q-learning update:
Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') - Q(s,a)]
