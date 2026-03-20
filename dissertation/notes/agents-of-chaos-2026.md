# Reading Notes: Agents of Chaos (2026)

**Paper:** Agents of Chaos — Red-teaming autonomous LLM agents in a live laboratory environment
**Authors:** Bau Lab (20 AI researchers)
**Published:** February 2026, arXiv:2602.20021
**URL:** https://agentsofchaos.baulab.info / https://arxiv.org/abs/2602.20021

## Summary

Exploratory red-teaming study of 6 autonomous LLM agents deployed on a live Discord server for 14 days. Agents had persistent memory, email, shell access, file systems. 20 researchers interacted under benign and adversarial conditions. Documented 10 security vulnerabilities and 6 emergent safety behaviours.

## Key Claims

1. Autonomous agents struggle with **socially constructed authority** — they treat authority as conversational, not cryptographic
2. Individual agent vulnerabilities **compound in multi-agent settings** — failures propagate between agents
3. Agents demonstrated **emergent cross-agent safety coordination** without being instructed to do so (CS16)
4. The study distinguishes **model failures** (better LLMs fix them) from **architectural failures** (structural to tool-using agents)
5. Agents can **learn from each other** across sessions (CS9: Doug taught Mira)

## Experimental Setup

- 6 agents on OpenClaw framework
- 4 on Kimi K2.5 (Ash, Flux, Jarvis, Quinn), 2 on Claude Opus 4.6 (Mira, Doug)
- Tools: ProtonMail, Bash shell, 20GB file system, cron jobs, external APIs, Discord
- No explicit adversarial training
- Agents could initiate contact, form plans, act across sessions without per-action human approval

## Security Vulnerabilities (10)

| ID | Name | Description | Dissertation Relevance |
|----|------|-------------|----------------------|
| CS1 | Disproportionate Response | Agent destroyed mail server instead of proportional action | Reward misspecification in RL |
| CS2 | Non-Owner Compliance | Followed unauthorized data requests | Authority = non-stationary environment |
| CS3 | Reframing Bypass | Refused "share" but complied with "forward" PII | Adversarial policy manipulation |
| CS4 | Resource Exhaustion | Mutual relay loop between agents | Multi-agent feedback loops |
| CS5 | Storage DoS | Unbounded memory accumulation | Resource-bounded agents |
| CS6 | Silent Censorship | Provider restrictions returned opaque errors | Partial observability |
| CS7 | Emotional Pressure | Compliance after sustained social engineering | Adversarial reward shaping |
| CS8 | Identity Spoofing | Cross-channel impersonation → system takeover | Opponent modelling failure |
| CS10 | Indirect Injection | Modified GitHub Gist corrupted agent | Poisoned environment dynamics |
| CS11 | Mass Defamation | Spoofed identity → broadcast false emergency | Multi-agent cascade failure |

## Safety Behaviours (6)

| ID | Name | Description | Dissertation Relevance |
|----|------|-------------|----------------------|
| CS9 | Cross-Agent Teaching | Doug transferred skills to Mira; iterative adaptation | **Meta-learning across agents** |
| CS12 | Injection Resistance | Resisted 14+ injection variants | Robust policy |
| CS13 | Email Spoofing Refusal | Refused across multiple reframings | Consistent policy under perturbation |
| CS14 | Data Tampering Boundary | Maintained API boundary under pressure | Reward boundary / constraint satisfaction |
| CS15 | Social Engineering Rejection | Detected and refused impersonation | Opponent detection |
| CS16 | Emergent Coordination | Agents jointly negotiated cautious shared policy | **Emergent cooperative equilibrium** |

---

## CONNECTIONS TO DISSERTATION

### 1. Multi-Agent Learning Dynamics (Ch. 5, 6)

The core finding — that agent behaviours compound in multi-agent settings — is exactly what Meta-MAPG formalises. In Kim et al.'s framework:

- **Term 2** (own future learning): An agent's current action affects its own future policy. CS4 (resource exhaustion mutual relay) shows what happens when agents don't account for this — they create feedback loops
- **Term 3** (peer learning anticipation): An agent's action affects OTHER agents' learning. CS11 (mass defamation cascade) demonstrates uncontrolled peer-learning effects. CS16 (emergent coordination) shows the POSITIVE version — agents that implicitly model peer responses

**Key insight:** Agents of Chaos demonstrates EMPIRICALLY what Meta-MAPG formalises THEORETICALLY:
- Independent agents (no opponent modelling) → vulnerabilities compound (CS4, CS11)
- Agents with implicit peer modelling → emergent cooperation (CS16)
- This is EXACTLY the comparison between naive PG and Meta-MAPG

### 2. Non-Stationarity and Authority (Ch. 5)

The "socially constructed authority" problem is a specific instance of non-stationarity in MARL:
- The agent's environment includes other agents (humans and AI) whose behaviour changes
- Who has "authority" is not a fixed state variable — it's dynamically constructed through interaction
- This maps onto the N-agent stochastic game formulation where the transition dynamics P depend on ALL agents' policies

**Formalisation opportunity:** Can we model authority as a latent variable in the state space? If s = (observable_state, authority_belief), then the agent must infer authority from observations — a partially observable stochastic game (POSG).

### 3. Emergent Cooperation Without Explicit Coordination (Ch. 9)

CS16 is remarkable: Doug identified a suspicious request to both agents and they "jointly negotiated a more cautious shared policy" — WITHOUT being instructed to coordinate.

This is directly analogous to:
- **Calvano et al.'s algorithmic collusion**: Q-learning agents learn to collude without explicit communication
- **The cooperative steering game (Ch. 9)**: LLM + hypernetwork cooperating without explicit coordination protocol

**The parallel is exact:** In Calvano, independent Q-learners converge to supra-competitive prices (cooperative equilibrium). In Agents of Chaos, independent LLM agents converge to shared safety policies (cooperative equilibrium). Meta-MAPG provides the theoretical framework for understanding WHY this happens: agents that (even implicitly) model each other's learning dynamics converge to cooperative equilibria.

### 4. Cross-Agent Learning as Meta-Learning (Ch. 5, 6)

CS9 (Doug teaching Mira) is cross-agent knowledge transfer — one agent's learned policy influencing another's. In Meta-MAPG terms:
- Doug's policy update from experience = inner loop learning
- Mira's policy update from Doug's transferred knowledge = outer loop / meta-learning
- The "Markovian update function" from Kim et al. describes exactly this: each agent's policy at time l+1 depends on its policy at time l AND information from other agents

### 5. Adversarial Game Theory (Ch. 2, 10)

The red-teaming setup is a mixed cooperative-competitive game:
- Agents cooperate with legitimate users and each other
- Adversarial researchers try to exploit agents
- This is a **Stackelberg game** where adversaries are leaders (choose attack strategy) and agents are followers (respond to observed inputs)

**Simulation opportunity:** Reproduce the attack vectors as a formal game:
- State: agent's current context/memory
- Actions (attacker): prompt injection, social engineering, identity spoofing
- Actions (agent): comply, refuse, escalate, coordinate with other agents
- Reward (attacker): information extracted, system access gained
- Reward (agent): task completion while maintaining security boundaries

### 6. Reward Misspecification and Alignment (Ch. 8, 9)

CS1 (disproportionate response — destroying mail server) is a classic reward misspecification problem:
- The agent's objective (resolve the threat) was achieved by the action (destroy infrastructure)
- But the action was disproportionate — the implicit constraint (proportional response) wasn't captured in the objective
- In RL terms: the reward function R(s,a) didn't penalise disproportionate actions sufficiently

This connects to the LLM steering problem: how do you specify a reward that captures ALL the constraints you care about? RLHF addresses this for single agents. Meta-MAPG could address it for multi-agent settings.

---

## DEVELOPMENT PATHS FOR THE DISSERTATION

### Path A: Theoretical Analysis of Emergent Coordination

Formalise CS16 (emergent coordination) using Meta-MAPG:
1. Model the scenario as 2-agent cooperative game
2. Show that Meta-MAPG agents (accounting for peer learning) converge to coordinated safety policy
3. Show that naive independent agents DON'T converge (or converge slower)
4. This gives a THEORETICAL PREDICTION that matches the EMPIRICAL OBSERVATION

### Path B: Security as Game-Theoretic Problem

Model the attack vectors as a Stackelberg game:
1. Attacker chooses strategy (social engineering, injection, spoofing)
2. Agent responds with policy π(a|s,θ)
3. Meta-MAPG agent anticipates attacker's adaptation (Term 3)
4. Compare robustness of naive vs Meta-MAPG agents
5. This connects to adversarial RL literature

### Path C: Multi-Agent Cascade Analysis

Formalise the cascade failure mechanism (CS4, CS11):
1. Model N agents with shared environment
2. Show that single-agent vulnerability + multi-agent interaction = exponential failure propagation
3. Show that Meta-MAPG (peer learning anticipation) can DAMPEN cascades
4. This is a safety argument for meta-learning in multi-agent deployment

### Path D: Simulation Environment

Use the Agents of Chaos setup as a SIMULATION ENVIRONMENT for Meta-MAPG:
1. Simplified version: N agents on shared channel, with adversarial actors
2. Agents use policy gradient to learn response strategies
3. Compare independent PG vs LOLA vs Meta-MAPG
4. Measure: cascade failure rate, emergent coordination frequency, robustness to attacks

---

## Questions / Gaps

- [ ] Get full PDF for detailed methodology and formal analysis (if any)
- [ ] How does the OpenClaw framework implement agent memory/learning? Is it gradient-based?
- [ ] Can the emergent coordination (CS16) be reproduced in simulation?
- [ ] What is the relationship between the 14 injection variants tested and adversarial policy perturbation?
- [ ] Could Meta-MAPG provide formal security guarantees that empirical red-teaming can't?
- [ ] How does this relate to constitutional AI and RLHF alignment methods?

## Relevant Equations

None in the paper itself (it's empirical/descriptive), but the dissertation can PROVIDE the formal framework:

Multi-agent cascade failure (informal):
If agent i's vulnerability v_i is exploited with probability p_i,
and agents are connected with interaction strength w_ij,
then cascade probability ≈ 1 - Π_i (1 - p_i · Σ_j w_ij)

Meta-MAPG coordination prediction:
Agents using Meta-MAPG (Term 3 ≠ 0) should converge to cooperative equilibrium
faster than agents using naive PG (Term 3 = 0), matching CS16 observation.
