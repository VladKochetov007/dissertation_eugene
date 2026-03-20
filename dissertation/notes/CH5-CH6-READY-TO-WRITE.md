# Ch.5-6 Meta-Learning in MARL — Ready to Write

**Status:** ✅ Reading complete (Kim 2021), proof extracted, structure ready

**Source:** Kim et al. (2021) — "A Policy Gradient Algorithm for Learning to Learn in Multiagent Reinforcement Learning"  
**Notes:** `~/Coding/maynard/dissertation/notes/kim2021.md`

---

## Overview

**Ch.5** sets up multi-agent RL framework + non-stationarity problem  
**Ch.6** proves Meta-MAPG theorem (extends Ch.4 PGT to multi-agent + meta-learning) — **YOUR PHD-LEVEL CONTRIBUTION!**

---

## Ch.5: Meta-Learning in Multiagent RL

### Structure (7 sections, ~10-12 pages)

**5.1 Introduction** (1 page)
- Multi-agent RL: agents learn simultaneously → environment is non-stationary
- Challenge: each agent's policy changes affect other agents
- Preview: meta-learning as solution

**5.2 Stochastic Games** (2 pages)
- Definition: M_n = ⟨I, S, A, P, R, γ⟩
- Joint policies, joint actions
- Value functions in multi-agent setting
- State distribution ρ_φ(s) under joint policy φ

**5.3 The Non-Stationarity Problem** (2 pages)
- Markov chain of policies representation (Figure 5.1)
- Sequential dependencies: φ_0 → τ_{φ_0} → φ_1 → τ_{φ_1} → ...
- Markovian update functions U^i(τ_φ, φ^i)
- Why standard RL fails: Markov property invalid from each agent's perspective

**5.4 Meta-Learning Framework** (2 pages)
- Meta-objective: optimize for adaptation performance
- Meta-value function V^i_{φ_0:ℓ+1}(s_0, φ^i_0)
- Inner-loop (adapt to current peers) vs outer-loop (learn to adapt)
- Distribution over initial peer policies p(φ^{-i}_0)

**5.5 Prior Work: Meta-PG** (1 page)
- Al-Shedivat et al. 2018
- Assumption: peers' learning independent of meta-agent
- Result: only "own learning" gradient (misses peer influence!)
- Limitation: treats other agents as external factors

**5.6 Prior Work: LOLA** (1 page)
- Foerster et al. 2018
- Approach: influence other agents' learning via first-order Taylor approximation
- Result: "peer learning" gradient (shapes opponent dynamics)
- Limitation: doesn't consider own learning dynamics

**5.7 Summary** (0.5 page)
- MARL non-stationarity = Markov chain of joint policies
- Meta-learning addresses fast adaptation
- Meta-PG + LOLA = partial solutions
- Next: unify both in Meta-MAPG (Ch.6)

**Total:** ~10-12 pages

---

## Ch.6: Meta-Multiagent Policy Gradient Theorem

### Structure (8 sections, ~12-15 pages) 🔴 **CORE CHAPTER**

**6.1 Introduction** (1 page)
- Problem: Meta-PG ignores peer learning, LOLA ignores own learning
- Goal: derive principled gradient accounting for BOTH
- Preview: Meta-MAPG theorem unifies both approaches

**6.2 The Meta-MAPG Theorem** (1 page)
- **Theorem 6.1 statement** (numbered theorem environment)
- Three gradient components:
  1. Current Policy (standard PG)
  2. Own Learning (how φ^i_0 affects own future policies)
  3. Peer Learning (how φ^i_0 affects peers' future policies)
- Significance: principled unification of Meta-PG + LOLA

**6.3 Proof** (5-6 pages) 🔴 **MOST IMPORTANT SECTION**

**6.3.1 Setup and Product Rule**
- Start with meta-value function
- Apply ∇_{φ^i_0}, product rule → 4 terms (A, B, C, D)

**6.3.2 Expanding Trajectory Probability (Term A)**
- Chain of trajectory probabilities p(τ_{φ_0:ℓ})
- Gradient hits EACH trajectory in the chain
- Result: sum of log-gradients for all ℓ'=0 to ℓ

**6.3.3 Log-Derivative Trick**
- Apply ∇ log x = ∇x / x to each trajectory
- **Crucial step:** log π(τ_{φ_ℓ}|φ^i_ℓ) + log π(τ_{φ_ℓ}|φ^{-i}_ℓ)
- Both own AND peer policies appear!

**6.3.4 Unrolling Q-Function Gradient (Terms B-D)**
- Repeatedly apply Bellman equation
- Cite Sutton & Barto 1998 (from Ch.4)
- Future joint policy φ_ℓ+1 appears

**6.3.5 Combining Terms**
- Terms A + B + C + D → final form
- Rearrange into three components (current, own, peer)
- **QED** ∎

**Key Insight Box:** The gradient naturally decomposes into three components because:
- Current policy: φ^i_0 directly affects τ_{φ_0}
- Own learning: φ^i_0 → φ^i_1:ℓ → τ_{φ_1:ℓ} (via inner-loop updates)
- Peer learning: φ^i_0 → φ^{-i}_1:ℓ → τ_{φ_1:ℓ} (via interaction effects!)

**6.4 Connection to Meta-PG** (1 page)
- **Corollary 6.1:** Meta-PG is a special case of Meta-MAPG
- Proof: If ∇_{φ^i_0} φ^{-i}_{ℓ'+1} = 0, then peer learning term vanishes
- When is this valid? Only if peers' policies treated as external (not realistic in MARL!)

**6.5 Connection to LOLA** (1 page)
- LOLA derives similar peer learning term via Taylor approximation
- Meta-MAPG derives it from first principles (gradient of meta-value)
- LOLA doesn't include own learning term
- Meta-MAPG = principled unification

**6.6 Example: Stateless Zero-Sum Game** (1-2 pages)
- Simple game: V^i_ℓ = φ^i_ℓ φ^j_ℓ, V^j_ℓ = -φ^i_ℓ φ^j_ℓ
- Derive Meta-MAPG gradient: ∇_{φ^i_0} V^i_{0:1} = φ^j_1 - α φ^i_1
- Derive Meta-PG gradient: ∇_{φ^i_0} V^i_{0:1} = φ^j_1 (misses -α φ^i_1 term!)
- Figure 6.1: Meta-PG fails (biased updates), Meta-MAPG converges

**6.7 Computational Considerations** (1 page)
- Computing peer learning gradient requires differentiating through inner-loop
- DiCE (Differentiable Monte Carlo Estimator) for efficiency
- Centralized vs decentralized meta-training (opponent modeling)

**6.8 Summary** (0.5 page)
- Meta-MAPG theorem extends single-agent PGT (Ch.4) to multi-agent + meta-learning
- Unifies Meta-PG (own learning) + LOLA (peer learning)
- Principled gradient derivation from first principles
- Next: convergence analysis (Ch.7), LLM application (Ch.8-9)

**Total:** ~12-15 pages

---

## Key Differences: Ch.4 vs Ch.6

| Aspect | Ch.4 (Single-Agent PGT) | Ch.6 (Meta-MAPG) |
|--------|------------------------|------------------|
| **Setting** | Single agent, stationary environment | N agents, non-stationary (learning agents) |
| **Policy** | π(a\|s,θ) | Joint policy φ = {φ^i, φ^{-i}} |
| **Performance** | v_{π_θ}(s_0) | V^i_{φ_0:ℓ+1}(s_0, φ^i_0) (meta-value) |
| **Gradient** | ∇_θ J(θ) | ∇_{φ^i_0} V^i_{φ_0:ℓ+1} |
| **Components** | 1 (current policy) | 3 (current + own + peer) |
| **Key insight** | No ∇_θ μ(s) needed | Peer learning emerges naturally! |
| **Proof structure** | Recursive Bellman unrolling | Product rule + chain of trajectories |

---

## Writing Strategy

### Option A: Sequential (Ch.5 → Ch.6)
1. Write Ch.5 (setup) — 10 pages, 1 day
2. Write Ch.6 (proof) — 15 pages, 2 days

**Total:** 3 days

### Option B: Core-first (recommended!)
1. Write Ch.6.3 (proof) — 6 pages, 1 day — **HARDEST, DO FIRST**
2. Write Ch.6.1-6.2 (intro + theorem statement) — 2 pages, 2 hours
3. Write Ch.6.4-6.8 (connections, example, summary) — 6 pages, 4 hours
4. Write Ch.5 (setup) — 10 pages, 1 day

**Total:** 2.5 days (more efficient — hardest part done first!)

---

## LaTeX Skeleton: Ch.6

```latex
\chapter{Meta-Multiagent Policy Gradient Theorem}
\label{ch:meta-mapg}

\section{Introduction}
% Problem: Meta-PG incomplete, LOLA incomplete
% Goal: derive principled gradient for multi-agent meta-learning
% Preview theorem

\section{The Meta-MAPG Theorem}
\begin{theorem}[Meta-Multiagent Policy Gradient]
\label{thm:meta-mapg}
For any stochastic game $\mathcal{M}_n$, the gradient of the meta-value function...
\end{theorem}
% Significance
% Three components explained

\section{Proof of Meta-MAPG Theorem}
\subsection{Setup and Product Rule}
% Eq. (6.1): meta-value function
% Eq. (6.2): apply gradient, product rule → 4 terms

\subsection{Expanding Trajectory Probability}
% Term A: chain of trajectories
% Eq. (6.3): p(τ_{φ_0:ℓ}) = p(τ_{φ_0}) × ... × p(τ_{φ_ℓ})
% Eq. (6.4): gradient of product

\subsection{Log-Derivative Trick}
% Eq. (6.5): ∇ log x = ∇x / x
% Eq. (6.6): log π(τ_{φ_ℓ}|φ^i_ℓ) + log π(τ_{φ_ℓ}|φ^{-i}_ℓ)
% **Key step:** both own and peer appear!

\subsection{Unrolling Q-Function Gradient}
% Terms B-D
% Eq. (6.7): ∇ Q^i_{φ_ℓ+1}(s_0, a_0) via Bellman
% Cite Sutton & Barto 1998

\subsection{Combining Terms}
% Eq. (6.8): Final form
% Current + Own + Peer
% QED ∎

\subsubsection{Key Insight}
% Boxed insight about decomposition

\section{Connection to Meta-PG}
\begin{corollary}[Meta-PG as Special Case]
\label{cor:meta-pg}
Meta-PG is obtained from Meta-MAPG when...
\end{corollary}

\section{Connection to LOLA}
% LOLA vs Meta-MAPG
% Principled derivation vs approximation

\section{Example: Stateless Zero-Sum Game}
% Simple game
% Meta-MAPG derivation
% Meta-PG derivation
% Figure: comparison

\section{Computational Considerations}
% DiCE for efficiency
% Centralized vs decentralized

\section{Summary}
% Key results
% Preview Ch.7 (convergence)
```

---

## Figures Needed

**Ch.5:**
- **Figure 5.1:** Markov chain of policies (φ_0 → τ_{φ_0} → φ_1 → ...)
- **Figure 5.2:** Probabilistic graph showing dependencies (from Kim Fig.1b)

**Ch.6:**
- **Figure 6.1:** Zero-sum game results (Meta-PG vs Meta-MAPG convergence)
- **Figure 6.2:** Three gradient components visualization (current, own, peer)

---

## Proof Writing Tips (Ch.6.3)

1. **Start with intuition:**
   "In multi-agent settings, the meta-agent's initial policy φ^i_0 affects performance in THREE ways: (1) directly via current trajectory, (2) indirectly via own adapted policies, (3) indirectly via peers' adapted policies. The Meta-MAPG theorem captures all three."

2. **Number every equation:**
   - Equation (6.1): Meta-value function
   - Equation (6.2): Gradient with 4 terms
   - Equation (6.3): Chain of trajectories
   - ...continue through (6.8)

3. **Cross-reference Ch.4:**
   - "As in the proof of Theorem 4.1 (single-agent PGT)..."
   - "Applying the same recursive Bellman unrolling from Section 4.4..."

4. **Highlight the crucial step:**
   When applying log-derivative trick, emphasize that log π(τ_{φ_ℓ}|φ_ℓ) splits into:
   ```
   log π(τ_{φ_ℓ}|φ_ℓ) = log π(τ_{φ_ℓ}|φ^i_ℓ) + log π(τ_{φ_ℓ}|φ^{-i}_ℓ)
   ```
   **This is where peer learning emerges!**

5. **Use "Key Insight" box:**
   After proof, add boxed text:
   > **Key Insight:** The peer learning gradient arises naturally from the chain rule when differentiating through the sequence of joint policies. Unlike LOLA, which approximates this term, Meta-MAPG derives it exactly from the gradient of the meta-objective.

---

## Time Estimate

**Ch.5:**
- Sec 5.1-5.4: 6 hours (setup, stochastic games, meta-learning)
- Sec 5.5-5.7: 4 hours (Meta-PG, LOLA, summary)
- **Total:** 10 hours → **1.5 days**

**Ch.6:**
- Sec 6.1-6.2: 3 hours (intro, theorem statement)
- **Sec 6.3: 10-12 hours** (proof — most time-consuming!)
- Sec 6.4-6.6: 4 hours (connections, example)
- Sec 6.7-6.8: 2 hours (computational, summary)
- **Total:** 20 hours → **2.5 days**

**Grand total:** 30 hours = **4 days of focused work** for Ch.5-6

---

## Dependencies

**Before writing Ch.6:**
- ✅ Ch.4 (Policy Gradient Theorem) — proof structure, notation
- ✅ Kim 2021 reading — DONE

**Reading Ch.6.3 proof requires:**
- Sutton & Barto 1998 (Bellman equation recursion) — cited from Ch.4
- Wei et al. 2018 (multiagent PG) — brief, can cite without full read

---

## Next Steps

**After Ch.4 is done:**
1. **Write Ch.5** (setup) — 1.5 days
2. **Write Ch.6** (Meta-MAPG theorem + proof) — 2.5 days

**OR (if Ch.4 taking longer):**
1. **Start Ch.6.3** (proof) now — hardest part, can work in parallel
2. Come back to Ch.5 later (easier setup chapter)

---

**Ready when you are!** 🎭

_— Лиля_
