# Ch.4 Policy Gradient Theorem — Ready to Write

**Status:** ✅ Reading complete, proof extracted, structure ready

**Source:** Sutton & Barto (2018), Chapter 13  
**Notes:** `~/Coding/maynard/dissertation/notes/sutton-barto-2018.md`

---

## Quick Start

You can now write Ch.4 directly. All the material you need is in `sutton-barto-2018.md`.

**Key sections:**
1. **Theorem statement** — line 42-74 of notes
2. **Detailed proof** — line 84-180 of notes  
3. **Algorithms** (REINFORCE, Actor-Critic) — line 182-198
4. **Writing checklist** — line 243-254

---

## Structure Recommendation

### Ch.4 Sections (10 sections total):

**4.1 Introduction** (1-2 pages)
- Why policy gradients vs. value-based methods?
- Advantages: stochastic policies, continuous action spaces, simpler approximation
- Preview: PGT eliminates need to differentiate state distribution

**4.2 Preliminaries** (1 page)
- MDP notation review (from Ch.3)
- Policy parameterization π(a|s,θ)
- Performance measure J(θ) = v_{π_θ}(s_0)

**4.3 Policy Gradient Theorem — Statement** (1 page)
- Theorem statement (Eq. 13.5)
- Explain ∝ symbol (proportionality constant = avg episode length)
- Significance: no ∇_θ μ(s) term!
- Intuition: why this is non-obvious and powerful

**4.4 Proof of PGT** (3-4 pages) 🔴 CORE SECTION
- **Step 1:** Gradient of value function (product rule)
- **Step 2:** Expand action-value gradient (Bellman equation)
- **Step 3:** Recursive substitution
- **Step 4:** Unrolling to infinite series
- **Step 5:** Define on-policy distribution μ(s)
- **Step 6:** Final form ∇J(θ) ∝ Σ_s μ(s) Σ_a q_π(s,a) ∇_θ π(a|s,θ)
- **Key insight box:** The telescoping cancellation of ∇_θ μ(s)

**4.5 Log-Derivative Trick** (1 page)
- Prove ∇_θ π(a|s,θ) = π(a|s,θ) ∇_θ ln π(a|s,θ)
- Convert PGT to expectation form
- Enables Monte Carlo sampling

**4.6 REINFORCE Algorithm** (2 pages)
- Derive update rule from PGT
- Monte Carlo estimate: G_t as unbiased sample of q_π(s_t, a_t)
- Pseudocode (box)
- Convergence properties (cite Sutton 1999)

**4.7 Baselines and Variance Reduction** (1-2 pages)
- Generalized PGT with baseline b(s)
- Prove baseline doesn't introduce bias
- Natural choice: b(s) = v̂(s,w)
- REINFORCE with baseline pseudocode

**4.8 Actor-Critic Methods** (1-2 pages)
- Bootstrap with value function critic
- TD error δ_t = r_{t+1} + γv̂(s_{t+1}) - v̂(s_t)
- δ_t estimates advantage A_π(s,a)
- Reduces variance vs. pure Monte Carlo

**4.9 Continuing Case** (1 page)
- Different performance measure: average reward rate
- Different μ definition (stationary distribution)
- Same proof structure applies

**4.10 Summary** (0.5 page)
- PGT is foundation for all policy gradient methods
- Key result: tractable gradient via sampling
- Next: extend to multi-agent (Ch.5), meta-learning (Ch.6)

**Total:** ~12-15 pages

---

## Writing Strategy

**Option A: Linear (recommended for first draft)**
1. Write Sec 4.1-4.3 (intro, setup, theorem statement) — 3 pages
2. Write Sec 4.4 (proof) — 4 pages — **FOCUS HERE FIRST**
3. Write Sec 4.5-4.8 (algorithms) — 5 pages
4. Write Sec 4.9-4.10 (continuing case, summary) — 2 pages

**Option B: Core-first (if time is tight)**
1. Write Sec 4.4 (proof) — 4 pages — ESSENTIAL
2. Write Sec 4.3 (theorem statement) — 1 page
3. Write Sec 4.6 (REINFORCE) — 2 pages  
4. Fill in the rest (intro, baselines, actor-critic, summary)

**My recommendation:** Option A. You can write Sec 4.1-4.3 quickly (straightforward), then focus energy on the proof (Sec 4.4), which is the hardest part.

---

## Proof Writing Tips

**For Sec 4.4 (the proof):**

1. **Start with intuition paragraph:**
   "The challenge in computing ∇J(θ) is that performance depends not only on the action selection (which we can easily differentiate) but also on the state distribution μ(s), which changes with the policy. Computing ∇_θ μ(s) would require knowing the environment dynamics, which we don't have access to. The Policy Gradient Theorem resolves this by showing that ∇J(θ) can be expressed in a form that depends only on μ(s) itself, not its derivative."

2. **Use clear step-by-step structure:**
   - Each step gets its own subsection or paragraph
   - State what you're doing before the math
   - After the math, interpret the result

3. **Use equation numbering:**
   - Number every key equation
   - Reference them in text ("substituting (4.7) into (4.5)...")

4. **Add "Key insight" box after Step 6:**
   Highlight the telescoping cancellation visually.

5. **Include Figure 4.1:**
   - Diagram showing state transition graph
   - Annotate with π(a|s), p(s'|s,a), η(s) visitation counts
   - Visualize the recursive unrolling

---

## Notation

Use Sutton-Barto conventions consistently:
- States: S, s, s'
- Actions: A, a, a'
- Policy: π(a|s,θ), parameter θ ∈ ℝ^d
- Value functions: v_π(s), q_π(s,a)
- On-policy distribution: μ(s) (normalized), η(s) (unnormalized)
- Gradient: ∇_θ or just ∇ when θ is clear from context
- Proportionality: ∝ (be clear about the constant)

**Define everything on first use**, even if it was in Ch.3.

---

## LaTeX Skeleton (ready to copy)

```latex
\chapter{Policy Gradient Theorem}
\label{ch:pg-theorem}

\section{Introduction}
% Why policy gradients?
% Problem: need to differentiate performance w.r.t. policy parameters
% Challenge: performance depends on state distribution, which depends on policy

\section{Preliminaries}
% MDP notation (brief review from Ch.3)
% Policy parameterization π(a|s,θ)
% Performance measure J(θ) = v_{π_θ}(s_0)

\section{The Policy Gradient Theorem}
% Theorem statement (numbered theorem environment)
% Significance
% Intuition

\section{Proof}
\subsection{Gradient of the State-Value Function}
% Step 1

\subsection{Expanding the Action-Value Gradient}
% Step 2

\subsection{Recursive Substitution}
% Step 3

\subsection{Unrolling the Recursion}
% Step 4

\subsection{The On-Policy Distribution}
% Step 5

\subsection{Final Form}
% Step 6

\subsection{Key Insight}
% Boxed insight about cancellation

\section{The Log-Derivative Trick}
% Score function lemma
% Expectation form

\section{REINFORCE: Monte Carlo Policy Gradient}
% Algorithm derivation
% Pseudocode
% Convergence

\section{Variance Reduction with Baselines}
% Generalized PGT
% Zero-bias property
% REINFORCE with baseline

\section{Actor-Critic Methods}
% Bootstrap with critic
% TD error as advantage estimate

\section{Continuing Case}
% Average reward formulation
% Different μ definition

\section{Summary}
% Key results
% Preview of Ch.5
```

---

## Time Estimate

- **Sec 4.1-4.3:** 2 hours (straightforward writing)
- **Sec 4.4 (proof):** 4-6 hours (careful derivation + LaTeX)
- **Sec 4.5-4.8:** 3-4 hours (algorithms are mostly pseudocode + explanation)
- **Sec 4.9-4.10:** 1 hour (brief)

**Total:** 10-13 hours for first draft.

**Revision:** 2-3 hours (proof-reading, equation checking, citations)

**Grand total:** ~15 hours → **2 days of focused work**

---

## Next Steps

1. ✅ **Reading:** Sutton & Barto Ch.13 — DONE  
2. 🟡 **Writing:** Ch.4 draft — START NOW
3. ⏳ **Next reading:** Kim 2021 (for Ch.5-6) — do while Ch.4 is resting

**Suggested workflow for today/tomorrow:**
- **Today (Feb 24 evening):** Write Sec 4.1-4.3 (intro + theorem statement) — 3 pages, 2 hours
- **Tomorrow (Feb 25 morning):** Write Sec 4.4 (proof) — 4 pages, 5 hours
- **Tomorrow (Feb 25 afternoon):** Write Sec 4.5-4.8 (algorithms) — 5 pages, 4 hours
- **Tomorrow (Feb 25 evening):** Write Sec 4.9-4.10 + revise — 2 pages, 2 hours

**Result:** Ch.4 draft complete in 1.5 days!

---

## Questions to Resolve While Writing

1. **Cite Sutton 1999 for REINFORCE convergence?** (or just state informally?)
2. **Include Figure 4.1 (state transition diagram)?** (recommended — helps visualize unrolling)
3. **Continuing case in main body or appendix?** (main body — it's short, and connects to Ch.5)
4. **How much detail on softmax policy gradient (Eq. 13.7)?** (include as example after Sec 4.5)

---

**You're ready to write. Go!** 🚀

_— Лиля_
