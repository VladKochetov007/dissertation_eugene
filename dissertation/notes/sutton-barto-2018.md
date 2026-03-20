# Reading Notes: Sutton & Barto (2018) — Reinforcement Learning: An Introduction

**Book:** Reinforcement Learning: An Introduction (2nd Edition)
**Authors:** Richard S. Sutton, Andrew G. Barto
**Key Chapters:** Ch. 3 (MDPs), Ch. 13 (Policy Gradient Methods)

## Key Claims

- The MDP formalism (S, A, P, R, γ) provides the foundational abstraction for sequential decision-making under uncertainty
- The Markov property makes the framework tractable: agent need not maintain full history to act optimally
- The Policy Gradient Theorem gives an exact expression for ∇J(θ) that does NOT require differentiating the state distribution μ_π — this is the crucial theoretical result
- Policy gradient methods have advantages over value-based methods: smooth interpolation, can represent stochastic optimal policies, natural mechanism for structural priors
- Trade-off: higher variance gradient estimates → need baselines and actor-critic methods

## Mathematical Setup

### MDP Tuple
M = ⟨S, A, P, R, γ⟩
- S: finite state set
- A: finite action set
- P: S × A × S → [0,1] transition function
- R: S × A × S → ℝ reward function
- γ ∈ [0,1) discount factor

### Markov Property
Pr{S_{t+1}=s', R_{t+1}=r | S_t, A_t} = Pr{S_{t+1}=s', R_{t+1}=r | S_0, A_0, R_1, ..., S_t, A_t}

### Value Functions
- State-value: v_π(s) = E_π[G_t | S_t = s]
- Action-value: q_π(s,a) = E_π[G_t | S_t = s, A_t = a]
- Relation: v_π(s) = Σ_a π(a|s) q_π(s,a)

### Bellman Equations
v_π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ v_π(s')]
q_π(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ Σ_{a'} π(a'|s') q_π(s',a')]

### On-Policy Distribution
η(s) = Σ_{t=0}^∞ γ^t Pr{S_t = s | S_0 = s_0, π}
μ_π(s) = η(s) / Σ_{s'} η(s')

## Key Theorems

### Policy Gradient Theorem (Ch. 13, Thm 13.1)

**Statement (Episodic case):**
∇J(θ) ∝ Σ_s μ(s) Σ_a q_π(s,a) ∇_θ π(a|s,θ)

where:
- J(θ) = v_{π_θ}(s_0) is the performance measure (value of start state)
- μ(s) is the on-policy state distribution under π
- ∝ means "proportional to" (constant = average episode length)

**Significance:** 
- The gradient does NOT require differentiating μ(s) w.r.t. θ
- Even though μ depends on θ through the policy!
- This makes gradient estimation tractable via sampling

**Why this matters:**
Performance depends on BOTH:
1. Action selections (easy to differentiate)
2. State distribution (environment-dependent, unknown)

Without PGT, we'd need ∇_θ μ(s), which requires environment dynamics.
PGT eliminates this dependency → can estimate gradient from experience alone.

### Log-derivative trick (Score function)
∇_θ π(a|s,θ) = π(a|s,θ) ∇_θ ln π(a|s,θ)

**Proof:** 
∇_θ π(a|s,θ) = π(a|s,θ) ∇_θ π(a|s,θ) / π(a|s,θ)
             = π(a|s,θ) ∇_θ ln π(a|s,θ)  (since ∇ ln x = ∇x / x)

**Application to PGT:**
∇J(θ) ∝ E_{s~μ, a~π} [q_π(s,a) ∇_θ ln π(a|s,θ)]

This is the **expectation form** — can estimate via sampling!

## Detailed Proof of Policy Gradient Theorem (for Ch.4)

### Setup
- Episodic case, no discounting (γ = 1)
- Performance: J(θ) = v_π(s_0)  
- Goal: Derive ∇J(θ) without differentiating μ(s)

### Step 1: Gradient of State-Value Function

Start with:
```
∇v_π(s) = ∇ [Σ_a π(a|s) q_π(s,a)]
```

Apply product rule:
```
∇v_π(s) = Σ_a [∇π(a|s) q_π(s,a) + π(a|s) ∇q_π(s,a)]
```

### Step 2: Expand Action-Value Gradient

Recall Bellman equation for q:
```
q_π(s,a) = Σ_{s',r} p(s',r|s,a) [r + v_π(s')]
```

Take gradient (reward r doesn't depend on θ):
```
∇q_π(s,a) = Σ_{s'} p(s'|s,a) ∇v_π(s')
```

### Step 3: Substitute Back

```
∇v_π(s) = Σ_a ∇π(a|s) q_π(s,a) + Σ_a π(a|s) Σ_{s'} p(s'|s,a) ∇v_π(s')
```

The first term is **direct effect** of policy change in state s.
The second term is **indirect effect** via changed successor state values.

### Step 4: Recursive Unrolling

The second term contains ∇v_π(s'), which we expand the same way:
```
∇v_π(s') = Σ_{a'} ∇π(a'|s') q_π(s',a') + Σ_{a'} π(a'|s') Σ_{s''} p(s''|s',a') ∇v_π(s'')
```

Substitute back → infinite series:
```
∇v_π(s) = Σ_{x∈S} Σ_{k=0}^∞ Pr(s→x, k, π) Σ_a ∇π(a|x) q_π(x,a)
```

where **Pr(s→x, k, π)** = probability of reaching state x from s in exactly k steps under π.

### Step 5: Define On-Policy Distribution

**Unnormalized** visitation count:
```
η(s) = Σ_{k=0}^∞ Pr(s_0→s, k, π)
```
This is the expected total number of visits to state s starting from s_0.

**Normalized** on-policy distribution:
```
μ(s) = η(s) / Σ_{s'} η(s')
```

### Step 6: Apply to Performance Measure

```
∇J(θ) = ∇v_π(s_0)
       = Σ_{s} η(s) Σ_a ∇π(a|s) q_π(s,a)
       = [Σ_{s'} η(s')] · Σ_s μ(s) Σ_a ∇π(a|s) q_π(s,a)
       ∝ Σ_s μ(s) Σ_a ∇π(a|s) q_π(s,a)
```

**Constant of proportionality:** Σ_{s'} η(s') = average episode length

**QED**

### Key Insight

The gradient of the state distribution **cancels out** during recursive unrolling!

Why?
- ∇v_π(s) contains both direct (∇π in current state) and indirect (changed state visitation) effects
- Recursive expansion via Bellman equation causes the indirect effects to telescope
- Final form depends only on μ(s), NOT ∇_θ μ(s)!

This is the **crucial theoretical result** that makes policy gradient methods tractable.

### For Dissertation Ch.4

**Include:**
1. Full derivation (Steps 1-6 above)
2. Emphasize the "no differentiation of state distribution" property
3. Connect to REINFORCE algorithm (uses Monte Carlo sampling of this gradient)
4. Discuss continuing case (different performance measure, same proof structure)
5. Prove the log-derivative trick explicitly
6. Show softmax policy gradient (Eq. 13.7) as concrete example

**Notation:**
- Use Sutton-Barto conventions (they match Kim 2021 paper)
- Clearly distinguish μ(s) vs. η(s) (normalized vs. unnormalized)
- Define Pr(s→x, k, π) rigorously (k-step transition probability)

## Key Algorithms

### REINFORCE
θ_{t+1} = θ_t + α G_t ∇_θ ln π(a_t|s_t, θ_t)
- Unbiased but high variance
- Uses sample return G_t as estimate of q_π(s_t, a_t)

### REINFORCE with Baseline
θ_{t+1} = θ_t + α (G_t - b(s_t)) ∇_θ ln π(a_t|s_t, θ_t)
- Baseline b(s) doesn't introduce bias (since Σ_a ∇π = 0)
- Natural choice: b(s) = v̂(s, w)

### Actor-Critic
- Actor: θ_{t+1} = θ_t + α δ_t ∇_θ ln π(a_t|s_t, θ_t)
- Critic: w_{t+1} = w_t + β δ_t ∇_w v̂(s_t, w_t)
- TD error: δ_t = r_{t+1} + γ v̂(s_{t+1}, w) - v̂(s_t, w)
- δ_t estimates the advantage A_π(s,a) = q_π(s,a) - v_π(s)

## Connections to Dissertation

- **Ch.3**: Direct foundation — MDP, policies, value functions, PGT all defined here
- **Ch.4**: Full proof of PGT comes from Ch. 13 material, worked through in detail
- **Ch.5**: LOLA and meta-learning extend the single-agent PGT to multi-agent
- **Ch.6**: Kim et al.'s Meta-MAPG theorem generalises Theorem 13.1 to N agents
- **Ch.9**: Cooperative steering game uses actor-critic as the base algorithm
- **Notation**: The dissertation follows Sutton-Barto conventions (S, A, P, R, γ, π, v, q, μ, η)

## Questions / Gaps

- [x] The PGT proof in Ch.13 — **DONE** (detailed proof documented above for Ch.4)
- [ ] Episodic vs continuing case: Understand continuing formulation (avg reward rate, different μ definition)
- [ ] Connection between single-agent μ_π and multi-agent joint state distribution in Kim et al.
- [ ] How does trajectory probability factorization extend to N agents with joint actions?
- [ ] Optimal baseline selection — is there a closed-form optimal b(s)? (probably not, but discuss)
- [ ] Softmax policy gradient derivation (Eq. 13.7) — work through algebra for Ch.4 example
- [ ] REINFORCE convergence proof — cite Sutton 1999 or include sketch?

## Ch.4 Writing Checklist

When writing Chapter 4:
- [ ] **Sec 4.1:** Introduction — why policy gradients? (vs value-based methods)
- [ ] **Sec 4.2:** Preliminaries — MDP notation, performance measure J(θ), policy parameterization
- [ ] **Sec 4.3:** Policy Gradient Theorem — statement, significance, intuition
- [ ] **Sec 4.4:** Proof of PGT — full derivation (use detailed proof above)
- [ ] **Sec 4.5:** REINFORCE algorithm — Monte Carlo sampling, eligibility vector
- [ ] **Sec 4.6:** Baselines and variance reduction — why baselines help, zero-bias property
- [ ] **Sec 4.7:** Actor-Critic methods — bootstrap with value function
- [ ] **Sec 4.8:** Continuing case formulation — average reward, different μ definition
- [ ] **Sec 4.9:** Examples — softmax policy (Eq. 13.7), Gaussian for continuous actions
- [ ] **Sec 4.10:** Summary — key results, convergence properties, preview of Ch.5 (multi-agent)

## Relevant Equations

Performance measure (episodic):
J(θ) = v_{π_θ}(s_0) = E_{π_θ}[G_0 | S_0 = s_0]

Trajectory probability:
p(τ | π) = d_0(s_0) Π_{t=0}^{H-1} π(a_t|s_t) P(s_{t+1}|s_t, a_t)

Discounted return along trajectory:
G(τ) = Σ_{t=0}^{H-1} γ^t r_{t+1}

## Personal Notes from Overleaf

- "curious — simple approach from the econ point of view, but easily modifiable" (re: MDP reward structure)
- Gradient policy methods learn parametrised policy π(a|s,θ) directly, as opposed to action-value estimates
- "Neural Nets fall under this category since this is precisely what we learn"
- Key: continuous policy parametrisation → smooth changes → stronger convergence guarantees
- In problems where optimal policy is stochastic (partially observable, game-theoretic), PG methods can represent these directly
- "The paper [Kim] is for the discrete case — maybe I can adjust for continuous and simulate?"
