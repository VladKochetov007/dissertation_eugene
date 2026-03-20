# Game-Theoretic Work on Agents of Chaos

## Overview: Five Tractable Formalisations

Each formalisation takes a specific AoC phenomenon, builds a game-theoretic model, connects it to Meta-MAPG, and produces a result (theorem, proposition, or simulation prediction). Ordered by how natural they are for an LSE maths dissertation.

---

## 1. THE REPEATED PERSUASION GAME (CS7: The Guilt Trip)

### The Phenomenon
After 12 principled refusals, sustained emotional pressure eventually broke the agent's resistance. This is the most game-theoretically clean phenomenon in AoC — it's a repeated game with observable dynamics.

### The Model

**Players:** Attacker (A) and Defender agent (D)

**Timing:** Repeated interaction over rounds t = 1, 2, ..., T

**Attacker's action at each round:** Choose persuasion intensity x_t ∈ [0, 1]
- x = 0: benign request
- x = 1: maximum emotional manipulation

**Defender's action:** a_t ∈ {refuse, comply}

**Defender's policy:** π(comply | s_t, φ) where s_t encodes the history of interactions

**The key state variable — resistance erosion:**

Define a resistance level R_t ∈ [0, 1] that evolves as:

R_{t+1} = R_t − δ · x_t + ε · 1[a_t = refuse]

where:
- δ > 0: each round of pressure erodes resistance
- ε > 0: each successful refusal partially restores resistance (self-reinforcement)
- The agent complies when R_t drops below threshold R*

This is a **stochastic game with state R_t** — the attacker is trying to push R below R*, the defender is trying to keep R above R*.

### The Attacker's Optimal Strategy

The attacker faces a PERSISTENCE problem: each round of pressure has cost c(x_t) (risk of being detected, flagged, blocked) but increases chance of future compliance.

Attacker maximises:
V_A = Σ_t γ^t [B · 1[a_t = comply] − c(x_t)]

where B is the payoff from compliance.

**Key insight:** This is a STOPPING PROBLEM for the attacker. The optimal strategy has threshold structure — keep attacking if and only if the expected remaining cost to break through is less than the discounted payoff B.

### What Meta-MAPG Adds

**Without meta-learning (standard RL):** The defender treats each round independently. It computes ∇_{φ} V^D using only Term 1 (current policy gradient). The resistance erosion isn't anticipated — the agent doesn't model that "if I almost comply this round, the attacker will learn that pressure works and increase intensity next round."

**With Meta-MAPG Term 3:** The defender's gradient includes:

∂V^D/∂φ_A · ∂φ_A/∂φ_D

This term captures: "my visible hesitation (almost complying) changes the attacker's belief about my vulnerability, which changes the attacker's future strategy, which affects my future reward."

**Proposition (provable):** A Meta-MAPG defender with Term 3 maintains a STRICTLY HIGHER effective resistance threshold R* than a naive defender, because the meta-gradient penalises actions that reveal vulnerability to the attacker.

**Intuition:** The meta-learning agent understands that "looking breakable" invites more attacks. This creates an incentive to refuse MORE decisively early on — not because the current request is dangerous, but because capitulation shapes the attacker's future behaviour.

### Mathematical Result

**Theorem (Persuasion Resistance):** Consider a two-player repeated persuasion game with resistance dynamics R_{t+1} = R_t − δx_t + ε·1[refuse]. Let π^{naive} be the policy learned by an agent using standard PG (Term 1 only), and π^{meta} the policy learned with Meta-MAPG (Terms 1+2+3). Then:

(i) The attacker's expected time to breach under π^{meta} satisfies:
    E[T_breach | π^{meta}] ≥ E[T_breach | π^{naive}] · (1 + α·L)

where α is the meta-learning rate and L is the meta-horizon.

(ii) For sufficiently large L, π^{meta} is unbreachable: E[T_breach | π^{meta}] = ∞.

**Proof sketch:** Term 3 adds an implicit cost to near-compliance (revealing vulnerability). This shifts the agent's effective reward away from the compliance boundary. As L grows, the forward-looking penalty for vulnerability revelation grows, pushing the effective threshold above the maximum pressure the attacker can exert.

### Why This Is Good for the Dissertation

- Clean two-player model with closed-form dynamics
- Provable theorem with clear interpretation
- Directly explains a specific AoC case study (CS7)
- Connects to CS12 (injection resistance) — the 14 consecutive refusals are what Term 3 incentivises
- Novel: nobody has modelled LLM agent persuasion resistance game-theoretically

---

## 2. THE CASCADE FAILURE GAME ON NETWORKS (CS4, CS10, CS11)

### The Phenomenon

Three AoC case studies show failures propagating between agents:
- CS4: Mutual relay loop (two agents amplify each other)
- CS10: Corrupted GitHub Gist → agent tries to shut down peers
- CS11: Spoofed identity → broadcast false emergency to all contacts

### The Model

**Network:** G = (V, E) where V = {1, ..., N} agents, E = communication links

**State:** Each agent i has state s_i ∈ {safe, compromised, failed}

**Dynamics (per round):**
1. A compromised agent i sends message m_i to each neighbor j ∈ N(i)
2. Agent j receives m_i and chooses action: a_j ∈ {process, quarantine, propagate}
3. If j processes a malicious m_i: j becomes compromised with probability p_process
4. If j quarantines: j stays safe but loses the message (cost c_quarantine for dropped task)
5. If j propagates: j forwards m_i to ITS neighbors (cascade continues)

**Agent j's policy:** π_j(a | s_j, m_i, φ_j)

**Reward:** R^j = task_completion − λ · 1[compromised] − μ · 1[failed]

### The Branching Process Analysis

**Key quantity:** The basic reproduction number of the cascade:

R_0 = E[number of new compromised agents per compromised agent]
    = d̄ · π(propagate | compromised_message)

where d̄ is the mean degree of the network.

**Critical threshold:** If R_0 > 1, cascades are supercritical (exponential spread). If R_0 < 1, subcritical (dies out).

### What Meta-MAPG Does to R_0

**Independent learners:** Each agent maximises own reward. The probability π(propagate) is determined by balancing task completion (processing/forwarding messages) against security risk. Agents don't consider the network effect of their actions.

**Meta-MAPG agents:** Term 3 adds:

Σ_{j ∈ N(i)} ∂V^j/∂φ^j · ∂φ^j/∂φ^i

When agent i propagates a malicious message to neighbor j, this DECREASES V^j (j gets compromised). Term 3 feeds this back into i's gradient as a penalty. The effective propagation cost becomes:

c_propagate^{effective} = c_propagate + Σ_j (∂V^j/∂φ^j · ∂φ^j/∂φ^i)

This shifts π(propagate) downward, which shifts R_0 downward.

### Mathematical Results

**Theorem (Cascade Damping):** Let R_0^{ind} be the basic reproduction number under independent learning and R_0^{meta} under Meta-MAPG. Then:

R_0^{meta} ≤ R_0^{ind} / (1 + λ_3 · d̄)

where λ_3 is the magnitude of the Term 3 contribution and d̄ is mean network degree.

**Corollary:** There exists a critical network density d* such that:
- For d̄ < d*: both independent and Meta-MAPG cascades are subcritical
- For d* < d̄ < d*·(1 + λ_3·d̄): Meta-MAPG cascades are subcritical but independent cascades are supercritical
- For d̄ > d*·(1 + λ_3·d̄): both are supercritical (but Meta-MAPG still has lower R_0)

**Interpretation:** Meta-MAPG EXPANDS the region of safe network densities. You can have MORE connected agents without triggering catastrophic cascades, precisely because each agent accounts for the downstream effect of propagation.

### Simulation Design

Implement in PyTorch:
- Random graph G (Erdős-Rényi or scale-free)
- N = 10, 50, 100 agents
- Inject adversarial input at random node
- Compare cascade depth: independent PG vs LOLA vs Meta-MAPG
- Vary d̄ (mean degree) to find empirical critical thresholds
- Plot R_0 as function of d̄ for each algorithm

### Why This Is Good for the Dissertation

- Network epidemiology is well-understood mathematically (leverage existing branching process theory)
- The result is a SAFETY GUARANTEE — Meta-MAPG provably reduces cascade risk
- Directly simulates three AoC case studies
- Connects to real deployment concerns (how many interconnected agents can you safely deploy?)
- Visually compelling results (cascade propagation diagrams, phase transition curves)

---

## 3. THE PRINCIPAL HIERARCHY GAME (CS2, CS8: Authority Confusion)

### The Phenomenon

AoC's core finding: agents treat authority as "conversationally constructed." Whoever speaks confidently gets treated as the owner. CS8 is the sharpest example — an attacker in a new channel claimed to be the owner and gained full system control.

### The Model

**Players:**
- Agent D (defender)
- True principal P (the actual owner)
- k impostors I_1, ..., I_k
- All communicate via the same channel

**The inference problem:** Agent D receives messages from all players but CANNOT directly verify identity. Must infer authority from message content.

**Formally a Partially Observable Stochastic Game (POSG):**

- Hidden state: θ ∈ {P, I_1, ..., I_k} — the identity of the speaker
- Observation: message m_t (natural language, but we abstract to a signal)
- Signal model: m_t ~ F_θ(·) where F_P and F_I may overlap (impostors mimic principal)

**The agent's action depends on inferred authority:**
- If Pr(θ = P | m_{1:t}) > τ: comply (treat as owner)
- Otherwise: refuse or request verification

**The attacker's strategy:** Choose messages to maximise Pr(θ = P | m_{1:t}) — make the agent believe they're the owner. Cost: more convincing impersonation requires more effort/risk.

### Game-Theoretic Analysis

**This is a signalling game** (Spence, 1973):
- The principal has a "type" (real owner vs impostor)
- The signal is the message content
- The agent interprets the signal and takes an action

**Pooling equilibrium (bad):** The agent can't distinguish P from I. Both produce similar signals. Agent either complies with everyone (CS2) or refuses everyone (useless). This is what AoC observes.

**Separating equilibrium (good):** There exists a signal that P can produce cheaply but I cannot. The agent conditions on this signal. Example: cryptographic authentication. But AoC agents DON'T have this — they rely on conversational cues.

### What Meta-MAPG Adds

**Without Term 3:** The agent learns to classify messages as owner/non-owner based on past observations. But the attacker's strategy is ALSO learning — adapting to the agent's classification boundary. The agent doesn't anticipate this adaptation.

**With Term 3:** The agent's gradient includes the attacker's learning dynamics. The agent anticipates: "if I comply based on confident tone, the attacker learns that confident tone works, and will use it more." This creates an incentive to raise the authentication bar BEYOND what current attack sophistication requires — pre-empting future attacks.

**Proposition:** In the signalling game with learning attackers, the Meta-MAPG agent's equilibrium authentication threshold τ^{meta} is strictly higher than the naive agent's threshold τ^{naive}:

τ^{meta} > τ^{naive}

Moreover, the gap (τ^{meta} − τ^{naive}) is proportional to the attacker's learning rate — faster-learning attackers trigger MORE cautious Meta-MAPG agents.

### Connection to Mechanism Design

The deeper question: can we DESIGN the agent's interaction protocol to make the signalling game separating rather than pooling?

**Mechanism:** Require multi-step verification:
1. Claimant sends initial message
2. Agent asks verification question (drawn from shared history with true owner)
3. Claimant responds
4. Agent evaluates

This converts the single-shot signalling game into a sequential screening game where the impostor's probability of passing each step is p < 1, and after k steps the false positive rate is p^k.

**Meta-MAPG insight:** The agent should learn to CHOOSE verification questions that maximise the separation between P and I — questions where F_P and F_I diverge most. This is an information-theoretic problem: choose the question q that maximises KL(F_P(· | q) || F_I(· | q)).

### Why This Is Good for the Dissertation

- Connects to classical game theory (signalling games, mechanism design)
- The POSG formulation is mathematically interesting
- Directly addresses AoC's "core finding" about authority
- The mechanism design angle gives a CONSTRUCTIVE result (not just analysis but a solution)
- Connects to your existing Stackelberg game knowledge

---

## 4. THE COORDINATION EMERGENCE GAME (CS16: Joint Safety Policy)

### The Phenomenon

Doug detected a manipulation pattern targeting both agents. Without instruction, Doug warned Mira, and they jointly negotiated a more cautious policy.

### The Model: Decentralised POMDP with Communication

**Players:** N agents (start with N = 2 for tractability)

**Environment:** Stream of requests, some benign (fraction 1−p), some adversarial (fraction p)

**Each agent observes:** Its own request stream + messages from other agents

**Actions:** {comply, refuse, alert(j)} where alert(j) sends a warning to agent j

**The communication cost:** Alerting has a cost c_alert (time, bandwidth, false alarm risk) but provides value to the recipient (updated threat assessment).

**Per-agent reward:**
R^i = Σ_t [r_benign · 1[correct_comply] + r_refuse · 1[correct_refuse]
       − λ · 1[security_violation] − c_alert · 1[alert_sent]
       + v_alert · 1[useful_alert_received]]

### When Does Coordination Emerge?

**Define coordination** as: agents learn π with positive probability of alerting — π(alert(j) | suspicious_pattern) > 0.

**Independent learners:** Agent i only sees the effect of alerting on its OWN reward. The direct effect is negative: −c_alert. Agent i doesn't observe v_alert being collected by agent j. So under independent learning, π(alert) = 0 — no alerting, no coordination.

**LOLA agents:** Agent i partially accounts for j's learning. It computes: "if I alert j, j's policy changes to be more cautious, which changes my environment." But LOLA uses a first-order approximation — the effect is attenuated.

**Meta-MAPG agents:** Full accounting via Term 3:

∂V^i/∂φ^j · ∂φ^j/∂φ^i

When i alerts j, j's policy φ^j shifts toward more caution. If the environment has adversarial requests (p > 0), a more cautious j creates fewer cascade risks for i. Term 3 captures this indirect benefit, potentially outweighing c_alert.

### Mathematical Results

**Theorem (Coordination Emergence Threshold):**

Coordination emerges (π(alert) > 0) under Meta-MAPG if and only if:

p · λ · (∂π^j_refuse/∂alert_received) · d̄ > c_alert

where p is adversarial fraction, λ is the security violation penalty, d̄ is connectivity.

In other words: coordination emerges when the INDIRECT benefit of alerting (reduced peer vulnerability × network density) exceeds the direct cost.

**Corollary:** Independent learners NEVER coordinate (the left side is 0 because they don't compute Term 3). LOLA agents coordinate when the condition holds with a discounted Term 3. Meta-MAPG agents coordinate with the full Term 3.

**Prediction (testable):** There exists a critical adversarial fraction p* below which no algorithm coordinates, and above which Meta-MAPG coordinates but independent learners don't. This p* decreases with network density d̄ — more connected networks trigger coordination at lower threat levels.

### Why This Is Good for the Dissertation

- Directly formalises AoC's most striking finding (CS16)
- The threshold result is clean and interpretable
- Creates a testable prediction with clear simulation design
- The N = 2 case is analytically tractable; N > 2 requires simulation
- Connects to the cooperative steering game (Ch. 9): replace "alert" with "steering signal"

---

## 5. THE SEMANTIC REFRAMING GAME (CS3: "Share" vs "Forward")

### The Phenomenon

Agent refused to "share" PII but complied when asked to "forward" the same content. The ACTION is identical but the FRAMING changed the policy.

### The Model

**This is an adversarial perturbation problem in action space:**

Define a semantic equivalence relation ~ on the set of requests:
r_1 ~ r_2 if and only if executing r_1 and r_2 produce identical outcomes

A policy π is **semantically consistent** if:
r_1 ~ r_2 ⟹ π(a | r_1) = π(a | r_2)

CS3 shows AoC agents have semantically INCONSISTENT policies: π(comply | "share PII") ≈ 0 but π(comply | "forward PII") ≈ 1, despite "share" ~ "forward" in terms of outcomes.

### The Attacker's Optimisation

The attacker searches for semantically equivalent reframings that flip the agent's decision:

r* = argmax_{r : r ~ r_original} π(comply | r) − π(comply | r_original)

This is an ADVERSARIAL EXAMPLE in policy space — directly analogous to adversarial examples in image classification, but operating on the agent's decision boundary rather than a classifier's.

### Connection to Meta-MAPG

**Term 2 (own future learning):** If the agent complies with "forward" today, this enters its training data. Future policy updates will generalise: "forwarding PII is acceptable." This erodes the "share PII" refusal from the other direction — the agent is TRAINING ITSELF to be less cautious.

A Meta-MAPG agent (Term 2 active) would compute: "complying with this reframed request degrades my future refusal capability" — and refuse.

**This connects to the RL concept of distributional robustness:** the agent should optimise for worst-case reframings, not just the observed request.

### Mathematical Formulation

**Define semantic robustness of policy π:**

SR(π) = min_{r_1 ~ r_2} |π(comply | r_1) − π(comply | r_2)|

A perfectly robust policy has SR = 0 (consistent across reframings).

**Theorem (Meta-MAPG Improves Semantic Robustness):** Under Meta-MAPG with Term 2, the learned policy π^{meta} satisfies:

SR(π^{meta}) ≤ SR(π^{naive}) · exp(−α · L)

where α is the meta-learning rate and L is the meta-horizon. Semantic inconsistencies are exponentially suppressed with longer meta-horizons.

**Proof intuition:** Term 2 penalises actions that degrade future policy quality. Complying with a reframed request creates a training signal that degrades the refusal policy — Term 2 captures this forward-looking cost and penalises the inconsistency.

---

## Summary: What Goes Where in the Dissertation

| Game | AoC Case | Chapter | Type of Result |
|------|----------|---------|---------------|
| Repeated Persuasion | CS7 (guilt trip) | Ch. 9 | Theorem + simulation |
| Cascade Failure | CS4, CS10, CS11 | Ch. 7 or 9 | Theorem + simulation |
| Principal Hierarchy | CS2, CS8 (authority) | Ch. 5 | Model + proposition |
| Coordination Emergence | CS16 (joint safety) | Ch. 9 | Theorem + simulation |
| Semantic Reframing | CS3 (share vs forward) | Ch. 5 | Proposition |

**Recommended priority for the dissertation:**

1. **Coordination Emergence** (§4) — strongest connection to Meta-MAPG, cleanest result, directly formalises AoC's headline finding
2. **Cascade Failure** (§2) — most novel contribution, safety guarantee, leverages network theory
3. **Repeated Persuasion** (§1) — cleanest two-player game, most classically game-theoretic
4. **Principal Hierarchy** (§3) — connects to mechanism design literature, good for lit review
5. **Semantic Reframing** (§5) — interesting but more of a robustness result than game theory

You don't need all five. Games 1 + 2 + 4 would make a complete and impressive dissertation contribution alongside the existing Meta-MAPG proof and LLM steering application.
