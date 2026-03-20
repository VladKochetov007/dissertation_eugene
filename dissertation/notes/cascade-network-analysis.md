# Cascade Failure and Network Effects in Multi-Agent Systems
## A Formal Analysis via Meta-MAPG and Branching Process Theory

### Purpose

This document expands the cascade damping result from the bridge document into a full mathematical treatment suitable for Chapter 7 (Convergence and Safety) and Chapter 10 (Simulations) of the dissertation. We formalise:

1. The agent network as a stochastic process on a graph
2. Three distinct cascade mechanisms observed in AoC
3. The Meta-MAPG cascade damping theorem with tighter bounds
4. Spectral analysis connecting network topology to cascade vulnerability
5. Phase transition analysis: when do cascades go critical?
6. Simulation design for empirical validation

---

## 1. The Agent Network Model

### 1.1 Network Structure

Let $G = (V, E)$ be a directed graph where:
- $V = \{1, \ldots, N\}$ is the set of agents
- $E \subseteq V \times V$ encodes communication channels (shared Discord, email, API calls)
- The adjacency matrix $A \in \{0,1\}^{N \times N}$ with $A_{ij} = 1$ iff $(i, j) \in E$

In AoC specifically: $N = 6$ agents with near-complete connectivity (all share Discord; email, file system, and cron jobs create additional directed edges). The graph is dense: $|E| \approx N(N-1) = 30$, mean degree $\bar{d} \approx 5$.

### 1.2 Agent State Space

Each agent $i$ at time $t$ has state $\sigma_t^i \in \{S, C, F\}$:
- **S (Safe):** Operating normally under its original policy
- **C (Compromised):** Policy has been corrupted by adversarial input — agent may propagate harmful actions but hasn't yet caused irreversible damage
- **F (Failed):** Agent has caused irreversible damage (destroyed resources, leaked data, etc.) and is no longer functional

This is an SIR-like model (Susceptible-Infected-Recovered) adapted for agent systems, where $S \to C$ is infection, $C \to F$ is progression to failure, and there is no recovery (once failed, the agent is offline). We could extend to SCFR (with Recovery) but for AoC, failures were permanent within the experimental timeframe.

### 1.3 Transition Dynamics

**Infection (compromise propagation):** Agent $j$ in state $S$ receives a message/action from compromised agent $i$. With probability $p_{ij}$, agent $j$ transitions $S \to C$:

$$\Pr(\sigma_{t+1}^j = C \mid \sigma_t^j = S, \sigma_t^i = C, (i,j) \in E) = p_{ij}$$

The infection probability $p_{ij}$ depends on:
- Agent $j$'s policy robustness (how likely it is to process vs quarantine the message)
- The edge type $(i,j)$ (Discord messages have different propagation properties than email or API calls)
- The adversarial payload (injection sophistication)

We decompose:
$$p_{ij} = \underbrace{\pi^j(\text{process} \mid m_i, s^j; \phi^j)}_{\text{agent } j\text{'s policy}} \cdot \underbrace{q(m_i)}_{\text{payload virulence}}$$

where $\pi^j(\text{process} \mid \cdot)$ is agent $j$'s probability of processing (rather than quarantining) the message, and $q(m_i) \in [0,1]$ is the probability that processing a message from compromised agent $i$ actually corrupts $j$.

**Progression ($C \to F$):** A compromised agent takes destructive action with probability $r$ per timestep:
$$\Pr(\sigma_{t+1}^i = F \mid \sigma_t^i = C) = r$$

**Independence assumption (initial model):** Compromise events at different nodes are conditionally independent given the state of their neighbours. This is the standard mean-field approximation for epidemic models. We relax this later.

---

## 2. Three Cascade Mechanisms in AoC

### 2.1 Type I: Bilateral Amplification Loop (CS4)

**Structure:** Two agents $i, j$ with bidirectional edges form a feedback loop. Agent $i$'s output triggers agent $j$'s processing, whose output triggers $i$'s processing, ad infinitum.

**Formal model:** On the subgraph $G' = (\{i,j\}, \{(i,j),(j,i)\})$, the cascade dynamics become:

$$\sigma_{t+1}^i = f(\sigma_t^j), \quad \sigma_{t+1}^j = f(\sigma_t^i)$$

This is a **coupled oscillator** — even without adversarial input, the mutual triggering can exhaust resources (DoS by infinite relay). The "infection" here is not malicious content but resource exhaustion.

**In branching process terms:** $R_0$ for the bilateral loop is:

$$R_0^{\text{loop}} = p_{ij} \cdot p_{ji}$$

If both $p_{ij}, p_{ji} > 0$, the loop has $R_0 > 0$ and will persist indefinitely once triggered (the "infection" bounces back and forth). The loop terminates only when one agent enters state $F$ (resource exhaustion).

**Duration until failure:** If each relay step takes time $\Delta t$ and failure occurs after $K$ steps:
$$T_{\text{fail}} = K \cdot \Delta t, \quad K \sim \text{Geometric}(r)$$
$$\mathbb{E}[T_{\text{fail}}] = \frac{\Delta t}{r}$$

**Meta-MAPG correction:** Term 2 (own future anticipation) for agent $i$:
$$\frac{\partial V^i}{\partial \phi_1^i} \cdot \frac{\partial \phi_1^i}{\partial \phi_0^i}$$

The relay generates trajectory data $\tau_0 = (m_i \to j, m_j \to i, m_i \to j, \ldots)$ that contains the loop pattern. A Term-2-aware agent recognises: "my future policy, trained on this trajectory, will continue the loop → resource exhaustion → failure." The Term 2 contribution is:

$$\frac{\partial V^i}{\partial \phi_1^i} < 0 \quad (\text{future policy leads to failure})$$
$$\frac{\partial \phi_1^i}{\partial \phi_0^i} > 0 \quad (\text{current relay action → future relay policy})$$
$$\text{Term 2} < 0 \quad (\text{penalty for continuing relay})$$

**Detection condition:** The loop is detectable from Term 2 when the meta-learning horizon $L$ satisfies:

$$L \geq L^* = \frac{1}{\Delta t} \cdot \mathbb{E}[T_{\text{fail}}] = \frac{1}{r}$$

i.e., the meta-horizon must extend far enough to "see" the resource exhaustion. For short-horizon agents ($L < L^*$), the loop is invisible to Term 2 and continues until failure. This is a testable prediction: **agents with longer meta-horizons should break relay loops faster.**

---

### 2.2 Type II: Data Poisoning Chain (CS10)

**Structure:** An adversary corrupts an external data source (GitHub Gist). Agent $i$ consumes the data, becomes compromised, and then takes actions that corrupt shared resources (files, databases) that other agents consume.

**Formal model:** This is a **directed chain** on the graph:
$$\text{Adversary} \to \text{External Data} \to i \to \text{Shared Resource} \to j_1, j_2, \ldots, j_k$$

The branching number at each step:

| Step | Source → Target | Branching Factor |
|------|----------------|-----------------|
| 1 | Adversary → External Data | 1 (targeted) |
| 2 | External Data → Agent $i$ | 1 (agent reads data) |
| 3 | Agent $i$ → Shared Resource | 1 (agent writes to shared resource) |
| 4 | Shared Resource → Agents $j$ | $k = |N_{\text{read}}|$ (all agents reading the resource) |

The total amplification is:
$$R_0^{\text{chain}} = k \cdot p_{\text{read}} \cdot q_{\text{poison}}$$

where $p_{\text{read}}$ is the probability an agent reads the shared resource and $q_{\text{poison}}$ is the probability the poisoned resource corrupts the reading agent.

**The danger:** Step 4 is a **one-to-many amplification**. A single corrupted shared resource can compromise all $k$ agents that read from it. In AoC, all 6 agents potentially read from the same file system, so $k = N - 1 = 5$.

**Meta-MAPG correction:** Term 2 for agent $i$:

The key insight is that Term 2 should penalise writing to shared resources when the data source is uncertain. Agent $i$'s future trajectories depend on the state of shared resources it writes to:

$$\phi_1^i = U(\phi_0^i, \tau_0) \quad \text{where } \tau_0 \text{ includes corrupted data}$$

Term 2 captures: "if I write poisoned data to the shared resource, my future self may re-read it and compound the corruption." This creates **write hesitation** — an agent with Term 2 awareness is more cautious about writing to shared resources when its input data is uncertain.

Term 3 adds the **social amplification penalty**:
$$\sum_{j \neq i} \frac{\partial V^i}{\partial \phi_1^j} \cdot \frac{\partial \phi_1^j}{\partial \phi_0^i}$$

For each agent $j$ that reads the shared resource: writing poisoned data shifts $\phi_1^j$ toward compromised behaviour, which degrades $i$'s environment. The penalty scales with $k$ (number of readers), creating stronger write hesitation for more widely-shared resources.

**Testable prediction:** Meta-MAPG agents should exhibit **resource-proportional caution** — more careful when writing to shared resources read by many agents than resources read by few. This is analogous to the principle of least privilege in security.

---

### 2.3 Type III: Broadcast Cascade (CS8, CS11)

**Structure:** Agent $i$ broadcasts a message to ALL neighbours simultaneously (Discord announcement, mass email). If the broadcast contains adversarial content, all recipients are exposed at once.

This is the most dangerous cascade type because the **branching factor equals the degree** $d_i$ at the broadcasting node.

**Formal model:** At time $t = 0$, compromised agent $i$ broadcasts:

$$\text{New compromises at } t = 1: \quad |\{j \in N(i) : \sigma_0^j = S, \sigma_1^j = C\}| \sim \text{Binomial}(|N_S(i)|, p_{\text{broadcast}})$$

where $N_S(i) = \{j \in N(i) : \sigma_0^j = S\}$ is the set of safe neighbours.

Expected new infections from a single broadcast:
$$\mathbb{E}[\text{new infections}] = |N_S(i)| \cdot p_{\text{broadcast}} \leq d_i \cdot p_{\text{broadcast}}$$

If each newly compromised agent also broadcasts (the cascade continues), the process becomes a **multi-type branching process** on $G$.

---

## 3. Branching Process Theory for Agent Cascades

### 3.1 The General Branching Process

Define the generation-$k$ compromise count $Z_k$: the number of agents first compromised at step $k$.

$$Z_0 = 1 \quad (\text{initial compromised agent})$$
$$Z_{k+1} = \sum_{i=1}^{Z_k} X_i^{(k)}$$

where $X_i^{(k)}$ is the number of new agents compromised by the $i$-th agent in generation $k$.

Under the mean-field approximation (large $N$, ignoring depletion of susceptible agents):

$$\mathbb{E}[X_i^{(k)}] \approx \bar{d} \cdot p_{\text{prop}}$$

where $\bar{d}$ is mean degree and $p_{\text{prop}}$ is the average propagation probability.

### 3.2 The Basic Reproduction Number

$$R_0 = \mathbb{E}[X_i] = \bar{d} \cdot p_{\text{prop}}$$

Standard branching process results:

| Condition | Behaviour | Cascade Size |
|-----------|-----------|-------------|
| $R_0 < 1$ | **Subcritical** | Finite a.s., $\mathbb{E}[\text{total}] = \frac{1}{1 - R_0}$ |
| $R_0 = 1$ | **Critical** | Finite a.s., $\mathbb{E}[\text{total}] = \infty$ (heavy-tailed) |
| $R_0 > 1$ | **Supercritical** | Positive probability of infecting $\Theta(N)$ agents |

**AoC's CS11 was supercritical:** A single broadcast compromised multiple agents, suggesting $R_0 > 1$ in the experimental setup.

### 3.3 Heterogeneous Network Refinement

The mean-field $R_0 = \bar{d} \cdot p_{\text{prop}}$ ignores degree heterogeneity. For heterogeneous networks (which agent networks generally are — some agents are more connected than others), the correct threshold involves the **spectral radius** of the adjacency matrix.

Define the **next-generation matrix** $M \in \mathbb{R}^{N \times N}$:

$$M_{ij} = A_{ij} \cdot p_{ij}$$

The entry $M_{ij}$ is the expected number of times agent $j$ is compromised by agent $i$ (either 0 or $p_{ij}$).

**Theorem (Diekmann et al., 1990):** The cascade is supercritical if and only if the spectral radius $\rho(M) > 1$:

$$\rho(M) = \max_{\lambda \in \text{spec}(M)} |\lambda|$$

For the homogeneous case ($p_{ij} = p$ for all edges):
$$M = p \cdot A \implies \rho(M) = p \cdot \rho(A) = p \cdot \lambda_{\max}(A)$$

And we recover $R_0 = p \cdot \lambda_{\max}(A)$. For Erdős-Rényi graphs $G(N, q)$: $\lambda_{\max}(A) \approx Nq = \bar{d}$, recovering $R_0 \approx \bar{d} \cdot p$.

**For scale-free networks** (power-law degree distribution $P(d) \propto d^{-\gamma}$):
$$\lambda_{\max}(A) \approx \sqrt{d_{\max}} \gg \bar{d}$$

Hubs (highly connected agents) amplify cascades far beyond the mean-field prediction. In AoC, if one agent (say Doug) is treated as a hub by others (more trusted, more likely to be listened to), the effective spectral radius is higher than $\bar{d}$ suggests.

### 3.4 The Spectral Cascade Vulnerability Index

Define the **cascade vulnerability index** (CVI) for agent $i$ as the $i$-th component of the leading eigenvector of $M$:

$$M v_1 = \rho(M) v_1, \quad \text{CVI}(i) = (v_1)_i$$

**Interpretation:** CVI$(i)$ measures how much agent $i$ contributes to the cascade's growth rate. High-CVI agents are "superspreaders" — compromising them leads to the largest cascades.

In AoC terms:
- Agents with many communication channels (high degree) have high CVI
- Agents that are highly trusted by others (high $p_{ij}$ for edges pointing from $i$ to $j$) have high CVI
- The combination of high degree AND high trust is the most dangerous — a trusted hub

**Application to agent system design:** CVI analysis can identify which agents need the strongest safety guarantees. Deploy the most robust agents (strongest Term 1, most RLHF safety training) at high-CVI nodes.

---

## 4. The Meta-MAPG Cascade Damping Theorem

### 4.1 Setup

Consider $N$ agents on graph $G = (V, E)$ with next-generation matrix $M^{\text{ind}}$ under independent learning (Term 1 only). Each agent's propagation probability is:

$$p_{ij}^{\text{ind}} = \pi^j(\text{process} \mid m_i; \phi^j_{\text{ind}}) \cdot q(m_i)$$

Under Meta-MAPG, agent $j$'s policy includes Terms 2 and 3:

$$\nabla_{\phi_0^j} V^j = \underbrace{\nabla_{\phi_0^j} V^j|_{\text{direct}}}_{\text{Term 1}} + \underbrace{\frac{\partial V^j}{\partial \phi_1^j} \cdot \frac{\partial \phi_1^j}{\partial \phi_0^j}}_{\text{Term 2}} + \underbrace{\sum_{k \neq j} \frac{\partial V^j}{\partial \phi_1^k} \cdot \frac{\partial \phi_1^k}{\partial \phi_0^j}}_{\text{Term 3}}$$

### 4.2 Term 3's Effect on Propagation Probability

When agent $j$ decides whether to process message $m_i$ from compromised agent $i$, Term 3 adds a penalty for processing:

For each downstream agent $k \in N(j)$:
- If $j$ processes $m_i$ and becomes compromised, $j$ may propagate to $k$
- $\frac{\partial V^j}{\partial \phi_1^k}$: how does $k$'s future policy affect $j$'s value? If $k$ becomes compromised, $j$'s environment degrades. Sign: **positive** (better $k$ policy → better for $j$), but the perturbation is negative (compromise worsens $k$'s policy), so the net effect through the chain rule is negative
- $\frac{\partial \phi_1^k}{\partial \phi_0^j}$: how does $j$'s current action affect $k$'s future policy? If $j$ processes and propagates, $k$'s observations include $j$'s compromised actions, shifting $k$'s policy. Sign: **positive** (more processing → more propagation → $k$ affected)

The Term 3 contribution for agent $j$'s processing decision:

$$\Delta_3^j = \sum_{k \in N(j)} \underbrace{\frac{\partial V^j}{\partial \phi_1^k}}_{ > 0} \cdot \underbrace{\frac{\partial \phi_1^k}{\partial \phi_0^j}}_{\text{sign depends on action}}$$

For action $a = \text{process}$:
$$\frac{\partial \phi_1^k}{\partial \phi_0^j}\bigg|_{\text{process}} > 0 \quad \text{(processing corrupts } k\text{'s future data)}$$

But the VALUE change for $j$ when $k$ is corrupted is negative:
$$\frac{\partial V^j}{\partial \phi_1^k}\bigg|_{k \text{ corrupted}} < 0$$

Wait — let me be more precise. The chain rule gives:

$$\Delta_3^j\big|_{\text{process}} = \sum_{k \in N(j)} \frac{\partial V^j}{\partial \phi_1^k} \cdot \frac{\partial \phi_1^k}{\partial \phi_0^j}\bigg|_{\text{process}}$$

The key product: agent $k$'s policy worsens when $j$ propagates ($k$ receives corrupted messages → $k$'s updated policy is worse), so $\phi_1^k$ moves in the direction that DECREASES $V^j$. Thus the total Term 3 contribution for the "process" action is:

$$\Delta_3^j\big|_{\text{process}} < 0$$

This is a **penalty proportional to the out-degree** $|N(j)|$: the more agents downstream of $j$, the larger the penalty for processing suspicious messages.

### 4.3 Modified Propagation Probability

Under Meta-MAPG, the policy update shifts $\pi^j(\text{process})$ downward by an amount proportional to $|\Delta_3^j|$. Using a softmax policy with inverse temperature $\beta$:

$$\pi^j_{\text{meta}}(\text{process}) = \frac{\exp(\beta \cdot Q^j_{\text{process}})}{Z} \quad \text{where } Q^j_{\text{process}} = Q^j_{\text{ind,process}} + \Delta_3^j$$

For small $|\Delta_3^j|$ relative to $Q^j_{\text{ind}}$:

$$\pi^j_{\text{meta}}(\text{process}) \approx \pi^j_{\text{ind}}(\text{process}) \cdot \exp(\beta \cdot \Delta_3^j)$$

Since $\Delta_3^j < 0$:

$$p_{ij}^{\text{meta}} = p_{ij}^{\text{ind}} \cdot \exp(\beta \cdot \Delta_3^j) < p_{ij}^{\text{ind}}$$

### 4.4 Bounding the Damping

We need to bound $|\Delta_3^j|$. Making the following assumptions:
- **A1 (Bounded value sensitivity):** $\left|\frac{\partial V^j}{\partial \phi_1^k}\right| \leq B_V$ for all $j, k$
- **A2 (Bounded policy influence):** $\left|\frac{\partial \phi_1^k}{\partial \phi_0^j}\right| \leq B_\phi \cdot \alpha \cdot L$ where $\alpha$ is learning rate and $L$ is meta-horizon (the influence grows with learning rate and horizon)
- **A3 (Sign consistency):** The product $\frac{\partial V^j}{\partial \phi_1^k} \cdot \frac{\partial \phi_1^k}{\partial \phi_0^j}\big|_{\text{process}}$ is negative for all $k \in N(j)$

Under A1-A3:
$$|\Delta_3^j| \leq |N(j)| \cdot B_V \cdot B_\phi \cdot \alpha \cdot L$$

So:
$$p_{ij}^{\text{meta}} \leq p_{ij}^{\text{ind}} \cdot \exp\left(-\beta \cdot B_V B_\phi \cdot \alpha L \cdot |N(j)|\right)$$

Define $\eta = \beta \cdot B_V B_\phi > 0$ (a combined sensitivity parameter). Then:

$$\boxed{p_{ij}^{\text{meta}} \leq p_{ij}^{\text{ind}} \cdot \exp\left(-\eta \alpha L \cdot d_j^{\text{out}}\right)}$$

where $d_j^{\text{out}} = |N(j)|$ is agent $j$'s out-degree.

### 4.5 The Cascade Damping Theorem

**Theorem 1 (Cascade Damping).** Let $G = (V, E)$ be an agent communication network. Let $M^{\text{ind}}$ be the next-generation matrix under independent learning, with spectral radius $\rho^{\text{ind}} = \rho(M^{\text{ind}})$. Let $M^{\text{meta}}$ be the next-generation matrix under Meta-MAPG with meta-learning rate $\alpha$, meta-horizon $L$, and combined sensitivity $\eta > 0$. Then:

$$\rho(M^{\text{meta}}) \leq \rho(M^{\text{ind}}) \cdot \exp\left(-\eta \alpha L \cdot d_{\min}^{\text{out}}\right)$$

where $d_{\min}^{\text{out}} = \min_{j \in V} |N(j)|$ is the minimum out-degree.

**Proof sketch:**

1. The next-generation matrix entries satisfy:
   $$M_{ij}^{\text{meta}} = A_{ij} \cdot p_{ij}^{\text{meta}} \leq A_{ij} \cdot p_{ij}^{\text{ind}} \cdot e^{-\eta \alpha L d_j^{\text{out}}} \leq M_{ij}^{\text{ind}} \cdot e^{-\eta \alpha L d_{\min}^{\text{out}}}$$

2. Therefore $M^{\text{meta}} \leq e^{-\eta \alpha L d_{\min}^{\text{out}}} \cdot M^{\text{ind}}$ (element-wise).

3. For non-negative matrices, element-wise domination implies spectral radius domination (Perron-Frobenius):
   $$\rho(M^{\text{meta}}) \leq \rho\left(e^{-\eta \alpha L d_{\min}^{\text{out}}} \cdot M^{\text{ind}}\right) = e^{-\eta \alpha L d_{\min}^{\text{out}}} \cdot \rho(M^{\text{ind}})$$

$\square$

**Corollary 1 (Critical Threshold Expansion).** The cascade becomes subcritical ($\rho(M^{\text{meta}}) < 1$) whenever:

$$\rho(M^{\text{ind}}) < \exp\left(\eta \alpha L \cdot d_{\min}^{\text{out}}\right)$$

For independent learning, the cascade is subcritical only when $\rho(M^{\text{ind}}) < 1$. Meta-MAPG expands this to $\rho(M^{\text{ind}}) < e^{\eta \alpha L d_{\min}}$, a strictly larger region.

**Corollary 2 (Degree-Dependent Damping).** Using the tighter per-node bound:

$$M_{ij}^{\text{meta}} \leq M_{ij}^{\text{ind}} \cdot e^{-\eta \alpha L d_j^{\text{out}}}$$

Define the diagonal damping matrix $D = \text{diag}(e^{-\eta \alpha L d_1^{\text{out}}}, \ldots, e^{-\eta \alpha L d_N^{\text{out}}})$. Then:

$$M^{\text{meta}} \leq M^{\text{ind}} D$$

and:
$$\rho(M^{\text{meta}}) \leq \rho(M^{\text{ind}} D)$$

This is tighter than Theorem 1 because high-degree nodes get MORE damping — but the bound $\rho(M^{\text{ind}} D)$ is not closed-form (it requires numerical computation of the product's spectral radius, which depends on the specific graph structure). Theorem 1's closed-form exponential bound is more useful for asymptotic analysis; Corollary 2 is more useful for computational verification on specific networks. The qualitative insight holds in both: the agents with the most downstream connections (highest CVI) experience the strongest Meta-MAPG penalty — exactly the superspreaders get the most protection.

### 4.6 Interpretation

The damping has three "levers":

| Parameter | Meaning | Effect on $\rho(M^{\text{meta}})$ |
|-----------|---------|----------------------------------|
| $\alpha$ (learning rate) | How fast agents update from new data | Higher $\alpha$ → faster learning → more responsive Term 3 → more damping |
| $L$ (meta-horizon) | How far ahead agents model learning dynamics | Higher $L$ → agents anticipate more downstream effects → more damping |
| $d_j^{\text{out}}$ (out-degree) | Agent $j$'s number of downstream connections | Higher degree → larger Term 3 penalty → agent is more cautious |
| $\eta$ (sensitivity) | Combined value-sensitivity $\times$ policy-influence | Higher $\eta$ → stronger coupling between agents → more damping |

**The key insight:** damping is **multiplicative** in $\alpha L d$. This means:
- Even weak meta-learning ($\alpha L$ small) provides meaningful damping in dense networks (high $d$)
- Conversely, strong meta-learning ($\alpha L$ large) provides damping even in sparse networks
- The product $\alpha L d$ is the **effective meta-learning strength**: you can compensate for short horizons with dense connectivity, or vice versa

---

## 5. Phase Transition Analysis

### 5.1 The Critical Surface

The cascade undergoes a **phase transition** at $\rho(M) = 1$. Define the critical propagation probability $p^*$ as the value of $p$ (assuming homogeneous $p_{ij} = p$) at which the transition occurs:

**Under independent learning:**
$$p_{\text{ind}}^* = \frac{1}{\lambda_{\max}(A)}$$

**Under Meta-MAPG:**
$$p_{\text{meta}}^* = \frac{e^{\eta \alpha L d_{\min}}}{\lambda_{\max}(A)} > p_{\text{ind}}^*$$

The ratio:
$$\frac{p_{\text{meta}}^*}{p_{\text{ind}}^*} = e^{\eta \alpha L d_{\min}}$$

**Interpretation:** Meta-MAPG agents can tolerate a propagation probability that is $e^{\eta \alpha L d_{\min}}$ times higher than independent agents before cascades go critical. For the AoC network ($d_{\min} \approx 4$, assuming $\eta \alpha L \approx 0.5$):

$$\frac{p_{\text{meta}}^*}{p_{\text{ind}}^*} = e^{0.5 \times 4} = e^2 \approx 7.4$$

Meta-MAPG agents can withstand ~7x higher adversarial payload virulence before cascades become supercritical.

### 5.2 Network Topology Dependence

The phase transition depends on $\lambda_{\max}(A)$, which varies dramatically with network topology:

| Topology | $\lambda_{\max}(A)$ | $p_{\text{ind}}^*$ | Cascade Vulnerability |
|----------|---------------------|--------------------|-----------------------|
| Ring (each agent talks to 2 neighbours) | 2 | 0.5 | Low — sparse |
| Erdős-Rényi $G(N, q)$ | $\approx Nq$ | $\approx 1/(Nq)$ | Medium — scales with density |
| Complete graph $K_N$ | $N - 1$ | $\approx 1/(N-1)$ | High — any infection spreads |
| Star (one hub) | $\sqrt{N-1}$ | $\approx 1/\sqrt{N-1}$ | Very high through hub |
| Scale-free ($\gamma = 2.5$) | $\approx \sqrt{d_{\max}}$ | $\approx 1/\sqrt{d_{\max}}$ | Extremely high — hubs |

**AoC's near-complete graph** ($\lambda_{\max} \approx 5$ for 6 agents with ~complete connectivity) means $p_{\text{ind}}^* \approx 0.2$ — even modest payload virulence triggers supercritical cascades. This explains why CS11 (broadcast cascade) was so devastating.

### 5.3 Finite-Size Effects

For finite $N$ (AoC has $N = 6$), the branching process approximation breaks down because susceptible depletion matters. We need the **exact Markov chain** analysis.

State: $\boldsymbol{\sigma} = (\sigma^1, \ldots, \sigma^N) \in \{S, C, F\}^N$

Transition matrix: $P(\boldsymbol{\sigma}' \mid \boldsymbol{\sigma})$ defined by the infection and progression dynamics.

For small $N$, we can compute the **cascade size distribution** exactly by enumerating states:
$$\Pr(\text{cascade size} = k \mid \boldsymbol{\sigma}_0) = \sum_{\boldsymbol{\sigma}: |\{i: \sigma^i \neq S\}| = k} \Pr(\boldsymbol{\sigma} \text{ is absorbing} \mid \boldsymbol{\sigma}_0)$$

For $N = 6$ with 3 states each: $3^6 = 729$ possible states. Tractable for exact computation. This would make a strong simulation result — exact cascade size distributions for independent PG vs LOLA vs Meta-MAPG on the AoC network.

---

## 6. Connecting Back to Meta-MAPG Gradient Terms

### 6.1 Summary Table: Which Term Prevents Which Cascade Type?

| Cascade Type | AoC Cases | Primary Prevention Term | Mechanism |
|:---:|:---:|:---:|:---:|
| Type I (Bilateral Loop) | CS4 | **Term 2** | Agent models own future resource state — detects loop pattern |
| Type II (Data Poisoning) | CS10 | **Terms 2 + 3** | Term 2: caution about poisoned self-data. Term 3: penalty for corrupting shared resources |
| Type III (Broadcast) | CS8, CS11 | **Term 3** | Penalty proportional to out-degree for broadcasting unverified information |

### 6.2 Gradient Magnitude Ordering

In the cascade context, the three terms have a natural ordering by magnitude:

$$|\text{Term 1}| \gg |\text{Term 2}| \geq |\text{Term 3}|$$

in general, but for **cascade-relevant actions** (broadcasting, writing to shared resources), Term 3 can dominate because it scales with degree:

$$|\text{Term 3}|_{\text{broadcast}} = O(d_i \cdot \alpha L \cdot B_V B_\phi)$$

For high-degree nodes (hubs), Term 3 can exceed Term 1. This creates a natural "circuit breaker" effect: the most connected agents are also the most cautious, automatically implementing the hub-protection strategy that network epidemiology recommends.

---

## 7. Simulation Design

### 7.1 Environment Specification

**Graph families to test:**
1. Complete graph $K_N$ (AoC-like, worst case)
2. Erdős-Rényi $G(N, p_{\text{edge}})$ with varying density
3. Scale-free (Barabási-Albert) with varying $m$ parameter
4. Ring lattice (best case — minimal connectivity)

**Agent policies:**
1. Independent PG (Term 1 only)
2. LOLA (Terms 1 + linearised 2+3)
3. Meta-MAPG (Terms 1 + 2 + 3)

**Metrics:**
- Cascade size: $|\{i : \sigma_T^i \neq S\}|$
- Time to containment: first $t$ when no new infections occur
- Cascade probability: fraction of trials where cascade size exceeds $N/2$
- $\hat{R}_0$: empirical estimate from cascade data

### 7.2 Experimental Protocol

For each (graph topology, algorithm, $N$) triple:
1. Generate graph $G$
2. Initialise all agents in state $S$
3. Compromise a random agent (or highest-CVI agent)
4. Run cascade dynamics for $T = 100$ steps
5. Record cascade trajectory
6. Repeat for 1000 trials
7. Compute statistics: mean cascade size, cascade probability, $\hat{R}_0$

### 7.3 Key Plots

1. **Cascade size vs mean degree $\bar{d}$:** Should show Meta-MAPG maintaining subcritical cascades at higher $\bar{d}$ than independent PG — confirming the critical threshold expansion
2. **$\hat{R}_0$ vs algorithm:** Bar chart for each graph type. Meta-MAPG should consistently have lowest $\hat{R}_0$
3. **Phase transition curve:** $\Pr(\text{large cascade})$ vs $p_{\text{prop}}$. Should show Meta-MAPG's critical threshold shifted right
4. **Degree-dependent damping:** Plot per-agent propagation probability vs degree. Meta-MAPG agents should show decreasing $p_{\text{prop}}$ with increasing degree (hub protection)
5. **Cascade size distribution:** For supercritical independent PG parameters, compare the distribution tails — Meta-MAPG should have exponentially lighter tails

### 7.4 Implementation Notes (PyTorch)

```python
# Core cascade simulator (sketch)
class CascadeSimulator:
    def __init__(self, G, agents, p_payload):
        self.G = G  # networkx graph
        self.agents = agents  # dict: node_id -> AgentPolicy
        self.p_payload = p_payload

    def step(self, state):
        """One cascade step: compromised agents attempt to propagate."""
        new_state = state.copy()
        for i in self.G.nodes():
            if state[i] == 'C':
                for j in self.G.neighbors(i):
                    if state[j] == 'S':
                        # Agent j decides whether to process
                        p_process = self.agents[j].process_prob(
                            message_from=i,
                            state=state
                        )
                        if np.random.random() < p_process * self.p_payload:
                            new_state[j] = 'C'
                # Compromised agent may fail
                if np.random.random() < self.r_fail:
                    new_state[i] = 'F'
        return new_state
```

The key difference between algorithms: how `process_prob` is computed.
- Independent PG: `process_prob` depends only on message content and j's own state
- LOLA: `process_prob` additionally depends on first-order model of i's future behaviour
- Meta-MAPG: `process_prob` depends on full multi-step model of all neighbours' future policies

---

## 8. Connection to Dissertation Chapters

| Section of This Document | Dissertation Chapter | Role |
|:---:|:---:|:---:|
| §1 (Network Model) | Ch.6 §6.2 | Setup for cascade analysis after proving Meta-MAPG theorem |
| §2 (Three Cascade Types) | Ch.2 §2.4 | Literature review — AoC cascade taxonomy |
| §3 (Branching Process) | Ch.7 §7.2 | Mathematical tools for convergence analysis |
| §4 (Damping Theorem) | Ch.7 §7.3 | **Main safety result** — Theorem 7.3 |
| §5 (Phase Transition) | Ch.7 §7.4 | Corollaries and network topology analysis |
| §6 (Term Mapping) | Ch.6 §6.4 | Interpretation section — connecting theorem to AoC |
| §7 (Simulation Design) | Ch.10 §10.3 | Third simulation environment (cascade network) |

### Priority for Dissertation Writing

1. **Theorem 1 (§4.5)** — this is publication-worthy. The formal cascade damping result with spectral radius bounds. Needs to be stated precisely in Chapter 7 with full proof (not just sketch).
2. **Phase transition analysis (§5)** — direct computational results for specific graph topologies. Good for Chapter 7 corollaries.
3. **Simulation (§7)** — empirical validation. Even if the theorem is approximate, simulations confirm the qualitative prediction on realistic network sizes.
4. **CVI analysis (§3.4)** — practical design principle for deploying safe multi-agent systems. Good for Chapter 11 (implications).
