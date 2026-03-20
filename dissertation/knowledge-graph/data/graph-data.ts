import { GraphNode, GraphEdge } from "../types/graph";

// ============================================================
// CHAPTERS
// ============================================================
const chapters: GraphNode[] = [
  {
    id: "ch1-introduction",
    label: "Ch 1: Introduction",
    category: "chapter",
    chapterNum: 1,
    status: "todo",
    description: "Motivation, contributions, dissertation roadmap.",
    core: true,
  },
  {
    id: "ch2-lit-review",
    label: "Ch 2: Literature Review",
    category: "chapter",
    chapterNum: 2,
    status: "todo",
    description: "Survey of Kim et al., HyperSteer, cooperative/Stackelberg games.",
    core: true,
  },
  {
    id: "ch3-policy-methods",
    label: "Ch 3: Policy Methods in RL",
    category: "chapter",
    chapterNum: 3,
    status: "draft",
    description: "MDPs, policies, value functions, policy gradient methods, REINFORCE, actor-critic. Single-agent foundation.",
    core: true,
  },
  {
    id: "ch4-pg-theorem-proof",
    label: "Ch 4: Policy Gradient Theorem",
    category: "chapter",
    chapterNum: 4,
    status: "todo",
    description: "Full proof of the PG theorem from Sutton Ch.13, worked through in detail.",
    core: true,
  },
  {
    id: "ch5-meta-learning-marl",
    label: "Ch 5: Meta-learning in MARL",
    category: "chapter",
    chapterNum: 5,
    status: "todo",
    description: "Multi-agent RL setup, Foerster's work, LOLA, meta-learning motivation.",
    core: true,
  },
  {
    id: "ch6-meta-mapg-theorem",
    label: "Ch 6: Meta-MAPG Theorem",
    category: "chapter",
    chapterNum: 6,
    status: "todo",
    description: "Full proof of the Meta-Multi-Agent Policy Gradient theorem from Kim et al.",
    core: true,
  },
  {
    id: "ch7-convergence",
    label: "Ch 7: Convergence Guarantees",
    category: "chapter",
    chapterNum: 7,
    status: "todo",
    description: "AMBITIOUS: Convergence analysis for meta-MAPG. Open research question.",
    core: false,
  },
  {
    id: "ch8-llm-steering",
    label: "Ch 8: LLM Steering",
    category: "chapter",
    chapterNum: 8,
    status: "todo",
    description: "Rewrite of ST310 project. Frozen LLM + hypernetwork as RL agents.",
    core: true,
  },
  {
    id: "ch9-cooperative-game",
    label: "Ch 9: Cooperative Steering Game",
    category: "chapter",
    chapterNum: 9,
    status: "todo",
    description: "Rewrite of whitepaper. Two-agent cooperative formulation for LLM steering.",
    core: true,
  },
  {
    id: "ch10-simulations",
    label: "Ch 10: Simulations",
    category: "chapter",
    chapterNum: 10,
    status: "todo",
    description: "Reproduce simulations from ST310 and Kim et al.",
    core: true,
  },
  {
    id: "ch11-conclusion",
    label: "Ch 11: Conclusion",
    category: "chapter",
    chapterNum: 11,
    status: "todo",
    description: "Summary of contributions, limitations, future work.",
    core: false,
  },
];

// ============================================================
// PAPERS
// ============================================================
const papers: GraphNode[] = [
  {
    id: "kim2021",
    label: "Kim et al. (2021)",
    category: "paper",
    authors: ["Dong-Ki Kim", "Miao Liu", "Matthew Riemer", "Chuangchuang Sun", "Marwa Abdulhai", "Golnaz Habibi", "Sebastian Lopez-Cot", "Gerald Tesauro", "Jonathan P. How"],
    year: 2021,
    venue: "arXiv",
    url: "https://arxiv.org/abs/2011.00382",
    description: "A policy gradient algorithm for learning to learn in multiagent RL. Derives the Meta-MAPG theorem.",
    chapters: ["ch2-lit-review", "ch5-meta-learning-marl", "ch6-meta-mapg-theorem", "ch10-simulations"],
    core: true,
  },
  {
    id: "sutton2018",
    label: "Sutton & Barto (2018)",
    category: "paper",
    authors: ["Richard S. Sutton", "Andrew G. Barto"],
    year: 2018,
    venue: "MIT Press",
    url: "http://incompleteideas.net/book/the-book-2nd.html",
    description: "Reinforcement Learning: An Introduction. The foundational RL textbook. Ch.3 (MDPs) and Ch.13 (Policy Gradient Methods).",
    chapters: ["ch3-policy-methods", "ch4-pg-theorem-proof"],
    core: true,
  },
  {
    id: "williams1992",
    label: "Williams (1992)",
    category: "paper",
    authors: ["Ronald J. Williams"],
    year: 1992,
    venue: "Machine Learning",
    description: "Simple statistical gradient-following algorithms for connectionist RL. Introduces REINFORCE.",
    chapters: ["ch3-policy-methods", "ch4-pg-theorem-proof"],
    core: true,
  },
  {
    id: "foerster2018lola",
    label: "Foerster et al. (2018) — LOLA",
    category: "paper",
    authors: ["Jakob Foerster", "Richard Y. Chen", "Maruan Al-Shedivat", "Shimon Whiteson", "Pieter Abbeel", "Igor Mordatch"],
    year: 2018,
    venue: "AAMAS",
    url: "https://arxiv.org/abs/1709.04326",
    description: "Learning with Opponent-Learning Awareness. Agents account for the learning steps of other agents.",
    chapters: ["ch2-lit-review", "ch5-meta-learning-marl"],
    core: true,
  },
  {
    id: "foerster2018dice",
    label: "Foerster et al. (2018) — DiCE",
    category: "paper",
    authors: ["Jakob Foerster", "Gregory Farquhar", "Maruan Al-Shedivat", "Tim Rocktäschel", "Eric P. Xing", "Shimon Whiteson"],
    year: 2018,
    venue: "ICML",
    description: "DiCE: The Infinitely Differentiable Monte-Carlo Estimator. Enables higher-order gradient estimation.",
    chapters: ["ch5-meta-learning-marl"],
    core: false,
  },
  {
    id: "wen2019",
    label: "Wen et al. (2019)",
    category: "paper",
    authors: ["Ying Wen", "Yaodong Yang", "Rui Luo", "Jun Wang", "Wei Pan"],
    year: 2019,
    venue: "NeurIPS",
    url: "https://arxiv.org/abs/1901.09207",
    description: "Probabilistic Recursive Reasoning for Multi-Agent RL. Introduces conditional independence assumption used in Kim et al.'s proof.",
    chapters: ["ch5-meta-learning-marl", "ch6-meta-mapg-theorem"],
    core: true,
  },
  {
    id: "wei2018",
    label: "Wei et al. (2018)",
    category: "paper",
    authors: ["Ermo Wei", "Drew Wicke", "David Freelan", "Sean Luke"],
    year: 2018,
    venue: "arXiv",
    description: "Multiagent Soft Q-Learning. Soft Q-learning in multi-agent settings, used in Kim et al.'s proof.",
    chapters: ["ch6-meta-mapg-theorem"],
    core: false,
  },
  {
    id: "finn2017maml",
    label: "Finn et al. (2017) — MAML",
    category: "paper",
    authors: ["Chelsea Finn", "Pieter Abbeel", "Sergey Levine"],
    year: 2017,
    venue: "ICML",
    url: "https://arxiv.org/abs/1703.03400",
    description: "Model-Agnostic Meta-Learning. Foundation for meta-learning via learning to learn through gradient updates.",
    chapters: ["ch5-meta-learning-marl"],
    core: true,
  },
  {
    id: "schulman2017ppo",
    label: "Schulman et al. (2017) — PPO",
    category: "paper",
    authors: ["John Schulman", "Filip Wolski", "Prafulla Dhariwal", "Alec Radford", "Oleg Klimov"],
    year: 2017,
    venue: "arXiv",
    url: "https://arxiv.org/abs/1707.06347",
    description: "Proximal Policy Optimization. Trust-region-like policy gradient with clipped surrogate objective.",
    chapters: ["ch3-policy-methods", "ch8-llm-steering"],
    core: false,
  },
  {
    id: "schulman2015trpo",
    label: "Schulman et al. (2015) — TRPO",
    category: "paper",
    authors: ["John Schulman", "Sergey Levine", "Philipp Moritz", "Michael I. Jordan", "Pieter Abbeel"],
    year: 2015,
    venue: "ICML",
    description: "Trust Region Policy Optimization. Monotonic improvement guarantees for policy gradient.",
    chapters: ["ch3-policy-methods"],
    core: false,
  },
  {
    id: "hypersteer",
    label: "HyperSteer (Mateusz)",
    category: "paper",
    authors: ["Mateusz"],
    year: 2025,
    description: "Hypernetwork-based LLM steering. Key reference for Chapter 8.",
    chapters: ["ch2-lit-review", "ch8-llm-steering"],
    core: true,
  },
  {
    id: "st310-project",
    label: "ST310 Course Project",
    category: "paper",
    authors: ["Yevhen Shcherbinin"],
    year: 2025,
    description: "Original course project on LLM steering using RL. Foundation for Chapters 8-9.",
    chapters: ["ch8-llm-steering", "ch9-cooperative-game", "ch10-simulations"],
    core: true,
  },
  {
    id: "littman1994",
    label: "Littman (1994)",
    category: "paper",
    authors: ["Michael L. Littman"],
    year: 1994,
    venue: "ICML",
    description: "Markov games as a framework for multi-agent RL. Introduces stochastic/Markov games formalism.",
    chapters: ["ch5-meta-learning-marl"],
    core: false,
  },
  {
    id: "lowe2017maddpg",
    label: "Lowe et al. (2017) — MADDPG",
    category: "paper",
    authors: ["Ryan Lowe", "Yi Wu", "Aviv Tamar", "Jean Harb", "Pieter Abbeel", "Igor Mordatch"],
    year: 2017,
    venue: "NeurIPS",
    description: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.",
    chapters: ["ch2-lit-review", "ch5-meta-learning-marl"],
    core: false,
  },
  {
    id: "zhang2021marl-survey",
    label: "Zhang et al. (2021) — MARL Survey",
    category: "paper",
    authors: ["Kaiqing Zhang", "Zhuoran Yang", "Tamer Başar"],
    year: 2021,
    venue: "Applied Mathematics & Optimization",
    description: "Multi-Agent RL: A Selective Overview of Theories and Algorithms.",
    chapters: ["ch2-lit-review", "ch5-meta-learning-marl"],
    core: false,
  },
  {
    id: "stackelberg-games",
    label: "Stackelberg Games (Lit)",
    category: "paper",
    description: "Literature on Stackelberg/leader-follower games in MARL context.",
    chapters: ["ch2-lit-review", "ch9-cooperative-game"],
    core: false,
    tags: ["game-theory", "cooperative"],
  },
  {
    id: "konda2000ac",
    label: "Konda & Tsitsiklis (2000)",
    category: "paper",
    authors: ["Vijay R. Konda", "John N. Tsitsiklis"],
    year: 2000,
    venue: "NeurIPS",
    description: "Actor-Critic Algorithms. Convergence proofs for two-timescale actor-critic methods.",
    chapters: ["ch3-policy-methods", "ch7-convergence"],
    core: false,
  },
  {
    id: "calvano2020",
    label: "Calvano et al. (2020)",
    category: "paper",
    authors: ["Emiliano Calvano", "Giacomo Calzolari", "Vincenzo Denicolò", "Sergio Pastorello"],
    year: 2020,
    venue: "AER",
    description: "Artificial Intelligence, Algorithmic Pricing, and Collusion. Shows Q-learning agents learn to collude in Bertrand competition.",
    chapters: ["ch2-lit-review", "ch10-simulations"],
    core: true,
  },
  {
    id: "prelim-presentation",
    label: "Prelim Presentation (Meta-MAPG Collusion)",
    category: "paper",
    authors: ["Yevhen Shcherbinin"],
    year: 2025,
    description: "Preliminary dissertation presentation. Meta-MAPG applied to algorithmic collusion detection in Bertrand competition.",
    chapters: ["ch1-introduction", "ch5-meta-learning-marl", "ch6-meta-mapg-theorem", "ch10-simulations"],
    core: true,
  },
  {
    id: "ouyang2022rlhf",
    label: "Ouyang et al. (2022) — InstructGPT",
    category: "paper",
    authors: ["Long Ouyang", "Jeff Wu", "Xu Jiang"],
    year: 2022,
    venue: "NeurIPS",
    description: "Training language models to follow instructions with human feedback. RLHF for LLMs.",
    chapters: ["ch8-llm-steering"],
    core: false,
  },
  {
    id: "agents-of-chaos-2026",
    label: "Agents of Chaos (2026)",
    category: "paper",
    authors: ["Bau Lab"],
    year: 2026,
    venue: "arXiv",
    url: "https://arxiv.org/abs/2602.20021",
    description: "Red-teaming study of 6 autonomous LLM agents. Documents 10 vulnerabilities and 6 emergent safety behaviours including spontaneous cross-agent coordination — empirical evidence for multi-agent learning dynamics that Meta-MAPG formalises.",
    chapters: ["ch2-lit-review", "ch5-meta-learning-marl", "ch8-llm-steering", "ch9-cooperative-game"],
    core: true,
  },
];

// ============================================================
// CONCEPTS
// ============================================================
const concepts: GraphNode[] = [
  {
    id: "mdp",
    label: "Markov Decision Process",
    category: "concept",
    description: "Tuple (S, A, P, R, γ) formalising sequential decision-making under uncertainty.",
    chapters: ["ch3-policy-methods"],
    core: true,
  },
  {
    id: "markov-property",
    label: "Markov Property",
    category: "concept",
    description: "Future states depend on past only through current state. Memoryless.",
    chapters: ["ch3-policy-methods"],
    core: true,
  },
  {
    id: "policy",
    label: "Policy π(a|s)",
    category: "concept",
    description: "Mapping from states to distributions over actions. Can be stochastic or deterministic.",
    chapters: ["ch3-policy-methods"],
    core: true,
  },
  {
    id: "parametrised-policy",
    label: "Parametrised Policy π(a|s,θ)",
    category: "concept",
    description: "Policy with learnable parameters θ ∈ ℝ^d. Foundation of policy gradient methods.",
    chapters: ["ch3-policy-methods"],
    core: true,
  },
  {
    id: "state-value-fn",
    label: "State-Value Function v_π(s)",
    category: "concept",
    description: "Expected discounted return from state s under policy π.",
    chapters: ["ch3-policy-methods"],
    core: true,
  },
  {
    id: "action-value-fn",
    label: "Action-Value Function q_π(s,a)",
    category: "concept",
    description: "Expected discounted return from state s, taking action a, then following π.",
    chapters: ["ch3-policy-methods"],
    core: true,
  },
  {
    id: "advantage-fn",
    label: "Advantage Function A_π(s,a)",
    category: "concept",
    description: "A_π(s,a) = q_π(s,a) − v_π(s). How much better action a is than average under π.",
    chapters: ["ch3-policy-methods"],
    core: true,
  },
  {
    id: "discounted-return",
    label: "Discounted Return G_t",
    category: "concept",
    description: "G_t = Σ γ^k R_{t+k+1}. Cumulative discounted reward from time t.",
    chapters: ["ch3-policy-methods"],
    core: true,
  },
  {
    id: "bellman-equations",
    label: "Bellman Equations",
    category: "concept",
    description: "Recursive consistency conditions for value functions. v_π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R + γ v_π(s')].",
    chapters: ["ch3-policy-methods"],
    core: true,
  },
  {
    id: "on-policy-distribution",
    label: "On-Policy Distribution μ_π(s)",
    category: "concept",
    description: "Normalised (discounted) state visitation frequency under policy π. Key to PG theorem.",
    chapters: ["ch3-policy-methods", "ch4-pg-theorem-proof"],
    core: true,
  },
  {
    id: "performance-measure",
    label: "Performance Measure J(θ)",
    category: "concept",
    description: "Scalar objective. Episodic: J(θ) = v_{π_θ}(s_0). Continuing: average reward.",
    chapters: ["ch3-policy-methods"],
    core: true,
  },
  {
    id: "log-derivative-trick",
    label: "Log-Derivative Trick",
    category: "concept",
    description: "∇π(a|s,θ) = π(a|s,θ) ∇ln π(a|s,θ). Converts gradient into expectation for MC estimation.",
    chapters: ["ch3-policy-methods", "ch4-pg-theorem-proof"],
    core: true,
  },
  {
    id: "td-error",
    label: "TD Error δ_t",
    category: "concept",
    description: "δ_t = r_{t+1} + γ v̂(s_{t+1}) − v̂(s_t). One-step estimate of advantage.",
    chapters: ["ch3-policy-methods"],
    core: false,
  },
  {
    id: "baseline",
    label: "Baseline b(s)",
    category: "concept",
    description: "State-dependent scalar subtracted from return for variance reduction. Does not bias gradient.",
    chapters: ["ch3-policy-methods"],
    core: true,
  },
  {
    id: "trajectory",
    label: "Trajectory τ",
    category: "concept",
    description: "Complete episode: τ = (s_0, a_0, r_1, ..., s_H). p(τ|π) factorises over policy and dynamics.",
    chapters: ["ch3-policy-methods", "ch5-meta-learning-marl"],
    core: true,
  },
  {
    id: "n-agent-game",
    label: "N-Agent Stochastic Game",
    category: "concept",
    description: "Multi-agent generalisation of MDP: M_n = ⟨I, S, A, P, R, γ⟩ with joint actions and per-agent rewards.",
    chapters: ["ch5-meta-learning-marl", "ch6-meta-mapg-theorem"],
    core: true,
  },
  {
    id: "conditional-independence",
    label: "Conditional Independence (Wen)",
    category: "concept",
    description: "π(aⁱ, a⁻ⁱ|s) = π(aⁱ|s) π(a⁻ⁱ|s). Key assumption in Kim et al.'s proof.",
    chapters: ["ch6-meta-mapg-theorem"],
    core: true,
  },
  {
    id: "markovian-update",
    label: "Markovian Update Function",
    category: "concept",
    description: "Each agent updates policy via a Markovian function after K trajectories. Sequential dependency: φ₁ depends on φ₀.",
    chapters: ["ch5-meta-learning-marl", "ch6-meta-mapg-theorem"],
    core: true,
  },
  {
    id: "meta-value-fn",
    label: "Meta-Value Function",
    category: "concept",
    description: "Evaluates reward given current policy AND future updates from that policy. Models the learning process itself.",
    chapters: ["ch5-meta-learning-marl", "ch6-meta-mapg-theorem"],
    core: true,
  },
  {
    id: "inner-loop",
    label: "Inner Loop (Policy Update)",
    category: "concept",
    description: "Single policy gradient step within one meta-learning iteration.",
    chapters: ["ch5-meta-learning-marl", "ch6-meta-mapg-theorem"],
    core: false,
  },
  {
    id: "outer-loop",
    label: "Outer Loop (Meta-Optimisation)",
    category: "concept",
    description: "Optimisation over the learning process itself — choosing initial parameters that lead to good post-update policies.",
    chapters: ["ch5-meta-learning-marl", "ch6-meta-mapg-theorem"],
    core: false,
  },
  {
    id: "non-stationarity",
    label: "Non-Stationarity in MARL",
    category: "concept",
    description: "Each agent's environment is non-stationary because other agents are simultaneously learning and changing their policies.",
    chapters: ["ch5-meta-learning-marl"],
    core: true,
  },
  {
    id: "cooperative-game",
    label: "Cooperative Game",
    category: "concept",
    description: "Agents share an objective or have aligned incentives. The LLM steering game is cooperative.",
    chapters: ["ch2-lit-review", "ch9-cooperative-game"],
    core: true,
  },
  {
    id: "hypernetwork",
    label: "Hypernetwork",
    category: "concept",
    description: "A neural network that generates weights for another network. Used in HyperSteer for LLM steering.",
    chapters: ["ch8-llm-steering", "ch9-cooperative-game"],
    core: true,
  },
  {
    id: "frozen-llm",
    label: "Frozen LLM (as Agent)",
    category: "concept",
    description: "Pre-trained LLM with frozen weights treated as one agent in a two-agent cooperative game.",
    chapters: ["ch8-llm-steering", "ch9-cooperative-game"],
    core: true,
  },
  {
    id: "bertrand-competition",
    label: "Bertrand Competition",
    category: "concept",
    description: "Price-setting oligopoly game. Firms simultaneously choose prices; consumers buy from cheapest. Key environment for algorithmic collusion experiments.",
    chapters: ["ch10-simulations"],
    core: true,
  },
  {
    id: "algorithmic-collusion",
    label: "Algorithmic Collusion",
    category: "concept",
    description: "Autonomous pricing agents converging to supra-competitive prices without explicit communication. Meta-MAPG can detect and analyse this.",
    chapters: ["ch10-simulations"],
    core: true,
  },
  {
    id: "three-term-gradient",
    label: "Three-Term Gradient Decomposition",
    category: "concept",
    description: "Meta-MAPG gradient decomposes into: (1) direct current-policy gradient, (2) own future learning anticipation, (3) peer learning anticipation. Terms 2-3 are what meta-learning adds.",
    chapters: ["ch6-meta-mapg-theorem"],
    core: true,
  },
  {
    id: "higher-order-gradients",
    label: "Higher-Order Gradients",
    category: "concept",
    description: "Differentiating through the learning process requires ∇²(gradient of gradient). Implemented via torch.autograd in PyTorch.",
    chapters: ["ch5-meta-learning-marl", "ch6-meta-mapg-theorem", "ch10-simulations"],
    core: true,
  },
  {
    id: "equilibrium-learning",
    label: "Equilibrium Learning",
    category: "concept",
    description: "Agents learning policies to satisfy game-theoretic equilibrium conditions. Most based on off-policy Q-learning.",
    chapters: ["ch5-meta-learning-marl"],
    core: false,
  },
  {
    id: "nash-equilibrium",
    label: "Nash Equilibrium",
    category: "concept",
    description: "Strategy profile where no agent can unilaterally improve. Standard solution concept in game theory.",
    chapters: ["ch5-meta-learning-marl", "ch7-convergence"],
    core: false,
  },
  {
    id: "stackelberg-equilibrium",
    label: "Stackelberg Equilibrium",
    category: "concept",
    description: "Leader-follower equilibrium. Leader commits to strategy, follower best-responds.",
    chapters: ["ch2-lit-review", "ch9-cooperative-game"],
    core: false,
  },
  {
    id: "emergent-coordination",
    label: "Emergent Multi-Agent Coordination",
    category: "concept",
    description: "Agents spontaneously developing cooperative policies without explicit coordination protocol. Observed empirically in Agents of Chaos (CS16) and algorithmically in Calvano et al.",
    chapters: ["ch5-meta-learning-marl", "ch9-cooperative-game"],
    core: true,
  },
  {
    id: "cascade-failure",
    label: "Multi-Agent Cascade Failure",
    category: "concept",
    description: "Individual agent vulnerability compounding through multi-agent interaction into exponential failure propagation. CS4 (resource exhaustion loop) and CS11 (mass defamation broadcast).",
    chapters: ["ch5-meta-learning-marl"],
    core: false,
  },
  {
    id: "authority-non-stationarity",
    label: "Socially Constructed Authority",
    category: "concept",
    description: "Authority as a non-stationary, conversationally constructed variable rather than fixed state. Agents treat confident communication as authority signal — a specific instance of MARL non-stationarity.",
    chapters: ["ch5-meta-learning-marl", "ch8-llm-steering"],
    core: false,
  },
  {
    id: "adversarial-robustness",
    label: "Adversarial Robustness (Agents)",
    category: "concept",
    description: "An agent's ability to maintain policy integrity under adversarial input: prompt injection, social engineering, identity spoofing. Framed as a Stackelberg game between attacker and agent.",
    chapters: ["ch8-llm-steering", "ch9-cooperative-game"],
    core: false,
  },
  {
    id: "cross-agent-meta-learning",
    label: "Cross-Agent Knowledge Transfer",
    category: "concept",
    description: "One agent's learned policy influencing another agent's behaviour. Doug teaching Mira (CS9) maps to outer-loop meta-learning in Kim et al.'s framework.",
    chapters: ["ch5-meta-learning-marl", "ch6-meta-mapg-theorem"],
    core: true,
  },
  {
    id: "reward-misspecification",
    label: "Reward Misspecification",
    category: "concept",
    description: "When the reward function R(s,a) fails to capture implicit constraints. CS1: agent destroyed mail server (achieved objective but violated proportionality). Central problem in LLM alignment.",
    chapters: ["ch8-llm-steering", "ch9-cooperative-game"],
    core: false,
  },
];

// ============================================================
// THEOREMS
// ============================================================
const theorems: GraphNode[] = [
  {
    id: "pg-theorem",
    label: "Policy Gradient Theorem",
    category: "theorem",
    description: "∇J(θ) ∝ Σ_s μ_π(s) Σ_a q_π(s,a) ∇_θ π(a|s,θ). Gradient doesn't require differentiating state distribution.",
    formalStatement: "\\nabla_{\\bm{\\theta}} J(\\bm{\\theta}) \\propto \\sum_{s \\in \\mathcal{S}} \\mu_\\pi(s) \\sum_{a \\in \\mathcal{A}} q_\\pi(s,a) \\nabla_{\\bm{\\theta}} \\pi(a|s, \\bm{\\theta})",
    chapters: ["ch3-policy-methods", "ch4-pg-theorem-proof"],
    core: true,
  },
  {
    id: "meta-mapg-theorem",
    label: "Meta-Multi-Agent PG Theorem",
    category: "theorem",
    description: "Generalisation of the PG theorem to multi-agent meta-learning. Accounts for each agent's policy update affecting all others.",
    chapters: ["ch6-meta-mapg-theorem"],
    core: true,
  },
  {
    id: "bellman-optimality",
    label: "Bellman Optimality Equations",
    category: "theorem",
    description: "v*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γ v*(s')]. Characterises optimal value functions.",
    chapters: ["ch3-policy-methods"],
    core: false,
  },
  {
    id: "baseline-unbiasedness",
    label: "Baseline Unbiasedness",
    category: "theorem",
    description: "Subtracting any state-dependent baseline b(s) from the return does not bias the policy gradient estimate.",
    formalStatement: "\\mathbb{E}_{a \\sim \\pi}[b(s) \\nabla_{\\bm{\\theta}} \\ln \\pi(a|s, \\bm{\\theta})] = 0",
    chapters: ["ch3-policy-methods", "ch4-pg-theorem-proof"],
    core: false,
  },
  {
    id: "det-optimal-exists",
    label: "Deterministic Optimal Policy Existence",
    category: "theorem",
    description: "For finite MDPs, at least one deterministic optimal policy always exists.",
    chapters: ["ch3-policy-methods"],
    core: false,
  },
];

// ============================================================
// ALGORITHMS
// ============================================================
const algorithms: GraphNode[] = [
  {
    id: "reinforce",
    label: "REINFORCE",
    category: "algorithm",
    description: "Monte Carlo policy gradient. Update: θ ← θ + α G_t ∇ln π(a_t|s_t,θ). Unbiased but high variance.",
    chapters: ["ch3-policy-methods", "ch10-simulations"],
    core: true,
  },
  {
    id: "actor-critic",
    label: "Actor-Critic",
    category: "algorithm",
    description: "Policy (actor) + value function (critic) updated simultaneously. Uses TD error as advantage estimate.",
    chapters: ["ch3-policy-methods", "ch10-simulations"],
    core: true,
  },
  {
    id: "ppo",
    label: "PPO",
    category: "algorithm",
    description: "Proximal Policy Optimization. Clipped surrogate objective for stable policy updates.",
    chapters: ["ch8-llm-steering"],
    core: false,
  },
  {
    id: "trpo",
    label: "TRPO",
    category: "algorithm",
    description: "Trust Region Policy Optimization. Monotonic improvement via KL-divergence constraint.",
    chapters: ["ch3-policy-methods"],
    core: false,
  },
  {
    id: "lola-algorithm",
    label: "LOLA",
    category: "algorithm",
    description: "Learning with Opponent-Learning Awareness. Agent differentiates through opponent's learning step.",
    chapters: ["ch5-meta-learning-marl"],
    core: true,
  },
  {
    id: "maml-algorithm",
    label: "MAML",
    category: "algorithm",
    description: "Model-Agnostic Meta-Learning. Learn initial parameters that adapt quickly via few gradient steps.",
    chapters: ["ch5-meta-learning-marl"],
    core: true,
  },
  {
    id: "meta-mapg-algorithm",
    label: "Meta-MAPG",
    category: "algorithm",
    description: "The algorithm from Kim et al. Meta-learning policy gradient for multi-agent settings.",
    chapters: ["ch6-meta-mapg-theorem", "ch10-simulations"],
    core: true,
  },
  {
    id: "maddpg-algorithm",
    label: "MADDPG",
    category: "algorithm",
    description: "Multi-Agent DDPG. Centralised training, decentralised execution.",
    chapters: ["ch2-lit-review", "ch5-meta-learning-marl"],
    core: false,
  },
  {
    id: "soft-q",
    label: "Soft Q-Learning",
    category: "algorithm",
    description: "Entropy-regularised Q-learning. Used in Wei et al. and referenced in Kim et al.'s proof.",
    chapters: ["ch6-meta-mapg-theorem"],
    core: false,
  },
];

// ============================================================
// APPLICATIONS
// ============================================================
const applications: GraphNode[] = [
  {
    id: "llm-steering",
    label: "LLM Steering",
    category: "application",
    description: "Using RL to steer a frozen LLM via a hypernetwork. The dissertation's main application.",
    chapters: ["ch8-llm-steering", "ch9-cooperative-game"],
    core: true,
  },
  {
    id: "cooperative-llm-game",
    label: "Cooperative LLM Steering Game",
    category: "application",
    description: "Frozen LLM + hypernetwork modelled as two cooperative agents.",
    chapters: ["ch9-cooperative-game"],
    core: true,
  },
  {
    id: "rlhf",
    label: "RLHF",
    category: "application",
    description: "Reinforcement Learning from Human Feedback. Standard paradigm for aligning LLMs.",
    chapters: ["ch8-llm-steering"],
    core: false,
  },
];

// ============================================================
// OPEN PROBLEMS
// ============================================================
const openProblems: GraphNode[] = [
  {
    id: "convergence-meta-mapg",
    label: "Convergence of Meta-MAPG",
    category: "open_problem",
    description: "Does the meta-MAPG algorithm converge? Under what conditions? This is the ambitious goal of Ch.7.",
    chapters: ["ch7-convergence"],
    core: true,
  },
  {
    id: "continuous-extension",
    label: "Continuous State-Action Extension",
    category: "open_problem",
    description: "Extend the discrete Meta-MAPG theorem to continuous state/action spaces.",
    chapters: ["ch7-convergence"],
    core: false,
  },
  {
    id: "optimal-k-h",
    label: "Optimal K and H Selection",
    category: "open_problem",
    description: "How to choose episode horizon H and number of trajectories K to control non-stationarity?",
    chapters: ["ch5-meta-learning-marl"],
    core: false,
  },
  {
    id: "formal-cascade-analysis",
    label: "Formal Cascade Failure Analysis",
    category: "open_problem",
    description: "Can Meta-MAPG (Term 3: peer learning anticipation) dampen multi-agent cascade failures? Theoretical prediction: meta-learning agents should prevent exponential failure propagation.",
    chapters: ["ch5-meta-learning-marl", "ch9-cooperative-game"],
    core: true,
  },
  {
    id: "security-game-formulation",
    label: "Agent Security as Stackelberg Game",
    category: "open_problem",
    description: "Formalise red-teaming attacks as Stackelberg game: attacker chooses strategy, agent responds with policy π(a|s,θ). Meta-MAPG agent anticipates attacker adaptation via Term 3.",
    chapters: ["ch2-lit-review", "ch9-cooperative-game"],
    core: false,
  },
];

// ============================================================
// KEY AUTHORS
// ============================================================
const authors: GraphNode[] = [
  {
    id: "author-foerster",
    label: "Jakob Foerster",
    category: "author",
    description: "Pioneer of opponent-learning awareness (LOLA, DiCE). Key figure in meta-learning MARL.",
    chapters: ["ch5-meta-learning-marl"],
    core: true,
  },
  {
    id: "author-sutton",
    label: "Richard Sutton",
    category: "author",
    description: "Co-author of the RL textbook. Temporal difference learning, policy gradient foundations.",
    chapters: ["ch3-policy-methods", "ch4-pg-theorem-proof"],
    core: true,
  },
  {
    id: "author-kim",
    label: "Dong-Ki Kim",
    category: "author",
    description: "Lead author of the Meta-MAPG paper. MIT LIDS.",
    chapters: ["ch6-meta-mapg-theorem"],
    core: true,
  },
];

// ============================================================
// ALL NODES
// ============================================================
export const nodes: GraphNode[] = [
  ...chapters,
  ...papers,
  ...concepts,
  ...theorems,
  ...algorithms,
  ...applications,
  ...openProblems,
  ...authors,
];

// ============================================================
// EDGES
// ============================================================
export const edges: GraphEdge[] = [
  // ---- Chapter dependencies (reading order / prerequisites) ----
  { source: "ch3-policy-methods", target: "ch4-pg-theorem-proof", type: "requires", strength: 0.9 },
  { source: "ch4-pg-theorem-proof", target: "ch5-meta-learning-marl", type: "requires", strength: 0.9 },
  { source: "ch5-meta-learning-marl", target: "ch6-meta-mapg-theorem", type: "requires", strength: 0.9 },
  { source: "ch6-meta-mapg-theorem", target: "ch7-convergence", type: "requires", strength: 0.7 },
  { source: "ch3-policy-methods", target: "ch8-llm-steering", type: "requires", strength: 0.8 },
  { source: "ch6-meta-mapg-theorem", target: "ch9-cooperative-game", type: "requires", strength: 0.9 },
  { source: "ch8-llm-steering", target: "ch9-cooperative-game", type: "requires", strength: 0.8 },

  // ---- Paper → Theorem (proves / introduces) ----
  { source: "sutton2018", target: "pg-theorem", type: "proves", strength: 0.9 },
  { source: "kim2021", target: "meta-mapg-theorem", type: "proves", strength: 1.0 },
  { source: "williams1992", target: "reinforce", type: "defines", strength: 1.0 },

  // ---- Paper citations ----
  { source: "kim2021", target: "sutton2018", type: "cites", strength: 0.8 },
  { source: "kim2021", target: "foerster2018lola", type: "cites", strength: 0.8 },
  { source: "kim2021", target: "wen2019", type: "cites", strength: 0.9 },
  { source: "kim2021", target: "wei2018", type: "cites", strength: 0.7 },
  { source: "kim2021", target: "finn2017maml", type: "cites", strength: 0.8 },
  { source: "foerster2018lola", target: "sutton2018", type: "cites", strength: 0.6 },
  { source: "lowe2017maddpg", target: "littman1994", type: "cites", strength: 0.6 },

  // ---- Extends / generalises ----
  { source: "meta-mapg-theorem", target: "pg-theorem", type: "extends", strength: 1.0 },
  { source: "n-agent-game", target: "mdp", type: "extends", strength: 1.0 },
  { source: "meta-value-fn", target: "state-value-fn", type: "extends", strength: 0.9 },
  { source: "lola-algorithm", target: "reinforce", type: "extends", strength: 0.7 },
  { source: "maml-algorithm", target: "reinforce", type: "extends", strength: 0.6 },
  { source: "meta-mapg-algorithm", target: "lola-algorithm", type: "extends", strength: 0.8 },
  { source: "meta-mapg-algorithm", target: "maml-algorithm", type: "extends", strength: 0.8 },
  { source: "actor-critic", target: "reinforce", type: "extends", strength: 0.9 },
  { source: "ppo", target: "trpo", type: "extends", strength: 0.8 },
  { source: "trpo", target: "reinforce", type: "extends", strength: 0.7 },

  // ---- Concept requires / builds on ----
  { source: "policy", target: "mdp", type: "requires", strength: 0.9 },
  { source: "parametrised-policy", target: "policy", type: "extends", strength: 0.9 },
  { source: "state-value-fn", target: "policy", type: "requires", strength: 0.8 },
  { source: "state-value-fn", target: "discounted-return", type: "requires", strength: 0.9 },
  { source: "action-value-fn", target: "state-value-fn", type: "requires", strength: 0.8 },
  { source: "advantage-fn", target: "action-value-fn", type: "requires", strength: 0.9 },
  { source: "advantage-fn", target: "state-value-fn", type: "requires", strength: 0.9 },
  { source: "bellman-equations", target: "state-value-fn", type: "requires", strength: 0.9 },
  { source: "bellman-equations", target: "action-value-fn", type: "requires", strength: 0.9 },
  { source: "on-policy-distribution", target: "policy", type: "requires", strength: 0.8 },
  { source: "performance-measure", target: "state-value-fn", type: "requires", strength: 0.8 },
  { source: "td-error", target: "advantage-fn", type: "requires", strength: 0.7, label: "estimates" },
  { source: "baseline", target: "state-value-fn", type: "requires", strength: 0.7 },
  { source: "trajectory", target: "mdp", type: "requires", strength: 0.7 },

  // ---- Theorem requires concepts ----
  { source: "pg-theorem", target: "on-policy-distribution", type: "requires", strength: 1.0 },
  { source: "pg-theorem", target: "action-value-fn", type: "requires", strength: 1.0 },
  { source: "pg-theorem", target: "parametrised-policy", type: "requires", strength: 1.0 },
  { source: "pg-theorem", target: "log-derivative-trick", type: "requires", strength: 0.8 },
  { source: "meta-mapg-theorem", target: "pg-theorem", type: "requires", strength: 1.0 },
  { source: "meta-mapg-theorem", target: "conditional-independence", type: "requires", strength: 0.9 },
  { source: "meta-mapg-theorem", target: "markovian-update", type: "requires", strength: 0.9 },
  { source: "meta-mapg-theorem", target: "meta-value-fn", type: "requires", strength: 0.9 },
  { source: "meta-mapg-theorem", target: "n-agent-game", type: "requires", strength: 0.9 },

  // ---- Algorithm implements concept ----
  { source: "reinforce", target: "pg-theorem", type: "implements", strength: 1.0 },
  { source: "actor-critic", target: "pg-theorem", type: "implements", strength: 0.9 },
  { source: "actor-critic", target: "td-error", type: "implements", strength: 0.9 },
  { source: "actor-critic", target: "baseline", type: "implements", strength: 0.8 },
  { source: "meta-mapg-algorithm", target: "meta-mapg-theorem", type: "implements", strength: 1.0 },

  // ---- Application applies concept ----
  { source: "llm-steering", target: "parametrised-policy", type: "applies", strength: 0.8 },
  { source: "llm-steering", target: "hypernetwork", type: "applies", strength: 0.9 },
  { source: "cooperative-llm-game", target: "n-agent-game", type: "applies", strength: 0.9 },
  { source: "cooperative-llm-game", target: "meta-mapg-algorithm", type: "applies", strength: 0.9 },
  { source: "cooperative-llm-game", target: "frozen-llm", type: "applies", strength: 1.0 },
  { source: "cooperative-llm-game", target: "hypernetwork", type: "applies", strength: 1.0 },
  { source: "rlhf", target: "ppo", type: "applies", strength: 0.7 },

  // ---- Chapter contains ----
  { source: "ch3-policy-methods", target: "mdp", type: "contains", strength: 0.8 },
  { source: "ch3-policy-methods", target: "pg-theorem", type: "contains", strength: 0.9 },
  { source: "ch3-policy-methods", target: "reinforce", type: "contains", strength: 0.8 },
  { source: "ch3-policy-methods", target: "actor-critic", type: "contains", strength: 0.8 },
  { source: "ch4-pg-theorem-proof", target: "pg-theorem", type: "contains", strength: 1.0 },
  { source: "ch6-meta-mapg-theorem", target: "meta-mapg-theorem", type: "contains", strength: 1.0 },
  { source: "ch6-meta-mapg-theorem", target: "meta-mapg-algorithm", type: "contains", strength: 1.0 },
  { source: "ch9-cooperative-game", target: "cooperative-llm-game", type: "contains", strength: 1.0 },
  { source: "ch8-llm-steering", target: "llm-steering", type: "contains", strength: 1.0 },

  // ---- Paper → Chapter (primary reference) ----
  { source: "ch3-policy-methods", target: "sutton2018", type: "cites", strength: 0.9 },
  { source: "ch6-meta-mapg-theorem", target: "kim2021", type: "cites", strength: 1.0 },
  { source: "ch5-meta-learning-marl", target: "foerster2018lola", type: "cites", strength: 0.8 },
  { source: "ch5-meta-learning-marl", target: "finn2017maml", type: "cites", strength: 0.7 },
  { source: "ch8-llm-steering", target: "hypersteer", type: "cites", strength: 0.9 },
  { source: "ch8-llm-steering", target: "st310-project", type: "cites", strength: 1.0 },

  // ---- Author → Paper ----
  { source: "author-foerster", target: "foerster2018lola", type: "defines", strength: 1.0 },
  { source: "author-foerster", target: "foerster2018dice", type: "defines", strength: 1.0 },
  { source: "author-sutton", target: "sutton2018", type: "defines", strength: 1.0 },
  { source: "author-kim", target: "kim2021", type: "defines", strength: 1.0 },

  // ---- Prelim presentation / collusion ----
  { source: "calvano2020", target: "bertrand-competition", type: "applies", strength: 0.9 },
  { source: "calvano2020", target: "algorithmic-collusion", type: "defines", strength: 0.9 },
  { source: "meta-mapg-algorithm", target: "algorithmic-collusion", type: "applies", strength: 0.8 },
  { source: "meta-mapg-algorithm", target: "bertrand-competition", type: "applies", strength: 0.8 },
  { source: "three-term-gradient", target: "meta-mapg-theorem", type: "requires", strength: 1.0 },
  { source: "higher-order-gradients", target: "meta-mapg-algorithm", type: "requires", strength: 0.9 },
  { source: "higher-order-gradients", target: "lola-algorithm", type: "requires", strength: 0.8 },
  { source: "ch10-simulations", target: "calvano2020", type: "cites", strength: 0.8 },
  { source: "ch10-simulations", target: "bertrand-competition", type: "contains", strength: 0.9 },
  { source: "ch6-meta-mapg-theorem", target: "three-term-gradient", type: "contains", strength: 1.0 },
  { source: "prelim-presentation", target: "kim2021", type: "cites", strength: 1.0 },
  { source: "prelim-presentation", target: "calvano2020", type: "cites", strength: 0.9 },
  { source: "prelim-presentation", target: "foerster2018lola", type: "cites", strength: 0.7 },

  // ---- Open problems ----
  { source: "convergence-meta-mapg", target: "meta-mapg-theorem", type: "extends", strength: 0.9 },
  { source: "convergence-meta-mapg", target: "konda2000ac", type: "cites", strength: 0.6 },
  { source: "continuous-extension", target: "meta-mapg-theorem", type: "extends", strength: 0.7 },

  // ---- Soft Q-learning connections (from reading notes) ----
  { source: "soft-q", target: "action-value-fn", type: "extends", strength: 0.8, label: "entropy-regularised" },
  { source: "wei2018", target: "soft-q", type: "defines", strength: 1.0 },
  { source: "meta-mapg-theorem", target: "soft-q", type: "requires", strength: 0.7, label: "used in proof" },
  { source: "ch6-meta-mapg-theorem", target: "wei2018", type: "cites", strength: 0.7 },

  // ---- LOLA as special case of Meta-MAPG ----
  { source: "foerster2018lola", target: "foerster2018dice", type: "cites", strength: 0.7 },
  { source: "lola-algorithm", target: "non-stationarity", type: "applies", strength: 0.8 },
  { source: "meta-mapg-algorithm", target: "three-term-gradient", type: "implements", strength: 1.0 },

  // ---- Calvano as baseline for simulations ----
  { source: "algorithmic-collusion", target: "nash-equilibrium", type: "requires", strength: 0.7, label: "supra-competitive vs Nash" },
  { source: "ch10-simulations", target: "kim2021", type: "cites", strength: 1.0 },
  { source: "ch10-simulations", target: "st310-project", type: "cites", strength: 0.9 },

  // ---- Agents of Chaos connections ----
  { source: "agents-of-chaos-2026", target: "non-stationarity", type: "applies", strength: 0.8, label: "authority as non-stationary" },
  { source: "agents-of-chaos-2026", target: "emergent-coordination", type: "defines", strength: 0.9 },
  { source: "agents-of-chaos-2026", target: "cascade-failure", type: "defines", strength: 0.9 },
  { source: "agents-of-chaos-2026", target: "cross-agent-meta-learning", type: "defines", strength: 0.8, label: "CS9: Doug→Mira" },
  { source: "agents-of-chaos-2026", target: "reward-misspecification", type: "applies", strength: 0.7, label: "CS1: disproportionate response" },

  // ---- Emergent coordination parallels ----
  { source: "emergent-coordination", target: "algorithmic-collusion", type: "equivalent_to", strength: 0.8, label: "same mechanism, different domain", bidirectional: true },
  { source: "emergent-coordination", target: "cooperative-game", type: "requires", strength: 0.7 },
  { source: "emergent-coordination", target: "meta-mapg-theorem", type: "requires", strength: 0.8, label: "Term 3 enables" },

  // ---- Cross-agent meta-learning ----
  { source: "cross-agent-meta-learning", target: "meta-value-fn", type: "requires", strength: 0.8, label: "outer loop" },
  { source: "cross-agent-meta-learning", target: "markovian-update", type: "requires", strength: 0.7 },

  // ---- Cascade failure analysis ----
  { source: "cascade-failure", target: "n-agent-game", type: "requires", strength: 0.8 },
  { source: "cascade-failure", target: "non-stationarity", type: "requires", strength: 0.7 },
  { source: "formal-cascade-analysis", target: "cascade-failure", type: "extends", strength: 0.9 },
  { source: "formal-cascade-analysis", target: "meta-mapg-theorem", type: "requires", strength: 0.8, label: "Term 3 dampens cascades" },

  // ---- Security game ----
  { source: "adversarial-robustness", target: "stackelberg-equilibrium", type: "requires", strength: 0.7 },
  { source: "security-game-formulation", target: "adversarial-robustness", type: "extends", strength: 0.8 },
  { source: "security-game-formulation", target: "meta-mapg-algorithm", type: "applies", strength: 0.7 },
  { source: "authority-non-stationarity", target: "non-stationarity", type: "extends", strength: 0.9, label: "specific instance" },

  // ---- Chapter citations ----
  { source: "ch5-meta-learning-marl", target: "agents-of-chaos-2026", type: "cites", strength: 0.8 },
  { source: "ch9-cooperative-game", target: "agents-of-chaos-2026", type: "cites", strength: 0.8 },
  { source: "ch9-cooperative-game", target: "emergent-coordination", type: "contains", strength: 0.9 },

  // ---- Reward misspecification ----
  { source: "reward-misspecification", target: "performance-measure", type: "critiques", strength: 0.7 },
  { source: "cooperative-llm-game", target: "reward-misspecification", type: "requires", strength: 0.6, label: "must avoid" },
];

// ============================================================
// HELPERS
// ============================================================
export function getCoreNodes(): GraphNode[] {
  return nodes.filter((n) => n.core);
}

export function getCoreEdges(): GraphEdge[] {
  const coreIds = new Set(getCoreNodes().map((n) => n.id));
  return edges.filter((e) => coreIds.has(e.source) && coreIds.has(e.target));
}

export function getNodesForChapter(chapterSlug: string): GraphNode[] {
  return nodes.filter(
    (n) => n.chapters?.includes(chapterSlug) || n.id === chapterSlug
  );
}

export function getEdgesForChapter(chapterSlug: string): GraphEdge[] {
  const chapterNodeIds = new Set(getNodesForChapter(chapterSlug).map((n) => n.id));
  return edges.filter(
    (e) => chapterNodeIds.has(e.source) && chapterNodeIds.has(e.target)
  );
}

export function getChapterNodes(): GraphNode[] {
  return nodes.filter((n) => n.category === "chapter");
}

export function getPaperNodes(): GraphNode[] {
  return nodes.filter((n) => n.category === "paper");
}
