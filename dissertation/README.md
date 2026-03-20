# LSE Mathematics Dissertation

**Title:** Meta-Learning Multi-Agent Policy Gradients with Applications to LLM Steering

**Candidate:** 007

**Supervisor:** TBD

## Structure

```
dissertation/
├── latex/                    # LaTeX source (syncs with Overleaf)
│   ├── main.tex              # Master document
│   ├── preamble/             # Style files, macros
│   ├── chapters/             # One .tex file per chapter
│   └── figures/              # All figures
├── papers/                   # PDF collection of references
├── notes/                    # Reading notes, scratchpad
├── knowledge-graph/          # Interactive knowledge graph + dissertation dashboard
│   ├── data/                 # JSON: papers, concepts, theorems, chapters
│   ├── lib/                  # Utilities
│   ├── types/                # TypeScript types
│   ├── app/                  # Next.js pages
│   └── components/           # React components
└── simulations/              # Code for Ch.10 experiments
```

## Knowledge Graph

The knowledge graph tracks papers, concepts, theorems, algorithms, and their connections.
Run with:

```bash
cd knowledge-graph
npm install
npm run dev
```

## Chapters

| Ch | Title | Status |
|----|-------|--------|
| 1  | Introduction | TODO |
| 2  | Literature Review | TODO |
| 3  | Policy Methods in RL | DRAFT |
| 4  | Policy Gradient Theorem (Proof) | TODO |
| 5  | Meta-learning in Multiagent Setups | TODO |
| 6  | Meta-learning Multiagent PG Theorem | TODO |
| 7  | Convergence Guarantees (AMBITIOUS) | TODO |
| 8  | Large Language Models Steering | TODO |
| 9  | Cooperative Steering Game | TODO |
| 10 | Simulations | TODO |
| 11 | Conclusion | TODO |
