// Node categories for an RL/math dissertation knowledge graph
export type NodeCategory =
  | "paper"        // Published papers (Kim et al., Sutton & Barto, etc.)
  | "author"       // Key researchers
  | "concept"      // Mathematical concepts (MDP, Bellman equation, etc.)
  | "theorem"      // Named theorems and results
  | "algorithm"    // Algorithms (REINFORCE, PPO, etc.)
  | "definition"   // Formal definitions
  | "application"  // Applications (LLM steering, cooperative games)
  | "chapter"      // Dissertation chapters
  | "open_problem" // Open questions and ambitious goals
  | "dataset";     // Datasets or environments

export type EdgeType =
  | "cites"          // Paper A cites Paper B
  | "proves"         // Paper proves theorem
  | "extends"        // Generalises or extends
  | "applies"        // Applies concept/theorem to domain
  | "defines"        // Introduces a definition
  | "implements"     // Implements an algorithm
  | "requires"       // Depends on (prerequisite)
  | "critiques"      // Challenges or limits
  | "contains"       // Chapter contains concept
  | "equivalent_to"; // Mathematically equivalent formulations

export interface GraphNode {
  id: string;
  label: string;
  category: NodeCategory;
  description?: string;
  // Paper-specific
  authors?: string[];
  year?: number;
  venue?: string;
  url?: string;
  // Chapter-specific
  chapterNum?: number;
  status?: "done" | "draft" | "outline" | "todo";
  // Theorem-specific
  formalStatement?: string; // LaTeX
  // General
  chapters?: string[];     // Which dissertation chapters reference this
  tags?: string[];
  core?: boolean;          // Core to the dissertation's argument
  // D3 force positions
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: EdgeType;
  strength?: number;  // 0-1, for force layout
  label?: string;
  bidirectional?: boolean;
}

export const CATEGORY_COLORS: Record<NodeCategory, string> = {
  paper:        "#3b82f6", // blue
  author:       "#6366f1", // indigo
  concept:      "#f59e0b", // amber
  theorem:      "#10b981", // emerald
  algorithm:    "#00d4aa", // teal
  definition:   "#8b5cf6", // violet
  application:  "#ec4899", // pink
  chapter:      "#ef4444", // red
  open_problem: "#f97316", // orange
  dataset:      "#64748b", // slate
};

export const CATEGORY_LABELS: Record<NodeCategory, string> = {
  paper:        "Papers",
  author:       "Authors",
  concept:      "Concepts",
  theorem:      "Theorems",
  algorithm:    "Algorithms",
  definition:   "Definitions",
  application:  "Applications",
  chapter:      "Chapters",
  open_problem: "Open Problems",
  dataset:      "Datasets / Envs",
};

export const EDGE_TYPE_LABELS: Record<EdgeType, string> = {
  cites:         "Cites",
  proves:        "Proves",
  extends:       "Extends",
  applies:       "Applies",
  defines:       "Defines",
  implements:    "Implements",
  requires:      "Requires",
  critiques:     "Critiques",
  contains:      "Contains",
  equivalent_to: "Equivalent to",
};

export const EDGE_TYPE_COLORS: Record<EdgeType, string> = {
  cites:         "#94a3b8",
  proves:        "#10b981",
  extends:       "#3b82f6",
  applies:       "#ec4899",
  defines:       "#8b5cf6",
  implements:    "#00d4aa",
  requires:      "#f59e0b",
  critiques:     "#ef4444",
  contains:      "#64748b",
  equivalent_to: "#6366f1",
};

export const STATUS_COLORS: Record<string, string> = {
  done:    "#10b981",
  draft:   "#f59e0b",
  outline: "#3b82f6",
  todo:    "#64748b",
};
