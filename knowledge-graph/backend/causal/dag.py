"""
Causal DAG Engine — structural operations on causal directed acyclic graphs.

Implements graph-theoretic operations required for Pearl's causal inference
framework: d-separation, Markov blankets, topological ordering, and path
enumeration. Built on NetworkX for algorithmic correctness.

This engine operates on the structural level — it answers questions about
the *shape* of causal relationships, not their *strength*. Strength estimation
is handled by the estimation module.

Usage:
    from graph.entities import CausalDAG
    engine = CausalDAGEngine.from_schema(dag_schema)
    is_sep = engine.is_d_separated("X", "Y", {"Z"})
    blanket = engine.markov_blanket("X")
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

from graph.entities import CausalDAG


class CausalDAGEngine:
    """Engine for structural operations on causal directed acyclic graphs.

    Wraps a NetworkX DiGraph and provides methods for d-separation testing,
    path enumeration, Markov blanket computation, and other operations
    central to Pearl's causal inference framework.

    Attributes:
        graph: The underlying NetworkX directed graph.
        node_names: Mapping from node ID to human-readable name.
    """

    def __init__(self, graph: Optional[nx.DiGraph] = None) -> None:
        """Initialize the engine with an optional pre-built graph.

        Args:
            graph: A NetworkX DiGraph. If None, an empty graph is created.
        """
        self.graph: nx.DiGraph = graph if graph is not None else nx.DiGraph()
        self.node_names: dict[str, str] = {}

    @classmethod
    def from_schema(cls, dag: CausalDAG) -> "CausalDAGEngine":
        """Construct a CausalDAGEngine from a CausalDAG Pydantic schema.

        Translates the entity-level CausalDAG definition into a NetworkX
        DiGraph suitable for algorithmic analysis.

        Args:
            dag: A CausalDAG schema object containing nodes and edges.

        Returns:
            A configured CausalDAGEngine instance.

        Raises:
            ValueError: If the resulting graph contains cycles.
        """
        G = nx.DiGraph()
        engine = cls(G)

        for node in dag.nodes:
            G.add_node(node.id, name=node.name, variable_type=node.type.value)
            engine.node_names[node.id] = node.name

        for edge in dag.edges:
            G.add_edge(
                edge.source,
                edge.target,
                edge_type=edge.type.value,
                strength=edge.strength,
            )

        if not nx.is_directed_acyclic_graph(G):
            raise ValueError(
                "The provided graph contains cycles and is not a valid DAG."
            )

        return engine

    # ------------------------------------------------------------------
    # Basic structural queries
    # ------------------------------------------------------------------

    def parents(self, node: str) -> set[str]:
        """Return the set of direct parents (predecessors) of a node.

        Args:
            node: The node ID.

        Returns:
            Set of parent node IDs.

        Raises:
            KeyError: If the node does not exist.
        """
        self._check_node(node)
        return set(self.graph.predecessors(node))

    def children(self, node: str) -> set[str]:
        """Return the set of direct children (successors) of a node.

        Args:
            node: The node ID.

        Returns:
            Set of child node IDs.

        Raises:
            KeyError: If the node does not exist.
        """
        self._check_node(node)
        return set(self.graph.successors(node))

    def ancestors(self, node: str) -> set[str]:
        """Return all ancestors of a node (transitive parents).

        Args:
            node: The node ID.

        Returns:
            Set of all ancestor node IDs (not including the node itself).

        Raises:
            KeyError: If the node does not exist.
        """
        self._check_node(node)
        return nx.ancestors(self.graph, node)

    def descendants(self, node: str) -> set[str]:
        """Return all descendants of a node (transitive children).

        Args:
            node: The node ID.

        Returns:
            Set of all descendant node IDs (not including the node itself).

        Raises:
            KeyError: If the node does not exist.
        """
        self._check_node(node)
        return nx.descendants(self.graph, node)

    # ------------------------------------------------------------------
    # D-separation
    # ------------------------------------------------------------------

    def is_d_separated(
        self,
        x: str,
        y: str,
        z_set: Optional[set[str]] = None,
    ) -> bool:
        """Test whether X and Y are d-separated given conditioning set Z.

        Uses the Bayes-Ball algorithm (via NetworkX) to determine whether
        all paths between X and Y are blocked by conditioning on Z.

        In Pearl's framework, d-separation implies conditional independence
        in every distribution compatible with the DAG structure.

        Args:
            x: First variable ID.
            y: Second variable ID.
            z_set: Set of variable IDs to condition on. Empty set if None.

        Returns:
            True if X and Y are d-separated given Z, False otherwise.

        Raises:
            KeyError: If any referenced node does not exist.
        """
        self._check_node(x)
        self._check_node(y)
        z = z_set or set()
        for node in z:
            self._check_node(node)

        return nx.d_separated(self.graph, {x}, {y}, z)

    # ------------------------------------------------------------------
    # Topological ordering
    # ------------------------------------------------------------------

    def topological_sort(self) -> list[str]:
        """Return nodes in topological order.

        In a causal DAG, topological order corresponds to a valid causal
        ordering: causes always appear before their effects.

        Returns:
            List of node IDs in topological order.

        Raises:
            ValueError: If the graph contains cycles.
        """
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Graph contains cycles; topological sort not possible.")
        return list(nx.topological_sort(self.graph))

    # ------------------------------------------------------------------
    # Path enumeration
    # ------------------------------------------------------------------

    def find_all_paths(self, source: str, target: str) -> list[list[str]]:
        """Find all directed paths from source to target.

        Args:
            source: Starting node ID.
            target: Ending node ID.

        Returns:
            List of paths, each a list of node IDs from source to target.

        Raises:
            KeyError: If source or target does not exist.
        """
        self._check_node(source)
        self._check_node(target)
        return list(nx.all_simple_paths(self.graph, source, target))

    # ------------------------------------------------------------------
    # Markov blanket
    # ------------------------------------------------------------------

    def markov_blanket(self, node: str) -> set[str]:
        """Compute the Markov blanket of a node.

        The Markov blanket consists of a node's parents, children, and
        the other parents of its children (co-parents / spouses). A node
        is conditionally independent of all other nodes given its Markov
        blanket.

        Args:
            node: The node ID.

        Returns:
            Set of node IDs forming the Markov blanket.

        Raises:
            KeyError: If the node does not exist.
        """
        self._check_node(node)
        blanket: set[str] = set()

        # Parents
        blanket.update(self.graph.predecessors(node))

        # Children
        children = set(self.graph.successors(node))
        blanket.update(children)

        # Co-parents (other parents of this node's children)
        for child in children:
            blanket.update(self.graph.predecessors(child))

        # Remove the node itself
        blanket.discard(node)
        return blanket

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_dag(self) -> bool:
        """Check whether the graph is a valid directed acyclic graph.

        Returns:
            True if the graph is a DAG, False if it contains cycles.
        """
        return nx.is_directed_acyclic_graph(self.graph)

    # ------------------------------------------------------------------
    # Matrix representation
    # ------------------------------------------------------------------

    def to_adjacency_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Convert the DAG to an adjacency matrix.

        Returns:
            Tuple of (adjacency_matrix, node_order) where adjacency_matrix
            is a numpy array and node_order lists the node IDs corresponding
            to rows/columns.
        """
        nodes = list(self.graph.nodes())
        matrix = nx.to_numpy_array(self.graph, nodelist=nodes)
        return matrix, nodes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_node(self, node: str) -> None:
        """Verify that a node exists in the graph.

        Args:
            node: The node ID to check.

        Raises:
            KeyError: If the node does not exist.
        """
        if node not in self.graph:
            raise KeyError(f"Node '{node}' not found in DAG.")
