"""
Pearl's do-calculus — causal effect identification from graphical models.

Implements the core identification algorithms from Judea Pearl's causal
inference framework:

- Backdoor criterion: finds valid adjustment sets to block confounding paths
- Frontdoor criterion: identifies mediator-based estimation strategies
- Effect identifiability: checks whether a causal effect can be estimated
  from observational data given the DAG structure
- Instrumental variables: finds variables satisfying the IV conditions

These algorithms operate on the structure of the DAG alone — they determine
*whether* and *how* a causal effect can be identified, not its magnitude.
Magnitude estimation is delegated to statistical methods (DoWhy, EconML).

References:
    Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
    Pearl, J. (2012). The do-calculus revisited. UAI 2012.

Usage:
    from causal.dag import CausalDAGEngine
    from causal.pearl import backdoor_criterion, identify_effect

    engine = CausalDAGEngine.from_schema(dag)
    adjustment_sets = backdoor_criterion(engine, treatment="X", outcome="Y")
    is_identifiable = identify_effect(engine, treatment="X", outcome="Y")
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx

from .dag import CausalDAGEngine


def backdoor_criterion(
    engine: CausalDAGEngine,
    treatment: str,
    outcome: str,
) -> list[set[str]]:
    """Find all valid backdoor adjustment sets for the causal effect of
    treatment on outcome.

    A set Z satisfies the backdoor criterion relative to (X, Y) if:
    1. No node in Z is a descendant of X.
    2. Z blocks every path between X and Y that contains an arrow into X
       (i.e., all backdoor paths from X to Y).

    Condition (2) is equivalent to: X and Y are d-separated given Z in
    the modified graph where all edges out of X have been removed.

    This function enumerates all minimal valid adjustment sets by testing
    subsets of non-descendant nodes.

    Args:
        engine: A CausalDAGEngine containing the causal DAG.
        treatment: The node ID of the treatment variable.
        outcome: The node ID of the outcome variable.

    Returns:
        A list of sets, where each set contains variable IDs that form a
        valid backdoor adjustment set. Returns an empty list if no valid
        adjustment set exists.

    Raises:
        KeyError: If treatment or outcome node does not exist.
    """
    engine._check_node(treatment)
    engine._check_node(outcome)

    # Build the manipulated graph: remove all edges OUT of treatment
    manipulated = engine.graph.copy()
    out_edges = list(manipulated.out_edges(treatment))
    manipulated.remove_edges_from(out_edges)

    # Candidate adjustment variables: all nodes except treatment, outcome,
    # and descendants of treatment in the ORIGINAL graph
    descendants_of_treatment = nx.descendants(engine.graph, treatment)
    all_nodes = set(engine.graph.nodes())
    candidates = all_nodes - {treatment, outcome} - descendants_of_treatment

    valid_sets: list[set[str]] = []

    # Test subsets of candidates, starting from smallest (empty set)
    # For efficiency, we limit search to subsets up to a reasonable size
    max_subset_size = min(len(candidates), 6)  # Practical limit for enumeration

    for size in range(0, max_subset_size + 1):
        for subset in combinations(sorted(candidates), size):
            z_set = set(subset)
            # Check d-separation in the manipulated graph
            if nx.d_separated(manipulated, {treatment}, {outcome}, z_set):
                # Check minimality: only keep if no proper subset already works
                is_minimal = True
                for existing in valid_sets:
                    if existing.issubset(z_set) and existing != z_set:
                        is_minimal = False
                        break
                if is_minimal:
                    valid_sets.append(z_set)

    return valid_sets


def frontdoor_criterion(
    engine: CausalDAGEngine,
    treatment: str,
    outcome: str,
) -> list[set[str]]:
    """Find sets of variables satisfying the frontdoor criterion.

    A set M satisfies the frontdoor criterion relative to (X, Y) if:
    1. M intercepts all directed paths from X to Y.
    2. There is no unblocked backdoor path from X to M.
    3. All backdoor paths from M to Y are blocked by X.

    The frontdoor criterion enables causal effect identification even when
    there are unmeasured confounders between X and Y, provided a suitable
    mediator set M exists.

    Args:
        engine: A CausalDAGEngine containing the causal DAG.
        treatment: The node ID of the treatment variable.
        outcome: The node ID of the outcome variable.

    Returns:
        A list of sets, where each set contains variable IDs satisfying
        the frontdoor criterion. Returns an empty list if no valid set exists.

    Raises:
        KeyError: If treatment or outcome node does not exist.
    """
    engine._check_node(treatment)
    engine._check_node(outcome)

    # Find all directed paths from treatment to outcome
    directed_paths = list(nx.all_simple_paths(engine.graph, treatment, outcome))

    if not directed_paths:
        return []

    # Candidate mediators: nodes on directed paths from treatment to outcome,
    # excluding treatment and outcome themselves
    mediator_candidates: set[str] = set()
    for path in directed_paths:
        mediator_candidates.update(path[1:-1])  # Exclude start and end

    if not mediator_candidates:
        return []

    valid_sets: list[set[str]] = []
    max_subset_size = min(len(mediator_candidates), 5)

    for size in range(1, max_subset_size + 1):
        for subset in combinations(sorted(mediator_candidates), size):
            m_set = set(subset)

            # Condition 1: M intercepts all directed paths from X to Y
            intercepts_all = True
            for path in directed_paths:
                intermediaries = set(path[1:-1])
                if not intermediaries.intersection(m_set):
                    intercepts_all = False
                    break

            if not intercepts_all:
                continue

            # Condition 2: No unblocked backdoor path from X to any M_i
            # (X and M_i are d-separated in the graph with edges out of X removed,
            #  conditioning on the empty set — but we check via d-separation
            #  in the original graph with M_i's parents as potential confounders)
            # Simplified check: no backdoor path from X to M not blocked
            condition_2 = True
            graph_no_x_out = engine.graph.copy()
            x_out_edges = list(graph_no_x_out.out_edges(treatment))
            graph_no_x_out.remove_edges_from(x_out_edges)

            for m_node in m_set:
                # Check if there's an unblocked path from treatment to m_node
                # in the graph without treatment's outgoing edges
                if nx.has_path(graph_no_x_out, treatment, m_node):
                    condition_2 = False
                    break
                if nx.has_path(graph_no_x_out, m_node, treatment):
                    # Check d-separation with empty conditioning
                    if not nx.d_separated(graph_no_x_out, {treatment}, {m_node}, set()):
                        condition_2 = False
                        break

            if not condition_2:
                continue

            # Condition 3: All backdoor paths from M to Y blocked by X
            condition_3 = True
            for m_node in m_set:
                graph_no_m_out = engine.graph.copy()
                m_out_edges = list(graph_no_m_out.out_edges(m_node))
                graph_no_m_out.remove_edges_from(m_out_edges)

                if not nx.d_separated(
                    graph_no_m_out, {m_node}, {outcome}, {treatment}
                ):
                    condition_3 = False
                    break

            if condition_3:
                valid_sets.append(m_set)

    return valid_sets


def identify_effect(
    engine: CausalDAGEngine,
    treatment: str,
    outcome: str,
) -> dict[str, object]:
    """Check whether the causal effect of treatment on outcome is identifiable.

    Attempts identification via the backdoor criterion first, then the
    frontdoor criterion. Returns a summary indicating whether the effect
    is identifiable and by which method.

    Args:
        engine: A CausalDAGEngine containing the causal DAG.
        treatment: The node ID of the treatment variable.
        outcome: The node ID of the outcome variable.

    Returns:
        Dictionary with keys:
        - "identifiable" (bool): whether the causal effect can be estimated
        - "method" (str | None): "backdoor", "frontdoor", or None
        - "adjustment_sets" (list[set[str]]): valid adjustment / mediator sets

    Raises:
        KeyError: If treatment or outcome node does not exist.
    """
    engine._check_node(treatment)
    engine._check_node(outcome)

    # Try backdoor criterion
    bd_sets = backdoor_criterion(engine, treatment, outcome)
    if bd_sets:
        return {
            "identifiable": True,
            "method": "backdoor",
            "adjustment_sets": bd_sets,
        }

    # Try frontdoor criterion
    fd_sets = frontdoor_criterion(engine, treatment, outcome)
    if fd_sets:
        return {
            "identifiable": True,
            "method": "frontdoor",
            "adjustment_sets": fd_sets,
        }

    return {
        "identifiable": False,
        "method": None,
        "adjustment_sets": [],
    }


def instrumental_variables(
    engine: CausalDAGEngine,
    treatment: str,
    outcome: str,
) -> list[str]:
    """Find valid instrumental variables for the effect of treatment on outcome.

    A variable Z is a valid instrument for the effect of X on Y if:
    1. Z is associated with X (there is a path from Z to X, or Z causes X).
    2. Z affects Y only through X (no direct path from Z to Y that bypasses X).
    3. Z and Y share no common causes (no confounding between Z and Y,
       conditional on possible observed covariates).

    For simplicity, this implementation checks:
    - Z has a directed path to treatment (relevance)
    - All directed paths from Z to outcome go through treatment (exclusion)
    - Z is not a descendant of treatment (exogeneity)

    Args:
        engine: A CausalDAGEngine containing the causal DAG.
        treatment: The node ID of the treatment variable.
        outcome: The node ID of the outcome variable.

    Returns:
        List of node IDs that qualify as instrumental variables.

    Raises:
        KeyError: If treatment or outcome node does not exist.
    """
    engine._check_node(treatment)
    engine._check_node(outcome)

    descendants_of_treatment = nx.descendants(engine.graph, treatment)
    all_nodes = set(engine.graph.nodes())
    candidates = all_nodes - {treatment, outcome} - descendants_of_treatment

    instruments: list[str] = []

    for z in sorted(candidates):
        # Relevance: Z must have a directed path to treatment
        if not nx.has_path(engine.graph, z, treatment):
            continue

        # Exclusion: all directed paths from Z to Y must pass through X
        z_to_y_paths = list(nx.all_simple_paths(engine.graph, z, outcome))
        if not z_to_y_paths:
            # Z has no path to Y at all — could still be valid if Z->X->Y exists
            # but we need the Z->X association to transmit to Y
            if nx.has_path(engine.graph, treatment, outcome):
                # Z -> X -> Y exists, and Z has no other path to Y: valid
                pass
            else:
                continue
        else:
            exclusion_holds = True
            for path in z_to_y_paths:
                if treatment not in path:
                    exclusion_holds = False
                    break
            if not exclusion_holds:
                continue

        # Exogeneity check: in the graph with edges out of treatment removed,
        # Z and Y should be d-separated (no confounding path)
        graph_no_x = engine.graph.copy()
        x_out_edges = list(graph_no_x.out_edges(treatment))
        graph_no_x.remove_edges_from(x_out_edges)

        if nx.d_separated(graph_no_x, {z}, {outcome}, set()):
            instruments.append(z)

    return instruments
