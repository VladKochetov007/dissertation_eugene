"""
Boyd's Destructive Deduction — shattering failing paradigms.

Implements the first half of Boyd's dialectical engine from "Destruction and
Creation" (1976). When a hypothesis or causal model enters Kuhnian crisis
(anomaly accumulation exceeds threshold), destructive deduction takes the
failing model and shatters it into its constituent parts.

The process:
1. Take a failing CausalDAG and its associated hypothesis
2. Evaluate each edge (causal claim) against available evidence
3. Classify edges as SUPPORTED (evidence backs the claim), UNSUPPORTED
   (no evidence for or against), or CONTRADICTED (evidence refutes the claim)
4. Produce a set of "shattered constituents" — individual causal claims
   with their evidence status
5. The supported and unsupported fragments become raw material for Boyd's
   creative induction (creation.py), which recombines them with new data
   to form a new paradigm

Boyd's insight: you MUST shatter existing patterns before creating new ones.
Attempting to patch a failing paradigm produces baroque epicycles (Kuhn's
observation). The only path to genuine novelty is through destruction first.

This maps onto the theological framework: the Fall is necessary before
Redemption. The existing pattern must be broken before a higher-order
synthesis can emerge. Destruction is not failure — it's the prerequisite
for creation.

References:
    Boyd, J. (1976). Destruction and Creation.
    Kuhn, T. (1962). The Structure of Scientific Revolutions.

Usage:
    deductor = DestructiveDeductor(store=store)
    result = deductor.shatter(hypothesis_id="h-001")
    supported = result.supported_constituents
    # Feed supported + new data into CreativeInductor
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from graph.entities import (
    CausalDAG,
    CausalEdge,
    Evidence,
    EvidenceType,
    Hypothesis,
    HypothesisStatus,
    Variable,
)
from graph.store import KnowledgeGraphStore
from causal.dag import CausalDAGEngine


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConstituentStatus(str, Enum):
    """Evidence status of a shattered constituent (individual causal claim).

    - supported: evidence backs this causal relationship
    - unsupported: no evidence for or against (unknown)
    - contradicted: evidence actively refutes this relationship
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ShatteredConstituent(BaseModel):
    """A single causal claim extracted from a shattered DAG.

    When a CausalDAG is destroyed, each edge becomes a constituent: an
    atomic causal claim (X -> Y) with its evidence status. These fragments
    are the raw material for creative induction.

    Attributes:
        id: Unique identifier for this constituent.
        source_variable: The source variable of the causal claim.
        target_variable: The target variable of the causal claim.
        edge: The original CausalEdge from the DAG.
        status: Whether evidence supports, contradicts, or is absent for this claim.
        supporting_evidence: IDs of evidence items supporting this claim.
        contradicting_evidence: IDs of evidence items contradicting this claim.
        confidence: Confidence in the evidence assessment (0.0 to 1.0).
        original_dag_id: ID of the DAG this constituent was extracted from.
        original_hypothesis_id: ID of the hypothesis this constituent belonged to.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this constituent.",
    )
    source_variable: Variable = Field(
        ...,
        description="The source variable of the causal claim.",
    )
    target_variable: Variable = Field(
        ...,
        description="The target variable of the causal claim.",
    )
    edge: CausalEdge = Field(
        ...,
        description="The original causal edge from the DAG.",
    )
    status: ConstituentStatus = Field(
        ...,
        description="Evidence status of this causal claim.",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="IDs of evidence items supporting this claim.",
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="IDs of evidence items contradicting this claim.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the evidence assessment.",
    )
    original_dag_id: str = Field(
        ...,
        description="ID of the DAG this constituent was extracted from.",
    )
    original_hypothesis_id: Optional[str] = Field(
        default=None,
        description="ID of the hypothesis this constituent belonged to.",
    )


class DestructionResult(BaseModel):
    """Result of Boyd's destructive deduction on a failing paradigm.

    Contains all the shattered constituents classified by their evidence
    status, plus metadata about the destruction process. The supported
    and unsupported constituents are the raw material for creative induction.

    Attributes:
        id: Unique identifier for this destruction result.
        hypothesis_id: ID of the hypothesis that was shattered.
        dag_id: ID of the CausalDAG that was shattered.
        supported_constituents: Causal claims backed by evidence (preserve these).
        unsupported_constituents: Causal claims with no evidence (investigate these).
        contradicted_constituents: Causal claims refuted by evidence (discard these).
        preserved_variables: Variables that appear in supported claims (still relevant).
        orphaned_variables: Variables that only appeared in contradicted claims.
        destruction_rationale: Explanation of why destruction was initiated.
        timestamp: When the destruction was performed.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this destruction result.",
    )
    hypothesis_id: str = Field(
        ...,
        description="ID of the hypothesis that was shattered.",
    )
    dag_id: str = Field(
        ...,
        description="ID of the CausalDAG that was shattered.",
    )
    supported_constituents: list[ShatteredConstituent] = Field(
        default_factory=list,
        description="Causal claims backed by evidence.",
    )
    unsupported_constituents: list[ShatteredConstituent] = Field(
        default_factory=list,
        description="Causal claims with no evidence for or against.",
    )
    contradicted_constituents: list[ShatteredConstituent] = Field(
        default_factory=list,
        description="Causal claims refuted by evidence.",
    )
    preserved_variables: list[Variable] = Field(
        default_factory=list,
        description="Variables that appear in supported claims.",
    )
    orphaned_variables: list[Variable] = Field(
        default_factory=list,
        description="Variables only in contradicted claims.",
    )
    destruction_rationale: str = Field(
        default="",
        description="Explanation of why destruction was initiated.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the destruction was performed.",
    )

    @property
    def total_constituents(self) -> int:
        """Total number of constituents produced by destruction."""
        return (
            len(self.supported_constituents)
            + len(self.unsupported_constituents)
            + len(self.contradicted_constituents)
        )

    @property
    def survival_rate(self) -> float:
        """Fraction of causal claims that survived destruction (supported)."""
        total = self.total_constituents
        if total == 0:
            return 0.0
        return len(self.supported_constituents) / total


# ---------------------------------------------------------------------------
# Destructive deductor
# ---------------------------------------------------------------------------


class DestructiveDeductor:
    """Boyd's destructive deduction engine.

    Takes a failing hypothesis and its causal DAG, evaluates each causal
    claim against available evidence, and shatters the model into classified
    constituents. The output feeds directly into creative induction.

    Attributes:
        store: The shared KnowledgeGraphStore.
        min_evidence_confidence: Minimum confidence for evidence to be considered.
    """

    def __init__(
        self,
        store: KnowledgeGraphStore,
        min_evidence_confidence: float = 0.3,
    ) -> None:
        """Initialize the destructive deductor.

        Args:
            store: The shared KnowledgeGraphStore.
            min_evidence_confidence: Minimum confidence threshold for
                                    evidence to affect constituent status.
        """
        self.store = store
        self.min_evidence_confidence = min_evidence_confidence

    def shatter(
        self,
        hypothesis_id: str,
        rationale: str = "",
    ) -> DestructionResult:
        """Perform destructive deduction on a hypothesis and its causal model.

        This is the core operation: take a failing paradigm, evaluate every
        causal claim against evidence, and produce classified fragments.

        The process:
        1. Retrieve the hypothesis and its causal DAG
        2. Retrieve all evidence linked to the hypothesis
        3. For each edge in the DAG, determine whether evidence supports
           or contradicts that specific causal claim
        4. Classify each edge as supported, unsupported, or contradicted
        5. Identify which variables are preserved vs. orphaned
        6. Update the hypothesis status to FALSIFIED

        Args:
            hypothesis_id: ID of the hypothesis to shatter.
            rationale: Explanation of why destruction was initiated.

        Returns:
            A DestructionResult containing the classified constituents.

        Raises:
            KeyError: If the hypothesis or its causal model doesn't exist.
            ValueError: If the hypothesis has no associated causal model.
        """
        # Retrieve the hypothesis
        hypothesis = self.store.get_hypothesis(hypothesis_id)
        if hypothesis is None:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found.")

        if not hypothesis.causal_model_id:
            raise ValueError(
                f"Hypothesis '{hypothesis_id}' has no associated causal model. "
                "Destructive deduction requires a CausalDAG to shatter."
            )

        # Retrieve the causal DAG
        dag = self.store.get_causal_dag(hypothesis.causal_model_id)
        if dag is None:
            raise KeyError(
                f"CausalDAG '{hypothesis.causal_model_id}' not found."
            )

        # Retrieve all evidence for this hypothesis
        evidence_items = self.store.get_evidence_for_hypothesis(hypothesis_id)

        # Build variable lookup
        var_lookup: dict[str, Variable] = {v.id: v for v in dag.nodes}

        # Evaluate each edge
        supported: list[ShatteredConstituent] = []
        unsupported: list[ShatteredConstituent] = []
        contradicted: list[ShatteredConstituent] = []

        for edge in dag.edges:
            constituent = self._evaluate_edge(
                edge=edge,
                var_lookup=var_lookup,
                evidence_items=evidence_items,
                dag_id=dag.id,
                hypothesis_id=hypothesis_id,
            )

            if constituent.status == ConstituentStatus.SUPPORTED:
                supported.append(constituent)
            elif constituent.status == ConstituentStatus.CONTRADICTED:
                contradicted.append(constituent)
            else:
                unsupported.append(constituent)

        # Identify preserved and orphaned variables
        preserved_var_ids: set[str] = set()
        for c in supported:
            preserved_var_ids.add(c.edge.source)
            preserved_var_ids.add(c.edge.target)
        for c in unsupported:
            preserved_var_ids.add(c.edge.source)
            preserved_var_ids.add(c.edge.target)

        all_var_ids = {v.id for v in dag.nodes}
        orphaned_var_ids = all_var_ids - preserved_var_ids

        preserved_vars = [var_lookup[vid] for vid in preserved_var_ids if vid in var_lookup]
        orphaned_vars = [var_lookup[vid] for vid in orphaned_var_ids if vid in var_lookup]

        # Mark the hypothesis as falsified
        self.store.update_hypothesis_status(hypothesis_id, HypothesisStatus.FALSIFIED)

        return DestructionResult(
            hypothesis_id=hypothesis_id,
            dag_id=dag.id,
            supported_constituents=supported,
            unsupported_constituents=unsupported,
            contradicted_constituents=contradicted,
            preserved_variables=preserved_vars,
            orphaned_variables=orphaned_vars,
            destruction_rationale=rationale or (
                f"Hypothesis '{hypothesis.title}' entered Kuhnian crisis. "
                f"Destructive deduction initiated to identify salvageable "
                f"causal claims from the failing paradigm."
            ),
        )

    def shatter_dag_only(
        self,
        dag: CausalDAG,
        evidence_items: list[Evidence],
        hypothesis_id: Optional[str] = None,
    ) -> DestructionResult:
        """Shatter a CausalDAG without modifying the knowledge graph.

        Useful for dry-run analysis or when working with DAGs not yet
        registered in the store.

        Args:
            dag: The CausalDAG to shatter.
            evidence_items: Evidence items to evaluate edges against.
            hypothesis_id: Optional hypothesis ID for provenance.

        Returns:
            A DestructionResult with classified constituents.
        """
        var_lookup: dict[str, Variable] = {v.id: v for v in dag.nodes}

        supported: list[ShatteredConstituent] = []
        unsupported: list[ShatteredConstituent] = []
        contradicted: list[ShatteredConstituent] = []

        for edge in dag.edges:
            constituent = self._evaluate_edge(
                edge=edge,
                var_lookup=var_lookup,
                evidence_items=evidence_items,
                dag_id=dag.id,
                hypothesis_id=hypothesis_id,
            )

            if constituent.status == ConstituentStatus.SUPPORTED:
                supported.append(constituent)
            elif constituent.status == ConstituentStatus.CONTRADICTED:
                contradicted.append(constituent)
            else:
                unsupported.append(constituent)

        preserved_var_ids: set[str] = set()
        for c in supported + unsupported:
            preserved_var_ids.add(c.edge.source)
            preserved_var_ids.add(c.edge.target)

        all_var_ids = {v.id for v in dag.nodes}
        orphaned_var_ids = all_var_ids - preserved_var_ids

        preserved_vars = [var_lookup[vid] for vid in preserved_var_ids if vid in var_lookup]
        orphaned_vars = [var_lookup[vid] for vid in orphaned_var_ids if vid in var_lookup]

        return DestructionResult(
            hypothesis_id=hypothesis_id or "",
            dag_id=dag.id,
            supported_constituents=supported,
            unsupported_constituents=unsupported,
            contradicted_constituents=contradicted,
            preserved_variables=preserved_vars,
            orphaned_variables=orphaned_vars,
            destruction_rationale="Dry-run destructive deduction.",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_edge(
        self,
        edge: CausalEdge,
        var_lookup: dict[str, Variable],
        evidence_items: list[Evidence],
        dag_id: str,
        hypothesis_id: Optional[str] = None,
    ) -> ShatteredConstituent:
        """Evaluate a single causal edge against available evidence.

        The evaluation heuristic:
        1. If the edge has explicit evidence IDs, use those
        2. Otherwise, check all evidence for the hypothesis and match
           by variable references in the evidence description
        3. Compute a net evidence score: supporting - contradicting
        4. Classify based on the net score

        Args:
            edge: The CausalEdge to evaluate.
            var_lookup: Mapping from variable ID to Variable.
            evidence_items: All evidence items for the hypothesis.
            dag_id: ID of the source DAG.
            hypothesis_id: ID of the source hypothesis.

        Returns:
            A classified ShatteredConstituent.
        """
        # Collect evidence that directly references this edge
        edge_evidence_ids = set(edge.evidence)
        supporting_ids: list[str] = []
        contradicting_ids: list[str] = []

        for ev in evidence_items:
            if ev.confidence < self.min_evidence_confidence:
                continue

            # Check if this evidence is directly linked to this edge
            is_relevant = ev.id in edge_evidence_ids

            # Heuristic: also check if evidence mentions the source or target
            # variable names in its description
            if not is_relevant:
                source_var = var_lookup.get(edge.source)
                target_var = var_lookup.get(edge.target)
                if source_var and target_var:
                    desc_lower = ev.description.lower()
                    if (
                        source_var.name.lower() in desc_lower
                        and target_var.name.lower() in desc_lower
                    ):
                        is_relevant = True

            if is_relevant:
                if ev.type == EvidenceType.SUPPORTING:
                    supporting_ids.append(ev.id)
                elif ev.type == EvidenceType.CONTRADICTING:
                    contradicting_ids.append(ev.id)

        # Determine status based on evidence balance
        if supporting_ids and not contradicting_ids:
            status = ConstituentStatus.SUPPORTED
            confidence = min(1.0, len(supporting_ids) * 0.3 + 0.2)
        elif contradicting_ids and not supporting_ids:
            status = ConstituentStatus.CONTRADICTED
            confidence = min(1.0, len(contradicting_ids) * 0.3 + 0.2)
        elif supporting_ids and contradicting_ids:
            # Mixed evidence — net score determines
            net = len(supporting_ids) - len(contradicting_ids)
            if net > 0:
                status = ConstituentStatus.SUPPORTED
                confidence = 0.3  # Lower confidence due to mixed evidence
            elif net < 0:
                status = ConstituentStatus.CONTRADICTED
                confidence = 0.3
            else:
                status = ConstituentStatus.UNSUPPORTED
                confidence = 0.2
        else:
            # No evidence at all
            status = ConstituentStatus.UNSUPPORTED
            confidence = 0.1

        source_var = var_lookup.get(edge.source, Variable(name=edge.source, type="observable"))
        target_var = var_lookup.get(edge.target, Variable(name=edge.target, type="observable"))

        return ShatteredConstituent(
            source_variable=source_var,
            target_variable=target_var,
            edge=edge,
            status=status,
            supporting_evidence=supporting_ids,
            contradicting_evidence=contradicting_ids,
            confidence=confidence,
            original_dag_id=dag_id,
            original_hypothesis_id=hypothesis_id,
        )
