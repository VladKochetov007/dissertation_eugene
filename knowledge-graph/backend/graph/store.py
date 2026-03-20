"""
Knowledge Graph Store — the nervous system of the Republic of AI Agents.

Implements an in-memory graph store backed by NetworkX, with entity registries
for typed lookup and temporal tracking of all mutations. Designed to be the
single source of truth connecting philosopher-kings, merchants, and warriors.

Usage:
    store = KnowledgeGraphStore()
    store.add_hypothesis(hypothesis)
    store.add_causal_dag(dag)
    neighbors = store.query_neighbors("node-id", depth=2)
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

import networkx as nx

from .entities import (
    CausalDAG,
    Evidence,
    Experiment,
    Hypothesis,
    HypothesisStatus,
    Variable,
)
from .knowledge_entities import (
    Concept,
    Domain,
    Era,
    HistoricalPeriod,
    Thinker,
    Tradition,
    Work,
)


class MutationRecord:
    """A timestamped record of a mutation to the knowledge graph.

    Every write operation is logged for auditability and temporal queries.
    """

    __slots__ = ("timestamp", "operation", "entity_type", "entity_id", "data")

    def __init__(
        self,
        operation: str,
        entity_type: str,
        entity_id: str,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        self.timestamp = datetime.now(timezone.utc)
        self.operation = operation
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.data = data or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the mutation record to a dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "data": self.data,
        }


class KnowledgeGraphStore:
    """In-memory knowledge graph backed by a NetworkX directed graph.

    Provides typed entity registries for fast lookup and a NetworkX DiGraph
    for structural queries (paths, neighbors, connectivity). All mutations
    are tracked with timestamps for temporal auditing.

    Attributes:
        graph: The underlying NetworkX directed graph.
        hypotheses: Registry mapping hypothesis IDs to Hypothesis objects.
        variables: Registry mapping variable IDs to Variable objects.
        causal_dags: Registry mapping DAG IDs to CausalDAG objects.
        evidence: Registry mapping evidence IDs to Evidence objects.
        experiments: Registry mapping experiment IDs to Experiment objects.
        mutations: Ordered list of all mutation records.
    """

    def __init__(self) -> None:
        """Initialize an empty knowledge graph store."""
        self.graph: nx.DiGraph = nx.DiGraph()

        # Operational entity registries — typed dictionaries for O(1) lookup
        self.hypotheses: dict[str, Hypothesis] = {}
        self.variables: dict[str, Variable] = {}
        self.causal_dags: dict[str, CausalDAG] = {}
        self.evidence: dict[str, Evidence] = {}
        self.experiments: dict[str, Experiment] = {}

        # Knowledge foundation entity registries
        self.thinkers: dict[str, Thinker] = {}
        self.concepts: dict[str, Concept] = {}
        self.traditions: dict[str, Tradition] = {}
        self.works: dict[str, Work] = {}
        self.historical_periods: dict[str, HistoricalPeriod] = {}
        self.domains: dict[str, Domain] = {}

        # Temporal tracking
        self.mutations: list[MutationRecord] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record_mutation(
        self,
        operation: str,
        entity_type: str,
        entity_id: str,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a mutation for temporal tracking."""
        self.mutations.append(
            MutationRecord(
                operation=operation,
                entity_type=entity_type,
                entity_id=entity_id,
                data=data,
            )
        )

    # ------------------------------------------------------------------
    # Hypothesis operations
    # ------------------------------------------------------------------

    def add_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        """Register a new hypothesis in the knowledge graph.

        Adds the hypothesis to the entity registry and creates a node in
        the graph with type metadata.

        Args:
            hypothesis: The Hypothesis to register.

        Returns:
            The registered Hypothesis (unchanged).

        Raises:
            ValueError: If a hypothesis with the same ID already exists.
        """
        if hypothesis.id in self.hypotheses:
            raise ValueError(f"Hypothesis with id '{hypothesis.id}' already exists.")

        self.hypotheses[hypothesis.id] = hypothesis
        self.graph.add_node(
            hypothesis.id,
            entity_type="hypothesis",
            label=hypothesis.title,
        )

        # Link hypothesis to its variables
        for var_id in hypothesis.variables:
            if var_id in self.variables:
                self.graph.add_edge(
                    hypothesis.id,
                    var_id,
                    relation="references_variable",
                )

        # Link hypothesis to its causal model
        if hypothesis.causal_model_id and hypothesis.causal_model_id in self.causal_dags:
            self.graph.add_edge(
                hypothesis.id,
                hypothesis.causal_model_id,
                relation="has_causal_model",
            )

        self._record_mutation("add", "hypothesis", hypothesis.id)
        return hypothesis

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Retrieve a hypothesis by ID.

        Args:
            hypothesis_id: The unique identifier of the hypothesis.

        Returns:
            The Hypothesis if found, None otherwise.
        """
        return self.hypotheses.get(hypothesis_id)

    def list_hypotheses(
        self,
        status: Optional[HypothesisStatus] = None,
    ) -> list[Hypothesis]:
        """List all hypotheses, optionally filtered by status.

        Args:
            status: If provided, only return hypotheses with this status.

        Returns:
            List of matching Hypothesis objects.
        """
        if status is None:
            return list(self.hypotheses.values())
        return [h for h in self.hypotheses.values() if h.status == status]

    def update_hypothesis_status(
        self,
        hypothesis_id: str,
        new_status: HypothesisStatus,
    ) -> Hypothesis:
        """Update the lifecycle status of a hypothesis.

        Args:
            hypothesis_id: The ID of the hypothesis to update.
            new_status: The new status to assign.

        Returns:
            The updated Hypothesis.

        Raises:
            KeyError: If the hypothesis does not exist.
        """
        if hypothesis_id not in self.hypotheses:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found.")

        hypothesis = self.hypotheses[hypothesis_id]
        old_status = hypothesis.status
        hypothesis.status = new_status
        hypothesis.updated_at = datetime.now(timezone.utc)

        self._record_mutation(
            "update_status",
            "hypothesis",
            hypothesis_id,
            {"old_status": old_status.value, "new_status": new_status.value},
        )
        return hypothesis

    # ------------------------------------------------------------------
    # Variable operations
    # ------------------------------------------------------------------

    def add_variable(self, variable: Variable) -> Variable:
        """Register a new variable in the knowledge graph.

        Args:
            variable: The Variable to register.

        Returns:
            The registered Variable.

        Raises:
            ValueError: If a variable with the same ID already exists.
        """
        if variable.id in self.variables:
            raise ValueError(f"Variable with id '{variable.id}' already exists.")

        self.variables[variable.id] = variable
        self.graph.add_node(
            variable.id,
            entity_type="variable",
            label=variable.name,
            variable_type=variable.type.value,
        )
        self._record_mutation("add", "variable", variable.id)
        return variable

    def get_variable(self, variable_id: str) -> Optional[Variable]:
        """Retrieve a variable by ID.

        Args:
            variable_id: The unique identifier of the variable.

        Returns:
            The Variable if found, None otherwise.
        """
        return self.variables.get(variable_id)

    # ------------------------------------------------------------------
    # Causal DAG operations
    # ------------------------------------------------------------------

    def add_causal_dag(self, dag: CausalDAG) -> CausalDAG:
        """Register a causal DAG and integrate its structure into the graph.

        Each variable node and causal edge from the DAG is added to the
        underlying NetworkX graph, creating a unified knowledge structure.

        Args:
            dag: The CausalDAG to register.

        Returns:
            The registered CausalDAG.

        Raises:
            ValueError: If a DAG with the same ID already exists.
        """
        if dag.id in self.causal_dags:
            raise ValueError(f"CausalDAG with id '{dag.id}' already exists.")

        self.causal_dags[dag.id] = dag

        # Add the DAG as a container node
        self.graph.add_node(
            dag.id,
            entity_type="causal_dag",
            label=f"DAG v{dag.version}",
        )

        # Add all variable nodes from the DAG
        for var in dag.nodes:
            if var.id not in self.variables:
                self.add_variable(var)
            # Link DAG to its variables
            self.graph.add_edge(dag.id, var.id, relation="contains_variable")

        # Add causal edges
        for edge in dag.edges:
            self.graph.add_edge(
                edge.source,
                edge.target,
                relation="causal",
                edge_type=edge.type.value,
                strength=edge.strength,
            )

        # Link to hypothesis if specified
        if dag.hypothesis_id and dag.hypothesis_id in self.hypotheses:
            self.graph.add_edge(
                dag.hypothesis_id,
                dag.id,
                relation="has_causal_model",
            )

        self._record_mutation("add", "causal_dag", dag.id)
        return dag

    def get_causal_dag(self, dag_id: str) -> Optional[CausalDAG]:
        """Retrieve a causal DAG by ID.

        Args:
            dag_id: The unique identifier of the DAG.

        Returns:
            The CausalDAG if found, None otherwise.
        """
        return self.causal_dags.get(dag_id)

    # ------------------------------------------------------------------
    # Evidence operations
    # ------------------------------------------------------------------

    def add_evidence(self, evidence_item: Evidence) -> Evidence:
        """Register new evidence and link it to its hypothesis.

        Args:
            evidence_item: The Evidence to register.

        Returns:
            The registered Evidence.

        Raises:
            ValueError: If evidence with the same ID already exists.
            KeyError: If the referenced hypothesis does not exist.
        """
        if evidence_item.id in self.evidence:
            raise ValueError(f"Evidence with id '{evidence_item.id}' already exists.")

        if evidence_item.hypothesis_id not in self.hypotheses:
            raise KeyError(
                f"Hypothesis '{evidence_item.hypothesis_id}' not found. "
                "Evidence must reference an existing hypothesis."
            )

        self.evidence[evidence_item.id] = evidence_item

        # Add evidence node and link to hypothesis
        self.graph.add_node(
            evidence_item.id,
            entity_type="evidence",
            label=f"Evidence ({evidence_item.type.value})",
        )
        self.graph.add_edge(
            evidence_item.id,
            evidence_item.hypothesis_id,
            relation=f"evidence_{evidence_item.type.value}",
        )

        # Update the hypothesis's evidence list
        hypothesis = self.hypotheses[evidence_item.hypothesis_id]
        if evidence_item.id not in hypothesis.evidence:
            hypothesis.evidence.append(evidence_item.id)
            hypothesis.updated_at = datetime.now(timezone.utc)

        self._record_mutation("add", "evidence", evidence_item.id)
        return evidence_item

    def get_evidence_for_hypothesis(self, hypothesis_id: str) -> list[Evidence]:
        """Retrieve all evidence linked to a specific hypothesis.

        Args:
            hypothesis_id: The ID of the hypothesis.

        Returns:
            List of Evidence objects linked to the hypothesis.
        """
        return [
            e for e in self.evidence.values()
            if e.hypothesis_id == hypothesis_id
        ]

    # ------------------------------------------------------------------
    # Experiment operations
    # ------------------------------------------------------------------

    def add_experiment(self, experiment: Experiment) -> Experiment:
        """Register a new experiment.

        Args:
            experiment: The Experiment to register.

        Returns:
            The registered Experiment.

        Raises:
            ValueError: If an experiment with the same ID already exists.
        """
        if experiment.id in self.experiments:
            raise ValueError(f"Experiment with id '{experiment.id}' already exists.")

        self.experiments[experiment.id] = experiment
        self.graph.add_node(
            experiment.id,
            entity_type="experiment",
            label=f"Experiment ({experiment.type.value})",
        )

        if experiment.hypothesis_id in self.hypotheses:
            self.graph.add_edge(
                experiment.id,
                experiment.hypothesis_id,
                relation="tests_hypothesis",
            )

        self._record_mutation("add", "experiment", experiment.id)
        return experiment

    # ------------------------------------------------------------------
    # Domain operations (knowledge foundation)
    # ------------------------------------------------------------------

    def add_domain(self, domain: Domain) -> Domain:
        """Register a knowledge domain."""
        if domain.id in self.domains:
            raise ValueError(f"Domain with id '{domain.id}' already exists.")

        self.domains[domain.id] = domain
        self.graph.add_node(domain.id, entity_type="domain", label=domain.name)

        if domain.parent_domain_id and domain.parent_domain_id in self.domains:
            self.graph.add_edge(domain.id, domain.parent_domain_id, relation="subdomain_of")

        self._record_mutation("add", "domain", domain.id)
        return domain

    def get_domain(self, domain_id: str) -> Optional[Domain]:
        return self.domains.get(domain_id)

    def list_domains(self) -> list[Domain]:
        return list(self.domains.values())

    # ------------------------------------------------------------------
    # Tradition operations (knowledge foundation)
    # ------------------------------------------------------------------

    def add_tradition(self, tradition: Tradition) -> Tradition:
        """Register a tradition (school of thought, religion, movement)."""
        if tradition.id in self.traditions:
            raise ValueError(f"Tradition with id '{tradition.id}' already exists.")

        self.traditions[tradition.id] = tradition
        self.graph.add_node(tradition.id, entity_type="tradition", label=tradition.name)

        for pid in tradition.parent_tradition_ids:
            if pid in self.traditions:
                self.graph.add_edge(tradition.id, pid, relation="child_of_tradition")

        self._record_mutation("add", "tradition", tradition.id)
        return tradition

    def get_tradition(self, tradition_id: str) -> Optional[Tradition]:
        return self.traditions.get(tradition_id)

    def list_traditions(self) -> list[Tradition]:
        return list(self.traditions.values())

    # ------------------------------------------------------------------
    # Historical period operations (knowledge foundation)
    # ------------------------------------------------------------------

    def add_historical_period(self, period: HistoricalPeriod) -> HistoricalPeriod:
        """Register a historical period / epoch."""
        if period.id in self.historical_periods:
            raise ValueError(f"HistoricalPeriod with id '{period.id}' already exists.")

        self.historical_periods[period.id] = period
        self.graph.add_node(period.id, entity_type="historical_period", label=period.name)

        self._record_mutation("add", "historical_period", period.id)
        return period

    def get_historical_period(self, period_id: str) -> Optional[HistoricalPeriod]:
        return self.historical_periods.get(period_id)

    def list_historical_periods(self) -> list[HistoricalPeriod]:
        return list(self.historical_periods.values())

    # ------------------------------------------------------------------
    # Thinker operations (knowledge foundation)
    # ------------------------------------------------------------------

    def add_thinker(self, thinker: Thinker) -> Thinker:
        """Register a thinker and create graph edges to related entities."""
        if thinker.id in self.thinkers:
            raise ValueError(f"Thinker with id '{thinker.id}' already exists.")

        self.thinkers[thinker.id] = thinker
        self.graph.add_node(thinker.id, entity_type="thinker", label=thinker.name)

        for tid in thinker.traditions:
            if tid in self.traditions:
                self.graph.add_edge(thinker.id, tid, relation="belongs_to_tradition")

        for did in thinker.domains:
            if did in self.domains:
                self.graph.add_edge(thinker.id, did, relation="works_in_domain")

        for cid in thinker.key_concepts:
            if cid in self.concepts:
                self.graph.add_edge(thinker.id, cid, relation="developed_concept")

        for wid in thinker.works:
            if wid in self.works:
                self.graph.add_edge(thinker.id, wid, relation="authored_work")

        for rel in thinker.related_thinkers:
            if rel.target_id in self.thinkers:
                self.graph.add_edge(
                    thinker.id, rel.target_id,
                    relation=rel.relation_type.value,
                )

        self._record_mutation("add", "thinker", thinker.id)
        return thinker

    def get_thinker(self, thinker_id: str) -> Optional[Thinker]:
        return self.thinkers.get(thinker_id)

    def list_thinkers(
        self,
        era: Optional[Era] = None,
        tradition_id: Optional[str] = None,
        domain_id: Optional[str] = None,
        tier: Optional[int] = None,
    ) -> list[Thinker]:
        results = list(self.thinkers.values())
        if era is not None:
            results = [t for t in results if t.era == era]
        if tradition_id is not None:
            results = [t for t in results if tradition_id in t.traditions]
        if domain_id is not None:
            results = [t for t in results if domain_id in t.domains]
        if tier is not None:
            results = [t for t in results if t.tier == tier]
        return results

    # ------------------------------------------------------------------
    # Concept operations (knowledge foundation)
    # ------------------------------------------------------------------

    def add_concept(self, concept: Concept) -> Concept:
        """Register a concept and create graph edges to related entities."""
        if concept.id in self.concepts:
            raise ValueError(f"Concept with id '{concept.id}' already exists.")

        self.concepts[concept.id] = concept
        self.graph.add_node(concept.id, entity_type="concept", label=concept.name)

        for did in concept.domain_ids:
            if did in self.domains:
                self.graph.add_edge(concept.id, did, relation="in_domain")

        if concept.originator_id and concept.originator_id in self.thinkers:
            self.graph.add_edge(concept.originator_id, concept.id, relation="originated_concept")

        for dev_id in concept.developer_ids:
            if dev_id in self.thinkers:
                self.graph.add_edge(dev_id, concept.id, relation="developed_concept")

        for tid in concept.tradition_ids:
            if tid in self.traditions:
                self.graph.add_edge(concept.id, tid, relation="part_of_tradition")

        for rel in concept.related_concepts:
            if rel.target_id in self.concepts:
                self.graph.add_edge(
                    concept.id, rel.target_id,
                    relation=f"{rel.relation_type.value}_concept",
                    description=rel.description,
                )

        self._record_mutation("add", "concept", concept.id)
        return concept

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        return self.concepts.get(concept_id)

    def list_concepts(
        self,
        domain_id: Optional[str] = None,
        tradition_id: Optional[str] = None,
    ) -> list[Concept]:
        results = list(self.concepts.values())
        if domain_id is not None:
            results = [c for c in results if domain_id in c.domain_ids]
        if tradition_id is not None:
            results = [c for c in results if tradition_id in c.tradition_ids]
        return results

    # ------------------------------------------------------------------
    # Work operations (knowledge foundation)
    # ------------------------------------------------------------------

    def add_work(self, work: Work) -> Work:
        """Register a work (book, paper, scripture, etc.)."""
        if work.id in self.works:
            raise ValueError(f"Work with id '{work.id}' already exists.")

        self.works[work.id] = work
        self.graph.add_node(work.id, entity_type="work", label=work.title)

        for aid in work.author_ids:
            if aid in self.thinkers:
                self.graph.add_edge(aid, work.id, relation="authored_work")

        for tid in work.tradition_ids:
            if tid in self.traditions:
                self.graph.add_edge(work.id, tid, relation="part_of_tradition")

        for cid in work.concepts_introduced:
            if cid in self.concepts:
                self.graph.add_edge(work.id, cid, relation="introduces_concept")

        for cid in work.concepts_developed:
            if cid in self.concepts:
                self.graph.add_edge(work.id, cid, relation="develops_concept")

        for ref_id in work.references_work_ids:
            if ref_id in self.works:
                self.graph.add_edge(work.id, ref_id, relation="references_work")

        self._record_mutation("add", "work", work.id)
        return work

    def get_work(self, work_id: str) -> Optional[Work]:
        return self.works.get(work_id)

    def list_works(
        self,
        author_id: Optional[str] = None,
        tradition_id: Optional[str] = None,
    ) -> list[Work]:
        results = list(self.works.values())
        if author_id is not None:
            results = [w for w in results if author_id in w.author_ids]
        if tradition_id is not None:
            results = [w for w in results if tradition_id in w.tradition_ids]
        return results

    # ------------------------------------------------------------------
    # Bulk operations (knowledge foundation)
    # ------------------------------------------------------------------

    def bulk_add(self, entities: list) -> dict[str, int]:
        """Add multiple entities in dependency order.

        Entities are sorted by type: Domains first, then Traditions,
        then HistoricalPeriods, then Thinkers, then Concepts, then Works.
        This ensures references resolve correctly during insertion.

        Returns:
            Dict mapping entity type name to count added.
        """
        type_order = {
            Domain: 0, Tradition: 1, HistoricalPeriod: 2,
            Thinker: 3, Concept: 4, Work: 5,
        }
        sorted_entities = sorted(entities, key=lambda e: type_order.get(type(e), 99))

        dispatch = {
            Domain: self.add_domain,
            Tradition: self.add_tradition,
            HistoricalPeriod: self.add_historical_period,
            Thinker: self.add_thinker,
            Concept: self.add_concept,
            Work: self.add_work,
        }

        counts: dict[str, int] = {}
        for entity in sorted_entities:
            entity_type = type(entity)
            add_fn = dispatch.get(entity_type)
            if add_fn:
                add_fn(entity)
                type_name = entity_type.__name__.lower()
                counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def resolve_deferred_edges(self) -> int:
        """Create graph edges for references that were unresolvable at insert time.

        Should be called after bulk loading to wire up all cross-references.

        Returns:
            Number of new edges created.
        """
        new_edges = 0

        for thinker in self.thinkers.values():
            for tid in thinker.traditions:
                if tid in self.traditions and not self.graph.has_edge(thinker.id, tid):
                    self.graph.add_edge(thinker.id, tid, relation="belongs_to_tradition")
                    new_edges += 1
            for did in thinker.domains:
                if did in self.domains and not self.graph.has_edge(thinker.id, did):
                    self.graph.add_edge(thinker.id, did, relation="works_in_domain")
                    new_edges += 1
            for cid in thinker.key_concepts:
                if cid in self.concepts and not self.graph.has_edge(thinker.id, cid):
                    self.graph.add_edge(thinker.id, cid, relation="developed_concept")
                    new_edges += 1
            for wid in thinker.works:
                if wid in self.works and not self.graph.has_edge(thinker.id, wid):
                    self.graph.add_edge(thinker.id, wid, relation="authored_work")
                    new_edges += 1
            for rel in thinker.related_thinkers:
                if rel.target_id in self.thinkers and not self.graph.has_edge(thinker.id, rel.target_id):
                    self.graph.add_edge(thinker.id, rel.target_id, relation=rel.relation_type.value)
                    new_edges += 1

        for concept in self.concepts.values():
            if concept.originator_id and concept.originator_id in self.thinkers:
                if not self.graph.has_edge(concept.originator_id, concept.id):
                    self.graph.add_edge(concept.originator_id, concept.id, relation="originated_concept")
                    new_edges += 1
            for dev_id in concept.developer_ids:
                if dev_id in self.thinkers and not self.graph.has_edge(dev_id, concept.id):
                    self.graph.add_edge(dev_id, concept.id, relation="developed_concept")
                    new_edges += 1
            for rel in concept.related_concepts:
                if rel.target_id in self.concepts and not self.graph.has_edge(concept.id, rel.target_id):
                    self.graph.add_edge(concept.id, rel.target_id, relation=f"{rel.relation_type.value}_concept")
                    new_edges += 1

        for work in self.works.values():
            for aid in work.author_ids:
                if aid in self.thinkers and not self.graph.has_edge(aid, work.id):
                    self.graph.add_edge(aid, work.id, relation="authored_work")
                    new_edges += 1
            for cid in work.concepts_introduced:
                if cid in self.concepts and not self.graph.has_edge(work.id, cid):
                    self.graph.add_edge(work.id, cid, relation="introduces_concept")
                    new_edges += 1
            for cid in work.concepts_developed:
                if cid in self.concepts and not self.graph.has_edge(work.id, cid):
                    self.graph.add_edge(work.id, cid, relation="develops_concept")
                    new_edges += 1

        for tradition in self.traditions.values():
            for pid in tradition.parent_tradition_ids:
                if pid in self.traditions and not self.graph.has_edge(tradition.id, pid):
                    self.graph.add_edge(tradition.id, pid, relation="child_of_tradition")
                    new_edges += 1

        for domain in self.domains.values():
            if domain.parent_domain_id and domain.parent_domain_id in self.domains:
                if not self.graph.has_edge(domain.id, domain.parent_domain_id):
                    self.graph.add_edge(domain.id, domain.parent_domain_id, relation="subdomain_of")
                    new_edges += 1

        return new_edges

    # ------------------------------------------------------------------
    # Knowledge-specific queries
    # ------------------------------------------------------------------

    def get_intellectual_lineage(self, thinker_id: str) -> dict[str, list[str]]:
        """Trace a thinker's intellectual lineage.

        Returns predecessors (who influenced them) and successors
        (who they influenced) via the graph's influence edges.
        """
        if thinker_id not in self.thinkers:
            raise KeyError(f"Thinker '{thinker_id}' not found.")

        predecessors = []
        successors = []

        # Incoming influence edges (others -> this thinker)
        for pred in self.graph.predecessors(thinker_id):
            edge_data = self.graph.get_edge_data(pred, thinker_id, {})
            rel = edge_data.get("relation", "")
            if rel in ("influenced", "student_of") and pred in self.thinkers:
                predecessors.append(pred)

        # Outgoing influence edges (this thinker -> others)
        for succ in self.graph.successors(thinker_id):
            edge_data = self.graph.get_edge_data(thinker_id, succ, {})
            rel = edge_data.get("relation", "")
            if rel in ("influenced", "student_of") and succ in self.thinkers:
                successors.append(succ)

        return {"predecessors": predecessors, "successors": successors}

    def get_concept_dialectic(self, concept_id: str) -> dict[str, list[str]]:
        """Return a concept's dialectical context: what it extends,
        what contradicts it, what synthesizes from it."""
        if concept_id not in self.concepts:
            raise KeyError(f"Concept '{concept_id}' not found.")

        result: dict[str, list[str]] = {
            "extends": [], "contradicts": [], "synthesizes": [],
            "formalizes": [], "analogous_to": [], "extended_by": [],
            "contradicted_by": [], "synthesized_by": [],
        }

        concept = self.concepts[concept_id]
        for rel in concept.related_concepts:
            key = rel.relation_type.value
            if key in result:
                result[key].append(rel.target_id)

        # Reverse: find concepts that reference this one
        for other in self.concepts.values():
            for rel in other.related_concepts:
                if rel.target_id == concept_id:
                    reverse_key = f"{rel.relation_type.value}d_by"
                    if reverse_key not in result:
                        reverse_key = f"{rel.relation_type.value}_by"
                    if reverse_key in result:
                        result[reverse_key].append(other.id)

        return result

    def search_by_manuscript_chapter(self, chapter: int) -> dict[str, list[str]]:
        """Return all entity IDs referencing a given manuscript chapter."""
        result: dict[str, list[str]] = {
            "thinkers": [], "concepts": [], "traditions": [],
            "works": [], "historical_periods": [], "domains": [],
        }

        for t in self.thinkers.values():
            if any(r.chapter == chapter for r in t.manuscript_refs):
                result["thinkers"].append(t.id)
        for c in self.concepts.values():
            if any(r.chapter == chapter for r in c.manuscript_refs):
                result["concepts"].append(c.id)
        for tr in self.traditions.values():
            if any(r.chapter == chapter for r in tr.manuscript_refs):
                result["traditions"].append(tr.id)
        for w in self.works.values():
            if any(r.chapter == chapter for r in w.manuscript_refs):
                result["works"].append(w.id)
        for hp in self.historical_periods.values():
            if any(r.chapter == chapter for r in hp.manuscript_refs):
                result["historical_periods"].append(hp.id)
        for d in self.domains.values():
            if any(r.chapter == chapter for r in d.manuscript_refs):
                result["domains"].append(d.id)

        return result

    # ------------------------------------------------------------------
    # Graph query operations
    # ------------------------------------------------------------------

    def query_neighbors(
        self,
        node_id: str,
        depth: int = 1,
    ) -> dict[str, list[str]]:
        """Find all neighbors of a node up to a given depth using BFS.

        Args:
            node_id: The starting node ID.
            depth: Maximum traversal depth (default 1).

        Returns:
            Dictionary mapping depth level (as string) to list of node IDs
            discovered at that level.

        Raises:
            KeyError: If the starting node does not exist in the graph.
        """
        if node_id not in self.graph:
            raise KeyError(f"Node '{node_id}' not found in graph.")

        visited: set[str] = {node_id}
        result: dict[str, list[str]] = {}
        queue: deque[tuple[str, int]] = deque()

        # Seed with immediate neighbors (both directions for full traversal)
        undirected = self.graph.to_undirected()
        for neighbor in undirected.neighbors(node_id):
            if neighbor not in visited:
                queue.append((neighbor, 1))
                visited.add(neighbor)

        while queue:
            current, current_depth = queue.popleft()
            if current_depth > depth:
                break

            level_key = str(current_depth)
            if level_key not in result:
                result[level_key] = []
            result[level_key].append(current)

            if current_depth < depth:
                for neighbor in undirected.neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, current_depth + 1))
                        visited.add(neighbor)

        return result

    def get_causal_path(
        self,
        source: str,
        target: str,
    ) -> list[list[str]]:
        """Find all directed causal paths between two nodes.

        Only follows edges in the causal direction (source -> target).

        Args:
            source: ID of the source node.
            target: ID of the target node.

        Returns:
            List of paths, where each path is a list of node IDs.

        Raises:
            KeyError: If source or target node does not exist.
        """
        if source not in self.graph:
            raise KeyError(f"Source node '{source}' not found in graph.")
        if target not in self.graph:
            raise KeyError(f"Target node '{target}' not found in graph.")

        try:
            return list(nx.all_simple_paths(self.graph, source, target))
        except nx.NetworkXError:
            return []

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize the entire knowledge graph store to a JSON string.

        Returns:
            JSON string containing all entities, graph structure, and
            mutation history.
        """
        data = {
            # Operational layer
            "hypotheses": {
                k: v.model_dump(mode="json") for k, v in self.hypotheses.items()
            },
            "variables": {
                k: v.model_dump(mode="json") for k, v in self.variables.items()
            },
            "causal_dags": {
                k: v.model_dump(mode="json") for k, v in self.causal_dags.items()
            },
            "evidence": {
                k: v.model_dump(mode="json") for k, v in self.evidence.items()
            },
            "experiments": {
                k: v.model_dump(mode="json") for k, v in self.experiments.items()
            },
            # Knowledge foundation layer
            "thinkers": {
                k: v.model_dump(mode="json") for k, v in self.thinkers.items()
            },
            "concepts": {
                k: v.model_dump(mode="json") for k, v in self.concepts.items()
            },
            "traditions": {
                k: v.model_dump(mode="json") for k, v in self.traditions.items()
            },
            "works": {
                k: v.model_dump(mode="json") for k, v in self.works.items()
            },
            "historical_periods": {
                k: v.model_dump(mode="json") for k, v in self.historical_periods.items()
            },
            "domains": {
                k: v.model_dump(mode="json") for k, v in self.domains.items()
            },
            # Graph and audit trail
            "graph": nx.node_link_data(self.graph),
            "mutations": [m.to_dict() for m in self.mutations],
        }
        return json.dumps(data, indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "KnowledgeGraphStore":
        """Deserialize a knowledge graph store from a JSON string.

        Reconstructs all entity registries and the NetworkX graph from
        a previously serialized store.

        Args:
            json_str: JSON string produced by to_json().

        Returns:
            A fully reconstructed KnowledgeGraphStore.
        """
        data = json.loads(json_str)
        store = cls()

        # Restore entities
        for _id, h_data in data.get("hypotheses", {}).items():
            store.hypotheses[_id] = Hypothesis.model_validate(h_data)

        for _id, v_data in data.get("variables", {}).items():
            store.variables[_id] = Variable.model_validate(v_data)

        for _id, d_data in data.get("causal_dags", {}).items():
            store.causal_dags[_id] = CausalDAG.model_validate(d_data)

        for _id, e_data in data.get("evidence", {}).items():
            store.evidence[_id] = Evidence.model_validate(e_data)

        for _id, ex_data in data.get("experiments", {}).items():
            store.experiments[_id] = Experiment.model_validate(ex_data)

        # Knowledge foundation layer
        for _id, t_data in data.get("thinkers", {}).items():
            store.thinkers[_id] = Thinker.model_validate(t_data)

        for _id, c_data in data.get("concepts", {}).items():
            store.concepts[_id] = Concept.model_validate(c_data)

        for _id, tr_data in data.get("traditions", {}).items():
            store.traditions[_id] = Tradition.model_validate(tr_data)

        for _id, w_data in data.get("works", {}).items():
            store.works[_id] = Work.model_validate(w_data)

        for _id, hp_data in data.get("historical_periods", {}).items():
            store.historical_periods[_id] = HistoricalPeriod.model_validate(hp_data)

        for _id, d_data in data.get("domains", {}).items():
            store.domains[_id] = Domain.model_validate(d_data)

        # Restore graph structure
        if "graph" in data:
            store.graph = nx.node_link_graph(data["graph"])

        return store

    # ------------------------------------------------------------------
    # Statistics and introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return summary statistics about the knowledge graph.

        Returns:
            Dictionary with counts of each entity type, graph nodes,
            graph edges, and total mutations.
        """
        return {
            "hypotheses": len(self.hypotheses),
            "variables": len(self.variables),
            "causal_dags": len(self.causal_dags),
            "evidence": len(self.evidence),
            "experiments": len(self.experiments),
            "thinkers": len(self.thinkers),
            "concepts": len(self.concepts),
            "traditions": len(self.traditions),
            "works": len(self.works),
            "historical_periods": len(self.historical_periods),
            "domains": len(self.domains),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "mutations": len(self.mutations),
        }
