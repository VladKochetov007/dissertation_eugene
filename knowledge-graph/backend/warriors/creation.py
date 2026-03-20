"""
Boyd's Creative Induction — synthesizing new paradigms from shattered parts.

Implements the second half of Boyd's dialectical engine from "Destruction and
Creation" (1976). After destructive deduction shatters a failing paradigm into
classified constituents, creative induction recombines the surviving fragments
with new data from merchants to synthesize a new hypothesis and causal model.

The process:
1. Take the supported and unsupported constituents from destructive deduction
2. Incorporate new data/observations from merchant agents
3. Find "common qualities" among fragments — shared variables, semantic
   similarity between concepts, structural patterns
4. Synthesize new causal edges from the recombined fragments + new data
5. Produce a new CausalDAG and Hypothesis that explains both the old
   evidence and the new observations that broke the old paradigm

Boyd's insight: creative induction finds the common threads across
DIFFERENT domains/fragments to produce genuinely new understanding.
This is the opposite of deduction (which breaks things apart) — it's
the synthetic, creative movement that generates novel concepts.

In the theological framework: this is the Resurrection after the Crucifixion.
The old paradigm died (destruction). The new paradigm emerges from the
fragments, incorporating what was true in the old while transcending its
limitations. The dialectical spiral ascends.

References:
    Boyd, J. (1976). Destruction and Creation.
    Hegel, G.W.F. — Science of Logic (thesis-antithesis-synthesis).

Usage:
    inductor = CreativeInductor(store=store)
    result = inductor.synthesize(
        destruction_result=destruction_result,
        new_observations=new_data,
        author="philosopher-king-001",
    )
    new_hypothesis = result.hypothesis
    new_dag = result.causal_dag
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from graph.entities import (
    CausalDAG,
    CausalEdge,
    EdgeType,
    Hypothesis,
    HypothesisStatus,
    Prediction,
    Variable,
    VariableType,
)
from graph.store import KnowledgeGraphStore

from .destruction import (
    ConstituentStatus,
    DestructionResult,
    ShatteredConstituent,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CommonQuality(BaseModel):
    """A shared quality discovered across shattered constituents.

    Common qualities are the "threads" that Boyd's creative induction
    identifies when examining fragments from different contexts. They
    represent structural patterns that survive paradigm destruction.

    Attributes:
        id: Unique identifier.
        description: Human-readable description of the shared quality.
        variable_ids: Variables that participate in this quality.
        constituent_ids: IDs of constituents that share this quality.
        similarity_score: Strength of the commonality (0.0 to 1.0).
        quality_type: What kind of commonality this is.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier.",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the shared quality.",
    )
    variable_ids: list[str] = Field(
        default_factory=list,
        description="Variable IDs participating in this quality.",
    )
    constituent_ids: list[str] = Field(
        default_factory=list,
        description="IDs of constituents sharing this quality.",
    )
    similarity_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Strength of the commonality.",
    )
    quality_type: str = Field(
        default="structural",
        description=(
            "Type of commonality: 'structural' (shared graph patterns), "
            "'semantic' (embedding similarity), 'variable' (shared variables), "
            "'evidential' (supported by same evidence)."
        ),
    )


class SynthesisResult(BaseModel):
    """Result of Boyd's creative induction — a new paradigm.

    Contains the newly synthesized hypothesis and causal model, along with
    provenance information linking back to the destruction result that
    produced the raw materials.

    Attributes:
        id: Unique identifier for this synthesis.
        hypothesis: The newly created Hypothesis.
        causal_dag: The newly created CausalDAG.
        common_qualities: Common qualities discovered during synthesis.
        source_destruction_id: ID of the DestructionResult that provided fragments.
        constituents_used: IDs of shattered constituents incorporated.
        new_variables_added: Variables added from new observations.
        new_edges_created: Number of new causal edges synthesized.
        synthesis_rationale: Explanation of the synthesis logic.
        timestamp: When the synthesis was performed.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this synthesis.",
    )
    hypothesis: Hypothesis = Field(
        ...,
        description="The newly created hypothesis.",
    )
    causal_dag: CausalDAG = Field(
        ...,
        description="The newly created causal DAG.",
    )
    common_qualities: list[CommonQuality] = Field(
        default_factory=list,
        description="Common qualities discovered during synthesis.",
    )
    source_destruction_id: str = Field(
        ...,
        description="ID of the DestructionResult that provided fragments.",
    )
    constituents_used: list[str] = Field(
        default_factory=list,
        description="IDs of shattered constituents incorporated.",
    )
    new_variables_added: list[str] = Field(
        default_factory=list,
        description="IDs of new variables added from observations.",
    )
    new_edges_created: int = Field(
        default=0,
        ge=0,
        description="Number of new causal edges synthesized.",
    )
    synthesis_rationale: str = Field(
        default="",
        description="Explanation of the synthesis logic.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the synthesis was performed.",
    )


# ---------------------------------------------------------------------------
# Creative inductor
# ---------------------------------------------------------------------------


class CreativeInductor:
    """Boyd's creative induction engine.

    Takes shattered constituents from destructive deduction and new
    observations from merchant agents, finds common qualities among
    the fragments, and synthesizes a new hypothesis and causal DAG.

    The synthesis process:
    1. Collect all preserved causal claims (supported + unsupported)
    2. Find common qualities: shared variables, semantic similarity,
       structural patterns
    3. Incorporate new variables from merchant observations
    4. Generate new causal edges by connecting fragments through
       common qualities and new data
    5. Assemble the new CausalDAG and Hypothesis

    Attributes:
        store: The shared KnowledgeGraphStore.
        similarity_threshold: Minimum embedding similarity for semantic matching.
    """

    def __init__(
        self,
        store: KnowledgeGraphStore,
        similarity_threshold: float = 0.6,
    ) -> None:
        """Initialize the creative inductor.

        Args:
            store: The shared KnowledgeGraphStore.
            similarity_threshold: Minimum cosine similarity between variable
                                 embeddings to consider them semantically related.
        """
        self.store = store
        self.similarity_threshold = similarity_threshold

    def synthesize(
        self,
        destruction_result: DestructionResult,
        new_observations: Optional[list[dict[str, Any]]] = None,
        new_variables: Optional[list[Variable]] = None,
        author: str = "system",
        title: Optional[str] = None,
        description: Optional[str] = None,
        register_in_store: bool = True,
    ) -> SynthesisResult:
        """Perform creative induction to synthesize a new paradigm.

        Takes the output of destructive deduction and optional new data,
        finds common qualities, and constructs a new hypothesis and
        causal model.

        Args:
            destruction_result: The DestructionResult from destructive deduction.
            new_observations: Optional list of new observation dicts from merchants.
                             Each should have at minimum: {'variable_name': str,
                             'variable_type': str, 'data': Any}.
            new_variables: Optional list of new Variable entities to incorporate.
            author: ID of the philosopher-king or system creating this hypothesis.
            title: Title for the new hypothesis. Auto-generated if not provided.
            description: Description for the new hypothesis. Auto-generated if not.
            register_in_store: Whether to register the new hypothesis and DAG
                              in the knowledge graph store.

        Returns:
            A SynthesisResult containing the new hypothesis and DAG.
        """
        new_observations = new_observations or []
        new_variables = new_variables or []

        # Step 1: Collect preserved constituents (supported + unsupported)
        preserved = (
            destruction_result.supported_constituents
            + destruction_result.unsupported_constituents
        )

        # Step 2: Find common qualities among preserved constituents
        common_qualities = self._find_common_qualities(preserved)

        # Step 3: Build the variable set for the new DAG
        # Start with preserved variables, add new ones
        variable_map: dict[str, Variable] = {}
        for var in destruction_result.preserved_variables:
            variable_map[var.id] = var

        # Add new variables from observations
        added_variable_ids: list[str] = []
        for obs in new_observations:
            var = Variable(
                name=obs.get("variable_name", f"new_var_{uuid.uuid4().hex[:8]}"),
                type=VariableType(obs.get("variable_type", "observable")),
                embedding=obs.get("embedding"),
                data_sources=obs.get("data_sources", []),
            )
            variable_map[var.id] = var
            added_variable_ids.append(var.id)

        for var in new_variables:
            if var.id not in variable_map:
                variable_map[var.id] = var
                added_variable_ids.append(var.id)

        # Step 4: Build the edge set for the new DAG
        edges: list[CausalEdge] = []
        constituents_used: list[str] = []

        # Preserve supported edges from the old paradigm
        for constituent in destruction_result.supported_constituents:
            # Ensure both endpoints are in our variable set
            if (
                constituent.edge.source in variable_map
                and constituent.edge.target in variable_map
            ):
                edges.append(constituent.edge)
                constituents_used.append(constituent.id)

        # Also include unsupported edges (they weren't contradicted, just unverified)
        for constituent in destruction_result.unsupported_constituents:
            if (
                constituent.edge.source in variable_map
                and constituent.edge.target in variable_map
            ):
                # Reduce strength to reflect uncertainty
                weakened_edge = CausalEdge(
                    source=constituent.edge.source,
                    target=constituent.edge.target,
                    type=constituent.edge.type,
                    strength=constituent.edge.strength * 0.5,  # Halve the strength
                    evidence=constituent.edge.evidence,
                )
                edges.append(weakened_edge)
                constituents_used.append(constituent.id)

        # Step 5: Generate new edges connecting fragments through common qualities
        new_edge_count = 0
        for quality in common_qualities:
            new_edges = self._edges_from_quality(quality, variable_map, edges)
            edges.extend(new_edges)
            new_edge_count += len(new_edges)

        # Step 6: Generate edges connecting new variables to existing structure
        if added_variable_ids:
            connection_edges = self._connect_new_variables(
                added_variable_ids, variable_map, edges
            )
            edges.extend(connection_edges)
            new_edge_count += len(connection_edges)

        # Step 7: Assemble the new CausalDAG
        new_dag = CausalDAG(
            nodes=list(variable_map.values()),
            edges=edges,
            version=1,
        )

        # Step 8: Create the new Hypothesis
        if title is None:
            old_hypothesis = self.store.get_hypothesis(destruction_result.hypothesis_id)
            old_title = old_hypothesis.title if old_hypothesis else "Unknown"
            title = f"Synthesis from '{old_title}' (post-destruction revision)"

        if description is None:
            description = (
                f"New hypothesis synthesized via Boyd's creative induction. "
                f"Incorporates {len(destruction_result.supported_constituents)} "
                f"supported causal claims from the previous paradigm, "
                f"{len(destruction_result.unsupported_constituents)} unverified claims, "
                f"and {len(added_variable_ids)} new variables from merchant observations. "
                f"{new_edge_count} new causal edges were synthesized from "
                f"{len(common_qualities)} common qualities discovered across fragments."
            )

        new_hypothesis = Hypothesis(
            author=author,
            title=title,
            description=description,
            causal_model_id=new_dag.id,
            variables=list(variable_map.keys()),
            predictions=[],  # Philosopher-kings should add falsification criteria
            status=HypothesisStatus.PROPOSED,
        )

        # Link DAG to hypothesis
        new_dag.hypothesis_id = new_hypothesis.id

        # Step 9: Register in the store if requested
        if register_in_store:
            self.store.add_hypothesis(new_hypothesis)
            self.store.add_causal_dag(new_dag)

        return SynthesisResult(
            hypothesis=new_hypothesis,
            causal_dag=new_dag,
            common_qualities=common_qualities,
            source_destruction_id=destruction_result.id,
            constituents_used=constituents_used,
            new_variables_added=added_variable_ids,
            new_edges_created=new_edge_count,
            synthesis_rationale=(
                f"Creative induction from destruction of hypothesis "
                f"'{destruction_result.hypothesis_id}'. "
                f"Found {len(common_qualities)} common qualities across "
                f"{len(preserved)} preserved constituents. "
                f"The new paradigm preserves what worked, discards what failed, "
                f"and incorporates {len(added_variable_ids)} new variables."
            ),
        )

    # ------------------------------------------------------------------
    # Common quality discovery
    # ------------------------------------------------------------------

    def _find_common_qualities(
        self,
        constituents: list[ShatteredConstituent],
    ) -> list[CommonQuality]:
        """Find common qualities among shattered constituents.

        Identifies three types of commonality:
        1. Variable sharing: constituents that share source or target variables
        2. Semantic similarity: variables with similar embeddings
        3. Structural patterns: shared edge types or strength ranges

        Args:
            constituents: The preserved constituents to analyze.

        Returns:
            List of discovered CommonQuality objects.
        """
        qualities: list[CommonQuality] = []

        if len(constituents) < 2:
            return qualities

        # Type 1: Variable sharing
        qualities.extend(self._find_variable_sharing(constituents))

        # Type 2: Semantic similarity (if embeddings are available)
        qualities.extend(self._find_semantic_similarity(constituents))

        # Type 3: Structural patterns
        qualities.extend(self._find_structural_patterns(constituents))

        return qualities

    def _find_variable_sharing(
        self,
        constituents: list[ShatteredConstituent],
    ) -> list[CommonQuality]:
        """Find constituents that share variables (common nodes in the graph).

        Args:
            constituents: Constituents to analyze.

        Returns:
            Common qualities based on shared variables.
        """
        # Map each variable to the constituents that reference it
        var_to_constituents: dict[str, list[str]] = {}

        for c in constituents:
            for var_id in [c.edge.source, c.edge.target]:
                if var_id not in var_to_constituents:
                    var_to_constituents[var_id] = []
                var_to_constituents[var_id].append(c.id)

        qualities: list[CommonQuality] = []
        for var_id, c_ids in var_to_constituents.items():
            if len(c_ids) >= 2:
                var = self.store.get_variable(var_id)
                var_name = var.name if var else var_id
                qualities.append(
                    CommonQuality(
                        description=(
                            f"Variable '{var_name}' is shared across "
                            f"{len(c_ids)} causal claims, suggesting it is "
                            f"a hub in the causal structure."
                        ),
                        variable_ids=[var_id],
                        constituent_ids=c_ids,
                        similarity_score=min(1.0, len(c_ids) / len(constituents)),
                        quality_type="variable",
                    )
                )

        return qualities

    def _find_semantic_similarity(
        self,
        constituents: list[ShatteredConstituent],
    ) -> list[CommonQuality]:
        """Find constituents with semantically similar variables (via embeddings).

        Uses cosine similarity between variable embeddings to identify
        conceptually related but structurally disconnected fragments.
        This is where the "creative" in creative induction comes from:
        finding connections that the original model didn't see.

        Args:
            constituents: Constituents to analyze.

        Returns:
            Common qualities based on semantic similarity.
        """
        # Collect all variables with embeddings
        var_embeddings: dict[str, tuple[Variable, np.ndarray]] = {}

        for c in constituents:
            for var in [c.source_variable, c.target_variable]:
                if var.embedding is not None and var.id not in var_embeddings:
                    var_embeddings[var.id] = (var, np.array(var.embedding))

        if len(var_embeddings) < 2:
            return []

        qualities: list[CommonQuality] = []
        var_ids = list(var_embeddings.keys())

        for i in range(len(var_ids)):
            for j in range(i + 1, len(var_ids)):
                vid_a, vid_b = var_ids[i], var_ids[j]
                var_a, emb_a = var_embeddings[vid_a]
                var_b, emb_b = var_embeddings[vid_b]

                similarity = self._cosine_similarity(emb_a, emb_b)

                if similarity >= self.similarity_threshold:
                    # Find which constituents use these variables
                    related_constituents = [
                        c.id
                        for c in constituents
                        if c.edge.source in (vid_a, vid_b)
                        or c.edge.target in (vid_a, vid_b)
                    ]

                    qualities.append(
                        CommonQuality(
                            description=(
                                f"Variables '{var_a.name}' and '{var_b.name}' are "
                                f"semantically similar (cosine={similarity:.3f}), "
                                f"suggesting a latent common cause or shared mechanism."
                            ),
                            variable_ids=[vid_a, vid_b],
                            constituent_ids=related_constituents,
                            similarity_score=float(similarity),
                            quality_type="semantic",
                        )
                    )

        return qualities

    def _find_structural_patterns(
        self,
        constituents: list[ShatteredConstituent],
    ) -> list[CommonQuality]:
        """Find structural patterns shared across constituents.

        Looks for shared edge types, similar strength magnitudes, and
        common graph motifs (chains, forks, colliders).

        Args:
            constituents: Constituents to analyze.

        Returns:
            Common qualities based on structural patterns.
        """
        qualities: list[CommonQuality] = []

        # Group by edge type
        by_type: dict[str, list[ShatteredConstituent]] = {}
        for c in constituents:
            edge_type = c.edge.type.value
            if edge_type not in by_type:
                by_type[edge_type] = []
            by_type[edge_type].append(c)

        for edge_type, group in by_type.items():
            if len(group) >= 2:
                # Check if strengths are in a similar range
                strengths = [c.edge.strength for c in group]
                strength_std = float(np.std(strengths)) if len(strengths) > 1 else 0.0

                if strength_std < 0.3:  # Low variance = consistent pattern
                    all_vars = set()
                    for c in group:
                        all_vars.add(c.edge.source)
                        all_vars.add(c.edge.target)

                    qualities.append(
                        CommonQuality(
                            description=(
                                f"Structural pattern: {len(group)} edges of type "
                                f"'{edge_type}' with consistent strength "
                                f"(mean={np.mean(strengths):.3f}, std={strength_std:.3f}). "
                                f"This suggests a systematic {edge_type} mechanism."
                            ),
                            variable_ids=sorted(all_vars),
                            constituent_ids=[c.id for c in group],
                            similarity_score=max(0.3, 1.0 - strength_std),
                            quality_type="structural",
                        )
                    )

        return qualities

    # ------------------------------------------------------------------
    # Edge generation
    # ------------------------------------------------------------------

    def _edges_from_quality(
        self,
        quality: CommonQuality,
        variable_map: dict[str, Variable],
        existing_edges: list[CausalEdge],
    ) -> list[CausalEdge]:
        """Generate new causal edges from a common quality.

        When a common quality links variables that aren't already directly
        connected, this may suggest a new causal relationship worth
        investigating.

        Args:
            quality: The common quality to generate edges from.
            variable_map: Available variables.
            existing_edges: Edges already in the new DAG (to avoid duplicates).

        Returns:
            List of new CausalEdge objects.
        """
        if quality.quality_type != "semantic" or len(quality.variable_ids) < 2:
            return []

        # Only generate edges from semantic similarity — the most creative type
        existing_pairs = {
            (e.source, e.target) for e in existing_edges
        }

        new_edges: list[CausalEdge] = []
        var_ids = quality.variable_ids

        for i in range(len(var_ids)):
            for j in range(i + 1, len(var_ids)):
                vid_a, vid_b = var_ids[i], var_ids[j]

                if vid_a not in variable_map or vid_b not in variable_map:
                    continue

                # Don't duplicate existing edges
                if (vid_a, vid_b) in existing_pairs or (vid_b, vid_a) in existing_pairs:
                    continue

                # Create a weak hypothetical edge — philosopher-kings should review
                new_edges.append(
                    CausalEdge(
                        source=vid_a,
                        target=vid_b,
                        type=EdgeType.CAUSAL,
                        strength=quality.similarity_score * 0.3,  # Weak initial strength
                        evidence=[],  # No evidence yet — needs testing
                    )
                )

        return new_edges

    def _connect_new_variables(
        self,
        new_variable_ids: list[str],
        variable_map: dict[str, Variable],
        existing_edges: list[CausalEdge],
    ) -> list[CausalEdge]:
        """Create hypothetical edges connecting new variables to existing structure.

        Uses embedding similarity to find the most likely connection points
        for new variables in the existing causal structure.

        Args:
            new_variable_ids: IDs of newly added variables.
            variable_map: All available variables.
            existing_edges: Existing edges in the DAG.

        Returns:
            List of new CausalEdge objects connecting new variables.
        """
        existing_var_ids = [
            vid for vid in variable_map if vid not in new_variable_ids
        ]
        existing_pairs = {(e.source, e.target) for e in existing_edges}

        new_edges: list[CausalEdge] = []

        for new_vid in new_variable_ids:
            new_var = variable_map.get(new_vid)
            if new_var is None or new_var.embedding is None:
                continue

            new_emb = np.array(new_var.embedding)
            best_match: Optional[tuple[str, float]] = None

            for existing_vid in existing_var_ids:
                existing_var = variable_map.get(existing_vid)
                if existing_var is None or existing_var.embedding is None:
                    continue

                existing_emb = np.array(existing_var.embedding)
                sim = self._cosine_similarity(new_emb, existing_emb)

                if sim >= self.similarity_threshold:
                    if best_match is None or sim > best_match[1]:
                        best_match = (existing_vid, sim)

            if best_match is not None:
                target_vid, sim = best_match
                if (new_vid, target_vid) not in existing_pairs:
                    new_edges.append(
                        CausalEdge(
                            source=new_vid,
                            target=target_vid,
                            type=EdgeType.CAUSAL,
                            strength=float(sim) * 0.2,  # Very weak — needs testing
                            evidence=[],
                        )
                    )

        return new_edges

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity in range [-1, 1].
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
