"""
Knowledge foundation entity models for the Knowledge Graph.

Defines the intellectual foundation layer — thinkers, concepts, traditions,
works, historical periods, and domains spanning 3000 years of human thought.
These entities live alongside the operational layer (Hypothesis, Variable,
CausalDAG, Evidence, Experiment) in the same graph, enabling cross-layer
connections between intellectual heritage and live hypothesis testing.

All models use Pydantic v2 for validation, serialization, and schema generation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Era(str, Enum):
    """Broad historical era classification."""

    ANCIENT = "ancient"  # before 500 CE
    MEDIEVAL = "medieval"  # 500-1500
    EARLY_MODERN = "early_modern"  # 1500-1800
    MODERN = "modern"  # 1800-1945
    CONTEMPORARY = "contemporary"  # 1945-present


class ConceptRelationType(str, Enum):
    """Type of relationship between two concepts.

    Captures the dialectical relationships central to the manuscript's
    Hegel-Popper-Kuhn-Pearl synthesis.
    """

    EXTENDS = "extends"
    CONTRADICTS = "contradicts"
    SYNTHESIZES = "synthesizes"
    PRECEDES = "precedes"
    FORMALIZES = "formalizes"
    APPLIES = "applies"
    ANALOGOUS_TO = "analogous_to"


class WorkType(str, Enum):
    """Type of intellectual work."""

    BOOK = "book"
    PAPER = "paper"
    SCRIPTURE = "scripture"
    DIALOGUE = "dialogue"
    ESSAY = "essay"
    TREATISE = "treatise"
    POEM = "poem"
    LECTURE = "lecture"


class ThinkerRelationType(str, Enum):
    """Type of relationship between thinkers."""

    INFLUENCED = "influenced"
    STUDENT_OF = "student_of"
    CONTEMPORARY_OF = "contemporary_of"
    OPPOSED = "opposed"
    COLLABORATED = "collaborated"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ManuscriptReference(BaseModel):
    """A reference to a specific location in the theology manuscript.

    Links knowledge entities back to the chapters where they appear,
    enabling the manuscript-site's interactive graph visualization.
    """

    chapter: int = Field(..., ge=0, le=41, description="Chapter number (0-41).")
    part: Optional[str] = Field(
        default=None,
        description="Part name, e.g. 'part2-epistemology'.",
    )
    section: Optional[str] = Field(
        default=None,
        description="Section heading within chapter.",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Relevance note for this reference.",
    )


class ConceptRelation(BaseModel):
    """A typed relationship from one concept to another.

    Captures the dialectical structure central to the manuscript:
    concepts extend, contradict, synthesize, or formalize each other.
    """

    target_id: str = Field(..., description="ID of the related concept.")
    relation_type: ConceptRelationType = Field(
        ..., description="Nature of the relationship."
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Brief explanation of the relationship.",
    )


class ThinkerRelation(BaseModel):
    """A typed relationship from one thinker to another."""

    target_id: str = Field(..., description="ID of the related thinker.")
    relation_type: ThinkerRelationType = Field(
        ..., description="Nature of the relationship."
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Brief explanation of the relationship.",
    )


# ---------------------------------------------------------------------------
# Core knowledge entities
# ---------------------------------------------------------------------------


class Thinker(BaseModel):
    """A historical or contemporary intellectual figure.

    Represents philosophers, theologians, scientists, mystics, and other
    thinkers whose ideas form the manuscript's intellectual foundation.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier (use deterministic slugs for seed data).",
    )
    name: str = Field(..., min_length=1, max_length=200)
    birth_year: Optional[int] = Field(
        default=None, description="Birth year (negative for BCE)."
    )
    death_year: Optional[int] = Field(
        default=None, description="Death year (None if alive)."
    )
    era: Era = Field(..., description="Broad historical era.")
    traditions: list[str] = Field(
        default_factory=list,
        description="IDs of Tradition entities this thinker belongs to.",
    )
    domains: list[str] = Field(
        default_factory=list,
        description="IDs of Domain entities this thinker works in.",
    )
    key_concepts: list[str] = Field(
        default_factory=list,
        description="IDs of Concept entities this thinker developed.",
    )
    works: list[str] = Field(
        default_factory=list,
        description="IDs of Work entities authored by this thinker.",
    )
    bio: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Brief biographical significance statement.",
    )
    manuscript_refs: list[ManuscriptReference] = Field(
        default_factory=list,
        description="References to manuscript chapters where this thinker appears.",
    )
    related_thinkers: list[ThinkerRelation] = Field(
        default_factory=list,
        description="Typed relationships to other thinkers.",
    )
    taxonomy_role: Optional[str] = Field(
        default=None,
        description="Role in normie/psycho/schizo taxonomy if applicable.",
    )
    tier: Optional[int] = Field(
        default=None,
        ge=1,
        le=3,
        description="Thinker tier from CLAUDE.md (1=foundational, 2=important, 3=supporting).",
    )
    embedding: Optional[list[float]] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Concept(BaseModel):
    """An intellectual idea, framework, or principle.

    Concepts are the atomic intellectual units of the knowledge graph.
    Examples: falsifiability, strange loop, do-calculus, samsaric cycle.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier.",
    )
    name: str = Field(..., min_length=1, max_length=300)
    description: str = Field(..., max_length=2000)
    domain_ids: list[str] = Field(
        default_factory=list,
        description="IDs of Domain entities this concept belongs to.",
    )
    originator_id: Optional[str] = Field(
        default=None,
        description="ID of the Thinker who originated this concept.",
    )
    developer_ids: list[str] = Field(
        default_factory=list,
        description="IDs of Thinkers who further developed this concept.",
    )
    tradition_ids: list[str] = Field(
        default_factory=list,
        description="IDs of Traditions this concept is part of.",
    )
    related_concepts: list[ConceptRelation] = Field(
        default_factory=list,
        description="Typed relationships to other concepts.",
    )
    manuscript_refs: list[ManuscriptReference] = Field(default_factory=list)
    claim_status: Optional[str] = Field(
        default=None,
        description="From Critical Interlude: analogy, pattern_observation, structural_analogy, structural_isomorphism, phenomenological_parallel.",
    )
    pearl_level: Optional[str] = Field(
        default=None,
        description="Pearl's causal hierarchy level: association, intervention, counterfactual.",
    )
    tags: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Tradition(BaseModel):
    """A school of thought, religion, or intellectual movement.

    Examples: Syriac Christianity, German Idealism, Theravada Buddhism,
    Rationalist Community (LessWrong), Vienna Circle.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier.",
    )
    name: str = Field(..., min_length=1, max_length=300)
    description: Optional[str] = Field(default=None, max_length=2000)
    time_period: Optional[str] = Field(
        default=None,
        description="Approximate period, e.g. '800-200 BCE' or '1781-present'.",
    )
    geographic_origin: Optional[str] = Field(default=None, max_length=200)
    parent_tradition_ids: list[str] = Field(
        default_factory=list,
        description="IDs of parent traditions this emerged from.",
    )
    sub_tradition_ids: list[str] = Field(
        default_factory=list,
        description="IDs of sub-traditions that branched from this.",
    )
    key_thinker_ids: list[str] = Field(default_factory=list)
    core_concept_ids: list[str] = Field(default_factory=list)
    key_work_ids: list[str] = Field(default_factory=list)
    manuscript_refs: list[ManuscriptReference] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Work(BaseModel):
    """A book, paper, text, scripture, or other intellectual product.

    Captures the specific texts referenced in the manuscript and
    Appendix B reading list.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier.",
    )
    title: str = Field(..., min_length=1, max_length=500)
    author_ids: list[str] = Field(
        default_factory=list,
        description="IDs of Thinker entities who authored this.",
    )
    year: Optional[int] = Field(
        default=None, description="Publication year (negative for BCE)."
    )
    work_type: WorkType = Field(default=WorkType.BOOK)
    tradition_ids: list[str] = Field(default_factory=list)
    concepts_introduced: list[str] = Field(
        default_factory=list,
        description="IDs of Concepts first introduced in this work.",
    )
    concepts_developed: list[str] = Field(
        default_factory=list,
        description="IDs of Concepts further developed in this work.",
    )
    references_work_ids: list[str] = Field(
        default_factory=list,
        description="IDs of other Works this work references or responds to.",
    )
    manuscript_refs: list[ManuscriptReference] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HistoricalPeriod(BaseModel):
    """An era or epoch in intellectual and spiritual history.

    Maps onto the manuscript's periodization: Axial Age, Enlightenment,
    Kuhnian paradigm periods, samsaric cycle phases.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier.",
    )
    name: str = Field(..., min_length=1, max_length=300)
    start_year: Optional[int] = Field(
        default=None, description="Approximate start (negative for BCE)."
    )
    end_year: Optional[int] = Field(
        default=None, description="Approximate end (None if ongoing)."
    )
    description: Optional[str] = Field(default=None, max_length=2000)
    dominant_paradigm: Optional[str] = Field(
        default=None,
        description="The Kuhnian paradigm dominant during this period.",
    )
    key_events: list[str] = Field(default_factory=list)
    thinker_ids: list[str] = Field(default_factory=list)
    tradition_ids: list[str] = Field(default_factory=list)
    concept_ids: list[str] = Field(default_factory=list)
    manuscript_refs: list[ManuscriptReference] = Field(default_factory=list)
    cycle_phase: Optional[str] = Field(
        default=None,
        description="Samsaric cycle phase: antichrist, prophetic, crucifixion, resurrection, pentecost, samsaric_turn.",
    )
    embedding: Optional[list[float]] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Domain(BaseModel):
    """A field of knowledge or intellectual discipline.

    Examples: philosophy, theology, mathematics, psychology, complexity science,
    causal inference, phenomenology.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier.",
    )
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    parent_domain_id: Optional[str] = Field(
        default=None,
        description="ID of the parent domain.",
    )
    sub_domain_ids: list[str] = Field(default_factory=list)
    key_thinker_ids: list[str] = Field(default_factory=list)
    key_concept_ids: list[str] = Field(default_factory=list)
    manuscript_refs: list[ManuscriptReference] = Field(default_factory=list)
    embedding: Optional[list[float]] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
