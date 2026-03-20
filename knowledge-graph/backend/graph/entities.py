"""
Entity models for the Knowledge Graph.

Defines the core data structures for the Republic of AI Agents architecture:
- Hypotheses: philosopher-king generated conjectures with Popperian falsification criteria
- Variables: observable, latent, and interventional nodes in causal models
- CausalEdge / CausalDAG: Pearl's causal graph structures
- DataSource: merchant agent data provenance
- Evidence: supporting or contradicting data linked to hypotheses
- Experiment: warrior agent testing pipelines

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


class HypothesisStatus(str, Enum):
    """Lifecycle status of a hypothesis in the knowledge graph.

    Mirrors the Kuhnian paradigm lifecycle:
    - proposed: initial conjecture by a philosopher-king
    - testing: warrior agents actively gathering evidence
    - validated: sufficient supporting evidence accumulated
    - falsified: Popperian falsification criteria met
    - paradigm: hypothesis promoted to foundational knowledge
    """

    PROPOSED = "proposed"
    TESTING = "testing"
    VALIDATED = "validated"
    FALSIFIED = "falsified"
    PARADIGM = "paradigm"


class VariableType(str, Enum):
    """Type of variable in a causal DAG.

    Maps onto Pearl's causal hierarchy:
    - observable: Level 1 — association, passively measurable
    - latent: unobserved confounders or mediators
    - intervention: Level 2 — variables subject to do-calculus interventions
    """

    OBSERVABLE = "observable"
    LATENT = "latent"
    INTERVENTION = "intervention"


class EdgeType(str, Enum):
    """Type of directed edge in a causal DAG."""

    CAUSAL = "causal"
    CONFOUNDING = "confounding"
    MEDIATING = "mediating"


class EvidenceType(str, Enum):
    """Whether a piece of evidence supports or contradicts a hypothesis."""

    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"


class ExperimentType(str, Enum):
    """Kind of experiment run by warrior agents."""

    AB_TEST = "ab_test"
    OBSERVATIONAL = "observational"
    INTERVENTION = "intervention"


class ExperimentStatus(str, Enum):
    """Lifecycle of an experiment."""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DataSourceType(str, Enum):
    """Category of data source managed by merchant agents."""

    API = "api"
    DATASET = "dataset"
    WEB_SCRAPER = "web_scraper"
    SENSOR = "sensor"
    MARKET_FEED = "market_feed"
    NEWS = "news"


# ---------------------------------------------------------------------------
# Helper sub-models
# ---------------------------------------------------------------------------


class Prediction(BaseModel):
    """A single falsifiable prediction attached to a hypothesis.

    Follows Popper's requirement: every genuine hypothesis must specify
    what observations would disprove it.
    """

    if_condition: str = Field(
        ...,
        alias="if",
        description="The intervention or condition being tested.",
    )
    then_outcome: str = Field(
        ...,
        alias="then",
        description="The expected outcome if the hypothesis is correct.",
    )
    falsification_criteria: str = Field(
        ...,
        description="What observation would definitively disprove this prediction.",
    )

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------


class Variable(BaseModel):
    """A node in a causal DAG representing a measurable or latent quantity.

    In the Republic architecture, variables are the atomic units of knowledge
    that merchant agents collect data about and warrior agents test.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the variable.",
    )
    name: str = Field(..., description="Human-readable variable name.")
    type: VariableType = Field(
        ..., description="Whether this variable is observable, latent, or interventional."
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Semantic embedding vector computed by sentence-transformers.",
    )
    data_sources: list[str] = Field(
        default_factory=list,
        description="IDs of merchant agents / data sources providing data for this variable.",
    )

    model_config = {"json_schema_extra": {"examples": [{"name": "market_price", "type": "observable"}]}}


class CausalEdge(BaseModel):
    """A directed edge in a causal DAG.

    Represents a causal, confounding, or mediating relationship between
    two variables, with an associated strength estimate and evidence trail.
    """

    source: str = Field(..., description="ID of the source variable.")
    target: str = Field(..., description="ID of the target variable.")
    type: EdgeType = Field(..., description="The nature of the causal relationship.")
    strength: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Estimated causal strength in [-1, 1].",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="IDs of evidence supporting this edge.",
    )


class CausalDAG(BaseModel):
    """A directed acyclic graph encoding causal relationships.

    This is the core data structure of Pearl's causal inference framework,
    serving as the shared language between philosopher-kings (who design them),
    merchants (who supply data for them), and warriors (who test them).
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this DAG version.",
    )
    nodes: list[Variable] = Field(
        default_factory=list,
        description="Variables (nodes) in the causal model.",
    )
    edges: list[CausalEdge] = Field(
        default_factory=list,
        description="Directed edges representing causal relationships.",
    )
    hypothesis_id: Optional[str] = Field(
        default=None,
        description="ID of the hypothesis this DAG was constructed to test.",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Version number, incremented on each structural update.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of DAG creation.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of last modification.",
    )


class Hypothesis(BaseModel):
    """A falsifiable conjecture registered by a philosopher-king.

    Central to the Republic architecture: philosopher-kings generate hypotheses,
    merchant agents gather data to test them, and warrior agents run experiments.
    Every hypothesis must include Popperian falsification criteria.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier.",
    )
    author: str = Field(
        ..., description="ID of the philosopher-king who proposed this hypothesis."
    )
    title: str = Field(..., min_length=1, max_length=500, description="Short title.")
    description: str = Field(
        ..., description="Detailed description of the hypothesis and its rationale."
    )
    causal_model_id: Optional[str] = Field(
        default=None,
        description="ID of the CausalDAG encoding this hypothesis's causal claims.",
    )
    variables: list[str] = Field(
        default_factory=list,
        description="IDs of variables referenced by this hypothesis.",
    )
    predictions: list[Prediction] = Field(
        default_factory=list,
        description="Falsifiable predictions derived from this hypothesis.",
    )
    status: HypothesisStatus = Field(
        default=HypothesisStatus.PROPOSED,
        description="Current lifecycle status.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="IDs of evidence items linked to this hypothesis.",
    )
    stake: float = Field(
        default=0.0,
        ge=0.0,
        description="Economic stake (skin in the game) backing this hypothesis.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of hypothesis creation.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of last modification.",
    )


class DataSource(BaseModel):
    """A data source managed by a merchant agent.

    Represents the provenance of data flowing into the knowledge graph,
    whether from online merchants (APIs, datasets) or offline merchants
    (sensors, physical measurements).
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier.",
    )
    name: str = Field(..., description="Human-readable data source name.")
    type: DataSourceType = Field(..., description="Category of data source.")
    url: Optional[str] = Field(
        default=None, description="URL or endpoint for the data source."
    )
    merchant_id: Optional[str] = Field(
        default=None,
        description="ID of the merchant agent responsible for this source.",
    )
    schema_info: Optional[dict] = Field(
        default=None,
        description="JSON schema or description of the data format.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of registration.",
    )


class Evidence(BaseModel):
    """A piece of evidence linked to a hypothesis.

    Evidence is gathered by merchant agents and evaluated by warrior agents.
    Each item either supports or contradicts a hypothesis, with an associated
    confidence score.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier.",
    )
    hypothesis_id: str = Field(
        ..., description="ID of the hypothesis this evidence relates to."
    )
    type: EvidenceType = Field(
        ..., description="Whether this evidence supports or contradicts the hypothesis."
    )
    description: str = Field(
        ..., description="Human-readable description of the evidence."
    )
    data_source_id: Optional[str] = Field(
        default=None,
        description="ID of the data source that produced this evidence.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this evidence, from 0 (no confidence) to 1 (certain).",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this evidence was recorded.",
    )


class Experiment(BaseModel):
    """An experiment run by warrior agents to test a hypothesis.

    Experiments connect the theoretical (hypotheses, causal models) to the
    empirical (data, outcomes). They represent the Popperian falsification
    process in action.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier.",
    )
    hypothesis_id: str = Field(
        ..., description="ID of the hypothesis being tested."
    )
    type: ExperimentType = Field(
        ..., description="Kind of experiment (A/B test, observational, intervention)."
    )
    status: ExperimentStatus = Field(
        default=ExperimentStatus.PLANNED,
        description="Current lifecycle status of the experiment.",
    )
    description: Optional[str] = Field(
        default=None, description="Human-readable description of the experiment design."
    )
    results: Optional[dict] = Field(
        default=None,
        description="Structured results once experiment completes.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of experiment creation.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of last status change.",
    )
