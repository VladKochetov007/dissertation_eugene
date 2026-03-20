"""
Era — temporal governance checkpoints for the Republic's knowledge graph.

An era defines the operating parameters for all agents in the Republic:
which data sources are valid, which evaluation criteria apply, what
thresholds define a validated hypothesis, and which causal models are
in the current paradigm.

The design draws from three sources:
1. Kuhn's paradigm theory — eras are paradigms within which normal science
   proceeds, with transitions representing paradigm shifts
2. OpenForage's era system — periodic governance snapshots that synchronize
   all agents to shared parameters, with era transitions governable through
   voting rather than crisis
3. Boyd's OODA loop — era transitions are the Orient phase at the systemic
   level: when the concept-reality mismatch exceeds tolerance, the system
   must reorient

Era lifecycle:
    ACTIVE → TRANSITIONING → ARCHIVED
    New era: PENDING → ACTIVE (when approved)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EraStatus(str, Enum):
    """Status of an era in its lifecycle."""

    PENDING = "pending"  # Proposed but not yet active
    ACTIVE = "active"  # Currently governing all agents
    TRANSITIONING = "transitioning"  # Being replaced — agents syncing to new era
    ARCHIVED = "archived"  # Historical record only


# ---------------------------------------------------------------------------
# Configuration — what an era defines
# ---------------------------------------------------------------------------


class EraConfig(BaseModel):
    """Configuration snapshot for an era.

    This is the 'paradigm' — the shared context within which all agents
    operate. When an era transitions, all agents must sync to the new
    config before resuming operations.

    Modeled after OpenForage's era snapshots, which define:
    - What features/data sources are valid
    - How to calculate evaluation metrics
    - What thresholds are required for acceptance
    - What functions/transformations are available
    """

    # Data source governance
    valid_data_sources: list[str] = Field(
        default_factory=list,
        description="IDs of data sources considered valid in this era.",
    )
    required_pearl_levels: list[str] = Field(
        default_factory=lambda: ["association"],
        description="Pearl hierarchy levels required for evidence (association, intervention, counterfactual).",
    )

    # Hypothesis evaluation criteria
    min_evidence_count: int = Field(
        default=3,
        ge=1,
        description="Minimum evidence items before a hypothesis can be validated.",
    )
    min_supporting_ratio: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum ratio of supporting vs contradicting evidence for validation.",
    )
    min_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum average confidence score across evidence for validation.",
    )
    require_falsification_criteria: bool = Field(
        default=True,
        description="Whether hypotheses must specify falsification criteria (Popperian norm).",
    )

    # Signal ensemble parameters (inspired by OpenForage's ensemble approach)
    min_independent_sources: int = Field(
        default=2,
        ge=1,
        description="Minimum number of independent merchant sources for evidence to count.",
    )
    signal_correlation_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Maximum correlation between signals before they are considered redundant.",
    )

    # Anomaly detection thresholds (Kuhnian crisis detection)
    anomaly_accumulation_threshold: int = Field(
        default=10,
        ge=1,
        description="Number of accumulated anomalies before triggering a paradigm crisis flag.",
    )
    anomaly_window_hours: float = Field(
        default=168.0,
        ge=1.0,
        description="Time window for counting anomalies (default: 1 week).",
    )

    # Custom parameters — extensible for domain-specific criteria
    custom: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific parameters for this era.",
    )


# ---------------------------------------------------------------------------
# Era model
# ---------------------------------------------------------------------------


class Era(BaseModel):
    """An era in the Republic's knowledge graph.

    Represents a temporal governance checkpoint — a Kuhnian paradigm
    within which agents operate under shared evaluation criteria.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique era identifier.",
    )
    number: int = Field(
        ...,
        ge=1,
        description="Sequential era number (monotonically increasing).",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable name for this era.",
    )
    description: str = Field(
        default="",
        description="Description of why this era was created and what changed.",
    )
    status: EraStatus = Field(
        default=EraStatus.PENDING,
        description="Current lifecycle status.",
    )
    config: EraConfig = Field(
        default_factory=EraConfig,
        description="The governing parameters for this era.",
    )

    # Timestamps
    proposed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this era was proposed.",
    )
    activated_at: Optional[datetime] = Field(
        default=None,
        description="When this era became active.",
    )
    archived_at: Optional[datetime] = Field(
        default=None,
        description="When this era was archived.",
    )

    # Governance
    proposed_by: str = Field(
        default="system",
        description="ID of the philosopher-king or governance process that proposed this era.",
    )
    approval_votes: int = Field(
        default=0,
        ge=0,
        description="Number of approval votes received.",
    )
    rejection_votes: int = Field(
        default=0,
        ge=0,
        description="Number of rejection votes received.",
    )

    # Metrics accumulated during the era
    hypotheses_proposed: int = Field(default=0, ge=0)
    hypotheses_validated: int = Field(default=0, ge=0)
    hypotheses_falsified: int = Field(default=0, ge=0)
    evidence_collected: int = Field(default=0, ge=0)
    anomalies_detected: int = Field(default=0, ge=0)
    merchant_contributions: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Era transition record
# ---------------------------------------------------------------------------


class EraTransition(BaseModel):
    """Record of a transition between eras.

    Captures what changed, why, and the state of the knowledge graph
    at the point of transition. This is the Republic's institutional
    memory of paradigm shifts.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique transition identifier.",
    )
    from_era_id: str = Field(
        ...,
        description="ID of the era being replaced.",
    )
    to_era_id: str = Field(
        ...,
        description="ID of the new era.",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Why the transition was triggered.",
    )
    transition_type: str = Field(
        default="governance",
        description="How the transition was triggered: governance (voted), crisis (anomaly threshold), manual.",
    )
    transitioned_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the transition occurred.",
    )

    # Snapshot of what changed
    config_changes: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of config fields that changed between eras.",
    )

    # State at transition point
    active_merchants_count: int = Field(default=0, ge=0)
    active_hypotheses_count: int = Field(default=0, ge=0)
    knowledge_graph_node_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Era Manager — orchestrates era lifecycle
# ---------------------------------------------------------------------------


class EraManager:
    """Manages era lifecycle for the Republic's knowledge graph.

    Handles era creation, activation, transition, and archival.
    Ensures exactly one era is ACTIVE at any time. Provides the
    synchronization mechanism that keeps all agents operating
    within the same paradigm.

    The era manager implements the Kirill Function at the systemic
    level: periodic review of whether the current paradigm is still
    producing useful results, with structured mechanisms for
    transitioning when it is not.
    """

    def __init__(self) -> None:
        self._eras: dict[str, Era] = {}
        self._transitions: list[EraTransition] = []
        self._current_era_id: Optional[str] = None

    @property
    def current_era(self) -> Optional[Era]:
        """Return the currently active era, or None if no era is active."""
        if self._current_era_id is None:
            return None
        return self._eras.get(self._current_era_id)

    @property
    def era_count(self) -> int:
        """Total number of eras (including archived)."""
        return len(self._eras)

    def propose_era(
        self,
        name: str,
        config: EraConfig,
        description: str = "",
        proposed_by: str = "system",
    ) -> Era:
        """Propose a new era.

        Creates an era in PENDING status. It must be activated through
        governance (approve_era) before it takes effect.

        Args:
            name: Human-readable name for the era.
            config: The governing parameters.
            description: Why this era is being proposed.
            proposed_by: ID of the proposer.

        Returns:
            The newly created Era in PENDING status.
        """
        era = Era(
            number=len(self._eras) + 1,
            name=name,
            description=description,
            config=config,
            proposed_by=proposed_by,
        )
        self._eras[era.id] = era
        logger.info(
            "Era proposed: #%d '%s' (id=%s) by %s",
            era.number,
            era.name,
            era.id,
            proposed_by,
        )
        return era

    def activate_era(self, era_id: str, reason: str = "Governance approval") -> EraTransition:
        """Activate a pending era, archiving the current one.

        This is the paradigm shift. All agents must sync to the new
        era's parameters after this call.

        Args:
            era_id: ID of the era to activate.
            reason: Why the transition is happening.

        Returns:
            EraTransition record documenting the change.

        Raises:
            KeyError: If era_id is not found.
            ValueError: If the era is not in PENDING status.
        """
        new_era = self._eras.get(era_id)
        if new_era is None:
            raise KeyError(f"Era '{era_id}' not found.")
        if new_era.status != EraStatus.PENDING:
            raise ValueError(
                f"Era '{era_id}' is in status {new_era.status}, expected PENDING."
            )

        # Archive current era if one exists
        old_era_id = self._current_era_id
        transition_record = None

        if old_era_id is not None:
            old_era = self._eras[old_era_id]
            old_era.status = EraStatus.ARCHIVED
            old_era.archived_at = datetime.now(timezone.utc)

            # Compute config changes
            old_config = old_era.config.model_dump()
            new_config = new_era.config.model_dump()
            changes = {
                k: {"from": old_config[k], "to": new_config[k]}
                for k in new_config
                if old_config.get(k) != new_config[k]
            }

            transition_record = EraTransition(
                from_era_id=old_era_id,
                to_era_id=era_id,
                reason=reason,
                config_changes=changes,
            )
            self._transitions.append(transition_record)

            logger.info(
                "Era transition: #%d '%s' → #%d '%s' (%d config changes). Reason: %s",
                old_era.number,
                old_era.name,
                new_era.number,
                new_era.name,
                len(changes),
                reason,
            )
        else:
            # First era — no transition record needed
            transition_record = EraTransition(
                from_era_id="genesis",
                to_era_id=era_id,
                reason="Initial era activation",
            )
            self._transitions.append(transition_record)

        # Activate new era
        new_era.status = EraStatus.ACTIVE
        new_era.activated_at = datetime.now(timezone.utc)
        self._current_era_id = era_id

        logger.info(
            "Era activated: #%d '%s' (id=%s)",
            new_era.number,
            new_era.name,
            new_era.id,
        )
        return transition_record

    def get_era(self, era_id: str) -> Optional[Era]:
        """Retrieve an era by ID."""
        return self._eras.get(era_id)

    def list_eras(self, status: Optional[EraStatus] = None) -> list[Era]:
        """List all eras, optionally filtered by status."""
        eras = list(self._eras.values())
        if status is not None:
            eras = [e for e in eras if e.status == status]
        return sorted(eras, key=lambda e: e.number)

    def get_transitions(self) -> list[EraTransition]:
        """Return the full history of era transitions.

        This is the Republic's institutional memory of paradigm shifts —
        when they happened, why, and what changed.
        """
        return list(self._transitions)

    def check_crisis_threshold(self) -> bool:
        """Check if the current era's anomaly count exceeds the crisis threshold.

        This is Kuhnian crisis detection: when accumulated anomalies
        pass the threshold, it signals that the current paradigm may
        need revision.

        Returns:
            True if anomalies exceed threshold, suggesting era transition.
        """
        era = self.current_era
        if era is None:
            return False
        return era.anomalies_detected >= era.config.anomaly_accumulation_threshold

    def record_anomaly(self) -> None:
        """Record an anomaly in the current era.

        Called by warrior agents when they detect a hypothesis prediction
        that diverges from observed data. When anomalies accumulate past
        the threshold, it signals a potential paradigm crisis.
        """
        era = self.current_era
        if era is None:
            logger.warning("Cannot record anomaly — no active era.")
            return
        era.anomalies_detected += 1
        if self.check_crisis_threshold():
            logger.warning(
                "PARADIGM CRISIS: Era #%d '%s' has %d anomalies (threshold: %d). "
                "Consider proposing a new era.",
                era.number,
                era.name,
                era.anomalies_detected,
                era.config.anomaly_accumulation_threshold,
            )

    def record_contribution(self) -> None:
        """Record a merchant contribution in the current era."""
        era = self.current_era
        if era is not None:
            era.merchant_contributions += 1

    def record_hypothesis_proposed(self) -> None:
        """Record a hypothesis proposal in the current era."""
        era = self.current_era
        if era is not None:
            era.hypotheses_proposed += 1

    def record_hypothesis_validated(self) -> None:
        """Record a hypothesis validation in the current era."""
        era = self.current_era
        if era is not None:
            era.hypotheses_validated += 1

    def record_hypothesis_falsified(self) -> None:
        """Record a hypothesis falsification in the current era."""
        era = self.current_era
        if era is not None:
            era.hypotheses_falsified += 1

    def record_evidence(self) -> None:
        """Record an evidence item collected in the current era."""
        era = self.current_era
        if era is not None:
            era.evidence_collected += 1
