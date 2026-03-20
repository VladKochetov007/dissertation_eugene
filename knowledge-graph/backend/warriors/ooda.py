"""
Boyd's OODA Loop — the decision cycle engine for warrior agents.

Implements John Boyd's Observe-Orient-Decide-Act loop as the fundamental
operating cycle for warrior agents in the Republic of AI Agents architecture.

Boyd's key insight: the entity that cycles through OODA faster than its
adversary gains a compounding advantage. In our context, faster OODA cycles
mean faster hypothesis testing, faster anomaly detection, and faster
adaptation of the knowledge graph to new evidence.

The OODA loop maps onto the Republic's epistemological stack:
- Observe: merchant agents supply data (Pearl's Level 1 — association)
- Orient: apply existing causal models to interpret data (Pearl's Level 2 — intervention)
- Decide: choose which hypothesis to test or update (Pearl's Level 3 — counterfactual)
- Act: execute tests, deploy changes, generate evidence

Each phase produces a typed data structure that feeds the next phase,
creating an auditable chain from raw observation to executed action.

References:
    Boyd, J. (1976). Destruction and Creation.
    Boyd, J. (1986). Patterns of Conflict.
    Osinga, F. (2007). Science, Strategy and War: The Strategic Theory of John Boyd.

Usage:
    loop = OODALoop(agent_id="warrior-001")
    loop.begin_cycle()
    observation = Observation(...)
    loop.record_observe(observation)
    orientation = Orientation(...)
    loop.record_orient(orientation)
    decision = Decision(...)
    loop.record_decide(decision)
    action = Action(...)
    loop.complete_cycle(action)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OODAPhase(str, Enum):
    """Current phase of the OODA loop.

    The loop progresses linearly through these phases within each cycle,
    but the overall process is continuous — completion of Act immediately
    begins the next Observe phase.
    """

    IDLE = "idle"
    OBSERVE = "observe"
    ORIENT = "orient"
    DECIDE = "decide"
    ACT = "act"


class DecisionType(str, Enum):
    """Type of decision a warrior agent can make during the Decide phase.

    Maps onto the warrior's available actions in the Republic:
    - test_hypothesis: run a statistical test on a hypothesis
    - update_model: modify a causal DAG based on new evidence
    - flag_anomaly: report an anomaly to philosopher-kings
    - deploy_hypothesis: promote a validated hypothesis to production
    - trigger_destruction: initiate Boyd's destructive deduction on a failing model
    - trigger_creation: initiate Boyd's creative induction to synthesize new model
    - no_action: observation doesn't warrant action (continue observing)
    """

    TEST_HYPOTHESIS = "test_hypothesis"
    UPDATE_MODEL = "update_model"
    FLAG_ANOMALY = "flag_anomaly"
    DEPLOY_HYPOTHESIS = "deploy_hypothesis"
    TRIGGER_DESTRUCTION = "trigger_destruction"
    TRIGGER_CREATION = "trigger_creation"
    NO_ACTION = "no_action"


# ---------------------------------------------------------------------------
# Phase data models
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """Data collected during the Observe phase.

    Observations are raw data from merchant agents or direct measurement.
    They are the inputs to the Orient phase, where causal models are
    applied to interpret them.

    The era_id field links every observation to the governance context
    in which it was collected — inspired by OpenForage's era system where
    all data is timestamped within an era's evaluation parameters.

    Attributes:
        id: Unique identifier for this observation.
        source_id: ID of the merchant agent or data source.
        era_id: ID of the era in which this observation was collected.
        data: The raw observation payload (flexible dict for heterogeneous data).
        variables_observed: IDs of knowledge graph variables this observation pertains to.
        timestamp: When the observation was recorded.
        metadata: Additional context about the observation (e.g., confidence, method).
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this observation.",
    )
    source_id: str = Field(
        ...,
        description="ID of the merchant agent or data source that produced this observation.",
    )
    era_id: Optional[str] = Field(
        default=None,
        description="ID of the era in which this observation was collected.",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw observation payload — flexible structure for heterogeneous data.",
    )
    variables_observed: list[str] = Field(
        default_factory=list,
        description="IDs of knowledge graph variables this observation relates to.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this observation was recorded.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context: confidence, collection method, etc.",
    )


class Orientation(BaseModel):
    """Interpreted observation from the Orient phase.

    Orientation is where the warrior applies existing causal models,
    cultural traditions (prior knowledge), and genetic heritage (built-in
    heuristics) to make sense of raw observations — Boyd's most critical
    phase. This is where the warrior's worldview shapes perception.

    In Pearl's terms, Orientation moves from Level 1 (what do I see?) to
    Level 2 (what would happen if I intervened?) by applying the causal
    DAG to the observation.

    Attributes:
        id: Unique identifier for this orientation.
        observation_id: ID of the observation being interpreted.
        causal_model_id: ID of the CausalDAG used for interpretation.
        hypothesis_ids: IDs of hypotheses relevant to this interpretation.
        interpretation: Structured interpretation of the observation.
        anomalies_detected: List of detected deviations from model predictions.
        confidence: Overall confidence in this interpretation (0.0 to 1.0).
        timestamp: When this orientation was produced.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this orientation.",
    )
    observation_id: str = Field(
        ...,
        description="ID of the observation being interpreted.",
    )
    causal_model_id: Optional[str] = Field(
        default=None,
        description="ID of the CausalDAG used to interpret this observation.",
    )
    hypothesis_ids: list[str] = Field(
        default_factory=list,
        description="IDs of hypotheses relevant to this interpretation.",
    )
    interpretation: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured interpretation: predicted vs. observed values, residuals, etc.",
    )
    anomalies_detected: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Deviations from model predictions. Each entry: "
            "{'variable_id': str, 'expected': Any, 'observed': Any, 'severity': float}."
        ),
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this interpretation, 0.0 (no confidence) to 1.0 (certain).",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this orientation was produced.",
    )


class Decision(BaseModel):
    """Chosen action from the Decide phase.

    The Decision encodes what the warrior agent has determined is the
    best course of action given the orientation. It specifies the type
    of action, the target entities, and the rationale.

    Attributes:
        id: Unique identifier for this decision.
        orientation_id: ID of the orientation that informed this decision.
        decision_type: The category of action chosen.
        target_ids: IDs of the entities this decision acts upon (hypotheses, DAGs, etc.).
        rationale: Human-readable explanation of why this action was chosen.
        parameters: Action-specific parameters (e.g., significance level, test type).
        priority: Priority level from 0.0 (low) to 1.0 (critical).
        timestamp: When this decision was made.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this decision.",
    )
    orientation_id: str = Field(
        ...,
        description="ID of the orientation that informed this decision.",
    )
    decision_type: DecisionType = Field(
        ...,
        description="The category of action chosen.",
    )
    target_ids: list[str] = Field(
        default_factory=list,
        description="IDs of knowledge graph entities this decision acts upon.",
    )
    rationale: str = Field(
        default="",
        description="Human-readable explanation for this decision.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters (significance level, test type, etc.).",
    )
    priority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Priority level: 0.0 = low, 1.0 = critical.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this decision was made.",
    )


class Action(BaseModel):
    """Execution result from the Act phase.

    An Action records what happened when the warrior executed a decision.
    It captures success/failure, any evidence produced, and side effects
    on the knowledge graph (hypothesis status changes, new evidence, etc.).

    Attributes:
        id: Unique identifier for this action.
        decision_id: ID of the decision that was executed.
        success: Whether the action completed successfully.
        evidence_ids: IDs of Evidence entities generated by this action.
        results: Structured results of the action execution.
        error: Error message if the action failed.
        side_effects: Changes made to the knowledge graph as a result.
        timestamp: When this action was executed.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this action.",
    )
    decision_id: str = Field(
        ...,
        description="ID of the decision that was executed.",
    )
    success: bool = Field(
        default=True,
        description="Whether the action completed successfully.",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="IDs of Evidence entities generated by this action.",
    )
    results: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured results of the action execution.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the action failed.",
    )
    side_effects: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Changes made to the knowledge graph. Each entry: "
            "{'entity_type': str, 'entity_id': str, 'operation': str, 'details': dict}."
        ),
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this action was executed.",
    )


# ---------------------------------------------------------------------------
# Cycle record
# ---------------------------------------------------------------------------


class OODACycle(BaseModel):
    """A complete record of one OODA loop iteration.

    Captures all four phases and their timing for performance analysis.
    Boyd's central thesis: faster cycles compound into decisive advantage.
    Tracking cycle duration is essential for optimizing warrior agent tempo.

    Attributes:
        id: Unique identifier for this cycle.
        agent_id: ID of the warrior agent that ran this cycle.
        cycle_number: Sequential cycle counter for this agent.
        observation: The Observe phase output.
        orientation: The Orient phase output.
        decision: The Decide phase output.
        action: The Act phase output.
        started_at: When the cycle began.
        completed_at: When the cycle finished (None if still running).
        duration_ms: Total cycle duration in milliseconds (None if incomplete).
        phase_durations_ms: Per-phase durations for fine-grained optimization.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this cycle.",
    )
    agent_id: str = Field(
        ...,
        description="ID of the warrior agent that ran this cycle.",
    )
    cycle_number: int = Field(
        default=0,
        ge=0,
        description="Sequential cycle counter for this agent.",
    )
    observation: Optional[Observation] = Field(
        default=None,
        description="The Observe phase output.",
    )
    orientation: Optional[Orientation] = Field(
        default=None,
        description="The Orient phase output.",
    )
    decision: Optional[Decision] = Field(
        default=None,
        description="The Decide phase output.",
    )
    action: Optional[Action] = Field(
        default=None,
        description="The Act phase output.",
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="When the cycle began.",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When the cycle finished.",
    )
    duration_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total cycle duration in milliseconds.",
    )
    phase_durations_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Per-phase durations: {'observe': ms, 'orient': ms, 'decide': ms, 'act': ms}.",
    )


# ---------------------------------------------------------------------------
# OODA Loop manager
# ---------------------------------------------------------------------------


class OODALoop:
    """Manages the Observe-Orient-Decide-Act cycle for a warrior agent.

    Handles phase transitions, timing, cycle history, and tempo tracking.
    Boyd's competitive advantage comes from cycling faster than the opponent:
    the loop manager tracks speed metrics to enable optimization.

    Attributes:
        agent_id: ID of the warrior agent this loop belongs to.
        current_phase: The phase currently being executed.
        current_cycle: The in-progress cycle record.
        history: Completed cycle records.
        cycle_count: Total cycles completed.
        max_phase_timeout_ms: Maximum time allowed for any single phase.
        target_cycle_ms: Target cycle duration (for tempo optimization).
    """

    def __init__(
        self,
        agent_id: str,
        max_phase_timeout_ms: float = 30_000.0,
        target_cycle_ms: float = 10_000.0,
    ) -> None:
        """Initialize an OODA loop for a warrior agent.

        Args:
            agent_id: ID of the warrior agent this loop belongs to.
            max_phase_timeout_ms: Maximum milliseconds allowed per phase before timeout.
            target_cycle_ms: Target cycle duration in milliseconds for tempo tracking.
        """
        self.agent_id = agent_id
        self.current_phase: OODAPhase = OODAPhase.IDLE
        self.current_cycle: Optional[OODACycle] = None
        self.history: list[OODACycle] = []
        self.cycle_count: int = 0
        self.max_phase_timeout_ms = max_phase_timeout_ms
        self.target_cycle_ms = target_cycle_ms

        # Internal timing state
        self._phase_start: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Cycle lifecycle
    # ------------------------------------------------------------------

    def begin_cycle(self) -> OODACycle:
        """Start a new OODA cycle.

        Creates a fresh cycle record and transitions to the Observe phase.
        If a previous cycle is in progress (incomplete), it is abandoned
        and recorded as incomplete in history.

        Returns:
            The newly created OODACycle.
        """
        # Abandon incomplete cycle if present
        if self.current_cycle is not None and self.current_cycle.completed_at is None:
            self.history.append(self.current_cycle)

        self.cycle_count += 1
        self.current_cycle = OODACycle(
            agent_id=self.agent_id,
            cycle_number=self.cycle_count,
            started_at=datetime.now(timezone.utc),
        )
        self.current_phase = OODAPhase.OBSERVE
        self._phase_start = datetime.now(timezone.utc)

        return self.current_cycle

    def record_observe(self, observation: Observation) -> None:
        """Record the output of the Observe phase and advance to Orient.

        Args:
            observation: The observation data collected.

        Raises:
            RuntimeError: If not in the Observe phase.
        """
        self._require_phase(OODAPhase.OBSERVE)
        self._record_phase_duration("observe")

        self.current_cycle.observation = observation
        self.current_phase = OODAPhase.ORIENT
        self._phase_start = datetime.now(timezone.utc)

    def record_orient(self, orientation: Orientation) -> None:
        """Record the output of the Orient phase and advance to Decide.

        Args:
            orientation: The interpreted observation.

        Raises:
            RuntimeError: If not in the Orient phase.
        """
        self._require_phase(OODAPhase.ORIENT)
        self._record_phase_duration("orient")

        self.current_cycle.orientation = orientation
        self.current_phase = OODAPhase.DECIDE
        self._phase_start = datetime.now(timezone.utc)

    def record_decide(self, decision: Decision) -> None:
        """Record the output of the Decide phase and advance to Act.

        Args:
            decision: The chosen action.

        Raises:
            RuntimeError: If not in the Decide phase.
        """
        self._require_phase(OODAPhase.DECIDE)
        self._record_phase_duration("decide")

        self.current_cycle.decision = decision
        self.current_phase = OODAPhase.ACT
        self._phase_start = datetime.now(timezone.utc)

    def complete_cycle(self, action: Action) -> OODACycle:
        """Record the Act phase output and complete the cycle.

        Computes total cycle duration, archives the cycle, and resets
        to IDLE state. The completed cycle is returned for inspection.

        Args:
            action: The execution result.

        Returns:
            The completed OODACycle with all phases and timing.

        Raises:
            RuntimeError: If not in the Act phase.
        """
        self._require_phase(OODAPhase.ACT)
        self._record_phase_duration("act")

        now = datetime.now(timezone.utc)
        self.current_cycle.action = action
        self.current_cycle.completed_at = now

        # Compute total duration
        if self.current_cycle.started_at:
            delta = now - self.current_cycle.started_at
            self.current_cycle.duration_ms = delta.total_seconds() * 1000.0

        completed = self.current_cycle
        self.history.append(completed)
        self.current_cycle = None
        self.current_phase = OODAPhase.IDLE
        self._phase_start = None

        return completed

    # ------------------------------------------------------------------
    # Tempo metrics (Boyd's competitive advantage)
    # ------------------------------------------------------------------

    def average_cycle_ms(self) -> Optional[float]:
        """Compute the average cycle duration across completed cycles.

        Returns:
            Average duration in milliseconds, or None if no completed cycles.
        """
        completed = [c for c in self.history if c.duration_ms is not None]
        if not completed:
            return None
        return sum(c.duration_ms for c in completed) / len(completed)

    def tempo_ratio(self) -> Optional[float]:
        """Compute the ratio of actual cycle time to target cycle time.

        A ratio < 1.0 means the agent is faster than target (good).
        A ratio > 1.0 means the agent is slower than target (needs optimization).

        Returns:
            Tempo ratio, or None if no completed cycles.
        """
        avg = self.average_cycle_ms()
        if avg is None:
            return None
        return avg / self.target_cycle_ms

    def phase_bottleneck(self) -> Optional[str]:
        """Identify the slowest phase across completed cycles.

        Boyd's insight: you optimize the loop by finding and fixing the
        bottleneck phase. This method identifies which phase consistently
        takes the most time.

        Returns:
            Name of the slowest phase, or None if no completed cycles.
        """
        completed = [c for c in self.history if c.phase_durations_ms]
        if not completed:
            return None

        phase_totals: dict[str, float] = {}
        phase_counts: dict[str, int] = {}

        for cycle in completed:
            for phase, duration in cycle.phase_durations_ms.items():
                phase_totals[phase] = phase_totals.get(phase, 0.0) + duration
                phase_counts[phase] = phase_counts.get(phase, 0) + 1

        if not phase_totals:
            return None

        phase_averages = {
            phase: phase_totals[phase] / phase_counts[phase]
            for phase in phase_totals
        }
        return max(phase_averages, key=phase_averages.get)

    def metrics(self) -> dict[str, Any]:
        """Return a summary of OODA loop performance metrics.

        Returns:
            Dictionary with cycle count, average duration, tempo ratio,
            bottleneck phase, and current state.
        """
        return {
            "agent_id": self.agent_id,
            "cycles_completed": len([c for c in self.history if c.completed_at]),
            "cycles_abandoned": len([c for c in self.history if c.completed_at is None]),
            "current_phase": self.current_phase.value,
            "average_cycle_ms": self.average_cycle_ms(),
            "target_cycle_ms": self.target_cycle_ms,
            "tempo_ratio": self.tempo_ratio(),
            "bottleneck_phase": self.phase_bottleneck(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_phase(self, expected: OODAPhase) -> None:
        """Verify the loop is in the expected phase.

        Args:
            expected: The phase that must be current.

        Raises:
            RuntimeError: If the current phase doesn't match.
        """
        if self.current_phase != expected:
            raise RuntimeError(
                f"OODA loop is in '{self.current_phase.value}' phase, "
                f"expected '{expected.value}'. Phases must proceed in order: "
                f"observe -> orient -> decide -> act."
            )
        if self.current_cycle is None:
            raise RuntimeError(
                "No active cycle. Call begin_cycle() before recording phase outputs."
            )

    def _record_phase_duration(self, phase_name: str) -> None:
        """Record the duration of the current phase.

        Args:
            phase_name: Name of the phase that just completed.
        """
        if self._phase_start is not None and self.current_cycle is not None:
            now = datetime.now(timezone.utc)
            delta = now - self._phase_start
            self.current_cycle.phase_durations_ms[phase_name] = (
                delta.total_seconds() * 1000.0
            )
