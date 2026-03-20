"""
Abstract Warrior Agent — the implementation arm of the Republic of AI Agents.

Warrior agents are responsible for testing hypotheses, detecting anomalies,
deploying validated knowledge, and reporting results back to philosopher-kings.
They operate through Boyd's OODA loop: Observe-Orient-Decide-Act in continuous
cycles, with each cycle producing evidence that updates the knowledge graph.

In the Republic architecture:
- Philosopher-kings generate hypotheses (Pearl's Level 3: counterfactual)
- Merchants gather data (Pearl's Levels 1-2: association and intervention)
- Warriors TEST and DEPLOY (the Popperian falsification engine)

Warriors are the epistemic immune system: they detect when existing paradigms
fail (Kuhnian anomaly accumulation), shatter failing models (Boyd's destructive
deduction), and synthesize new ones (Boyd's creative induction).

Usage:
    class MyWarrior(WarriorAgent):
        async def observe(self) -> Observation:
            ...
        async def orient(self, observation: Observation) -> Orientation:
            ...
        async def decide(self, orientation: Orientation) -> Decision:
            ...
        async def act(self, decision: Decision) -> Action:
            ...

    warrior = MyWarrior(agent_id="w-001", store=store)
    await warrior.run(max_cycles=100)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from graph.entities import (
    Evidence,
    EvidenceType,
    Hypothesis,
    HypothesisStatus,
)
from graph.store import KnowledgeGraphStore

from eras import EraManager, ContributionRegistry, Contribution, ContributionType

from .ooda import (
    Action,
    Decision,
    DecisionType,
    Observation,
    OODACycle,
    OODALoop,
    OODAPhase,
    Orientation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent status
# ---------------------------------------------------------------------------


class WarriorStatus(str, Enum):
    """Operational status of a warrior agent.

    - idle: agent is initialized but not running
    - running: agent is actively cycling through OODA
    - paused: agent is temporarily suspended (can resume)
    - stopped: agent has been permanently stopped
    - error: agent encountered an unrecoverable error
    """

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Status report model
# ---------------------------------------------------------------------------


class WarriorStatusReport(BaseModel):
    """Status report sent to philosopher-kings.

    Warriors report their operational state, performance metrics, and
    findings to the philosopher-king layer. These reports are the primary
    feedback mechanism in the Republic's governance loop.

    Attributes:
        agent_id: ID of the reporting warrior.
        agent_name: Human-readable name of the warrior.
        status: Current operational status.
        ooda_metrics: Performance metrics from the OODA loop.
        hypotheses_tested: Total hypotheses tested in this session.
        anomalies_detected: Total anomalies detected in this session.
        evidence_generated: Total evidence items produced.
        current_targets: IDs of hypotheses currently being targeted.
        last_cycle_at: Timestamp of the most recent completed cycle.
        uptime_seconds: Seconds since agent was started.
        errors: Recent error messages (last 10).
    """

    agent_id: str = Field(..., description="ID of the reporting warrior.")
    agent_name: str = Field(default="", description="Human-readable agent name.")
    status: WarriorStatus = Field(..., description="Current operational status.")
    era_id: Optional[str] = Field(
        default=None,
        description="ID of the era this warrior is operating within.",
    )
    ooda_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Performance metrics from the OODA loop.",
    )
    hypotheses_tested: int = Field(
        default=0, ge=0, description="Total hypotheses tested."
    )
    anomalies_detected: int = Field(
        default=0, ge=0, description="Total anomalies detected."
    )
    evidence_generated: int = Field(
        default=0, ge=0, description="Total evidence items produced."
    )
    current_targets: list[str] = Field(
        default_factory=list,
        description="IDs of hypotheses currently being targeted.",
    )
    last_cycle_at: Optional[datetime] = Field(
        default=None, description="Timestamp of most recent completed cycle."
    )
    uptime_seconds: float = Field(
        default=0.0, ge=0.0, description="Seconds since agent was started."
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Recent error messages (most recent 10).",
    )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class WarriorAgent(ABC):
    """Abstract base class for all warrior agents in the Republic.

    Subclasses must implement the four OODA phases: observe(), orient(),
    decide(), and act(). The base class handles the run loop, timing,
    metrics tracking, error handling, and knowledge graph interaction.

    The run loop cycles through OODA continuously until stopped or until
    the maximum cycle count is reached. Each cycle produces an auditable
    OODACycle record stored in the OODA loop history.

    Attributes:
        agent_id: Unique identifier for this agent.
        name: Human-readable name.
        store: Reference to the shared KnowledgeGraphStore.
        ooda: The OODA loop manager tracking cycles and tempo.
        status: Current operational status.
        hypotheses_tested: Counter for hypotheses tested.
        anomalies_detected: Counter for anomalies detected.
        evidence_generated: Counter for evidence items produced.
        current_targets: IDs of hypotheses this agent is currently working on.
        started_at: When the agent was started (None if not yet started).
        errors: Recent error log (capped at 100 entries).
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "WarriorAgent",
        store: Optional[KnowledgeGraphStore] = None,
        era_manager: Optional[EraManager] = None,
        contribution_registry: Optional[ContributionRegistry] = None,
        max_phase_timeout_ms: float = 30_000.0,
        target_cycle_ms: float = 10_000.0,
        cycle_delay_seconds: float = 1.0,
    ) -> None:
        """Initialize a warrior agent.

        Args:
            agent_id: Unique ID. Generated if not provided.
            name: Human-readable name for this agent.
            store: The shared KnowledgeGraphStore. Can be set later.
            era_manager: Optional EraManager for era-aware operations.
                When provided, anomalies are reported to the era system
                and all evidence is tagged with the current era ID.
            contribution_registry: Optional ContributionRegistry for tracking
                warrior contributions (test results, anomaly reports) with
                attribution and quality scoring — inspired by OpenForage's
                signal evaluation pipeline.
            max_phase_timeout_ms: Maximum time per OODA phase in milliseconds.
            target_cycle_ms: Target OODA cycle duration for tempo tracking.
            cycle_delay_seconds: Delay between cycles (prevents busy-spinning).
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name
        self.store = store or KnowledgeGraphStore()
        self.era_manager = era_manager
        self.contribution_registry = contribution_registry
        self.ooda = OODALoop(
            agent_id=self.agent_id,
            max_phase_timeout_ms=max_phase_timeout_ms,
            target_cycle_ms=target_cycle_ms,
        )
        self.status = WarriorStatus.IDLE
        self.cycle_delay_seconds = cycle_delay_seconds

        # Counters
        self.hypotheses_tested: int = 0
        self.anomalies_detected: int = 0
        self.evidence_generated: int = 0
        self.current_targets: list[str] = []
        self.started_at: Optional[datetime] = None
        self.errors: list[str] = []

        # Internal control
        self._stop_requested = False

    # ------------------------------------------------------------------
    # Abstract OODA phase methods (subclasses MUST implement)
    # ------------------------------------------------------------------

    @abstractmethod
    async def observe(self) -> Observation:
        """Observe phase: collect data from merchants and the knowledge graph.

        Subclasses should query merchant agents, read from data sources,
        or poll the knowledge graph for new data relevant to their mission.

        Returns:
            An Observation containing the collected data.
        """
        ...

    @abstractmethod
    async def orient(self, observation: Observation) -> Orientation:
        """Orient phase: interpret observation using existing causal models.

        This is Boyd's most critical phase. The warrior applies its current
        worldview (causal models, prior evidence, heuristics) to make sense
        of raw data. Anomalies detected here trigger the Kuhnian crisis
        detection pathway.

        Args:
            observation: Raw data from the Observe phase.

        Returns:
            An Orientation with interpretation and any anomalies detected.
        """
        ...

    @abstractmethod
    async def decide(self, orientation: Orientation) -> Decision:
        """Decide phase: choose the best course of action.

        Based on the interpreted observation, decide what to do:
        test a hypothesis, flag an anomaly, trigger destruction/creation,
        deploy a validated model, or take no action.

        Args:
            orientation: Interpreted observation from the Orient phase.

        Returns:
            A Decision specifying the chosen action and its parameters.
        """
        ...

    @abstractmethod
    async def act(self, decision: Decision) -> Action:
        """Act phase: execute the decision.

        Carry out the chosen action: run a statistical test, update the
        knowledge graph, generate evidence, flag an anomaly, etc.

        Args:
            decision: The chosen action from the Decide phase.

        Returns:
            An Action recording the execution result and any side effects.
        """
        ...

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self, max_cycles: Optional[int] = None) -> list[OODACycle]:
        """Run the OODA loop continuously until stopped or max_cycles reached.

        Each iteration cycles through observe -> orient -> decide -> act,
        producing an OODACycle record. Errors in any phase are caught and
        logged; the loop continues unless a fatal error occurs.

        Args:
            max_cycles: Maximum number of cycles to run. None for unlimited.

        Returns:
            List of completed OODACycle records from this run.
        """
        self.status = WarriorStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self._stop_requested = False
        completed_cycles: list[OODACycle] = []

        logger.info(
            "Warrior '%s' (%s) starting OODA loop (max_cycles=%s)",
            self.name,
            self.agent_id,
            max_cycles,
        )

        cycles_run = 0
        while not self._stop_requested:
            if max_cycles is not None and cycles_run >= max_cycles:
                break

            try:
                cycle = await self._run_single_cycle()
                if cycle is not None:
                    completed_cycles.append(cycle)
                cycles_run += 1
            except Exception as exc:
                error_msg = f"Cycle {cycles_run + 1} failed: {exc}"
                logger.error(error_msg, exc_info=True)
                self._record_error(error_msg)
                # Continue running unless stop was requested
                if self._stop_requested:
                    break

            # Delay between cycles to prevent busy-spinning
            if not self._stop_requested:
                await asyncio.sleep(self.cycle_delay_seconds)

        self.status = WarriorStatus.STOPPED if self._stop_requested else WarriorStatus.IDLE
        logger.info(
            "Warrior '%s' completed %d cycles.",
            self.name,
            cycles_run,
        )
        return completed_cycles

    async def run_single_cycle(self) -> Optional[OODACycle]:
        """Run exactly one OODA cycle.

        Convenience method for testing or manual step-through.

        Returns:
            The completed OODACycle, or None if the cycle failed.
        """
        return await self._run_single_cycle()

    def stop(self) -> None:
        """Request the agent to stop after the current cycle completes."""
        self._stop_requested = True
        logger.info("Stop requested for warrior '%s'.", self.name)

    def pause(self) -> None:
        """Pause the agent. Can be resumed with resume()."""
        self.status = WarriorStatus.PAUSED
        self._stop_requested = True
        logger.info("Warrior '%s' paused.", self.name)

    def resume(self) -> None:
        """Resume a paused agent."""
        if self.status == WarriorStatus.PAUSED:
            self.status = WarriorStatus.IDLE
            self._stop_requested = False
            logger.info("Warrior '%s' resumed.", self.name)

    # ------------------------------------------------------------------
    # Status reporting (for philosopher-kings)
    # ------------------------------------------------------------------

    def status_report(self) -> WarriorStatusReport:
        """Generate a status report for philosopher-kings.

        Returns a structured report summarizing the agent's operational
        state, performance metrics, and findings.

        Returns:
            A WarriorStatusReport with current metrics and state.
        """
        uptime = 0.0
        if self.started_at:
            delta = datetime.now(timezone.utc) - self.started_at
            uptime = delta.total_seconds()

        last_cycle_at = None
        if self.ooda.history:
            last_completed = [c for c in self.ooda.history if c.completed_at]
            if last_completed:
                last_cycle_at = last_completed[-1].completed_at

        current_era_id = None
        if self.era_manager is not None and self.era_manager.current_era is not None:
            current_era_id = self.era_manager.current_era.id

        return WarriorStatusReport(
            agent_id=self.agent_id,
            agent_name=self.name,
            status=self.status,
            era_id=current_era_id,
            ooda_metrics=self.ooda.metrics(),
            hypotheses_tested=self.hypotheses_tested,
            anomalies_detected=self.anomalies_detected,
            evidence_generated=self.evidence_generated,
            current_targets=list(self.current_targets),
            last_cycle_at=last_cycle_at,
            uptime_seconds=uptime,
            errors=self.errors[-10:],
        )

    # ------------------------------------------------------------------
    # Knowledge graph helpers
    # ------------------------------------------------------------------

    def get_testable_hypotheses(self) -> list[Hypothesis]:
        """Retrieve hypotheses that are ready for testing.

        Returns hypotheses in PROPOSED or TESTING status from the
        knowledge graph store.

        Returns:
            List of testable Hypothesis objects.
        """
        proposed = self.store.list_hypotheses(status=HypothesisStatus.PROPOSED)
        testing = self.store.list_hypotheses(status=HypothesisStatus.TESTING)
        return proposed + testing

    def record_evidence(
        self,
        hypothesis_id: str,
        evidence_type: EvidenceType,
        description: str,
        confidence: float = 0.5,
        data_source_id: Optional[str] = None,
    ) -> Evidence:
        """Create and register a new Evidence item in the knowledge graph.

        Convenience method for warrior agents to produce evidence from
        test results. Automatically increments the evidence counter.

        When an EraManager is connected, evidence is tagged with the current
        era and the era's evidence counter is incremented. When a
        ContributionRegistry is connected, the evidence is also submitted
        as a tracked contribution for reputation scoring.

        Args:
            hypothesis_id: ID of the hypothesis this evidence relates to.
            evidence_type: Whether the evidence supports or contradicts.
            description: Human-readable description of the evidence.
            confidence: Confidence score from 0.0 to 1.0.
            data_source_id: Optional ID of the data source.

        Returns:
            The newly created and registered Evidence.
        """
        evidence = Evidence(
            hypothesis_id=hypothesis_id,
            type=evidence_type,
            description=description,
            confidence=confidence,
            data_source_id=data_source_id,
        )
        self.store.add_evidence(evidence)
        self.evidence_generated += 1

        # Report to era system
        if self.era_manager is not None:
            self.era_manager.record_evidence()

        # Submit as tracked contribution for reputation scoring
        if self.contribution_registry is not None and self.era_manager is not None:
            era = self.era_manager.current_era
            if era is not None:
                contribution_type = (
                    ContributionType.HYPOTHESIS_TEST
                    if evidence_type in (EvidenceType.SUPPORTING, EvidenceType.CONTRADICTING)
                    else ContributionType.EVIDENCE
                )
                contribution = Contribution(
                    era_id=era.id,
                    agent_id=self.agent_id,
                    agent_type="warrior",
                    contribution_type=contribution_type,
                    target_entity_id=hypothesis_id,
                    payload={
                        "evidence_id": evidence.id,
                        "evidence_type": evidence_type.value,
                        "description": description,
                    },
                    local_score=confidence,
                )
                self.contribution_registry.submit(contribution)

        return evidence

    def update_hypothesis_status(
        self,
        hypothesis_id: str,
        new_status: HypothesisStatus,
    ) -> Hypothesis:
        """Update a hypothesis's status in the knowledge graph.

        Convenience method that delegates to the store and logs the change.

        Args:
            hypothesis_id: ID of the hypothesis to update.
            new_status: The new status to assign.

        Returns:
            The updated Hypothesis.
        """
        hypothesis = self.store.update_hypothesis_status(hypothesis_id, new_status)
        logger.info(
            "Warrior '%s' updated hypothesis '%s' -> %s",
            self.name,
            hypothesis_id,
            new_status.value,
        )
        return hypothesis

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    async def _run_single_cycle(self) -> Optional[OODACycle]:
        """Execute one complete OODA cycle with error handling.

        Returns:
            The completed OODACycle, or None if any phase failed.
        """
        cycle = self.ooda.begin_cycle()

        try:
            # Observe
            observation = await self.observe()
            self.ooda.record_observe(observation)

            # Orient
            orientation = await self.orient(observation)
            self.ooda.record_orient(orientation)

            # Track anomalies from orientation and report to era system
            if orientation.anomalies_detected:
                self.anomalies_detected += len(orientation.anomalies_detected)
                # Report each anomaly to the era manager for Kuhnian crisis tracking
                if self.era_manager is not None:
                    for _ in orientation.anomalies_detected:
                        self.era_manager.record_anomaly()

            # Decide
            decision = await self.decide(orientation)
            self.ooda.record_decide(decision)

            # Track hypothesis testing
            if decision.decision_type == DecisionType.TEST_HYPOTHESIS:
                self.hypotheses_tested += 1

            # Act
            action = await self.act(decision)
            completed = self.ooda.complete_cycle(action)

            return completed

        except Exception as exc:
            error_msg = f"OODA phase '{self.ooda.current_phase.value}' failed: {exc}"
            self._record_error(error_msg)
            logger.error(error_msg, exc_info=True)
            # Reset the loop state for next cycle
            self.ooda.current_phase = OODAPhase.IDLE
            self.ooda.current_cycle = None
            return None

    def _record_error(self, error: str) -> None:
        """Record an error, keeping only the most recent 100 entries.

        Args:
            error: The error message to record.
        """
        self.errors.append(error)
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]
