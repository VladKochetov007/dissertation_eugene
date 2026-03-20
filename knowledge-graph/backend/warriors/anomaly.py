"""
Anomaly Detection — Kuhnian crisis detection for the knowledge graph.

Implements anomaly tracking and crisis detection based on Kuhn's model of
scientific revolutions. In Kuhn's framework:

1. Normal science: the current paradigm (validated hypotheses) explains
   observations successfully, with occasional minor anomalies.
2. Crisis: anomalies accumulate past a manageable threshold. The paradigm
   can no longer explain what's being observed. Baroque epicycles appear.
3. Revolution: a new paradigm replaces the old one (Boyd's destruction
   + creation cycle).

This module tracks anomalies against existing hypotheses and causal models,
detects when anomaly accumulation reaches crisis levels, and triggers the
appropriate response: either update the model or initiate destructive
deduction to shatter it.

Anomaly types:
- Prediction failure: a hypothesis's prediction is contradicted by observation
- Unexpected confounder: a variable not in the model explains variance
- Model drift: the causal model's predictions degrade over time
- Structural anomaly: observed causal relationships contradict the DAG structure

References:
    Kuhn, T. (1962). The Structure of Scientific Revolutions.
    Boyd, J. (1976). Destruction and Creation.

Usage:
    detector = AnomalyDetector(store=store, crisis_threshold=5)
    anomaly = detector.record_anomaly(hypothesis_id, anomaly_type, ...)
    status = detector.check_crisis(hypothesis_id)
"""

from __future__ import annotations

import uuid
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


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AnomalyType(str, Enum):
    """Types of anomalies detectable against existing paradigms.

    Each type corresponds to a specific failure mode of the current model:
    - prediction_failure: hypothesis prediction contradicted by data
    - unexpected_confounder: unmodeled variable explains significant variance
    - model_drift: model accuracy degrades over time (concept drift)
    - structural_anomaly: observed relationships contradict the DAG structure
    """

    PREDICTION_FAILURE = "prediction_failure"
    UNEXPECTED_CONFOUNDER = "unexpected_confounder"
    MODEL_DRIFT = "model_drift"
    STRUCTURAL_ANOMALY = "structural_anomaly"


class AnomalySeverity(str, Enum):
    """Severity of a detected anomaly.

    Maps onto Kuhn's distinction between minor anomalies (absorbed by
    the paradigm) and critical anomalies (threaten paradigm integrity):
    - low: minor deviation, within noise tolerance
    - medium: notable deviation, warrants monitoring
    - high: significant deviation, requires investigation
    - critical: paradigm-threatening anomaly, may trigger crisis
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CrisisStatus(str, Enum):
    """Kuhnian crisis status for a hypothesis or causal model.

    - normal: anomalies within acceptable range (normal science phase)
    - accumulating: anomalies building but not yet at crisis threshold
    - crisis: anomaly threshold exceeded — paradigm is failing
    - revolution: destructive deduction initiated, new paradigm emerging
    """

    NORMAL = "normal"
    ACCUMULATING = "accumulating"
    CRISIS = "crisis"
    REVOLUTION = "revolution"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Anomaly(BaseModel):
    """A single detected anomaly against an existing hypothesis or model.

    Anomalies are the raw material of paradigm change. Each one represents
    an observation that the current model cannot adequately explain.

    Attributes:
        id: Unique identifier for this anomaly.
        hypothesis_id: ID of the hypothesis this anomaly challenges.
        causal_model_id: ID of the causal DAG that failed to predict this.
        anomaly_type: Category of anomaly detected.
        severity: How severe this anomaly is relative to the paradigm.
        description: Human-readable description of the anomaly.
        expected_value: What the model predicted.
        observed_value: What was actually observed.
        deviation: Quantitative deviation from prediction (e.g., residual).
        variable_ids: IDs of variables involved in the anomaly.
        evidence_id: ID of the Evidence entity generated from this anomaly.
        timestamp: When this anomaly was detected.
        metadata: Additional context (data source, method, etc.).
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this anomaly.",
    )
    hypothesis_id: str = Field(
        ...,
        description="ID of the hypothesis this anomaly challenges.",
    )
    causal_model_id: Optional[str] = Field(
        default=None,
        description="ID of the causal DAG that failed to predict this.",
    )
    anomaly_type: AnomalyType = Field(
        ...,
        description="Category of anomaly detected.",
    )
    severity: AnomalySeverity = Field(
        default=AnomalySeverity.MEDIUM,
        description="How severe this anomaly is.",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the anomaly.",
    )
    expected_value: Optional[Any] = Field(
        default=None,
        description="What the model predicted.",
    )
    observed_value: Optional[Any] = Field(
        default=None,
        description="What was actually observed.",
    )
    deviation: Optional[float] = Field(
        default=None,
        description="Quantitative deviation from prediction.",
    )
    variable_ids: list[str] = Field(
        default_factory=list,
        description="IDs of variables involved in this anomaly.",
    )
    evidence_id: Optional[str] = Field(
        default=None,
        description="ID of the Evidence entity generated from this anomaly.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this anomaly was detected.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (data source, method, etc.).",
    )


class CrisisReport(BaseModel):
    """Report on the crisis status of a hypothesis.

    Summarizes the anomaly situation for a philosopher-king to evaluate.
    When status reaches CRISIS, this report contains the information
    needed to decide whether to initiate destructive deduction.

    Attributes:
        hypothesis_id: ID of the hypothesis being assessed.
        status: Current Kuhnian crisis status.
        total_anomalies: Total number of anomalies recorded.
        anomalies_by_type: Count of anomalies per type.
        anomalies_by_severity: Count of anomalies per severity.
        crisis_score: Numerical crisis score (0.0 = stable, 1.0 = full crisis).
        threshold: The crisis threshold being used.
        most_recent_anomaly: Timestamp of the most recent anomaly.
        recommendation: Suggested action for philosopher-kings.
        anomaly_ids: IDs of all anomalies contributing to this report.
    """

    hypothesis_id: str = Field(
        ...,
        description="ID of the hypothesis being assessed.",
    )
    status: CrisisStatus = Field(
        ...,
        description="Current Kuhnian crisis status.",
    )
    total_anomalies: int = Field(
        default=0, ge=0, description="Total anomalies recorded."
    )
    anomalies_by_type: dict[str, int] = Field(
        default_factory=dict,
        description="Count per anomaly type.",
    )
    anomalies_by_severity: dict[str, int] = Field(
        default_factory=dict,
        description="Count per severity level.",
    )
    crisis_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0.0 = stable paradigm, 1.0 = full crisis.",
    )
    threshold: float = Field(
        default=0.0,
        description="Crisis threshold used for this assessment.",
    )
    most_recent_anomaly: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the most recent anomaly.",
    )
    recommendation: str = Field(
        default="",
        description="Suggested action for philosopher-kings.",
    )
    anomaly_ids: list[str] = Field(
        default_factory=list,
        description="IDs of anomalies contributing to this report.",
    )


# ---------------------------------------------------------------------------
# Anomaly detector
# ---------------------------------------------------------------------------

# Severity weights for computing the weighted crisis score.
_SEVERITY_WEIGHTS: dict[AnomalySeverity, float] = {
    AnomalySeverity.LOW: 0.25,
    AnomalySeverity.MEDIUM: 0.5,
    AnomalySeverity.HIGH: 1.0,
    AnomalySeverity.CRITICAL: 2.0,
}


class AnomalyDetector:
    """Kuhnian crisis detection engine for the knowledge graph.

    Tracks anomalies against hypotheses and causal models, computes
    crisis scores, and triggers appropriate responses when anomaly
    accumulation exceeds the crisis threshold.

    The crisis score is a severity-weighted sum of anomalies normalized
    by the crisis threshold. This reflects Kuhn's insight that not all
    anomalies are equally threatening to a paradigm: a single critical
    anomaly can trigger crisis while many low-severity ones are absorbed
    as "noise" during normal science.

    Attributes:
        store: The shared KnowledgeGraphStore.
        crisis_threshold: Weighted anomaly score at which crisis is declared.
        anomalies: All recorded anomalies, indexed by hypothesis ID.
        crisis_statuses: Current crisis status per hypothesis.
    """

    def __init__(
        self,
        store: KnowledgeGraphStore,
        crisis_threshold: float = 5.0,
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            store: The shared KnowledgeGraphStore.
            crisis_threshold: Severity-weighted anomaly score at which a
                             hypothesis enters crisis. Default 5.0 means
                             (for example) 5 medium anomalies, or 2.5 high,
                             or 10 low anomalies trigger crisis.
        """
        self.store = store
        self.crisis_threshold = crisis_threshold

        # Anomaly storage: hypothesis_id -> list of anomalies
        self.anomalies: dict[str, list[Anomaly]] = {}

        # Crisis status tracking
        self.crisis_statuses: dict[str, CrisisStatus] = {}

    # ------------------------------------------------------------------
    # Anomaly recording
    # ------------------------------------------------------------------

    def record_anomaly(
        self,
        hypothesis_id: str,
        anomaly_type: AnomalyType,
        description: str,
        severity: AnomalySeverity = AnomalySeverity.MEDIUM,
        expected_value: Optional[Any] = None,
        observed_value: Optional[Any] = None,
        deviation: Optional[float] = None,
        variable_ids: Optional[list[str]] = None,
        causal_model_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        generate_evidence: bool = True,
    ) -> Anomaly:
        """Record a new anomaly against a hypothesis.

        Creates an Anomaly record, optionally generates a contradicting
        Evidence item in the knowledge graph, and updates the crisis
        score for the affected hypothesis.

        Args:
            hypothesis_id: ID of the hypothesis this anomaly challenges.
            anomaly_type: Category of anomaly.
            description: Human-readable description.
            severity: How severe the anomaly is.
            expected_value: What the model predicted.
            observed_value: What was actually observed.
            deviation: Quantitative deviation.
            variable_ids: IDs of variables involved.
            causal_model_id: ID of the causal DAG that failed.
            metadata: Additional context.
            generate_evidence: Whether to create a contradicting Evidence entity.

        Returns:
            The newly created Anomaly.
        """
        anomaly = Anomaly(
            hypothesis_id=hypothesis_id,
            causal_model_id=causal_model_id,
            anomaly_type=anomaly_type,
            severity=severity,
            description=description,
            expected_value=expected_value,
            observed_value=observed_value,
            deviation=deviation,
            variable_ids=variable_ids or [],
            metadata=metadata or {},
        )

        # Store the anomaly
        if hypothesis_id not in self.anomalies:
            self.anomalies[hypothesis_id] = []
        self.anomalies[hypothesis_id].append(anomaly)

        # Generate contradicting evidence if requested and hypothesis exists
        if generate_evidence and hypothesis_id in self.store.hypotheses:
            confidence = _SEVERITY_WEIGHTS.get(severity, 0.5) / 2.0
            confidence = min(confidence, 1.0)

            evidence = Evidence(
                hypothesis_id=hypothesis_id,
                type=EvidenceType.CONTRADICTING,
                description=f"Anomaly [{anomaly_type.value}]: {description}",
                confidence=confidence,
            )
            self.store.add_evidence(evidence)
            anomaly.evidence_id = evidence.id

        # Update crisis status
        self._update_crisis_status(hypothesis_id)

        return anomaly

    # ------------------------------------------------------------------
    # Crisis assessment
    # ------------------------------------------------------------------

    def check_crisis(self, hypothesis_id: str) -> CrisisReport:
        """Assess the crisis status of a hypothesis.

        Computes the severity-weighted crisis score, determines the
        Kuhnian phase, and generates a recommendation for philosopher-kings.

        Args:
            hypothesis_id: ID of the hypothesis to assess.

        Returns:
            A CrisisReport summarizing the anomaly situation.
        """
        anomaly_list = self.anomalies.get(hypothesis_id, [])

        if not anomaly_list:
            return CrisisReport(
                hypothesis_id=hypothesis_id,
                status=CrisisStatus.NORMAL,
                total_anomalies=0,
                crisis_score=0.0,
                threshold=self.crisis_threshold,
                recommendation="No anomalies detected. Paradigm stable (normal science).",
            )

        # Count by type and severity
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        weighted_score = 0.0

        for anomaly in anomaly_list:
            by_type[anomaly.anomaly_type.value] = (
                by_type.get(anomaly.anomaly_type.value, 0) + 1
            )
            by_severity[anomaly.severity.value] = (
                by_severity.get(anomaly.severity.value, 0) + 1
            )
            weighted_score += _SEVERITY_WEIGHTS.get(anomaly.severity, 0.5)

        # Normalize to [0, 1] crisis score
        crisis_score = min(1.0, weighted_score / self.crisis_threshold)

        # Determine crisis status
        status = self._score_to_status(crisis_score)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            status, crisis_score, by_type, by_severity
        )

        most_recent = max(a.timestamp for a in anomaly_list)

        return CrisisReport(
            hypothesis_id=hypothesis_id,
            status=status,
            total_anomalies=len(anomaly_list),
            anomalies_by_type=by_type,
            anomalies_by_severity=by_severity,
            crisis_score=crisis_score,
            threshold=self.crisis_threshold,
            most_recent_anomaly=most_recent,
            recommendation=recommendation,
            anomaly_ids=[a.id for a in anomaly_list],
        )

    def get_anomalies(
        self,
        hypothesis_id: str,
        anomaly_type: Optional[AnomalyType] = None,
        min_severity: Optional[AnomalySeverity] = None,
    ) -> list[Anomaly]:
        """Retrieve anomalies for a hypothesis with optional filtering.

        Args:
            hypothesis_id: ID of the hypothesis.
            anomaly_type: Filter to a specific anomaly type.
            min_severity: Filter to anomalies at or above this severity.

        Returns:
            List of matching Anomaly objects.
        """
        anomaly_list = self.anomalies.get(hypothesis_id, [])

        if anomaly_type is not None:
            anomaly_list = [a for a in anomaly_list if a.anomaly_type == anomaly_type]

        if min_severity is not None:
            min_weight = _SEVERITY_WEIGHTS.get(min_severity, 0.0)
            anomaly_list = [
                a
                for a in anomaly_list
                if _SEVERITY_WEIGHTS.get(a.severity, 0.0) >= min_weight
            ]

        return anomaly_list

    def get_hypotheses_in_crisis(self) -> list[str]:
        """Return IDs of all hypotheses currently in crisis.

        Returns:
            List of hypothesis IDs with crisis status >= CRISIS.
        """
        return [
            h_id
            for h_id, status in self.crisis_statuses.items()
            if status in (CrisisStatus.CRISIS, CrisisStatus.REVOLUTION)
        ]

    def mark_revolution(self, hypothesis_id: str) -> None:
        """Mark a hypothesis as undergoing revolution (destructive deduction initiated).

        Args:
            hypothesis_id: ID of the hypothesis entering revolution.
        """
        self.crisis_statuses[hypothesis_id] = CrisisStatus.REVOLUTION

    def clear_anomalies(self, hypothesis_id: str) -> None:
        """Clear all anomalies for a hypothesis (after paradigm shift completes).

        Called after a successful revolution — the new paradigm starts
        with a clean slate. This is the Kuhnian fresh start.

        Args:
            hypothesis_id: ID of the hypothesis to clear.
        """
        self.anomalies.pop(hypothesis_id, None)
        self.crisis_statuses[hypothesis_id] = CrisisStatus.NORMAL

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_crisis_status(self, hypothesis_id: str) -> None:
        """Recompute the crisis status for a hypothesis after a new anomaly.

        Args:
            hypothesis_id: ID of the hypothesis to update.
        """
        anomaly_list = self.anomalies.get(hypothesis_id, [])
        weighted_score = sum(
            _SEVERITY_WEIGHTS.get(a.severity, 0.5) for a in anomaly_list
        )
        crisis_score = min(1.0, weighted_score / self.crisis_threshold)

        # Don't downgrade from REVOLUTION status
        current = self.crisis_statuses.get(hypothesis_id, CrisisStatus.NORMAL)
        if current == CrisisStatus.REVOLUTION:
            return

        self.crisis_statuses[hypothesis_id] = self._score_to_status(crisis_score)

    @staticmethod
    def _score_to_status(crisis_score: float) -> CrisisStatus:
        """Map a crisis score to a Kuhnian status.

        Args:
            crisis_score: Normalized score from 0.0 to 1.0.

        Returns:
            The corresponding CrisisStatus.
        """
        if crisis_score < 0.3:
            return CrisisStatus.NORMAL
        elif crisis_score < 0.7:
            return CrisisStatus.ACCUMULATING
        else:
            return CrisisStatus.CRISIS

    @staticmethod
    def _generate_recommendation(
        status: CrisisStatus,
        crisis_score: float,
        by_type: dict[str, int],
        by_severity: dict[str, int],
    ) -> str:
        """Generate a human-readable recommendation for philosopher-kings.

        Args:
            status: Current crisis status.
            crisis_score: Normalized crisis score.
            by_type: Anomaly counts by type.
            by_severity: Anomaly counts by severity.

        Returns:
            A recommendation string.
        """
        if status == CrisisStatus.NORMAL:
            return (
                "Paradigm stable. Anomalies within normal range. "
                "Continue normal science operations."
            )
        elif status == CrisisStatus.ACCUMULATING:
            dominant_type = max(by_type, key=by_type.get) if by_type else "unknown"
            return (
                f"Anomalies accumulating (score: {crisis_score:.2f}). "
                f"Dominant anomaly type: {dominant_type}. "
                "Consider reviewing the causal model for these specific failure modes. "
                "Not yet at crisis level — continued monitoring recommended."
            )
        elif status == CrisisStatus.CRISIS:
            critical_count = by_severity.get("critical", 0)
            high_count = by_severity.get("high", 0)
            return (
                f"CRISIS: Anomaly threshold exceeded (score: {crisis_score:.2f}). "
                f"Critical anomalies: {critical_count}, High: {high_count}. "
                "The current paradigm is failing to explain observations. "
                "Recommend initiating Boyd's destructive deduction to shatter "
                "the failing model and identify which causal claims are still "
                "supported by evidence."
            )
        else:
            return (
                "Revolution in progress. Destructive deduction has been initiated. "
                "Await completion of destruction + creation cycle."
            )
