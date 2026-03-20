"""
Results Collection and Reporting — the warrior's voice to philosopher-kings.

Handles the collection of results from deployed hypotheses, generation of
Evidence entities from real-world outcomes, and structured reporting back
to the philosopher-king layer. This module closes the Republic's governance
loop: philosopher-kings propose, merchants gather, warriors test and deploy,
and feedback reports the outcomes.

The feedback loop is essential for:
1. Validating deployed paradigms against ongoing observations
2. Accumulating evidence that may trigger future Kuhnian crises
3. Providing philosopher-kings with actionable intelligence for new hypotheses
4. Maintaining temporal awareness of how models perform over time

In Boyd's framework, feedback is what makes the OODA loop continuous rather
than one-shot. Each action's results become the next cycle's observations.

Usage:
    collector = FeedbackCollector(store=store)
    collector.record_outcome(
        hypothesis_id="h-001",
        prediction_index=0,
        observed_outcome="price increased by 5%",
        matches_prediction=True,
        confidence=0.85,
    )
    report = collector.generate_report(hypothesis_id="h-001")
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


class OutcomeMatch(str, Enum):
    """Whether an observed outcome matches the hypothesis prediction.

    - match: observation consistent with prediction
    - mismatch: observation contradicts prediction
    - partial: observation partially consistent
    - unknown: cannot determine match (insufficient data)
    """

    MATCH = "match"
    MISMATCH = "mismatch"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Outcome(BaseModel):
    """A single observed outcome from a deployed hypothesis.

    Outcomes are the real-world results that close the feedback loop.
    Each outcome is compared against the hypothesis's predictions to
    determine whether the paradigm continues to hold.

    Attributes:
        id: Unique identifier for this outcome.
        hypothesis_id: ID of the hypothesis being evaluated.
        prediction_index: Index of the prediction being checked (in hypothesis.predictions).
        observed_outcome: Description of what was actually observed.
        matches_prediction: Whether the observation matches the prediction.
        confidence: Confidence in the match assessment (0.0 to 1.0).
        data_source_id: ID of the data source that produced this observation.
        evidence_id: ID of the Evidence entity generated from this outcome.
        details: Additional outcome-specific information.
        timestamp: When this outcome was recorded.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this outcome.",
    )
    hypothesis_id: str = Field(
        ...,
        description="ID of the hypothesis being evaluated.",
    )
    prediction_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Index of the prediction in hypothesis.predictions being checked.",
    )
    observed_outcome: str = Field(
        ...,
        description="Description of what was actually observed.",
    )
    matches_prediction: OutcomeMatch = Field(
        ...,
        description="Whether the observation matches the prediction.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the match assessment.",
    )
    data_source_id: Optional[str] = Field(
        default=None,
        description="ID of the data source that produced this observation.",
    )
    evidence_id: Optional[str] = Field(
        default=None,
        description="ID of the Evidence entity generated from this outcome.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional outcome-specific information.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this outcome was recorded.",
    )


class FeedbackReport(BaseModel):
    """Structured feedback report for philosopher-kings.

    Summarizes the real-world performance of a hypothesis, including
    all observed outcomes, evidence generated, and an overall health
    assessment.

    Attributes:
        id: Unique identifier for this report.
        hypothesis_id: ID of the hypothesis being reported on.
        hypothesis_title: Title of the hypothesis.
        hypothesis_status: Current status of the hypothesis.
        total_outcomes: Number of outcomes recorded.
        matches: Number of outcomes matching predictions.
        mismatches: Number of outcomes contradicting predictions.
        partial_matches: Number of partial matches.
        unknown_matches: Number of undetermined outcomes.
        match_rate: Fraction of outcomes that match predictions.
        average_confidence: Average confidence across outcomes.
        evidence_generated: Number of Evidence entities generated.
        health_assessment: Overall health assessment of the hypothesis.
        recommendation: Recommended action for philosopher-kings.
        outcomes: All recorded outcomes.
        generated_at: When this report was created.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this report.",
    )
    hypothesis_id: str = Field(
        ...,
        description="ID of the hypothesis being reported on.",
    )
    hypothesis_title: str = Field(
        default="",
        description="Title of the hypothesis.",
    )
    hypothesis_status: str = Field(
        default="",
        description="Current status of the hypothesis.",
    )
    total_outcomes: int = Field(
        default=0, ge=0, description="Number of outcomes recorded."
    )
    matches: int = Field(
        default=0, ge=0, description="Outcomes matching predictions."
    )
    mismatches: int = Field(
        default=0, ge=0, description="Outcomes contradicting predictions."
    )
    partial_matches: int = Field(
        default=0, ge=0, description="Partial matches."
    )
    unknown_matches: int = Field(
        default=0, ge=0, description="Undetermined outcomes."
    )
    match_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction matching predictions."
    )
    average_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Average confidence across outcomes."
    )
    evidence_generated: int = Field(
        default=0, ge=0, description="Evidence entities generated."
    )
    health_assessment: str = Field(
        default="",
        description="Overall health assessment: 'healthy', 'degrading', or 'failing'.",
    )
    recommendation: str = Field(
        default="",
        description="Recommended action for philosopher-kings.",
    )
    outcomes: list[Outcome] = Field(
        default_factory=list,
        description="All recorded outcomes.",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this report was created.",
    )


# ---------------------------------------------------------------------------
# Feedback collector
# ---------------------------------------------------------------------------


class FeedbackCollector:
    """Collects results from deployed hypotheses and reports to philosopher-kings.

    Maintains a registry of outcomes per hypothesis, generates Evidence
    entities from observations, and produces structured reports assessing
    hypothesis health in production.

    Attributes:
        store: The shared KnowledgeGraphStore.
        outcomes: Registry of outcomes indexed by hypothesis ID.
        reports: Archive of generated reports.
    """

    def __init__(self, store: KnowledgeGraphStore) -> None:
        """Initialize the feedback collector.

        Args:
            store: The shared KnowledgeGraphStore.
        """
        self.store = store
        self.outcomes: dict[str, list[Outcome]] = {}
        self.reports: list[FeedbackReport] = []

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        hypothesis_id: str,
        observed_outcome: str,
        matches_prediction: OutcomeMatch,
        prediction_index: Optional[int] = None,
        confidence: float = 0.5,
        data_source_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        generate_evidence: bool = True,
    ) -> Outcome:
        """Record an observed outcome for a deployed hypothesis.

        Creates an Outcome record and optionally generates an Evidence
        entity in the knowledge graph. Matching outcomes produce supporting
        evidence; mismatches produce contradicting evidence.

        Args:
            hypothesis_id: ID of the hypothesis being evaluated.
            observed_outcome: Description of the observation.
            matches_prediction: Whether the observation matches predictions.
            prediction_index: Index of the specific prediction being checked.
            confidence: Confidence in the match assessment.
            data_source_id: ID of the data source.
            details: Additional outcome information.
            generate_evidence: Whether to create an Evidence entity.

        Returns:
            The newly created Outcome.
        """
        outcome = Outcome(
            hypothesis_id=hypothesis_id,
            prediction_index=prediction_index,
            observed_outcome=observed_outcome,
            matches_prediction=matches_prediction,
            confidence=confidence,
            data_source_id=data_source_id,
            details=details or {},
        )

        # Store the outcome
        if hypothesis_id not in self.outcomes:
            self.outcomes[hypothesis_id] = []
        self.outcomes[hypothesis_id].append(outcome)

        # Generate evidence if requested and hypothesis exists
        if generate_evidence and hypothesis_id in self.store.hypotheses:
            evidence = self._outcome_to_evidence(outcome)
            if evidence is not None:
                outcome.evidence_id = evidence.id

        return outcome

    def record_batch_outcomes(
        self,
        hypothesis_id: str,
        outcomes: list[dict[str, Any]],
        generate_evidence: bool = True,
    ) -> list[Outcome]:
        """Record multiple outcomes at once.

        Convenience method for bulk outcome recording, e.g., after
        processing a batch of data from merchant agents.

        Args:
            hypothesis_id: ID of the hypothesis being evaluated.
            outcomes: List of outcome dicts with keys: 'observed_outcome',
                     'matches_prediction', and optional 'prediction_index',
                     'confidence', 'data_source_id', 'details'.
            generate_evidence: Whether to create Evidence entities.

        Returns:
            List of created Outcome objects.
        """
        results: list[Outcome] = []
        for outcome_data in outcomes:
            outcome = self.record_outcome(
                hypothesis_id=hypothesis_id,
                observed_outcome=outcome_data["observed_outcome"],
                matches_prediction=OutcomeMatch(outcome_data["matches_prediction"]),
                prediction_index=outcome_data.get("prediction_index"),
                confidence=outcome_data.get("confidence", 0.5),
                data_source_id=outcome_data.get("data_source_id"),
                details=outcome_data.get("details"),
                generate_evidence=generate_evidence,
            )
            results.append(outcome)
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, hypothesis_id: str) -> FeedbackReport:
        """Generate a feedback report for a hypothesis.

        Computes aggregate statistics across all recorded outcomes,
        assesses the hypothesis's health in production, and produces
        a recommendation for philosopher-kings.

        Args:
            hypothesis_id: ID of the hypothesis to report on.

        Returns:
            A FeedbackReport with aggregate metrics and recommendations.
        """
        hypothesis = self.store.get_hypothesis(hypothesis_id)
        outcome_list = self.outcomes.get(hypothesis_id, [])

        # Count outcomes by match type
        matches = sum(1 for o in outcome_list if o.matches_prediction == OutcomeMatch.MATCH)
        mismatches = sum(1 for o in outcome_list if o.matches_prediction == OutcomeMatch.MISMATCH)
        partials = sum(1 for o in outcome_list if o.matches_prediction == OutcomeMatch.PARTIAL)
        unknowns = sum(1 for o in outcome_list if o.matches_prediction == OutcomeMatch.UNKNOWN)

        total = len(outcome_list)
        deterministic = matches + mismatches + partials
        match_rate = (
            (matches + 0.5 * partials) / deterministic if deterministic > 0 else 0.0
        )

        avg_confidence = (
            sum(o.confidence for o in outcome_list) / total if total > 0 else 0.0
        )

        evidence_count = sum(
            1 for o in outcome_list if o.evidence_id is not None
        )

        # Assess health
        health, recommendation = self._assess_health(
            match_rate=match_rate,
            mismatches=mismatches,
            total=total,
            hypothesis=hypothesis,
        )

        report = FeedbackReport(
            hypothesis_id=hypothesis_id,
            hypothesis_title=hypothesis.title if hypothesis else "",
            hypothesis_status=hypothesis.status.value if hypothesis else "",
            total_outcomes=total,
            matches=matches,
            mismatches=mismatches,
            partial_matches=partials,
            unknown_matches=unknowns,
            match_rate=match_rate,
            average_confidence=avg_confidence,
            evidence_generated=evidence_count,
            health_assessment=health,
            recommendation=recommendation,
            outcomes=outcome_list,
        )

        self.reports.append(report)
        return report

    def get_outcomes(
        self,
        hypothesis_id: str,
        match_type: Optional[OutcomeMatch] = None,
    ) -> list[Outcome]:
        """Retrieve outcomes for a hypothesis with optional filtering.

        Args:
            hypothesis_id: ID of the hypothesis.
            match_type: If provided, filter to outcomes with this match type.

        Returns:
            List of matching Outcome objects.
        """
        outcomes = self.outcomes.get(hypothesis_id, [])
        if match_type is not None:
            outcomes = [o for o in outcomes if o.matches_prediction == match_type]
        return outcomes

    def get_latest_report(self, hypothesis_id: str) -> Optional[FeedbackReport]:
        """Retrieve the most recent feedback report for a hypothesis.

        Args:
            hypothesis_id: ID of the hypothesis.

        Returns:
            The most recent FeedbackReport, or None if no reports exist.
        """
        relevant = [
            r for r in self.reports if r.hypothesis_id == hypothesis_id
        ]
        if not relevant:
            return None
        return max(relevant, key=lambda r: r.generated_at)

    # ------------------------------------------------------------------
    # Hypothesis status updates
    # ------------------------------------------------------------------

    def update_hypothesis_from_feedback(
        self,
        hypothesis_id: str,
        mismatch_threshold: int = 3,
    ) -> Optional[HypothesisStatus]:
        """Evaluate whether feedback warrants a hypothesis status change.

        If accumulated mismatches exceed the threshold, the hypothesis
        should be flagged for review (transitioned back to TESTING status
        so warrior agents can re-evaluate it).

        This does NOT directly falsify a hypothesis — that requires
        formal testing through the hypothesis_test module. This method
        only flags hypotheses that are underperforming in production.

        Args:
            hypothesis_id: ID of the hypothesis to evaluate.
            mismatch_threshold: Number of mismatches before flagging.

        Returns:
            The new HypothesisStatus if changed, None if no change needed.
        """
        outcome_list = self.outcomes.get(hypothesis_id, [])
        mismatches = [
            o for o in outcome_list
            if o.matches_prediction == OutcomeMatch.MISMATCH
            and o.confidence >= 0.5
        ]

        if len(mismatches) >= mismatch_threshold:
            hypothesis = self.store.get_hypothesis(hypothesis_id)
            if hypothesis and hypothesis.status == HypothesisStatus.PARADIGM:
                # Flag for re-evaluation — don't falsify directly
                new_status = HypothesisStatus.TESTING
                self.store.update_hypothesis_status(hypothesis_id, new_status)
                return new_status

        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _outcome_to_evidence(self, outcome: Outcome) -> Optional[Evidence]:
        """Convert an Outcome to an Evidence entity and register it.

        Args:
            outcome: The outcome to convert.

        Returns:
            The created Evidence entity, or None if the hypothesis doesn't exist.
        """
        if outcome.hypothesis_id not in self.store.hypotheses:
            return None

        # Determine evidence type from match assessment
        if outcome.matches_prediction == OutcomeMatch.MATCH:
            evidence_type = EvidenceType.SUPPORTING
            confidence = outcome.confidence
        elif outcome.matches_prediction == OutcomeMatch.MISMATCH:
            evidence_type = EvidenceType.CONTRADICTING
            confidence = outcome.confidence
        elif outcome.matches_prediction == OutcomeMatch.PARTIAL:
            evidence_type = EvidenceType.SUPPORTING
            confidence = outcome.confidence * 0.5  # Reduced for partial match
        else:
            # Unknown — record as weak supporting evidence
            evidence_type = EvidenceType.SUPPORTING
            confidence = 0.1

        prediction_info = ""
        if outcome.prediction_index is not None:
            hypothesis = self.store.get_hypothesis(outcome.hypothesis_id)
            if hypothesis and outcome.prediction_index < len(hypothesis.predictions):
                pred = hypothesis.predictions[outcome.prediction_index]
                prediction_info = f" Prediction: if {pred.if_condition} then {pred.then_outcome}."

        description = (
            f"Production feedback [{outcome.matches_prediction.value}]: "
            f"{outcome.observed_outcome}.{prediction_info}"
        )

        evidence = Evidence(
            hypothesis_id=outcome.hypothesis_id,
            type=evidence_type,
            description=description,
            confidence=confidence,
            data_source_id=outcome.data_source_id,
        )
        self.store.add_evidence(evidence)
        return evidence

    @staticmethod
    def _assess_health(
        match_rate: float,
        mismatches: int,
        total: int,
        hypothesis: Optional[Hypothesis],
    ) -> tuple[str, str]:
        """Assess the health of a hypothesis based on feedback.

        Args:
            match_rate: Fraction of outcomes matching predictions.
            mismatches: Count of mismatching outcomes.
            total: Total outcomes recorded.
            hypothesis: The Hypothesis entity.

        Returns:
            Tuple of (health_assessment, recommendation).
        """
        if total == 0:
            return (
                "insufficient_data",
                "No outcomes recorded yet. Continue monitoring and collecting data.",
            )

        if total < 5:
            return (
                "insufficient_data",
                f"Only {total} outcomes recorded. Need at least 5 for reliable assessment.",
            )

        if match_rate >= 0.8:
            return (
                "healthy",
                "Hypothesis is performing well in production. "
                "Predictions match observations at a high rate. "
                "Continue monitoring for drift.",
            )
        elif match_rate >= 0.5:
            return (
                "degrading",
                f"Hypothesis performance is degrading (match rate: {match_rate:.1%}). "
                f"{mismatches} mismatches detected. "
                "Consider initiating anomaly investigation. "
                "The model may need updating or the paradigm may be shifting.",
            )
        else:
            return (
                "failing",
                f"Hypothesis is FAILING in production (match rate: {match_rate:.1%}). "
                f"{mismatches} mismatches out of {total} outcomes. "
                "Recommend immediate review. Consider rollback and initiating "
                "Boyd's destructive deduction to identify what's breaking.",
            )
