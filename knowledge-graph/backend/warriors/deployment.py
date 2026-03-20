"""
Hypothesis Deployment Pipeline — promoting validated knowledge to production.

Handles the lifecycle of deploying validated hypotheses from the knowledge
graph into production systems. In the Republic architecture, deployment is
the moment when a hypothesis — tested by warriors, validated against evidence —
becomes operational knowledge that influences decisions.

The deployment pipeline:
1. Pre-deployment validation: verify the hypothesis meets deployment criteria
2. Deployment: promote the hypothesis to PARADIGM status, snapshot the DAG
3. Monitoring: track deployment health via ongoing OODA cycles
4. Rollback: revert to previous paradigm if the deployed hypothesis fails

This is the Kuhnian "normal science" phase: a validated paradigm becomes
the operating framework until anomalies accumulate sufficiently to trigger
the next crisis-destruction-creation cycle.

Usage:
    pipeline = DeploymentPipeline(store=store)
    deployment = pipeline.deploy(hypothesis_id="h-001")
    status = pipeline.get_deployment_status(deployment.id)
    pipeline.rollback(deployment.id, reason="Anomaly accumulation in production")
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from graph.entities import (
    CausalDAG,
    Evidence,
    EvidenceType,
    Hypothesis,
    HypothesisStatus,
)
from graph.store import KnowledgeGraphStore


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DeploymentStatus(str, Enum):
    """Lifecycle status of a hypothesis deployment.

    - pending: deployment requested but not yet executed
    - validating: pre-deployment checks in progress
    - deployed: hypothesis is live in production
    - monitoring: actively watching for post-deployment anomalies
    - rolled_back: deployment was reverted
    - failed: deployment failed during execution
    """

    PENDING = "pending"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    MONITORING = "monitoring"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class DeploymentRecord(BaseModel):
    """Record of a hypothesis deployment.

    Captures the full lifecycle of deploying a hypothesis to production,
    including pre-deployment checks, deployment timestamp, monitoring
    state, and rollback history.

    Attributes:
        id: Unique identifier for this deployment.
        hypothesis_id: ID of the deployed hypothesis.
        causal_model_id: ID of the CausalDAG snapshot at deployment time.
        status: Current deployment status.
        deployed_at: When the hypothesis was deployed.
        deployed_by: ID of the agent or user who initiated deployment.
        previous_paradigm_id: ID of the hypothesis this deployment replaced.
        validation_results: Results of pre-deployment validation checks.
        rollback_reason: Reason for rollback, if applicable.
        rolled_back_at: When the deployment was rolled back, if applicable.
        metadata: Additional deployment context.
        created_at: When this deployment record was created.
        updated_at: When this record was last updated.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this deployment.",
    )
    hypothesis_id: str = Field(
        ...,
        description="ID of the deployed hypothesis.",
    )
    causal_model_id: Optional[str] = Field(
        default=None,
        description="ID of the CausalDAG snapshot at deployment time.",
    )
    status: DeploymentStatus = Field(
        default=DeploymentStatus.PENDING,
        description="Current deployment status.",
    )
    deployed_at: Optional[datetime] = Field(
        default=None,
        description="When the hypothesis was deployed.",
    )
    deployed_by: str = Field(
        default="system",
        description="ID of the agent or user who initiated deployment.",
    )
    previous_paradigm_id: Optional[str] = Field(
        default=None,
        description="ID of the hypothesis this deployment replaced.",
    )
    validation_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Results of pre-deployment validation checks.",
    )
    rollback_reason: Optional[str] = Field(
        default=None,
        description="Reason for rollback, if applicable.",
    )
    rolled_back_at: Optional[datetime] = Field(
        default=None,
        description="When the deployment was rolled back.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional deployment context.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this deployment record was created.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this record was last updated.",
    )


class ValidationCheck(BaseModel):
    """Result of a single pre-deployment validation check.

    Attributes:
        name: Name of the validation check.
        passed: Whether the check passed.
        message: Human-readable result message.
        details: Check-specific details.
    """

    name: str = Field(..., description="Name of the validation check.")
    passed: bool = Field(..., description="Whether the check passed.")
    message: str = Field(default="", description="Human-readable result message.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Check-specific details.",
    )


# ---------------------------------------------------------------------------
# Deployment pipeline
# ---------------------------------------------------------------------------


class DeploymentPipeline:
    """Pipeline for deploying validated hypotheses to production.

    Manages the full deployment lifecycle: validation, deployment, monitoring,
    and rollback. Maintains a registry of all deployments and the current
    active paradigm.

    Attributes:
        store: The shared KnowledgeGraphStore.
        deployments: Registry of all deployment records.
        active_paradigm_id: ID of the currently deployed (active) hypothesis.
        min_supporting_evidence: Minimum supporting evidence required for deployment.
        min_confidence: Minimum average evidence confidence for deployment.
    """

    def __init__(
        self,
        store: KnowledgeGraphStore,
        min_supporting_evidence: int = 3,
        min_confidence: float = 0.6,
    ) -> None:
        """Initialize the deployment pipeline.

        Args:
            store: The shared KnowledgeGraphStore.
            min_supporting_evidence: Minimum number of supporting evidence items
                                    required before a hypothesis can be deployed.
            min_confidence: Minimum average confidence across supporting evidence.
        """
        self.store = store
        self.deployments: dict[str, DeploymentRecord] = {}
        self.active_paradigm_id: Optional[str] = None
        self.min_supporting_evidence = min_supporting_evidence
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Deployment lifecycle
    # ------------------------------------------------------------------

    def deploy(
        self,
        hypothesis_id: str,
        deployed_by: str = "system",
        force: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> DeploymentRecord:
        """Deploy a validated hypothesis to production.

        Performs pre-deployment validation, promotes the hypothesis to
        PARADIGM status, and records the deployment. If a previous paradigm
        exists, it is superseded but not deleted.

        Args:
            hypothesis_id: ID of the hypothesis to deploy.
            deployed_by: ID of the agent or user initiating deployment.
            force: If True, skip validation checks (use with caution).
            metadata: Additional deployment context.

        Returns:
            The DeploymentRecord for this deployment.

        Raises:
            KeyError: If the hypothesis doesn't exist.
            ValueError: If validation fails and force is False.
        """
        hypothesis = self.store.get_hypothesis(hypothesis_id)
        if hypothesis is None:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found.")

        # Create deployment record
        record = DeploymentRecord(
            hypothesis_id=hypothesis_id,
            causal_model_id=hypothesis.causal_model_id,
            deployed_by=deployed_by,
            previous_paradigm_id=self.active_paradigm_id,
            metadata=metadata or {},
        )

        # Pre-deployment validation
        record.status = DeploymentStatus.VALIDATING
        validation_checks = self._validate_for_deployment(hypothesis)
        record.validation_results = {
            "checks": [c.model_dump() for c in validation_checks],
            "all_passed": all(c.passed for c in validation_checks),
        }

        if not all(c.passed for c in validation_checks) and not force:
            record.status = DeploymentStatus.FAILED
            failed_checks = [c.name for c in validation_checks if not c.passed]
            record.validation_results["failed_checks"] = failed_checks
            self.deployments[record.id] = record
            raise ValueError(
                f"Deployment validation failed for hypothesis '{hypothesis_id}'. "
                f"Failed checks: {failed_checks}. Use force=True to override."
            )

        # Execute deployment
        now = datetime.now(timezone.utc)

        # Promote hypothesis to PARADIGM status
        self.store.update_hypothesis_status(hypothesis_id, HypothesisStatus.PARADIGM)

        # Update deployment record
        record.status = DeploymentStatus.DEPLOYED
        record.deployed_at = now
        record.updated_at = now

        # Record the previous paradigm being superseded
        if self.active_paradigm_id:
            old_hypothesis = self.store.get_hypothesis(self.active_paradigm_id)
            if old_hypothesis and old_hypothesis.status == HypothesisStatus.PARADIGM:
                # Demote old paradigm to VALIDATED (it's still valid, just no longer active)
                self.store.update_hypothesis_status(
                    self.active_paradigm_id, HypothesisStatus.VALIDATED
                )

        # Set new active paradigm
        self.active_paradigm_id = hypothesis_id
        self.deployments[record.id] = record

        return record

    def rollback(
        self,
        deployment_id: str,
        reason: str = "",
    ) -> DeploymentRecord:
        """Roll back a deployment, reverting to the previous paradigm.

        Args:
            deployment_id: ID of the deployment to roll back.
            reason: Explanation for the rollback.

        Returns:
            The updated DeploymentRecord.

        Raises:
            KeyError: If the deployment doesn't exist.
            ValueError: If the deployment is not in a rollback-able state.
        """
        record = self.deployments.get(deployment_id)
        if record is None:
            raise KeyError(f"Deployment '{deployment_id}' not found.")

        if record.status not in (
            DeploymentStatus.DEPLOYED,
            DeploymentStatus.MONITORING,
        ):
            raise ValueError(
                f"Deployment '{deployment_id}' is in status '{record.status.value}' "
                f"and cannot be rolled back. Only DEPLOYED or MONITORING deployments "
                f"can be rolled back."
            )

        now = datetime.now(timezone.utc)

        # Demote the current hypothesis from PARADIGM
        self.store.update_hypothesis_status(
            record.hypothesis_id, HypothesisStatus.VALIDATED
        )

        # Restore the previous paradigm if available
        if record.previous_paradigm_id:
            previous = self.store.get_hypothesis(record.previous_paradigm_id)
            if previous:
                self.store.update_hypothesis_status(
                    record.previous_paradigm_id, HypothesisStatus.PARADIGM
                )
                self.active_paradigm_id = record.previous_paradigm_id
            else:
                self.active_paradigm_id = None
        else:
            self.active_paradigm_id = None

        # Update deployment record
        record.status = DeploymentStatus.ROLLED_BACK
        record.rollback_reason = reason
        record.rolled_back_at = now
        record.updated_at = now

        return record

    # ------------------------------------------------------------------
    # Status and monitoring
    # ------------------------------------------------------------------

    def get_deployment_status(self, deployment_id: str) -> DeploymentRecord:
        """Retrieve the current status of a deployment.

        Args:
            deployment_id: ID of the deployment.

        Returns:
            The DeploymentRecord.

        Raises:
            KeyError: If the deployment doesn't exist.
        """
        record = self.deployments.get(deployment_id)
        if record is None:
            raise KeyError(f"Deployment '{deployment_id}' not found.")
        return record

    def get_active_deployment(self) -> Optional[DeploymentRecord]:
        """Retrieve the currently active deployment, if any.

        Returns:
            The active DeploymentRecord, or None if no paradigm is deployed.
        """
        if self.active_paradigm_id is None:
            return None

        for record in self.deployments.values():
            if (
                record.hypothesis_id == self.active_paradigm_id
                and record.status in (DeploymentStatus.DEPLOYED, DeploymentStatus.MONITORING)
            ):
                return record
        return None

    def list_deployments(
        self,
        status: Optional[DeploymentStatus] = None,
    ) -> list[DeploymentRecord]:
        """List all deployments, optionally filtered by status.

        Args:
            status: If provided, only return deployments with this status.

        Returns:
            List of matching DeploymentRecord objects.
        """
        records = list(self.deployments.values())
        if status is not None:
            records = [r for r in records if r.status == status]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def set_monitoring(self, deployment_id: str) -> DeploymentRecord:
        """Transition a deployed hypothesis to monitoring status.

        Args:
            deployment_id: ID of the deployment to transition.

        Returns:
            The updated DeploymentRecord.

        Raises:
            KeyError: If the deployment doesn't exist.
            ValueError: If the deployment is not in DEPLOYED status.
        """
        record = self.deployments.get(deployment_id)
        if record is None:
            raise KeyError(f"Deployment '{deployment_id}' not found.")

        if record.status != DeploymentStatus.DEPLOYED:
            raise ValueError(
                f"Only DEPLOYED deployments can transition to MONITORING. "
                f"Current status: {record.status.value}."
            )

        record.status = DeploymentStatus.MONITORING
        record.updated_at = datetime.now(timezone.utc)
        return record

    # ------------------------------------------------------------------
    # Pre-deployment validation
    # ------------------------------------------------------------------

    def _validate_for_deployment(
        self,
        hypothesis: Hypothesis,
    ) -> list[ValidationCheck]:
        """Run pre-deployment validation checks on a hypothesis.

        Checks:
        1. Hypothesis is in VALIDATED status (has passed testing)
        2. Sufficient supporting evidence exists
        3. Evidence confidence meets the minimum threshold
        4. Hypothesis has associated causal model
        5. Hypothesis has falsification criteria (predictions)

        Args:
            hypothesis: The Hypothesis to validate.

        Returns:
            List of ValidationCheck results.
        """
        checks: list[ValidationCheck] = []

        # Check 1: Status must be VALIDATED
        checks.append(
            ValidationCheck(
                name="status_validated",
                passed=hypothesis.status == HypothesisStatus.VALIDATED,
                message=(
                    "Hypothesis is VALIDATED."
                    if hypothesis.status == HypothesisStatus.VALIDATED
                    else f"Hypothesis status is '{hypothesis.status.value}', expected 'validated'."
                ),
                details={"current_status": hypothesis.status.value},
            )
        )

        # Check 2: Sufficient supporting evidence
        evidence_items = self.store.get_evidence_for_hypothesis(hypothesis.id)
        supporting = [e for e in evidence_items if e.type == EvidenceType.SUPPORTING]
        checks.append(
            ValidationCheck(
                name="sufficient_evidence",
                passed=len(supporting) >= self.min_supporting_evidence,
                message=(
                    f"Found {len(supporting)} supporting evidence items "
                    f"(minimum: {self.min_supporting_evidence})."
                ),
                details={
                    "supporting_count": len(supporting),
                    "contradicting_count": len(evidence_items) - len(supporting),
                    "minimum_required": self.min_supporting_evidence,
                },
            )
        )

        # Check 3: Evidence confidence
        if supporting:
            avg_confidence = sum(e.confidence for e in supporting) / len(supporting)
        else:
            avg_confidence = 0.0

        checks.append(
            ValidationCheck(
                name="evidence_confidence",
                passed=avg_confidence >= self.min_confidence,
                message=(
                    f"Average supporting evidence confidence: {avg_confidence:.3f} "
                    f"(minimum: {self.min_confidence})."
                ),
                details={
                    "average_confidence": avg_confidence,
                    "minimum_required": self.min_confidence,
                },
            )
        )

        # Check 4: Has causal model
        has_model = hypothesis.causal_model_id is not None
        checks.append(
            ValidationCheck(
                name="has_causal_model",
                passed=has_model,
                message=(
                    "Hypothesis has an associated causal model."
                    if has_model
                    else "Hypothesis has no associated causal model."
                ),
                details={"causal_model_id": hypothesis.causal_model_id},
            )
        )

        # Check 5: Has falsification criteria
        has_predictions = len(hypothesis.predictions) > 0
        checks.append(
            ValidationCheck(
                name="has_falsification_criteria",
                passed=has_predictions,
                message=(
                    f"Hypothesis has {len(hypothesis.predictions)} falsifiable predictions."
                    if has_predictions
                    else "Hypothesis has no falsifiable predictions (Popperian criterion violated)."
                ),
                details={"prediction_count": len(hypothesis.predictions)},
            )
        )

        return checks
