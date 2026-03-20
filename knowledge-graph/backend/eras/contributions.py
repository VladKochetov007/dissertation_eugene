"""
Contribution Tracking — recording and evaluating agent contributions.

Inspired by OpenForage's signal submission pipeline, where agents discover
signals, submit them for evaluation, and are compensated based on their
out-of-sample performance. The Republic's version tracks merchant data
contributions and warrior hypothesis test results, attributing them to
specific agents within specific eras.

The contribution system serves three functions:

1. ATTRIBUTION: which agent contributed which evidence, enabling reputation
   tracking and accountability (the Popperian skin-in-the-game mechanism)

2. QUALITY EVALUATION: distinguishing in-sample goodness (the merchant's
   local assessment) from out-of-sample goodness (the warrior's validation),
   preventing the submission of evidence that looks good locally but fails
   to generalize (the overfitting problem, stated epistemologically)

3. ENSEMBLE AGGREGATION: combining contributions from multiple independent
   merchants to produce collective evidence stronger than any individual
   signal — the blessing of dimensionality applied to knowledge production
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


class ContributionStatus(str, Enum):
    """Lifecycle of a contribution through the evaluation pipeline.

    Mirrors OpenForage's found → useful distinction:
    - SUBMITTED: agent has submitted the contribution (local evaluation passed)
    - VERIFIED: in-sample evaluation confirmed by server (found)
    - VALIDATED: out-of-sample evaluation passed (useful)
    - REJECTED: failed verification or validation
    - SUPERSEDED: replaced by a better contribution in a later era
    """

    SUBMITTED = "submitted"
    VERIFIED = "verified"
    VALIDATED = "validated"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class ContributionType(str, Enum):
    """What kind of contribution this is."""

    EVIDENCE = "evidence"  # Merchant-submitted data/observation
    HYPOTHESIS_TEST = "hypothesis_test"  # Warrior test result
    ANOMALY_REPORT = "anomaly_report"  # Warrior-detected anomaly
    CAUSAL_MODEL = "causal_model"  # Philosopher-king DAG submission
    FEATURE = "feature"  # New data feature/transformation
    DATA_SOURCE = "data_source"  # New data source registration


# ---------------------------------------------------------------------------
# Contribution model
# ---------------------------------------------------------------------------


class Contribution(BaseModel):
    """A single agent contribution to the knowledge graph.

    Every piece of evidence, test result, anomaly report, or causal model
    submitted to the Republic is recorded as a Contribution. This enables
    attribution, quality tracking, and reputation computation.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique contribution identifier.",
    )
    era_id: str = Field(
        ...,
        description="ID of the era in which this contribution was made.",
    )
    agent_id: str = Field(
        ...,
        description="ID of the merchant/warrior/philosopher-king who submitted this.",
    )
    agent_type: str = Field(
        default="merchant",
        description="Type of agent: merchant, warrior, philosopher_king.",
    )
    contribution_type: ContributionType = Field(
        ...,
        description="What kind of contribution this is.",
    )
    status: ContributionStatus = Field(
        default=ContributionStatus.SUBMITTED,
        description="Current evaluation status.",
    )

    # The contribution content
    target_entity_id: Optional[str] = Field(
        default=None,
        description="ID of the entity this contribution relates to (hypothesis, variable, etc.).",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="The contribution data itself.",
    )

    # Evaluation scores (inspired by OpenForage's in-sample / out-sample split)
    local_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Agent's local (in-sample) quality assessment.",
    )
    verified_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Server's verified (in-sample) quality score.",
    )
    validated_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Out-of-sample quality score (the real test).",
    )
    uniqueness_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How unique/orthogonal this contribution is relative to existing ones.",
    )
    marginal_improvement: Optional[float] = Field(
        default=None,
        description="How much this contribution improves the ensemble (can be negative).",
    )

    # Timestamps
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    verified_at: Optional[datetime] = None
    validated_at: Optional[datetime] = None

    # Provenance
    data_source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of data sources used to produce this contribution.",
    )


# ---------------------------------------------------------------------------
# Agent reputation
# ---------------------------------------------------------------------------


class AgentReputation(BaseModel):
    """Reputation record for an agent in the Republic.

    Reputation is earned through demonstrated quality — successful
    contributions that survive out-of-sample validation — not through
    credentials, institutional affiliation, or social connection.

    This is the Popperian meritocracy operationalized: you earn standing
    in the Republic by making predictions that survive attempts to
    falsify them.
    """

    agent_id: str = Field(..., description="The agent's unique identifier.")
    agent_type: str = Field(default="merchant", description="merchant, warrior, or philosopher_king.")

    # Cumulative metrics
    total_contributions: int = Field(default=0, ge=0)
    verified_contributions: int = Field(default=0, ge=0)
    validated_contributions: int = Field(default=0, ge=0)
    rejected_contributions: int = Field(default=0, ge=0)

    # Quality metrics
    avg_validated_score: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_uniqueness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_marginal_improvement: float = Field(default=0.0)

    # Derived reputation score (0-1)
    reputation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite reputation score weighted toward validated out-of-sample performance.",
    )

    # Era-specific tracking
    contributions_by_era: dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of era_id to contribution count.",
    )

    # Timestamps
    first_contribution_at: Optional[datetime] = None
    last_contribution_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Contribution Registry
# ---------------------------------------------------------------------------


class ContributionRegistry:
    """Registry tracking all contributions and agent reputations.

    Serves as the Republic's institutional memory of who contributed
    what, when, and how well it held up under scrutiny. Combined with
    the Era system, it provides the full temporal audit trail of
    knowledge production.
    """

    def __init__(self) -> None:
        self._contributions: dict[str, Contribution] = {}
        self._reputations: dict[str, AgentReputation] = {}

        # Indices for fast lookups
        self._by_era: dict[str, list[str]] = {}  # era_id -> [contribution_ids]
        self._by_agent: dict[str, list[str]] = {}  # agent_id -> [contribution_ids]
        self._by_entity: dict[str, list[str]] = {}  # entity_id -> [contribution_ids]

    def submit(self, contribution: Contribution) -> Contribution:
        """Submit a new contribution.

        Records the contribution and updates the agent's reputation
        tracking. The contribution enters the evaluation pipeline
        in SUBMITTED status.

        Args:
            contribution: The contribution to submit.

        Returns:
            The stored contribution.
        """
        self._contributions[contribution.id] = contribution

        # Update indices
        self._by_era.setdefault(contribution.era_id, []).append(contribution.id)
        self._by_agent.setdefault(contribution.agent_id, []).append(contribution.id)
        if contribution.target_entity_id:
            self._by_entity.setdefault(
                contribution.target_entity_id, []
            ).append(contribution.id)

        # Ensure reputation record exists
        self._ensure_reputation(contribution.agent_id, contribution.agent_type)
        rep = self._reputations[contribution.agent_id]
        rep.total_contributions += 1
        rep.contributions_by_era[contribution.era_id] = (
            rep.contributions_by_era.get(contribution.era_id, 0) + 1
        )
        rep.last_contribution_at = datetime.now(timezone.utc)
        if rep.first_contribution_at is None:
            rep.first_contribution_at = rep.last_contribution_at

        logger.info(
            "Contribution submitted: %s by agent %s (type=%s, era=%s)",
            contribution.id,
            contribution.agent_id,
            contribution.contribution_type.value,
            contribution.era_id,
        )
        return contribution

    def verify(self, contribution_id: str, score: float) -> Contribution:
        """Mark a contribution as verified (in-sample evaluation passed).

        Args:
            contribution_id: ID of the contribution to verify.
            score: Verified quality score (0-1).

        Returns:
            The updated contribution.

        Raises:
            KeyError: If contribution not found.
            ValueError: If contribution is not in SUBMITTED status.
        """
        contrib = self._get(contribution_id)
        if contrib.status != ContributionStatus.SUBMITTED:
            raise ValueError(
                f"Contribution {contribution_id} is {contrib.status}, expected SUBMITTED."
            )

        contrib.status = ContributionStatus.VERIFIED
        contrib.verified_score = score
        contrib.verified_at = datetime.now(timezone.utc)

        rep = self._reputations.get(contrib.agent_id)
        if rep:
            rep.verified_contributions += 1

        logger.info(
            "Contribution verified: %s (score=%.3f)",
            contribution_id,
            score,
        )
        return contrib

    def validate(
        self,
        contribution_id: str,
        score: float,
        uniqueness: float = 0.5,
        marginal_improvement: float = 0.0,
    ) -> Contribution:
        """Mark a contribution as validated (out-of-sample evaluation passed).

        This is the real test — does the contribution hold up against
        data the agent hasn't seen? Only validated contributions enter
        the knowledge graph as trusted evidence.

        Args:
            contribution_id: ID of the contribution to validate.
            score: Out-of-sample quality score (0-1).
            uniqueness: How orthogonal this is to existing contributions (0-1).
            marginal_improvement: How much this improves the ensemble.

        Returns:
            The updated contribution.
        """
        contrib = self._get(contribution_id)
        if contrib.status != ContributionStatus.VERIFIED:
            raise ValueError(
                f"Contribution {contribution_id} is {contrib.status}, expected VERIFIED."
            )

        contrib.status = ContributionStatus.VALIDATED
        contrib.validated_score = score
        contrib.uniqueness_score = uniqueness
        contrib.marginal_improvement = marginal_improvement
        contrib.validated_at = datetime.now(timezone.utc)

        # Update reputation with running averages
        rep = self._reputations.get(contrib.agent_id)
        if rep:
            rep.validated_contributions += 1
            n = rep.validated_contributions
            rep.avg_validated_score = (
                rep.avg_validated_score * (n - 1) + score
            ) / n
            rep.avg_uniqueness_score = (
                rep.avg_uniqueness_score * (n - 1) + uniqueness
            ) / n
            rep.avg_marginal_improvement = (
                rep.avg_marginal_improvement * (n - 1) + marginal_improvement
            ) / n
            self._recompute_reputation(rep)

        logger.info(
            "Contribution validated: %s (score=%.3f, uniqueness=%.3f, marginal=%.4f)",
            contribution_id,
            score,
            uniqueness,
            marginal_improvement,
        )
        return contrib

    def reject(self, contribution_id: str, reason: str = "") -> Contribution:
        """Reject a contribution that failed verification or validation.

        Args:
            contribution_id: ID of the contribution to reject.
            reason: Why it was rejected.

        Returns:
            The updated contribution.
        """
        contrib = self._get(contribution_id)
        contrib.status = ContributionStatus.REJECTED

        rep = self._reputations.get(contrib.agent_id)
        if rep:
            rep.rejected_contributions += 1
            self._recompute_reputation(rep)

        logger.info(
            "Contribution rejected: %s (reason: %s)",
            contribution_id,
            reason or "unspecified",
        )
        return contrib

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_contributions_for_era(self, era_id: str) -> list[Contribution]:
        """Get all contributions for a specific era."""
        ids = self._by_era.get(era_id, [])
        return [self._contributions[cid] for cid in ids if cid in self._contributions]

    def get_contributions_by_agent(self, agent_id: str) -> list[Contribution]:
        """Get all contributions by a specific agent."""
        ids = self._by_agent.get(agent_id, [])
        return [self._contributions[cid] for cid in ids if cid in self._contributions]

    def get_contributions_for_entity(self, entity_id: str) -> list[Contribution]:
        """Get all contributions targeting a specific entity."""
        ids = self._by_entity.get(entity_id, [])
        return [self._contributions[cid] for cid in ids if cid in self._contributions]

    def get_reputation(self, agent_id: str) -> Optional[AgentReputation]:
        """Get reputation record for an agent."""
        return self._reputations.get(agent_id)

    def get_leaderboard(self, top_n: int = 10) -> list[AgentReputation]:
        """Get the top N agents by reputation score.

        This is the Republic's meritocratic ranking — standing earned
        through demonstrated predictive accuracy, not credentials.
        """
        reps = sorted(
            self._reputations.values(),
            key=lambda r: r.reputation_score,
            reverse=True,
        )
        return reps[:top_n]

    def get_validated_for_entity(self, entity_id: str) -> list[Contribution]:
        """Get only validated contributions for an entity.

        These are the contributions that survived out-of-sample testing
        and are trusted evidence in the knowledge graph.
        """
        return [
            c for c in self.get_contributions_for_entity(entity_id)
            if c.status == ContributionStatus.VALIDATED
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get(self, contribution_id: str) -> Contribution:
        """Retrieve a contribution by ID, raising KeyError if not found."""
        if contribution_id not in self._contributions:
            raise KeyError(f"Contribution '{contribution_id}' not found.")
        return self._contributions[contribution_id]

    def _ensure_reputation(self, agent_id: str, agent_type: str) -> None:
        """Ensure a reputation record exists for the agent."""
        if agent_id not in self._reputations:
            self._reputations[agent_id] = AgentReputation(
                agent_id=agent_id,
                agent_type=agent_type,
            )

    def _recompute_reputation(self, rep: AgentReputation) -> None:
        """Recompute composite reputation score.

        Heavily weights out-of-sample performance (validated_score)
        over volume, penalizes rejections, and rewards uniqueness.
        This mirrors OpenForage's emphasis on out-of-sample goodness
        to discourage overfitting.
        """
        if rep.total_contributions == 0:
            rep.reputation_score = 0.0
            return

        # Validation rate (what fraction of submissions survive OOS testing)
        validation_rate = (
            rep.validated_contributions / max(rep.total_contributions, 1)
        )

        # Rejection penalty
        rejection_rate = (
            rep.rejected_contributions / max(rep.total_contributions, 1)
        )

        # Composite: weighted toward OOS quality and uniqueness
        rep.reputation_score = max(0.0, min(1.0, (
            0.35 * rep.avg_validated_score
            + 0.25 * validation_rate
            + 0.20 * rep.avg_uniqueness_score
            + 0.10 * min(1.0, rep.avg_marginal_improvement * 10)  # Scale marginal improvement
            - 0.10 * rejection_rate
        )))
