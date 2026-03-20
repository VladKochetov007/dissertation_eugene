"""
Era System — Kuhnian paradigm management for the Republic of AI Agents.

Inspired by OpenForage's era system, which organizes agent operations into
temporal checkpoints that define what features are valid, what evaluation
criteria apply, and what thresholds constitute success.

In the Republic's framework, an era is a Kuhnian paradigm: a shared context
within which normal science (hypothesis testing, data gathering, evidence
evaluation) proceeds productively. Era transitions are paradigm shifts:
the evaluation criteria change, new data sources become available, old
hypotheses may be invalidated, and all agents synchronize to the new
parameters.

The era system makes paradigm shifts routine and governable rather than
traumatic and contested. It is the Kirill Function operationalized as
infrastructure.
"""

from .era import Era, EraConfig, EraStatus, EraTransition, EraManager
from .contributions import (
    Contribution,
    ContributionStatus,
    ContributionType,
    AgentReputation,
    ContributionRegistry,
)

__all__ = [
    "Era",
    "EraConfig",
    "EraStatus",
    "EraTransition",
    "EraManager",
    "Contribution",
    "ContributionStatus",
    "ContributionType",
    "AgentReputation",
    "ContributionRegistry",
]
