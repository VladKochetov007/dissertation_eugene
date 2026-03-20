"""
Warriors package — the implementation arm of the Republic of AI Agents.

Warriors test hypotheses, detect anomalies, deploy validated knowledge, and
report results back to philosopher-kings. They operate through Boyd's OODA
loop (Observe-Orient-Decide-Act) and implement his dialectical engine:
destructive deduction shatters failing paradigms, creative induction
synthesizes new ones from the fragments.

The warrior framework integrates:
- Boyd's OODA loop: continuous observe-orient-decide-act cycles
- Popperian falsification: statistical hypothesis testing with falsification criteria
- Kuhnian crisis detection: anomaly accumulation triggers paradigm shifts
- Boyd's destruction + creation: the dialectical engine for paradigm evolution
- Deployment pipeline: promoting validated knowledge to production
- Feedback collection: real-world results reporting back to philosopher-kings
"""

# OODA loop
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

# Base warrior agent
from .base import (
    WarriorAgent,
    WarriorStatus,
    WarriorStatusReport,
)

# Hypothesis testing
from .hypothesis_test import (
    HypothesisTester,
    TestMethod,
    TestResult,
    TestVerdict,
)

# Anomaly detection (Kuhnian crisis)
from .anomaly import (
    Anomaly,
    AnomalyDetector,
    AnomalySeverity,
    AnomalyType,
    CrisisReport,
    CrisisStatus,
)

# Boyd's destructive deduction
from .destruction import (
    ConstituentStatus,
    DestructiveDeductor,
    DestructionResult,
    ShatteredConstituent,
)

# Boyd's creative induction
from .creation import (
    CommonQuality,
    CreativeInductor,
    SynthesisResult,
)

# Deployment pipeline
from .deployment import (
    DeploymentPipeline,
    DeploymentRecord,
    DeploymentStatus,
    ValidationCheck,
)

# Feedback collection
from .feedback import (
    FeedbackCollector,
    FeedbackReport,
    Outcome,
    OutcomeMatch,
)

__all__ = [
    # OODA loop
    "OODALoop",
    "OODAPhase",
    "OODACycle",
    "Observation",
    "Orientation",
    "Decision",
    "DecisionType",
    "Action",
    # Base warrior agent
    "WarriorAgent",
    "WarriorStatus",
    "WarriorStatusReport",
    # Hypothesis testing
    "HypothesisTester",
    "TestMethod",
    "TestResult",
    "TestVerdict",
    # Anomaly detection
    "AnomalyDetector",
    "Anomaly",
    "AnomalyType",
    "AnomalySeverity",
    "CrisisReport",
    "CrisisStatus",
    # Destructive deduction
    "DestructiveDeductor",
    "DestructionResult",
    "ShatteredConstituent",
    "ConstituentStatus",
    # Creative induction
    "CreativeInductor",
    "CommonQuality",
    "SynthesisResult",
    # Deployment
    "DeploymentPipeline",
    "DeploymentRecord",
    "DeploymentStatus",
    "ValidationCheck",
    # Feedback
    "FeedbackCollector",
    "FeedbackReport",
    "Outcome",
    "OutcomeMatch",
]
