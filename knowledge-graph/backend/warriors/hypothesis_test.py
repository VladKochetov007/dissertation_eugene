"""
Hypothesis Testing — the Popperian falsification engine.

Implements statistical hypothesis testing as the formal mechanism through
which warrior agents evaluate philosopher-king conjectures. Each test
produces Evidence entities that update the knowledge graph, moving
hypotheses through their lifecycle: proposed -> testing -> validated/falsified.

Supports three testing modalities:
1. A/B test evaluation: compare treatment and control groups
2. Causal effect estimation: estimate P(Y | do(X)) using adjustment sets
3. Counterfactual comparison: compare observed outcome to counterfactual

The module integrates with the causal DAG engine for identifying valid
adjustment sets and with the knowledge graph store for reading hypotheses
and writing evidence.

References:
    Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
    Popper, K. (1959). The Logic of Scientific Discovery.

Usage:
    tester = HypothesisTester(store=store)
    result = tester.test_hypothesis(hypothesis, method="causal_effect")
    evidence = tester.generate_evidence(hypothesis.id, result)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats as scipy_stats

from graph.entities import (
    CausalDAG,
    Evidence,
    EvidenceType,
    Experiment,
    ExperimentStatus,
    ExperimentType,
    Hypothesis,
    HypothesisStatus,
)
from graph.store import KnowledgeGraphStore
from causal.dag import CausalDAGEngine
from causal.pearl import backdoor_criterion, identify_effect


# ---------------------------------------------------------------------------
# Enums and result models
# ---------------------------------------------------------------------------


class TestMethod(str, Enum):
    """Statistical testing method used by the warrior.

    - ab_test: frequentist comparison of treatment vs. control
    - causal_effect: estimate causal effect using Pearl's framework
    - counterfactual: compare observed to counterfactual outcome
    - correlation: basic correlation analysis (Level 1 only)
    """

    AB_TEST = "ab_test"
    CAUSAL_EFFECT = "causal_effect"
    COUNTERFACTUAL = "counterfactual"
    CORRELATION = "correlation"


class TestVerdict(str, Enum):
    """Outcome of a hypothesis test.

    - supported: evidence consistent with hypothesis predictions
    - contradicted: evidence inconsistent with hypothesis predictions
    - inconclusive: insufficient evidence to determine either way
    """

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"


class TestResult(BaseModel):
    """Structured result of a statistical hypothesis test.

    Captures the full context of a test: method used, statistical measures,
    verdict, and the raw data that produced the result. Designed to be
    fully auditable — a philosopher-king should be able to reproduce or
    critique the reasoning from this record alone.

    Attributes:
        id: Unique identifier for this result.
        hypothesis_id: ID of the hypothesis being tested.
        method: The statistical method used.
        verdict: Overall verdict (supported, contradicted, inconclusive).
        statistic: The test statistic value (t-stat, z-score, etc.).
        p_value: P-value from the statistical test.
        effect_size: Estimated effect size (Cohen's d, ATE, etc.).
        confidence_interval: (lower, upper) bounds of the confidence interval.
        significance_level: Alpha level used for the test.
        sample_size: Number of observations used.
        adjustment_set: Variables conditioned on (for causal effect estimation).
        details: Additional method-specific details.
        timestamp: When this test was executed.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this result.",
    )
    hypothesis_id: str = Field(
        ...,
        description="ID of the hypothesis tested.",
    )
    method: TestMethod = Field(
        ...,
        description="The statistical method used.",
    )
    verdict: TestVerdict = Field(
        ...,
        description="Overall verdict of the test.",
    )
    statistic: Optional[float] = Field(
        default=None,
        description="Test statistic value (t-stat, z-score, etc.).",
    )
    p_value: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="P-value from the statistical test.",
    )
    effect_size: Optional[float] = Field(
        default=None,
        description="Estimated effect size (Cohen's d, ATE, etc.).",
    )
    confidence_interval: Optional[tuple[float, float]] = Field(
        default=None,
        description="(lower, upper) bounds of the confidence interval.",
    )
    significance_level: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Alpha level used for significance determination.",
    )
    sample_size: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of observations used in the test.",
    )
    adjustment_set: Optional[list[str]] = Field(
        default=None,
        description="Variable IDs conditioned on (for causal effect estimation).",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional method-specific details and diagnostics.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this test was executed.",
    )


# ---------------------------------------------------------------------------
# Hypothesis tester
# ---------------------------------------------------------------------------


class HypothesisTester:
    """Statistical hypothesis testing engine for warrior agents.

    Provides methods for testing hypotheses using various statistical
    approaches, generating Evidence entities from results, and managing
    experiments in the knowledge graph.

    The tester integrates three epistemological levels:
    1. Correlation (Pearl Level 1): basic statistical association
    2. Causal effect (Pearl Level 2): interventional estimation with adjustment
    3. Counterfactual (Pearl Level 3): what-if reasoning about unobserved scenarios

    Attributes:
        store: The shared KnowledgeGraphStore.
        default_significance: Default alpha level for significance testing.
    """

    def __init__(
        self,
        store: KnowledgeGraphStore,
        default_significance: float = 0.05,
    ) -> None:
        """Initialize the hypothesis tester.

        Args:
            store: The shared KnowledgeGraphStore for reading hypotheses
                   and writing evidence/experiments.
            default_significance: Default significance level (alpha) for tests.
        """
        self.store = store
        self.default_significance = default_significance

    # ------------------------------------------------------------------
    # Core testing methods
    # ------------------------------------------------------------------

    def run_ab_test(
        self,
        hypothesis_id: str,
        treatment_data: np.ndarray,
        control_data: np.ndarray,
        significance_level: Optional[float] = None,
    ) -> TestResult:
        """Evaluate an A/B test for a hypothesis.

        Performs a two-sample t-test comparing treatment and control groups.
        Computes effect size (Cohen's d) and confidence interval for the
        difference in means.

        Args:
            hypothesis_id: ID of the hypothesis being tested.
            treatment_data: Array of outcome values for the treatment group.
            control_data: Array of outcome values for the control group.
            significance_level: Alpha level. Uses default if not provided.

        Returns:
            A TestResult with the t-test results and verdict.

        Raises:
            ValueError: If either data array is empty.
        """
        if len(treatment_data) == 0 or len(control_data) == 0:
            raise ValueError("Treatment and control data arrays must be non-empty.")

        alpha = significance_level or self.default_significance

        # Two-sample independent t-test
        t_stat, p_value = scipy_stats.ttest_ind(
            treatment_data, control_data, equal_var=False
        )

        # Effect size: Cohen's d
        pooled_std = np.sqrt(
            (np.std(treatment_data, ddof=1) ** 2 + np.std(control_data, ddof=1) ** 2) / 2
        )
        cohens_d = (
            (np.mean(treatment_data) - np.mean(control_data)) / pooled_std
            if pooled_std > 0
            else 0.0
        )

        # Confidence interval for difference in means
        mean_diff = np.mean(treatment_data) - np.mean(control_data)
        se_diff = np.sqrt(
            np.var(treatment_data, ddof=1) / len(treatment_data)
            + np.var(control_data, ddof=1) / len(control_data)
        )
        df = len(treatment_data) + len(control_data) - 2
        t_critical = scipy_stats.t.ppf(1 - alpha / 2, df)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff

        # Determine verdict
        if p_value < alpha:
            verdict = TestVerdict.SUPPORTED if cohens_d > 0 else TestVerdict.CONTRADICTED
        else:
            verdict = TestVerdict.INCONCLUSIVE

        return TestResult(
            hypothesis_id=hypothesis_id,
            method=TestMethod.AB_TEST,
            verdict=verdict,
            statistic=float(t_stat),
            p_value=float(p_value),
            effect_size=float(cohens_d),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            significance_level=alpha,
            sample_size=len(treatment_data) + len(control_data),
            details={
                "treatment_mean": float(np.mean(treatment_data)),
                "control_mean": float(np.mean(control_data)),
                "treatment_std": float(np.std(treatment_data, ddof=1)),
                "control_std": float(np.std(control_data, ddof=1)),
                "treatment_n": len(treatment_data),
                "control_n": len(control_data),
                "degrees_of_freedom": df,
            },
        )

    def run_causal_effect_test(
        self,
        hypothesis_id: str,
        treatment_var: str,
        outcome_var: str,
        observed_data: dict[str, np.ndarray],
        dag: CausalDAG,
        significance_level: Optional[float] = None,
    ) -> TestResult:
        """Estimate the causal effect of treatment on outcome using adjustment.

        Uses Pearl's backdoor criterion to identify valid adjustment sets,
        then estimates the causal effect by conditioning on the adjustment
        variables. Falls back to correlation analysis if no valid adjustment
        set exists.

        This is the warrior's primary tool for moving from Pearl Level 1
        (association) to Pearl Level 2 (intervention) without physically
        intervening.

        Args:
            hypothesis_id: ID of the hypothesis being tested.
            treatment_var: ID of the treatment variable.
            outcome_var: ID of the outcome variable.
            observed_data: Dictionary mapping variable IDs to numpy arrays
                          of observed values. All arrays must have equal length.
            dag: The CausalDAG encoding the hypothesized causal relationships.
            significance_level: Alpha level. Uses default if not provided.

        Returns:
            A TestResult with the causal effect estimate and verdict.

        Raises:
            ValueError: If required variables are missing from observed_data.
            KeyError: If treatment or outcome variable not in the DAG.
        """
        alpha = significance_level or self.default_significance

        if treatment_var not in observed_data or outcome_var not in observed_data:
            raise ValueError(
                "observed_data must contain arrays for both treatment and outcome variables."
            )

        # Build the causal engine from the DAG
        engine = CausalDAGEngine.from_schema(dag)

        # Find valid adjustment sets via backdoor criterion
        identification = identify_effect(engine, treatment_var, outcome_var)

        adjustment_set: Optional[list[str]] = None
        method_detail = "unadjusted_correlation"

        if identification["identifiable"] and identification["adjustment_sets"]:
            # Use the smallest valid adjustment set
            sorted_sets = sorted(identification["adjustment_sets"], key=len)
            best_set = sorted_sets[0]

            # Check that all adjustment variables have data
            if all(v in observed_data for v in best_set):
                adjustment_set = sorted(best_set)
                method_detail = f"{identification['method']}_adjustment"

        treatment = observed_data[treatment_var]
        outcome = observed_data[outcome_var]
        n = len(treatment)

        if adjustment_set and len(adjustment_set) > 0:
            # Adjusted analysis: partial correlation controlling for confounders
            # Using simple linear regression approach: regress Y on X controlling for Z
            confounders = np.column_stack(
                [observed_data[z] for z in adjustment_set]
            )
            # Residualize treatment and outcome with respect to confounders
            treatment_resid = self._residualize(treatment, confounders)
            outcome_resid = self._residualize(outcome, confounders)

            # Test the partial association
            if np.std(treatment_resid) > 0 and np.std(outcome_resid) > 0:
                correlation, p_value = scipy_stats.pearsonr(treatment_resid, outcome_resid)
            else:
                correlation, p_value = 0.0, 1.0

            effect_size = float(correlation)
        else:
            # Unadjusted correlation (Pearl Level 1 fallback)
            if np.std(treatment) > 0 and np.std(outcome) > 0:
                correlation, p_value = scipy_stats.pearsonr(treatment, outcome)
            else:
                correlation, p_value = 0.0, 1.0
            effect_size = float(correlation)

        # Confidence interval for correlation
        if n > 3:
            z_r = np.arctanh(correlation) if abs(correlation) < 1.0 else np.sign(correlation) * 3.0
            se_z = 1.0 / np.sqrt(n - 3)
            z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
            ci_lower = float(np.tanh(z_r - z_crit * se_z))
            ci_upper = float(np.tanh(z_r + z_crit * se_z))
        else:
            ci_lower, ci_upper = -1.0, 1.0

        # Determine verdict
        if p_value < alpha:
            verdict = TestVerdict.SUPPORTED
        elif p_value > 1 - alpha:
            verdict = TestVerdict.CONTRADICTED
        else:
            verdict = TestVerdict.INCONCLUSIVE

        return TestResult(
            hypothesis_id=hypothesis_id,
            method=TestMethod.CAUSAL_EFFECT,
            verdict=verdict,
            statistic=float(correlation),
            p_value=float(p_value),
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            significance_level=alpha,
            sample_size=n,
            adjustment_set=adjustment_set,
            details={
                "identification_method": method_detail,
                "identifiable": identification["identifiable"],
                "n_adjustment_sets_found": len(identification["adjustment_sets"]),
            },
        )

    def run_counterfactual_test(
        self,
        hypothesis_id: str,
        observed_outcome: float,
        counterfactual_outcome: float,
        uncertainty: float = 0.0,
        significance_level: Optional[float] = None,
    ) -> TestResult:
        """Compare an observed outcome to a counterfactual estimate.

        This is Pearl's Level 3 — counterfactual reasoning. Given an
        observed outcome and a model-estimated counterfactual (what would
        have happened without intervention), determine whether the
        intervention had a meaningful effect.

        Args:
            hypothesis_id: ID of the hypothesis being tested.
            observed_outcome: The actually observed outcome value.
            counterfactual_outcome: The model-estimated counterfactual value.
            uncertainty: Standard error of the counterfactual estimate.
            significance_level: Alpha level. Uses default if not provided.

        Returns:
            A TestResult with the counterfactual comparison.
        """
        alpha = significance_level or self.default_significance

        effect = observed_outcome - counterfactual_outcome

        if uncertainty > 0:
            z_score = effect / uncertainty
            p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(z_score))))
            z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
            ci_lower = effect - z_crit * uncertainty
            ci_upper = effect + z_crit * uncertainty
        else:
            # No uncertainty estimate — can only report the point estimate
            z_score = float("inf") if effect != 0 else 0.0
            p_value = 0.0 if effect != 0 else 1.0
            ci_lower = effect
            ci_upper = effect

        # Determine verdict
        if uncertainty > 0 and p_value < alpha:
            verdict = TestVerdict.SUPPORTED if effect > 0 else TestVerdict.CONTRADICTED
        elif uncertainty == 0 and effect != 0:
            verdict = TestVerdict.SUPPORTED if effect > 0 else TestVerdict.CONTRADICTED
        else:
            verdict = TestVerdict.INCONCLUSIVE

        return TestResult(
            hypothesis_id=hypothesis_id,
            method=TestMethod.COUNTERFACTUAL,
            verdict=verdict,
            statistic=float(z_score) if np.isfinite(z_score) else None,
            p_value=p_value,
            effect_size=float(effect),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            significance_level=alpha,
            sample_size=1,
            details={
                "observed_outcome": observed_outcome,
                "counterfactual_outcome": counterfactual_outcome,
                "uncertainty": uncertainty,
            },
        )

    # ------------------------------------------------------------------
    # Evidence and experiment management
    # ------------------------------------------------------------------

    def generate_evidence(
        self,
        hypothesis_id: str,
        result: TestResult,
        data_source_id: Optional[str] = None,
    ) -> Evidence:
        """Create an Evidence entity from a test result and register it.

        Translates the statistical verdict into an EvidenceType (supporting
        or contradicting) and computes a confidence score from the p-value.

        Args:
            hypothesis_id: ID of the hypothesis this evidence relates to.
            result: The TestResult from a hypothesis test.
            data_source_id: Optional ID of the data source used.

        Returns:
            The newly created and registered Evidence entity.
        """
        if result.verdict == TestVerdict.SUPPORTED:
            evidence_type = EvidenceType.SUPPORTING
        elif result.verdict == TestVerdict.CONTRADICTED:
            evidence_type = EvidenceType.CONTRADICTING
        else:
            # Inconclusive results still get recorded as supporting with low confidence
            evidence_type = EvidenceType.SUPPORTING

        # Confidence derived from p-value: lower p-value = higher confidence
        if result.p_value is not None:
            confidence = max(0.0, min(1.0, 1.0 - result.p_value))
        else:
            confidence = 0.5

        # For inconclusive results, cap confidence low
        if result.verdict == TestVerdict.INCONCLUSIVE:
            confidence = min(confidence, 0.3)

        description = (
            f"Test method: {result.method.value}. "
            f"Verdict: {result.verdict.value}. "
            f"Effect size: {result.effect_size}. "
            f"P-value: {result.p_value}. "
            f"Sample size: {result.sample_size}."
        )

        evidence = Evidence(
            hypothesis_id=hypothesis_id,
            type=evidence_type,
            description=description,
            confidence=confidence,
            data_source_id=data_source_id,
        )
        self.store.add_evidence(evidence)
        return evidence

    def create_experiment(
        self,
        hypothesis_id: str,
        experiment_type: ExperimentType,
        description: Optional[str] = None,
    ) -> Experiment:
        """Create and register a new Experiment in the knowledge graph.

        Args:
            hypothesis_id: ID of the hypothesis being tested.
            experiment_type: Kind of experiment (A/B, observational, intervention).
            description: Human-readable description of the experiment design.

        Returns:
            The newly created and registered Experiment.
        """
        experiment = Experiment(
            hypothesis_id=hypothesis_id,
            type=experiment_type,
            status=ExperimentStatus.PLANNED,
            description=description,
        )
        self.store.add_experiment(experiment)
        return experiment

    def complete_experiment(
        self,
        experiment_id: str,
        results: dict[str, Any],
        success: bool = True,
    ) -> Experiment:
        """Mark an experiment as completed with results.

        Args:
            experiment_id: ID of the experiment to complete.
            results: Structured results of the experiment.
            success: Whether the experiment completed successfully.

        Returns:
            The updated Experiment.

        Raises:
            KeyError: If the experiment does not exist.
        """
        experiment = self.store.experiments.get(experiment_id)
        if experiment is None:
            raise KeyError(f"Experiment '{experiment_id}' not found.")

        experiment.status = ExperimentStatus.COMPLETED if success else ExperimentStatus.FAILED
        experiment.results = results
        experiment.updated_at = datetime.now(timezone.utc)

        return experiment

    def evaluate_hypothesis(
        self,
        hypothesis_id: str,
        required_supporting: int = 3,
        max_contradicting: int = 1,
    ) -> HypothesisStatus:
        """Evaluate whether a hypothesis should be validated or falsified.

        Examines all evidence linked to the hypothesis and determines
        whether it has accumulated enough supporting evidence (with
        sufficiently few contradictions) to be validated, or whether
        contradictions warrant falsification.

        This implements the Popperian principle: a single strong
        falsification outweighs many weak confirmations.

        Args:
            hypothesis_id: ID of the hypothesis to evaluate.
            required_supporting: Minimum supporting evidence items needed.
            max_contradicting: Maximum contradicting items before falsification.

        Returns:
            The recommended new HypothesisStatus.
        """
        evidence_items = self.store.get_evidence_for_hypothesis(hypothesis_id)

        supporting = [e for e in evidence_items if e.type == EvidenceType.SUPPORTING]
        contradicting = [e for e in evidence_items if e.type == EvidenceType.CONTRADICTING]

        # Popperian asymmetry: falsification is decisive
        # Weight contradicting evidence by confidence
        high_confidence_contradictions = [
            e for e in contradicting if e.confidence >= 0.7
        ]

        if len(high_confidence_contradictions) > max_contradicting:
            return HypothesisStatus.FALSIFIED

        # Validation requires sufficient high-confidence supporting evidence
        high_confidence_support = [
            e for e in supporting if e.confidence >= 0.5
        ]

        if len(high_confidence_support) >= required_supporting:
            return HypothesisStatus.VALIDATED

        # Still testing
        return HypothesisStatus.TESTING

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _residualize(target: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        """Compute residuals of target after regressing on covariates.

        Uses ordinary least squares to remove the linear effect of
        covariates from the target variable.

        Args:
            target: The variable to residualize (1D array).
            covariates: Matrix of covariate values (2D array).

        Returns:
            Residuals of the target after removing covariate effects.
        """
        # Add intercept
        n = covariates.shape[0]
        X = np.column_stack([np.ones(n), covariates])

        # OLS: beta = (X'X)^{-1} X'y
        try:
            beta = np.linalg.lstsq(X, target, rcond=None)[0]
            predicted = X @ beta
            return target - predicted
        except np.linalg.LinAlgError:
            # If regression fails, return original (no adjustment)
            return target
