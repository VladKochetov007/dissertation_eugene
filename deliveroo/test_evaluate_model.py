"""Tests for evaluate_model. Run with: python -m pytest test_evaluate_model.py -v"""

import math
import sys
import unittest
from unittest.mock import patch

# Stub pipeline_utils so we can import evaluate_model without the real package
import types

config_module = types.ModuleType("pipeline_utils.config")
config_module.MODEL_PERF_THRESHOLD = 0.5

pipeline_utils = types.ModuleType("pipeline_utils")
pipeline_utils.config = config_module

sys.modules["pipeline_utils"] = pipeline_utils
sys.modules["pipeline_utils.config"] = config_module

from evaluate_model import ModelPerformanceError, _clip, _log_loss, evaluate_model


class TestClip(unittest.TestCase):
    def test_value_in_range(self):
        self.assertEqual(_clip(0.5), 0.5)

    def test_clips_zero(self):
        self.assertGreater(_clip(0.0), 0.0)

    def test_clips_one(self):
        self.assertLess(_clip(1.0), 1.0)

    def test_negative(self):
        self.assertGreater(_clip(-0.5), 0.0)

    def test_above_one(self):
        self.assertLess(_clip(1.5), 1.0)


class TestLogLoss(unittest.TestCase):
    def test_perfect_predictions(self):
        """Perfect predictions should give loss close to 0."""
        y_true = [1, 0, 1, 0]
        y_pred = [0.999, 0.001, 0.999, 0.001]
        loss = _log_loss(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0, places=2)

    def test_worst_predictions(self):
        """Inverted predictions should give high loss."""
        y_true = [1, 0, 1, 0]
        y_pred = [0.001, 0.999, 0.001, 0.999]
        loss = _log_loss(y_true, y_pred)
        self.assertGreater(loss, 5.0)

    def test_uniform_predictions(self):
        """Predicting 0.5 for everything should give log(2) ≈ 0.6931."""
        y_true = [1, 0, 1, 0]
        y_pred = [0.5, 0.5, 0.5, 0.5]
        loss = _log_loss(y_true, y_pred)
        self.assertAlmostEqual(loss, math.log(2), places=4)

    def test_single_positive(self):
        """Single sample, label=1, pred=0.8 → -log(0.8)."""
        loss = _log_loss([1], [0.8])
        self.assertAlmostEqual(loss, -math.log(0.8), places=6)

    def test_single_negative(self):
        """Single sample, label=0, pred=0.3 → -log(0.7)."""
        loss = _log_loss([0], [0.3])
        self.assertAlmostEqual(loss, -math.log(0.7), places=6)

    def test_handles_zero_pred(self):
        """pred=0.0 should not raise due to clipping."""
        loss = _log_loss([1], [0.0])
        self.assertTrue(math.isfinite(loss))

    def test_handles_one_pred(self):
        """pred=1.0 should not raise due to clipping."""
        loss = _log_loss([0], [1.0])
        self.assertTrue(math.isfinite(loss))


class TestEvaluateModel(unittest.TestCase):
    def test_returns_loss(self):
        loss = evaluate_model([1, 0], [0.9, 0.1])
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            evaluate_model([], [])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            evaluate_model([1, 0], [0.5])

    def test_below_threshold_passes(self):
        """Good predictions should pass threshold check."""
        # With threshold=0.5, near-perfect preds should pass
        loss = evaluate_model([1, 0, 1], [0.95, 0.05, 0.95])
        self.assertLess(loss, 0.5)

    def test_above_threshold_raises(self):
        """Bad predictions should raise ModelPerformanceError."""
        with self.assertRaises(ModelPerformanceError):
            evaluate_model([1, 0, 1, 0], [0.1, 0.9, 0.1, 0.9])

    def test_threshold_boundary(self):
        """Loss exactly at threshold should pass (not strictly greater)."""
        import evaluate_model as em

        known_loss = _log_loss([1, 0], [0.5, 0.5])  # = log(2)
        original = em.MODEL_PERF_THRESHOLD
        try:
            em.MODEL_PERF_THRESHOLD = known_loss
            loss = evaluate_model([1, 0], [0.5, 0.5])
            self.assertAlmostEqual(loss, known_loss, places=6)
        finally:
            em.MODEL_PERF_THRESHOLD = original


if __name__ == "__main__":
    unittest.main()
