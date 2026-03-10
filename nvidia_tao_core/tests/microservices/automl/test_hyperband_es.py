# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for automl/hyperband_es.py - HyperBand with Early Stopping"""

import numpy as np
from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.automl.hyperband_es import HyperBandES
from nvidia_tao_core.microservices.utils.automl_utils import JobStates


PATCH_PREFIX = 'nvidia_tao_core.microservices.automl'


def _make_hbes(job_id="job_hbes_test", max_epochs=9, reduction_factor=3,
               epoch_multiplier=1, early_stop_threshold=0.8,
               min_early_stop_epochs=3, metric="loss"):
    """Helper to create a HyperBandES instance with mocked dependencies."""
    with patch(f'{PATCH_PREFIX}.hyperband.save_job_specs'), \
         patch(f'{PATCH_PREFIX}.hyperband.get_job_specs',
               return_value={"training_config": {"num_epochs": 10}}), \
         patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_automl_custom_param_ranges',
               return_value={}), \
         patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_job_specs',
               return_value={}):
        job_context = Mock()
        job_context.id = job_id
        job_context.handler_id = f"exp_{job_id}"
        parameters = [{"parameter": "learning_rate"}]
        brain = HyperBandES(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters,
            max_epochs=max_epochs,
            reduction_factor=reduction_factor,
            epoch_multiplier=epoch_multiplier,
            early_stop_threshold=early_stop_threshold,
            min_early_stop_epochs=min_early_stop_epochs
        )
        brain.metric = metric
        if metric == "loss":
            brain.reverse_sort = False
        return brain


def _make_rec(rec_id, status, result):
    """Helper to create a mock recommendation/history record."""
    rec = Mock()
    rec.id = rec_id
    rec.status = status
    rec.result = result
    return rec


class TestHyperBandESInitialization:
    """Test HyperBandES initialization and inheritance."""

    def test_inherits_hyperband_brackets(self):
        """Verify bracket/rung structure is inherited from HyperBand."""
        brain = _make_hbes(max_epochs=9, reduction_factor=3)
        assert brain.ni is not None
        assert brain.ri is not None
        assert len(brain.ni) > 0
        assert len(brain.ri) > 0

    def test_es_specific_attributes(self):
        """Verify ES-specific attributes are initialized correctly."""
        brain = _make_hbes(early_stop_threshold=0.9, min_early_stop_epochs=5)
        assert brain.confidence_threshold == 0.9
        assert brain.min_epochs_for_prediction == 5
        assert brain.learning_curves == {}
        assert brain.early_stopped_configs == set()
        assert brain.configs_to_cancel == set()

    def test_default_es_parameters(self):
        """Verify default early stopping parameters."""
        brain = _make_hbes()
        assert brain.confidence_threshold == 0.8
        assert brain.min_epochs_for_prediction == 3


class TestPowerLawModel:
    """Test the power law learning curve model."""

    def test_power_law_basic(self):
        """Power law model returns expected values."""
        result = HyperBandES._power_law_model(np.array([1.0]), 1.0, -0.5, 0.0)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_power_law_decay(self):
        """Power law with negative exponent produces decreasing values."""
        x = np.array([1.0, 2.0, 4.0, 8.0])
        y = HyperBandES._power_law_model(x, 1.0, -0.5, 0.0)
        for i in range(len(y) - 1):
            assert y[i] > y[i + 1], "Power law with b<0 should decay"

    def test_power_law_offset(self):
        """Power law c parameter shifts the curve vertically."""
        x = np.array([2.0])
        y_no_offset = HyperBandES._power_law_model(x, 1.0, -0.5, 0.0)
        y_offset = HyperBandES._power_law_model(x, 1.0, -0.5, 0.5)
        np.testing.assert_allclose(y_offset - y_no_offset, [0.5], atol=1e-6)


class TestExponentialModel:
    """Test the exponential learning curve model."""

    def test_exponential_decay(self):
        """Exponential model with positive b produces decreasing values."""
        x = np.array([1.0, 2.0, 4.0, 8.0])
        y = HyperBandES._exponential_model(x, 1.0, 0.5, 0.0)
        for i in range(len(y) - 1):
            assert y[i] > y[i + 1], "Exponential with b>0 should decay"

    def test_exponential_approaches_offset(self):
        """Exponential model converges to c for large x."""
        x_large = np.array([100.0])
        y = HyperBandES._exponential_model(x_large, 1.0, 1.0, 0.3)
        np.testing.assert_allclose(y, [0.3], atol=1e-6)


class TestPredictFinalPerformance:
    """Test learning curve prediction logic."""

    def test_too_few_points_returns_none(self):
        """Returns None when fewer data points than min_epochs_for_prediction."""
        brain = _make_hbes(min_early_stop_epochs=3)
        curve = [(1, 0.5), (2, 0.45)]
        predicted, confidence = brain._predict_final_performance("cfg0", curve)
        assert predicted is None
        assert confidence == 0.0

    def test_prediction_with_clean_power_law_data(self):
        """Prediction succeeds with data that fits a power law curve well."""
        brain = _make_hbes(max_epochs=9, reduction_factor=3, min_early_stop_epochs=3)
        brain.bracket = "0"

        a, b, c = 0.5, -0.5, 0.2
        epochs = [1, 2, 3, 4, 5]
        metrics = [a * (e ** b) + c for e in epochs]
        curve = list(zip(epochs, metrics))

        predicted, confidence = brain._predict_final_performance("cfg0", curve)
        assert predicted is not None
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5, "Clean power law data should yield high confidence"

    def test_prediction_confidence_bounds(self):
        """Confidence is always clamped to [0, 1]."""
        brain = _make_hbes(max_epochs=9, reduction_factor=3, min_early_stop_epochs=3)
        brain.bracket = "0"

        curve = [(1, 0.9), (2, 0.7), (3, 0.55), (4, 0.45), (5, 0.38)]
        predicted, confidence = brain._predict_final_performance("cfg0", curve)
        if predicted is not None:
            assert 0.0 <= confidence <= 1.0

    def test_prediction_with_constant_data(self):
        """Constant metrics (ss_tot=0) don't crash — returns 0 confidence."""
        brain = _make_hbes(max_epochs=9, reduction_factor=3, min_early_stop_epochs=3)
        brain.bracket = "0"

        curve = [(1, 0.5), (2, 0.5), (3, 0.5)]
        predicted, confidence = brain._predict_final_performance("cfg0", curve)
        # Either fit fails (None) or confidence is 0 due to ss_tot=0
        if predicted is not None:
            assert confidence == 0.0


class TestShouldEarlyStop:
    """Test early stopping decision logic."""

    def test_no_stop_when_already_stopped(self):
        """Configs already in early_stopped_configs are skipped."""
        brain = _make_hbes()
        brain.early_stopped_configs.add("cfg0")
        assert brain._should_early_stop("cfg0", 0.5) is False

    def test_no_stop_with_insufficient_data_points(self):
        """No stopping until min_early_stop_epochs data points collected."""
        brain = _make_hbes(min_early_stop_epochs=5)
        for epoch in range(1, 5):
            result = brain._should_early_stop("cfg0", 0.5 - epoch * 0.01)
            assert result is False
        assert len(brain.learning_curves["cfg0"]) == 4

    def test_no_stop_without_comparison_baseline(self):
        """No stopping when no other configs exist to compare against."""
        brain = _make_hbes(min_early_stop_epochs=3)
        brain.bracket = "0"
        for epoch in range(1, 6):
            result = brain._should_early_stop("cfg0", 0.5 - epoch * 0.01)
        assert result is False, "Cannot stop without comparison baseline"

    def test_learning_curve_uses_observation_count(self):
        """Verify data points use observation count (not epoch) as x-axis."""
        brain = _make_hbes(min_early_stop_epochs=10)
        brain._should_early_stop("cfg0", 0.5)
        brain._should_early_stop("cfg0", 0.45)
        brain._should_early_stop("cfg1", 0.6)

        assert len(brain.learning_curves["cfg0"]) == 2
        assert brain.learning_curves["cfg0"][0] == (1, 0.5)
        assert brain.learning_curves["cfg0"][1] == (2, 0.45)
        assert len(brain.learning_curves["cfg1"]) == 1
        assert brain.learning_curves["cfg1"][0] == (1, 0.6)

    def test_duplicate_results_skipped(self):
        """Same result value is not added as a new observation."""
        brain = _make_hbes(min_early_stop_epochs=10)
        brain._should_early_stop("cfg0", 0.5)
        brain._should_early_stop("cfg0", 0.5)
        brain._should_early_stop("cfg0", 0.45)
        brain._should_early_stop("cfg0", 0.45)
        assert len(brain.learning_curves["cfg0"]) == 2

    def test_early_stop_on_clearly_worse_config_lower_is_better(self):
        """A config predicted to be much worse than best is stopped (loss metric)."""
        brain = _make_hbes(
            min_early_stop_epochs=3, early_stop_threshold=0.0, metric="loss"
        )
        brain.bracket = "0"

        brain.learning_curves["good_cfg"] = [(1, 0.3), (2, 0.25), (3, 0.22)]

        with patch.object(brain, '_predict_final_performance',
                          return_value=(0.8, 0.95)):
            brain.learning_curves["bad_cfg"] = [(1, 0.7), (2, 0.72)]
            result = brain._should_early_stop("bad_cfg", 0.75)

        # predicted 0.8 > 0.22 * 1.05 = 0.231 → should stop
        assert result is True
        assert "bad_cfg" in brain.early_stopped_configs

    def test_no_stop_when_config_is_competitive(self):
        """A config predicted close to best is NOT stopped."""
        brain = _make_hbes(
            min_early_stop_epochs=3, early_stop_threshold=0.0, metric="loss"
        )
        brain.bracket = "0"

        brain.learning_curves["other_cfg"] = [(1, 0.30), (2, 0.28), (3, 0.26)]

        with patch.object(brain, '_predict_final_performance',
                          return_value=(0.25, 0.95)):
            brain.learning_curves["test_cfg"] = [(1, 0.32), (2, 0.29)]
            result = brain._should_early_stop("test_cfg", 0.27)

        # predicted 0.25 < 0.26 * 1.05 = 0.273 → should NOT stop
        assert result is False

    def test_early_stop_higher_is_better(self):
        """Early stopping works correctly when higher metric is better."""
        brain = _make_hbes(
            min_early_stop_epochs=3, early_stop_threshold=0.0, metric="accuracy"
        )
        brain.reverse_sort = True
        brain.bracket = "0"

        brain.learning_curves["good_cfg"] = [(1, 0.80), (2, 0.85), (3, 0.88)]

        with patch.object(brain, '_predict_final_performance',
                          return_value=(0.60, 0.95)):
            brain.learning_curves["bad_cfg"] = [(1, 0.50), (2, 0.52)]
            result = brain._should_early_stop("bad_cfg", 0.53)

        # predicted 0.60 < 0.88 * 0.95 = 0.836 → should stop
        assert result is True

    def test_low_confidence_prevents_early_stop(self):
        """No stopping when prediction confidence is below threshold."""
        brain = _make_hbes(
            min_early_stop_epochs=3, early_stop_threshold=0.9, metric="loss"
        )
        brain.bracket = "0"

        brain.learning_curves["other_cfg"] = [(1, 0.20), (2, 0.18), (3, 0.17)]

        with patch.object(brain, '_predict_final_performance',
                          return_value=(0.9, 0.5)):
            brain.learning_curves["test_cfg"] = [(1, 0.7), (2, 0.68)]
            result = brain._should_early_stop("test_cfg", 0.65)

        # confidence 0.5 < threshold 0.9 → should NOT stop
        assert result is False


class TestGenerateRecommendations:
    """Test the overridden generate_recommendations with early stopping."""

    def test_running_config_checked_for_early_stop(self):
        """Running configs with non-zero results are evaluated for stopping."""
        brain = _make_hbes(min_early_stop_epochs=3)

        rec_running = _make_rec("cfg0", JobStates.running, 0.45)
        rec_running.job_id = "job_cfg0"
        rec_done = _make_rec("cfg1", JobStates.success, 0.30)
        rec_done.job_id = "job_cfg1"
        history = [rec_running, rec_done]

        with patch.object(HyperBandES.__bases__[0], 'generate_recommendations',
                          return_value=[]):
            with patch.object(brain, '_should_early_stop', return_value=False) as mock_stop:
                brain.generate_recommendations(history)
                mock_stop.assert_called_once_with("cfg0", 0.45)

    def test_early_stopped_config_added_to_cancel_set(self):
        """Config that should be stopped is added to configs_to_cancel."""
        brain = _make_hbes(min_early_stop_epochs=3)

        rec = _make_rec("cfg0", JobStates.running, 0.45)
        rec.job_id = "job_cfg0"
        history = [rec]

        with patch.object(HyperBandES.__bases__[0], 'generate_recommendations',
                          return_value=[]):
            with patch.object(brain, '_should_early_stop', return_value=True):
                brain.generate_recommendations(history)
                assert "job_cfg0" in brain.configs_to_cancel

    def test_early_stopped_configs_appear_as_failure_to_parent(self):
        """Early-stopped configs are passed to parent as failures to unblock the barrier."""
        brain = _make_hbes(min_early_stop_epochs=3)
        brain.early_stopped_configs.add("cfg0")

        rec = _make_rec("cfg0", JobStates.running, 0.45)
        rec.job_id = "job_cfg0"
        history = [rec]

        parent_received_history = []

        def capture_parent_call(hist):
            parent_received_history.extend(hist)
            return []

        with patch.object(HyperBandES.__bases__[0], 'generate_recommendations',
                          side_effect=capture_parent_call):
            brain.generate_recommendations(history)

        assert len(parent_received_history) == 1
        assert parent_received_history[0].status == JobStates.failure

    def test_zero_result_not_checked(self):
        """Running configs with result == 0.0 are skipped (no metrics yet)."""
        brain = _make_hbes()

        rec = _make_rec("cfg0", JobStates.running, 0.0)
        rec.job_id = "job_cfg0"
        history = [rec]

        with patch.object(HyperBandES.__bases__[0], 'generate_recommendations',
                          return_value=[]):
            with patch.object(brain, '_should_early_stop') as mock_stop:
                brain.generate_recommendations(history)
                mock_stop.assert_not_called()

    def test_non_running_configs_not_checked(self):
        """Only running configs are evaluated for early stopping."""
        brain = _make_hbes()

        rec_success = _make_rec("cfg0", JobStates.success, 0.30)
        rec_success.job_id = "job0"
        rec_failure = _make_rec("cfg1", JobStates.failure, 0.50)
        rec_failure.job_id = "job1"
        rec_pending = _make_rec("cfg2", JobStates.pending, 0.0)
        rec_pending.job_id = "job2"
        history = [rec_success, rec_failure, rec_pending]

        with patch.object(HyperBandES.__bases__[0], 'generate_recommendations',
                          return_value=[]):
            with patch.object(brain, '_should_early_stop') as mock_stop:
                brain.generate_recommendations(history)
                mock_stop.assert_not_called()

    def test_parent_recommendations_returned(self):
        """Parent class recommendations are passed through unchanged."""
        brain = _make_hbes()
        expected_recs = [{"learning_rate": 0.005}]

        with patch.object(HyperBandES.__bases__[0], 'generate_recommendations',
                          return_value=expected_recs):
            result = brain.generate_recommendations([])
            assert result == expected_recs

    def test_configs_to_cancel_reset_each_call(self):
        """configs_to_cancel is cleared at the start of each call."""
        brain = _make_hbes()
        brain.configs_to_cancel = {"old_job_1"}

        with patch.object(HyperBandES.__bases__[0], 'generate_recommendations',
                          return_value=[]):
            brain.generate_recommendations([])

        assert brain.configs_to_cancel == set()


class TestStatePersistence:
    """Test save_state and load_state for ES-specific data."""

    def test_save_state_includes_es_data(self):
        """save_state persists learning_curves and early_stopped_configs."""
        brain = _make_hbes()
        brain.learning_curves = {"cfg0": [(1, 0.5), (2, 0.4)]}
        brain.early_stopped_configs = {"cfg1", "cfg2"}

        base_state = {
            "bracket": "0", "sh_iter": 0, "expt_iter": 0,
            "complete": False, "epoch_number": 0
        }
        saved_payloads = []

        def capture_save(job_id, state_dict):
            saved_payloads.append((job_id, dict(state_dict)))

        with patch.object(HyperBandES.__bases__[0], 'save_state'), \
             patch(f'{PATCH_PREFIX}.hyperband_es.get_automl_brain_info',
                   return_value=dict(base_state)), \
             patch('nvidia_tao_core.microservices.utils.stateless_handler_utils.save_automl_brain_info',
                   side_effect=capture_save):
            brain.save_state()

        assert len(saved_payloads) == 1
        saved_job_id, saved_dict = saved_payloads[0]
        assert saved_job_id == brain.job_context.id
        assert saved_dict["learning_curves"] == {"cfg0": [(1, 0.5), (2, 0.4)]}
        assert set(saved_dict["early_stopped_configs"]) == {"cfg1", "cfg2"}

    def test_load_state_fresh(self):
        """load_state with no existing brain creates a fresh instance."""
        with patch(f'{PATCH_PREFIX}.hyperband_es.get_automl_brain_info',
                   return_value=None):
            brain = _make_hbes.__wrapped__ if hasattr(_make_hbes, '__wrapped__') else None
            # Use load_state directly
            with patch(f'{PATCH_PREFIX}.hyperband.save_job_specs'), \
                 patch(f'{PATCH_PREFIX}.hyperband.get_job_specs',
                       return_value={"training_config": {"num_epochs": 10}}), \
                 patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_automl_custom_param_ranges',
                       return_value={}), \
                 patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_job_specs',
                       return_value={}):
                job_context = Mock()
                job_context.id = "job_load_fresh"
                job_context.handler_id = "exp_load_fresh"
                brain = HyperBandES.load_state(
                    job_context=job_context,
                    root="/path/to/root/subdir",
                    network="image_classification",
                    parameters=[{"parameter": "learning_rate"}],
                    max_epochs=9,
                    reduction_factor=3,
                    epoch_multiplier=1,
                )
                assert isinstance(brain, HyperBandES)
                assert brain.learning_curves == {}
                assert brain.early_stopped_configs == set()

    def test_load_state_restores_es_data(self):
        """load_state restores learning_curves and early_stopped_configs."""
        saved_state = {
            "bracket": "0",
            "sh_iter": 1,
            "expt_iter": 2,
            "complete": False,
            "epoch_number": 3,
            "learning_curves": {"cfg0": [(1, 0.5), (2, 0.4)]},
            "early_stopped_configs": ["cfg1"]
        }
        with patch(f'{PATCH_PREFIX}.hyperband_es.get_automl_brain_info',
                   return_value=saved_state), \
             patch(f'{PATCH_PREFIX}.hyperband.save_job_specs'), \
             patch(f'{PATCH_PREFIX}.hyperband.get_job_specs',
                   return_value={"training_config": {"num_epochs": 10}}), \
             patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_automl_custom_param_ranges',
                   return_value={}), \
             patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_job_specs',
                   return_value={}):
            job_context = Mock()
            job_context.id = "job_load_restore"
            job_context.handler_id = "exp_load_restore"
            brain = HyperBandES.load_state(
                job_context=job_context,
                root="/path/to/root/subdir",
                network="image_classification",
                parameters=[{"parameter": "learning_rate"}],
                max_epochs=9,
                reduction_factor=3,
                epoch_multiplier=1,
            )
            assert brain.bracket == "0"
            assert brain.sh_iter == 1
            assert brain.expt_iter == 2
            assert brain.epoch_number == 3
            assert brain.learning_curves == {"cfg0": [(1, 0.5), (2, 0.4)]}
            assert brain.early_stopped_configs == {"cfg1"}

    def test_load_state_without_es_keys(self):
        """load_state works when saved state has no ES-specific keys (upgrade path)."""
        saved_state = {
            "bracket": "0",
            "sh_iter": 0,
            "expt_iter": 0,
            "complete": False,
            "epoch_number": 0,
        }
        with patch(f'{PATCH_PREFIX}.hyperband_es.get_automl_brain_info',
                   return_value=saved_state), \
             patch(f'{PATCH_PREFIX}.hyperband.save_job_specs'), \
             patch(f'{PATCH_PREFIX}.hyperband.get_job_specs',
                   return_value={"training_config": {"num_epochs": 10}}), \
             patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_automl_custom_param_ranges',
                   return_value={}), \
             patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_job_specs',
                   return_value={}):
            job_context = Mock()
            job_context.id = "job_load_no_es"
            job_context.handler_id = "exp_load_no_es"
            brain = HyperBandES.load_state(
                job_context=job_context,
                root="/path/to/root/subdir",
                network="image_classification",
                parameters=[{"parameter": "learning_rate"}],
                max_epochs=9,
                reduction_factor=3,
                epoch_multiplier=1,
            )
            assert brain.learning_curves == {}
            assert brain.early_stopped_configs == set()


class TestEndToEndEarlyStopFlow:
    """Integration-style tests for the full early stopping flow."""

    def test_full_early_stop_scenario(self):
        """Simulate a multi-observation flow where one config gets stopped."""
        brain = _make_hbes(
            min_early_stop_epochs=3, early_stop_threshold=0.0, metric="loss"
        )
        brain.bracket = "0"

        # Simulate good config building a learning curve
        brain.learning_curves["good"] = [
            (1, 0.40), (2, 0.32), (3, 0.27), (4, 0.24), (5, 0.22)
        ]

        # Simulate bad config — gradually feed data points
        brain._should_early_stop("bad", 0.90)
        brain._should_early_stop("bad", 0.88)

        # At observation 3 we have enough points; mock prediction to return poor result
        with patch.object(brain, '_predict_final_performance',
                          return_value=(0.75, 0.95)):
            stopped = brain._should_early_stop("bad", 0.85)

        # predicted 0.75 > 0.22 * 1.05 = 0.231 → stop
        assert stopped is True
        assert "bad" in brain.early_stopped_configs

        # Subsequent calls for same config return False (already stopped)
        assert brain._should_early_stop("bad", 0.80) is False

    def test_multiple_configs_independent_tracking(self):
        """Each config has its own independent learning curve with observation counts."""
        brain = _make_hbes(min_early_stop_epochs=10)

        for epoch in range(1, 6):
            brain._should_early_stop("cfg_a", 0.5 - epoch * 0.02)
            brain._should_early_stop("cfg_b", 0.8 - epoch * 0.01)
            brain._should_early_stop("cfg_c", 0.3 - epoch * 0.03)

        assert len(brain.learning_curves["cfg_a"]) == 5
        assert len(brain.learning_curves["cfg_b"]) == 5
        assert len(brain.learning_curves["cfg_c"]) == 5
        # Observation counts should be sequential 1..5
        assert [x for x, _ in brain.learning_curves["cfg_a"]] == [1, 2, 3, 4, 5]

    def test_margin_boundary_lower_is_better(self):
        """Config exactly at the 5% margin boundary is NOT stopped."""
        brain = _make_hbes(
            min_early_stop_epochs=3, early_stop_threshold=0.0, metric="loss"
        )
        brain.bracket = "0"

        # Best other config: 0.40
        brain.learning_curves["other"] = [(1, 0.45), (2, 0.42), (3, 0.40)]

        # predicted = 0.40 * 1.05 = 0.42 exactly at boundary
        with patch.object(brain, '_predict_final_performance',
                          return_value=(0.42, 0.99)):
            brain.learning_curves["test"] = [(1, 0.50), (2, 0.47)]
            result = brain._should_early_stop("test", 0.44)

        # 0.42 == 0.40 * 1.05 → NOT strictly greater → should NOT stop
        assert result is False

    def test_margin_boundary_higher_is_better(self):
        """Config exactly at the 5% margin boundary is NOT stopped (higher is better)."""
        brain = _make_hbes(
            min_early_stop_epochs=3, early_stop_threshold=0.0, metric="accuracy"
        )
        brain.reverse_sort = True
        brain.bracket = "0"

        # Best other config: 0.80
        brain.learning_curves["other"] = [(1, 0.70), (2, 0.75), (3, 0.80)]

        # predicted = 0.80 * 0.95 = 0.76 exactly at boundary
        with patch.object(brain, '_predict_final_performance',
                          return_value=(0.76, 0.99)):
            brain.learning_curves["test"] = [(1, 0.65), (2, 0.68)]
            result = brain._should_early_stop("test", 0.70)

        # 0.76 == 0.80 * 0.95 → NOT strictly less → should NOT stop
        assert result is False
