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

"""Unit tests for BOHB generate_recommendations workflow"""

import numpy as np
from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.automl.bohb import BOHB
from nvidia_tao_core.microservices.utils.automl_utils import ResumeRecommendation, JobStates


PATCH_PREFIX = 'nvidia_tao_core.microservices.automl'

PARAMETERS = [
    {"parameter": "learning_rate", "value_type": "float",
     "valid_min": 0.001, "valid_max": 0.1}
]


def _make_bohb(job_id="job_bohb_test", max_epochs=9, reduction_factor=3,
               epoch_multiplier=1, metric="loss"):
    with patch(f'{PATCH_PREFIX}.bohb.save_job_specs'), \
         patch(f'{PATCH_PREFIX}.bohb.get_job_specs',
               return_value={"training_config": {"num_epochs": 10}}), \
         patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_automl_custom_param_ranges',
               return_value={}), \
         patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_job_specs',
               return_value={}):
        job_context = Mock()
        job_context.id = job_id
        job_context.handler_id = f"exp_{job_id}"
        return BOHB(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=list(PARAMETERS),
            max_epochs=max_epochs,
            reduction_factor=reduction_factor,
            epoch_multiplier=epoch_multiplier,
            metric=metric,
        )


def _make_rec(rec_id, status, result, specs=None, job_id=None):
    rec = Mock()
    rec.id = rec_id
    rec.status = status
    rec.result = result
    rec.specs = specs if specs is not None else {"learning_rate": 0.05}
    rec.job_id = job_id if job_id is not None else f"job_{rec_id}"
    return rec


def _build_completed_history(count, results):
    history = []
    for i in range(count):
        specs = {"learning_rate": 0.001 + i * 0.01}
        history.append(_make_rec(i, JobStates.success, results[i], specs=dict(specs), job_id=f"job_{i}"))
    return history


class TestBOHBWorkflow:
    """Test generate_recommendations workflow for BOHB."""

    def test_bohb_bracket_structure(self):
        """Verify ni/ri brackets for max_epochs=9, reduction_factor=3."""
        brain = _make_bohb()
        assert brain.ni["0"] == [9, 3, 1]
        assert brain.ri["0"] == [1, 3, 9]
        assert brain.ni["1"] == [5, 1]
        assert brain.ri["1"] == [3, 9]
        assert brain.ni["2"] == [3]
        assert brain.ri["2"] == [9]

    def test_bohb_initial_launch(self):
        """First call with empty history returns ni[0][0] new config dicts."""
        brain = _make_bohb()
        fixed_config = {"learning_rate": 0.042}
        with patch.object(brain, '_tpe_suggest', return_value=np.array([0.5])), \
             patch.object(brain, '_generate_parameters_from_suggestions',
                          return_value=fixed_config), \
             patch(f'{PATCH_PREFIX}.bohb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs = brain.generate_recommendations([])

        assert len(recs) == 9
        for r in recs:
            assert isinstance(r, dict)
            assert r["learning_rate"] == 0.042
        assert brain.last_launched_count == 9

    def test_bohb_waits_for_completion(self):
        """Returns [] when experiments are still running."""
        brain = _make_bohb()
        brain.sh_iter = 0
        brain.expt_iter = 9
        brain.last_launched_count = 9

        running = [_make_rec(i, JobStates.running, 0.0) for i in range(9)]
        with patch(f'{PATCH_PREFIX}.bohb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs = brain.generate_recommendations(running)

        assert recs == []

    def test_bohb_promotes_best_configs(self):
        """After all configs complete, top-k are promoted as ResumeRecommendations."""
        brain = _make_bohb()
        brain.sh_iter = 0
        brain.expt_iter = 9
        brain.last_launched_count = 9

        results = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        history = _build_completed_history(9, results)

        with patch(f'{PATCH_PREFIX}.bohb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs = brain.generate_recommendations(history)

        assert len(recs) == 3
        for r in recs:
            assert isinstance(r, ResumeRecommendation)
        promoted_ids = {r.id for r in recs}
        assert promoted_ids == {8, 7, 6}

    def test_bohb_complete_bracket_flow(self):
        """Full bracket 0: launch 9 -> promote 3 -> promote 1 -> bracket done."""
        brain = _make_bohb()
        fixed_config = {"learning_rate": 0.05}

        with patch.object(brain, '_tpe_suggest', return_value=np.array([0.5])), \
             patch.object(brain, '_generate_parameters_from_suggestions',
                          return_value=fixed_config), \
             patch(f'{PATCH_PREFIX}.bohb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs_1 = brain.generate_recommendations([])
        assert len(recs_1) == 9

        results_rung0 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        history = _build_completed_history(9, results_rung0)
        with patch(f'{PATCH_PREFIX}.bohb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs_2 = brain.generate_recommendations(history)
        assert len(recs_2) == 3
        assert all(isinstance(r, ResumeRecommendation) for r in recs_2)
        promoted_3_ids = {r.id for r in recs_2}
        assert promoted_3_ids == {8, 7, 6}

        for rec in history:
            if rec.id in promoted_3_ids:
                rec.result = rec.result * 0.5
        with patch(f'{PATCH_PREFIX}.bohb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs_3 = brain.generate_recommendations(history)
        assert len(recs_3) == 1
        assert isinstance(recs_3[0], ResumeRecommendation)
        assert recs_3[0].id == 8

        for rec in history:
            if rec.id == 8:
                rec.result = 0.01
        with patch.object(brain, '_tpe_suggest', return_value=np.array([0.5])), \
             patch.object(brain, '_generate_parameters_from_suggestions',
                          return_value=fixed_config), \
             patch(f'{PATCH_PREFIX}.bohb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            brain.generate_recommendations(history)
        assert brain.bracket == "1"

    def test_bohb_done_flag(self):
        """done() returns False while running, True when complete with no pending."""
        brain = _make_bohb()
        assert brain.done() is False

        brain.complete = True
        brain.last_launched_count = 5
        assert brain.done() is False

        brain.last_launched_count = 0
        assert brain.done() is True


class TestBOHBTPEAttributes:
    """Test BOHB TPE-specific attribute initialization."""

    def test_tpe_attributes_initialized(self):
        brain = _make_bohb()
        assert brain.num_samples == 64
        assert brain.min_points_in_model == 10
        assert brain.quantile == 0.15
        assert brain.budget_observations == {}

    def test_custom_tpe_attributes(self):
        with patch(f'{PATCH_PREFIX}.bohb.save_job_specs'), \
             patch(f'{PATCH_PREFIX}.bohb.get_job_specs',
                   return_value={"training_config": {"num_epochs": 10}}), \
             patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_automl_custom_param_ranges',
                   return_value={}), \
             patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_job_specs',
                   return_value={}):
            from unittest.mock import Mock
            job_context = Mock()
            job_context.id = "job_bohb_custom"
            job_context.handler_id = "exp_bohb_custom"
            brain = BOHB(
                job_context=job_context,
                root="/path/to/root/subdir",
                network="image_classification",
                parameters=list(PARAMETERS),
                max_epochs=9,
                reduction_factor=3,
                epoch_multiplier=1,
                kde_samples=128,
                top_n_percent=20.0,
                min_points_in_model=5,
            )
        assert brain.num_samples == 128
        assert brain.min_points_in_model == 5
        assert brain.quantile == 0.20


class TestBOHBObservationEncoding:
    """Test _normalize_value_to_observation encodes all param types correctly."""

    def test_float_roundtrip(self):
        brain = _make_bohb()
        param = {"parameter": "learning_rate", "value_type": "float",
                 "valid_min": 0.001, "valid_max": 0.1}
        encoded = brain._normalize_value_to_observation(param, 0.0505)
        assert 0.0 <= encoded <= 1.0
        import numpy as np
        log_min, log_max = np.log10(0.001), np.log10(0.1)
        expected = (np.log10(0.0505) - log_min) / (log_max - log_min)
        assert abs(encoded - expected) < 1e-6

    def test_int_roundtrip(self):
        brain = _make_bohb()
        param = {"parameter": "batch_size", "value_type": "int",
                 "valid_min": 8, "valid_max": 64}
        encoded = brain._normalize_value_to_observation(param, 32)
        assert 0.0 <= encoded <= 1.0
        expected = (32 - 8) / (64 - 8)
        assert abs(encoded - expected) < 1e-6

    def test_categorical_roundtrip(self):
        brain = _make_bohb()
        param = {"parameter": "optimizer", "value_type": "categorical",
                 "valid_options": ["sgd", "adam", "rmsprop"]}
        enc_sgd = brain._normalize_value_to_observation(param, "sgd")
        enc_adam = brain._normalize_value_to_observation(param, "adam")
        enc_rmsprop = brain._normalize_value_to_observation(param, "rmsprop")
        assert abs(enc_sgd - 0.5 / 3) < 1e-6
        assert abs(enc_adam - 1.5 / 3) < 1e-6
        assert abs(enc_rmsprop - 2.5 / 3) < 1e-6
        # Verify decode consistency: int(encoded * len) == original index
        assert int(enc_sgd * 3) == 0
        assert int(enc_adam * 3) == 1
        assert int(enc_rmsprop * 3) == 2

    def test_bool_encoding(self):
        brain = _make_bohb()
        param = {"parameter": "use_augment", "value_type": "bool"}
        enc_true = brain._normalize_value_to_observation(param, True)
        enc_false = brain._normalize_value_to_observation(param, False)
        assert enc_true == 0.75
        assert enc_false == 0.25
        # Verify decode consistency: encoded >= 0.5 matches original
        assert (enc_true >= 0.5) is True
        assert (enc_false >= 0.5) is False

    def test_none_value_defaults(self):
        brain = _make_bohb()
        for vt in ["float", "int", "categorical", "bool"]:
            param = {"parameter": "x", "value_type": vt, "valid_min": 0, "valid_max": 1}
            assert brain._normalize_value_to_observation(param, None) == 0.5

    def test_not_constant_across_types(self):
        """Different values for non-float types should produce different encodings."""
        brain = _make_bohb()
        param = {"parameter": "batch_size", "value_type": "int",
                 "valid_min": 8, "valid_max": 64}
        enc_8 = brain._normalize_value_to_observation(param, 8)
        enc_64 = brain._normalize_value_to_observation(param, 64)
        assert enc_8 != enc_64


class TestBOHBTPEFallbackToRandom:
    """Test BOHB falls back to random when insufficient observations."""

    def test_tpe_random_when_below_min_points(self):
        """_tpe_suggest returns random array when observations < min_points_in_model."""
        brain = _make_bohb()
        brain.budget_observations = {1: [(np.array([0.5]), 0.3)]}
        np.random.seed(42)
        result = brain._tpe_suggest(budget=1)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(brain.parameters)

    def test_tpe_random_when_empty_observations(self):
        brain = _make_bohb()
        result = brain._tpe_suggest(budget=1)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(brain.parameters)


class TestBOHBBuildKDE:
    """Test KDE building logic."""

    def test_build_kde_returns_none_below_threshold(self):
        brain = _make_bohb()
        data = np.array([[0.1], [0.2]])
        kde = brain._build_kde(data)
        assert kde is None

    def test_build_kde_succeeds_with_enough_data(self):
        brain = _make_bohb()
        np.random.seed(42)
        data = np.random.rand(15, 1)
        kde = brain._build_kde(data)
        assert kde is not None

    def test_sample_from_kde_clipped(self):
        brain = _make_bohb()
        np.random.seed(42)
        data = np.random.rand(15, 1)
        kde = brain._build_kde(data)
        samples = brain._sample_from_kde(kde, 10)
        assert samples is not None
        assert samples.shape == (10, 1)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_sample_from_none_kde(self):
        brain = _make_bohb()
        assert brain._sample_from_kde(None, 10) is None


class TestBOHBTPEKicksin:
    """Test BOHB switches from random to TPE after enough observations."""

    def test_tpe_uses_kde_with_enough_observations(self):
        """After min_points_in_model observations, _tpe_suggest uses KDE path."""
        brain = _make_bohb()
        # Disable random fraction so the test deterministically reaches the KDE path
        brain.random_fraction = 0.0
        np.random.seed(42)
        budget = 1
        brain.budget_observations[budget] = []
        for i in range(12):
            config = np.random.rand(len(brain.parameters))
            brain.budget_observations[budget].append((config, 0.5 + i * 0.01))

        with patch.object(brain, '_build_kde') as mock_kde, \
             patch.object(brain, '_sample_from_kde', return_value=np.random.rand(64, 1)):
            mock_kde_obj = Mock()
            mock_kde_obj.pdf = Mock(return_value=np.array([0.5]))
            mock_kde_obj.factor = 0.5
            mock_kde.return_value = mock_kde_obj
            brain._tpe_suggest(budget=budget)
            assert mock_kde.call_count >= 1
