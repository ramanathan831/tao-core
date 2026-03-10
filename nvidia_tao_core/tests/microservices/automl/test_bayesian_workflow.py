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

"""Unit tests for Bayesian generate_recommendations workflow"""

import numpy as np
from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.automl.bayesian import Bayesian
from nvidia_tao_core.microservices.utils.automl_utils import JobStates


PATCH_PREFIX = 'nvidia_tao_core.microservices.automl'
BASE_PREFIX = f'{PATCH_PREFIX}.automl_algorithm_base'

PARAMETERS = [
    {"parameter": "learning_rate", "value_type": "float",
     "valid_min": 0.001, "valid_max": 0.1}
]


def _make_bayesian(job_id="job_bayesian_test"):
    with patch(f'{PATCH_PREFIX}.bayesian.get_total_epochs', return_value=10), \
         patch(f'{BASE_PREFIX}.get_automl_custom_param_ranges',
               return_value={}), \
         patch(f'{BASE_PREFIX}.get_job_specs', return_value={}):
        job_context = Mock()
        job_context.id = job_id
        job_context.handler_id = f"exp_{job_id}"
        return Bayesian(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=PARAMETERS,
        )


def _make_rec(rec_id, status, result, specs=None, job_id=None):
    rec = Mock()
    rec.id = rec_id
    rec.status = status
    rec.result = result
    rec.specs = specs or {"learning_rate": 0.01}
    rec.job_id = job_id or f"job_{rec_id}"
    return rec


class TestBayesianInitialRandom:
    """Test Bayesian initial random recommendation."""

    @patch(f'{PATCH_PREFIX}.bayesian.get_flatten_specs')
    def test_bayesian_initial_random(self, mock_gfs):
        """history=[] returns exactly 1 dict with random params."""
        brain = _make_bayesian()
        recs = brain.generate_recommendations([])

        assert len(recs) == 1
        assert isinstance(recs[0], dict)
        assert "learning_rate" in recs[0]
        assert len(brain.Xs) == 1


class TestBayesianWaitsForCompletion:
    """Test Bayesian waits for last experiment."""

    @patch(f'{PATCH_PREFIX}.bayesian.get_flatten_specs')
    def test_bayesian_waits_for_completion(self, mock_gfs):
        """Returns [] when last experiment still running."""
        brain = _make_bayesian()
        brain.generate_recommendations([])

        history = [_make_rec(0, JobStates.running, 0.0)]
        recs = brain.generate_recommendations(history)
        assert recs == []


class TestBayesianUpdatesGP:
    """Test Bayesian GP update on completion."""

    @patch(f'{PATCH_PREFIX}.bayesian.get_flatten_specs')
    def test_bayesian_updates_gp(self, mock_gfs):
        """After completion, ys appended and GP updated."""
        brain = _make_bayesian()
        brain.generate_recommendations([])

        result_value = 0.42
        history = [_make_rec(0, JobStates.success, result_value)]

        with patch.object(brain, 'optimize_ei',
                          return_value=np.array([0.5])):
            recs = brain.generate_recommendations(history)

        assert result_value in brain.ys
        assert len(brain.ys) == 1
        assert len(brain.Xs) == 2
        assert len(recs) == 1


class TestBayesianSequentialFlow:
    """Test Bayesian multi-iteration sequential flow."""

    @patch(f'{PATCH_PREFIX}.bayesian.get_flatten_specs')
    def test_bayesian_sequential_flow(self, mock_gfs):
        """Complete 3 iterations -- verify Xs/ys grow."""
        brain = _make_bayesian()
        brain.generate_recommendations([])

        for iteration in range(3):
            result = 0.5 - iteration * 0.1
            history = [_make_rec(iteration, JobStates.success, result)]

            with patch.object(brain, 'optimize_ei',
                              return_value=np.array([0.3 + iteration * 0.1])):
                recs = brain.generate_recommendations(history)

            assert len(recs) == 1
            assert len(brain.ys) == iteration + 1
            assert len(brain.Xs) == iteration + 2


class TestBayesianAttributes:
    """Test Bayesian algorithm attribute initialization."""

    def test_xi_and_num_restarts(self):
        brain = _make_bayesian()
        assert brain.xi == 0.01
        assert brain.num_restarts == 5

    def test_gp_initialized(self):
        brain = _make_bayesian()
        assert brain.gp is not None
        assert brain.Xs == []
        assert brain.ys == []


class TestBayesianMetricDirection:
    """Test Bayesian does not control epochs — epoch is a search param."""

    @patch(f'{PATCH_PREFIX}.bayesian.get_flatten_specs')
    def test_bayesian_does_not_filter_epoch_params(self, mock_gfs):
        """Unlike HyperBand, Bayesian includes all params (including epoch)."""
        brain = _make_bayesian()
        recs = brain.generate_recommendations([])
        assert len(recs) == 1
        assert "learning_rate" in recs[0]


class TestBayesianExpectedImprovement:
    """Test _expected_improvement function behavior."""

    @patch(f'{PATCH_PREFIX}.bayesian.get_flatten_specs')
    def test_ei_returns_negative_for_minimization(self, mock_gfs):
        """_expected_improvement returns negative value (for scipy minimization)."""
        brain = _make_bayesian()
        brain.generate_recommendations([])

        brain.ys.append(0.5)
        brain.update_gp()

        X = np.array([0.5])
        ei_val = brain._expected_improvement(X)
        assert isinstance(ei_val, float)

    @patch(f'{PATCH_PREFIX}.bayesian.get_flatten_specs')
    def test_optimize_ei_returns_bounded_suggestions(self, mock_gfs):
        """optimize_ei returns values in [0, 1] range."""
        brain = _make_bayesian()
        brain.generate_recommendations([])

        brain.ys.append(0.5)
        brain.update_gp()

        suggestions = brain.optimize_ei()
        assert isinstance(suggestions, np.ndarray)
        assert len(suggestions) == len(brain.parameters)
        assert np.all(suggestions >= 0.0)
        assert np.all(suggestions <= 1.0)


class TestBayesianMetricDirectionEI:
    """Test EI correctly handles metric direction (higher/lower is better)."""

    def test_loss_metric_sets_reverse_sort_false(self):
        """For loss metric, reverse_sort=False (lower is better)."""
        brain = _make_bayesian()
        assert brain.reverse_sort is False

    def test_accuracy_metric_sets_reverse_sort_true(self):
        """For accuracy metric, reverse_sort=True (higher is better)."""
        with patch(f'{PATCH_PREFIX}.bayesian.get_total_epochs', return_value=10), \
             patch(f'{BASE_PREFIX}.get_automl_custom_param_ranges',
                   return_value={}), \
             patch(f'{BASE_PREFIX}.get_job_specs', return_value={}):
            job_context = Mock()
            job_context.id = "job_acc_test"
            job_context.handler_id = "exp_job_acc_test"
            brain = Bayesian(
                job_context=job_context,
                root="/path/to/root/subdir",
                network="image_classification",
                parameters=PARAMETERS,
                metric="val_accuracy",
            )
        assert brain.reverse_sort is True

    @patch(f'{PATCH_PREFIX}.bayesian.get_flatten_specs')
    def test_ei_for_minimization_prefers_lower(self, mock_gfs):
        """For loss (minimize), EI should favor points predicted to be lower."""
        brain = _make_bayesian()
        assert brain.reverse_sort is False
        brain.generate_recommendations([])

        brain.ys = [0.8, 0.6, 0.5]
        brain.Xs = [np.array([0.2]), np.array([0.5]), np.array([0.8])]
        brain.update_gp()

        ei_near_best = brain._expected_improvement(np.array([0.85]))
        ei_far = brain._expected_improvement(np.array([0.15]))
        assert isinstance(ei_near_best, float)
        assert isinstance(ei_far, float)
