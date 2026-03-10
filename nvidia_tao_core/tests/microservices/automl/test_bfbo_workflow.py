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

"""Unit tests for BFBO generate_recommendations workflow"""

import numpy as np
from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.automl.bfbo import BFBO
from nvidia_tao_core.microservices.utils.automl_utils import JobStates


PATCH_PREFIX = 'nvidia_tao_core.microservices.automl'
BASE_PREFIX = f'{PATCH_PREFIX}.automl_algorithm_base'

PARAMETERS = [
    {"parameter": "learning_rate", "value_type": "float",
     "valid_min": 0.001, "valid_max": 0.1}
]


def _make_bfbo(job_id="job_bfbo_test"):
    with patch(f'{PATCH_PREFIX}.bfbo.get_total_epochs', return_value=10), \
         patch(f'{BASE_PREFIX}.get_automl_custom_param_ranges',
               return_value={}), \
         patch(f'{BASE_PREFIX}.get_job_specs', return_value={}):
        job_context = Mock()
        job_context.id = job_id
        job_context.handler_id = f"exp_{job_id}"
        return BFBO(
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


class TestBFBOInitialRandom:
    """Test BFBO initial random recommendation."""

    @patch(f'{PATCH_PREFIX}.bfbo.get_flatten_specs')
    def test_bfbo_initial_random(self, mock_gfs):
        """history=[] returns exactly 1 dict."""
        brain = _make_bfbo()
        recs = brain.generate_recommendations([])

        assert len(recs) == 1
        assert isinstance(recs[0], dict)
        assert "learning_rate" in recs[0]
        assert len(brain.Xs) == 1


class TestBFBOWaitsForCompletion:
    """Test BFBO waits for last experiment."""

    @patch(f'{PATCH_PREFIX}.bfbo.get_flatten_specs')
    def test_bfbo_waits_for_completion(self, mock_gfs):
        """Returns [] when last experiment still running."""
        brain = _make_bfbo()
        brain.generate_recommendations([])

        history = [_make_rec(0, JobStates.running, 0.0)]
        recs = brain.generate_recommendations(history)
        assert recs == []


class TestBFBOKappaDecay:
    """Test BFBO kappa decay after each iteration."""

    @patch(f'{PATCH_PREFIX}.bfbo.get_flatten_specs')
    def test_bfbo_kappa_decay(self, mock_gfs):
        """After each iteration, kappa *= kappa_decay (down to kappa_min)."""
        brain = _make_bfbo()
        initial_kappa = brain.kappa
        kappa_decay = brain.kappa_decay
        kappa_min = brain.kappa_min

        brain.generate_recommendations([])

        expected_kappa = initial_kappa
        for iteration in range(10):
            history = [_make_rec(iteration, JobStates.success, 0.5 - iteration * 0.03)]

            with patch.object(brain, 'optimize_ucb',
                              return_value=np.array([0.5])):
                brain.generate_recommendations(history)

            expected_kappa = max(kappa_min, expected_kappa * kappa_decay)
            assert abs(brain.kappa - expected_kappa) < 1e-10


class TestBFBOSequentialFlow:
    """Test BFBO multi-iteration sequential flow."""

    @patch(f'{PATCH_PREFIX}.bfbo.get_flatten_specs')
    def test_bfbo_sequential_flow(self, mock_gfs):
        """Complete 3 iterations -- verify Xs/ys grow and kappa decays."""
        brain = _make_bfbo()
        initial_kappa = brain.kappa

        brain.generate_recommendations([])

        prev_kappa = initial_kappa
        for iteration in range(3):
            result = 0.5 - iteration * 0.1
            history = [_make_rec(iteration, JobStates.success, result)]

            with patch.object(brain, 'optimize_ucb',
                              return_value=np.array([0.3 + iteration * 0.1])):
                recs = brain.generate_recommendations(history)

            assert len(recs) == 1
            assert len(brain.ys) == iteration + 1
            assert len(brain.Xs) == iteration + 2
            assert brain.kappa < prev_kappa
            prev_kappa = brain.kappa


class TestBFBOAttributes:
    """Test BFBO algorithm attribute initialization."""

    def test_kappa_initial_values(self):
        brain = _make_bfbo()
        assert brain.kappa == 2.0
        assert brain.kappa_decay == 0.95
        assert brain.kappa_min == 0.5

    def test_penalization_attributes(self):
        brain = _make_bfbo()
        assert brain.local_penalization is True
        assert brain.penalization_radius == 0.1
        assert brain.num_restarts == 10


class TestBFBOUpperConfidenceBound:
    """Test _upper_confidence_bound function."""

    @patch(f'{PATCH_PREFIX}.bfbo.get_flatten_specs')
    def test_ucb_returns_negative_for_minimization(self, mock_gfs):
        """_upper_confidence_bound returns negative UCB (for scipy minimization)."""
        brain = _make_bfbo()
        brain.generate_recommendations([])

        brain.ys.append(0.5)
        brain.update_gp()

        X = np.array([0.5])
        ucb_val = brain._upper_confidence_bound(X)
        assert isinstance(ucb_val, float)

    @patch(f'{PATCH_PREFIX}.bfbo.get_flatten_specs')
    def test_ucb_with_local_penalization(self, mock_gfs):
        """UCB value changes with local penalization near existing points."""
        brain = _make_bfbo()
        brain.generate_recommendations([])

        brain.ys.append(0.5)
        brain.update_gp()

        far_point = np.array([0.9])
        near_point = brain.Xs[0]

        ucb_far = brain._upper_confidence_bound(far_point)
        ucb_near = brain._upper_confidence_bound(near_point)
        assert ucb_far != ucb_near

    @patch(f'{PATCH_PREFIX}.bfbo.get_flatten_specs')
    def test_optimize_ucb_returns_bounded(self, mock_gfs):
        """optimize_ucb returns values in [0, 1]."""
        brain = _make_bfbo()
        brain.generate_recommendations([])

        brain.ys.append(0.5)
        brain.update_gp()

        suggestions = brain.optimize_ucb()
        assert isinstance(suggestions, np.ndarray)
        assert len(suggestions) == len(brain.parameters)
        assert np.all(suggestions >= 0.0)
        assert np.all(suggestions <= 1.0)


class TestBFBOLocalPenalization:
    """Test local penalization discourages querying similar points."""

    @patch(f'{PATCH_PREFIX}.bfbo.get_flatten_specs')
    def test_penalization_at_existing_point_is_small(self, mock_gfs):
        """Penalization factor is small (near 0) when querying at an existing point."""
        brain = _make_bfbo()
        brain.generate_recommendations([])

        brain.ys.append(0.5)
        brain.update_gp()

        existing_point = np.array(brain.Xs[0])
        distances = np.linalg.norm(np.array(brain.Xs) - existing_point.reshape(1, -1), axis=1)
        penalization = np.prod(np.tanh(distances / brain.penalization_radius))
        assert penalization < 0.01
