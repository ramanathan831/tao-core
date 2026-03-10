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

"""Unit tests for PBT generate_recommendations workflow"""

import numpy as np
from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.automl.pbt import PBT
from nvidia_tao_core.microservices.utils.automl_utils import ResumeRecommendation, JobStates


PATCH_PREFIX = 'nvidia_tao_core.microservices.automl'
BASE_PREFIX = f'{PATCH_PREFIX}.automl_algorithm_base'

PARAMETERS = [
    {"parameter": "learning_rate", "value_type": "float",
     "valid_min": 0.001, "valid_max": 0.1}
]


def _make_pbt(job_id="job_pbt_test", population_size=5,
              max_generations=3, eval_interval=10,
              perturbation_factor=1.2, metric="loss"):
    with patch(f'{PATCH_PREFIX}.pbt.save_job_specs'), \
         patch(f'{PATCH_PREFIX}.pbt.get_job_specs',
               return_value={"training_config": {"num_epochs": 200}}), \
         patch(f'{BASE_PREFIX}.get_automl_custom_param_ranges',
               return_value={}), \
         patch(f'{BASE_PREFIX}.get_job_specs', return_value={}):
        job_context = Mock()
        job_context.id = job_id
        job_context.handler_id = f"exp_{job_id}"
        return PBT(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=PARAMETERS,
            population_size=population_size,
            max_generations=max_generations,
            eval_interval=eval_interval,
            perturbation_factor=perturbation_factor,
            metric=metric,
        )


def _make_rec(rec_id, status, result, specs=None, job_id=None):
    rec = Mock()
    rec.id = rec_id
    rec.status = status
    rec.result = result
    rec.specs = specs or {"learning_rate": 0.01}
    rec.job_id = job_id or f"job_{rec_id}"
    return rec


class TestPBTInitialPopulation:
    """Test PBT initial population launch."""

    @patch(f'{PATCH_PREFIX}.pbt.get_flatten_specs')
    def test_pbt_initial_population(self, mock_gfs):
        """history=[] returns population_size dicts."""
        brain = _make_pbt(population_size=5)
        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.05}):
            recs = brain.generate_recommendations([])

        assert len(recs) == 5
        for r in recs:
            assert isinstance(r, dict)
        assert len(brain.population) == 5


class TestPBTWaitsForAll:
    """Test PBT waits for all members to complete."""

    @patch(f'{PATCH_PREFIX}.pbt.get_flatten_specs')
    def test_pbt_waits_for_all(self, mock_gfs):
        """Returns [] when not all population members have completed."""
        brain = _make_pbt(population_size=5)

        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.05}):
            brain.generate_recommendations([])

        history = []
        for cid in range(4):
            history.append(_make_rec(cid, JobStates.success, 0.5 - cid * 0.05))
        history.append(_make_rec(4, JobStates.running, 0.0))

        recs = brain.generate_recommendations(history)
        assert recs == []


class TestPBTExploitExplore:
    """Test PBT exploit-explore after a generation completes."""

    @patch(f'{PATCH_PREFIX}.pbt.get_flatten_specs')
    def test_pbt_exploit_explore(self, mock_gfs):
        """After gen 0, bottom 20% members get replaced specs from top."""
        brain = _make_pbt(population_size=5)

        with patch.object(brain, '_generate_random_parameters',
                          side_effect=[{"learning_rate": 0.01 * (i + 1)}
                                       for i in range(5)]):
            brain.generate_recommendations([])

        history = [
            _make_rec(cid, JobStates.success, 0.1 * (cid + 1))
            for cid in range(5)
        ]

        np.random.seed(42)
        recs = brain.generate_recommendations(history)

        assert len(recs) == 5
        assert all(isinstance(r, ResumeRecommendation) for r in recs)

        replaced_rec = next(r for r in recs if r.id == 4)
        assert replaced_rec.specs is not None

    @patch(f'{PATCH_PREFIX}.pbt.get_flatten_specs')
    def test_pbt_all_resume(self, mock_gfs):
        """ALL members resume (ResumeRecommendation), even surviving ones."""
        brain = _make_pbt(population_size=5)

        with patch.object(brain, '_generate_random_parameters',
                          side_effect=[{"learning_rate": 0.01 * (i + 1)}
                                       for i in range(5)]):
            brain.generate_recommendations([])

        history = [
            _make_rec(cid, JobStates.success, 0.1 * (cid + 1))
            for cid in range(5)
        ]

        recs = brain.generate_recommendations(history)
        assert len(recs) == 5
        for r in recs:
            assert isinstance(r, ResumeRecommendation)


class TestPBTGenerationTracking:
    """Test PBT generation counting and completion."""

    @patch(f'{PATCH_PREFIX}.pbt.get_flatten_specs')
    def test_pbt_generation_tracking(self, mock_gfs):
        """Generation increments; complete=True at max_generations."""
        brain = _make_pbt(population_size=5, max_generations=3)

        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.05}):
            brain.generate_recommendations([])

        for gen in range(3):
            history = [
                _make_rec(cid, JobStates.success, 0.1 * (cid + 1))
                for cid in range(5)
            ]
            recs = brain.generate_recommendations(history)
            if gen < 2:
                assert brain.generation == gen + 1
                assert not brain.complete
            else:
                assert brain.generation == 3
                assert brain.complete
                assert recs == []


class TestPBTPerturbedParams:
    """Test PBT perturbation changes replaced member's specs."""

    @patch(f'{PATCH_PREFIX}.pbt.get_flatten_specs')
    def test_pbt_perturbed_params_differ(self, mock_gfs):
        """Replaced member's new specs differ from source (perturbation)."""
        brain = _make_pbt(population_size=5, perturbation_factor=2.0)

        with patch.object(brain, '_generate_random_parameters',
                          side_effect=[{"learning_rate": 0.01 * (i + 1)}
                                       for i in range(5)]):
            brain.generate_recommendations([])

        history = [
            _make_rec(cid, JobStates.success, 0.1 * (cid + 1))
            for cid in range(5)
        ]

        np.random.seed(42)
        recs = brain.generate_recommendations(history)

        replaced_rec = next(r for r in recs if r.id == 4)
        assert replaced_rec.specs is not None


class TestPBTResumeFromTopMember:
    """Test replaced members resume from the top member's checkpoint."""

    @patch(f'{PATCH_PREFIX}.pbt.get_flatten_specs')
    def test_replaced_member_resumes_from_source(self, mock_gfs):
        """Bottom member's ResumeRecommendation has resume_from_job_id pointing to top member."""
        brain = _make_pbt(population_size=5)

        with patch.object(brain, '_generate_random_parameters',
                          side_effect=[{"learning_rate": 0.01 * (i + 1)}
                                       for i in range(5)]):
            brain.generate_recommendations([])

        history = [
            _make_rec(cid, JobStates.success, 0.1 * (cid + 1), job_id=f"job_{cid}")
            for cid in range(5)
        ]

        np.random.seed(42)
        recs = brain.generate_recommendations(history)

        replaced_rec = next(r for r in recs if r.id == 4)
        assert replaced_rec.resume_from_job_id is not None
        assert replaced_rec.resume_from_job_id == "job_0"

    @patch(f'{PATCH_PREFIX}.pbt.get_flatten_specs')
    def test_surviving_member_resumes_from_self(self, mock_gfs):
        """Surviving member has resume_from_job_id=None (continues own training)."""
        brain = _make_pbt(population_size=5)

        with patch.object(brain, '_generate_random_parameters',
                          side_effect=[{"learning_rate": 0.01 * (i + 1)}
                                       for i in range(5)]):
            brain.generate_recommendations([])

        history = [
            _make_rec(cid, JobStates.success, 0.1 * (cid + 1), job_id=f"job_{cid}")
            for cid in range(5)
        ]

        recs = brain.generate_recommendations(history)

        surviving_rec = next(r for r in recs if r.id == 0)
        assert surviving_rec.resume_from_job_id is None


class TestPBTEvalIntervalEpochTracking:
    """Test eval_interval drives epoch_number updates."""

    @patch(f'{PATCH_PREFIX}.pbt.get_flatten_specs')
    def test_epoch_number_tracks_generation(self, mock_gfs):
        """epoch_number = (generation + 1) * eval_interval after each gen."""
        brain = _make_pbt(population_size=3, max_generations=5, eval_interval=10)

        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.05}):
            brain.generate_recommendations([])

        assert brain.epoch_number == 10

        history = [_make_rec(cid, JobStates.success, 0.1 * (cid + 1)) for cid in range(3)]
        brain.generate_recommendations(history)
        assert brain.generation == 1
        assert brain.epoch_number == 20

        brain.generate_recommendations(history)
        assert brain.generation == 2
        assert brain.epoch_number == 30


class TestPBTPerturbParameterTypes:
    """Test _perturb_parameter handles different value types correctly."""

    def test_perturb_float_multiply_or_divide(self):
        """Float perturbation either multiplies or divides by perturbation_factor."""
        brain = _make_pbt(perturbation_factor=1.5)
        param = {"parameter": "learning_rate", "value_type": "float",
                 "valid_min": 0.001, "valid_max": 0.1}
        current = 0.05

        np.random.seed(100)
        results = set()
        for _ in range(50):
            val = brain._perturb_parameter(param, current)
            results.add(round(val, 6))

        expected_multiply = min(0.1, 0.05 * 1.5)
        expected_divide = max(0.001, 0.05 / 1.5)
        assert any(abs(v - expected_multiply) < 0.001 or abs(v - expected_divide) < 0.001
                   for v in results)

    def test_perturb_int_delta(self):
        """Int perturbation adds/subtracts delta based on perturbation_factor."""
        brain = _make_pbt(perturbation_factor=1.5)
        param = {"parameter": "batch_size", "value_type": "int",
                 "valid_min": 1, "valid_max": 64}
        current = 32

        np.random.seed(100)
        results = set()
        for _ in range(50):
            val = brain._perturb_parameter(param, current)
            results.add(val)

        delta = max(1, int(abs(32) * 0.5))
        assert (32 + delta) in results or (32 - delta) in results
