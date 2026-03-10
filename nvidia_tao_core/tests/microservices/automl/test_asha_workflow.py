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

"""Unit tests for ASHA generate_recommendations workflow"""

from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.automl.asha import ASHA
from nvidia_tao_core.microservices.utils.automl_utils import ResumeRecommendation, JobStates


PATCH_PREFIX = 'nvidia_tao_core.microservices.automl'
BASE_PREFIX = f'{PATCH_PREFIX}.automl_algorithm_base'

PARAMETERS = [
    {"parameter": "learning_rate", "value_type": "float",
     "valid_min": 0.001, "valid_max": 0.1}
]


def _make_asha(job_id="job_asha_test", max_epochs=9, reduction_factor=3,
               epoch_multiplier=1, max_concurrent=4, metric="loss"):
    with patch(f'{PATCH_PREFIX}.asha.save_job_specs'), \
         patch(f'{PATCH_PREFIX}.asha.get_job_specs',
               return_value={"training_config": {"num_epochs": 10}}), \
         patch(f'{BASE_PREFIX}.get_automl_custom_param_ranges',
               return_value={}), \
         patch(f'{BASE_PREFIX}.get_job_specs', return_value={}):
        job_context = Mock()
        job_context.id = job_id
        job_context.handler_id = f"exp_{job_id}"
        return ASHA(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=PARAMETERS,
            max_epochs=max_epochs,
            reduction_factor=reduction_factor,
            epoch_multiplier=epoch_multiplier,
            max_concurrent=max_concurrent,
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


class TestASHARungCalculation:
    """Test ASHA rung (resource level) calculation."""

    def test_asha_rung_calculation(self):
        """Verify rungs=[1,3,9] for max_epochs=9, rf=3, em=1."""
        brain = _make_asha(max_epochs=9, reduction_factor=3, epoch_multiplier=1)
        assert brain.rungs == [1, 3, 9]

    def test_asha_rung_calculation_with_epoch_multiplier(self):
        """Rungs scale with epoch_multiplier."""
        brain = _make_asha(max_epochs=9, reduction_factor=3, epoch_multiplier=2)
        assert brain.rungs == [2, 6, 18]


class TestASHAInitialLaunch:
    """Test ASHA initial launch behaviour (history=[])."""

    @patch(f'{PATCH_PREFIX}.asha.get_flatten_specs')
    def test_asha_initial_launch(self, mock_gfs):
        """history=[] returns max_concurrent new dicts."""
        brain = _make_asha(max_concurrent=4)
        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.05}):
            recs = brain.generate_recommendations([])

        assert len(recs) == 4
        for r in recs:
            assert isinstance(r, dict)
            assert "learning_rate" in r

    @patch(f'{PATCH_PREFIX}.asha.get_flatten_specs')
    def test_asha_initial_launch_respects_max_trials(self, mock_gfs):
        """max_trials < max_concurrent limits initial launch count."""
        brain = _make_asha(max_concurrent=4)
        brain.max_trials = 2
        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.05}):
            recs = brain.generate_recommendations([])

        assert len(recs) == 2


class TestASHAPromotionQuota:
    """Test ASHA promotion quota logic."""

    @patch(f'{PATCH_PREFIX}.asha.get_flatten_specs')
    def test_asha_promotion_quota(self, mock_gfs):
        """After 3 completions at rung 0 with rf=3, quota=1 -> promotes best."""
        brain = _make_asha(max_epochs=9, reduction_factor=3, max_concurrent=4)
        for cid in range(3):
            brain.config_to_rung[cid] = 0
            brain.active_configs.add(cid)
            brain.config_specs[cid] = {"learning_rate": 0.01 * (cid + 1)}
        brain.next_config_id = 3
        brain.total_configs_started = 3

        history = [
            _make_rec(0, JobStates.success, 0.5),
            _make_rec(1, JobStates.success, 0.3),
            _make_rec(2, JobStates.success, 0.8),
        ]

        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.07}):
            recs = brain.generate_recommendations(history)

        resume_recs = [r for r in recs if isinstance(r, ResumeRecommendation)]
        assert len(resume_recs) >= 1
        promoted_ids = {r.id for r in resume_recs}
        assert 1 in promoted_ids

    @patch(f'{PATCH_PREFIX}.asha.get_flatten_specs')
    def test_asha_failure_counts_toward_quota(self, mock_gfs):
        """Failures count toward completions but cannot be promoted."""
        brain = _make_asha(max_epochs=9, reduction_factor=3, max_concurrent=4)
        for cid in range(3):
            brain.config_to_rung[cid] = 0
            brain.active_configs.add(cid)
            brain.config_specs[cid] = {"learning_rate": 0.01 * (cid + 1)}
        brain.next_config_id = 3
        brain.total_configs_started = 3

        history = [
            _make_rec(0, JobStates.success, 0.5),
            _make_rec(1, JobStates.failure, 0.0),
            _make_rec(2, JobStates.success, 0.8),
        ]

        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.07}):
            recs = brain.generate_recommendations(history)

        resume_recs = [r for r in recs if isinstance(r, ResumeRecommendation)]
        promoted_ids = {r.id for r in resume_recs}
        assert 1 not in promoted_ids
        assert 0 in promoted_ids


class TestASHAAsyncBehaviour:
    """Test ASHA async (no waiting) behaviour."""

    @patch(f'{PATCH_PREFIX}.asha.get_flatten_specs')
    def test_asha_async_no_waiting(self, mock_gfs):
        """New configs launched immediately when slots available."""
        brain = _make_asha(max_epochs=9, reduction_factor=3, max_concurrent=4)
        brain.config_to_rung[0] = 0
        brain.config_to_rung[1] = 0
        brain.active_configs = {0, 1}
        brain.config_specs[0] = {"learning_rate": 0.01}
        brain.config_specs[1] = {"learning_rate": 0.02}
        brain.next_config_id = 2
        brain.total_configs_started = 2

        history = [
            _make_rec(0, JobStates.success, 0.5),
            _make_rec(1, JobStates.running, 0.0),
        ]

        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.07}):
            recs = brain.generate_recommendations(history)

        assert len(recs) > 0, "ASHA should launch new configs without waiting"


class TestASHAPromotesBest:
    """Test ASHA promotes configs with best results."""

    @patch(f'{PATCH_PREFIX}.asha.get_flatten_specs')
    def test_asha_promotes_best_by_result(self, mock_gfs):
        """Promotes the config with best result (lowest loss)."""
        brain = _make_asha(max_epochs=9, reduction_factor=3, max_concurrent=6)
        brain.reverse_sort = False

        for cid in range(6):
            brain.config_to_rung[cid] = 0
            brain.active_configs.add(cid)
            brain.config_specs[cid] = {"learning_rate": 0.01 * (cid + 1)}
        brain.next_config_id = 6
        brain.total_configs_started = 6

        results = [0.5, 0.2, 0.9, 0.1, 0.6, 0.3]
        history = [
            _make_rec(cid, JobStates.success, results[cid])
            for cid in range(6)
        ]

        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.07}):
            recs = brain.generate_recommendations(history)

        resume_recs = [r for r in recs if isinstance(r, ResumeRecommendation)]
        promoted_ids = {r.id for r in resume_recs}
        assert 3 in promoted_ids
        assert 1 in promoted_ids


class TestASHADone:
    """Test ASHA done() behavior."""

    def test_done_initially_false(self):
        brain = _make_asha()
        assert brain.done() is False

    def test_done_returns_complete_flag(self):
        brain = _make_asha()
        brain.complete = True
        assert brain.done() is True


class TestASHAMinTopConfigs:
    """Test min_top_configs controls stopping."""

    @patch(f'{PATCH_PREFIX}.asha.get_flatten_specs')
    def test_stops_when_min_top_reached(self, mock_gfs):
        """Algorithm completes when min_top_configs reach the max rung."""
        brain = _make_asha(max_epochs=9, reduction_factor=3, max_concurrent=4)
        brain.min_top_configs = 2
        brain.completed_configs = {0, 1}
        brain.active_configs = set()
        brain.total_configs_started = 10

        history = [_make_rec(0, JobStates.success, 0.3), _make_rec(1, JobStates.success, 0.4)]

        recs = brain.generate_recommendations(history)
        assert brain.complete is True
        assert recs == []

    @patch(f'{PATCH_PREFIX}.asha.get_flatten_specs')
    def test_continues_when_below_min_top(self, mock_gfs):
        """Algorithm continues when fewer than min_top_configs have completed all rungs."""
        brain = _make_asha(max_epochs=9, reduction_factor=3, max_concurrent=4)
        brain.min_top_configs = 5
        brain.completed_configs = {0, 1}
        brain.active_configs = set()
        brain.next_config_id = 5
        brain.total_configs_started = 5

        history = [_make_rec(0, JobStates.success, 0.3), _make_rec(1, JobStates.success, 0.4)]

        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.05}):
            recs = brain.generate_recommendations(history)
        assert brain.complete is False
        assert len(recs) > 0


class TestASHAMultiRungPromotion:
    """Test config promotion across multiple rungs."""

    @patch(f'{PATCH_PREFIX}.asha.get_flatten_specs')
    def test_promotion_creates_resume_recommendation(self, mock_gfs):
        """Promoted config becomes a ResumeRecommendation at next rung."""
        brain = _make_asha(max_epochs=9, reduction_factor=3, max_concurrent=4)

        for cid in range(3):
            brain.config_to_rung[cid] = 0
            brain.active_configs.add(cid)
            brain.config_specs[cid] = {"learning_rate": 0.01 * (cid + 1)}
        brain.next_config_id = 3
        brain.total_configs_started = 3

        history = [
            _make_rec(0, JobStates.success, 0.5),
            _make_rec(1, JobStates.success, 0.2),
            _make_rec(2, JobStates.success, 0.8),
        ]

        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.07}):
            recs = brain.generate_recommendations(history)

        resume_recs = [r for r in recs if isinstance(r, ResumeRecommendation)]
        assert len(resume_recs) >= 1
        promoted_config = resume_recs[0]
        assert promoted_config.id == 1
        assert brain.config_to_rung[1] == 1


class TestASHAPromotionPriority:
    """Test promotions take priority over new configs."""

    @patch(f'{PATCH_PREFIX}.asha.get_flatten_specs')
    def test_pending_promotions_launched_first(self, mock_gfs):
        """Pending promotions are launched before new random configs."""
        brain = _make_asha(max_epochs=9, reduction_factor=3, max_concurrent=4)

        brain.pending_promotions = [(1, 3)]
        brain.config_specs[1] = {"learning_rate": 0.02}
        brain.active_configs = set()
        brain.next_config_id = 5
        brain.total_configs_started = 5

        history = [_make_rec(1, JobStates.success, 0.3)]

        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.07}):
            recs = brain.generate_recommendations(history)

        assert isinstance(recs[0], ResumeRecommendation)
        assert recs[0].id == 1


class TestASHAConfigEpochTargets:
    """Test per-config epoch targets for mixed-rung parallel execution."""

    @patch(f'{PATCH_PREFIX}.asha.get_flatten_specs')
    def test_initial_launch_sets_epoch_targets(self, mock_gfs):
        """Initial launch assigns rung-0 epoch target to each config."""
        brain = _make_asha(max_epochs=9, reduction_factor=3, max_concurrent=4)
        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.05}):
            recs = brain.generate_recommendations([])
        assert len(recs) == 4
        for cid in range(4):
            assert brain.config_epoch_targets[cid] == 1

    @patch(f'{PATCH_PREFIX}.asha.get_flatten_specs')
    def test_promotion_sets_correct_epoch_target(self, mock_gfs):
        """Promoted configs get their target rung's epoch count."""
        brain = _make_asha(max_epochs=9, reduction_factor=3, max_concurrent=4)
        brain.pending_promotions = [(1, 3)]
        brain.config_specs[1] = {"learning_rate": 0.02}
        brain.active_configs = set()
        brain.next_config_id = 5
        brain.total_configs_started = 5

        history = [_make_rec(1, JobStates.success, 0.3)]

        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.07}):
            recs = brain.generate_recommendations(history)

        assert brain.config_epoch_targets[1] == 3

    @patch(f'{PATCH_PREFIX}.asha.get_flatten_specs')
    def test_mixed_rung_batch_has_different_targets(self, mock_gfs):
        """A batch with a promotion and new config has different epoch targets."""
        brain = _make_asha(max_epochs=9, reduction_factor=3, max_concurrent=4)
        brain.pending_promotions = [(1, 3)]
        brain.config_specs[1] = {"learning_rate": 0.02}
        brain.active_configs = set()
        brain.next_config_id = 5
        brain.total_configs_started = 5

        history = [_make_rec(1, JobStates.success, 0.3)]

        with patch.object(brain, '_generate_random_parameters',
                          return_value={"learning_rate": 0.07}):
            recs = brain.generate_recommendations(history)

        assert brain.config_epoch_targets[1] == 3
        if len(recs) > 1:
            new_config_id = 5
            assert brain.config_epoch_targets.get(new_config_id) == 1
