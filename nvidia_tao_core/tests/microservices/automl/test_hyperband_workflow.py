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

"""Unit tests for Hyperband generate_recommendations workflow"""

from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.automl.hyperband import HyperBand
from nvidia_tao_core.microservices.utils.automl_utils import ResumeRecommendation, JobStates


PATCH_PREFIX = 'nvidia_tao_core.microservices.automl'

PARAMETERS = [
    {"parameter": "learning_rate", "value_type": "float",
     "valid_min": 0.001, "valid_max": 0.1}
]


def _make_hyperband(job_id="job_hb_test", max_epochs=9, reduction_factor=3,
                    epoch_multiplier=1, metric="loss"):
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
        return HyperBand(
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


class TestHyperbandWorkflow:
    """Test generate_recommendations workflow for HyperBand."""

    def test_hyperband_bracket_structure(self):
        """Verify ni/ri brackets for max_epochs=9, reduction_factor=3."""
        brain = _make_hyperband()
        assert brain.ni["0"] == [9, 3, 1]
        assert brain.ri["0"] == [1, 3, 9]
        assert brain.ni["1"] == [5, 1]
        assert brain.ri["1"] == [3, 9]
        assert brain.ni["2"] == [3]
        assert brain.ri["2"] == [9]

    def test_hyperband_initial_launch(self):
        """First call with empty history returns ni[0][0] new config dicts."""
        brain = _make_hyperband()
        fixed_config = {"learning_rate": 0.042}
        with patch.object(brain, '_generate_random_parameters', return_value=fixed_config), \
             patch(f'{PATCH_PREFIX}.hyperband.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs = brain.generate_recommendations([])

        assert len(recs) == 9
        for r in recs:
            assert isinstance(r, dict)
            assert r["learning_rate"] == 0.042
        assert brain.last_launched_count == 9

    def test_hyperband_waits_for_completion(self):
        """Returns [] when experiments are still running."""
        brain = _make_hyperband()
        brain.sh_iter = 0
        brain.expt_iter = 9
        brain.last_launched_count = 9

        running = [_make_rec(i, JobStates.running, 0.0) for i in range(9)]
        with patch(f'{PATCH_PREFIX}.hyperband.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs = brain.generate_recommendations(running)

        assert recs == []

    def test_hyperband_promotes_best_configs(self):
        """After all configs complete, top-k are promoted as ResumeRecommendations."""
        brain = _make_hyperband()
        brain.sh_iter = 0
        brain.expt_iter = 9
        brain.last_launched_count = 9

        results = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        history = _build_completed_history(9, results)

        with patch(f'{PATCH_PREFIX}.hyperband.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs = brain.generate_recommendations(history)

        assert len(recs) == 3
        for r in recs:
            assert isinstance(r, ResumeRecommendation)
        promoted_ids = {r.id for r in recs}
        assert promoted_ids == {8, 7, 6}

    def test_hyperband_complete_bracket_flow(self):
        """Full bracket 0: launch 9 -> promote 3 -> promote 1 -> bracket done."""
        brain = _make_hyperband()
        fixed_config = {"learning_rate": 0.05}

        with patch.object(brain, '_generate_random_parameters', return_value=fixed_config), \
             patch(f'{PATCH_PREFIX}.hyperband.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs_1 = brain.generate_recommendations([])
        assert len(recs_1) == 9

        results_rung0 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        history = _build_completed_history(9, results_rung0)
        with patch(f'{PATCH_PREFIX}.hyperband.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs_2 = brain.generate_recommendations(history)
        assert len(recs_2) == 3
        assert all(isinstance(r, ResumeRecommendation) for r in recs_2)
        promoted_3_ids = {r.id for r in recs_2}
        assert promoted_3_ids == {8, 7, 6}

        for rec in history:
            if rec.id in promoted_3_ids:
                rec.result = rec.result * 0.5
        with patch(f'{PATCH_PREFIX}.hyperband.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs_3 = brain.generate_recommendations(history)
        assert len(recs_3) == 1
        assert isinstance(recs_3[0], ResumeRecommendation)
        assert recs_3[0].id == 8

        for rec in history:
            if rec.id == 8:
                rec.result = 0.01
        with patch.object(brain, '_generate_random_parameters', return_value=fixed_config), \
             patch(f'{PATCH_PREFIX}.hyperband.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            brain.generate_recommendations(history)
        assert brain.bracket == "1"

    def test_hyperband_done_flag(self):
        """done() returns False while running, True when complete with no pending."""
        brain = _make_hyperband()
        assert brain.done() is False

        brain.complete = True
        brain.last_launched_count = 5
        assert brain.done() is False

        brain.last_launched_count = 0
        assert brain.done() is True


class TestHyperbandEpochMultiplier:
    """Test epoch_multiplier scaling of ri values."""

    def test_epoch_multiplier_scales_ri(self):
        """ri values are used with epoch_multiplier when setting epoch_number."""
        brain = _make_hyperband(max_epochs=9, reduction_factor=3, epoch_multiplier=2)
        assert brain.ri["0"] == [1, 3, 9]
        assert brain.epoch_multiplier == 2

    def test_epoch_number_uses_multiplier(self):
        """epoch_number = ri * epoch_multiplier after initial launch."""
        brain = _make_hyperband(max_epochs=9, reduction_factor=3, epoch_multiplier=2)
        fixed_config = {"learning_rate": 0.05}
        with patch.object(brain, '_generate_random_parameters', return_value=fixed_config), \
             patch(f'{PATCH_PREFIX}.hyperband.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            brain.generate_recommendations([])
        assert brain.epoch_number == 1 * 2  # ri[0][0] * epoch_multiplier


class TestHyperbandIsEpochParameter:
    """Test _is_epoch_parameter filters epoch params from search space."""

    def test_simple_epoch_names(self):
        brain = _make_hyperband()
        assert brain._is_epoch_parameter("num_epochs") is True
        assert brain._is_epoch_parameter("epochs") is True
        assert brain._is_epoch_parameter("epoch") is True
        assert brain._is_epoch_parameter("n_epochs") is True
        assert brain._is_epoch_parameter("max_iters") is True

    def test_nested_epoch_names(self):
        brain = _make_hyperband()
        assert brain._is_epoch_parameter("training_config.num_epochs") is True
        assert brain._is_epoch_parameter("train.epoch") is True
        assert brain._is_epoch_parameter("train_config.max_epochs") is True

    def test_non_epoch_params_not_filtered(self):
        brain = _make_hyperband()
        assert brain._is_epoch_parameter("learning_rate") is False
        assert brain._is_epoch_parameter("train.optm_lr") is False
        assert brain._is_epoch_parameter("policy.lora.r") is False

    def test_epoch_params_skipped_in_random_generation(self):
        """_generate_random_parameters skips epoch-controlling params."""
        params = [
            {"parameter": "learning_rate", "value_type": "float", "valid_min": 0.001, "valid_max": 0.1},
            {"parameter": "train.epoch", "value_type": "int", "valid_min": 1, "valid_max": 20},
        ]
        with patch(f'{PATCH_PREFIX}.hyperband.save_job_specs'), \
             patch(f'{PATCH_PREFIX}.hyperband.get_job_specs',
                   return_value={"training_config": {"num_epochs": 10}}), \
             patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_automl_custom_param_ranges',
                   return_value={}), \
             patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_job_specs',
                   return_value={}):
            from unittest.mock import Mock
            job_context = Mock()
            job_context.id = "job_epoch_test"
            job_context.handler_id = "exp_epoch_test"
            brain = HyperBand(
                job_context=job_context,
                root="/path/to/root/subdir",
                network="image_classification",
                parameters=params,
                max_epochs=9,
                reduction_factor=3,
                epoch_multiplier=1,
            )
        specs = brain._generate_random_parameters()
        assert "learning_rate" in specs
        assert "train.epoch" not in specs


class TestHyperbandMetricDirection:
    """Test metric direction affects promotion sort order."""

    def test_loss_metric_lower_is_better(self):
        brain = _make_hyperband(metric="loss")
        assert brain.reverse_sort is False

    def test_accuracy_metric_higher_is_better(self):
        brain = _make_hyperband(metric="accuracy")
        assert brain.reverse_sort is True

    def test_kpi_metric_higher_is_better(self):
        brain = _make_hyperband(metric="kpi")
        assert brain.reverse_sort is True

    def test_evaluation_cost_lower_is_better(self):
        brain = _make_hyperband(metric="evaluation_cost")
        assert brain.reverse_sort is False

    def test_promotes_highest_when_higher_is_better(self):
        """With accuracy metric, the highest-result configs are promoted."""
        brain = _make_hyperband(metric="accuracy")
        brain.sh_iter = 0
        brain.expt_iter = 9
        brain.last_launched_count = 9

        results = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        history = _build_completed_history(9, results)

        with patch(f'{PATCH_PREFIX}.hyperband.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs = brain.generate_recommendations(history)

        assert len(recs) == 3
        promoted_ids = {r.id for r in recs}
        assert promoted_ids == {8, 7, 6}
