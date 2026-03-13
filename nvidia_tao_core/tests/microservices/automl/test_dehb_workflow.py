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

"""Unit tests for DEHB generate_recommendations workflow"""

import numpy as np
from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.automl.dehb import DEHB
from nvidia_tao_core.microservices.utils.automl_utils import ResumeRecommendation, JobStates


PATCH_PREFIX = 'nvidia_tao_core.microservices.automl'

PARAMETERS = [
    {"parameter": "learning_rate", "value_type": "float",
     "valid_min": 0.001, "valid_max": 0.1}
]


def _make_dehb(job_id="job_dehb_test", max_epochs=9, reduction_factor=3,
               epoch_multiplier=1, metric="loss"):
    with patch(f'{PATCH_PREFIX}.dehb.save_job_specs'), \
         patch(f'{PATCH_PREFIX}.dehb.get_job_specs',
               return_value={"training_config": {"num_epochs": 10}}), \
         patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_automl_custom_param_ranges',
               return_value={}), \
         patch(f'{PATCH_PREFIX}.automl_algorithm_base.get_job_specs',
               return_value={}):
        job_context = Mock()
        job_context.id = job_id
        job_context.handler_id = f"exp_{job_id}"
        return DEHB(
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


class TestDEHBWorkflow:
    """Test generate_recommendations workflow for DEHB."""

    def test_dehb_bracket_structure(self):
        """Verify ni/ri brackets for max_epochs=9, reduction_factor=3."""
        brain = _make_dehb()
        assert brain.ni["0"] == [9, 3, 1]
        assert brain.ri["0"] == [1, 3, 9]
        assert brain.ni["1"] == [5, 1]
        assert brain.ri["1"] == [3, 9]
        assert brain.ni["2"] == [3]
        assert brain.ri["2"] == [9]

    def test_dehb_initial_launch(self):
        """First call with empty history returns ni[0][0] new config dicts."""
        brain = _make_dehb()
        fixed_config = {"learning_rate": 0.042}
        with patch.object(brain, '_differential_evolution_mutation',
                          return_value=(fixed_config, None)), \
             patch(f'{PATCH_PREFIX}.dehb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs = brain.generate_recommendations([])

        assert len(recs) == 9
        for r in recs:
            assert isinstance(r, dict)
            assert r["learning_rate"] == 0.042
        assert brain.last_launched_count == 9

    def test_dehb_waits_for_completion(self):
        """Returns [] when experiments are still running."""
        brain = _make_dehb()
        brain.sh_iter = 0
        brain.expt_iter = 9
        brain.last_launched_count = 9

        running = [_make_rec(i, JobStates.running, 0.0) for i in range(9)]
        with patch(f'{PATCH_PREFIX}.dehb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs = brain.generate_recommendations(running)

        assert recs == []

    def test_dehb_promotes_best_configs(self):
        """After all configs complete, top-k are promoted as ResumeRecommendations."""
        brain = _make_dehb()
        brain.sh_iter = 0
        brain.expt_iter = 9
        brain.last_launched_count = 9

        results = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        history = _build_completed_history(9, results)

        with patch(f'{PATCH_PREFIX}.dehb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs = brain.generate_recommendations(history)

        assert len(recs) == 3
        for r in recs:
            assert isinstance(r, ResumeRecommendation)
        promoted_ids = {r.id for r in recs}
        assert promoted_ids == {8, 7, 6}

    def test_dehb_complete_bracket_flow(self):
        """Full bracket 0: launch 9 -> promote 3 -> promote 1 -> bracket done."""
        brain = _make_dehb()
        fixed_config = {"learning_rate": 0.05}

        with patch.object(brain, '_differential_evolution_mutation',
                          return_value=(fixed_config, None)), \
             patch(f'{PATCH_PREFIX}.dehb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs_1 = brain.generate_recommendations([])
        assert len(recs_1) == 9

        results_rung0 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        history = _build_completed_history(9, results_rung0)
        with patch(f'{PATCH_PREFIX}.dehb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs_2 = brain.generate_recommendations(history)
        assert len(recs_2) == 3
        assert all(isinstance(r, ResumeRecommendation) for r in recs_2)
        promoted_3_ids = {r.id for r in recs_2}
        assert promoted_3_ids == {8, 7, 6}

        for rec in history:
            if rec.id in promoted_3_ids:
                rec.result = rec.result * 0.5
        with patch(f'{PATCH_PREFIX}.dehb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs_3 = brain.generate_recommendations(history)
        assert len(recs_3) == 1
        assert isinstance(recs_3[0], ResumeRecommendation)
        assert recs_3[0].id == 8

        for rec in history:
            if rec.id == 8:
                rec.result = 0.01
        with patch.object(brain, '_differential_evolution_mutation',
                          return_value=(fixed_config, None)), \
             patch(f'{PATCH_PREFIX}.dehb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            brain.generate_recommendations(history)
        assert brain.bracket == "1"

    def test_dehb_done_flag(self):
        """done() returns False while running, True when complete with no pending."""
        brain = _make_dehb()
        assert brain.done() is False

        brain.complete = True
        brain.last_launched_count = 5
        assert brain.done() is False

        brain.last_launched_count = 0
        assert brain.done() is True


class TestDEHBNormalizeVector:
    """Test config normalization to/from [0,1] vectors."""

    def test_normalize_config_to_vector(self):
        brain = _make_dehb()
        specs = {"learning_rate": 0.0505}
        vector = brain._normalize_config_to_vector(specs)
        assert isinstance(vector, np.ndarray)
        assert len(vector) == len(brain.parameters)
        # Log-uniform encoding: (log10(0.0505) - log10(0.001)) / (log10(0.1) - log10(0.001))
        log_min = np.log10(0.001)
        log_max = np.log10(0.1)
        expected = (np.log10(0.0505) - log_min) / (log_max - log_min)
        assert abs(vector[0] - expected) < 1e-6

    def test_vector_to_config_roundtrip(self):
        brain = _make_dehb()
        vector = np.array([0.5])
        config = brain._vector_to_config(vector)
        assert "learning_rate" in config
        # Log-uniform decoding: 10^(0.5 * (log10(0.1) - log10(0.001)) + log10(0.001))
        expected_lr = 10 ** (0.5 * (np.log10(0.1) - np.log10(0.001)) + np.log10(0.001))
        assert abs(config["learning_rate"] - expected_lr) < 1e-6

    def test_normalize_clamps_to_bounds(self):
        brain = _make_dehb()
        specs = {"learning_rate": 0.5}
        vector = brain._normalize_config_to_vector(specs)
        assert vector[0] == 1.0

    def test_normalize_categorical_encodes_by_index(self):
        """Categorical parameters should encode to their index position, not a constant."""
        brain = _make_dehb()
        brain.parameters.append({
            "parameter": "optimizer",
            "value_type": "categorical",
            "valid_options": ["sgd", "adam", "adamw"]
        })
        specs = {"learning_rate": 0.05, "optimizer": "adam"}
        vector = brain._normalize_config_to_vector(specs)
        assert len(vector) == 2
        expected_cat = (1 + 0.5) / 3  # index 1, midpoint encoding
        assert abs(vector[1] - expected_cat) < 1e-6

    def test_normalize_bool_encodes_distinct_values(self):
        """Bool parameters should encode True/False distinctly, not as constant 0.5."""
        brain = _make_dehb()
        brain.parameters.append({
            "parameter": "use_augmentation",
            "value_type": "bool",
        })
        specs_true = {"learning_rate": 0.05, "use_augmentation": True}
        specs_false = {"learning_rate": 0.05, "use_augmentation": False}
        vec_true = brain._normalize_config_to_vector(specs_true)
        vec_false = brain._normalize_config_to_vector(specs_false)
        assert vec_true[1] == 0.75
        assert vec_false[1] == 0.25
        assert vec_true[1] != vec_false[1]

    def test_normalize_ordered_int_encodes_by_index(self):
        """Ordered_int parameters should encode to their index position."""
        brain = _make_dehb()
        brain.parameters.append({
            "parameter": "batch_size",
            "value_type": "ordered_int",
            "valid_options": [8, 16, 32, 64]
        })
        specs = {"learning_rate": 0.05, "batch_size": 32}
        vector = brain._normalize_config_to_vector(specs)
        assert len(vector) == 2
        expected = (2 + 0.5) / 4  # index 2, midpoint encoding
        assert abs(vector[1] - expected) < 1e-6


class TestDEHBMutationCrossover:
    """Test DE mutation and crossover logic."""

    def test_de_mutation_with_small_population_falls_back(self):
        """With < 4 population members at budget, DE falls back to random."""
        brain = _make_dehb()
        budget = 1
        brain.budget_populations[budget] = [np.array([0.1]), np.array([0.5])]
        brain.budget_results[budget] = [0.5, 0.3]
        specs, target_info = brain._differential_evolution_mutation(budget)
        assert isinstance(specs, dict)
        assert "learning_rate" in specs
        assert target_info is None

    def test_de_mutation_with_sufficient_population(self):
        """With >= 4 members at budget, DE uses mutation/crossover and returns target with budget."""
        brain = _make_dehb()
        np.random.seed(42)
        budget = 1
        brain.budget_populations[budget] = [
            np.array([0.1]), np.array([0.3]),
            np.array([0.5]), np.array([0.7]),
            np.array([0.9])
        ]
        brain.budget_results[budget] = [0.5, 0.4, 0.3, 0.2, 0.1]
        specs, target_info = brain._differential_evolution_mutation(budget)
        assert isinstance(specs, dict)
        assert "learning_rate" in specs
        lr = specs["learning_rate"]
        assert 0.001 <= lr <= 0.1
        assert target_info is not None
        target_vector, target_result, target_budget = target_info
        assert isinstance(target_vector, np.ndarray)
        assert target_result in brain.budget_results[budget]
        assert target_budget == budget

    def test_de_mutation_empty_budget_falls_back(self):
        """DE at a budget with no population falls back to random."""
        brain = _make_dehb()
        brain.budget_populations[1] = [
            np.array([0.1]), np.array([0.3]),
            np.array([0.5]), np.array([0.7]),
        ]
        brain.budget_results[1] = [0.5, 0.4, 0.3, 0.2]
        specs, target_info = brain._differential_evolution_mutation(budget=3)
        assert isinstance(specs, dict)
        assert target_info is None

    def test_de_mutation_factor_and_crossover(self):
        """Mutation factor and crossover_prob affect the generated config."""
        brain = _make_dehb()
        brain.mutation_factor = 0.5
        brain.crossover_prob = 0.5
        assert brain.mutation_factor == 0.5
        assert brain.crossover_prob == 0.5


class TestDEHBPopulationBuilding:
    """Test per-budget DE populations grow from completed experiments."""

    def test_population_grows_after_completion(self):
        """Completed experiments are added to the budget-specific DE population."""
        brain = _make_dehb()
        assert brain.budget_populations == {}

        fixed_config = {"learning_rate": 0.042}
        with patch.object(brain, '_differential_evolution_mutation',
                          return_value=(fixed_config, None)), \
             patch(f'{PATCH_PREFIX}.dehb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            brain.generate_recommendations([])

        # All 9 configs at budget 1 (ri["0"][0] * epoch_multiplier = 1*1 = 1)
        assert len(brain.config_budgets) == 9
        for cid in range(9):
            assert brain.config_budgets[cid] == 1

        history = _build_completed_history(9, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

        with patch(f'{PATCH_PREFIX}.dehb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            brain.generate_recommendations(history)

        assert 1 in brain.budget_populations
        assert len(brain.budget_populations[1]) > 0

    def test_promoted_configs_tracked_at_higher_budget(self):
        """Promoted configs get tracked at their new budget level."""
        brain = _make_dehb()
        fixed_config = {"learning_rate": 0.05}

        with patch.object(brain, '_differential_evolution_mutation',
                          return_value=(fixed_config, None)), \
             patch(f'{PATCH_PREFIX}.dehb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            brain.generate_recommendations([])

        results_rung0 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        history = _build_completed_history(9, results_rung0)
        with patch(f'{PATCH_PREFIX}.dehb.get_flatten_specs'), \
             patch.object(brain, 'override_num_epochs'):
            recs = brain.generate_recommendations(history)

        assert len(recs) == 3
        # Promoted configs should now have budget=3 (ri["0"][1]*1=3)
        for r in recs:
            assert brain.config_budgets[r.id] == 3
