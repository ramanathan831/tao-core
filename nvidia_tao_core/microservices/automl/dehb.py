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

"""DEHB (Differential Evolution HyperBand) AutoML algorithm modules"""
import numpy as np
import math
import logging

from nvidia_tao_core.microservices.utils.automl_utils import (
    ResumeRecommendation, JobStates, get_valid_range, clamp_value,
    get_valid_options, get_option_weights, fix_input_dimension
)
from nvidia_tao_core.microservices.automl.automl_algorithm_base import AutoMLAlgorithmBase
from nvidia_tao_core.microservices.utils.handler_utils import get_flatten_specs
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    save_job_specs,
    get_job_specs,
    save_automl_brain_info,
    get_automl_brain_info
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DEHB(AutoMLAlgorithmBase):
    """DEHB (Differential Evolution HyperBand) AutoML algorithm class

    DEHB combines Differential Evolution for configuration sampling with
    HyperBand's resource allocation strategy. Instead of TPE or random sampling,
    it uses differential evolution's mutation and crossover operators.
    """

    def __init__(self, job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                 mutation_factor=0.5, crossover_prob=0.5, metric="loss"):
        """Initialize the DEHB algorithm class

        Args:
            root: handler root
            network: model we are running AutoML on
            parameters: automl sweepable parameters
            max_epochs: the maximum amount of resource that can be allocated to a single configuration
            reduction_factor: reduction factor for successive halving
            epoch_multiplier: multiplying factor for epochs
            mutation_factor: differential weight (F) for mutation, typically 0.5
            crossover_prob: crossover probability (CR), typically 0.5
            metric: metric to optimize (e.g., 'loss', 'val_accuracy', 'mIoU')
        """
        super().__init__(job_context, root, network, parameters)
        self.epoch_multiplier = int(epoch_multiplier)
        self.metric = metric
        self.ni = {}
        self.ri = {}
        self.brackets_and_sh_sequence(max_epochs, reduction_factor)
        self.epoch_number = 0

        # Differential Evolution parameters
        self.mutation_factor = float(mutation_factor)
        self.crossover_prob = float(crossover_prob)

        # State variables
        self.bracket = "0"
        self.override_num_epochs(self.ri[self.bracket][-1] * self.epoch_multiplier)
        self.sh_iter = 0
        self.experiments_considered = []
        self.expt_iter = 0
        self.complete = False

        # Determine reverse_sort based on metric (same logic as controller)
        # Default: higher is better (accuracy, mIoU, etc.)
        self.reverse_sort = True
        # For loss metrics: lower is better
        if metric == "loss" or "loss" in metric.lower() or metric.lower() in ("evaluation_cost",):
            self.reverse_sort = False
        # Track how many configs were launched in current rung (for parallel execution)
        self.last_launched_count = 0

        # DE-specific: per-budget subpopulations (DEHB paper Section 3.2)
        # Each budget level maintains its own population for DE evolution
        self.budget_populations = {}  # budget -> list of config vectors
        self.budget_results = {}  # budget -> list of corresponding results
        self.config_budgets = {}  # config_id -> budget it was last evaluated at
        # DE selection: maps config_id -> (target_vector, target_result, budget)
        self.trial_targets = {}

        logger.info(
            f"DEHB initialized with max_epochs={max_epochs}, "
            f"reduction_factor={reduction_factor}, epoch_multiplier={self.epoch_multiplier}, "
            f"mutation_factor={mutation_factor}, crossover_prob={crossover_prob}"
        )

    def brackets_and_sh_sequence(self, max_epochs, reduction_factor):
        """Generate ni,ri arrays based on max_epochs and reduction_factor values"""
        smax = int(np.log(max_epochs) / np.log(reduction_factor))
        for itr, s in enumerate(range(smax, -1, -1)):
            self.ni[str(itr)] = []
            self.ri[str(itr)] = []
            n = int(math.ceil((smax + 1) * (reduction_factor**s) / (s + 1)))
            r = int(max_epochs / (reduction_factor**s))
            for s_idx in range(s + 1):
                ni = int(n * (reduction_factor**(-s_idx)))
                ri = int(r * (reduction_factor**s_idx))
                self.ni[str(itr)].append(ni)
                self.ri[str(itr)].append(ri)

    def override_num_epochs(self, num_epochs):
        """Override num epochs parameter in train spec file"""
        spec = get_job_specs(self.job_context.id)
        for key1 in spec:
            if key1 in ("training_config", "train_config", "train"):
                for key2 in spec[key1]:
                    if key2 in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch"):
                        spec[key1][key2] = num_epochs
                    elif key2 in ("train_config"):
                        for key3 in spec[key1][key2]:
                            if key3 == "runner":
                                for key4 in spec[key1][key2][key3]:
                                    if key4 == "max_epochs":
                                        spec[key1][key2][key3][key4] = num_epochs
            elif key1 in ("num_epochs"):
                spec[key1] = num_epochs
        save_job_specs(self.job_context.id, spec)

    def _normalize_config_to_vector(self, specs):
        """Convert a configuration dict to normalized vector [0, 1]^d

        Uses get_valid_range() for float/int to stay consistent with
        _vector_to_config() decoding, especially when custom_ranges apply.

        Args:
            specs: configuration dict

        Returns:
            numpy array of normalized values
        """
        vector = []
        for param in self.parameters:
            param_name = param["parameter"]
            value = specs.get(param_name)
            param_type = param.get("value_type")

            if param_type in ("float", "int", "integer"):
                try:
                    v_min, v_max = get_valid_range(param, self.parent_params, self.custom_ranges)
                except (TypeError, ValueError):
                    v_min, v_max = 0, 1
                if isinstance(v_min, list):
                    v_min = float(v_min[0]) if v_min else 0.0
                if isinstance(v_max, list):
                    v_max = float(v_max[0]) if v_max else 1.0
                v_min, v_max = float(v_min), float(v_max)
                if v_max > v_min and value is not None:
                    try:
                        actual = float(value) if not isinstance(value, list) else float(value[0])
                        normalized = (actual - v_min) / (v_max - v_min)
                        vector.append(np.clip(normalized, 0.0, 1.0))
                    except (TypeError, ValueError):
                        vector.append(0.5)
                else:
                    vector.append(0.5)

            elif param_type in ("categorical", "ordered"):
                valid_options = get_valid_options(param, self.custom_ranges)
                if valid_options and valid_options != "" and value is not None:
                    try:
                        idx = list(valid_options).index(value)
                        vector.append((idx + 0.5) / len(valid_options))
                    except ValueError:
                        str_options = [str(o) for o in valid_options]
                        if str(value) in str_options:
                            idx = str_options.index(str(value))
                            vector.append((idx + 0.5) / len(valid_options))
                        else:
                            vector.append(0.5)
                else:
                    vector.append(0.5)

            elif param_type == "ordered_int":
                valid_options = get_valid_options(param, self.custom_ranges)
                if valid_options and valid_options != "" and value is not None:
                    try:
                        int_val = int(value)
                        int_options = [int(o) for o in valid_options]
                        idx = int_options.index(int_val)
                        vector.append((idx + 0.5) / len(valid_options))
                    except (ValueError, TypeError):
                        vector.append(0.5)
                else:
                    vector.append(0.5)

            elif param_type == "bool":
                if value is not None:
                    vector.append(0.75 if bool(value) else 0.25)
                else:
                    vector.append(0.5)

            else:
                vector.append(0.5)

        return np.array(vector)

    def _vector_to_config(self, vector):
        """Convert normalized vector to configuration dict

        Args:
            vector: numpy array of normalized values [0, 1]

        Returns:
            configuration dict
        """
        specs = {}
        for i, param in enumerate(self.parameters):
            param_name = param["parameter"]
            normalized_value = np.clip(vector[i], 0.0, 1.0)

            # Convert back to parameter value
            param_type = param.get("value_type")
            math_cond = param.get("math_cond", None)

            if param_type == "float":
                v_min, v_max = get_valid_range(param, self.parent_params, self.custom_ranges)
                value = normalized_value * (v_max - v_min) + v_min
                value = clamp_value(value, v_min, v_max)
                specs[param_name] = value

            elif param_type in ("int", "integer"):
                v_min, v_max = get_valid_range(param, self.parent_params, self.custom_ranges)
                # Map continuous to discrete integer
                continuous_value = normalized_value * (v_max - v_min) + v_min
                value = int(round(continuous_value))

                # Apply math conditions if specified
                # Skip relational constraints (like "> depends_on") as they're handled in base class
                if math_cond and type(math_cond) is str and "depends_on" not in math_cond:
                    parts = math_cond.split(" ")
                    if len(parts) >= 2:
                        operator = parts[0]
                        factor = int(parts[1])
                        if operator == "^":
                            value = int(self._apply_power_constraint_with_equal_priority(
                                v_min, v_max, factor, value))
                        elif operator == "/":
                            value = fix_input_dimension(value, factor)

                value = int(max(v_min, min(v_max, value)))
                specs[param_name] = value

            elif param_type in ("categorical", "ordered"):
                valid_options = get_valid_options(param, self.custom_ranges)
                if valid_options and valid_options != "":
                    # Map normalized value to discrete index
                    idx = int(normalized_value * len(valid_options))
                    idx = min(idx, len(valid_options) - 1)

                    # Handle weighted options
                    weights = get_option_weights(param, self.custom_ranges)
                    if weights and len(weights) == len(valid_options):
                        # Use normalized_value to pick from weighted distribution
                        sorted_pairs = sorted(zip(valid_options, weights), key=lambda x: x[1], reverse=True)
                        cumulative = 0
                        total_weight = sum(weights)
                        for option, weight in sorted_pairs:
                            cumulative += weight / total_weight
                            if normalized_value <= cumulative:
                                specs[param_name] = option
                                break
                        else:
                            specs[param_name] = sorted_pairs[0][0]
                    else:
                        specs[param_name] = valid_options[idx]
                else:
                    specs[param_name] = param.get("default_value")

            elif param_type == "ordered_int":
                valid_options = get_valid_options(param, self.custom_ranges)
                if valid_options and valid_options != "":
                    idx = int(normalized_value * len(valid_options))
                    idx = min(idx, len(valid_options) - 1)

                    weights = get_option_weights(param, self.custom_ranges)
                    if weights and len(weights) == len(valid_options):
                        sorted_pairs = sorted(zip(valid_options, weights), key=lambda x: x[1], reverse=True)
                        cumulative = 0
                        total_weight = sum(weights)
                        for option, weight in sorted_pairs:
                            cumulative += weight / total_weight
                            if normalized_value <= cumulative:
                                specs[param_name] = int(option)
                                break
                        else:
                            specs[param_name] = int(sorted_pairs[0][0])
                    else:
                        specs[param_name] = int(valid_options[idx])
                else:
                    default_val = param.get("default_value")
                    specs[param_name] = int(default_val) if default_val else 0

            elif param_type == "bool":
                # Map to binary
                specs[param_name] = normalized_value >= 0.5

            else:
                # Use base class method for complex types (lists, dicts, etc.)
                specs[param_name] = self.generate_automl_param_rec_value(param)

        return specs

    def _differential_evolution_mutation(self, budget):
        """Generate new configuration using DE/rand/1/bin mutation strategy.

        Uses the subpopulation at the given budget level for DE evolution.
        Falls back to random sampling if the budget's subpopulation has < 4 members.

        Args:
            budget: The budget (epoch) level for this configuration.

        Returns:
            tuple: (config_dict, target_info) where target_info is
                   (target_vector, target_result, budget) for DE selection, or None for random.
        """
        pop = self.budget_populations.get(budget, [])
        pop_results = self.budget_results.get(budget, [])

        if len(pop) < 4:
            logger.info(
                f"Insufficient population for DE at budget {budget} "
                f"(size={len(pop)}), using random sampling"
            )
            return self._generate_random_parameters(), None

        base_idx = np.random.randint(len(pop))
        base_vector = pop[base_idx]

        indices = list(range(len(pop)))
        indices.remove(base_idx)
        r1, r2 = np.random.choice(indices, size=2, replace=False)

        # Mutation: v = base + F * (x_r1 - x_r2)
        mutant_vector = base_vector + self.mutation_factor * (
            pop[r1] - pop[r2]
        )
        mutant_vector = np.clip(mutant_vector, 0.0, 1.0)

        # Binomial crossover with unconditional j_rand
        j_rand = np.random.randint(len(base_vector))
        trial_vector = np.copy(base_vector)
        for i in range(len(trial_vector)):
            if np.random.rand() < self.crossover_prob or i == j_rand:
                trial_vector[i] = mutant_vector[i]

        target_info = (np.copy(base_vector), pop_results[base_idx], budget)
        logger.info(f"Generated configuration via DE mutation at budget {budget}")
        return self._vector_to_config(trial_vector), target_info

    def _generate_random_parameters(self):
        """Generate random parameter values"""
        hyperparam_dict = {}
        for param in self.parameters:
            name = param["parameter"]
            rec = self.generate_automl_param_rec_value(param)
            logger.info(f"Generated random parameter in DEHB: {name} = {rec}")
            hyperparam_dict[name] = rec
        return hyperparam_dict

    def save_state(self):
        """Save the DEHB algorithm related variables to brain metadata"""
        state_dict = {}
        state_dict["bracket"] = self.bracket
        state_dict["sh_iter"] = self.sh_iter
        state_dict["expt_iter"] = self.expt_iter
        state_dict["complete"] = self.complete
        state_dict["epoch_number"] = self.epoch_number
        state_dict["epoch_multiplier"] = self.epoch_multiplier
        state_dict["ni"] = self.ni
        state_dict["ri"] = self.ri
        state_dict["last_launched_count"] = self.last_launched_count
        state_dict["metric"] = self.metric
        # Save per-budget DE populations
        state_dict["budget_populations"] = {
            str(k): [p.tolist() for p in v]
            for k, v in self.budget_populations.items()
        }
        state_dict["budget_results"] = {
            str(k): v for k, v in self.budget_results.items()
        }
        state_dict["config_budgets"] = {
            str(k): v for k, v in self.config_budgets.items()
        }
        state_dict["trial_targets"] = {
            str(k): (v[0].tolist(), v[1], v[2])
            for k, v in self.trial_targets.items()
        }

        save_automl_brain_info(self.job_context.id, state_dict)

    @staticmethod
    def load_state(job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                   mutation_factor=0.5, crossover_prob=0.5, metric="loss"):
        """Load the DEHB algorithm related variables from brain metadata"""
        json_loaded = get_automl_brain_info(job_context.id)
        if not json_loaded:
            return DEHB(job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                        mutation_factor, crossover_prob, metric)

        # Load metric from state (with fallback to parameter)
        loaded_metric = json_loaded.get("metric", metric)
        brain = DEHB(job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                     mutation_factor, crossover_prob, loaded_metric)
        # Load state
        brain.bracket = json_loaded["bracket"]
        brain.sh_iter = json_loaded["sh_iter"]
        brain.expt_iter = json_loaded["expt_iter"]
        brain.complete = json_loaded["complete"]
        brain.epoch_number = json_loaded["epoch_number"]
        brain.last_launched_count = json_loaded.get("last_launched_count", 0)

        # Load per-budget DE populations
        if "budget_populations" in json_loaded:
            brain.budget_populations = {
                int(k): [np.array(p) for p in v]
                for k, v in json_loaded["budget_populations"].items()
            }
            brain.budget_results = {
                int(k): v
                for k, v in json_loaded["budget_results"].items()
            }
        elif "population" in json_loaded and json_loaded["population"]:
            # Backward compat: migrate old global population to budget 0
            brain.budget_populations[0] = [np.array(p) for p in json_loaded["population"]]
            brain.budget_results[0] = json_loaded["population_results"]

        if "config_budgets" in json_loaded:
            brain.config_budgets = {
                int(k): v for k, v in json_loaded["config_budgets"].items()
            }

        if "trial_targets" in json_loaded:
            brain.trial_targets = {}
            for k, v in json_loaded["trial_targets"].items():
                if len(v) == 3:
                    brain.trial_targets[int(k)] = (np.array(v[0]), v[1], v[2])
                else:
                    # Backward compat: old 2-tuple format -> budget 0
                    brain.trial_targets[int(k)] = (np.array(v[0]), v[1], 0)

        return brain

    def _generate_one_recommendation(self, history):
        """Updates the counter variables and performs successive halving with DE"""
        if self.complete:
            return None

        num = self.ni[self.bracket][self.sh_iter]
        if self.expt_iter == num:
            self.expt_iter = 0
            self.sh_iter += 1
        if self.sh_iter == len(self.ni[self.bracket]):
            self.sh_iter = 0
            self.bracket = str(int(self.bracket) + 1)
            if self.bracket in self.ri.keys():
                self.override_num_epochs(self.ri[self.bracket][-1] * self.epoch_multiplier)
        if int(self.bracket) > int(max(list(self.ni.keys()), key=int)):
            logger.info(f"DEHB: All brackets complete (bracket={self.bracket} > max), setting complete=True")
            self.complete = True
            return None

        if self.sh_iter == 0:
            budget = self.ri[self.bracket][self.sh_iter] * self.epoch_multiplier
            specs, target_info = self._differential_evolution_mutation(budget)
            self.epoch_number = budget
            config_id = len(history) + self.expt_iter
            self.config_budgets[config_id] = budget
            if target_info is not None:
                self.trial_targets[config_id] = target_info
            to_return = specs
        else:
            # Do successive halving
            lower = -1 * self.ni.get(self.bracket, [0])[0]

            if self.expt_iter == 0:
                if self.sh_iter == 1:
                    self.experiments_considered = sorted(
                        history[lower:],
                        key=lambda rec: rec.result,
                        reverse=self.reverse_sort
                    )[0:self.ni[self.bracket][self.sh_iter]]
                else:
                    for experiment in self.experiments_considered:
                        experiment.result = history[experiment.id].result
                    self.experiments_considered = sorted(
                        self.experiments_considered,
                        key=lambda rec: rec.result,
                        reverse=self.reverse_sort
                    )[0:self.ni[self.bracket][self.sh_iter]]

            budget = self.ri[self.bracket][self.sh_iter] * self.epoch_multiplier
            self.epoch_number = budget
            rec_id = self.experiments_considered[self.expt_iter].id
            self.config_budgets[rec_id] = budget
            resumerec = ResumeRecommendation(
                rec_id,
                self.experiments_considered[self.expt_iter].specs,
                self.experiments_considered[self.expt_iter].job_id
            )
            to_return = resumerec
        self.expt_iter += 1

        return to_return

    def done(self):
        """Return if DEHB algorithm is complete or not.

        Returns True only if all recommendations have been issued AND all have completed.
        Checks last_launched_count to ensure running experiments finish before declaring done.
        """
        logger.info(f"DEHB done() called: complete={self.complete}, last_launched_count={self.last_launched_count}")

        if not self.complete:
            logger.info("DEHB done() returning False: not complete yet")
            return False

        # If complete flag is set but we still have running experiments, not done yet
        if self.last_launched_count > 0:
            logger.warning(f"DEHB done() returning False: {self.last_launched_count} experiments still running!")
            return False

        logger.info("DEHB done() returning True: all recommendations issued and completed")
        return True

    @property
    def max_concurrent(self):
        """Maximum number of concurrent experiments for DEHB.

        Returns the maximum ni value across all brackets to allow full parallelism.
        """
        max_ni = 1
        for bracket_ni in self.ni.values():
            if bracket_ni:
                max_ni = max([max_ni] + bracket_ni)
        return max_ni

    def generate_recommendations(self, history):
        """Generates recommendations for the controller to run (supports parallel execution)"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)

        logger.info(
            f"DEHB generate_recommendations: complete={self.complete}, "
            f"last_launched_count={self.last_launched_count}, history_len={len(history)}"
        )

        if self.complete:
            # Check if final experiments have completed
            # NOTE: Resumed experiments reuse the same ID, so we can't rely on history indices
            # Instead, check for ANY pending/running experiments
            if self.last_launched_count > 0 and history:
                logger.info(
                    f"DEHB checking final batch completion: last_launched_count={self.last_launched_count}, "
                    f"history_len={len(history)}"
                )

                # Log status of ALL experiments
                for i, exp in enumerate(history):
                    logger.info(
                        f"  history[{i}]: id={exp.id}, status={exp.status}, "
                        f"job_id={exp.job_id}, result={exp.result}"
                    )

                # Check for ANY pending/started/running experiments (resumed experiments reuse IDs!)
                any_running = any(
                    exp.status in [JobStates.pending, JobStates.started, JobStates.running]
                    for exp in history
                )

                logger.info(f"DEHB final batch check: any_running={any_running}")

                if not any_running:
                    logger.info(
                        f"DEHB: All experiments complete, resetting last_launched_count "
                        f"from {self.last_launched_count} to 0"
                    )
                    self.last_launched_count = 0  # Reset counter after final batch completes
                else:
                    logger.info(
                        f"DEHB: Experiments still running, keeping last_launched_count={self.last_launched_count}"
                    )
            return []

        # DE selection with per-budget subpopulations (DEHB paper Section 3.2)
        for rec in history:
            if rec.status == JobStates.success and rec.result != 0.0:
                budget = self.config_budgets.get(rec.id)
                if budget is None:
                    continue

                config_vector = self._normalize_config_to_vector(rec.specs)

                if budget not in self.budget_populations:
                    self.budget_populations[budget] = []
                    self.budget_results[budget] = []

                pop = self.budget_populations[budget]
                is_duplicate = any(np.allclose(config_vector, p) for p in pop)
                if is_duplicate:
                    self.trial_targets.pop(rec.id, None)
                    continue

                target_info = self.trial_targets.pop(rec.id, None)

                if target_info is not None:
                    target_vector, target_result, target_budget = target_info
                    trial_is_better = (
                        (self.reverse_sort and rec.result > target_result) or
                        (not self.reverse_sort and rec.result < target_result)
                    )
                    if trial_is_better:
                        target_pop = self.budget_populations.get(target_budget, [])
                        target_idx = None
                        for idx, p in enumerate(target_pop):
                            if np.allclose(p, target_vector):
                                target_idx = idx
                                break
                        if target_idx is not None:
                            self.budget_populations[target_budget][target_idx] = config_vector
                            self.budget_results[target_budget][target_idx] = rec.result
                            logger.info(
                                f"DE selection: trial replaced target at budget {target_budget} "
                                f"(pop_size={len(target_pop)})"
                            )
                        else:
                            pop.append(config_vector)
                            self.budget_results[budget].append(rec.result)
                            logger.info(
                                f"DE selection: target gone, added trial at budget {budget} "
                                f"(pop_size={len(pop)})"
                            )
                    else:
                        logger.info("DE selection: trial worse than target, discarded")
                else:
                    pop.append(config_vector)
                    self.budget_results[budget].append(rec.result)
                    logger.info(
                        f"Added config to budget {budget} population (pop_size={len(pop)})"
                    )

        # Initial case: launch all configs for first rung in parallel
        if history == []:
            num_configs_in_rung = self.ni[self.bracket][self.sh_iter]
            recommendations = []
            for _ in range(num_configs_in_rung):
                rec = self._generate_one_recommendation(history)
                if type(rec) is dict:
                    recommendations.append(rec)
            self.last_launched_count = len(recommendations)
            self.track_id = len(recommendations) - 1 if recommendations else 0
            logger.info(
                f"DEHB: Launching {len(recommendations)} parallel configs for "
                f"bracket {self.bracket}, rung {self.sh_iter}"
            )
            return recommendations

        # Check if all experiments from last launch are complete
        # NOTE: Resumed experiments reuse IDs, so check for ANY running experiments
        if self.last_launched_count > 0:
            logger.info("DEHB: Checking if last batch complete before generating new recommendations")

            # Check for ANY pending/started/running experiments (handles resumed experiments with reused IDs)
            any_running = any(
                exp.status in [JobStates.pending, JobStates.started, JobStates.running]
                for exp in history
            )

            logger.info(f"DEHB: any_running={any_running} (last_launched_count={self.last_launched_count})")

            if any_running:
                logger.info("DEHB: Experiments still running, waiting before generating new recommendations")
                return []  # Wait for current batch to complete

            logger.info("DEHB: All experiments complete, proceeding to generate new recommendations")

        # All previous experiments are done, generate next batch
        num_configs_before = self.ni[self.bracket][self.sh_iter] if not self.complete else 0

        recommendations = []
        for _ in range(max(num_configs_before, 1)):  # At least try to generate one
            rec = self._generate_one_recommendation(history)
            if rec is None:
                break
            recommendations.append(rec)

            # After first recommendation, check if state changed (new rung/bracket)
            if len(recommendations) == 1:
                num_configs_in_new_rung = self.ni[self.bracket][self.sh_iter] if not self.complete else 0
                # If we moved to a new state, generate rest of configs for new rung
                if num_configs_in_new_rung != num_configs_before:
                    for _ in range(num_configs_in_new_rung - 1):
                        rec = self._generate_one_recommendation(history)
                        if rec:
                            recommendations.append(rec)
                        else:
                            break
                    break

        self.last_launched_count = len(recommendations)

        if recommendations:
            last_rec = recommendations[-1]
            if type(last_rec) is dict:
                self.track_id = len(history) + len(recommendations) - 1
            elif type(last_rec) is ResumeRecommendation:
                self.track_id = last_rec.id

            logger.info(
                f"DEHB: Launching {len(recommendations)} recommendation(s) for "
                f"bracket {self.bracket}, rung {self.sh_iter}"
            )

        return recommendations
