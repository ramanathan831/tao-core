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

"""LLM-powered AutoML algorithm (Option 1 / Approach 1).

Plugs into the existing BrainFactory and Controller as `automl_algorithm = "llm"`.
Uses an LLM to generate hyperparameter recommendations instead of GP/Hyperband/DE math.
"""
import json
import logging
import os
import numpy as np
from typing import Any, Dict, List, Optional

from nvidia_tao_core.microservices.automl.automl_algorithm_base import AutoMLAlgorithmBase
from nvidia_tao_core.microservices.automl.llm_client import LLMClient
from nvidia_tao_core.microservices.automl.prompts.llm_brain_prompts import (
    build_recommendation_with_reasoning_prompt,
)
from nvidia_tao_core.microservices.utils.automl_utils import (
    JobStates, get_valid_options
)
from nvidia_tao_core.microservices.utils.handler_utils import get_total_epochs, get_flatten_specs
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    save_automl_brain_info, get_automl_brain_info
)

logger = logging.getLogger(__name__)


class LLMBrain(AutoMLAlgorithmBase):
    """LLM-powered AutoML algorithm that uses an LLM to generate hyperparameter recommendations."""

    def __init__(
        self,
        job_context,
        root,
        network,
        parameters,
        llm_params: Optional[Dict[str, Any]] = None,
        metric: str = "kpi",
    ):
        """Initialize the LLMBrain."""
        super().__init__(job_context, root, network, parameters)
        self.llm_client = LLMClient(params=llm_params)
        self.metric = metric
        self.experiment_history: List[Dict[str, Any]] = []
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_metric: Optional[float] = None
        self.external_knowledge: Optional[str] = None
        self.reverse_sort = True

        self.num_epochs_per_experiment = get_total_epochs(
            job_context, os.path.join(self.handler_root, "specs")
        )

    def generate_recommendations(self, history):
        """Generate hyperparameter recommendations using an LLM."""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)

        # Update internal history from controller's recommendation objects
        self._sync_history(history)

        if history and history[-1].status not in [JobStates.success, JobStates.failure]:
            return []

        # Determine metric direction
        metric_direction = "minimize" if not self.reverse_sort else "maximize"

        messages = build_recommendation_with_reasoning_prompt(
            parameters=self.parameters,
            history=self.experiment_history,
            best_config=self.best_config,
            best_metric=self.best_metric,
            network=self.network,
            metric_name=self.metric,
            metric_direction=metric_direction,
        )

        response = self.llm_client.chat(messages, json_mode=True, temperature=0.7)

        if not response.ok:
            logger.warning("LLM call failed: %s. Falling back to random.", response.error)
            return self._random_fallback()

        config = self._parse_llm_response(response)
        if config is None:
            logger.warning("Could not parse LLM response. Falling back to random.")
            return self._random_fallback()

        # Validate and clamp values to schema bounds
        validated = self._validate_config(config)
        logger.info("LLM recommendation: %s", json.dumps(validated))

        return [validated]

    def _sync_history(self, recommendations):
        """Sync experiment history from controller's recommendation objects."""
        for rec in recommendations:
            if rec.status in [JobStates.success, JobStates.failure]:
                rec_id = rec.id
                if rec_id >= len(self.experiment_history):
                    entry = {
                        "config": rec.specs if hasattr(rec, 'specs') else {},
                        "metric": rec.result if rec.result is not None else 0.0,
                        "status": "success" if rec.status == JobStates.success else "failure",
                    }
                    self.experiment_history.append(entry)

                    if rec.status == JobStates.success:
                        is_better = (
                            self.best_metric is None or
                            (self.reverse_sort and rec.result > self.best_metric) or
                            (not self.reverse_sort and rec.result < self.best_metric)
                        )
                        if is_better:
                            self.best_metric = rec.result
                            self.best_config = rec.specs if hasattr(rec, 'specs') else {}

    def _parse_llm_response(self, response) -> Optional[Dict[str, Any]]:
        """Extract configuration from LLM response (handles both raw and reasoning formats)."""
        data = response.json_content
        if data is None:
            return None

        if isinstance(data, dict):
            if "config" in data:
                reasoning = data.get("reasoning", "")
                if reasoning:
                    logger.info("LLM reasoning: %s", reasoning)
                return data["config"]
            # Direct config format
            param_names = {p["parameter"] for p in self.parameters}
            if any(k in param_names for k in data.keys()):
                return data

        return None

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM-proposed values -- trust the LLM's exact values, only check types.

        Unlike traditional algorithms that search within a range, LLMBrain proposes
        specific values with reasoning. We validate types and enforce hard schema
        constraints (non-negative, valid enum) but do NOT clamp to narrow ranges.
        """
        validated = {}
        for param_dict in self.parameters:
            name = param_dict["parameter"]
            if name not in config:
                validated[name] = self.generate_automl_param_rec_value(param_dict)
                continue

            value = config[name]
            dtype = param_dict.get("value_type", "")

            if dtype == "float":
                try:
                    value = float(value)
                    hard_min = param_dict.get("valid_min")
                    if hard_min not in (None, '', "", "inf", "-inf"):
                        hard_min = float(hard_min)
                        if not np.isinf(hard_min) and value < hard_min:
                            logger.warning(
                                "LLM proposed %s=%s below hard min %s, using min",
                                name, value, hard_min,
                            )
                            value = hard_min
                except (ValueError, TypeError):
                    value = self.generate_automl_param_rec_value(param_dict)

            elif dtype in ("int", "integer"):
                try:
                    value = int(round(float(value)))
                    hard_min = param_dict.get("valid_min")
                    if hard_min not in (None, '', ""):
                        value = max(int(hard_min), value)
                except (ValueError, TypeError):
                    value = self.generate_automl_param_rec_value(param_dict)

            elif dtype in ("categorical", "ordered"):
                valid_options = get_valid_options(param_dict, self.custom_ranges)
                if valid_options and value not in valid_options:
                    value = self.generate_automl_param_rec_value(param_dict)

            elif dtype == "bool":
                if not isinstance(value, bool):
                    value = str(value).lower() in ("true", "1", "yes")

            validated[name] = value

        return validated

    def _random_fallback(self) -> List[Dict[str, Any]]:
        """Generate a random recommendation when LLM fails."""
        recommendations = []
        for param_dict in self.parameters:
            value = self.generate_automl_param_rec_value(param_dict)
            recommendations.append(value)
        return [dict(zip([p["parameter"] for p in self.parameters], recommendations))]

    def save_state(self):
        """Save LLM brain state to MongoDB."""
        state = {
            "experiment_history": self.experiment_history,
            "best_config": self.best_config,
            "best_metric": self.best_metric,
            "llm_usage": self.llm_client.get_usage_summary(),
        }
        save_automl_brain_info(self.job_context.id, state)

    @staticmethod
    def load_state(job_context, root, network, parameters, llm_params=None, metric="kpi"):
        """Load LLM brain state from MongoDB."""
        state = get_automl_brain_info(job_context.id)
        brain = LLMBrain(job_context, root, network, parameters, llm_params, metric)

        if state:
            brain.experiment_history = state.get("experiment_history", [])
            brain.best_config = state.get("best_config")
            brain.best_metric = state.get("best_metric")
            logger.info(
                "Loaded LLM brain state: %d experiments, best_metric=%s",
                len(brain.experiment_history), brain.best_metric,
            )

        return brain
