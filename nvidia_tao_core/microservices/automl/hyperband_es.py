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

"""HyperBand with Early Stopping (Learning Curve Prediction) AutoML algorithm modules"""
import os
import numpy as np
import logging
from scipy.optimize import curve_fit

from nvidia_tao_core.microservices.utils.automl_utils import JobStates
from nvidia_tao_core.microservices.automl.hyperband import HyperBand
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_automl_brain_info
)

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


class HyperBandES(HyperBand):
    """HyperBand with Early Stopping via Learning Curve Prediction

    Extends HyperBand with predictive early stopping using learning curve
    extrapolation. Stops configurations early if predicted final performance
    is unlikely to be competitive.

    The controller reads `configs_to_cancel` after generate_recommendations()
    to actually terminate the cloud jobs for early-stopped configurations.
    """

    def __init__(self, job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                 early_stop_threshold=0.8, min_early_stop_epochs=3):
        """Initialize the HyperBand ES algorithm class

        Args:
            root: handler root
            network: model we are running AutoML on
            parameters: automl sweepable parameters
            max_epochs: the maximum amount of resource that can be allocated to a single configuration
            reduction_factor: reduction factor for successive halving
            epoch_multiplier: multiplying factor for epochs
            early_stop_threshold: confidence threshold for early stopping (0-1)
            min_early_stop_epochs: minimum observations before attempting prediction
        """
        super().__init__(job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier)

        self.min_epochs_for_prediction = int(min_early_stop_epochs)
        self.confidence_threshold = float(early_stop_threshold)

        # Track learning curves: config_id -> [(observation_count, metric), ...]
        self.learning_curves = {}

        # Track early stopped configs
        self.early_stopped_configs = set()

        # Cancellation signal: set of job_ids the controller should cancel
        self.configs_to_cancel = set()

        logger.info(
            f"HyperBandES initialized with early_stop_threshold={early_stop_threshold}, "
            f"min_early_stop_epochs={min_early_stop_epochs}"
        )

    @staticmethod
    def _power_law_model(x, a, b, c):
        """Power law learning curve model: y = a * x^b + c

        This model captures the typical behavior of neural network training
        where performance improves according to a power law.
        """
        return a * np.power(x, b) + c

    @staticmethod
    def _exponential_model(x, a, b, c):
        """Exponential learning curve model: y = a * exp(-b * x) + c

        Alternative model for learning curves that plateau exponentially.
        """
        return a * np.exp(-b * x) + c

    def _predict_final_performance(self, config_id, current_curve, prediction_horizon=None):
        """Predict final performance using learning curve extrapolation

        Uses observation counts as x-axis and extrapolates to a prediction
        horizon (default: 2x current observations) to estimate future performance.

        Args:
            config_id: configuration ID
            current_curve: list of (observation_count, metric) tuples
            prediction_horizon: observation count to predict at (default: 2x current)

        Returns:
            tuple: (predicted_final, confidence)
                - predicted_final: predicted metric at prediction horizon
                - confidence: confidence in prediction (0-1)
        """
        if len(current_curve) < self.min_epochs_for_prediction:
            return None, 0.0

        epochs = np.array([e for e, _ in current_curve], dtype=float)
        metrics = np.array([m for _, m in current_curve], dtype=float)

        if prediction_horizon is None:
            prediction_horizon = max(len(current_curve) * 2, self.min_epochs_for_prediction * 2)

        ss_tot = np.sum((metrics - np.mean(metrics)) ** 2)
        if ss_tot == 0:
            return None, 0.0

        best_predicted = None
        best_confidence = 0.0

        p0 = [metrics[0] - metrics[-1], -0.5, metrics[-1]]

        # Try power law: y = a * x^b + c
        try:
            popt, _ = curve_fit(self._power_law_model, epochs, metrics, p0=p0, maxfev=1000)
            predicted = self._power_law_model(prediction_horizon, *popt)
            ss_res = np.sum((metrics - self._power_law_model(epochs, *popt)) ** 2)
            r2 = max(0, min(1, 1 - ss_res / ss_tot))
            if r2 > best_confidence:
                best_predicted, best_confidence = predicted, r2
        except Exception:
            pass

        # Try exponential: y = a * exp(-b * x) + c
        try:
            p0_exp = [metrics[0] - metrics[-1], 0.5, metrics[-1]]
            popt, _ = curve_fit(self._exponential_model, epochs, metrics, p0=p0_exp, maxfev=1000)
            predicted = self._exponential_model(prediction_horizon, *popt)
            ss_res = np.sum((metrics - self._exponential_model(epochs, *popt)) ** 2)
            r2 = max(0, min(1, 1 - ss_res / ss_tot))
            if r2 > best_confidence:
                best_predicted, best_confidence = predicted, r2
        except Exception:
            pass

        if best_predicted is not None:
            logger.info(
                f"Config {config_id}: Predicted performance at horizon {prediction_horizon} "
                f"= {best_predicted:.4f} (confidence={best_confidence:.2f})"
            )

        return best_predicted, best_confidence

    def _should_early_stop(self, config_id, current_result):
        """Determine if a configuration should be stopped early

        Uses observation count (number of unique metric readings) as the
        x-axis for learning curve fitting, rather than epoch numbers which
        are not available from the history.

        Args:
            config_id: configuration ID
            current_result: current metric value

        Returns:
            bool: True if should stop early
        """
        if config_id in self.early_stopped_configs:
            return False

        if config_id not in self.learning_curves:
            self.learning_curves[config_id] = []

        # Skip if result hasn't changed (no new training progress)
        curve = self.learning_curves[config_id]
        if curve and curve[-1][1] == current_result:
            return False

        obs_count = len(curve) + 1
        curve.append((obs_count, current_result))

        if len(curve) < self.min_epochs_for_prediction:
            return False

        predicted_final, confidence = self._predict_final_performance(
            config_id, curve
        )

        if predicted_final is None or confidence < self.confidence_threshold:
            return False

        # Get best performance seen so far across all configs
        all_results = []
        for rec_id, other_curve in self.learning_curves.items():
            if rec_id != config_id and other_curve:
                all_results.append(other_curve[-1][1])

        if not all_results:
            return False

        if self.reverse_sort:
            current_best = max(all_results)
        else:
            current_best = min(all_results)

        if abs(current_best) < 1e-10:
            return False

        margin = 0.05
        if self.reverse_sort:
            should_stop = predicted_final < current_best * (1 - margin)
        else:
            should_stop = predicted_final > current_best * (1 + margin)

        if should_stop:
            self.early_stopped_configs.add(config_id)
            logger.info(
                f"Early stopping config {config_id}: predicted={predicted_final:.4f}, "
                f"current_best={current_best:.4f}, observations={obs_count}"
            )

        return should_stop

    def generate_recommendations(self, history):
        """Generates recommendations with predictive early stopping.

        Fixes three architectural issues with the original implementation:
        1. Signals cancellation via configs_to_cancel (not via deep-copy mutation)
        2. Excludes early-stopped configs from parent's any_running barrier
        3. Uses observation counts for learning curves (not rung target epochs)
        """
        self.configs_to_cancel = set()

        # Check for early stopping on running configs BEFORE calling parent
        for rec in history:
            if rec.status == JobStates.running and rec.result != 0.0:
                if self._should_early_stop(rec.id, rec.result):
                    logger.info(
                        f"Early stop triggered for config {rec.id} (job {rec.job_id})"
                    )
                    self.configs_to_cancel.add(rec.job_id)

        # Build filtered history for parent: early-stopped configs appear as
        # failures with penalty results so they're never promoted by SH
        filtered_history = []
        for rec in history:
            if rec.id in self.early_stopped_configs:
                from copy import copy
                rec_copy = copy(rec)
                rec_copy.status = JobStates.failure
                rec_copy.result = 1e7 if not self.reverse_sort else 1e-7
                filtered_history.append(rec_copy)
            else:
                filtered_history.append(rec)

        recommendations = super().generate_recommendations(filtered_history)

        return recommendations

    @staticmethod
    def load_state(job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                   metric="loss", min_epochs_for_prediction=3, confidence_threshold=0.8):
        """Load the HyperBandES algorithm related variables from brain metadata"""
        json_loaded = get_automl_brain_info(job_context.id)
        if not json_loaded:
            return HyperBandES(
                job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                min_epochs_for_prediction, confidence_threshold
            )

        brain = HyperBandES(
            job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
            min_epochs_for_prediction, confidence_threshold
        )
        # Load base HyperBand state
        brain.bracket = json_loaded["bracket"]
        brain.sh_iter = json_loaded["sh_iter"]
        brain.expt_iter = json_loaded["expt_iter"]
        brain.complete = json_loaded["complete"]
        brain.epoch_number = json_loaded["epoch_number"]
        brain.last_launched_count = json_loaded.get("last_launched_count", 0)

        # Load ES-specific state if available
        if "learning_curves" in json_loaded:
            # Convert string keys back to int if needed
            brain.learning_curves = {}
            for k, v in json_loaded["learning_curves"].items():
                try:
                    brain.learning_curves[int(k)] = v
                except (ValueError, TypeError):
                    brain.learning_curves[k] = v
        if "early_stopped_configs" in json_loaded:
            brain.early_stopped_configs = set(json_loaded["early_stopped_configs"])

        return brain

    def save_state(self):
        """Save the HyperBandES algorithm related variables to brain metadata"""
        super().save_state()

        from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_automl_brain_info

        state_dict = get_automl_brain_info(self.job_context.id)
        # Convert int keys to string for JSON serialization
        state_dict["learning_curves"] = {
            str(k): v for k, v in self.learning_curves.items()
        }
        state_dict["early_stopped_configs"] = list(self.early_stopped_configs)

        save_automl_brain_info(self.job_context.id, state_dict)
