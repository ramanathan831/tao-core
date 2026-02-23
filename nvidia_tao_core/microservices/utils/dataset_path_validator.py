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

"""Validates dataset paths and backend compatibility"""
import logging

logger = logging.getLogger(__name__)

VALID_PROTOCOLS = ["aws", "s3", "azure", "lustre", "file", "local"]
# Backends that do not allow local/file paths (must use cloud or shared filesystem)
REMOTE_BACKENDS = ["slurm", "lepton", "nvcf"]


def validate_dataset_path(path: str, backend_type: str = None) -> tuple:
    """Validates dataset path format and backend compatibility.

    Args:
        path: Dataset path with or without protocol prefix
        backend_type: Backend type string (local, slurm, lepton, nvcf)

    Returns:
        tuple: (is_valid, error_message)
    """
    if not path:
        return False, "Dataset path cannot be empty"

    # Parse protocol
    if "://" in path:
        protocol = path.split("://", 1)[0].lower()
    else:
        protocol = "local"

    # Normalize s3:// to aws://
    if protocol == "s3":
        protocol = "aws"

    # Validate protocol format
    if protocol not in VALID_PROTOCOLS:
        return False, f"Invalid protocol '{protocol}'. Supported: {', '.join(VALID_PROTOCOLS)}"

    # Backend-specific validation
    if backend_type:
        # Local/file paths not allowed for remote backends (slurm, lepton, nvcf)
        if backend_type in REMOTE_BACKENDS and protocol in ["file", "local"]:
            backend_specific_msg = {
                "slurm": "Use lustre:// for shared filesystem access.",
                "lepton": "Use cloud storage (aws://, azure://) with workspace credentials.",
                "nvcf": "Use cloud storage (aws://, azure://) with workspace credentials."
            }
            hint = backend_specific_msg.get(backend_type, "Use cloud storage or shared filesystem.")
            return False, (
                f"Local paths (file:// or no prefix) not allowed for {backend_type.upper()} backend. "
                f"{hint}"
            )

    return True, ""


def validate_all_dataset_paths(experiment_metadata: dict, backend_type: str = None) -> tuple:
    """Validates all dataset paths in experiment metadata.

    Args:
        experiment_metadata: Experiment metadata dictionary
        backend_type: Backend type string for validation

    Returns:
        tuple: (is_valid, error_message)
    """
    # Collect all paths to validate
    paths_to_validate = []

    train_paths = experiment_metadata.get("train_dataset_paths", [])
    if train_paths:
        paths_to_validate.extend([(p, "train_dataset_paths") for p in train_paths])

    eval_path = experiment_metadata.get("eval_dataset_path")
    if eval_path:
        paths_to_validate.append((eval_path, "eval_dataset_path"))

    inference_path = experiment_metadata.get("inference_dataset_path")
    if inference_path:
        paths_to_validate.append((inference_path, "inference_dataset_path"))

    calibration_path = experiment_metadata.get("calibration_dataset_path")
    if calibration_path:
        paths_to_validate.append((calibration_path, "calibration_dataset_path"))

    # Validate each path
    for path, field_name in paths_to_validate:
        is_valid, error_msg = validate_dataset_path(path, backend_type)
        if not is_valid:
            return False, f"Invalid {field_name}: {error_msg}"

    return True, ""
