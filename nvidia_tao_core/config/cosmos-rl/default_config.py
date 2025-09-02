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

"""Default config file"""

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    BOOL_FIELD,
    FLOAT_FIELD,
    STR_FIELD,
    INT_FIELD,
    DATACLASS_FIELD
)


@dataclass
class DatasetConfig:
    """Dataset config."""

    annotation_path: Optional[str] = STR_FIELD(
        default_value="data/sft/annotations.json",
        value="data/sft/annotations.json",
        display_name="Annotation path",
        description="Path to the annotation file"
    )

    media_dir: Optional[str] = STR_FIELD(
        default_value="data/sft/train2017",
        value="data/sft/train2017",
        display_name="Media directory path",
        description="Path to the media directory"
    )


@dataclass
class TrainCheckpointConfig:
    """Train checkpoint config."""

    enable_checkpoint: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="Enable checkpoint",
        description="Enable checkpoint."
    )


@dataclass
class TrainPolicyConfig:
    """Train policy config."""

    type: str = STR_FIELD(
        value="sft",
        default_value="sft",
        valid_options="sft",
        display_name="Type",
        description="Type of policy."
    )

    mini_batch: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        valid_max="inf",
        display_name="Mini batch",
        description="Mini batch."
    )


@dataclass
class TrainFP8Config:
    """Train FP8 config."""

    enable_fp8: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Enable FP8",
        description="Enable FP8."
    )

    fp8_recipe: str = STR_FIELD(
        value="dynamic_scaling",
        default_value="dynamic_scaling",
        valid_options="dynamic_scaling,delayed_scaling",
        display_name="FP8 recipe",
        description="Recipe for weight scale calculation."
    )

    quant_recipe: str = STR_FIELD(
        value="rowwise",
        default_value="rowwise",
        valid_options="rowwise,tensorwise",
        display_name="Quant recipe",
        description="Quantization strategy for weight."
    )


@dataclass
class TrainConfig:
    """Train Config."""

    epoch: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="Number of Epochs",
        description="The number of epochs.",
        popular="yes"
    )

    compile: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Compile",
        description="Whether to compile the model.",
        popular="yes"
    )

    train_batch_per_replica: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="Train batch per replica",
        description="The number of batches per replica during training.",
        popular="yes"
    )

    output_dir: str = STR_FIELD(
        value="output",
        default_value="output",
        display_name="Output directory",
        description="Output directory."
    )

    optm_lr: float = FLOAT_FIELD(
        value=1e-6,
        default_value=1e-6,
        valid_min=0,
        valid_max="inf",
        display_name="Learning rate",
        description="Learning rate."
    )

    optm_impl: str = STR_FIELD(
        value="fused",
        default_value="fused",
        valid_options="fused,foreach,for-loop",
        display_name="Implementation type",
        description="Implementation type for optimizer. More info: https://pytorch.org/docs/stable/optim.html",
    )

    optm_weight_decay: float = FLOAT_FIELD(
        value=0.01,
        default_value=0.01,
        valid_min=0,
        valid_max="inf",
        display_name="Weight decay",
        description="Weight decay."
    )

    optm_min_lr_factor: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0,
        valid_max="inf",
        display_name="Minimum learning rate factor",
        description="Minimum learning rate factor."
    )

    optm_grad_norm_clip: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0,
        valid_max="inf",
        display_name="Gradient norm clip",
        description="Gradient norm clip."
    )

    epoch: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="Number of Epochs",
        description="The number of epochs.",
        popular="yes"
    )

    ckpt: TrainCheckpointConfig = DATACLASS_FIELD(TrainCheckpointConfig(), description="Train checkpoint config.")
    train_policy: TrainPolicyConfig = DATACLASS_FIELD(TrainPolicyConfig(), description="Train policy config.")
    fp8: TrainFP8Config = DATACLASS_FIELD(TrainFP8Config(), description="Train FP8 config.")


@dataclass
class PolicyParallelismConfig:
    """Policy parallelism config."""

    tp_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="TP size",
        description="TP size."
    )

    cp_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="CP size",
        description="CP size."
    )

    dp_shard_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="DP shard size",
        description="DP shard size."
    )

    pp_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="PP size",
        description="PP size."
    )


@dataclass
class PolicyConfig:
    """Policy config."""

    model_name_or_path: str = STR_FIELD(
        value="nvidia/Cosmos-Reason1-7B",
        default_value="nvidia/Cosmos-Reason1-7B",
        display_name="Model name or path",
        description="Model name or path."
    )

    model_max_length: int = INT_FIELD(
        value=4096,
        default_value=4096,
        valid_min=1,
        valid_max="inf",
        display_name="Model max length",
        description="Model max length."
    )
    parallelism: PolicyParallelismConfig = DATACLASS_FIELD(
        PolicyParallelismConfig(), description="Policy parallelism config."
    )


@dataclass
class CustomConfig:
    """Custom config."""

    dataset: DatasetConfig = DATACLASS_FIELD(DatasetConfig(), description="Dataset config.")


@dataclass
class ExperimentConfig:
    """Experiment config."""

    train: TrainConfig = DATACLASS_FIELD(TrainConfig(), description="Train config.")
    policy: PolicyConfig = DATACLASS_FIELD(PolicyConfig(), description="Policy config.")
    redis: str = STR_FIELD(
        value="12800",
        default_value="12800",
        display_name="Redis",
        description="Redis."
    )
    results_dir: str = STR_FIELD(
        value="/results",
        default_value="/results",
        display_name="Output directory",
        description="Output directory."
    )
    custom: CustomConfig = DATACLASS_FIELD(CustomConfig(), description="Custom config.")
