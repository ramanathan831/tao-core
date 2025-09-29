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

from typing import List, Optional, Union
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    BOOL_FIELD,
    FLOAT_FIELD,
    STR_FIELD,
    INT_FIELD,
    DATACLASS_FIELD,
    LIST_FIELD,
    UNION_FIELD
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

    media_path: Optional[str] = STR_FIELD(
        default_value="data/sft/train2017",
        value="data/sft/train2017",
        display_name="Media directory path",
        description="Path to the media directory"
    )
    system_prompt: Optional[str] = STR_FIELD(
        default_value="",
        value="",
        display_name="System prompt",
        description="System prompt."
    )


@dataclass
class LoggingConfig:
    """Validation config."""

    logger: List[str] = LIST_FIELD(
        arrList=["console", "tao"],
        valid_options=["console", "tao"],
        display_name="Logger",
        description="Logger to use."
    )
    project_name: str = STR_FIELD(
        value="cosmos-rl",
        default_value="cosmos-rl",
        display_name="Project name",
        description="Project name."
    )
    experiment_name: str = STR_FIELD(
        value="cosmos-rl",
        default_value="cosmos-rl",
        display_name="Experiment name",
        description="Experiment name."
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
    save_freq_in_epoch: int = INT_FIELD(
        value=10,
        default_value=10,
        valid_min=1,
        valid_max="inf",
        display_name="Save frequency",
        description="Save every N epochs."
    )
    save_mode: str = STR_FIELD(
        value="sync",
        default_value="sync",
        valid_options="async,sync",
        display_name="Save mode",
        description="Checkpoint save mode for training."
    )
    max_keep: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=-1,
        valid_max="inf",
        display_name="Max keep",
        description="Maximum number of checkpoints to keep. If set to -1, all checkpoints will be kept."
    )
    export_safetensors: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="Export safetensors",
        description="Export HuggingFace compatible format."
    )


@dataclass
class TrainPolicyDatasetConfig:
    """Train policy dataset config."""

    name: str = STR_FIELD(
        value="its",
        default_value="its",
        display_name="Dataset name",
        description="Name of the dataset."
    )

    test_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="Test size",
        description="Size of the test dataset."
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

    enable_dataset_cache: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="Enable dataset cache",
        description="Enable dataset caching for faster loading."
    )

    dataloader_num_workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=0,
        valid_max="inf",
        display_name="Dataloader num workers",
        description="Number of worker processes for data loading."
    )

    dataloader_prefetch_factor: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        display_name="Dataloader prefetch factor",
        description="Number of batches to prefetch per worker."
    )

    conversation_column_name: str = STR_FIELD(
        value="conversations",
        default_value="conversations",
        display_name="Conversation column name",
        description="Name of the column containing conversations in the dataset."
    )

    dataset: TrainPolicyDatasetConfig = DATACLASS_FIELD(TrainPolicyDatasetConfig(), description="Dataset config.")


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

    resume: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Resume",
        description="Whether to resume training."
    )

    epoch: int = INT_FIELD(
        value=10,
        default_value=10,
        valid_min=10,
        valid_max=20,
        display_name="Number of Epochs",
        description="The number of epochs.",
        popular="yes",
        parent_param="TRUE",
        automl_enabled="TRUE",
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
        description="Learning rate.",
        automl_enabled="TRUE"
    )

    optm_impl: str = STR_FIELD(
        value="foreach",
        default_value="foreach",
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

    epsilon: float = FLOAT_FIELD(
        value=1e-8,
        default_value=1e-8,
        valid_min=0,
        valid_max="inf",
        display_name="Epsilon",
        description="Epsilon value for optimizer."
    )

    optm_name: str = STR_FIELD(
        value="AdamW",
        default_value="AdamW",
        valid_options="AdamW,Adam",
        display_name="Optimizer name",
        description="Name of the optimizer to use.",
        automl_enabled="TRUE"
    )

    optm_betas: List[float] = LIST_FIELD(
        arrList=[0.9, 0.999],
        display_name="Optimizer betas",
        description="Beta parameters for Adam/AdamW optimizer.",
        automl_enabled="TRUE",
        value_type="list_2"
    )

    optm_warmup_epochs: Optional[Union[int, float]] = UNION_FIELD(
        value=0,
        union_types=["int", "float", "NoneType"],
        default_value=0,
        valid_min=0,
        valid_max="inf",
        math_cond="/ 2",
        display_name="Warmup epochs",
        description="Number of warmup epochs for learning rate scheduler (epochs / 2).",
        automl_enabled="TRUE",
        depends_on="train.epoch"
    )

    async_tp_enabled: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Async TP enabled",
        description="Enable asynchronous tensor parallelism."
    )

    master_dtype: str = STR_FIELD(
        value="float32",
        default_value="float32",
        valid_options="float32,float16,bfloat16",
        display_name="Master dtype",
        description="Master data type for training."
    )

    param_dtype: str = STR_FIELD(
        value="bfloat16",
        default_value="bfloat16",
        valid_options="float32,float16,bfloat16",
        display_name="Parameter dtype",
        description="Parameter data type for training."
    )

    fsdp_reduce_dtype: str = STR_FIELD(
        value="float32",
        default_value="float32",
        valid_options="float32,float16,bfloat16",
        display_name="FSDP reduce dtype",
        description="Data type for FSDP reduction operations."
    )

    fsdp_offload: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="FSDP offload",
        description="Enable FSDP parameter offloading."
    )

    fsdp_reshard_after_forward: str = STR_FIELD(
        value="default",
        default_value="default",
        valid_options="default,true,false",
        display_name="FSDP reshard after forward",
        description="FSDP reshard after forward pass."
    )

    sync_weight_interval: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="Sync weight interval",
        description="Interval for weight synchronization."
    )

    ckpt: TrainCheckpointConfig = DATACLASS_FIELD(TrainCheckpointConfig(), description="Train checkpoint config.")
    train_policy: TrainPolicyConfig = DATACLASS_FIELD(TrainPolicyConfig(), description="Train policy config.")
    fp8: TrainFP8Config = DATACLASS_FIELD(TrainFP8Config(), description="Train FP8 config.")


@dataclass
class ValidationConfig:
    """Validation config."""

    enable: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="Enable validation",
        description="Whether to enable validation."
    )
    freq_in_epoch: int = INT_FIELD(
        value=10,
        default_value=10,
        valid_min=1,
        valid_max="inf",
        display_name="Validation frequency",
        description="Validation frequency."
    )


@dataclass
class PolicyParallelismConfig:
    """Policy parallelism config."""

    n_init_replicas: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="N init replicas",
        description="Number of initial replicas."
    )

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

    dp_replicate_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="DP replicate size",
        description="DP replicate size."
    )

    pp_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="PP size",
        description="PP size."
    )

    cp_rotate_method: str = STR_FIELD(
        value="allgather",
        default_value="allgather",
        valid_options="allgather,p2p",
        display_name="CP rotate method",
        description="Context parallelism rotation method."
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

    model_gradient_checkpointing: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="Model gradient checkpointing",
        description="Enable gradient checkpointing to save memory during training."
    )

    parallelism: PolicyParallelismConfig = DATACLASS_FIELD(
        PolicyParallelismConfig(), description="Policy parallelism config."
    )


@dataclass
class VisionConfig:
    """Vision config."""

    fps: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max=3,
        display_name="FPS",
        description="Frames per second for vision processing.",
        automl_enabled="TRUE"
    )

    total_pixels: int = INT_FIELD(
        value=313600,
        default_value=313600,
        valid_min=1,
        valid_max="inf",
        display_name="Total pixels",
        description="Total number of pixels for vision processing."
    )


@dataclass
class CustomConfig:
    """Custom config."""

    dataset: DatasetConfig = DATACLASS_FIELD(DatasetConfig(), description="Dataset config.")
    vision: VisionConfig = DATACLASS_FIELD(VisionConfig(), description="Vision config.")


@dataclass
class ExperimentConfig:
    """Experiment config."""

    train: TrainConfig = DATACLASS_FIELD(TrainConfig(), description="Train config.")
    validation: ValidationConfig = DATACLASS_FIELD(ValidationConfig(), description="Validation config.")
    policy: PolicyConfig = DATACLASS_FIELD(PolicyConfig(), description="Policy config.")
    logging: LoggingConfig = DATACLASS_FIELD(LoggingConfig(), description="Logging config.")
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
