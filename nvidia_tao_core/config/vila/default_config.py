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
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    DATACLASS_FIELD
)


@dataclass
class TrainerConfig:
    """Train Config."""

    num_epochs: int = INT_FIELD(value=1, default_value=1, valid_min=1, valid_max="inf", display_name="Number of Epochs", description="The number of epochs.", popular="yes")
    batch_size: int = INT_FIELD(value=8, default_value=8, valid_min=1, valid_max="inf", display_name="Batch size", description="The batch size during training.", popular="yes")
    learning_rate: float = FLOAT_FIELD(value=1e-5, default_value=1e-5, valid_min=0, valid_max="inf", description="Learning rate. Default is 1e-5.", display_name="Learning rate", popular="yes")
    weight_decay: float = FLOAT_FIELD(value=0., default_value=0., display_name="Weight decay", description="The weight decay coefficient.", popular="yes")
    warmup_ratio: float = FLOAT_FIELD(value=0.03, default_value=0.03, valid_min=0, valid_max="inf", description="Warmup ratio.", display_name="Warmup ratio", popular="yes")
    gradient_accumulation_steps: int = INT_FIELD(value=2, default_value=2, valid_min=1, valid_max="inf", display_name="Gradient accumulation steps.", description="Gradient accumulation steps. Your effective batch size is gradient_accumulation_steps * batch_size * num_gpus * num_nodes", popular="yes")


@dataclass
class SystemConfig:
    """GPU and Multinode System config."""

    num_gpus: Optional[int] = INT_FIELD(value=1, default_value=1, description="Number of gpus. Should range from 1 - 8.", display_name="Number of GPUs", valid_min=1, popular="yes")
    num_nodes: Optional[int] = INT_FIELD(value=1, default_value=1, description="Number of nodes. Only set if using multinode.", display_name="Number of Nodes", valid_min=1, popular="yes")
    master_addr: Optional[str] = STR_FIELD(value="127.0.0.1", default_value="127.0.0.1", description="Master address. Only set if using multinode.", display_name="Master address")
    node_rank: Optional[int] = INT_FIELD(value=0, default_value=0, description="Node rank. Only set if using multinode.", display_name="Node rank", valid_min=0)
    port: Optional[int] = INT_FIELD(value=24501, default_value=24501, description="Port number. Default is 24501", display_name="Port number")


@dataclass
class ExperimentConfig:
    """Experiment config."""

    model_path: str = STR_FIELD(value="/models/vila", default_value="/models/vila", display_name="Pretrained model path", description="Pretrained model path")
    results_dir: str = STR_FIELD(value="/results/lora", default_value="/results/lora", display_name="Output directory", description="Output directory. Must contain `lora` in the output name.")
    dataset_name: Optional[str] = STR_FIELD(value="scienceqa", default_value="scienceqa", display_name="Dataset name", description="Dataset name. Default is scienceqa. Dataset name must be registered at `llava/data/registry/datasets/default.yaml`.")

    llm_mode: Optional[str] = STR_FIELD(value="lora", default_value="lora", valid_options="freeze,ft,lora", display_name="LLM mode", description="LLM mode: freeze, ft, or lora. Default is lora")
    vision_mode: Optional[str] = STR_FIELD(value="ft", default_value="ft", valid_options="freeze,ft,lora", display_name="Vision tower mode", description="Vision tower mode: freeze, ft, or lora. Default is ft")
    disable_wandb: Optional[str] = STR_FIELD(value="true", default_value="true", valid_options="true,false", display_name="Vision tower mode", description="Enable or disable wandb logging")

    trainer: TrainerConfig = DATACLASS_FIELD(TrainerConfig(), description="Trainer config")
    system: SystemConfig = DATACLASS_FIELD(SystemConfig(), description="GPU and Multinode System config")
