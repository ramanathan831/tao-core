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

"""Common config fields across all models"""

from dataclasses import dataclass

from omegaconf import MISSING
from typing import Optional, List

from nvidia_tao_core.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)
from nvidia_tao_core.config.common.mlops import WandBConfig


@dataclass
class CuDNNConfig:
    """Common CuDNN config"""

    benchmark: bool = BOOL_FIELD(value=False)
    deterministic: bool = BOOL_FIELD(value=True)


@dataclass
class TrainConfig:
    """Common train experiment config."""

    num_gpus: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Number of GPUs",
        description="""The number of GPUs to run the train job.""",
    )
    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        display_name="GPU IDs",
        description="""
        List of GPU IDs to run the training on. The length of this list
        must be equal to the number of gpus in train.num_gpus.""")
    num_nodes: int = INT_FIELD(
        value=1,
        display_name="Number of nodes",
        description="Number of nodes to run the training on. If > 1, then multi-node is enabled.",
        valid_min=1,
    )
    seed: int = INT_FIELD(
        value=1234,
        default_value=1234,
        valid_min=-1,
        valid_max="inf",
        description="The seed for the initializer in PyTorch. If < 0, disable fixed seed.",
        display_name="Seed for randomization",
    )
    cudnn: CuDNNConfig = DATACLASS_FIELD(CuDNNConfig())

    num_epochs: int = INT_FIELD(
        value=10,
        valid_min=1,
        valid_max="inf",
        description="Number of epochs to run the training.",
        display_name="Number of epochs",
    )
    checkpoint_interval: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Checkpoint interval",
        description="The interval (in epochs) at which a checkpoint will be saved. Helps resume training.",
    )
    validation_interval: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Validation interval",
        description="""
        The interval (in epochs) at which a evaluation
        will be triggered on the validation dataset.""",
    )

    resume_training_checkpoint_path: Optional[str] = STR_FIELD(
        value=None,
        description="Path to the checkpoint to resume training from.",
        display_name="Resume checkpoint path."
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """)


@dataclass
class EvaluateConfig:
    """Common eval experiment config."""

    num_gpus: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Number of GPUs",
        description="""The number of GPUs to run the evaluation job.""",
    )
    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        display_name="GPU IDs",
        description="""
        List of GPU IDs to run the evaluation on. The length of this list
        must be equal to the number of gpus in evaluate.num_gpus.""")
    num_nodes: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Number of nodes",
        description="Number of nodes to run the evaluation on. If > 1, then multi-node is enabled.",
    )
    checkpoint: str = STR_FIELD(
        value=MISSING,
        description="Path to the checkpoint used for evaluation.",
        display_name="Checkpoint path."
    )
    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        description="""Path to the TensorRT engine to be used for evaluation.
                    This only works with :code:`tao-deploy`.""",
        display_name="TensorRT Engine"
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """)


@dataclass
class InferenceConfig:
    """Common inference experiment config."""

    num_gpus: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Number of GPUs",
        description="""The number of GPUs to run the inference job.""",
    )
    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        display_name="GPU IDs",
        description="""
        List of GPU IDs to run the inference on. The length of this list
        must be equal to the number of gpus in inference.num_gpus.""")
    num_nodes: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Number of nodes",
        description="Number of nodes to run the inference on. If > 1, then multi-node is enabled.",
    )
    checkpoint: str = STR_FIELD(
        value=MISSING,
        description="Path to the checkpoint used for inference.",
        display_name="Checkpoint path."
    )
    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        description="""Path to the TensorRT engine to be used for inference.
                    This only works with :code:`tao-deploy`.""",
        display_name="TensorRT Engine"
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """)


# TAO Deploy configs

@dataclass
class CalibrationConfig:
    """Calibration config."""

    cal_image_dir: List[str] = LIST_FIELD(
        arrList=MISSING,
        display_name="Calibration image directories",
        description="""List of image directories to be used for calibration
                    when running Post Training Quantization using TensorRT.""",
    )
    cal_cache_file: str = STR_FIELD(
        value=MISSING,
        display_name="Calibration cache file",
        description="""The path to save the calibration cache file containing
                    scales that were generated during Post Training Quantization.""",
    )
    cal_batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="""The batch size of the input TensorRT to run calibration on.""",
        display_name="Calibration batch size",
    )
    cal_batches: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="""The number of input tensor batches to run calibration on.
                    It is recommended to use atleast 10% of the training images.""",
        display_name="Number of calibration batches",
    )


@dataclass
class TrtConfig:
    """Trt config."""

    workspace_size: int = INT_FIELD(
        value=1024,
        default_value=1024,
        valid_min=0,
        description="""The size (in MB) of the workspace TensorRT has
                    to run it's optimization tactics and generate the
                    TensorRT engine.""",
        display_name="Max workspace size",
    )
    min_batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="""The minimum batch size in the optimization profile for
                    the input tensor of the TensorRT engine.""",
        display_name="Min batch size",
    )
    opt_batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="""The optimum batch size in the optimization profile for
                    the input tensor of the TensorRT engine.""",
        display_name="Optimum batch size",
    )
    max_batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="""The maximum batch size in the optimization profile for
                    the input tensor of the TensorRT engine.""",
        display_name="Maximum batch size",
    )


@dataclass
class GenTrtEngineConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        description="""The index of the GPU to build the TensorRT engine.""",
        display_name="GPU ID",
    )
    onnx_file: str = STR_FIELD(
        value=MISSING,
        display_name="ONNX file",
        description="""
        Path to the ONNX model file.
        """
    )
    trt_engine: Optional[str] = STR_FIELD(
        value=MISSING,
        description="""Path to the TensorRT engine generated should be stored.
                    This only works with :code:`tao-deploy`.""",
        display_name="TensorRT engine"
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-1,
        description="""The batch size of the input Tensor for the engine.
                    A value of :code:`-1` implies dynamic tensor shapes.""",
        display_name="Batch size"
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Verbose",
        description="""Flag to enable verbose TensorRT logging."""
    )
    # TODO @seanf: remove these after TRT upgrade as they're unnecessary
    input_channel: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=3,
        valid_max=3,
        description="Input channel.",
        display_name="Input Channel"
    )
    input_width: int = INT_FIELD(
        value=512,
        default_value=512,
        valid_min=512,
        valid_max=512,
        description="Input width.",
        display_name="Input Width"
    )
    input_height: int = INT_FIELD(
        value=512,
        default_value=512,
        valid_min=512,
        valid_max=512,
        description="Input height.",
        display_name="Input Height"
    )
    opset_version: int = INT_FIELD(
        value=16,
        default_value=16,
        valid_min=16,
        valid_max=16,
        description="ONNX opset version.",
        display_name="Opset Version",
        popular="16"
    )


@dataclass
class CommonExperimentConfig:
    """Common experiment config."""

    model_name: Optional[str] = STR_FIELD(
        value=None,
        display_name="Model name",
        description="Name of model if invoking task via :code:`model_agnostic`"
    )
    encryption_key: Optional[str] = STR_FIELD(
        value=None,
        display_name="Encryption key",
        description="Key for encrypting model checkpoints"
    )
    results_dir: Optional[str] = STR_FIELD(
        value="/results",
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """
    )
    wandb: WandBConfig = DATACLASS_FIELD(
        WandBConfig(
            project="TAO Toolkit",
            name="TAO Toolkit training experiment",
            tags=["training", "tao-toolkit"]
        )
    )
