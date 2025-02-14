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

"""Classification Default config file"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_core.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    DICT_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)
from nvidia_tao_core.config.common.common_config import (
    CommonExperimentConfig,
    TrainConfig,
    EvaluateConfig,
    GenTrtEngineConfig,
    InferenceConfig,
    TrtConfig,
    CalibrationConfig
)

@dataclass
class OptimConfig:
    """Optimizer config."""

    monitor_name: str = STR_FIELD(value="val_loss", default_value="val_loss", description="Monitor Name")
    optim: str = STR_FIELD(value="adamw", default_value="adamw", description="Optimizer", valid_options="adamw,adam,sgd")
    lr: float = FLOAT_FIELD(value=0.00006, default_value=0.00006, valid_min=0, valid_max="inf", automl_enabled="TRUE", description="Optimizer learning rate")
    policy: str = STR_FIELD(value="linear", default_value="linear", valid_options="linear,step", description="Optimizer policy")
    momentum: float = FLOAT_FIELD(value=0.9, default_value=0.9, math_cond="> 0.0", display_name="momentum - AdamW", description="The momentum for the AdamW optimizer.", automl_enabled="TRUE")
    weight_decay: float = FLOAT_FIELD(value=0.01, default_value=0.01, math_cond="> 0.0", display_name="weight decay", description="The weight decay coefficient.", automl_enabled="TRUE")

@dataclass
class LossConfig:
    """Loss config."""

    type: str = STR_FIELD(value="CrossEntropyLoss", default_value="CrossEntropyLoss", description="Loss type", valid_options="CrossEntropyLoss")
    label_smooth_val: float = FLOAT_FIELD(value=0.0, default_value=0.0, valid_min=0, valid_max=1, description="Label smoothing value")

@dataclass
class HeadConfig:
    """Configuration parameters for Head."""

    type: str = STR_FIELD(value="TAOLinearClsHead", value_type="ordered", valid_options="TAOLinearClsHead,LogisticRegressionHead", description="Type of classification head")
    binary: bool = BOOL_FIELD(value=False, description="Flag to specify binary classification")
    num_classes: int = INT_FIELD(value=1000, default_value=20, valid_min=2, valid_max="inf", description="Number of classes")
    in_channels: int = INT_FIELD(value=448, description="Number of backbone input channels to head")  # Mapped to differenct channels based according to the backbone used in the fan_model.py
    custom_args: Optional[Dict[Any, Any]] = DICT_FIELD(None, default_value=None, description="custom head arguments")
    loss: LossConfig = DATACLASS_FIELD(LossConfig())
    topk: List[int] = LIST_FIELD([1], description="k value for Topk accuracy")


@dataclass
class BackboneConfig:
    """Configuration parameters for Backbone."""

    type: str = STR_FIELD(value="fan_small_12_p4_hybrid", default_value="fan_small_12_p4_hybrid", description="Backbone architure", display_name="Backbone architectures", valid_options=",".join(["fan_tiny_8_p4_hybrid", "fan_large_16_p4_hybrid", "fan_small_12_p4_hybrid", "fan_base_16_p4_hybrid", "vit_large_nvdinov2", "vit_giant_nvdinov2", "vit_base_nvclip_16_siglip", "vit_huge_nvclip_14_siglip"]), automl_enabled="TRUE")
    feat_downsample: bool = BOOL_FIELD(value=False, default_value=False, display_name="Feature downsample", description="Feature downsample for fan base backbone")
    pretrained_backbone_path: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to the pretrained model")
    freeze_backbone: bool = BOOL_FIELD(value=False, default_value=False, description="Flag to freeze backbone", automl_enabled="TRUE")

@dataclass
class ModelConfig:
    """ Model config."""

    backbone: BackboneConfig = DATACLASS_FIELD(BackboneConfig())
    head: HeadConfig = DATACLASS_FIELD(HeadConfig())


@dataclass
class RandomFlip:
    """RandomFlip augmentation config."""

    vflip_probability: float = FLOAT_FIELD(value=0.5, default_value=0.5, valid_min=0, valid_max=1, description="Vertical Flip probability", automl_enabled="TRUE")
    hflip_probability: float = FLOAT_FIELD(value=0.5, default_value=0.5, valid_min=0, valid_max=1, description="Horizontal Flip probability", automl_enabled="TRUE")
    enable: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable augmentation", automl_enabled="TRUE")


@dataclass
class RandomRotation:
    """RandomRotation augmentation config."""

    rotate_probability: float = FLOAT_FIELD(value=0.5, default_value=0.5, valid_min=0, valid_max=1, description="Random Rotate probability", automl_enabled="TRUE")
    angle_list: List[float] = LIST_FIELD(arrList=[90, 180, 270], default_value=[90, 180, 270], description="Random rotate angle probability")
    enable: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable augmentation", automl_enabled="TRUE")


@dataclass
class RandomColor:
    """RandomColor augmentation config."""

    brightness: float = FLOAT_FIELD(value=0.3, default_value=0.3, math_cond="> 0.0", description="Random Color Brightness", automl_enabled="TRUE")
    contrast: float = FLOAT_FIELD(value=0.3, default_value=0.3, math_cond="> 0.0", description="Random Color Contrast", automl_enabled="TRUE")
    saturation: float = FLOAT_FIELD(value=0.3, default_value=0.3, math_cond="> 0.0", description="Random Color Saturation", automl_enabled="TRUE")
    hue: float = FLOAT_FIELD(value=0, default_value=0, math_cond="> 0.0", description="Random Color Hue", automl_enabled="TRUE")
    enable: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable Random Color", automl_enabled="TRUE")
    color_probability: float = FLOAT_FIELD(value=0.5, default_value=0.5, valid_min=0, valid_max=1, description="Random Color Probability", automl_enabled="TRUE")


@dataclass
class RandomCropWithScale:
    """RandomCropWithScale augmentation config."""

    scale_range: List[float] = LIST_FIELD(arrList=[1, 1.2], default_value=[1, 1.2], description="Random Scale range")  # non configurable here
    enable: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable Random Crop with Scale", automl_enabled="TRUE")


@dataclass
class AugmentationConfig:
    """Augmentation config."""

    random_flip: RandomFlip = DATACLASS_FIELD(RandomFlip())
    random_rotate: RandomRotation = DATACLASS_FIELD(RandomRotation())
    random_color: RandomColor = DATACLASS_FIELD(RandomColor())
    with_scale_random_crop: RandomCropWithScale = DATACLASS_FIELD(RandomCropWithScale())
    with_random_blur: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable with_random_blur")
    with_random_crop: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable with_random_crop")
    mean: List[float] = LIST_FIELD(arrList=[0.485, 0.456, 0.406], default_value=[0.485, 0.456, 0.406], description="Mean for the augmentation", display_name="Mean")  # non configurable here
    std: List[float] = LIST_FIELD(arrList=[0.229, 0.224, 0.225], default_value=[0.229, 0.224, 0.225], description="Standard deviation for the augmentation", display_name="Standard Deviation")  # non configurable here


@dataclass
class DataPathFormat:
    """Dataset Path experiment config."""

    csv_path: str = STR_FIELD(value=MISSING, default_value="", description="Path to csv file for dataset")
    images_dir: str = STR_FIELD(value=MISSING, default_value="", description="Path to images directory for dataset")


@dataclass
class TrainData:
    """Train Data Dataclass"""

    data_prefix: Optional[str] = STR_FIELD(value="", default_value="", description="Dataset directory path")


@dataclass
class ValData:
    """Validation Data Dataclass"""

    data_prefix: Optional[str] = STR_FIELD(value=None, default_value="", description="Dataset directory path")


@dataclass
class TestData:
    """Test Data Dataclass"""

    data_prefix: Optional[str] = STR_FIELD(value=None, default_value="", description="Dataset directory path")


@dataclass
class DatasetConfig:
    """Segmentation Dataset Config."""

    root_dir: str = STR_FIELD(value=MISSING, default_value="", description="Path to root directory for dataset")
    dataset: str = STR_FIELD(value="CLDataset", default_value="CLDataset", valid_options="Dataset", description="dataset class")
    num_classes: int = INT_FIELD(value=2, default_value=2, description="The number of classes in the training data", math_cond=">0", valid_min=2, valid_max="inf")
    img_size: int = INT_FIELD(value=224, default_value=224, description="The input image size")
    batch_size: int = INT_FIELD(value=8, default_value=8, valid_min=1, valid_max="inf", description="Batch size", display_name="Batch Size", automl_enabled="TRUE")
    workers: int = INT_FIELD(value=8, default_value=1, valid_min=0, valid_max="inf", description="Workers", display_name="Workers", automl_enabled="TRUE")
    shuffle: bool = BOOL_FIELD(value=True, default_value=True, description="Shuffle dataloader")
    augmentation: AugmentationConfig = DATACLASS_FIELD(AugmentationConfig())
    train: TrainData = DATACLASS_FIELD(TrainData())
    val: ValData = DATACLASS_FIELD(ValData())
    test: TestData = DATACLASS_FIELD(TestData())


@dataclass
class TensorBoardLogger:
    """Configuration for the tensorboard logger."""

    enabled: bool = BOOL_FIELD(value=False, default_value=False, description="Flag to enable tensorboard")
    infrequent_logging_frequency: int = INT_FIELD(value=2, default_value=2, valid_min=0, valid_max="inf", description="infrequent_logging_frequency")  # Defined per epoch


@dataclass
class TrainExpConfig(TrainConfig):
    """Train Config."""

    optim: OptimConfig = DATACLASS_FIELD(OptimConfig())
    pretrained_model_path: Optional[str] = STR_FIELD(value=None, default_value="", description="Pretrained model path", display_name="pretrained model path")
    tensorboard: Optional[TensorBoardLogger] = DATACLASS_FIELD(TensorBoardLogger())
    enable_ema: bool = BOOL_FIELD(value=False, default_value=False, description="Flag to enable EMA")


@dataclass
class EvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    vis_after_n_batches: int = INT_FIELD(value=16, default_value=1, valid_min=1, valid_max="inf", description="Visualize evaluation segmentation results after n batches")
    batch_size: int = INT_FIELD(value=-1, default_value=8, valid_min=1, valid_max="inf", description="Batch size", display_name="Batch Size")
    checkpoint: str = STR_FIELD(value=MISSING, default_value="", description="Path to checkpoint file", display_name="Path to checkpoint file")


@dataclass
class InferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    vis_after_n_batches: int = INT_FIELD(value=16, default_value=1, valid_min=1, valid_max="inf", description="Visualize evaluation segmentation results after n batches")
    batch_size: int = INT_FIELD(value=-1, default_value=8, valid_min=1, valid_max="inf", description="Batch size", display_name="Batch Size")
    checkpoint: str = STR_FIELD(value=MISSING, default_value="", description="Path to checkpoint file", display_name="Path to checkpoint file")


@dataclass
class ExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = STR_FIELD(value=None, default_value="", description="Results directory", display_name="Results directory")
    gpu_id: int = INT_FIELD(value=0, default_value=0, description="GPU ID", display_name="GPU ID", value_min=0)
    checkpoint: str = STR_FIELD(value=MISSING, default_value="", description="Path to checkpoint file", display_name="Path to checkpoint file")
    onnx_file: Optional[str] = STR_FIELD(value=MISSING, default_value="", description="ONNX file", display_name="ONNX file")
    on_cpu: bool = BOOL_FIELD(value=False, default_value=False, description="Flag to export on cpu", display_name="On CPU")
    input_channel: int = INT_FIELD(value=3, default_value=3, description="Input channel", display_name="Input channel")
    input_width: int = INT_FIELD(value=224, default_value=224, description="Input width", display_name="Input width", valid_min=128)
    input_height: int = INT_FIELD(value=224, default_value=224, description="Input height", display_name="Input height", valid_min=128)
    opset_version: int = INT_FIELD(value=17, default_value=12, valid_min=1, display_name="opset version", description="""Operator set version of the ONNX model used to generate the TensorRT engine.""")
    batch_size: int = INT_FIELD(value=-1, default_value=-1, description="Batch size", display_name="Batch size", valid_min=0)
    verbose: bool = BOOL_FIELD(value=False, default_value=False, description="Verbose", display_name="Verbose")


@dataclass
class TrtConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(value="FP32", default_value="fp16", description="Data type", display_name="Data type")
    calibration: CalibrationConfig = DATACLASS_FIELD(CalibrationConfig())


@dataclass
class GenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: TrtConfig = DATACLASS_FIELD(TrtConfig())


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: ModelConfig = DATACLASS_FIELD(ModelConfig())
    dataset: DatasetConfig = DATACLASS_FIELD(DatasetConfig())
    train: TrainExpConfig = DATACLASS_FIELD(TrainExpConfig())
    evaluate: EvalExpConfig = DATACLASS_FIELD(EvalExpConfig())
    inference: InferenceExpConfig = DATACLASS_FIELD(InferenceExpConfig())
    export: ExportExpConfig = DATACLASS_FIELD(ExportExpConfig())
    gen_trt_engine: GenTrtEngineExpConfig = DATACLASS_FIELD(GenTrtEngineExpConfig())
