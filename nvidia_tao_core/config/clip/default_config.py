# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""CLIP experiment configuration."""

from dataclasses import dataclass
from typing import List, Optional
from omegaconf import MISSING

from nvidia_tao_core.config.common.common_config import (
    CommonExperimentConfig,
    GenTrtEngineConfig,
    TrainConfig,
    TrtConfig,
)
from nvidia_tao_core.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)


# =============================================================================
# Model Config
# =============================================================================
@dataclass
class CLIPModelConfig:
    """CLIP model configuration."""

    type: str = STR_FIELD(
        value="siglip2-so400m-patch16-256",
        default_value="siglip2-so400m-patch16-256",
        description="CLIP model type. "
                    "C-RADIO: c-radio_v3-h, c-radio_v3-l, c-radio_v3-b, c-radio_v3-g; "
                    "SigLIP2: siglip2-so400m-patch16-naflex (NaFlex), siglip2-so400m-patch14-224, "
                    "siglip2-so400m-patch14-384, siglip2-so400m-patch16-256, "
                    "siglip2-so400m-patch16-384, siglip2-so400m-patch16-512; "
                    "OpenCLIP: ViT-L-14-SigLIP-CLIPA-224, ViT-L-14-SigLIP-CLIPA-336, "
                    "ViT-H-14-SigLIP-CLIPA-224.",
        display_name="Model Type",
    )
    adaptor_name: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Text adaptor for C-RADIO models (ignored for other model types). "
                    "'siglip' (SigLIP2 text encoder) or 'clip' (DFN CLIP text encoder). "
                    "When None, defaults to 'siglip' at runtime.",
        display_name="Adaptor Name",
    )
    freeze_vision_encoder: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="If True, freeze vision encoder weights during training.",
        display_name="Freeze Vision Encoder",
    )
    freeze_text_encoder: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="If True, freeze text encoder weights during training.",
        display_name="Freeze Text Encoder",
    )
    image_size: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Input image resolution for training transforms. "
                    "Common values: 224 (RADIO/OpenCLIP), 384 (SigLIP2-g), "
                    "256 (SigLIP2-so400m). "
                    "Must be a multiple of the model's patch size (typically 14 or 16).",
        display_name="Image Size",
    )
    init_logit_scale: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        description="Override for the initial logit scale (log-space). "
                    "When None, automatically set from train.loss_type: "
                    "2.3026 (SigLIP) or 2.6592 (CLIP). "
                    "Set manually only with caution, as incorrect values "
                    "can destabilize training.",
        display_name="Initial Logit Scale",
    )
    init_logit_bias: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        description="Override for the initial logit bias. "
                    "When None, automatically set from train.loss_type: "
                    "-10.0 (SigLIP) or 0.0 (CLIP). "
                    "Set manually only with caution, as incorrect values "
                    "can destabilize training.",
        display_name="Initial Logit Bias",
    )


# =============================================================================
# Dataset Config
# =============================================================================
@dataclass
class CLIPAugmentationConfig:
    """Data augmentation configuration for CLIP training.

    To disable augmentations:
        - scale: [1.0, 1.0]       -> disables random resize crop scaling
        - color_jitter: []        -> disables color jitter
        - grayscale: 0.0          -> disables grayscale
    """

    scale: List[float] = LIST_FIELD(
        arrList=[0.4, 1.0],
        default_value=[0.4, 1.0],
        description="Scale range [min, max] for random resized crop. Set to [1.0, 1.0] to disable.",
        display_name="Scale Range",
    )
    color_jitter: List[float] = LIST_FIELD(
        arrList=[0.8, 0.32, 0.32, 0.32, 0.08],
        default_value=[0.8, 0.32, 0.32, 0.32, 0.08],
        description="Color jitter [prob, brightness, contrast, saturation, hue]. Set to [] to disable.",
        display_name="Color Jitter",
    )
    grayscale: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        valid_min=0.0,
        valid_max=1.0,
        description="Probability of grayscale conversion. Set to 0.0 to disable.",
        display_name="Grayscale",
    )


@dataclass
class CLIPDataPathConfig:
    """Dataset path configuration for custom image-text datasets."""

    root_dir: str = STR_FIELD(
        value=MISSING,
        default_value=MISSING,
        description="Root directory containing the images.",
        display_name="Root Directory",
    )
    image_list_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to text file listing image filenames. If None, all images in root_dir are used.",
        display_name="Image List File",
    )
    label_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Directory containing text caption files. If None, uses root_dir.",
        display_name="Label Directory",
    )
    label_suffix: str = STR_FIELD(
        value=".txt",
        default_value=".txt",
        description="File extension for caption files (e.g., '.txt').",
        display_name="Label Suffix",
    )


@dataclass
class CLIPWDSConfig:
    """WebDataset (sharded) configuration for large-scale CLIP training."""

    root_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Root directory containing WebDataset shards (required when type='wds').",
        display_name="Root Directory",
    )
    shard_list_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to text file listing shard URLs/paths.",
        display_name="Shard List File",
    )
    samples_per_shard: int = INT_FIELD(
        value=10000,
        default_value=10000,
        valid_min=1,
        description="Number of samples per shard (used for progress tracking).",
        display_name="Samples per Shard",
    )


@dataclass
class CLIPTrainDataConfig:
    """Training data configuration."""

    type: str = STR_FIELD(
        value="custom",
        default_value="custom",
        valid_options="wds,custom",
        description="Dataset type: 'custom' for filesystem-based or 'wds' for WebDataset.",
        display_name="Dataset Type",
    )
    datasets: List[CLIPDataPathConfig] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description="List of dataset path configurations (used when type='custom').",
        display_name="Datasets",
    )
    wds: Optional[CLIPWDSConfig] = DATACLASS_FIELD(
        CLIPWDSConfig(),
        default_value=CLIPWDSConfig(),
        description="WebDataset configuration (used when type='wds').",
    )
    batch_size: int = INT_FIELD(
        value=128,
        default_value=128,
        valid_min=1,
        description="Training batch size per GPU.",
        display_name="Batch Size",
    )
    num_workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=0,
        description="Number of data loading worker processes.",
        display_name="Number of Workers",
    )


@dataclass
class CLIPValDataConfig:
    """Validation data configuration."""

    type: str = STR_FIELD(
        value="classification",
        default_value="classification",
        valid_options="classification,custom",
        description="Validation dataset type: 'classification' (torchvision ImageFolder) or 'custom'.",
        display_name="Dataset Type",
    )
    dataset: CLIPDataPathConfig = DATACLASS_FIELD(
        CLIPDataPathConfig(),
        default_value=CLIPDataPathConfig(),
        description="Validation dataset path configuration.",
    )
    batch_size: int = INT_FIELD(
        value=64,
        default_value=64,
        valid_min=1,
        description="Validation batch size per GPU.",
        display_name="Batch Size",
    )
    num_workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=0,
        description="Number of data loading worker processes.",
        display_name="Number of Workers",
    )
    split: str = STR_FIELD(
        value="val",
        default_value="val",
        description="Dataset split to use for zero-shot evaluation (e.g., 'train', 'val').",
        display_name="Split",
    )
    templates_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to custom prompt templates file for zero-shot classification.",
        display_name="Templates File",
    )
    classnames_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to custom class names file for zero-shot classification.",
        display_name="Class Names File",
    )


@dataclass
class CLIPDatasetConfig:
    """Dataset configuration for CLIP training and evaluation."""

    train: CLIPTrainDataConfig = DATACLASS_FIELD(
        CLIPTrainDataConfig(),
        default_value=CLIPTrainDataConfig(),
        description="Training dataset configuration.",
    )
    val: CLIPValDataConfig = DATACLASS_FIELD(
        CLIPValDataConfig(),
        default_value=CLIPValDataConfig(),
        description="Validation dataset configuration.",
    )
    augmentation: CLIPAugmentationConfig = DATACLASS_FIELD(
        CLIPAugmentationConfig(),
        default_value=CLIPAugmentationConfig(),
        description="Data augmentation configuration.",
    )
    pin_memory: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Pin memory in DataLoader for faster GPU transfer.",
        display_name="Pin Memory",
    )
    seed: int = INT_FIELD(
        value=42,
        default_value=42,
        description="Random seed for data loading and shuffling.",
        display_name="Random Seed",
    )


# =============================================================================
# Training Config
# =============================================================================
@dataclass
class CLIPOptimConfig:
    """Optimizer configuration for CLIP training."""

    optimizer_type: str = STR_FIELD(
        value="adamw",
        default_value="adamw",
        valid_options="adamw,lamb",
        description="Optimizer type: 'adamw' (AdamW) or 'lamb' (LAMB).",
        display_name="Optimizer Type",
    )
    vision_lr: float = FLOAT_FIELD(
        value=1e-4,
        default_value=1e-4,
        valid_min=0,
        valid_max="inf",
        description="Learning rate for the vision encoder.",
        display_name="Vision LR",
    )
    text_lr: float = FLOAT_FIELD(
        value=1e-4,
        default_value=1e-4,
        valid_min=0,
        valid_max="inf",
        description="Learning rate for the text encoder.",
        display_name="Text LR",
    )
    weight_decay: float = FLOAT_FIELD(
        value=1e-4,
        default_value=1e-4,
        valid_min=0,
        valid_max="inf",
        description="Weight decay (L2 regularization) coefficient.",
        display_name="Weight Decay",
    )
    betas: List[float] = LIST_FIELD(
        arrList=[0.9, 0.95],
        default_value=[0.9, 0.95],
        description="Adam/LAMB beta parameters [beta1, beta2] for momentum.",
        display_name="Betas",
    )
    eps: float = FLOAT_FIELD(
        value=1e-6,
        default_value=1e-6,
        valid_min=0,
        description="Epsilon for numerical stability.",
        display_name="Epsilon",
    )
    warmup_steps: int = INT_FIELD(
        value=100,
        default_value=100,
        valid_min=0,
        description="Number of linear warmup steps for learning rate.",
        display_name="Warmup Steps",
    )
    scheduler: str = STR_FIELD(
        value="cosine",
        default_value="cosine",
        valid_options="cosine,constant,linear",
        description="LR schedule after warmup: "
                    "'cosine' (cosine decay to 0), "
                    "'constant' (hold at base LR), "
                    "'linear' (linear decay to 0).",
        display_name="LR Scheduler",
    )


@dataclass
class CLIPTrainConfig(TrainConfig):
    """CLIP training configuration."""

    optim: CLIPOptimConfig = DATACLASS_FIELD(
        CLIPOptimConfig(),
        default_value=CLIPOptimConfig(),
        description="Optimizer configuration with per-tower learning rates.",
    )
    loss_type: str = STR_FIELD(
        value="siglip",
        default_value="siglip",
        valid_options="siglip,clip",
        description="Contrastive loss function: 'siglip' (sigmoid) or 'clip' (softmax).",
        display_name="Loss Type",
    )
    precision: str = STR_FIELD(
        value="fp16",
        default_value="fp16",
        valid_options="fp16,fp32,bf16",
        description="Training precision: fp16 (mixed), fp32 (full), or bf16 (bfloat16).",
        display_name="Precision",
    )
    grad_clip_norm: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        description="Maximum gradient norm for clipping. Set to None to disable.",
        display_name="Gradient Clip Norm",
    )
    grad_checkpointing: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable gradient checkpointing to reduce memory at cost of speed.",
        display_name="Gradient Checkpointing",
    )
    distributed_strategy: str = STR_FIELD(
        value="ddp",
        default_value="ddp",
        valid_options="ddp,fsdp",
        description="Distributed training strategy: 'ddp' or 'fsdp' (fully sharded).",
        display_name="Distributed Strategy",
    )
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to pretrained model checkpoint for fine-tuning.",
        display_name="Pretrained Model Path",
    )
    val_check_interval: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="Run validation every N training steps. If None, validates at end of epoch.",
        display_name="Validation Check Interval",
    )


# =============================================================================
# Inference/Eval Config
# =============================================================================
@dataclass
class CLIPInferenceEvalConfig:
    """Configuration for CLIP inference and evaluation."""

    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to trained model checkpoint (.ckpt or .pth). "
                    "Not required for TRT-based evaluation.",
        display_name="Checkpoint Path",
    )
    batch_size: int = INT_FIELD(
        value=64,
        default_value=64,
        valid_min=1,
        description="Batch size for inference/evaluation.",
        display_name="Batch Size",
    )
    num_gpus: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="Number of GPUs to use.",
        display_name="Number of GPUs",
    )
    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        default_value=[0],
        description="List of GPU device IDs to use.",
        display_name="GPU IDs",
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Directory to save inference/evaluation results.",
        display_name="Results Directory",
    )
    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to TensorRT engine for TRT-based evaluation/inference.",
        display_name="TRT Engine Path",
    )
    # Inference-specific fields
    image_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Directory containing images for inference (inference only).",
        display_name="Image Directory",
    )
    text_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to text file with prompts for inference (inference only).",
        display_name="Text File",
    )
    tokenizer_name: str = STR_FIELD(
        value="openai/clip-vit-large-patch14",
        default_value="openai/clip-vit-large-patch14",
        description="HuggingFace tokenizer name for zero-shot classification. "
                    "Must match the tokenizer used during ONNX export. "
                    "Common values: 'openai/clip-vit-large-patch14' (CLIP/C-RADIO with DFN adapter), "
                    "'google/siglip2-so400m-patch14-384' (SigLIP2/C-RADIO with SigLIP adapter).",
        display_name="Tokenizer Name",
    )


# =============================================================================
# Export Config
# =============================================================================
@dataclass
class CLIPExportConfig:
    """ONNX export configuration for CLIP models."""

    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to trained model checkpoint (.ckpt or .pth). "
                    "If null, exports directly from HuggingFace pretrained weights.",
        display_name="Checkpoint Path",
    )
    onnx_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Output ONNX file path (without extension for 'separate' encoder_type).",
        display_name="ONNX File Path",
    )
    encoder_type: str = STR_FIELD(
        value="combined",
        default_value="combined",
        valid_options="combined,separate",
        description="Export mode: 'combined' (single ONNX with both encoders), "
                    "'separate' (two ONNX files: vision and text).",
        display_name="Encoder Type",
    )
    opset_version: int = INT_FIELD(
        value=17,
        default_value=17,
        valid_min=11,
        description="ONNX opset version for export.",
        display_name="ONNX Opset Version",
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        description="Export batch size. Use -1 for dynamic batch size.",
        display_name="Batch Size",
    )
    input_height: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=32,
        description="Input image height for vision encoder export.",
        display_name="Input Height",
    )
    input_width: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=32,
        description="Input image width for vision encoder export.",
        display_name="Input Width",
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        description="GPU device ID to use for export.",
        display_name="GPU ID",
    )
    on_cpu: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="If True, export on CPU instead of GPU.",
        display_name="On CPU",
    )
    input_channel: int = INT_FIELD(
        value=3,
        default_value=3,
        description="Number of channels in the input image.",
        display_name="Input Channel",
        valid_min=1,
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable verbose ONNX export logging.",
        display_name="Verbose",
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Directory to save exported ONNX models.",
        display_name="Results Directory",
    )


# =============================================================================
# TRT Engine Config
# =============================================================================
@dataclass
class CLIPTrtConfig(TrtConfig):
    """CLIP TensorRT configuration."""

    data_type: str = STR_FIELD(
        value="fp32",
        default_value="fp32",
        valid_options="fp32,fp16",
        description="TensorRT precision: FP32 or FP16.",
        display_name="Data Type",
    )


@dataclass
class CLIPGenTrtEngineConfig(GenTrtEngineConfig):
    """CLIP TRT engine generation config."""

    tensorrt: CLIPTrtConfig = DATACLASS_FIELD(CLIPTrtConfig(), default_value=CLIPTrtConfig())


# =============================================================================
# Experiment Config
# =============================================================================
@dataclass
class CLIPExperimentConfig(CommonExperimentConfig):
    """CLIP experiment config."""

    model_name: Optional[str] = STR_FIELD(
        value="clip",
        default_value="clip",
        description="Name of model for task invocation.",
        display_name="Model Name",
    )
    model: CLIPModelConfig = DATACLASS_FIELD(
        CLIPModelConfig(),
        default_value=CLIPModelConfig(),
        description="Model config.",
    )
    dataset: CLIPDatasetConfig = DATACLASS_FIELD(
        CLIPDatasetConfig(),
        default_value=CLIPDatasetConfig(),
        description="Dataset config.",
    )
    train: CLIPTrainConfig = DATACLASS_FIELD(
        CLIPTrainConfig(),
        default_value=CLIPTrainConfig(),
        description="Training config.",
    )
    evaluate: CLIPInferenceEvalConfig = DATACLASS_FIELD(
        CLIPInferenceEvalConfig(),
        default_value=CLIPInferenceEvalConfig(),
        description="Evaluation config.",
    )
    inference: CLIPInferenceEvalConfig = DATACLASS_FIELD(
        CLIPInferenceEvalConfig(),
        default_value=CLIPInferenceEvalConfig(),
        description="Inference config.",
    )
    export: CLIPExportConfig = DATACLASS_FIELD(
        CLIPExportConfig(),
        default_value=CLIPExportConfig(),
        description="Export config.",
    )
    gen_trt_engine: CLIPGenTrtEngineConfig = DATACLASS_FIELD(
        CLIPGenTrtEngineConfig(),
        default_value=CLIPGenTrtEngineConfig(),
        description="TensorRT engine generation config.",
    )
