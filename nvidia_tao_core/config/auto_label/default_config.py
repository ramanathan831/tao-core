# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Default config file."""

from typing import List, Optional, Dict
from dataclasses import dataclass
from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)
from nvidia_tao_core.config.grounding_dino.dataset import GDINOAugmentationConfig
from nvidia_tao_core.config.grounding_dino.model import GDINOModelConfig
from nvidia_tao_core.config.grounding_dino.train import GDINOTrainExpConfig
from nvidia_tao_core.config.mal.default_config import (
    MALTrainExpConfig,
    MALEvalExpConfig,
    MALInferenceExpConfig,
    MALDatasetConfig,
    MALModelConfig,
)


@dataclass
class MALConfig:
    """MAL config."""

    dataset: MALDatasetConfig = DATACLASS_FIELD(
        MALDatasetConfig(),
        description="Configuration parameters for MAL dataset"
    )
    train: MALTrainExpConfig = DATACLASS_FIELD(
        MALTrainExpConfig(),
        description="Configuration parameters for MAL train"
    )
    model: MALModelConfig = DATACLASS_FIELD(
        MALModelConfig(),
        description="Configuration parameters for MAL model"
    )
    inference: MALInferenceExpConfig = DATACLASS_FIELD(
        MALInferenceExpConfig(),
        description="Configuration parameters for MAL inference"
    )
    evaluate: MALEvalExpConfig = DATACLASS_FIELD(
        MALEvalExpConfig(),
        description="Configuration parameters for MAL evaluation"
    )
    checkpoint: Optional[str] = STR_FIELD(
        None,
        default_value="",
        description="MAL model checkpoint path",
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Result directory",
    )


@dataclass
class GDINOConfig:
    """Grounding DINO config."""

    @dataclass
    class GDINODataConfig:
        """DINO dataset config used for auto-labeling."""

        image_dir: Optional[str] = STR_FIELD(
            None,
            default_value="",
            description="Image root directory",
        )
        noun_chunk_path: Optional[str] = STR_FIELD(
            value=None,
            default_value=""
        )
        class_names: Optional[List[str]] = LIST_FIELD(
            arrList=[],
            description="List of classes to run auto-labeling"
        )
        augmentation: GDINOAugmentationConfig = DATACLASS_FIELD(
            GDINOAugmentationConfig(),
            description="Configuration parameters for Grounding DINO augmenation"
        )

    train: GDINOTrainExpConfig = DATACLASS_FIELD(
        GDINOTrainExpConfig(),
        description="Configuration parameters for Grounding DINO train"
    )
    model: GDINOModelConfig = DATACLASS_FIELD(
        GDINOModelConfig(),
        description="Configuration parameters for Grounding DINO model"
    )
    dataset: GDINODataConfig = DATACLASS_FIELD(
        GDINODataConfig(),
        description="Configuration parameters for Grounding DINO dataset"
    )

    checkpoint: Optional[str] = STR_FIELD(
        None,
        default_value="",
        description="Grounding model checkpoint path",
    )

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Result directory",
    )

    iteration_scheduler: List[Dict[str, float]] = LIST_FIELD(
        arrList=[{"conf_threshold": 0.5, "nms_threshold": 0.0}],
        default_values=[{"conf_threshold": 0.5, "nms_threshold": 0.0}],
        description="""The list of iteration schedule. Default is one iteration with confidence threshold of 0.5.
                    Next iteration eliminates classes/noun chunks that have been already detected."""
    )
    visualize: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to enable visualization of bounding boxes."
    )


@dataclass
class VideoCotGeminiConfig:
    """Gemini API configuration for video CoT pipeline."""

    api_key: str = STR_FIELD(
        value="",
        default_value="",
        description="Google Gemini API key (or set GOOGLE_API_KEY env var)",
    )
    model: str = STR_FIELD(
        value="gemini-3.1-flash-lite-preview",
        default_value="gemini-3.1-flash-lite-preview",
        description="Gemini model name",
    )
    media_resolution: str = STR_FIELD(
        value="MEDIA_RESOLUTION_LOW",
        default_value="MEDIA_RESOLUTION_LOW",
        description="Media resolution for video input",
    )
    temperature: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        description="Sampling temperature",
    )
    max_output_tokens: int = INT_FIELD(
        value=8192,
        default_value=8192,
        description="Maximum output tokens",
    )
    timeout: int = INT_FIELD(
        value=120,
        default_value=120,
        description="Request timeout in seconds",
    )


@dataclass
class VideoCotOpenAIConfig:
    """OpenAI-compatible endpoint configuration for video CoT pipeline."""

    api_key: str = STR_FIELD(
        value="",
        default_value="",
        description="API key for OpenAI-compatible endpoint",
    )
    base_url: str = STR_FIELD(
        value="",
        default_value="",
        description="Base URL for OpenAI-compatible endpoint",
    )
    model_name: str = STR_FIELD(
        value="",
        default_value="",
        description="Model name for OpenAI-compatible endpoint",
    )
    temperature: float = FLOAT_FIELD(
        value=0.7,
        default_value=0.7,
        description="Sampling temperature",
    )
    max_tokens: int = INT_FIELD(
        value=4096,
        default_value=4096,
        description="Maximum output tokens",
    )
    timeout: int = INT_FIELD(
        value=60,
        default_value=60,
        description="Request timeout in seconds",
    )


@dataclass
class VideoCotLLMConfig:
    """LLM backend selection and configuration for video CoT pipeline."""

    backend: str = STR_FIELD(
        value="gemini",
        default_value="gemini",
        description="LLM backend to use",
        valid_options="gemini,openai",
    )
    gemini: VideoCotGeminiConfig = DATACLASS_FIELD(
        VideoCotGeminiConfig(),
        description="Gemini API configuration",
    )
    openai: VideoCotOpenAIConfig = DATACLASS_FIELD(
        VideoCotOpenAIConfig(),
        description="OpenAI-compatible endpoint configuration",
    )


@dataclass
class VideoCotWorkflowConfig:
    """Pipeline execution parameters for video CoT."""

    steps: List[str] = LIST_FIELD(
        arrList=["0", "1a", "1b", "1c", "2", "3", "4"],
        default_values=["0", "1a", "1b", "1c", "2", "3", "4"],
        description="Pipeline steps to execute",
    )
    mode: str = STR_FIELD(
        value="auto",
        default_value="auto",
        description="Pipeline mode: auto (VLM classifies), anomaly, or normal",
        valid_options="auto,anomaly,normal",
    )
    max_workers: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        description="Maximum concurrent workers for video processing",
    )
    max_video_length_sec: int = INT_FIELD(
        value=300,
        default_value=300,
        description="Maximum video length in seconds",
    )
    chunk_duration_options: List[int] = LIST_FIELD(
        arrList=[5, 10, 15, 20, 30],
        default_values=[5, 10, 15, 20, 30],
        description="Chunk duration options in seconds",
    )
    max_chunks: int = INT_FIELD(
        value=10,
        default_value=10,
        description="Maximum number of chunks per video",
    )
    highlight_before_sec: float = FLOAT_FIELD(
        value=3.0,
        default_value=3.0,
        description="Seconds to include before anomaly timestamp in highlight clip",
    )
    highlight_after_sec: float = FLOAT_FIELD(
        value=3.0,
        default_value=3.0,
        description="Seconds to include after anomaly timestamp in highlight clip",
    )
    long_video_threshold_sec: int = INT_FIELD(
        value=60,
        default_value=60,
        description="Duration threshold (seconds) above which videos are sampled as frames",
    )
    long_video_sample_fps: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="Frame sampling rate for long videos",
    )
    long_video_max_frames: int = INT_FIELD(
        value=60,
        default_value=60,
        description="Maximum frames to sample from long videos",
    )
    qa_types: List[str] = LIST_FIELD(
        arrList=["mcq", "bcq", "open_qa"],
        default_values=["mcq", "bcq", "open_qa"],
        description="QA types to generate",
    )


@dataclass
class VideoCotDataConfig:
    """Input data specification for video CoT pipeline.

    At least one of ``video_root`` or ``input_jsonl_files`` must be provided.
    Both may be used together — the resulting video lists are merged.

    When using ``input_jsonl_files``, each JSONL file should contain one JSON
    object per line with at least a ``"video_path"`` (or ``"video"``) field::

        {"video_path": "/absolute/path/to/video.mp4"}

    Additional fields are allowed. If ``filter_field`` is set, only entries
    where that boolean field is truthy are included.
    """

    video_root: str = STR_FIELD(
        value="",
        default_value="",
        description="Root directory containing input videos (walked recursively). "
                    "At least one of video_root or input_jsonl_files must be provided; both may be used together.",
    )
    input_jsonl_files: List[str] = LIST_FIELD(
        arrList=[],
        default_values=[],
        description="Optional list of JSONL files listing video paths. "
                    "Each line must have a 'video_path' (or 'video') field. "
                    "Can be used instead of or in addition to video_root.",
    )
    filter_field: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Optional boolean field name to filter entries in input JSONL files",
    )


@dataclass
class VideoCotConfig:
    """Video Chain-of-Thought annotation pipeline configuration."""

    vlm: VideoCotLLMConfig = DATACLASS_FIELD(
        VideoCotLLMConfig(),
        description="VLM (vision-language model) configuration for video steps",
    )
    llm: VideoCotLLMConfig = DATACLASS_FIELD(
        VideoCotLLMConfig(),
        description="LLM (text-only) configuration for text steps",
    )
    workflow: VideoCotWorkflowConfig = DATACLASS_FIELD(
        VideoCotWorkflowConfig(),
        description="Pipeline workflow parameters",
    )
    data: VideoCotDataConfig = DATACLASS_FIELD(
        VideoCotDataConfig(),
        description="Input data configuration",
    )
    output_format: str = STR_FIELD(
        value="both",
        default_value="both",
        description="Output format: qa, daft, or both",
        valid_options="qa,daft,both",
    )
    prompts_module: str = STR_FIELD(
        value="",
        default_value="",
        description="Optional Python module path for custom prompt templates",
    )


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        default_value=[0],
        description="Indices of GPUs to use"
    )
    num_gpus: int = INT_FIELD(value=1,
                              default_value=1,
                              description="Number of GPUs to use")
    batch_size: int = INT_FIELD(value=4,
                                default_value=4,
                                valid_min=1,
                                description="Batch size")
    num_workers: int = INT_FIELD(value=8,
                                 default_value=8,
                                 valid_min=1,
                                 description="Number of workers for dataloader")

    autolabel_type: str = STR_FIELD(
        value="mal",
        default_value="mal",
        description="Type of auto-labeling to run",
        valid_options="mal,grounding_dino,video_cot"
    )

    mal: MALConfig = DATACLASS_FIELD(
        MALConfig(),
        description="Configuration parameters for MAL"
    )
    grounding_dino: GDINOConfig = DATACLASS_FIELD(
        GDINOConfig(),
        description="Configuration parameters for Grounding DINO"
    )
    video_cot: VideoCotConfig = DATACLASS_FIELD(
        VideoCotConfig(),
        description="Configuration parameters for Video CoT pipeline"
    )

    results_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Result directory",
    )

    def __post_init__(self):
        """assertion check."""
        valid_types = ["mal", "grounding_dino", "video_cot"]
        assert self.autolabel_type in valid_types, \
            f"Invalid option encountered. {self.autolabel_type}"
