# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
    DATACLASS_FIELD,
    BOOL_FIELD,
)


@dataclass
class InferenceConfig:
    """Inference config."""

    inference_type: str = STR_FIELD(
        default_value="text2world",
        value="text2world",
        display_name="Inference type",
        description="Type of inference to run",
        valid_options="text2world,image2world,video2world",
    )
    input_path: Optional[str] = STR_FIELD(
        default_value="",
        value="",
        display_name="Input path",
        description="Path to input image/video file. Required for image2world and video2world inference types."
    )
    prompt: str = STR_FIELD(
        default_value="",
        value="",
        display_name="Prompt",
        description="Text prompt for video generation"
    )
    negative_prompt: str = STR_FIELD(
        default_value="The video captures a series of frames showing ugly scenes, static with no motion, "
                      "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
                      "pixelated images, poorly lit areas, underexposed and overexposed scenes, "
                      "poor color balance, washed out colors, choppy sequences, jerky movements, "
                      "low frame rate, artifacting, color banding, unnatural transitions, "
                      "outdated special effects, fake elements, unconvincing visuals, "
                      "poorly edited content, jump cuts, visual noise, and flickering. "
                      "Overall, the video is of poor quality.",
        value="",
        display_name="Negative prompt",
        description="Negative prompt describing what to avoid in the generated video"
    )
    model: str = STR_FIELD(
        default_value="2B/post-trained",
        value="2B/post-trained",
        display_name="Model",
        description="Model variant to use for inference (e.g. 2B/post-trained, 2B/pre-trained, "
                    "2B/distilled, 14B/post-trained, 14B/pre-trained)"
    )
    checkpoint_path: Optional[str] = STR_FIELD(
        default_value="",
        value="",
        display_name="Checkpoint path",
        description="Path to model checkpoint (DCP directory or .pt file). "
                    "If empty, uses the default checkpoint for the selected model."
    )
    num_gpus: Optional[int] = INT_FIELD(
        default_value=1,
        value=1,
        valid_min=1,
        valid_max=8,
        display_name="Number of GPUs",
        description="Number of GPUs to use for inference (context parallelism)"
    )
    num_output_frames: int = INT_FIELD(
        default_value=77,
        value=77,
        valid_min=1,
        valid_max=500,
        display_name="Number of output frames",
        description="Number of video frames to generate"
    )
    num_steps: int = INT_FIELD(
        default_value=35,
        value=35,
        valid_min=1,
        valid_max=100,
        display_name="Number of diffusion steps",
        description="Number of diffusion denoising steps"
    )
    guidance: int = INT_FIELD(
        default_value=7,
        value=7,
        valid_min=0,
        valid_max=7,
        display_name="Guidance scale",
        description="Guidance scale (0-7). Higher values make the video adhere more closely to the prompt."
    )
    seed: int = INT_FIELD(
        default_value=0,
        value=0,
        valid_min=0,
        valid_max="inf",
        display_name="Random seed",
        description="Random seed for reproducibility"
    )
    enable_lora: bool = BOOL_FIELD(
        default_value=False,
        value=False,
        display_name="Enable LoRA",
        description="Enable LoRA model loading for inference"
    )
    base_model_path: Optional[str] = STR_FIELD(
        default_value="",
        value="",
        display_name="Base model path",
        description="Path to base model for LoRA inference (used when enable_lora is True)"
    )


@dataclass
class ExperimentConfig:
    """Experiment config."""

    results_dir: str = STR_FIELD(
        default_value="",
        value="",
        display_name="Results directory",
        description="Directory to save inference results"
    )
    inference: InferenceConfig = DATACLASS_FIELD(InferenceConfig(), description="Inference config.")
