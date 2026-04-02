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

"""Default config file for DEFT."""

from dataclasses import dataclass
from typing import Optional
from omegaconf import MISSING
from nvidia_tao_core.config.utils.types import STR_FIELD, DATACLASS_FIELD


@dataclass
class DataConfig:
    """Dataset configuration for KPI gap analysis."""

    predictions_json: str = STR_FIELD(
        value=MISSING,
        default_value="<path to predictions JSON>",
        description="Path to predictions JSON file with response and gt fields."
    )
    videos_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Directory containing videos. If empty, video_id in predictions are treated as absolute paths."
    )


@dataclass
class ExperimentConfig:
    """Experiment configuration for KPI gap analysis."""

    data: DataConfig = DATACLASS_FIELD(DataConfig())
    results_dir: Optional[str] = STR_FIELD(
        value=MISSING,
        default_value="<path to output directory>",
        description="Output directory for kpi_gaps.jsonl and kpi_gaps_report.txt."
    )
