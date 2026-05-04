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

"""Configuration hyperparameter schema to deploy the model."""

from dataclasses import dataclass
from typing import Optional

from nvidia_tao_core.config.utils.types import (
    DATACLASS_FIELD,
    INT_FIELD,
    STR_FIELD,
)
from nvidia_tao_core.config.common.common_config import (
    GenTrtEngineConfig,
    TrtConfig
)


@dataclass
class DepthNetTrtConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(
        value="FP32",
        default_value="FP32",
        description="The precision to be set for building the TensorRT engine.",
        display_name="data type",
        valid_options=",".join(["FP32", "FP16", "BF16"])
    )


@dataclass
class DepthNetGenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: DepthNetTrtConfig = DATACLASS_FIELD(
        DepthNetTrtConfig(),
        description="Hyper parameters to configure the TensorRT Engine builder.",
        display_name="TensorRT hyper params."
    )
    min_height: Optional[int] = INT_FIELD(
        value=None, default_value=None, valid_min=14,
        description="Minimum input height for dynamic-shape engine.",
        display_name="Min input height",
    )
    opt_height: Optional[int] = INT_FIELD(
        value=None, default_value=None, valid_min=14,
        description="Optimum input height for dynamic-shape engine.",
        display_name="Opt input height",
    )
    max_height: Optional[int] = INT_FIELD(
        value=None, default_value=None, valid_min=14,
        description="Maximum input height for dynamic-shape engine.",
        display_name="Max input height",
    )
    min_width: Optional[int] = INT_FIELD(
        value=None, default_value=None, valid_min=14,
        description="Minimum input width for dynamic-shape engine.",
        display_name="Min input width",
    )
    opt_width: Optional[int] = INT_FIELD(
        value=None, default_value=None, valid_min=14,
        description="Optimum input width for dynamic-shape engine.",
        display_name="Opt input width",
    )
    max_width: Optional[int] = INT_FIELD(
        value=None, default_value=None, valid_min=14,
        description="Maximum input width for dynamic-shape engine.",
        display_name="Max input width",
    )

    def __post_init__(self):
        """Validate min ≤ opt ≤ max for height and width when all are set."""
        for axis, lo, opt, hi in (
            ("height", self.min_height, self.opt_height, self.max_height),
            ("width", self.min_width, self.opt_width, self.max_width),
        ):
            if lo is None or opt is None or hi is None:
                continue
            if not (lo <= opt <= hi):
                raise ValueError(
                    f"Dynamic-shape {axis} bounds must satisfy "
                    f"min ≤ opt ≤ max; got min={lo}, opt={opt}, max={hi}."
                )
