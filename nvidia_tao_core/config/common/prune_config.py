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

"""Default config file."""

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    INT_FIELD,
    STR_FIELD,
    FLOAT_FIELD
)


@dataclass
class PruneConfig:
    """Prune config."""

    mode: str = STR_FIELD(value="amount", valid_options=["amount", "threshold", "experimental_hybrid"])
    amount: Optional[float] = FLOAT_FIELD(value=None)
    threshold: Optional[float] = FLOAT_FIELD(value=None)
    granularity: int = INT_FIELD(value=8)
    raw_prune_score: str = STR_FIELD(value="L1", valid_options=["L1", "L2"])
