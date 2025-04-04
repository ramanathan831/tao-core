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

"""Generating JSON schemas"""

from nvidia_tao_core.api_utils import dataclass2json_converter


def generate_schema(neural_network_name, action=""):
    """Generates JSON schema for network"""
    imported_module = dataclass2json_converter.import_module_from_path(
        f"nvidia_tao_core.config.{neural_network_name}.default_config"
    )
    if neural_network_name == "bevfusion" and action == "dataset_convert":
        expConfig = imported_module.BEVFusionDataConvertExpConfig()
    if neural_network_name == "stylegan_xl" and action == "dataset_convert":
        imported_module = dataclass2json_converter.import_module_from_path(
            f"nvidia_tao_core.config.{neural_network_name}.dataset"
        )
        expConfig = imported_module.DataConvertExpConfig()
    else:
        expConfig = imported_module.ExperimentConfig()
    json_with_meta_config = dataclass2json_converter.dataclass_to_json(expConfig)
    return dataclass2json_converter.create_json_schema(json_with_meta_config)
