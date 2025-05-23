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

"""Api tests to generate schema for networks"""

import pytest

from nvidia_tao_core.microservices.constants import TAO_NETWORKS
from nvidia_tao_core.microservices.enum_constants import _get_network_architectures
from nvidia_tao_core.scripts.generate_schema import generate_schema

EXCLUDED_KEYWORDS = ['maxine', 'monai', 'vlm']
config_networks = [
    network for network in _get_network_architectures()
    if not any(keyword in network for keyword in EXCLUDED_KEYWORDS)
]
constant_networks = [
    network for network in TAO_NETWORKS
    if not any(keyword in network for keyword in EXCLUDED_KEYWORDS)
]


@pytest.mark.parametrize("network", config_networks)
def test_networks_from_enum(network):
    """Test schema from api network_arch enum"""
    schema = generate_schema(network)
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "default" in schema


@pytest.mark.parametrize("network", constant_networks)
def test_networks_from_constants(network):
    """Test schema from TAO_NETWORKS constant"""
    schema = generate_schema(network)
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "default" in schema
