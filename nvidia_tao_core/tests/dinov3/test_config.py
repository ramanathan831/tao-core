# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for the DINOv3 dataclass config and its jsonschema conversion.

The DINOv3 config in tao-core is the source of truth for the generated skill
schemas in tao-skills-external and must stay aligned with
``nvidia_tao_pytorch.config.dinov3.default_config`` (bug 6465432).
"""

from nvidia_tao_core.api_utils.dataclass2json_converter import (
    create_json_schema,
    dataclass_to_json,
)
from nvidia_tao_core.config.dinov3.default_config import (
    ExperimentConfig,
    SUPPORTED_BACKBONES,
    map_params,
)

# Must match nvidia_tao_pytorch.config.dinov3.default_config.SUPPORTED_BACKBONES.
EXPECTED_BACKBONES = ["vit_s", "vit_s_plus", "vit_b", "vit_l", "vit_h_plus", "vit_7b"]


def test_supported_backbones_match_tao_pytorch():
    assert SUPPORTED_BACKBONES == EXPECTED_BACKBONES


def test_map_params_cover_all_backbones():
    for param, per_arch in map_params.items():
        assert sorted(per_arch.keys()) == sorted(EXPECTED_BACKBONES), (
            f"map_params[{param!r}] does not cover all supported backbones"
        )


def test_backbone_enum_in_json_schema():
    schema = create_json_schema(dataclass_to_json(ExperimentConfig()))
    backbone = schema["properties"]["model"]["properties"]["backbone"]["properties"]
    assert backbone["teacher_type"]["enum"] == EXPECTED_BACKBONES
    assert backbone["student_type"]["enum"] == EXPECTED_BACKBONES
    assert backbone["teacher_type"]["default"] == "vit_b"
    assert backbone["student_type"]["default"] == "vit_b"


def test_experiment_config_actions():
    schema = create_json_schema(dataclass_to_json(ExperimentConfig()))
    for action in ("train", "inference", "export", "gen_trt_engine", "convert"):
        assert action in schema["properties"], f"missing {action} section"
