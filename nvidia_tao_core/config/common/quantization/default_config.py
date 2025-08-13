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

"""Configuration classes for quantization framework.

This module defines dataclasses for specifying quantization configuration at various levels of
granularity, including model, layer, weight, and activation. These configuration classes are used to
control quantization behavior in the TAO Toolkit.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from nvidia_tao_core.config.utils.types import (
    DICT_FIELD,
    DATACLASS_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)


@dataclass
class BaseQuantizationConfig:
    """Base configuration for quantization.

    Parameters
    ----------
    dtype : str
        Data type to use for quantization (see ``SupportedDtype`` for valid values), e.g.,
        "int8", "fp8_e4m3fn", "fp8_e5m2".
    observer_or_fake_quant : str
        Name of the observer (PTQ) or fake quant (QAT) to use for collecting statistics (must be
        registered).
    quant_axis : int or None, optional
        Axis along which to apply per-channel quantization. If ``None``, per-tensor quantization is
        used.
    observer_or_fake_quant_kwargs : dict or None, optional
        Additional keyword arguments to pass to the observer or fake quant constructor. If not
        provided, defaults to an empty dictionary.

    Notes
    -----
    This is the base class for all quantization configurations. It provides the common parameters
    needed for both weight and activation quantization.

    See Also
    --------
    SupportedDtype
        Enumeration of valid quantization data types.
    """

    dtype: str = STR_FIELD(  # type: ignore
        "",
        description="Data type to use for quantization (e.g., 'int8', 'fp8_e4m3fn', 'fp8_e5m2')",
        display_name="Quantization data type",
        required="yes",
    )
    observer_or_fake_quant: str = STR_FIELD(  # type: ignore
        "",
        description="Name of the observer (PTQ) or fake quant (QAT) to use for collecting statistics",
        display_name="Observer or fake quant",
        required="yes",
    )
    quant_axis: Optional[int] = INT_FIELD(  # type: ignore
        None,
        description="Axis along which to apply per-channel quantization. If None, per-tensor quantization is used",
        display_name="Quantization axis",
    )
    observer_or_fake_quant_kwargs: Optional[Dict[str, Any]] = DICT_FIELD(
        {},
        description="Additional keyword arguments to pass to the observer or fake quant constructor",
        display_name="Observer kwargs",
    )


@dataclass
class WeightQuantizationConfig(BaseQuantizationConfig):
    """Configuration for weight quantization.

    Inherits all parameters from ``BaseQuantizationConfig``. This configuration is specifically for
    quantizing model weights.

    See Also
    --------
    BaseQuantizationConfig
        Base class for quantization configurations.
    ActivationQuantizationConfig
        Configuration for activation quantization.
    """
    # No additional parameters for now; all inherited from BaseQuantizationConfig.


@dataclass
class ActivationQuantizationConfig(BaseQuantizationConfig):
    """Configuration for activation quantization.

    Inherits all parameters from ``BaseQuantizationConfig``. This configuration is specifically for
    quantizing model activations.

    See Also
    --------
    BaseQuantizationConfig
        Base class for quantization configurations.
    WeightQuantizationConfig
        Configuration for weight quantization.
    """
    # No additional parameters for now; all inherited from BaseQuantizationConfig.


@dataclass
class LayerQuantizationConfig:
    """Configuration for quantizing a single module or set of modules.

    Parameters
    ----------
    module_name : str
        Name or pattern specifying the module(s) to quantize. Can be:

        - Exact module name (e.g., "layers.0.conv1")
        - Layer type (e.g., "Linear", "Conv2d")
        - Wildcard pattern (e.g., "conv*", "*.linear")

    weights : WeightQuantizationConfig or None, optional
        Weight quantization configuration. If ``None``, weights are not quantized and left in the
        original precision.
    activations : ActivationQuantizationConfig or None, optional
        Activation quantization configuration. If ``None``, activations are not quantized and left in
        the original precision.

    Notes
    -----
    This configuration allows fine-grained control over quantization at the layer level. You can
    specify different quantization strategies for weights and activations of the same layer.
    """

    module_name: str = STR_FIELD(  # type: ignore
        "",
        description="Name or pattern specifying the module(s) to quantize",
        display_name="Module name",
        required="yes",
    )
    weights: Optional[WeightQuantizationConfig] = DATACLASS_FIELD(
        None,
        description="Weight quantization configuration. If None, weights are not quantized",
        display_name="Weight quantization",
    )
    activations: Optional[ActivationQuantizationConfig] = DATACLASS_FIELD(
        None,
        description="Activation quantization configuration. If None, activations are not quantized",
        display_name="Activation quantization",
    )


@dataclass
class ModelQuantizationConfig:
    """Top-level configuration for model quantization.

    Parameters
    ----------
    backend : str or None, optional
        The quantization backend to use. Defaults to "modelopt".
    mode : str or None, optional
        The quantization mode to use. Defaults to "ptq".
    layers : list of LayerQuantizationConfig or None, optional
        List of per-module quantization configurations. Each entry specifies how a particular module
        or set of modules should be quantized. If ``None`` or empty list, no specific layer
        quantization is applied.
    skip_names : list of str or None, optional
        List of module names or patterns to exclude from quantization. Each entry can be:

        - An exact module name (e.g., "layers.0.conv1")
        - A layer type (e.g., "Linear", "Conv2d")
        - A wildcard pattern (e.g., "conv*", "*.linear")

        Any module whose name matches any pattern in this list will be excluded from quantization. If
        ``None`` or empty list, no modules are excluded.
    model_path : str, optional
        Path to the model to be quantized.
    results_dir : str, optional
        Path to where all the assets generated from a quantization task are stored.
    runtime : str, optional
        The runtime to use. Defaults to "ONNX". Export saves the model in the specified runtime
        format.

    Notes
    -----
    This is the main configuration class that orchestrates the entire quantization process. It
    combines global settings with layer-specific configurations to provide a comprehensive
    quantization strategy.

    Pattern syntax
    --------------
    The ``module_name`` and ``skip_names`` fields accept wildcard patterns interpreted using
    Python's ``fnmatch.fnmatch``. Wildcards ``*`` and ``?`` are supported and matching is
    case-sensitive. A pattern is first applied to the module's qualified graph name (e.g.,
    ``backbone.layer1.0.conv1``). If that does not match, the module's class name (e.g., ``Conv2d``)
    is checked. See ``nvidia_tao_pytorch.core.quantization.utils.match_layer`` for implementation
    details.

    Examples
    --------
    Basic static PTQ configuration::

        config = ModelQuantizationConfig(
            backend="modelopt",
            layers=[
                LayerQuantizationConfig(
                    module_name="conv*",
                    weights=WeightQuantizationConfig(
                        mode="static_ptq",
                        dtype="int8",
                        observer_or_fake_quant="my_observer"
                    ),
                    activations=ActivationQuantizationConfig(
                        mode="static_ptq",
                        dtype="int8",
                        observer_or_fake_quant="my_observer"
                    )
                )
            ],
            skip_names=["final_layer"]
        )

    QAT configuration with mixed precision::

        config = ModelQuantizationConfig(
            backend="modelopt",
            layers=[
                LayerQuantizationConfig(
                    module_name="Linear",
                    weights=WeightQuantizationConfig(
                        mode="qat",
                        dtype="fp8_e4m3fn",
                        observer_or_fake_quant="my_fake_quant"
                    ),
                    activations=ActivationQuantizationConfig(
                        mode="qat",
                        dtype="fp8_e4m3fn",
                        observer_or_fake_quant="my_fake_quant"
                    )
                )
            ]
        )

    See Also
    --------
    LayerQuantizationConfig
        Configuration for individual layer quantization.
    WeightQuantizationConfig
        Configuration for weight quantization.
    ActivationQuantizationConfig
        Configuration for activation quantization.
    """

    backend: Optional[str] = STR_FIELD(  # type: ignore
        "modelopt",
        description="The quantization backend to use",
        display_name="Quantization backend",
    )
    mode: Optional[str] = STR_FIELD(
        "ptq",
        description="The quantization mode to use",
        display_name="Quantization mode",
    )
    layers: Optional[List[LayerQuantizationConfig]] = LIST_FIELD(
        [],
        description="List of per-module quantization configurations",
        display_name="Layer quantization configs",
    )
    skip_names: Optional[List[str]] = LIST_FIELD(
        [],
        description="List of module/layer names or patterns to exclude from quantization",
        display_name="Skip names",
    )
    model_path: Optional[str] = STR_FIELD(
        value="",
        display_name="Model path",
        description="Path to the model to be quantized.",
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="Results directory",
        description="Path to where all the assets generated from a task are stored.",
    )
    runtime: Optional[str] = STR_FIELD(
        value="ONNX",
        display_name="Runtime",
        description="The runtime to use. Export saves the model in the specified runtime format.",
    )
