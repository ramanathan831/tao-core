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

"""Default config file for Sparse4D."""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    FLOAT_FIELD,
    LIST_FIELD,
    DICT_FIELD,
    DATACLASS_FIELD
)
from nvidia_tao_core.config.common.common_config import (
    CommonExperimentConfig,
    ExportConfig,
    TrainConfig,
    EvaluateConfig,
    InferenceConfig,
)


@dataclass
class Sparse4DOptimizerConfig:
    """Optimizer config for Sparse4D."""

    type: str = STR_FIELD(value="adamw", default_value="adamw", description="Optimizer type", valid_options="adamw,adam,sgd", display_name="Optimizer type")
    lr: float = FLOAT_FIELD(value=5e-5, default_value=5e-5, valid_min=0, valid_max="inf", automl_enabled="TRUE", description="Learning rate", display_name="Learning rate")
    weight_decay: float = FLOAT_FIELD(value=0.001, default_value=0.001, math_cond=">= 0.0", description="Weight decay coefficient", display_name="Weight decay coefficient")
    momentum: float = FLOAT_FIELD(value=0.9, default_value=0.9, math_cond=">= 0.0", description="Momentum for SGD", display_name="Momentum for SGD")
    paramwise_cfg: Optional[Dict[str, Any]] = DICT_FIELD(
        hashMap={"custom_keys": {"img_backbone": {"lr_mult": 0.2}}},
        description="Parameters-wise configuration",
        default_value={"custom_keys": {"img_backbone": {"lr_mult": 0.2}}},
        display_name="Parameters-wise configuration"
    )
    grad_clip: Optional[Dict[str, Any]] = DICT_FIELD(
        hashMap={"max_norm": 25, "norm_type": 2},
        description="Gradient clipping configuration",
        default_value={"max_norm": 25, "norm_type": 2},
        display_name="Gradient clipping configuration"
    )
    lr_scheduler: Dict[str, Any] = DICT_FIELD(
        hashMap={
            "policy": "cosine",
            "warmup": "linear",
            "warmup_iters": 500,
            "warmup_ratio": 0.333333,
            "min_lr_ratio": 0.001
        },
        description="Learning rate scheduler configuration",
        default_value={
            "policy": "cosine",
            "warmup": "linear",
            "warmup_iters": 500,
            "warmup_ratio": 0.333333,
            "min_lr_ratio": 0.001
        },
        display_name="Learning rate scheduler configuration"
    )


@dataclass
class Sparse4DTrainConfig(TrainConfig):
    """Training configuration for Sparse4D."""

    validation_interval: float = FLOAT_FIELD(value=0.5, default_value=0.5, valid_min=0, valid_max="inf", description="Validation interval in epochs", display_name="Validation interval in epochs")
    checkpoint_interval: float = FLOAT_FIELD(value=0.5, default_value=0.5, valid_min=0, valid_max="inf", description="Checkpoint interval in epochs", display_name="Checkpoint interval in epochs")
    pretrained_model_path: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to pretrained model", display_name="Path to pretrained model")
    optim: Sparse4DOptimizerConfig = DATACLASS_FIELD(Sparse4DOptimizerConfig(), description="Optimizer configuration", display_name="Optimizer configuration")


@dataclass
class Sparse4DBackboneConfig:
    """Backbone configuration for Sparse4D."""

    type: str = STR_FIELD(value="ResNet", default_value="ResNet", description="Backbone type", valid_options="ResNet", display_name="Backbone type")
    depth: int = INT_FIELD(value=101, default_value=101, valid_min=18, valid_max=152, description="ResNet depth", valid_options="18,34,50,101,152", display_name="ResNet depth")
    num_stages: int = INT_FIELD(value=4, default_value=4, valid_min=1, valid_max=4, description="Number of stages", display_name="Number of stages")
    frozen_stages: int = INT_FIELD(value=-1, default_value=-1, valid_min=-1, valid_max=4, description="Frozen stages (-1 for none)", display_name="Frozen stages (-1 for none)")
    norm_eval: bool = BOOL_FIELD(value=True, default_value=True, description="Set BatchNorm layers to eval mode", display_name="Set BatchNorm layers to eval mode")
    style: str = STR_FIELD(value="pytorch", default_value="pytorch", description="ResNet style", valid_options="pytorch,caffe", display_name="ResNet style")
    with_cp: bool = BOOL_FIELD(value=True, default_value=True, description="Use checkpoint to save memory", display_name="Use checkpoint to save memory")
    out_indices: Tuple[int, ...] = LIST_FIELD(arrList=[0, 1, 2, 3], default_value=[0, 1, 2, 3], description="Output indices", display_name="Output indices")
    norm_cfg: Dict[str, Any] = DICT_FIELD(
        hashMap={"type": "BN", "requires_grad": False},
        description="Normalization configuration",
        default_value={"type": "BN", "requires_grad": False},
        display_name="Normalization configuration"
    )
    pretrained_backbone_path: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to pretrained backbone weights", display_name="Path to pretrained backbone weights")


@dataclass
class Sparse4DNeckConfig:
    """Neck configuration for Sparse4D."""

    type: str = STR_FIELD(value="FPN", default_value="FPN", description="Neck type", valid_options="FPN", display_name="Neck type")
    num_outs: int = INT_FIELD(value=4, default_value=4, valid_min=1, valid_max="inf", description="Number of output levels", display_name="Number of output levels")
    start_level: int = INT_FIELD(value=0, default_value=0, valid_min=0, valid_max="inf", description="Start level for FPN", display_name="Start level for FPN")
    out_channels: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Output channels", display_name="Output channels")
    in_channels: List[int] = LIST_FIELD(arrList=[256, 512, 1024, 2048], default_value=[256, 512, 1024, 2048], description="Input channels", display_name="Input channels")
    add_extra_convs: str = STR_FIELD(value="on_output", default_value="on_output", description="Type of extra conv", valid_options="on_input,on_lateral,on_output,False", display_name="Type of extra conv")
    relu_before_extra_convs: bool = BOOL_FIELD(value=True, default_value=True, description="Apply ReLU before extra convs", display_name="Apply ReLU before extra convs")


@dataclass
class Sparse4DDepthBranchConfig:
    """Depth branch configuration for Sparse4D."""

    type: str = STR_FIELD(value="dense_depth", default_value="dense_depth", description="Depth branch type", display_name="Depth branch type")
    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions", display_name="Embedding dimensions")
    num_depth_layers: int = INT_FIELD(value=3, default_value=3, valid_min=1, valid_max="inf", description="Number of depth layers", display_name="Number of depth layers")
    loss_weight: float = FLOAT_FIELD(value=0.2, default_value=0.2, valid_min=0, valid_max="inf", description="Weight for depth loss", display_name="Weight for depth loss")


@dataclass
class Sparse4DInstanceBankConfig:
    """Instance bank configuration for Sparse4D."""

    num_anchor: int = INT_FIELD(value=900, default_value=900, valid_min=1, valid_max="inf", description="Number of anchors", display_name="Number of anchors")
    anchor: str = STR_FIELD(value="", default_value="", description="Path to anchor file", display_name="Path to anchor file")
    num_temp_instances: int = INT_FIELD(value=600, default_value=600, valid_min=0, valid_max="inf", description="Number of temporal instances", display_name="Number of temporal instances")
    confidence_decay: float = FLOAT_FIELD(value=0.8, default_value=0.8, valid_min=0, valid_max=1, description="Confidence decay factor", display_name="Confidence decay factor")
    feat_grad: bool = BOOL_FIELD(value=False, default_value=False, description="Enable gradients for features", display_name="Enable gradients for features")
    default_time_interval: float = FLOAT_FIELD(value=0.033333, default_value=0.033333, valid_min=0, valid_max="inf", description="Default time interval", display_name="Default time interval")
    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions", display_name="Embedding dimensions")
    use_temporal_align: bool = BOOL_FIELD(value=False, default_value=False, description="Use temporal alignment", display_name="Use temporal alignment")


@dataclass
class Sparse4DAnchorEncoderConfig:
    """Anchor encoder configuration for Sparse4D."""

    type: str = STR_FIELD(value="SparseBox3DEncoder", default_value="SparseBox3DEncoder", description="Anchor encoder type", display_name="Anchor encoder type")
    vel_dims: int = INT_FIELD(value=3, default_value=3, valid_min=1, valid_max="inf", description="Velocity dimensions", display_name="Velocity dimensions")
    embed_dims: List[int] = LIST_FIELD(arrList=[128, 32, 32, 64], default_value=[128, 32, 32, 64], description="Embedding dimensions", display_name="Embedding dimensions")
    mode: str = STR_FIELD(value="cat", default_value="cat", description="Mode", valid_options="cat,add", display_name="Mode")
    output_fc: bool = BOOL_FIELD(value=False, default_value=False, description="Output FC", display_name="Output FC")
    in_loops: int = INT_FIELD(value=1, default_value=1, valid_min=1, valid_max="inf", description="In loops", display_name="In loops")
    out_loops: int = INT_FIELD(value=4, default_value=4, valid_min=1, valid_max="inf", description="Out loops", display_name="Out loops")
    pos_embed_only: bool = BOOL_FIELD(value=False, default_value=False, description="Pos embed only", display_name="Pos embed only")


@dataclass
class Sparse4DKpsGeneratorConfig:
    """KPS generator configuration for Sparse4D."""

    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions", display_name="Embedding dimensions")
    num_learnable_pts: int = INT_FIELD(value=6, default_value=6, valid_min=1, valid_max="inf", description="Number of learnable points", display_name="Number of learnable points")
    fix_scale: List[List[Any]] = LIST_FIELD(
        arrList=[
            [0, 0, 0],
            [0.45, 0, 0],
            [-0.45, 0, 0],
            [0, 0.45, 0],
            [0, -0.45, 0],
            [0, 0, 0.45],
            [0, 0, -0.45]
        ],
        default_value=[
            [0, 0, 0],
            [0.45, 0, 0],
            [-0.45, 0, 0],
            [0, 0.45, 0],
            [0, -0.45, 0],
            [0, 0, 0.45],
            [0, 0, -0.45]
        ],
        description="Fixed scale",
        display_name="Fixed scale"
    )


@dataclass
class Sparse4DLossClsConfig:
    """Classification loss configuration for Sparse4D."""

    type: str = STR_FIELD(value="focal", default_value="focal", description="Classification loss type", display_name="Classification loss type")
    use_sigmoid: bool = BOOL_FIELD(value=True, default_value=True, description="Use sigmoid", display_name="Use sigmoid")
    gamma: float = FLOAT_FIELD(value=2.0, default_value=2.0, valid_min=0, valid_max="inf", description="Focal loss gamma", display_name="Focal loss gamma")
    alpha: float = FLOAT_FIELD(value=0.25, default_value=0.25, valid_min=0, valid_max=1, description="Focal loss alpha", display_name="Focal loss alpha")
    loss_weight: float = FLOAT_FIELD(value=2.0, default_value=2.0, valid_min=0, valid_max="inf", description="Loss weight", display_name="Loss weight")


@dataclass
class Sparse4DTrainDatasetConfig:
    """Training dataset configuration for Sparse4D."""

    ann_file: str = STR_FIELD(value=MISSING, default_value="", description="Path to annotation file", display_name="Path to annotation file")
    test_mode: bool = BOOL_FIELD(value=False, default_value=False, description="Test mode", display_name="Test mode")
    use_valid_flag: bool = BOOL_FIELD(value=True, default_value=True, description="Use valid flag", display_name="Use valid flag")
    with_seq_flag: bool = BOOL_FIELD(value=True, default_value=True, description="With sequence flag", display_name="With sequence flag")
    sequences_split_num: int = INT_FIELD(value=100, default_value=100, valid_min=1, valid_max="inf", description="Number of sequences", display_name="Number of sequences")
    keep_consistent_seq_aug: bool = BOOL_FIELD(value=True, default_value=True, description="Keep consistent sequence augmentation", display_name="Keep consistent sequence augmentation")
    same_scene_in_batch: bool = BOOL_FIELD(value=True, default_value=True, description="Same scene in batch", display_name="Same scene in batch")


@dataclass
class Sparse4DValDatasetConfig:
    """Validation dataset configuration for Sparse4D."""

    ann_file: str = STR_FIELD(value=MISSING, default_value="", description="Path to annotation file", display_name="Path to annotation file")
    test_mode: bool = BOOL_FIELD(value=False, default_value=False, description="Test mode", display_name="Test mode")
    use_valid_flag: bool = BOOL_FIELD(value=True, default_value=True, description="Use valid flag", display_name="Use valid flag")
    tracking: bool = BOOL_FIELD(value=True, default_value=True, description="Tracking", display_name="Tracking")
    tracking_threshold: float = FLOAT_FIELD(value=0.2, default_value=0.2, valid_min=0, valid_max=1, description="Tracking threshold", display_name="Tracking threshold")
    same_scene_in_batch: bool = BOOL_FIELD(value=True, default_value=True, description="Same scene in batch", display_name="Same scene in batch")


@dataclass
class Sparse4DTestDatasetConfig:
    """Test dataset configuration for Sparse4D."""

    ann_file: str = STR_FIELD(value=MISSING, default_value="", description="Path to annotation file", display_name="Path to annotation file")
    test_mode: bool = BOOL_FIELD(value=True, default_value=True, description="Test mode", display_name="Test mode")
    use_valid_flag: bool = BOOL_FIELD(value=True, default_value=True, description="Use valid flag", display_name="Use valid flag")
    tracking: bool = BOOL_FIELD(value=True, default_value=True, description="Tracking", display_name="Tracking")
    tracking_threshold: float = FLOAT_FIELD(value=0.2, default_value=0.2, valid_min=0, valid_max=1, description="Tracking threshold", display_name="Tracking threshold")
    same_scene_in_batch: bool = BOOL_FIELD(value=True, default_value=True, description="Same scene in batch", display_name="Same scene in batch")


@dataclass
class Sparse4DLossRegConfig:
    """Regression loss configuration for Sparse4D."""

    type: str = STR_FIELD(value="sparse_box_3d", default_value="sparse_box_3d", description="Regression loss type", display_name="Regression loss type")
    box_weight: float = FLOAT_FIELD(value=0.25, default_value=0.25, valid_min=0, valid_max="inf", description="Box loss weight", display_name="Box loss weight")


@dataclass
class Sparse4DLossIDConfig:
    """ID loss configuration for Sparse4D."""

    type: str = STR_FIELD(value="cross_entropy_label_smooth", default_value="cross_entropy_label_smooth", description="ID loss type", display_name="ID loss type")
    num_ids: int = INT_FIELD(value=70, default_value=70, valid_min=1, valid_max="inf", description="Number of IDs", display_name="Number of IDs")


@dataclass
class Sparse4DLossConfig:
    """Loss configuration for Sparse4D."""

    cls: Sparse4DLossClsConfig = DATACLASS_FIELD(Sparse4DLossClsConfig(), description="Classification loss config", display_name="Classification loss config")
    reg: Sparse4DLossRegConfig = DATACLASS_FIELD(Sparse4DLossRegConfig(), description="Regression loss config", display_name="Regression loss config")
    id: Sparse4DLossIDConfig = DATACLASS_FIELD(Sparse4DLossIDConfig(), description="ID loss config", display_name="ID loss config")


@dataclass
class Sparse4DDecoderConfig:
    """Decoder configuration for Sparse4D."""

    type: str = STR_FIELD(value="SparseBox3DDecoder", default_value="SparseBox3DDecoder", description="Decoder type", display_name="Decoder type")
    score_threshold: float = FLOAT_FIELD(value=0.05, default_value=0.05, valid_min=0, valid_max=1, description="Score threshold", display_name="Score threshold")


@dataclass
class Sparse4DBNNeckConfig:
    """BNNeck configuration for Sparse4D."""

    type: str = STR_FIELD(value="bnneck", default_value="bnneck", description="BNNeck type", display_name="BNNeck type")
    feat_dim: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Feature dimension", display_name="Feature dimension")
    num_ids: int = INT_FIELD(value=70, default_value=70, valid_min=1, valid_max="inf", description="Number of IDs", display_name="Number of IDs")


@dataclass
class Sparse4DVisibilityNetConfig:
    """VisibilityNet configuration for Sparse4D."""

    type: str = STR_FIELD(value="visibility_net", default_value="visibility_net", description="VisibilityNet type", display_name="VisibilityNet type")
    embedding_dim: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimension", display_name="Embedding dimension")
    hidden_channels: int = INT_FIELD(value=32, default_value=32, valid_min=1, valid_max="inf", description="Hidden channels", display_name="Hidden channels")


@dataclass
class Sparse4DSamplerConfig:
    """Sampler configuration for Sparse4D."""

    num_dn_groups: int = INT_FIELD(value=5, default_value=5, valid_min=1, valid_max="inf", description="Number of DN groups", display_name="Number of DN groups")
    num_temp_dn_groups: int = INT_FIELD(value=3, default_value=3, valid_min=0, valid_max="inf", description="Number of temporal DN groups", display_name="Number of temporal DN groups")
    dn_noise_scale: List[float] = LIST_FIELD(
        arrList=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        default_value=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        description="DN noise scale",
        display_name="DN noise scale"
    )
    max_dn_gt: int = INT_FIELD(value=128, default_value=128, valid_min=1, valid_max="inf", description="Maximum DN ground truth", display_name="Maximum DN ground truth")
    add_neg_dn: bool = BOOL_FIELD(value=True, default_value=True, description="Add negative DN", display_name="Add negative DN")
    cls_weight: float = FLOAT_FIELD(value=2.0, default_value=2.0, valid_min=0, valid_max="inf", description="Classification weight", display_name="Classification weight")
    box_weight: float = FLOAT_FIELD(value=0.25, default_value=0.25, valid_min=0, valid_max="inf", description="Box weight", display_name="Box weight")
    reg_weights: List[float] = LIST_FIELD(
        arrList=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        default_value=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        description="Regression weights",
        display_name="Regression weights"
    )
    use_temporal_align: bool = BOOL_FIELD(value=False, default_value=False, description="Use temporal alignment", display_name="Use temporal alignment")


@dataclass
class Sparse4DDeformableModelConfig:
    """Deformable model configuration for Sparse4D."""

    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions")
    num_groups: int = INT_FIELD(value=8, default_value=8, valid_min=1, valid_max="inf", description="Number of groups")
    num_levels: int = INT_FIELD(value=4, default_value=4, valid_min=1, valid_max="inf", description="Number of levels", display_name="Number of levels")
    attn_drop: float = FLOAT_FIELD(value=0.15, default_value=0.15, valid_min=0, valid_max=1, description="Attention dropout", display_name="Attention dropout")
    use_deformable_func: bool = BOOL_FIELD(value=True, default_value=True, description="Use deformable function", display_name="Use deformable function")
    use_camera_embed: bool = BOOL_FIELD(value=False, default_value=False, description="Use camera embedding", display_name="Use camera embedding")
    residual_mode: str = STR_FIELD(value="cat", default_value="cat", description="Residual mode", valid_options="cat,add", display_name="Residual mode")
    num_cams: int = INT_FIELD(value=6, default_value=6, valid_min=1, valid_max="inf", description="Number of cameras", display_name="Number of cameras")
    max_num_cams: int = INT_FIELD(value=20, default_value=20, valid_min=1, valid_max="inf", description="Maximum number of cameras", display_name="Maximum number of cameras")
    proj_drop: float = FLOAT_FIELD(value=0.0, default_value=0.0, valid_min=0, valid_max=1, description="Projection dropout", display_name="Projection dropout")
    attn_drop: float = FLOAT_FIELD(value=0.0, default_value=0.0, valid_min=0, valid_max=1, description="Attention dropout", display_name="Attention dropout")
    kps_generator: Sparse4DKpsGeneratorConfig = DATACLASS_FIELD(Sparse4DKpsGeneratorConfig(), description="KPS generator config", display_name="KPS generator config")


@dataclass
class Sparse4DRefineLayerConfig:
    """Refine layer configuration for Sparse4D."""

    type: str = STR_FIELD(value="sparse_box_3d_refinement_module", default_value="sparse_box_3d_refinement_module", description="Refine layer type", display_name="Refine layer type")
    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions", display_name="Embedding dimensions")
    num_cls: int = INT_FIELD(value=8, default_value=8, valid_min=1, valid_max="inf", description="Number of classes", display_name="Number of classes")
    refine_yaw: bool = BOOL_FIELD(value=True, default_value=True, description="Refine yaw", display_name="Refine yaw")
    with_quality_estimation: bool = BOOL_FIELD(value=True, default_value=True, description="With quality estimation", display_name="With quality estimation")


@dataclass
class Sparse4DGraphModelConfig:
    """Graph model configuration for Sparse4D."""

    type: str = STR_FIELD(value="MultiheadAttention", default_value="MultiheadAttention", description="Graph model type", display_name="Graph model type")
    embed_dims: int = INT_FIELD(value=512, default_value=512, valid_min=1, valid_max="inf", description="Embedding dimensions", display_name="Embedding dimensions")
    num_heads: int = INT_FIELD(value=8, default_value=8, valid_min=1, valid_max="inf", description="Number of heads", display_name="Number of heads")
    batch_first: bool = BOOL_FIELD(value=True, default_value=True, description="Batch first", display_name="Batch first")
    dropout: float = FLOAT_FIELD(value=0.1, default_value=0.1, valid_min=0, valid_max=1, description="Dropout rate", display_name="Dropout rate")


@dataclass
class Sparse4DNormLayerConfig:
    """Norm layer configuration for Sparse4D."""

    type: str = STR_FIELD(value="LN", default_value="LN", description="Norm layer type", display_name="Norm layer type")
    normalized_shape: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Normalized shape", display_name="Normalized shape")


@dataclass
class Sparse4DActConfig:
    """Activation configuration for Sparse4D."""

    type: str = STR_FIELD(value="ReLU", default_value="ReLU", description="Activation type", display_name="Activation type")
    inplace: bool = BOOL_FIELD(value=True, default_value=True, description="Inplace", display_name="Inplace")


@dataclass
class Sparse4DFFNConfig:
    """FFN configuration for Sparse4D."""

    type: str = STR_FIELD(value="AsymmetricFFN", default_value="AsymmetricFFN", description="FFN type", display_name="FFN type")
    in_channels: int = INT_FIELD(value=512, default_value=512, valid_min=1, valid_max="inf", description="In channels", display_name="In channels")
    pre_norm: Sparse4DNormLayerConfig = DATACLASS_FIELD(Sparse4DNormLayerConfig(), description="Pre-norm config", display_name="Pre-norm config")
    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions", display_name="Embedding dimensions")
    feedforward_channels: int = INT_FIELD(value=1024, default_value=1024, valid_min=1, valid_max="inf", description="Feedforward channels", display_name="Feedforward channels")
    num_fcs: int = INT_FIELD(value=2, default_value=2, valid_min=1, valid_max="inf", description="Number of feedforward channels", display_name="Number of feedforward channels")
    ffn_drop: float = FLOAT_FIELD(value=0.1, default_value=0.1, valid_min=0, valid_max=1, description="FFN dropout", display_name="FFN dropout")
    act_cfg: Sparse4DActConfig = DATACLASS_FIELD(Sparse4DActConfig(), description="Activation config", display_name="Activation config")


@dataclass
class Sparse4DHeadConfig:
    """Head configuration for Sparse4D."""

    type: str = STR_FIELD(value="sparse4d", default_value="sparse4d", description="Head type", display_name="Head type")
    num_output: int = INT_FIELD(value=300, default_value=300, valid_min=1, valid_max="inf", description="Number of output instances", display_name="Number of output instances")
    cls_threshold_to_reg: float = FLOAT_FIELD(value=0.05, default_value=0.05, valid_min=0, valid_max=1, description="Classification threshold for regression", display_name="Classification threshold for regression")
    decouple_attn: bool = BOOL_FIELD(value=True, default_value=True, description="Decouple attention", display_name="Decouple attention")
    return_feature: bool = BOOL_FIELD(value=True, default_value=True, description="Return instance features", display_name="Return instance features")
    use_reid_sampling: bool = BOOL_FIELD(value=False, default_value=False, description="Use Re-ID sampling", display_name="Use Re-ID sampling")
    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions", display_name="Embedding dimensions")
    reid_dims: int = INT_FIELD(value=0, default_value=0, valid_min=1, valid_max="inf", description="Re-ID dimensions", display_name="Re-ID dimensions")
    num_groups: int = INT_FIELD(value=8, default_value=8, valid_min=1, valid_max="inf", description="Number of groups", display_name="Number of groups")
    num_decoder: int = INT_FIELD(value=6, default_value=6, valid_min=1, valid_max="inf", description="Number of decoder layers", display_name="Number of decoder layers")
    num_single_frame_decoder: int = INT_FIELD(value=1, default_value=1, valid_min=1, valid_max="inf", description="Number of single-frame decoder layers", display_name="Number of single-frame decoder layers")
    drop_out: float = FLOAT_FIELD(value=0.1, default_value=0.1, valid_min=0, valid_max=1, description="Dropout rate", display_name="Dropout rate")
    temporal: bool = BOOL_FIELD(value=True, default_value=True, description="Enable temporal modeling", display_name="Enable temporal modeling")
    with_quality_estimation: bool = BOOL_FIELD(value=True, default_value=True, description="Enable quality estimation", display_name="Enable quality estimation")
    operation_order: List[str] = LIST_FIELD(
        arrList=[
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine"
        ],
        default_value=[
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine"
        ],
        description="Operation order",
        display_name="Operation order"
    )
    visibility_net: Sparse4DVisibilityNetConfig = DATACLASS_FIELD(Sparse4DVisibilityNetConfig(), description="Visibility net config", display_name="Visibility net config")
    instance_bank: Sparse4DInstanceBankConfig = DATACLASS_FIELD(Sparse4DInstanceBankConfig(), description="Instance bank config", display_name="Instance bank config")
    anchor_encoder: Sparse4DAnchorEncoderConfig = DATACLASS_FIELD(Sparse4DAnchorEncoderConfig(), description="Anchor encoder config", display_name="Anchor encoder config")
    sampler: Sparse4DSamplerConfig = DATACLASS_FIELD(Sparse4DSamplerConfig(), description="Sampler config", display_name="Sampler config")
    reg_weights: List[float] = LIST_FIELD(
        arrList=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        default_value=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        description="Regression weights",
        display_name="Regression weights"
    )
    loss: Sparse4DLossConfig = DATACLASS_FIELD(Sparse4DLossConfig(), description="Loss config", display_name="Loss config")
    bnneck: Sparse4DBNNeckConfig = DATACLASS_FIELD(Sparse4DBNNeckConfig(), description="BN neck config", display_name="BN neck config")
    deformable_model: Sparse4DDeformableModelConfig = DATACLASS_FIELD(Sparse4DDeformableModelConfig(), description="Deformable model config", display_name="Deformable model config")
    refine_layer: Sparse4DRefineLayerConfig = DATACLASS_FIELD(Sparse4DRefineLayerConfig(), description="Refine layer config", display_name="Refine layer config")
    valid_vel_weight: float = FLOAT_FIELD(value=10.0, default_value=10.0, valid_min=0, valid_max="inf", description="Valid velocity weight", display_name="Valid velocity weight")
    graph_model: Sparse4DGraphModelConfig = DATACLASS_FIELD(Sparse4DGraphModelConfig(), description="Graph model config", display_name="Graph model config")
    temp_graph_model: Sparse4DGraphModelConfig = DATACLASS_FIELD(Sparse4DGraphModelConfig(), description="Temp graph model config", display_name="Temp graph model config")
    decoder: Sparse4DDecoderConfig = DATACLASS_FIELD(Sparse4DDecoderConfig(), description="Decoder config", display_name="Decoder config")
    norm_layer: Sparse4DNormLayerConfig = DATACLASS_FIELD(Sparse4DNormLayerConfig(), description="Norm layer config", display_name="Norm layer config")
    ffn: Sparse4DFFNConfig = DATACLASS_FIELD(Sparse4DFFNConfig(), description="FFN config", display_name="FFN config")


@dataclass
class Sparse4DModelConfig:
    """Model configuration for Sparse4D."""

    type: str = STR_FIELD(value="sparse4d", default_value="sparse4d", description="Model type", display_name="Model type")
    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions", display_name="Embedding dimensions")
    use_grid_mask: bool = BOOL_FIELD(value=True, default_value=True, description="Use grid mask", display_name="Use grid mask")
    use_deformable_func: bool = BOOL_FIELD(value=True, default_value=True, description="Use deformable function", display_name="Use deformable function")
    input_shape: List[int] = LIST_FIELD(arrList=[1408, 512], default_value=[1408, 512], description="Input image shape", display_name="Input image shape")
    backbone: Sparse4DBackboneConfig = DATACLASS_FIELD(Sparse4DBackboneConfig(), description="Backbone config", display_name="Backbone config")
    neck: Sparse4DNeckConfig = DATACLASS_FIELD(Sparse4DNeckConfig(), description="Neck config", display_name="Neck config")
    depth_branch: Sparse4DDepthBranchConfig = DATACLASS_FIELD(Sparse4DDepthBranchConfig(), description="Depth branch config", display_name="Depth branch config")
    head: Sparse4DHeadConfig = DATACLASS_FIELD(Sparse4DHeadConfig(), description="Head config", display_name="Head config")
    use_temporal_align: bool = BOOL_FIELD(value=False, default_value=False, description="Use temporal alignment", display_name="Use temporal alignment")


@dataclass
class Sparse4DAugmentationConfig:
    """Augmentation configuration for Sparse4D."""

    resize_lim: List[float] = LIST_FIELD(arrList=[0.7, 0.77], default_value=[0.7, 0.77], description="Resize limits", display_name="Resize limits")
    final_dim: List[int] = LIST_FIELD(arrList=[512, 1408], default_value=[512, 1408], description="Final dimensions", display_name="Final dimensions")
    bot_pct_lim: List[float] = LIST_FIELD(arrList=[0.0, 0.0], default_value=[0.0, 0.0], description="Bottom percentage limits", display_name="Bottom percentage limits")
    rot_lim: List[float] = LIST_FIELD(arrList=[-5.4, 5.4], default_value=[-5.4, 5.4], description="Rotation limits in degrees", display_name="Rotation limits in degrees")
    image_size: List[int] = LIST_FIELD(arrList=[1080, 1920], default_value=[1080, 1920], description="Original image size", display_name="Original image size")
    rand_flip: bool = BOOL_FIELD(value=True, default_value=True, description="Random flip", display_name="Random flip")
    rot3d_range: List[float] = LIST_FIELD(arrList=[-0.3925, 0.3925], default_value=[-0.3925, 0.3925], description="3D rotation range in radians", display_name="3D rotation range in radians")


@dataclass
class Sparse4DNormalizeConfig:
    """Normalization configuration for Sparse4D."""

    mean: List[float] = LIST_FIELD(arrList=[123.675, 116.28, 103.53], default_value=[123.675, 116.28, 103.53], description="Mean values for normalization", display_name="Mean values for normalization")
    std: List[float] = LIST_FIELD(arrList=[58.395, 57.12, 57.375], default_value=[58.395, 57.12, 57.375], description="Standard deviation values for normalization", display_name="Standard deviation values for normalization")
    to_rgb: bool = BOOL_FIELD(value=True, default_value=True, description="Convert to RGB", display_name="Convert to RGB")


@dataclass
class Sparse4DSequencesConfig:
    """Sequences configuration for Sparse4D."""

    split_num: int = INT_FIELD(value=100, default_value=100, valid_min=1, valid_max="inf", description="Number of sequence splits", display_name="Number of sequence splits")
    keep_consistent_aug: bool = BOOL_FIELD(value=True, default_value=True, description="Keep consistent augmentation", display_name="Keep consistent augmentation")
    same_scene_in_batch: bool = BOOL_FIELD(value=True, default_value=True, description="Keep same scene in batch", display_name="Keep same scene in batch")


@dataclass
class Sparse4DTrackingConfig:
    """Tracking configuration for Sparse4D."""

    enabled: bool = BOOL_FIELD(value=True, default_value=True, description="Enable tracking", display_name="Enable tracking")
    threshold: float = FLOAT_FIELD(value=0.2, default_value=0.2, valid_min=0, valid_max=1, description="Tracking threshold", display_name="Tracking threshold")


@dataclass
class Omniverse3DDetTrackDatasetConfig:
    """Dataset configuration for Sparse4D."""

    type: str = STR_FIELD(value="omniverse_3d_det_track", default_value="omniverse_3d_det_track", description="Dataset type", display_name="Dataset type")
    batch_size: int = INT_FIELD(value=2, default_value=2, valid_min=1, valid_max="inf", description="Batch size", display_name="Batch size")
    use_h5_file: bool = BOOL_FIELD(value=True, default_value=True, description="Use H5 file", display_name="Use H5 file")
    num_frames: int = INT_FIELD(value=200, default_value=200, valid_min=1, valid_max="inf", description="Number of frames", display_name="Number of frames")
    num_bev_groups: int = INT_FIELD(value=1, default_value=1, valid_min=1, valid_max="inf", description="Number of BEV groups", display_name="Number of BEV groups")
    data_root: str = STR_FIELD(value=MISSING, default_value="", description="Path to data root", display_name="Path to data root")
    anno_root: str = STR_FIELD(value=MISSING, default_value="", description="Path to annotation root", display_name="Path to annotation root")
    classes: List[str] = LIST_FIELD(
        arrList=["person", "humanoid", "nova_carter", "transporter", "forklift", "box", "pallet", "crate"],
        default_value=["person", "humanoid", "nova_carter", "transporter", "forklift", "box", "pallet", "crate"],
        description="Classes to detect",
        display_name="Classes to detect"
    )
    num_workers: int = INT_FIELD(value=4, default_value=4, valid_min=0, valid_max="inf", description="Number of workers", display_name="Number of workers")
    num_ids: int = INT_FIELD(value=70, default_value=70, valid_min=1, valid_max="inf", description="Number of IDs", display_name="Number of IDs")
    augmentation: Sparse4DAugmentationConfig = DATACLASS_FIELD(Sparse4DAugmentationConfig(), description="Augmentation config", display_name="Augmentation config")
    normalize: Sparse4DNormalizeConfig = DATACLASS_FIELD(Sparse4DNormalizeConfig(), description="Normalize config", display_name="Normalize config")
    sequences: Sparse4DSequencesConfig = DATACLASS_FIELD(Sparse4DSequencesConfig(), description="Sequences config", display_name="Sequences config")
    train_dataset: Sparse4DTrainDatasetConfig = DATACLASS_FIELD(Sparse4DTrainDatasetConfig(), description="Train dataset config", display_name="Train dataset config")
    val_dataset: Sparse4DValDatasetConfig = DATACLASS_FIELD(Sparse4DValDatasetConfig(), description="Val dataset config", display_name="Val dataset config")
    test_dataset: Sparse4DTestDatasetConfig = DATACLASS_FIELD(Sparse4DTestDatasetConfig(), description="Test dataset config", display_name="Test dataset config")


@dataclass
class Sparse4DEvaluateConfig(EvaluateConfig):
    """Evaluation configuration for Sparse4D."""

    metrics: List[str] = LIST_FIELD(arrList=["detection"], default_value=["detection"], description="Metrics to evaluate", display_name="Metrics to evaluate")
    tracking: Sparse4DTrackingConfig = DATACLASS_FIELD(Sparse4DTrackingConfig(), description="Tracking config", display_name="Tracking config")


@dataclass
class Sparse4DInferenceConfig(InferenceConfig):
    """Inference configuration for Sparse4D."""

    checkpoint: str = STR_FIELD(value=MISSING, default_value="", description="Path to checkpoint file", display_name="Path to checkpoint file")
    jsonfile_prefix: str = STR_FIELD(value="sparse4d_pred", default_value="sparse4d_pred", description="JSON file prefix", display_name="JSON file prefix")
    output_nvschema: bool = BOOL_FIELD(value=True, default_value=True, description="Output NVSchema", display_name="Output NVSchema")
    tracking: Sparse4DTrackingConfig = DATACLASS_FIELD(Sparse4DTrackingConfig())


@dataclass
class Sparse4DExportConfig(ExportConfig):
    """Export configuration for Sparse4D."""

    gpu_id: int = INT_FIELD(value=0, default_value=0, valid_min=0, valid_max="inf", description="GPU ID for export", display_name="GPU ID for export")
    onnx_file: str = STR_FIELD(value=MISSING, default_value="${export.results_dir}/sparse4d.onnx", description="Path to output ONNX file", display_name="Path to output ONNX file")


@dataclass
class Sparse4DVisConfig:
    """Visualization configuration for Sparse4D."""

    show: bool = BOOL_FIELD(value=True, default_value=True, description="Show visualization", display_name="Show visualization")
    vis_dir: str = STR_FIELD(value="./vis", default_value="./vis", description="Visualization directory", display_name="Visualization directory")
    vis_score_threshold: float = FLOAT_FIELD(value=0.25, default_value=0.25, valid_min=0, valid_max=1, description="Visualization score threshold", display_name="Visualization score threshold")
    n_images_col: int = INT_FIELD(value=6, default_value=6, valid_min=1, valid_max="inf", description="Number of images per column", display_name="Number of images per column")
    viz_down_sample: int = INT_FIELD(value=3, default_value=3, valid_min=1, valid_max="inf", description="Visualization down sample", display_name="Visualization down sample")


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment configuration for Sparse4D."""

    train: Sparse4DTrainConfig = DATACLASS_FIELD(Sparse4DTrainConfig(), description="Train config", display_name="Train config")
    model: Sparse4DModelConfig = DATACLASS_FIELD(Sparse4DModelConfig(), description="Model config", display_name="Model config")
    dataset: Omniverse3DDetTrackDatasetConfig = DATACLASS_FIELD(Omniverse3DDetTrackDatasetConfig(), description="Dataset config", display_name="Dataset config")
    inference: Sparse4DInferenceConfig = DATACLASS_FIELD(Sparse4DInferenceConfig(), description="Inference config", display_name="Inference config")
    evaluate: Sparse4DEvaluateConfig = DATACLASS_FIELD(Sparse4DEvaluateConfig(), description="Evaluate config", display_name="Evaluate config")
    export: Sparse4DExportConfig = DATACLASS_FIELD(Sparse4DExportConfig(), description="Export config", display_name="Export config")
    vis: Sparse4DVisConfig = DATACLASS_FIELD(Sparse4DVisConfig(), description="Vis config", display_name="Vis config")
