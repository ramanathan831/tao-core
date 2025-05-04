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

    type: str = STR_FIELD(value="adamw", default_value="adamw", description="Optimizer type", valid_options="adamw,adam,sgd")
    lr: float = FLOAT_FIELD(value=5e-5, default_value=5e-5, valid_min=0, valid_max="inf", automl_enabled="TRUE", description="Learning rate")
    weight_decay: float = FLOAT_FIELD(value=0.001, default_value=0.001, math_cond=">= 0.0", description="Weight decay coefficient")
    momentum: float = FLOAT_FIELD(value=0.9, default_value=0.9, math_cond=">= 0.0", description="Momentum for SGD")
    paramwise_cfg: Optional[Dict[str, Any]] = DICT_FIELD(
        hashMap={"custom_keys": {"img_backbone": {"lr_mult": 0.2}}},
        description="Parameters-wise configuration",
        default_value={"custom_keys": {"img_backbone": {"lr_mult": 0.2}}}
    )
    grad_clip: Optional[Dict[str, Any]] = DICT_FIELD(
        hashMap={"max_norm": 25, "norm_type": 2},
        description="Gradient clipping configuration",
        default_value={"max_norm": 25, "norm_type": 2}
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
        }
    )


@dataclass
class Sparse4DTrainConfig(TrainConfig):
    """Training configuration for Sparse4D."""

    validation_interval: float = FLOAT_FIELD(value=0.5, default_value=0.5, valid_min=0, valid_max="inf", description="Validation interval in epochs")
    checkpoint_interval: float = FLOAT_FIELD(value=0.5, default_value=0.5, valid_min=0, valid_max="inf", description="Checkpoint interval in epochs")
    pretrained_model_path: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to pretrained model")
    optim: Sparse4DOptimizerConfig = DATACLASS_FIELD(Sparse4DOptimizerConfig())


@dataclass
class Sparse4DBackboneConfig:
    """Backbone configuration for Sparse4D."""

    type: str = STR_FIELD(value="ResNet", default_value="ResNet", description="Backbone type", valid_options="resnet")
    depth: int = INT_FIELD(value=101, default_value=101, valid_min=18, valid_max=152, description="ResNet depth", valid_options="18,34,50,101,152")
    num_stages: int = INT_FIELD(value=4, default_value=4, valid_min=1, valid_max=4, description="Number of stages")
    frozen_stages: int = INT_FIELD(value=-1, default_value=-1, valid_min=-1, valid_max=4, description="Frozen stages (-1 for none)")
    norm_eval: bool = BOOL_FIELD(value=True, default_value=True, description="Set BatchNorm layers to eval mode")
    style: str = STR_FIELD(value="pytorch", default_value="pytorch", description="ResNet style", valid_options="pytorch,caffe")
    with_cp: bool = BOOL_FIELD(value=True, default_value=True, description="Use checkpoint to save memory")
    out_indices: Tuple[int, ...] = LIST_FIELD(arrList=[0, 1, 2, 3], default_value=[0, 1, 2, 3], description="Output indices")
    norm_cfg: Dict[str, Any] = DICT_FIELD(
        hashMap={"type": "BN", "requires_grad": False},
        description="Normalization configuration",
        default_value={"type": "BN", "requires_grad": False}
    )
    pretrained_backbone_path: Optional[str] = STR_FIELD(value=None, default_value="", description="Path to pretrained backbone weights")


@dataclass
class Sparse4DNeckConfig:
    """Neck configuration for Sparse4D."""

    type: str = STR_FIELD(value="FPN", default_value="FPN", description="Neck type", valid_options="FPN")
    num_outs: int = INT_FIELD(value=4, default_value=4, valid_min=1, valid_max="inf", description="Number of output levels")
    start_level: int = INT_FIELD(value=0, default_value=0, valid_min=0, valid_max="inf", description="Start level for FPN")
    out_channels: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Output channels")
    in_channels: List[int] = LIST_FIELD(arrList=[256, 512, 1024, 2048], default_value=[256, 512, 1024, 2048], description="Input channels")
    add_extra_convs: str = STR_FIELD(value="on_output", default_value="on_output", description="Type of extra conv", valid_options="on_input,on_lateral,on_output,False")
    relu_before_extra_convs: bool = BOOL_FIELD(value=True, default_value=True, description="Apply ReLU before extra convs")


@dataclass
class Sparse4DDepthBranchConfig:
    """Depth branch configuration for Sparse4D."""

    type: str = STR_FIELD(value="dense_depth", default_value="dense_depth", description="Depth branch type")
    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions")
    num_depth_layers: int = INT_FIELD(value=3, default_value=3, valid_min=1, valid_max="inf", description="Number of depth layers")
    loss_weight: float = FLOAT_FIELD(value=0.2, default_value=0.2, valid_min=0, valid_max="inf", description="Weight for depth loss")


@dataclass
class Sparse4DInstanceBankConfig:
    """Instance bank configuration for Sparse4D."""

    num_anchor: int = INT_FIELD(value=900, default_value=900, valid_min=1, valid_max="inf", description="Number of anchors")
    anchor: str = STR_FIELD(value="", default_value="", description="Path to anchor file")
    num_temp_instances: int = INT_FIELD(value=600, default_value=600, valid_min=0, valid_max="inf", description="Number of temporal instances")
    confidence_decay: float = FLOAT_FIELD(value=0.8, default_value=0.8, valid_min=0, valid_max=1, description="Confidence decay factor")
    feat_grad: bool = BOOL_FIELD(value=False, default_value=False, description="Enable gradients for features")
    default_time_interval: float = FLOAT_FIELD(value=0.033333, default_value=0.033333, valid_min=0, valid_max="inf", description="Default time interval")
    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions")
    use_temporal_align: bool = BOOL_FIELD(value=False, default_value=False, description="Use temporal alignment")


@dataclass
class Sparse4DAnchorEncoderConfig:
    """Anchor encoder configuration for Sparse4D."""

    type: str = STR_FIELD(value="SparseBox3DEncoder", default_value="SparseBox3DEncoder", description="Anchor encoder type")
    vel_dims: int = INT_FIELD(value=3, default_value=3, valid_min=1, valid_max="inf", description="Velocity dimensions")
    embed_dims: List[int] = LIST_FIELD(arrList=[128, 32, 32, 64], default_value=[128, 32, 32, 64], description="Embedding dimensions")
    mode: str = STR_FIELD(value="cat", default_value="cat", description="Mode", valid_options="cat,add")
    output_fc: bool = BOOL_FIELD(value=False, default_value=False, description="Output FC")
    in_loops: int = INT_FIELD(value=1, default_value=1, valid_min=1, valid_max="inf", description="In loops")
    out_loops: int = INT_FIELD(value=4, default_value=4, valid_min=1, valid_max="inf", description="Out loops")
    pos_embed_only: bool = BOOL_FIELD(value=False, default_value=False, description="Pos embed only")


@dataclass
class Sparse4DKpsGeneratorConfig:
    """KPS generator configuration for Sparse4D."""

    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions")
    num_learnable_pts: int = INT_FIELD(value=6, default_value=6, valid_min=1, valid_max="inf", description="Number of learnable points")
    fix_scale: List[List[Any]] = LIST_FIELD(arrList=[
        [0, 0, 0],
        [0.45, 0, 0],
        [-0.45, 0, 0],
        [0, 0.45, 0],
        [0, -0.45, 0],
        [0, 0, 0.45],
        [0, 0, -0.45]
    ], default_value=[
        [0, 0, 0],
        [0.45, 0, 0],
        [-0.45, 0, 0],
        [0, 0.45, 0],
        [0, -0.45, 0],
        [0, 0, 0.45],
        [0, 0, -0.45]
    ], description="Fixed scale")


@dataclass
class Sparse4DLossClsConfig:
    """Classification loss configuration for Sparse4D."""

    type: str = STR_FIELD(value="focal", default_value="focal", description="Classification loss type")
    use_sigmoid: bool = BOOL_FIELD(value=True, default_value=True, description="Use sigmoid")
    gamma: float = FLOAT_FIELD(value=2.0, default_value=2.0, valid_min=0, valid_max="inf", description="Focal loss gamma")
    alpha: float = FLOAT_FIELD(value=0.25, default_value=0.25, valid_min=0, valid_max=1, description="Focal loss alpha")
    loss_weight: float = FLOAT_FIELD(value=2.0, default_value=2.0, valid_min=0, valid_max="inf", description="Loss weight")


@dataclass
class Sparse4DTrainDatasetConfig:
    """Training dataset configuration for Sparse4D."""

    ann_file: str = STR_FIELD(value=MISSING, default_value="", description="Path to annotation file")
    test_mode: bool = BOOL_FIELD(value=False, default_value=False, description="Test mode")
    use_valid_flag: bool = BOOL_FIELD(value=True, default_value=True, description="Use valid flag")
    with_seq_flag: bool = BOOL_FIELD(value=True, default_value=True, description="With sequence flag")
    sequences_split_num: int = INT_FIELD(value=100, default_value=100, valid_min=1, valid_max="inf", description="Number of sequences")
    keep_consistent_seq_aug: bool = BOOL_FIELD(value=True, default_value=True, description="Keep consistent sequence augmentation")
    same_scene_in_batch: bool = BOOL_FIELD(value=True, default_value=True, description="Same scene in batch")


@dataclass
class Sparse4DValDatasetConfig:
    """Validation dataset configuration for Sparse4D."""

    ann_file: str = STR_FIELD(value=MISSING, default_value="", description="Path to annotation file")
    test_mode: bool = BOOL_FIELD(value=False, default_value=False, description="Test mode")
    use_valid_flag: bool = BOOL_FIELD(value=True, default_value=True, description="Use valid flag")
    tracking: bool = BOOL_FIELD(value=True, default_value=True, description="Tracking")
    tracking_threshold: float = FLOAT_FIELD(value=0.2, default_value=0.2, valid_min=0, valid_max=1, description="Tracking threshold")
    same_scene_in_batch: bool = BOOL_FIELD(value=True, default_value=True, description="Same scene in batch")


@dataclass
class Sparse4DTestDatasetConfig:
    """Test dataset configuration for Sparse4D."""

    ann_file: str = STR_FIELD(value=MISSING, default_value="", description="Path to annotation file")
    test_mode: bool = BOOL_FIELD(value=True, default_value=True, description="Test mode")
    use_valid_flag: bool = BOOL_FIELD(value=True, default_value=True, description="Use valid flag")
    tracking: bool = BOOL_FIELD(value=True, default_value=True, description="Tracking")
    tracking_threshold: float = FLOAT_FIELD(value=0.2, default_value=0.2, valid_min=0, valid_max=1, description="Tracking threshold")
    same_scene_in_batch: bool = BOOL_FIELD(value=True, default_value=True, description="Same scene in batch")


@dataclass
class Sparse4DLossRegConfig:
    """Regression loss configuration for Sparse4D."""

    type: str = STR_FIELD(value="sparse_box_3d", default_value="sparse_box_3d", description="Regression loss type")
    box_weight: float = FLOAT_FIELD(value=0.25, default_value=0.25, valid_min=0, valid_max="inf", description="Box loss weight")


@dataclass
class Sparse4DLossIDConfig:
    """ID loss configuration for Sparse4D."""

    type: str = STR_FIELD(value="cross_entropy_label_smooth", default_value="cross_entropy_label_smooth", description="ID loss type")
    num_ids: int = INT_FIELD(value=70, default_value=70, valid_min=1, valid_max="inf", description="Number of IDs")


@dataclass
class Sparse4DLossConfig:
    """Loss configuration for Sparse4D."""

    cls: Sparse4DLossClsConfig = DATACLASS_FIELD(Sparse4DLossClsConfig())
    reg: Sparse4DLossRegConfig = DATACLASS_FIELD(Sparse4DLossRegConfig())
    id: Sparse4DLossIDConfig = DATACLASS_FIELD(Sparse4DLossIDConfig())


@dataclass
class Sparse4DDecoderConfig:
    """Decoder configuration for Sparse4D."""

    type: str = STR_FIELD(value="SparseBox3DDecoder", default_value="SparseBox3DDecoder", description="Decoder type")
    score_threshold: float = FLOAT_FIELD(value=0.05, default_value=0.05, valid_min=0, valid_max=1, description="Score threshold")


@dataclass
class Sparse4DBNNeckConfig:
    """BNNeck configuration for Sparse4D."""

    type: str = STR_FIELD(value="bnneck", default_value="bnneck", description="BNNeck type")
    feat_dim: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Feature dimension")
    num_ids: int = INT_FIELD(value=70, default_value=70, valid_min=1, valid_max="inf", description="Number of IDs")


@dataclass
class Sparse4DVisibilityNetConfig:
    """VisibilityNet configuration for Sparse4D."""

    type: str = STR_FIELD(value="visibility_net", default_value="visibility_net", description="VisibilityNet type")
    embedding_dim: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimension")
    hidden_channels: int = INT_FIELD(value=32, default_value=32, valid_min=1, valid_max="inf", description="Hidden channels")


@dataclass
class Sparse4DSamplerConfig:
    """Sampler configuration for Sparse4D."""

    num_dn_groups: int = INT_FIELD(value=5, default_value=5, valid_min=1, valid_max="inf", description="Number of DN groups")
    num_temp_dn_groups: int = INT_FIELD(value=3, default_value=3, valid_min=0, valid_max="inf", description="Number of temporal DN groups")
    dn_noise_scale: List[float] = LIST_FIELD(
        arrList=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        default_value=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        description="DN noise scale"
    )
    max_dn_gt: int = INT_FIELD(value=128, default_value=128, valid_min=1, valid_max="inf", description="Maximum DN ground truth")
    add_neg_dn: bool = BOOL_FIELD(value=True, default_value=True, description="Add negative DN")
    cls_weight: float = FLOAT_FIELD(value=2.0, default_value=2.0, valid_min=0, valid_max="inf", description="Classification weight")
    box_weight: float = FLOAT_FIELD(value=0.25, default_value=0.25, valid_min=0, valid_max="inf", description="Box weight")
    reg_weights: List[float] = LIST_FIELD(
        arrList=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        default_value=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        description="Regression weights"
    )
    use_temporal_align: bool = BOOL_FIELD(value=False, default_value=False, description="Use temporal alignment")


@dataclass
class Sparse4DDeformableModelConfig:
    """Deformable model configuration for Sparse4D."""

    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions")
    num_groups: int = INT_FIELD(value=8, default_value=8, valid_min=1, valid_max="inf", description="Number of groups")
    num_levels: int = INT_FIELD(value=4, default_value=4, valid_min=1, valid_max="inf", description="Number of levels")
    attn_drop: float = FLOAT_FIELD(value=0.15, default_value=0.15, valid_min=0, valid_max=1, description="Attention dropout")
    use_deformable_func: bool = BOOL_FIELD(value=True, default_value=True, description="Use deformable function")
    use_camera_embed: bool = BOOL_FIELD(value=False, default_value=False, description="Use camera embedding")
    residual_mode: str = STR_FIELD(value="cat", default_value="cat", description="Residual mode", valid_options="cat,add")
    num_cams: int = INT_FIELD(value=6, default_value=6, valid_min=1, valid_max="inf", description="Number of cameras")
    max_num_cams: int = INT_FIELD(value=20, default_value=20, valid_min=1, valid_max="inf", description="Maximum number of cameras")
    proj_drop: float = FLOAT_FIELD(value=0.0, default_value=0.0, valid_min=0, valid_max=1, description="Projection dropout")
    attn_drop: float = FLOAT_FIELD(value=0.0, default_value=0.0, valid_min=0, valid_max=1, description="Attention dropout")
    kps_generator: Sparse4DKpsGeneratorConfig = DATACLASS_FIELD(Sparse4DKpsGeneratorConfig())


@dataclass
class Sparse4DRefineLayerConfig:
    """Refine layer configuration for Sparse4D."""

    type: str = STR_FIELD(value="sparse_box_3d_refinement_module", default_value="sparse_box_3d_refinement_module", description="Refine layer type")
    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions")
    num_cls: int = INT_FIELD(value=8, default_value=8, valid_min=1, valid_max="inf", description="Number of classes")
    refine_yaw: bool = BOOL_FIELD(value=True, default_value=True, description="Refine yaw")
    with_quality_estimation: bool = BOOL_FIELD(value=True, default_value=True, description="With quality estimation")


@dataclass
class Sparse4DGraphModelConfig:
    """Graph model configuration for Sparse4D."""

    type: str = STR_FIELD(value="MultiheadAttention", default_value="MultiheadAttention", description="Graph model type")
    embed_dims: int = INT_FIELD(value=512, default_value=512, valid_min=1, valid_max="inf", description="Embedding dimensions")
    num_heads: int = INT_FIELD(value=8, default_value=8, valid_min=1, valid_max="inf", description="Number of heads")
    batch_first: bool = BOOL_FIELD(value=True, default_value=True, description="Batch first")
    dropout: float = FLOAT_FIELD(value=0.1, default_value=0.1, valid_min=0, valid_max=1, description="Dropout rate")


@dataclass
class Sparse4DNormLayerConfig:
    """Norm layer configuration for Sparse4D."""

    type: str = STR_FIELD(value="LN", default_value="LN", description="Norm layer type")
    normalized_shape: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Normalized shape")


@dataclass
class Sparse4DActConfig:
    """Activation configuration for Sparse4D."""

    type: str = STR_FIELD(value="ReLU", default_value="ReLU", description="Activation type")
    inplace: bool = BOOL_FIELD(value=True, default_value=True, description="Inplace")


@dataclass
class Sparse4DFFNConfig:
    """FFN configuration for Sparse4D."""

    type: str = STR_FIELD(value="AsymmetricFFN", default_value="AsymmetricFFN", description="FFN type")
    in_channels: int = INT_FIELD(value=512, default_value=512, valid_min=1, valid_max="inf", description="In channels")
    pre_norm: Sparse4DNormLayerConfig = DATACLASS_FIELD(Sparse4DNormLayerConfig())
    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions")
    feedforward_channels: int = INT_FIELD(value=1024, default_value=1024, valid_min=1, valid_max="inf", description="Feedforward channels")
    num_fcs: int = INT_FIELD(value=2, default_value=2, valid_min=1, valid_max="inf", description="Number of feedforward channels")
    ffn_drop: float = FLOAT_FIELD(value=0.1, default_value=0.1, valid_min=0, valid_max=1, description="FFN dropout")
    act_cfg: Sparse4DActConfig = DATACLASS_FIELD(Sparse4DActConfig())


@dataclass
class Sparse4DHeadConfig:
    """Head configuration for Sparse4D."""

    type: str = STR_FIELD(value="sparse4d", default_value="sparse4d", description="Head type")
    num_output: int = INT_FIELD(value=300, default_value=300, valid_min=1, valid_max="inf", description="Number of output instances")
    cls_threshold_to_reg: float = FLOAT_FIELD(value=0.05, default_value=0.05, valid_min=0, valid_max=1, description="Classification threshold for regression")
    decouple_attn: bool = BOOL_FIELD(value=True, default_value=True, description="Decouple attention")
    return_feature: bool = BOOL_FIELD(value=True, default_value=True, description="Return instance features")
    use_reid_sampling: bool = BOOL_FIELD(value=False, default_value=False, description="Use Re-ID sampling")
    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions")
    reid_dims: int = INT_FIELD(value=0, default_value=0, valid_min=1, valid_max="inf", description="Re-ID dimensions")
    num_groups: int = INT_FIELD(value=8, default_value=8, valid_min=1, valid_max="inf", description="Number of groups")
    num_decoder: int = INT_FIELD(value=6, default_value=6, valid_min=1, valid_max="inf", description="Number of decoder layers")
    num_single_frame_decoder: int = INT_FIELD(value=1, default_value=1, valid_min=1, valid_max="inf", description="Number of single-frame decoder layers")
    drop_out: float = FLOAT_FIELD(value=0.1, default_value=0.1, valid_min=0, valid_max=1, description="Dropout rate")
    temporal: bool = BOOL_FIELD(value=True, default_value=True, description="Enable temporal modeling")
    with_quality_estimation: bool = BOOL_FIELD(value=True, default_value=True, description="Enable quality estimation")
    operation_order: List[str] = LIST_FIELD(arrList=[
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
        "deformable", "ffn", "norm", "refine"
    ], default_value=[
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
        "deformable", "ffn", "norm", "refine"
    ], description="Operation order")
    visibility_net: Sparse4DVisibilityNetConfig = DATACLASS_FIELD(Sparse4DVisibilityNetConfig())
    instance_bank: Sparse4DInstanceBankConfig = DATACLASS_FIELD(Sparse4DInstanceBankConfig())
    anchor_encoder: Sparse4DAnchorEncoderConfig = DATACLASS_FIELD(Sparse4DAnchorEncoderConfig())
    sampler: Sparse4DSamplerConfig = DATACLASS_FIELD(Sparse4DSamplerConfig())
    reg_weights: List[float] = LIST_FIELD(
        arrList=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        default_value=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        description="Regression weights"
    )
    loss: Sparse4DLossConfig = DATACLASS_FIELD(Sparse4DLossConfig())
    bnneck: Sparse4DBNNeckConfig = DATACLASS_FIELD(Sparse4DBNNeckConfig())
    deformable_model: Sparse4DDeformableModelConfig = DATACLASS_FIELD(Sparse4DDeformableModelConfig())
    refine_layer: Sparse4DRefineLayerConfig = DATACLASS_FIELD(Sparse4DRefineLayerConfig())
    valid_vel_weight: float = FLOAT_FIELD(value=10.0, default_value=10.0, valid_min=0, valid_max="inf", description="Valid velocity weight")
    graph_model: Sparse4DGraphModelConfig = DATACLASS_FIELD(Sparse4DGraphModelConfig())
    temp_graph_model: Sparse4DGraphModelConfig = DATACLASS_FIELD(Sparse4DGraphModelConfig())
    decoder: Sparse4DDecoderConfig = DATACLASS_FIELD(Sparse4DDecoderConfig())
    norm_layer: Sparse4DNormLayerConfig = DATACLASS_FIELD(Sparse4DNormLayerConfig())
    ffn: Sparse4DFFNConfig = DATACLASS_FIELD(Sparse4DFFNConfig())


@dataclass
class Sparse4DModelConfig:
    """Model configuration for Sparse4D."""

    type: str = STR_FIELD(value="sparse4d", default_value="sparse4d", description="Model type")
    embed_dims: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf", description="Embedding dimensions")
    use_grid_mask: bool = BOOL_FIELD(value=True, default_value=True, description="Use grid mask")
    use_deformable_func: bool = BOOL_FIELD(value=True, default_value=True, description="Use deformable function")
    input_shape: List[int] = LIST_FIELD(arrList=[1408, 512], default_value=[1408, 512], description="Input image shape")
    backbone: Sparse4DBackboneConfig = DATACLASS_FIELD(Sparse4DBackboneConfig())
    neck: Sparse4DNeckConfig = DATACLASS_FIELD(Sparse4DNeckConfig())
    depth_branch: Sparse4DDepthBranchConfig = DATACLASS_FIELD(Sparse4DDepthBranchConfig())
    head: Sparse4DHeadConfig = DATACLASS_FIELD(Sparse4DHeadConfig())
    use_temporal_align: bool = BOOL_FIELD(value=False, default_value=False, description="Use temporal alignment")


@dataclass
class Sparse4DAugmentationConfig:
    """Augmentation configuration for Sparse4D."""

    resize_lim: List[float] = LIST_FIELD(arrList=[0.7, 0.77], default_value=[0.7, 0.77], description="Resize limits")
    final_dim: List[int] = LIST_FIELD(arrList=[512, 1408], default_value=[512, 1408], description="Final dimensions")
    bot_pct_lim: List[float] = LIST_FIELD(arrList=[0.0, 0.0], default_value=[0.0, 0.0], description="Bottom percentage limits")
    rot_lim: List[float] = LIST_FIELD(arrList=[-5.4, 5.4], default_value=[-5.4, 5.4], description="Rotation limits in degrees")
    image_size: List[int] = LIST_FIELD(arrList=[1080, 1920], default_value=[1080, 1920], description="Original image size")
    rand_flip: bool = BOOL_FIELD(value=True, default_value=True, description="Random flip")
    rot3d_range: List[float] = LIST_FIELD(arrList=[-0.3925, 0.3925], default_value=[-0.3925, 0.3925], description="3D rotation range in radians")


@dataclass
class Sparse4DNormalizeConfig:
    """Normalization configuration for Sparse4D."""

    mean: List[float] = LIST_FIELD(arrList=[123.675, 116.28, 103.53], default_value=[123.675, 116.28, 103.53], description="Mean values for normalization")
    std: List[float] = LIST_FIELD(arrList=[58.395, 57.12, 57.375], default_value=[58.395, 57.12, 57.375], description="Standard deviation values for normalization")
    to_rgb: bool = BOOL_FIELD(value=True, default_value=True, description="Convert to RGB")


@dataclass
class Sparse4DSequencesConfig:
    """Sequences configuration for Sparse4D."""

    split_num: int = INT_FIELD(value=100, default_value=100, valid_min=1, valid_max="inf", description="Number of sequence splits")
    keep_consistent_aug: bool = BOOL_FIELD(value=True, default_value=True, description="Keep consistent augmentation")
    same_scene_in_batch: bool = BOOL_FIELD(value=True, default_value=True, description="Keep same scene in batch")


@dataclass
class Sparse4DTrackingConfig:
    """Tracking configuration for Sparse4D."""

    enabled: bool = BOOL_FIELD(value=True, default_value=True, description="Enable tracking")
    threshold: float = FLOAT_FIELD(value=0.2, default_value=0.2, valid_min=0, valid_max=1, description="Tracking threshold")


@dataclass
class Omniverse3DDetTrackDatasetConfig:
    """Dataset configuration for Sparse4D."""

    type: str = STR_FIELD(value="omniverse_3d_det_track", default_value="omniverse_3d_det_track", description="Dataset type")
    batch_size: int = INT_FIELD(value=2, default_value=2, valid_min=1, valid_max="inf", description="Batch size")
    use_h5_file: bool = BOOL_FIELD(value=True, default_value=True, description="Use H5 file")
    num_frames: int = INT_FIELD(value=200, default_value=200, valid_min=1, valid_max="inf", description="Number of frames")
    num_bev_groups: int = INT_FIELD(value=1, default_value=1, valid_min=1, valid_max="inf", description="Number of BEV groups")
    data_root: str = STR_FIELD(value=MISSING, default_value="", description="Path to data root")
    anno_root: str = STR_FIELD(value=MISSING, default_value="", description="Path to annotation root")
    classes: List[str] = LIST_FIELD(arrList=["person", "humanoid", "nova_carter", "transporter", "forklift", "box", "pallet", "crate"],
                                  default_value=["person", "humanoid", "nova_carter", "transporter", "forklift", "box", "pallet", "crate"],
                                  description="Classes to detect")
    num_workers: int = INT_FIELD(value=4, default_value=4, valid_min=0, valid_max="inf", description="Number of workers")
    num_ids: int = INT_FIELD(value=70, default_value=70, valid_min=1, valid_max="inf", description="Number of IDs")
    augmentation: Sparse4DAugmentationConfig = DATACLASS_FIELD(Sparse4DAugmentationConfig())
    normalize: Sparse4DNormalizeConfig = DATACLASS_FIELD(Sparse4DNormalizeConfig())
    sequences: Sparse4DSequencesConfig = DATACLASS_FIELD(Sparse4DSequencesConfig())
    train_dataset: Sparse4DTrainDatasetConfig = DATACLASS_FIELD(Sparse4DTrainDatasetConfig())
    val_dataset: Sparse4DValDatasetConfig = DATACLASS_FIELD(Sparse4DValDatasetConfig())
    test_dataset: Sparse4DTestDatasetConfig = DATACLASS_FIELD(Sparse4DTestDatasetConfig())


@dataclass
class Sparse4DEvaluateConfig(EvaluateConfig):
    """Evaluation configuration for Sparse4D."""

    metrics: List[str] = LIST_FIELD(arrList=["detection"], default_value=["detection"], description="Metrics to evaluate")
    tracking: Sparse4DTrackingConfig = DATACLASS_FIELD(Sparse4DTrackingConfig())


@dataclass
class Sparse4DInferenceConfig(InferenceConfig):
    """Inference configuration for Sparse4D."""

    checkpoint: str = STR_FIELD(value=MISSING, default_value="", description="Path to checkpoint file", display_name="Path to checkpoint file")
    jsonfile_prefix: str = STR_FIELD(value="sparse4d_pred", default_value="sparse4d_pred", description="JSON file prefix")
    output_nvschema: bool = BOOL_FIELD(value=True, default_value=True, description="Output NVSchema")
    tracking: Sparse4DTrackingConfig = DATACLASS_FIELD(Sparse4DTrackingConfig())


@dataclass
class Sparse4DExportConfig(ExportConfig):
    """Export configuration for Sparse4D."""

    gpu_id: int = INT_FIELD(value=0, default_value=0, valid_min=0, valid_max="inf", description="GPU ID for export")
    onnx_file: str = STR_FIELD(value=MISSING, default_value="${export.results_dir}/sparse4d.onnx", description="Path to output ONNX file")


@dataclass
class Sparse4DVisConfig:
    """Visualization configuration for Sparse4D."""

    show: bool = BOOL_FIELD(value=True, default_value=True, description="Show visualization")
    vis_dir: str = STR_FIELD(value="./vis", default_value="./vis", description="Visualization directory")
    vis_score_threshold: float = FLOAT_FIELD(value=0.25, default_value=0.25, valid_min=0, valid_max=1, description="Visualization score threshold")
    n_images_col: int = INT_FIELD(value=6, default_value=6, valid_min=1, valid_max="inf", description="Number of images per column")
    viz_down_sample: int = INT_FIELD(value=3, default_value=3, valid_min=1, valid_max="inf", description="Visualization down sample")


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment configuration for Sparse4D."""

    train: Sparse4DTrainConfig = DATACLASS_FIELD(Sparse4DTrainConfig())
    model: Sparse4DModelConfig = DATACLASS_FIELD(Sparse4DModelConfig())
    dataset: Omniverse3DDetTrackDatasetConfig = DATACLASS_FIELD(Omniverse3DDetTrackDatasetConfig())
    inference: Sparse4DInferenceConfig = DATACLASS_FIELD(Sparse4DInferenceConfig())
    evaluate: Sparse4DEvaluateConfig = DATACLASS_FIELD(Sparse4DEvaluateConfig())
    export: Sparse4DExportConfig = DATACLASS_FIELD(Sparse4DExportConfig())
    vis: Sparse4DVisConfig = DATACLASS_FIELD(Sparse4DVisConfig())
