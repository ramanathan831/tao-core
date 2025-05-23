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

"""Constants unifying individual networks."""

epochs_mapper = {
    "action_recognition": "train.num_epochs",
    "analytics": "",
    "annotations": "",
    "augmentation": "",
    "auto_label": "",
    "classification_pyt": "train.num_epochs",
    "classification_tf2": "train.num_epochs",
    "deformable_detr": "train.num_epochs",
    "detectnet_v2": "training_config.num_epochs",
    "dino": "train.num_epochs",
    "efficientdet_tf2": "train.num_epochs",
    "mal": "train.num_epochs",
    "ml_recog": "train.num_epochs",
    "ocdnet": "train.num_epochs",
    "ocrnet": "train.num_epochs",
    "optical_inspection": "train.num_epochs",
    "pointpillars": "train.num_epochs",
    "pose_classification": "train.num_epochs",
    "re_identification": "train.num_epochs",
    "segformer": "train.num_epochs",
    "unet": "training_config.epochs",
}

backbone_mapper = {
    "action_recognition": "model.backbone",
    "analytics": "",
    "annotations": "",
    "augmentation": "",
    "auto_label": "",
    "classification_pyt": "model.backbone.type",
    "classification_tf2": "model.backbone",
    "deformable_detr": "model.backbone",
    "detectnet_v2": "model_config.arch",
    "dino": "model.backbone",
    "efficientdet_tf2": "model.name",
    "mal": "model.arch",
    "ml_recog": "model.backbone",
    "ocdnet": "model.backbone",
    "ocrnet": "model.backbone",
    "optical_inspection": "model.backbone",
    "pointpillars": "model.backbone_2d.name",
    "pose_classification": "",
    "re_identification": "model.backbone",
    "segformer": "model.backbone.type",
    "unet": "model_config.arch",
}

image_size_mapper = {
    "action_recognition": "model.input_height,model.input_width",
    "analytics": "",
    "annotations": "",
    "augmentation": "",
    "auto_label": "",
    "classification_pyt": "",
    "classification_tf2": "model.input_height,model.input_width",
    "deformable_detr": "",
    "detectnet_v2": (
        "augmentation_config.preprocessing.output_image_height,"
        "augmentation_config.preprocessing.output_image_width"
    ),
    "dino": "",
    "efficientdet_tf2": "model.input_height,model.input_width",
    "mal": "",
    "ml_recog": "model.input_height,model.input_width",
    "ocdnet": "",
    "ocrnet": "model.input_height,model.input_width",
    "optical_inspection": "dataset.image_height,dataset.image_width",
    "pointpillars": "",
    "pose_classification": "",
    "re_identification": "model.input_height,model.input_width",
    "segformer": "dataset.segment.img_size",
    "unet": "model_config.model_input_height,model_config.model_input_width",
}

# Include your network if it has spec fields to load full network as PTM and loading backbone portion alone
ptm_mapper = {
    "backbone": {
        "classification_pyt": "model.backbone.pretrained_backbone_path",
        "segformer": "model.backbone.pretrained_backbone_path",
        "visual_changenet": "model.backbone.pretrained_backbone_path",
        "dino": "model.pretrained_backbone_path",
        "grounding_dino": "model.pretrained_backbone_path",
        "mask_grounding_dino": "model.pretrained_backbone_path",
    },
    "end_to_end": {
        "classification_pyt": "train.pretrained_model_path",
        "visual_changenet": "train.pretrained_model_path",
        "segformer": "train.pretrained_model_path",
        "dino": "train.pretrained_model_path",
    }
}
