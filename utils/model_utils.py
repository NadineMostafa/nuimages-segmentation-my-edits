"""
Source: https://github.com/pytorch/vision

"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from utils import transforms as T


def get_model_instance_segmentation(num_classes=33, pretrained=True, freeze_backbone=True):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)

    # Freeze backbone layers
    if freeze_backbone:
        for name, parameter in model.backbone.named_parameters():
            parameter.requires_grad_(False)

    # Replace prediction heads
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))

def load_model(model_filename, num_classes=33):
    model = get_model_instance_segmentation(num_classes, pretrained=False)
    model.load_state_dict(str(model_filename))
    return model