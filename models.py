import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_student_model(num_classes):
    """Builds the compact ResNet-18 student."""
    backbone = resnet_fpn_backbone('resnet18', pretrained=True)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

def get_teacher_model(num_classes, pretrained_weights_path=None):
    """Builds the heavy ResNet-101 teacher and strictly freezes its parameters."""
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    model = FasterRCNN(backbone, num_classes=num_classes)
    
    if pretrained_weights_path:
        print(f"Loading Teacher weights from: {pretrained_weights_path}")
        model.load_state_dict(torch.load(pretrained_weights_path, map_location='cpu'))
        
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    return model