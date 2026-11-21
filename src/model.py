"""
Model Architecture Definition:
Pretrained EfficientNet-B0 with custom binary classification head.
"""
import torch
import torch.nn as nn
from torchvision import models

def build_efficientnet_b0(pretrained=True, dropout_rate=0.3):
    """
    Builds EfficientNet-B0 for binary classification.
    Output: Raw logit (pass through sigmoid for probability).
    """
    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, 1)  # 1 output logit
    )
    return model

def load_trained_model(weights_path, device='cpu'):
    """
    Loads saved model weights.
    """
    model = build_efficientnet_b0(pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
