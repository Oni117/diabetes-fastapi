from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


# ===============================
# CBAM MODULE
# Same as training code
# ===============================
class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // ratio, channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(
            self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg, max_], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ===============================
# MODEL — EfficientNet-B1 + CBAM
# Same architecture as your training code
# ===============================
class EfficientNet_CBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = models.efficientnet_b1(weights=None)

        self.features = backbone.features
        self.cbam = CBAM(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ===============================
# LOAD MODEL
# ===============================
def load_model(model_path, num_classes, device):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = EfficientNet_CBAM(num_classes=num_classes)

    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location=device,
        )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {}

    for key, value in state_dict.items():
        clean_key = key.replace("module.", "")
        clean_state_dict[clean_key] = value

    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    return model