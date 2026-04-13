# ============================================================
# NOTE: This is a DEVELOPMENT/SETUP version only.
# Contains the DeepCNN and SiameseNetwork architecture.
# Must match exactly what was used during training.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()

        self.features = nn.Sequential(
            # Stack 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Stack 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Stack 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Stack 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

            # Stack 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding head
        self.embedding = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        
        )

    def forward(self, x):
        x = self.features(x)        # [B, 256, H, W]
        x = self.gap(x)             # [B, 256, 1, 1]
        x = torch.flatten(x, 1)     # [B, 256]
        x = self.embedding(x)       # [B, 128]
        
        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)

        return x


class SiameseNetwork(nn.Module):
    def __init__(self, backbone):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone

    def forward(self, x1, x2):
        emb_a = self.backbone(x1)
        emb_b = self.backbone(x2)
        dist = F.pairwise_distance(emb_a, emb_b)
        return dist, emb_a, emb_b