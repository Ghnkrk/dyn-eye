"""
DINOv2 Feature Extractor

Extracts 384-dimensional feature vectors from image crops
using a pre-trained DINOv2 ViT-S/14 model.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("dinov2_extractor")

# Preprocessing for DINOv2 (224x224, standard ImageNet stats)
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class DinoV2Extractor:
    """
    Extracts feature embeddings from images using DINOv2 (ViT-S/14).
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Initialising DINOv2 (ViT-S/14) extractor on {self.device}")

        # Load pre-trained DINOv2 model from torch hub
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            log.error(f"Failed to load DINOv2 model: {e}")
            raise

    @torch.no_grad()
    def extract_single(self, image_path: str) -> np.ndarray:
        """Extract a 384-d L2-normalized feature vector from a single image."""
        img = Image.open(image_path).convert("RGB")
        tensor = _preprocess(img).unsqueeze(0).to(self.device)
        features = self.model(tensor)
        # Apply L2 normalization for robust similarity matching
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features.squeeze().cpu().numpy()

    @torch.no_grad()
    def extract_batch(
        self, image_paths: list[str], batch_size: int = cfg.FEATURE_BATCH_SIZE
    ) -> np.ndarray:
        """
        Extract features for a list of images in batches with L2 normalization.
        Returns an (N, 384) numpy array.
        """
        all_features = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(_preprocess(img))
                except Exception as e:
                    log.warning(f"Skipping {p}: {e}")
                    # Placeholder zero vector
                    tensors.append(torch.zeros(3, 224, 224))

            batch_tensor = torch.stack(tensors).to(self.device)
            feats = self.model(batch_tensor)
            # Apply L2 normalization for robust similarity matching
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            all_features.append(feats.cpu().numpy())

            log.debug(
                f"Extracted DINOv2 features for batch {start//batch_size + 1} "
                f"({len(batch_paths)} images)"
            )

        result = np.vstack(all_features)
        log.info(f"DINOv2 feature extraction complete: {result.shape}")
        return result
