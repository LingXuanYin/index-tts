"""
indextts/vc/kmeans_quantizer.py

K-means quantizer for HuBERT-soft content features (design D3).

Key design points:
  - Hard cluster assignment: each frame -> nearest cluster center vector
  - quantize_to_vector returns (B, T, D) with D = feature dim (256)
    NOT (B, T, n_clusters) — dimension is unchanged, values are cluster centers
  - Enables a second pass of speaker information removal after HuBERT-soft
  - Default n_clusters = 200 (configurable via config_vc.yaml)

Usage:
    q = KMeansQuantizer(n_clusters=200, dim=256)
    q.train(features_np)          # features_np: (N, 256) numpy array
    q.save("checkpoints/vc/kmeans_200.pt")

    # Load for inference / training
    q2 = KMeansQuantizer(n_clusters=200, dim=256)
    q2.load("checkpoints/vc/kmeans_200.pt")

    # Quantize
    features = torch.randn(B, T, 256)
    quantized = q2.quantize_to_vector(features)  # (B, T, 256) cluster centers

    # Bypass mode (debug / ablation)
    passthrough = q2.bypass(features)  # (B, T, 256) unchanged
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


class KMeansQuantizer:
    """K-means quantizer mapping content features to cluster center vectors.

    Args:
        n_clusters: Number of cluster centroids. Default 200 (design D3).
        dim: Feature dimensionality. Must match content encoder output dim (256).
        random_state: Reproducibility seed for MiniBatchKMeans.
    """

    def __init__(
        self,
        n_clusters: int = 200,
        dim: int = 256,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.dim = dim
        self._random_state = random_state
        self._kmeans: Optional[MiniBatchKMeans] = None
        self._centers: Optional[torch.Tensor] = None  # (n_clusters, dim)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, features: np.ndarray) -> None:
        """Fit MiniBatchKMeans on content features.

        Args:
            features: Float numpy array of shape (N, dim) or (B, T, dim).
                      Will be reshaped to (-1, dim) if 3D.
        """
        if features.ndim == 3:
            features = features.reshape(-1, self.dim)
        elif features.ndim != 2:
            raise ValueError(f"features must be 2D or 3D, got shape {features.shape}")

        if features.shape[1] != self.dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.dim}, got {features.shape[1]}"
            )

        self._kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self._random_state,
            batch_size=min(1024, len(features)),
            n_init="auto",
        )
        self._kmeans.fit(features)
        self._centers = torch.from_numpy(
            self._kmeans.cluster_centers_.astype(np.float32)
        )  # (n_clusters, dim)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save cluster centers to a .pt file.

        File format: {"centers": Tensor(n_clusters, dim), "n_clusters": int, "dim": int}

        Args:
            path: Output file path. Parent directory must exist.
        """
        if self._centers is None:
            raise RuntimeError("Must call train() or load() before save().")
        torch.save(
            {
                "centers": self._centers,
                "n_clusters": self.n_clusters,
                "dim": self.dim,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load cluster centers from a .pt file saved by save().

        Args:
            path: Path to the saved .pt file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"KMeans checkpoint not found: {path!r}. "
                "Run preprocessing with --train_kmeans to create it."
            )
        data = torch.load(path, map_location="cpu")
        self._centers = data["centers"].float()  # (n_clusters, dim)
        # Optionally update n_clusters / dim from file for safety
        if "n_clusters" in data:
            loaded_n = data["n_clusters"]
            if loaded_n != self.n_clusters:
                self.n_clusters = loaded_n
        if "dim" in data:
            loaded_dim = data["dim"]
            if loaded_dim != self.dim:
                self.dim = loaded_dim

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    @property
    def cluster_centers(self) -> Optional[torch.Tensor]:
        """Cluster center tensor of shape (n_clusters, dim), or None if untrained."""
        return self._centers

    def quantize_to_vector(
        self,
        features: torch.Tensor,
        mode: str = "hard",
    ) -> torch.Tensor:
        """Map each feature frame to its nearest cluster center vector.

        IMPORTANT: output dim = feature dim (256), NOT n_clusters.
        This is intentional — the quantized output is a cluster center vector,
        not a one-hot or integer index.

        Args:
            features: Float tensor of shape (B, T, D).
            mode: Only "hard" is supported (nearest-neighbor assignment).

        Returns:
            Float tensor of shape (B, T, D) where each frame is replaced by
            the nearest cluster center vector.

        Raises:
            RuntimeError: If train() or load() has not been called.
        """
        if self._centers is None:
            raise RuntimeError(
                "KMeansQuantizer has no cluster centers. "
                "Call train() or load() first."
            )
        if mode != "hard":
            raise ValueError(f"Only 'hard' quantization is supported, got {mode!r}")

        B, T, D = features.shape
        centers = self._centers.to(features.device)  # (K, D)

        # Reshape to (B*T, D) for efficient distance computation
        flat = features.reshape(B * T, D)  # (N, D)

        # Compute squared L2 distances: (N, K)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        flat_norm = (flat ** 2).sum(dim=-1, keepdim=True)        # (N, 1)
        center_norm = (centers ** 2).sum(dim=-1, keepdim=True).T  # (1, K)
        dot = flat @ centers.T                                     # (N, K)
        dists = flat_norm + center_norm - 2 * dot                  # (N, K)

        # Nearest center index
        indices = dists.argmin(dim=-1)  # (N,)

        # Look up center vectors
        quantized_flat = centers[indices]  # (N, D)
        return quantized_flat.reshape(B, T, D)

    def quantize_hard(self, features: torch.Tensor) -> torch.Tensor:
        """Return cluster indices (long tensor) for each frame.

        Args:
            features: Float tensor of shape (B, T, D).

        Returns:
            Long tensor of shape (B, T) with cluster indices in [0, n_clusters).
        """
        if self._centers is None:
            raise RuntimeError(
                "KMeansQuantizer has no cluster centers. "
                "Call train() or load() first."
            )
        B, T, D = features.shape
        centers = self._centers.to(features.device)
        flat = features.reshape(B * T, D)

        flat_norm = (flat ** 2).sum(dim=-1, keepdim=True)
        center_norm = (centers ** 2).sum(dim=-1, keepdim=True).T
        dot = flat @ centers.T
        dists = flat_norm + center_norm - 2 * dot
        indices = dists.argmin(dim=-1)  # (N,)
        return indices.reshape(B, T)

    def bypass(self, features: torch.Tensor) -> torch.Tensor:
        """Pass features through without quantization (debug / ablation mode).

        Useful for comparing quantized vs. raw HuBERT-soft features during
        training experiments (config flag: kmeans.enabled: false).

        Args:
            features: Float tensor of shape (B, T, D).

        Returns:
            The same tensor, unchanged.
        """
        return features
