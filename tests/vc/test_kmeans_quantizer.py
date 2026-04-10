"""
Task 3.3a - Tests for KMeansQuantizer (TDD - written BEFORE implementation)

KMeansQuantizer:
  - train(features_np): fit MiniBatchKMeans on (N, D) features
  - save(path): save cluster centers as torch tensor
  - load(path): load cluster centers from saved file
  - quantize_to_vector(features_tensor): (B, T, D) -> (B, T, D) cluster centers
  - bypass(features_tensor): (B, T, D) -> (B, T, D) identity pass-through

Key constraint: quantize_to_vector returns (B, T, 256) NOT (B, T, n_clusters)
"""
import sys
import os
import unittest
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestKMeansQuantizerTrain(unittest.TestCase):
    """Test training and persistence."""

    N_CLUSTERS = 10   # small for tests
    DIM = 256
    N_SAMPLES = 200   # small dataset

    def _make_random_features(self, n=200, dim=256):
        return np.random.randn(n, dim).astype(np.float32)

    def _make_quantizer(self, n_clusters=None):
        from indextts.vc.kmeans_quantizer import KMeansQuantizer
        n_clusters = n_clusters or self.N_CLUSTERS
        return KMeansQuantizer(n_clusters=n_clusters, dim=self.DIM)

    def test_train_sets_cluster_centers(self):
        """After train(), quantizer must have cluster_centers_ with shape (n_clusters, D)."""
        q = self._make_quantizer()
        feats = self._make_random_features()
        q.train(feats)
        centers = q.cluster_centers
        self.assertIsNotNone(centers, "cluster_centers must not be None after train()")
        self.assertEqual(centers.shape, (self.N_CLUSTERS, self.DIM))

    def test_train_cluster_centers_are_float(self):
        """Cluster centers must be float32 tensors."""
        q = self._make_quantizer()
        feats = self._make_random_features()
        q.train(feats)
        self.assertEqual(q.cluster_centers.dtype, torch.float32)

    def test_save_and_load_roundtrip(self):
        """save() + load() must preserve cluster centers exactly."""
        q = self._make_quantizer()
        feats = self._make_random_features()
        q.train(feats)
        original_centers = q.cluster_centers.clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        try:
            q.save(tmp_path)

            q2 = self._make_quantizer()
            q2.load(tmp_path)

            torch.testing.assert_close(
                q2.cluster_centers,
                original_centers,
                msg="Loaded cluster centers must match saved ones"
            )
        finally:
            os.unlink(tmp_path)

    def test_save_creates_file(self):
        """save() must create a file at the given path."""
        q = self._make_quantizer()
        feats = self._make_random_features()
        q.train(feats)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        os.unlink(tmp_path)  # remove it first

        try:
            q.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path), "save() must create the file")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_without_train_raises(self):
        """quantize_to_vector without training or loading must raise."""
        q = self._make_quantizer()
        feats = torch.randn(2, 10, self.DIM)
        with self.assertRaises((RuntimeError, AttributeError, ValueError)):
            q.quantize_to_vector(feats)


class TestKMeansQuantizeToVector(unittest.TestCase):
    """Test quantize_to_vector output shape and semantics."""

    DIM = 256
    N_CLUSTERS = 10

    def _make_trained_quantizer(self, n_clusters=None, dim=None):
        from indextts.vc.kmeans_quantizer import KMeansQuantizer
        n_clusters = n_clusters or self.N_CLUSTERS
        dim = dim or self.DIM
        q = KMeansQuantizer(n_clusters=n_clusters, dim=dim)
        feats = np.random.randn(300, dim).astype(np.float32)
        q.train(feats)
        return q

    def test_output_shape_is_B_T_D_not_n_clusters(self):
        """CRITICAL: quantize_to_vector returns (B, T, D) NOT (B, T, n_clusters).

        The cluster index embedding lookup table maps index -> center vector.
        Output dim must equal feature dim (256), not number of clusters (10).
        """
        q = self._make_trained_quantizer(n_clusters=10, dim=self.DIM)
        B, T, D = 2, 20, self.DIM
        feats = torch.randn(B, T, D)
        result = q.quantize_to_vector(feats)
        self.assertEqual(result.shape, (B, T, D),
                         f"Output shape must be ({B}, {T}, {D}), got {result.shape}. "
                         f"Output must be cluster CENTER VECTORS (dim={D}), "
                         f"NOT cluster indices (n_clusters={q.n_clusters})")

    def test_output_values_are_cluster_centers(self):
        """Each output vector must be one of the cluster center vectors."""
        q = self._make_trained_quantizer()
        feats = torch.randn(1, 5, self.DIM)
        result = q.quantize_to_vector(feats)
        centers = q.cluster_centers  # (n_clusters, D)

        # Each output frame must match one of the cluster centers exactly
        for t in range(result.shape[1]):
            frame = result[0, t]  # (D,)
            # Check if this frame matches any center
            dists = torch.norm(centers - frame.unsqueeze(0), dim=-1)  # (n_clusters,)
            min_dist = dists.min().item()
            self.assertAlmostEqual(
                min_dist, 0.0, places=4,
                msg=f"Frame {t} output must be exactly one cluster center; "
                    f"min dist to any center = {min_dist:.6f}"
            )

    def test_output_same_dtype_as_input(self):
        """Output dtype must be float32."""
        q = self._make_trained_quantizer()
        feats = torch.randn(1, 10, self.DIM)
        result = q.quantize_to_vector(feats)
        self.assertEqual(result.dtype, torch.float32)

    def test_batch_consistency(self):
        """Processing each sample independently must match batched processing."""
        q = self._make_trained_quantizer()
        feats = torch.randn(3, 10, self.DIM)

        batched = q.quantize_to_vector(feats)
        for b in range(feats.shape[0]):
            single = q.quantize_to_vector(feats[b:b+1])
            torch.testing.assert_close(
                batched[b:b+1], single,
                msg=f"Batch sample {b} must match single-sample result"
            )


class TestKMeansQuantizerBypass(unittest.TestCase):
    """Test bypass mode (identity pass-through for debugging)."""

    DIM = 256

    def _make_quantizer(self):
        from indextts.vc.kmeans_quantizer import KMeansQuantizer
        return KMeansQuantizer(n_clusters=10, dim=self.DIM)

    def test_bypass_returns_input_unchanged(self):
        """bypass() must return the input tensor unchanged."""
        q = self._make_quantizer()
        feats = torch.randn(2, 15, self.DIM)
        result = q.bypass(feats)
        torch.testing.assert_close(result, feats)

    def test_bypass_shape_preserved(self):
        """bypass() output shape must equal input shape."""
        q = self._make_quantizer()
        feats = torch.randn(3, 8, self.DIM)
        result = q.bypass(feats)
        self.assertEqual(result.shape, feats.shape)

    def test_bypass_works_without_training(self):
        """bypass() must work even without training (no model needed)."""
        q = self._make_quantizer()
        feats = torch.randn(1, 5, self.DIM)
        # Should not raise
        result = q.bypass(feats)
        self.assertEqual(result.shape, feats.shape)


class TestKMeansQuantizerInterface(unittest.TestCase):
    """Test public interface completeness."""

    def test_has_required_methods(self):
        """KMeansQuantizer must have: train, save, load, quantize_to_vector, bypass."""
        from indextts.vc.kmeans_quantizer import KMeansQuantizer
        required = ['train', 'save', 'load', 'quantize_to_vector', 'bypass']
        for method in required:
            self.assertTrue(
                hasattr(KMeansQuantizer, method),
                f"KMeansQuantizer must have method: {method}"
            )

    def test_n_clusters_attribute(self):
        """n_clusters attribute must be accessible."""
        from indextts.vc.kmeans_quantizer import KMeansQuantizer
        q = KMeansQuantizer(n_clusters=200, dim=256)
        self.assertEqual(q.n_clusters, 200)

    def test_dim_attribute(self):
        """dim attribute must be accessible."""
        from indextts.vc.kmeans_quantizer import KMeansQuantizer
        q = KMeansQuantizer(n_clusters=10, dim=256)
        self.assertEqual(q.dim, 256)


if __name__ == '__main__':
    unittest.main()
