"""
Task 2.1 - Tests for MyModel VC mode (use_gpt_latent=False)

TDD: tests written BEFORE confirming implementation details.
"""
import sys
import os
import unittest
import torch
import tempfile

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from munch import Munch
from indextts.s2mel.modules.commons import MyModel, load_checkpoint2


def _make_minimal_args(
    use_f0_condition: bool = False,
    n_f0_bins: int = 512,
    in_channels: int = 1024,
):
    """Build a minimal Munch config that lets MyModel.__init__ succeed.

    The structure mirrors the real checkpoints/config.yaml s2mel section:
    args = the s2mel sub-config passed directly to MyModel/CFM/DiT.
    Top-level fields: dit_type, reg_loss_type, DiT, length_regulator.
    """
    args = Munch(
        # top-level fields used by CFM / BASECFM
        dit_type="DiT",
        reg_loss_type="l1",
        length_regulator=Munch(
            channels=512,
            sampling_ratios=(1, 1, 1, 1),
            is_discrete=False,
            in_channels=in_channels,
            vector_quantize=False,
            content_codebook_size=2048,
            n_codebooks=1,
            quantizer_dropout=0.0,
            f0_condition=use_f0_condition,
            n_f0_bins=n_f0_bins,
        ),
        DiT=Munch(
            hidden_dim=512,
            num_heads=8,
            depth=13,
            class_dropout_prob=0.1,
            block_size=8192,
            in_channels=80,
            style_condition=True,
            final_layer_type='wavenet',
            target='mel',
            content_dim=512,
            content_codebook_size=1024,
            content_type='discrete',
            f0_condition=use_f0_condition,
            n_f0_bins=n_f0_bins,
            content_codebooks=1,
            is_causal=False,
            long_skip_connection=True,
            zero_prompt_speech_token=False,
            time_as_token=False,
            style_as_token=False,
            uvit_skip_connection=True,
            add_resblock_in_transformer=False,
        ),
        wavenet=Munch(
            hidden_dim=512,
            num_layers=8,
            kernel_size=5,
            dilation_rate=1,
            p_dropout=0.2,
            style_condition=True,
        ),
        style_encoder=Munch(dim=192),
    )
    return args


class TestMyModelVCMode(unittest.TestCase):
    """Test MyModel with use_gpt_latent=False (VC mode)."""

    def test_vc_mode_no_gpt_layer(self):
        """use_gpt_latent=False should NOT register gpt_layer in models dict."""
        args = _make_minimal_args()
        model = MyModel(args, use_gpt_latent=False)
        self.assertNotIn(
            'gpt_layer', model.models,
            "VC mode (use_gpt_latent=False) must not register gpt_layer"
        )

    def test_vc_mode_has_cfm_and_length_regulator(self):
        """VC mode model must have cfm and length_regulator."""
        args = _make_minimal_args()
        model = MyModel(args, use_gpt_latent=False)
        self.assertIn('cfm', model.models)
        self.assertIn('length_regulator', model.models)

    def test_tts_mode_has_gpt_layer(self):
        """use_gpt_latent=True (TTS mode) must register gpt_layer (regression guard)."""
        args = _make_minimal_args()
        model = MyModel(args, use_gpt_latent=True)
        self.assertIn(
            'gpt_layer', model.models,
            "TTS mode (use_gpt_latent=True) must register gpt_layer"
        )

    def test_tts_mode_gpt_layer_shape(self):
        """gpt_layer must be Linear(1280→256→128→1024)."""
        args = _make_minimal_args()
        model = MyModel(args, use_gpt_latent=True)
        gpt_layer = model.models['gpt_layer']
        self.assertIsInstance(gpt_layer, torch.nn.Sequential)
        layers = list(gpt_layer.children())
        self.assertEqual(len(layers), 3)
        self.assertEqual(layers[0].in_features, 1280)
        self.assertEqual(layers[0].out_features, 256)
        self.assertEqual(layers[2].out_features, 1024)

    def test_f0_condition_registers_f0_embedding(self):
        """When f0_condition=True, length_regulator must have f0_embedding."""
        args = _make_minimal_args(use_f0_condition=True, n_f0_bins=256)
        model = MyModel(args, use_gpt_latent=False)
        lr = model.models['length_regulator']
        self.assertTrue(
            hasattr(lr, 'f0_embedding'),
            "f0_condition=True must register f0_embedding on InterpolateRegulator"
        )
        # f0_embedding should be Embedding(n_f0_bins=256, channels)
        emb = lr.f0_embedding
        self.assertIsInstance(emb, torch.nn.Embedding)
        self.assertEqual(emb.num_embeddings, 256)

    def test_load_checkpoint2_shape_mismatch_does_not_crash(self):
        """load_checkpoint2 must silently skip mismatched shapes (not raise)."""
        args = _make_minimal_args(use_f0_condition=False, n_f0_bins=512)
        model = MyModel(args, use_gpt_latent=False)

        # Build a fake checkpoint with shape-mismatched weights
        fake_state = {
            "net": {
                "cfm": {},          # empty – all keys will be skipped gracefully
                "length_regulator": {
                    # Intentionally wrong shape to trigger the skip branch
                    "content_in_proj.weight": torch.zeros(512, 999),  # wrong in_features
                },
            },
            "epoch": 0,
            "iters": 0,
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(fake_state, tmp_path)
            # Must not raise; shape-mismatched keys should be silently skipped
            load_checkpoint2(model, optimizer=None, path=tmp_path, load_only_params=True)
        finally:
            os.unlink(tmp_path)

    def test_vc_config_in_channels_256(self):
        """VC config uses in_channels=256 (HuBERT-soft dim)."""
        args = _make_minimal_args(in_channels=256)
        model = MyModel(args, use_gpt_latent=False)
        lr = model.models['length_regulator']
        # content_in_proj should be Linear(256 → channels)
        self.assertTrue(
            hasattr(lr, 'content_in_proj'),
            "InterpolateRegulator must have content_in_proj when in_channels is set"
        )
        proj = lr.content_in_proj
        self.assertEqual(proj.in_features, 256)


if __name__ == '__main__':
    unittest.main()
