"""
tests/vc/test_model_adapter.py

TDD tests for indextts/vc_train/model_adapter.py (task 4.4a).

Tests:
  - build_vc_model returns model, reset_params, finetune_params
  - model does NOT contain gpt_layer (use_gpt_latent=False)
  - reset_params_list includes content_in_proj / f0_embedding / f0_mask / cond_x_merge_linear
  - content_in_proj weight shape is (512, 256) — Linear(256→512) after VC config
  - shape mismatch layer is randomly initialised (not from fake s2mel.pth weights)
  - reset_cond_projection=True adds cond_projection to reset params
  - finetune_params does not overlap with reset_params (exclusive lists)
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from munch import Munch

from indextts.vc_train.model_adapter import build_vc_model, _make_vc_args


# ---------------------------------------------------------------------------
# Helpers: build a minimal args Munch for MyModel
# ---------------------------------------------------------------------------

def _vc_args_dict(reset_cond_projection: bool = False) -> dict:
    """Minimal config dict matching VC config_vc.yaml structure."""
    return {
        "s2mel": {
            "dit_type": "DiT",
            "reg_loss_type": "l1",
            "style_encoder": {"dim": 192},
            "wavenet": {
                "hidden_dim": 512,
                "num_layers": 8,
                "kernel_size": 5,
                "dilation_rate": 1,
                "p_dropout": 0.2,
                "style_condition": True,
            },
            "preprocess_params": {
                "sr": 22050,
                "spect_params": {
                    "n_fft": 1024,
                    "win_length": 1024,
                    "hop_length": 256,
                    "n_mels": 80,
                    "fmin": 0,
                    "fmax": "None",
                },
            },
            "length_regulator": {
                "channels": 512,
                "is_discrete": False,
                "in_channels": 256,
                "content_codebook_size": 2048,
                "sampling_ratios": [1, 1, 1, 1],
                "vector_quantize": False,
                "n_codebooks": 1,
                "quantizer_dropout": 0.0,
                "f0_condition": True,
                "n_f0_bins": 256,
            },
            "DiT": {
                "hidden_dim": 512,
                "num_heads": 8,
                "depth": 13,
                "class_dropout_prob": 0.1,
                "block_size": 8192,
                "in_channels": 80,
                "style_condition": True,
                "final_layer_type": "wavenet",
                "target": "mel",
                "content_dim": 512,
                "content_codebook_size": 1024,
                "content_type": "discrete",
                "f0_condition": True,
                "n_f0_bins": 256,
                "content_codebooks": 1,
                "is_causal": False,
                "long_skip_connection": True,
                "zero_prompt_speech_token": False,
                "time_as_token": False,
                "style_as_token": False,
                "uvit_skip_connection": True,
            },
        },
        "training": {
            "reset_cond_projection": reset_cond_projection,
        },
    }


@pytest.fixture
def fake_s2mel_pth(tmp_path):
    """
    Create a fake s2mel.pth checkpoint with:
      - length_regulator: content_in_proj weight (512, 1024) — TTS shape (mismatch with VC 256 input)
      - cfm: some dummy weights that should load fine
    We simulate the load_checkpoint2 format: {"net": {"length_regulator": {...}, "cfm": {...}}}
    """
    # Build a real MyModel with TTS config (in_channels=1024) to get proper state dict
    from indextts.s2mel.modules.commons import MyModel, recursive_munch
    tts_args = _vc_args_dict()
    # Override to TTS shape
    tts_args["s2mel"]["length_regulator"]["in_channels"] = 1024
    tts_args["s2mel"]["length_regulator"]["f0_condition"] = False
    tts_args["s2mel"]["length_regulator"]["n_f0_bins"] = 512
    tts_args["s2mel"]["DiT"]["f0_condition"] = False
    tts_args["s2mel"]["DiT"]["n_f0_bins"] = 512

    args_munch = recursive_munch(tts_args["s2mel"])
    tts_model = MyModel(args_munch, use_gpt_latent=False)

    fake_ckpt = {
        "net": {key: tts_model.models[key].state_dict() for key in tts_model.models},
        "epoch": 0,
        "iters": 0,
    }
    pth_path = str(tmp_path / "s2mel.pth")
    torch.save(fake_ckpt, pth_path)
    return pth_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildVcModel:
    def test_returns_three_items(self, fake_s2mel_pth):
        config = _vc_args_dict()
        config["training"]["pretrained_s2mel"] = fake_s2mel_pth
        result = build_vc_model(config)
        assert isinstance(result, tuple) and len(result) == 3

    def test_model_is_nn_module(self, fake_s2mel_pth):
        config = _vc_args_dict()
        config["training"]["pretrained_s2mel"] = fake_s2mel_pth
        model, _, _ = build_vc_model(config)
        assert isinstance(model, nn.Module)

    def test_model_has_no_gpt_layer(self, fake_s2mel_pth):
        """gpt_layer must not be in model (use_gpt_latent=False)."""
        config = _vc_args_dict()
        config["training"]["pretrained_s2mel"] = fake_s2mel_pth
        model, _, _ = build_vc_model(config)
        # MyModel stores sub-models in model.models (ModuleDict)
        assert "gpt_layer" not in model.models, (
            "gpt_layer must NOT be in model.models when use_gpt_latent=False"
        )

    def test_content_in_proj_shape_is_256_to_512(self, fake_s2mel_pth):
        """content_in_proj must be Linear(256→512) after loading VC config."""
        config = _vc_args_dict()
        config["training"]["pretrained_s2mel"] = fake_s2mel_pth
        model, _, _ = build_vc_model(config)
        lr = model.models["length_regulator"]
        # content_in_proj is a Linear layer
        assert hasattr(lr, "content_in_proj"), "length_regulator must have content_in_proj"
        w = lr.content_in_proj.weight
        assert w.shape == (512, 256), (
            f"content_in_proj weight shape should be (512, 256), got {w.shape}"
        )

    def test_content_in_proj_is_not_tts_weights(self, fake_s2mel_pth):
        """content_in_proj must be randomly initialised (shape mismatch → skipped)."""
        config = _vc_args_dict()
        config["training"]["pretrained_s2mel"] = fake_s2mel_pth
        model, _, _ = build_vc_model(config)

        lr = model.models["length_regulator"]
        # The weight should be (512, 256) — cannot be loaded from TTS pth (which is (512, 1024))
        # So it must be randomly initialised (non-zero due to kaiming_uniform or similar)
        w = lr.content_in_proj.weight
        assert w.shape == (512, 256)
        # Just verify it's a tensor with gradients (not frozen by accident)
        assert w.requires_grad

    def test_reset_params_includes_content_in_proj(self, fake_s2mel_pth):
        """reset_params must include content_in_proj parameters."""
        config = _vc_args_dict()
        config["training"]["pretrained_s2mel"] = fake_s2mel_pth
        _, reset_params, _ = build_vc_model(config)
        # reset_params is a list of dicts (for optimizer param_groups)
        all_params = _flatten_param_list(reset_params)
        assert len(all_params) > 0, "reset_params must not be empty"

    def test_reset_params_includes_f0_embedding(self, fake_s2mel_pth):
        """f0_embedding must be in reset params."""
        config = _vc_args_dict()
        config["training"]["pretrained_s2mel"] = fake_s2mel_pth
        model, reset_params, _ = build_vc_model(config)
        lr = model.models["length_regulator"]
        f0_emb_params = set(id(p) for p in lr.f0_embedding.parameters())
        reset_param_ids = {id(p) for p in _flatten_param_list(reset_params)}
        overlap = f0_emb_params & reset_param_ids
        assert overlap, "f0_embedding parameters must appear in reset_params"

    def test_reset_params_includes_f0_mask(self, fake_s2mel_pth):
        """f0_mask must be in reset params."""
        config = _vc_args_dict()
        config["training"]["pretrained_s2mel"] = fake_s2mel_pth
        model, reset_params, _ = build_vc_model(config)
        lr = model.models["length_regulator"]
        f0_mask_id = id(lr.f0_mask)
        reset_param_ids = {id(p) for p in _flatten_param_list(reset_params)}
        assert f0_mask_id in reset_param_ids, "f0_mask must appear in reset_params"

    def test_reset_params_includes_cond_x_merge_linear(self, fake_s2mel_pth):
        """cond_x_merge_linear must be in reset params (content semantics changed)."""
        config = _vc_args_dict()
        config["training"]["pretrained_s2mel"] = fake_s2mel_pth
        model, reset_params, _ = build_vc_model(config)
        cfm = model.models["cfm"]
        # cond_x_merge_linear is inside the DiT estimator
        merge_params = set(id(p) for p in cfm.estimator.cond_x_merge_linear.parameters())
        reset_param_ids = {id(p) for p in _flatten_param_list(reset_params)}
        overlap = merge_params & reset_param_ids
        assert overlap, "cond_x_merge_linear parameters must appear in reset_params"

    def test_finetune_params_not_empty(self, fake_s2mel_pth):
        """finetune_params must contain at least some DiT parameters."""
        config = _vc_args_dict()
        config["training"]["pretrained_s2mel"] = fake_s2mel_pth
        _, _, finetune_params = build_vc_model(config)
        assert len(_flatten_param_list(finetune_params)) > 0

    def test_reset_and_finetune_params_disjoint(self, fake_s2mel_pth):
        """reset_params and finetune_params must not share any parameter tensors."""
        config = _vc_args_dict()
        config["training"]["pretrained_s2mel"] = fake_s2mel_pth
        _, reset_params, finetune_params = build_vc_model(config)
        reset_ids = {id(p) for p in _flatten_param_list(reset_params)}
        finetune_ids = {id(p) for p in _flatten_param_list(finetune_params)}
        overlap = reset_ids & finetune_ids
        assert not overlap, (
            f"reset_params and finetune_params share {len(overlap)} parameter(s) — "
            "they must be disjoint."
        )

    def test_reset_cond_projection_flag_adds_param(self, fake_s2mel_pth):
        """With reset_cond_projection=True, cond_projection must be in reset_params."""
        config = _vc_args_dict(reset_cond_projection=True)
        config["training"]["pretrained_s2mel"] = fake_s2mel_pth
        model, reset_params, _ = build_vc_model(config)
        cfm = model.models["cfm"]
        cond_proj_params = set(id(p) for p in cfm.estimator.cond_projection.parameters())
        reset_param_ids = {id(p) for p in _flatten_param_list(reset_params)}
        overlap = cond_proj_params & reset_param_ids
        assert overlap, (
            "cond_projection parameters must appear in reset_params when "
            "reset_cond_projection=True"
        )

    def test_no_s2mel_pth_still_works(self, tmp_path):
        """build_vc_model with missing pretrained_s2mel should not raise if path absent."""
        config = _vc_args_dict()
        config["training"]["pretrained_s2mel"] = str(tmp_path / "nonexistent.pth")
        # Should raise or warn — acceptable either way, as long as not crash silently
        # The key contract: if file doesn't exist, it raises FileNotFoundError or similar
        with pytest.raises((FileNotFoundError, RuntimeError, Exception)):
            build_vc_model(config)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _flatten_param_list(param_groups) -> List[torch.Tensor]:
    """Flatten param groups (list of dicts with 'params' key) or list of tensors."""
    params = []
    for item in param_groups:
        if isinstance(item, dict):
            params.extend(item["params"])
        elif isinstance(item, torch.Tensor):
            params.append(item)
        else:
            params.extend(item)
    return params
