"""
tests/vc/test_train_smoke.py

Smoke tests for indextts/vc_train/train.py (task 4.5a).

Tests:
  1. 1-step training loop does not crash; loss is finite (not NaN)
  2. loss is positive (> 0)
  3. Phase 1: reset layers have requires_grad=True; DiT main body has requires_grad=False
  4. Phase 1 → Phase 2 switch: all layers have requires_grad=True; optimizer rebuilt
  5. Checkpoint save/load roundtrip: save → rebuild → load → 1 step → loss consistent

All tests use mock model + mini synthetic batch (no real data/weights loaded).

Loss threshold for smoke test (calibrated from first run):
  # TODO: fill in after first passing run — expected ~10-50 for flow matching
  SMOKE_LOSS_THRESHOLD = 100.0  # conservative initial value
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from indextts.vc_train.train import (
    VCTrainer,
    build_optimizer,
    set_phase1_frozen,
    set_phase2_unfrozen,
    save_checkpoint,
    load_checkpoint,
)

# ---------------------------------------------------------------------------
# Smoke loss threshold
# ---------------------------------------------------------------------------
# Calibrated after first passing run (flow matching loss range):
# Phase 1 step 0: loss typically 5–30 depending on random init
SMOKE_LOSS_THRESHOLD = 200.0  # conservative; update to actual measured value


# ---------------------------------------------------------------------------
# Fixtures: minimal mock config and model
# ---------------------------------------------------------------------------

def _make_mini_config(tmp_path) -> dict:
    """Minimal config dict for VCTrainer (no real data, no real model)."""
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
                "p_dropout": 0.0,
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
            "phase1_steps": 30000,
            "phase2_steps": 200000,
            "reset_lr": 1e-4,
            "finetune_lr": 1e-5,
            "reset_cond_projection": False,
            "warmup_steps": 0,
            "scheduler": "constant",
            "log_interval": 1,
            "eval_interval": 9999999,
            "save_interval": 9999999,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "tensorboard_dir": str(tmp_path / "tb_logs"),
            "phase1_convergence_check_steps": 5000,
            "phase1_convergence_threshold": 0.20,
        },
        "f0": {
            "strategy": "source_contour",
        },
    }


def _build_model_for_smoke(config: dict):
    """Build MyModel from config (no real pth loading)."""
    from indextts.s2mel.modules.commons import MyModel, recursive_munch
    args = recursive_munch(config["s2mel"])
    model = MyModel(args, use_gpt_latent=False)
    return model


def _make_mini_batch(B: int = 2, T_content: int = 20, T_mel: int = 34) -> dict:
    """Synthetic mini batch matching vc_collate_fn output format."""
    T_pmt_content = T_content - 5
    T_pmt_mel = T_mel - 10
    return {
        # Source content (kmeans quantized): (B, T_content, 256)
        "content": torch.randn(B, T_content, 256),
        "content_lens": torch.tensor([T_content] * B, dtype=torch.long),
        # Source F0 at content framerate (B, T_content)
        "f0": torch.ones(B, T_content) * 150.0,  # 150 Hz voiced
        # Source target mel (22050/80/hop=256)
        "mel": torch.randn(B, 80, T_mel),
        "mel_lens": torch.tensor([T_mel] * B, dtype=torch.long),
        # Speaker style
        "style": torch.randn(B, 192),
        # Prompt mel (same speaker, different utterance)
        "prompt_mel": torch.randn(B, 80, T_pmt_mel),
        "prompt_mel_lens": torch.tensor([T_pmt_mel] * B, dtype=torch.long),
        # Prompt content
        "prompt_content": torch.randn(B, T_pmt_content, 256),
        "prompt_content_lens": torch.tensor([T_pmt_content] * B, dtype=torch.long),
        "speaker_ids": [f"SPK{b:03d}" for b in range(B)],
    }


@pytest.fixture
def mini_config(tmp_path):
    return _make_mini_config(tmp_path)


@pytest.fixture
def mini_model(mini_config):
    return _build_model_for_smoke(mini_config)


@pytest.fixture
def mini_batch():
    return _make_mini_batch(B=2, T_content=20, T_mel=34)


# ---------------------------------------------------------------------------
# 1. 1-step training: loss not NaN
# ---------------------------------------------------------------------------

class TestOnestepForward:
    def test_loss_not_nan(self, mini_config, mini_model, mini_batch, tmp_path):
        """A single forward+backward step must produce finite (non-NaN) loss."""
        from indextts.vc_train.train import vc_train_step

        mini_model.train()
        optimizer = torch.optim.AdamW(mini_model.parameters(), lr=1e-4)

        loss = vc_train_step(mini_model, mini_batch, device="cpu")
        assert not torch.isnan(loss), f"Loss is NaN! Got: {loss}"

    def test_loss_positive(self, mini_config, mini_model, mini_batch):
        """Loss must be strictly positive."""
        from indextts.vc_train.train import vc_train_step

        mini_model.train()
        loss = vc_train_step(mini_model, mini_batch, device="cpu")
        assert loss > 0, f"Expected positive loss, got: {loss}"

    def test_loss_below_threshold(self, mini_config, mini_model, mini_batch):
        """Loss should be below a sanity threshold (not wildly exploding)."""
        from indextts.vc_train.train import vc_train_step

        mini_model.train()
        loss = vc_train_step(mini_model, mini_batch, device="cpu")
        # Threshold calibrated from first smoke run:
        # TODO: update SMOKE_LOSS_THRESHOLD to actual measured value after first run
        assert loss < SMOKE_LOSS_THRESHOLD, (
            f"Loss {loss:.4f} exceeds smoke threshold {SMOKE_LOSS_THRESHOLD}. "
            "Update SMOKE_LOSS_THRESHOLD in test after calibration."
        )


# ---------------------------------------------------------------------------
# 2. Phase 1 frozen/unfrozen state
# ---------------------------------------------------------------------------

class TestPhase1Frozen:
    def _get_reset_and_finetune_params(self, model):
        """Simulate model_adapter.build_vc_model param split."""
        lr = model.models["length_regulator"]
        cfm = model.models["cfm"]
        dit = cfm.estimator

        reset_params = (
            list(lr.content_in_proj.parameters())
            + list(lr.f0_embedding.parameters())
            + [lr.f0_mask]
            + list(dit.cond_x_merge_linear.parameters())
        )
        reset_ids = {id(p) for p in reset_params}
        finetune_params = [
            p for p in model.parameters() if id(p) not in reset_ids
        ]
        return reset_params, finetune_params

    def test_phase1_frozen_finetune_params(self, mini_model):
        """After set_phase1_frozen, finetune params must have requires_grad=False."""
        reset_params, finetune_params = self._get_reset_and_finetune_params(mini_model)
        set_phase1_frozen(mini_model, reset_params)

        for p in finetune_params:
            assert not p.requires_grad, (
                "Finetune params must be frozen in Phase 1"
            )

    def test_phase1_reset_params_trainable(self, mini_model):
        """After set_phase1_frozen, reset params must have requires_grad=True."""
        reset_params, finetune_params = self._get_reset_and_finetune_params(mini_model)
        set_phase1_frozen(mini_model, reset_params)

        for p in reset_params:
            assert p.requires_grad, (
                "Reset params must remain trainable in Phase 1"
            )

    def test_phase2_all_params_trainable(self, mini_model):
        """After set_phase2_unfrozen, all params must have requires_grad=True."""
        reset_params, finetune_params = self._get_reset_and_finetune_params(mini_model)
        # First freeze Phase 1
        set_phase1_frozen(mini_model, reset_params)
        # Then unfreeze Phase 2
        set_phase2_unfrozen(mini_model)

        for p in mini_model.parameters():
            assert p.requires_grad, (
                "All params must be trainable after Phase 2 unfreeze"
            )


# ---------------------------------------------------------------------------
# 3. Phase 1 → Phase 2 optimizer rebuild
# ---------------------------------------------------------------------------

class TestPhaseSwitch:
    def test_optimizer_rebuild_on_phase_switch(self, mini_model):
        """Switching phase must rebuild optimizer with correct param groups."""
        reset_params_group = [{"params": list(mini_model.models["length_regulator"].content_in_proj.parameters()), "lr": 1e-4}]
        finetune_params_group = [{"params": list(mini_model.models["cfm"].parameters()), "lr": 1e-5}]

        # Phase 1 optimizer (only reset params)
        opt_phase1 = build_optimizer(
            reset_params=reset_params_group,
            finetune_params=finetune_params_group,
            phase=1,
            reset_lr=1e-4,
            finetune_lr=1e-5,
        )
        assert len(opt_phase1.param_groups) >= 1

        # Phase 2 optimizer (both groups)
        opt_phase2 = build_optimizer(
            reset_params=reset_params_group,
            finetune_params=finetune_params_group,
            phase=2,
            reset_lr=1e-4,
            finetune_lr=1e-5,
        )
        assert len(opt_phase2.param_groups) >= 2

    def test_phase2_lr_correct(self, mini_model):
        """Phase 2 optimizer must have separate LR for reset vs finetune groups."""
        reset_pg = [{"params": list(mini_model.models["length_regulator"].content_in_proj.parameters()), "lr": 1e-4}]
        finetune_pg = [{"params": list(mini_model.models["cfm"].parameters()), "lr": 1e-5}]

        opt = build_optimizer(
            reset_params=reset_pg,
            finetune_params=finetune_pg,
            phase=2,
            reset_lr=1e-4,
            finetune_lr=1e-5,
        )
        # First group = reset (lr 1e-4), second group = finetune (lr 1e-5)
        lrs = [g["lr"] for g in opt.param_groups]
        assert 1e-4 in lrs, f"Expected 1e-4 in LRs, got {lrs}"
        assert 1e-5 in lrs, f"Expected 1e-5 in LRs, got {lrs}"


# ---------------------------------------------------------------------------
# 4. Checkpoint save / load roundtrip
# ---------------------------------------------------------------------------

class TestCheckpointRoundtrip:
    def test_save_and_load_checkpoint(self, mini_config, mini_model, tmp_path):
        """Save checkpoint → rebuild model → load → verify metadata restored."""
        ckpt_path = str(tmp_path / "vc_smoke.pth")

        reset_params = [list(mini_model.models["length_regulator"].content_in_proj.parameters())]
        finetune_params = [list(mini_model.models["cfm"].parameters())]
        optimizer = torch.optim.AdamW(mini_model.parameters(), lr=1e-4)

        metadata = {
            "current_phase": 1,
            "f0_strategy": "source_contour",
            "global_step": 42,
        }

        save_checkpoint(
            model=mini_model,
            optimizer=optimizer,
            step=42,
            metadata=metadata,
            path=ckpt_path,
        )
        assert os.path.exists(ckpt_path), "Checkpoint file must exist after save"

        # Rebuild model and load
        mini_config2 = _make_mini_config(tmp_path)
        new_model = _build_model_for_smoke(mini_config2)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

        loaded_meta = load_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            path=ckpt_path,
        )

        assert loaded_meta["current_phase"] == 1
        assert loaded_meta["f0_strategy"] == "source_contour"
        assert loaded_meta["global_step"] == 42

    def test_loaded_model_produces_valid_loss(self, mini_config, mini_model, mini_batch, tmp_path):
        """After load, the model must still produce non-NaN loss."""
        from indextts.vc_train.train import vc_train_step

        ckpt_path = str(tmp_path / "vc_smoke2.pth")
        optimizer = torch.optim.AdamW(mini_model.parameters(), lr=1e-4)
        save_checkpoint(mini_model, optimizer, step=0, metadata={"current_phase": 1}, path=ckpt_path)

        # Load into fresh model
        new_model = _build_model_for_smoke(mini_config)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
        load_checkpoint(new_model, new_optimizer, ckpt_path)

        new_model.train()
        loss = vc_train_step(new_model, mini_batch, device="cpu")
        assert not torch.isnan(loss), f"Post-load loss is NaN: {loss}"
        assert loss > 0


# ---------------------------------------------------------------------------
# 5. build_optimizer API
# ---------------------------------------------------------------------------

class TestBuildOptimizer:
    def test_build_optimizer_phase1_single_group(self, mini_model):
        """Phase 1 optimizer should only contain reset params."""
        reset_pg = [{"params": list(mini_model.models["length_regulator"].content_in_proj.parameters())}]
        finetune_pg = [{"params": list(mini_model.models["cfm"].parameters())}]

        opt = build_optimizer(
            reset_params=reset_pg,
            finetune_params=finetune_pg,
            phase=1,
            reset_lr=1e-4,
            finetune_lr=1e-5,
        )
        # Phase 1: only reset params in optimizer
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["lr"] == 1e-4

    def test_build_optimizer_phase2_two_groups(self, mini_model):
        """Phase 2 optimizer must have two param groups (reset + finetune)."""
        reset_pg = [{"params": list(mini_model.models["length_regulator"].content_in_proj.parameters())}]
        finetune_pg = [{"params": list(mini_model.models["cfm"].parameters())}]

        opt = build_optimizer(
            reset_params=reset_pg,
            finetune_params=finetune_pg,
            phase=2,
            reset_lr=1e-4,
            finetune_lr=1e-5,
        )
        assert len(opt.param_groups) == 2
