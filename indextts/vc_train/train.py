"""
indextts/vc_train/train.py

VC finetuning training script.

Architecture:
  - accelerate-driven launch (single/multi GPU, bf16/fp16 via accelerate config)
  - Two-phase finetune (design D7):
      Phase 1 (0~phase1_steps): only train reset layers (content_in_proj, f0_embedding,
               f0_mask, cond_x_merge_linear), DiT main body frozen
      Phase 2 (phase1_steps~phase2_steps): unfreeze all; reset layers lr=reset_lr,
               DiT layers lr=finetune_lr (10x smaller)
  - AdamW optimizer with per-group LR
  - Cosine / constant LR schedule with warmup
  - Checkpoint saves: model weights + optimizer state + metadata
      metadata: current_phase / f0_strategy / global_step
  - Tensorboard: loss / lr / grad_norm / val metrics

CLI:
  accelerate launch indextts/vc_train/train.py \\
      --config indextts/vc_train/config_vc.yaml \\
      [--resume_from checkpoints/vc/step_30000.pth] \\
      [--single_phase]          # bypass 2-phase, train all from step 0
      [--f0_strategy source_contour]

Design constraints:
  - NO hardcoded batch_size / device / precision in this file
  - All compute config via accelerate + CLI args
  - TDD: vc_train_step / build_optimizer / set_phase1_frozen / set_phase2_unfrozen
         are standalone functions for testability
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

from indextts.vc.f0_encoder import VALID_F0_STRATEGIES, STRATEGY_SOURCE_CONTOUR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions (testable without accelerate / full config)
# ---------------------------------------------------------------------------

def vc_train_step(
    model: nn.Module,
    batch: Dict[str, Any],
    device: str = "cpu",
) -> torch.Tensor:
    """Execute one VC training forward pass (design D1 data flow).

    Implements:
      Step 1: source content → length_regulator(f0) → cond_src (B, T_tgt, 512)
      Step 2: prompt content → length_regulator(f0=None) → cond_prompt (B, T_pmt, 512)
      Step 3: mu = cat([cond_prompt, cond_src], dim=1).transpose(1, 2) → (B, 512, T_total)
      Step 4: x1 = cat([prompt_mel, mel], dim=2) → (B, 80, T_total)
      Step 5: loss = cfm.forward(x1, x_lens, prompt_lens, mu, style)

    Args:
        model: MyModel instance (must be in train() mode).
        batch: Dict from vc_collate_fn with keys:
            content (B, T_content, 256), content_lens (B,),
            f0 (B, T_content), mel (B, 80, T_mel), mel_lens (B,),
            style (B, 192),
            prompt_mel (B, 80, T_pmt_mel), prompt_mel_lens (B,),
            prompt_content (B, T_pmt_content, 256), prompt_content_lens (B,)
        device: Target device string (moved here if tensors are on different device).

    Returns:
        Scalar loss tensor (with gradient).
    """
    # Move batch to device
    def _to(t):
        return t.to(device) if isinstance(t, torch.Tensor) else t

    content = _to(batch["content"])          # (B, T_content, 256)
    content_lens = _to(batch["content_lens"])  # (B,)
    f0 = _to(batch["f0"])                    # (B, T_content)
    mel = _to(batch["mel"])                  # (B, 80, T_mel)
    mel_lens = _to(batch["mel_lens"])        # (B,)
    style = _to(batch["style"])              # (B, 192)
    prompt_mel = _to(batch["prompt_mel"])    # (B, 80, T_pmt_mel)
    prompt_mel_lens = _to(batch["prompt_mel_lens"])  # (B,)
    prompt_content = _to(batch["prompt_content"])    # (B, T_pmt_content, 256)
    prompt_content_lens = _to(batch["prompt_content_lens"])  # (B,)

    length_regulator = model.models["length_regulator"]
    cfm = model.models["cfm"]

    # Ensure DiT transformer caches are initialised (required before forward pass)
    # Uses max_seq_length from config block_size; batch_size from actual batch
    B = content.shape[0]
    T_total = (mel_lens + prompt_mel_lens).max().item()
    cfm.estimator.setup_caches(max_batch_size=B, max_seq_length=max(T_total + 64, 512))

    # Step 1: source content → cond_src
    # length_regulator.forward(x, ylens, f0) → (out, olens, ...)
    cond_src_out = length_regulator(
        content,            # (B, T_content, 256)
        ylens=mel_lens,     # (B,) target mel frame counts
        f0=f0,              # (B, T_content) F0 in Hz
    )
    cond_src = cond_src_out[0]   # (B, T_tgt_mel, 512)

    # Step 2: prompt content → cond_prompt (no F0, use f0_mask)
    cond_pmt_out = length_regulator(
        prompt_content,          # (B, T_pmt_content, 256)
        ylens=prompt_mel_lens,   # (B,) prompt mel frame counts
        f0=None,                 # prompt uses f0_mask
    )
    cond_pmt = cond_pmt_out[0]  # (B, T_pmt_mel, 512)

    # Step 3: concatenate mu (time-first format for DiT cond_projection)
    # DiT.forward receives cond as (B, T, 512) (time-first), applies Linear on last dim
    # Note: design.md says channel-first but DiT code uses time-first (see:
    #   diffusion_transformer.py:214 cond [2,1863,512]->[2,1863,512] via cond_projection)
    # The cfm.forward docstring "(batch, mel_timesteps, 512)" is correct here.
    mu = torch.cat([cond_pmt, cond_src], dim=1)   # (B, T_total, 512) time-first

    # Step 4: construct x1 = [prompt_mel | target_mel]
    x1 = torch.cat([prompt_mel, mel], dim=2)        # (B, 80, T_total)
    x_lens = prompt_mel_lens + mel_lens             # (B,)
    prompt_lens = prompt_mel_lens                   # (B,)

    # Step 5: flow matching loss (only computed over target range, not prompt)
    loss, _ = cfm.forward(
        x1=x1,
        x_lens=x_lens,
        prompt_lens=prompt_lens,
        mu=mu,
        style=style,
    )

    return loss


def set_phase1_frozen(model: nn.Module, reset_params: List[nn.Parameter]) -> None:
    """Set Phase 1 parameter freeze state.

    Reset layers: requires_grad=True (will be trained)
    All other layers: requires_grad=False (frozen)

    Args:
        model: MyModel instance.
        reset_params: List of Parameter objects that should remain trainable.
    """
    reset_ids = {id(p) for p in reset_params}

    # Freeze all
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze reset layers
    for p in reset_params:
        p.requires_grad_(True)


def set_phase2_unfrozen(model: nn.Module) -> None:
    """Set Phase 2 parameter state: unfreeze all model parameters.

    Args:
        model: MyModel instance.
    """
    for p in model.parameters():
        p.requires_grad_(True)


def build_optimizer(
    reset_params: List[Dict],
    finetune_params: List[Dict],
    phase: int,
    reset_lr: float,
    finetune_lr: float,
) -> torch.optim.AdamW:
    """Build AdamW optimizer with appropriate param groups for the given phase.

    Phase 1: only reset_params (single param group, lr=reset_lr)
    Phase 2: reset_params + finetune_params (two groups, different LRs)

    Args:
        reset_params: List of param group dicts (reset layers).
        finetune_params: List of param group dicts (DiT main body).
        phase: 1 or 2.
        reset_lr: Learning rate for reset layers.
        finetune_lr: Learning rate for finetune layers (Phase 2 only).

    Returns:
        Configured AdamW optimizer.
    """
    if phase == 1:
        # Flatten reset params into one group with reset_lr
        all_reset = []
        for pg in reset_params:
            if isinstance(pg, dict):
                all_reset.extend(pg["params"])
            else:
                all_reset.extend(pg)
        param_groups = [{"params": all_reset, "lr": reset_lr}]

    elif phase == 2:
        # Two groups: reset (lr=reset_lr) + finetune (lr=finetune_lr)
        all_reset = []
        for pg in reset_params:
            if isinstance(pg, dict):
                all_reset.extend(pg["params"])
            else:
                all_reset.extend(pg)

        all_finetune = []
        for pg in finetune_params:
            if isinstance(pg, dict):
                all_finetune.extend(pg["params"])
            else:
                all_finetune.extend(pg)

        param_groups = [
            {"params": all_reset, "lr": reset_lr},
            {"params": all_finetune, "lr": finetune_lr},
        ]
    else:
        raise ValueError(f"phase must be 1 or 2, got {phase}")

    return torch.optim.AdamW(param_groups, weight_decay=1e-2)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: Dict,
    path: str,
) -> None:
    """Save training checkpoint to path.

    Format:
        {
            "net": {key: state_dict for key in model.models},
            "optimizer": optimizer.state_dict(),
            "step": int,
            "metadata": dict with current_phase / f0_strategy / global_step / ...
        }

    Args:
        model: MyModel instance.
        optimizer: Current optimizer.
        step: Current global training step.
        metadata: Dict containing current_phase, f0_strategy, and any extra fields.
        path: Output path for the checkpoint file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Collect model state dicts
    net = {}
    if hasattr(model, "models"):
        for key in model.models:
            net[key] = model.models[key].state_dict()
    else:
        net["model"] = model.state_dict()

    checkpoint = {
        "net": net,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "metadata": {**metadata, "global_step": step},
    }
    torch.save(checkpoint, path)
    logger.info("Checkpoint saved to %s (step=%d)", path, step)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
) -> Dict:
    """Load checkpoint from path.

    Handles shape mismatches gracefully (same logic as load_checkpoint2):
    mismatched keys are skipped with a warning, allowing partial loading.

    Args:
        model: MyModel instance to restore weights into.
        optimizer: Optimizer to restore state into (None = skip optimizer).
        path: Path to the checkpoint file.

    Returns:
        metadata dict (contains current_phase, f0_strategy, global_step, etc.)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path!r}")

    state = torch.load(path, map_location="cpu")
    net = state.get("net", {})

    # Load model weights (with shape mismatch tolerance)
    if hasattr(model, "models"):
        for key in model.models:
            if key in net:
                model_sd = model.models[key].state_dict()
                filtered = {
                    k: v for k, v in net[key].items()
                    if k in model_sd and v.shape == model_sd[k].shape
                }
                skipped = set(net[key].keys()) - set(filtered.keys())
                if skipped:
                    logger.warning("load_checkpoint: skipped keys (shape mismatch): %s", skipped)
                model.models[key].load_state_dict(filtered, strict=False)
    else:
        model.load_state_dict(net.get("model", {}), strict=False)

    # Restore optimizer
    if optimizer is not None and "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception as e:
            logger.warning("Could not restore optimizer state: %s", e)

    metadata = state.get("metadata", {})
    if "step" in state and "global_step" not in metadata:
        metadata["global_step"] = state["step"]

    return metadata


def get_lr_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    scheduler_type: str = "cosine",
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a LR scheduler with linear warmup + cosine/constant decay.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps (linear ramp from 0 to base lr).
        total_steps: Total number of training steps.
        scheduler_type: "cosine" or "constant".

    Returns:
        A LambdaLR scheduler.
    """
    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        if scheduler_type == "constant":
            return 1.0
        # Cosine decay
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# VCTrainer class (full training loop)
# ---------------------------------------------------------------------------

class VCTrainer:
    """High-level VC training coordinator.

    Wraps accelerate, data loading, checkpointing, tensorboard.
    All GPU/precision config is deferred to accelerate.

    Args:
        config: Full config dict (loaded from config_vc.yaml).
        resume_from: Optional checkpoint path to resume from.
        single_phase: If True, skip Phase 1; train all params from step 0.
        f0_strategy: Override F0 strategy from config.
    """

    def __init__(
        self,
        config: Dict,
        resume_from: Optional[str] = None,
        single_phase: bool = False,
        f0_strategy: Optional[str] = None,
    ):
        self.config = config
        self.training_cfg = config.get("training", {})
        self.resume_from = resume_from
        self.single_phase = single_phase

        self.f0_strategy = f0_strategy or config.get("f0", {}).get("strategy", STRATEGY_SOURCE_CONTOUR)
        if self.f0_strategy not in VALID_F0_STRATEGIES:
            raise ValueError(
                f"Unknown f0_strategy: {self.f0_strategy!r}. "
                f"Valid options: {VALID_F0_STRATEGIES}"
            )

        self.phase1_steps = self.training_cfg.get("phase1_steps", 30000)
        self.phase2_steps = self.training_cfg.get("phase2_steps", 200000)
        self.reset_lr = float(self.training_cfg.get("reset_lr", 1e-4))
        self.finetune_lr = float(self.training_cfg.get("finetune_lr", 1e-5))
        self.warmup_steps = int(self.training_cfg.get("warmup_steps", 2000))
        self.scheduler_type = self.training_cfg.get("scheduler", "cosine")
        self.log_interval = int(self.training_cfg.get("log_interval", 100))
        self.eval_interval = int(self.training_cfg.get("eval_interval", 5000))
        self.save_interval = int(self.training_cfg.get("save_interval", 5000))
        self.checkpoint_dir = self.training_cfg.get("checkpoint_dir", "checkpoints/vc")
        self.tensorboard_dir = self.training_cfg.get("tensorboard_dir", "checkpoints/vc/tb_logs")

        self.global_step = 0
        self.current_phase = 1 if not single_phase else 2

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.writer: Optional[SummaryWriter] = None
        self.reset_params = None
        self.finetune_params = None

    def setup(self):
        """Initialise model, optimizer, tensorboard. Call before train()."""
        from indextts.vc_train.model_adapter import build_vc_model

        pretrained_pth = self.training_cfg.get("pretrained_s2mel", "checkpoints/s2mel.pth")
        self.config["training"]["pretrained_s2mel"] = pretrained_pth

        self.model, self.reset_params, self.finetune_params = build_vc_model(self.config)
        self.model.train()

        # Freeze state
        if self.single_phase:
            set_phase2_unfrozen(self.model)
            phase = 2
        else:
            reset_flat = [p for pg in self.reset_params for p in (pg["params"] if isinstance(pg, dict) else pg)]
            set_phase1_frozen(self.model, reset_flat)
            phase = 1
        self.current_phase = phase

        # Build optimizer
        self.optimizer = build_optimizer(
            reset_params=self.reset_params,
            finetune_params=self.finetune_params,
            phase=phase,
            reset_lr=self.reset_lr,
            finetune_lr=self.finetune_lr,
        )

        # Scheduler
        total_steps = self.phase2_steps if not self.single_phase else self.phase1_steps
        self.scheduler = get_lr_schedule(
            self.optimizer, self.warmup_steps, total_steps, self.scheduler_type
        )

        # Resume
        if self.resume_from:
            meta = load_checkpoint(self.model, self.optimizer, self.resume_from)
            self.global_step = meta.get("global_step", 0)
            self.current_phase = meta.get("current_phase", self.current_phase)
            logger.info(
                "Resumed from %s (step=%d, phase=%d)",
                self.resume_from, self.global_step, self.current_phase,
            )
            # Restore scheduler step
            for _ in range(self.global_step):
                self.scheduler.step()

        # Tensorboard
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        logger.info("Tensorboard writer at %s", self.tensorboard_dir)

    def _switch_to_phase2(self):
        """Switch from Phase 1 to Phase 2: unfreeze DiT, rebuild optimizer."""
        logger.info("=== Switching to Phase 2 (step=%d) ===", self.global_step)
        set_phase2_unfrozen(self.model)
        self.current_phase = 2
        self.optimizer = build_optimizer(
            reset_params=self.reset_params,
            finetune_params=self.finetune_params,
            phase=2,
            reset_lr=self.reset_lr,
            finetune_lr=self.finetune_lr,
        )
        remaining_steps = self.phase2_steps - self.global_step
        self.scheduler = get_lr_schedule(
            self.optimizer, 0, remaining_steps, self.scheduler_type
        )
        if self.writer:
            self.writer.add_scalar("train/phase", 2, self.global_step)

    def train_step(self, batch: Dict, device: str) -> float:
        """Execute one training step (forward + backward + optimizer step).

        Args:
            batch: Collated batch dict from vc_collate_fn.
            device: Device string for tensor placement.

        Returns:
            Loss value as Python float.
        """
        self.model.train()
        self.optimizer.zero_grad()

        loss = vc_train_step(self.model, batch, device=device)
        loss.backward()

        # Gradient norm logging
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        loss_val = loss.item()

        # Tensorboard
        if self.writer and self.global_step % self.log_interval == 0:
            self.writer.add_scalar("train/loss", loss_val, self.global_step)
            self.writer.add_scalar("train/grad_norm", grad_norm, self.global_step)
            for i, pg in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f"train/lr_group{i}", pg["lr"], self.global_step)

        # Phase switch check
        if (
            not self.single_phase
            and self.current_phase == 1
            and self.global_step >= self.phase1_steps
        ):
            self._switch_to_phase2()

        return loss_val

    def save(self):
        """Save checkpoint with metadata."""
        ckpt_path = os.path.join(
            self.checkpoint_dir, f"step_{self.global_step:07d}.pth"
        )
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=self.global_step,
            metadata={
                "current_phase": self.current_phase,
                "f0_strategy": self.f0_strategy,
                "global_step": self.global_step,
            },
            path=ckpt_path,
        )

    def close(self):
        if self.writer:
            self.writer.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="VC Finetune Training")
    parser.add_argument("--config", required=True, help="Path to config_vc.yaml")
    parser.add_argument("--resume_from", default=None, help="Checkpoint path to resume")
    parser.add_argument(
        "--single_phase",
        action="store_true",
        help="Bypass two-phase finetune; train all params from step 0",
    )
    parser.add_argument(
        "--f0_strategy",
        default=None,
        choices=list(VALID_F0_STRATEGIES),
        help="Override F0 strategy from config",
    )
    return parser.parse_args()


def main():
    """Main training entry point (accelerate launch compatible)."""
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    from indextts.vc_train.dataset import VCDataset, vc_collate_fn
    from indextts.vc_train.manifest import read_jsonl

    args = _parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    accelerator = Accelerator()
    device = accelerator.device

    logging.basicConfig(level=logging.INFO)
    logger.info("Training config: %s", args.config)
    logger.info("F0 strategy: %s", args.f0_strategy or config.get("f0", {}).get("strategy"))
    logger.info("VC model will load on setup — no preload at import time")

    trainer = VCTrainer(
        config=config,
        resume_from=args.resume_from,
        single_phase=args.single_phase,
        f0_strategy=args.f0_strategy,
    )
    trainer.setup()

    # Build dataset
    train_cfg = config.get("training", {})
    datasets_cfg = config.get("datasets", [])
    training_entries = []
    for ds_cfg in datasets_cfg:
        train_manifest = os.path.join(ds_cfg["path"], "train.jsonl")
        if os.path.exists(train_manifest):
            entries = read_jsonl(train_manifest)
            # Convert VCManifestEntry to the dict format VCDataset expects
            for e in entries:
                training_entries.append({
                    "audio_path": e.audio_path,
                    "feature_base": e.audio_path.replace(".wav", ""),
                    "speaker_id": e.speaker_id,
                    "language": e.language,
                    "duration_s": e.duration_s,
                    "text": e.text,
                })

    dataset = VCDataset(entries=training_entries, f0_strategy=trainer.f0_strategy)
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # batch_size is controlled via accelerate config
        collate_fn=vc_collate_fn,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )

    trainer.model, trainer.optimizer, dataloader = accelerator.prepare(
        trainer.model, trainer.optimizer, dataloader
    )

    # Training loop
    total_steps = train_cfg.get("phase2_steps", 200000)
    step = trainer.global_step

    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break

            loss = trainer.train_step(batch, device=str(device))
            step = trainer.global_step

            if step % train_cfg.get("log_interval", 100) == 0:
                logger.info("step=%d phase=%d loss=%.4f", step, trainer.current_phase, loss)

            if step % train_cfg.get("save_interval", 5000) == 0:
                trainer.save()

    trainer.save()
    trainer.close()
    logger.info("Training complete at step %d", trainer.global_step)


if __name__ == "__main__":
    main()
