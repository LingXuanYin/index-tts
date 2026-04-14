"""Full VC training pipeline v2 — incorporates F5-TTS/seed-vc best practices.

Key improvements over v1:
  1. EMA (weight EMA, decay=0.9999) — stabilizes inference even after grad spikes
  2. accelerator.backward() + accelerator.clip_grad_norm_() — multi-GPU safe
  3. Relaxed grad_clip=10.0 (seed-vc style) instead of aggressive 5.0 + skip
  4. Correct mel via indextts.s2mel.modules.audio.mel_spectrogram (log mel)
  5. bf16 autocast via accelerator (not manual)
"""
import torch
import os
import sys
import time
import glob
import json
import argparse
import numpy as np
import math

sys.path.insert(0, ".")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preprocess-only", action="store_true")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-duration", type=float, default=8.0)
    p.add_argument("--kmeans-clusters", type=int, default=200)
    p.add_argument("--eval-interval", type=int, default=5000)
    p.add_argument("--save-interval", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--aishell3-dir", default="/root/autodl-tmp/data_aishell3/train/wav")
    p.add_argument("--output-dir", default="/root/autodl-tmp/index-tts/data/vc/aishell3")
    p.add_argument("--resume", default=None)
    return p.parse_args()

# ============================================================
# TRAINING (v2: EMA + accelerator + relaxed grad clip)
# ============================================================
def train(args):
    from munch import Munch
    import yaml
    from torch.utils.tensorboard import SummaryWriter
    from torch.optim.lr_scheduler import LambdaLR
    from accelerate import Accelerator
    from indextts.vc_train.model_adapter import build_vc_model
    from indextts.vc_train.train import vc_train_step
    from indextts.vc_train.dataset import VCDataset, vc_collate_fn
    from indextts.vc_train.manifest import read_jsonl
    from indextts.vc.kmeans_quantizer import KMeansQuantizer

    # --- Accelerator (handles bf16/fp16, distributed, grad accum) ---
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    is_main = accelerator.is_main_process

    # --- Config ---
    with open("checkpoints/config.yaml") as f:
        cfg = Munch.fromDict(yaml.safe_load(f))
    cfg.s2mel.length_regulator.in_channels = 256
    cfg.s2mel.length_regulator.f0_condition = True
    cfg.s2mel.length_regulator.n_f0_bins = 256
    cfg.s2mel.DiT.f0_condition = True
    if not hasattr(cfg, "training"):
        cfg.training = Munch()
    cfg.training.pretrained_s2mel = "checkpoints/s2mel.pth"
    cfg.training.reset_cond_projection = False

    # --- Model ---
    model, reset_params, finetune_params = build_vc_model(cfg)
    if is_main:
        print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # --- EMA (manual implementation, no deepcopy/device issues) ---
    ema_decay = args.ema_decay
    ema_params = None  # Will be initialized after accelerator.prepare

    # --- Dataset ---
    kmeans = KMeansQuantizer(n_clusters=args.kmeans_clusters)
    kmeans.load(os.path.join(args.output_dir, "kmeans_codebook.pt"))

    entries_raw = read_jsonl(os.path.join(args.output_dir, "manifest.jsonl"))
    # Filter: max duration + remove single-utterance speakers (self-reconstruction degrades quality)
    spk_count = {}
    for m in entries_raw:
        spk_count[m.speaker_id] = spk_count.get(m.speaker_id, 0) + 1
    entries = []
    n_dur_skip = 0
    n_spk_skip = 0
    for m in entries_raw:
        if m.duration_s > args.max_duration:
            n_dur_skip += 1
            continue
        if spk_count[m.speaker_id] < 2:
            n_spk_skip += 1
            continue
        stem = os.path.splitext(os.path.basename(m.audio_path))[0]
        e = m.to_dict()
        e["feature_base"] = os.path.join(args.output_dir, "preprocessed", stem)
        entries.append(e)
    if is_main:
        n_spk = len(set(e["speaker_id"] for e in entries))
        print(f"Dataset: {len(entries)} utterances, {n_spk} speakers "
              f"(filtered: {n_dur_skip} too long, {n_spk_skip} single-spk)")

    ds = VCDataset(entries=entries, f0_strategy="source_contour")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=vc_collate_fn, drop_last=True, num_workers=4,
    )

    # --- Optimizer (single phase, all params) ---
    for g in reset_params + finetune_params:
        for p in g["params"]:
            p.requires_grad_(True)
    all_groups = reset_params + finetune_params
    optimizer = torch.optim.AdamW(all_groups, lr=args.lr, weight_decay=1e-2)

    # --- LR schedule: warmup + cosine decay ---
    def warmup_cosine(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine)

    # --- Prepare with accelerator ---
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    # Initialize EMA params (clone model weights AFTER accelerator.prepare moved them to GPU)
    unwrapped_model = accelerator.unwrap_model(model)
    ema_params = {n: p.data.clone() for n, p in unwrapped_model.named_parameters()}
    if is_main:
        print(f"EMA initialized (decay={ema_decay}, {len(ema_params)} params)")

    # --- Resume ---
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Load into unwrapped model
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("global_step", 0)
        if ema_params is not None and "ema_params" in ckpt:
            for n in ema_params:
                if n in ckpt["ema_params"]:
                    ema_params[n] = ckpt["ema_params"][n].to(device)
        if is_main:
            print(f"Resumed from {args.resume} at step {start_step}")

    # --- Training loop ---
    writer = SummaryWriter("runs/train_v2") if is_main else None
    os.makedirs("checkpoints/vc", exist_ok=True)

    global_step = start_step
    losses_window = []
    best_avg = float("inf")
    t0 = time.time()
    nan_count = 0

    if is_main:
        print(f"\nTraining v2: {args.steps} steps (from {start_step})")
        print(f"  lr={args.lr}, warmup={args.warmup_steps}, grad_clip={args.grad_clip}")
        print(f"  EMA={'enabled' if ema_params else 'disabled'}, bf16=True (accelerator)")
        print("=" * 70)

    model.train()
    while global_step < args.steps:
        for batch in loader:
            if global_step >= args.steps:
                break

            # --- Forward + backward via accelerator ---
            with accelerator.accumulate(model):
                loss = vc_train_step(
                    accelerator.unwrap_model(model), batch, device=str(device)
                )

                # NaN guard — CRASH LOUD, don't silently skip
                if not torch.isfinite(loss):
                    nan_count += 1
                    if is_main:
                        # Diagnostic dump before crashing
                        unwrapped_diag = accelerator.unwrap_model(model)
                        print(f"\n{'='*60}")
                        print(f"FATAL: NaN/Inf loss at step {global_step} (consecutive: {nan_count})")
                        print(f"  loss value: {loss.item()}")
                        print(f"  lr: {optimizer.param_groups[0]['lr']:.2e}")
                        # Check model weights
                        w_norms = []
                        for n, p in unwrapped_diag.named_parameters():
                            if p.data.isnan().any():
                                print(f"  NaN in weight: {n} shape={p.shape}")
                            w_norms.append((n, p.data.norm().item()))
                        top_w = sorted(w_norms, key=lambda x: -x[1])[:5]
                        print(f"  Top weight norms: {[(n, f'{v:.2f}') for n, v in top_w]}")
                        # Check batch
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                has_nan = v.isnan().any().item()
                                has_inf = v.isinf().any().item()
                                print(f"  batch[{k}]: shape={v.shape} nan={has_nan} inf={has_inf} range=[{v.min():.3f},{v.max():.3f}]")
                        print(f"{'='*60}")

                    if nan_count >= 3:
                        raise RuntimeError(
                            f"Training crashed: {nan_count} consecutive NaN losses at step {global_step}. "
                            f"Last good checkpoint: check checkpoints/vc/ for most recent v2_step*.pth. "
                            f"Likely cause: lr too high ({optimizer.param_groups[0]['lr']:.2e}) or data issue."
                        )
                    optimizer.zero_grad()
                    global_step += 1
                    continue

                accelerator.backward(loss)

                # Gradient clipping (via accelerator for multi-GPU safety)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                else:
                    grad_norm = torch.tensor(0.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # --- EMA update (only on sync steps) ---
            if ema_params is not None and accelerator.sync_gradients:
                with torch.no_grad():
                    for n, p in accelerator.unwrap_model(model).named_parameters():
                        if n in ema_params:
                            ema_params[n].mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            # --- Logging ---
            lv = loss.item()
            if np.isfinite(lv):
                losses_window.append(lv)
            if len(losses_window) > 100:
                losses_window.pop(0)
            avg100 = np.nanmean(losses_window) if losses_window else float("nan")
            lr_now = optimizer.param_groups[0]["lr"]

            if avg100 < best_avg and np.isfinite(avg100):
                best_avg = avg100

            if is_main:
                gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                if writer:
                    writer.add_scalar("train/loss", lv, global_step)
                    writer.add_scalar("train/avg100", avg100, global_step)
                    writer.add_scalar("train/grad_norm", gn, global_step)
                    writer.add_scalar("train/lr", lr_now, global_step)

                if global_step % 500 == 0:
                    elapsed = time.time() - t0
                    sps = (global_step - start_step + 1) / max(elapsed, 1)
                    eta = (args.steps - global_step) / max(sps, 0.01)
                    print(
                        f"Step {global_step:6d}/{args.steps} | "
                        f"loss={lv:.4f} avg100={avg100:.4f} | "
                        f"grad={gn:.1f} lr={lr_now:.2e} | "
                        f"{sps:.1f} step/s ETA {eta/3600:.1f}h"
                    )

            # --- Checkpoint ---
            if is_main and (global_step + 1) % args.save_interval == 0:
                unwrapped = accelerator.unwrap_model(model)
                ckpt_data = {
                    "model_state_dict": unwrapped.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step + 1,
                    "loss": lv,
                    "avg100": avg100,
                    "f0_strategy": "source_contour",
                    "nan_count": nan_count,
                }
                if ema_params is not None:
                    ckpt_data["ema_params"] = ema_params
                ckpt_path = f"checkpoints/vc/v2_step{global_step+1}.pth"
                torch.save(ckpt_data, ckpt_path)
                # Also save EMA-only weights for inference (swap EMA into model state_dict)
                if ema_params is not None:
                    ema_state = unwrapped.state_dict()
                    for n in ema_params:
                        if n in ema_state:
                            ema_state[n] = ema_params[n]
                    ema_path = f"checkpoints/vc/v2_ema_step{global_step+1}.pth"
                    torch.save({
                        "model_state_dict": ema_state,
                        "global_step": global_step + 1,
                        "f0_strategy": "source_contour",
                    }, ema_path)
                print(f"  >> Saved {ckpt_path}" + (f" + {ema_path}" if ema_params else ""))

                # Auto-cleanup: keep only last 3 checkpoints to prevent disk full
                import re as _re
                def _step_num(path):
                    m = _re.search(r'step(\d+)', path)
                    return int(m.group(1)) if m else 0
                for prefix in ["v2_step", "v2_ema_step"]:
                    old_ckpts = sorted(glob.glob(f"checkpoints/vc/{prefix}*.pth"), key=_step_num)
                    while len(old_ckpts) > 3:
                        removed = old_ckpts.pop(0)
                        os.remove(removed)
                        print(f"    Cleanup: deleted {os.path.basename(removed)}")

            global_step += 1

    # --- Final save ---
    if is_main:
        unwrapped = accelerator.unwrap_model(model)
        final_path = "checkpoints/vc/v2_final.pth"
        torch.save({
            "model_state_dict": unwrapped.state_dict(),
            "global_step": global_step,
            "f0_strategy": "source_contour",
        }, final_path)
        if ema_params is not None:
            ema_state = unwrapped.state_dict()
            for n in ema_params:
                if n in ema_state:
                    ema_state[n] = ema_params[n]
            ema_final = "checkpoints/vc/v2_ema_final.pth"
            torch.save({
                "model_state_dict": ema_state,
                "global_step": global_step,
                "f0_strategy": "source_contour",
            }, ema_final)
            print(f"EMA checkpoint: {ema_final}")
        if writer:
            writer.close()

        total = time.time() - t0
        print("=" * 70)
        print(f"Done: {global_step} steps in {total/3600:.1f}h")
        print(f"Final loss={lv:.4f} | Best avg100={best_avg:.4f}")
        print(f"NaN steps skipped: {nan_count}")
        print(f"Checkpoint: {final_path}")

if __name__ == "__main__":
    args = parse_args()
    if not args.preprocess_only:
        train(args)
