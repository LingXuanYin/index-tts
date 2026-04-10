"""
indextts/vc_train/model_adapter.py

Build and configure a MyModel instance for VC finetuning.

Key responsibilities:
  1. Load TTS s2mel.pth weights into a VC-configured MyModel
     (use_gpt_latent=False, in_channels=256, f0_condition=True, n_f0_bins=256)
  2. Handle shape mismatches automatically via load_checkpoint2's filtered loading:
     - content_in_proj: (512, 1024) in TTS → (512, 256) in VC → skipped, random init
     - f0_embedding:    Embedding(512, ...) → Embedding(256, ...) → skipped, random init
  3. Identify and return two disjoint parameter lists:
     - reset_params:   layers that need fast training (reset to random; new semantics)
     - finetune_params: DiT main body layers (existing knowledge; slow lr)
  4. Support reset_cond_projection config flag (experiment for Phase 1 ablation)

Parameter split (design D7):
  reset_params:
    - length_regulator.content_in_proj (shape mismatch → always reset)
    - length_regulator.f0_embedding    (n_f0_bins mismatch → always reset)
    - length_regulator.f0_mask         (untrainedparameter → always reset)
    - cfm.estimator.cond_x_merge_linear (content semantics changed → always reset)
    - cfm.estimator.cond_projection    (optional, controlled by reset_cond_projection flag)
  finetune_params:
    - everything else in model.models["cfm"] and model.models["length_regulator"]
      EXCEPT the reset_params above

Usage:
    from indextts.vc_train.model_adapter import build_vc_model
    model, reset_params, finetune_params = build_vc_model(config_dict)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from munch import Munch

from indextts.s2mel.modules.commons import MyModel, load_checkpoint2, recursive_munch


def _make_vc_args(config: Dict) -> Munch:
    """Convert config dict → Munch args object for MyModel.__init__.

    Args:
        config: Dict matching config_vc.yaml structure (must have config["s2mel"]).

    Returns:
        Munch object with s2mel fields at top level (the format MyModel expects).
    """
    return recursive_munch(config["s2mel"])


def build_vc_model(
    config: Dict,
) -> Tuple[MyModel, List[Dict], List[Dict]]:
    """Build and initialise a MyModel for VC finetuning.

    Steps:
      1. Build MyModel with VC config (in_channels=256, f0_condition=True, use_gpt_latent=False)
      2. Load TTS s2mel.pth weights; shape-mismatched layers are silently skipped
         (load_checkpoint2 uses strict=False + shape filter)
      3. Collect reset_params and finetune_params as separate param group dicts

    Args:
        config: Config dict, must have:
            - config["s2mel"]: MyModel architecture config
            - config["training"]["pretrained_s2mel"]: path to s2mel.pth
            - config["training"]["reset_cond_projection"]: bool (default False)

    Returns:
        Tuple of (model, reset_params, finetune_params) where:
          - model: MyModel instance in eval() mode after loading weights
          - reset_params: list of {"params": [...]} dicts for optimizer param group
          - finetune_params: list of {"params": [...]} dicts for optimizer param group
            (disjoint from reset_params)

    Raises:
        FileNotFoundError: If pretrained_s2mel path does not exist.
    """
    training_cfg = config.get("training", {})
    pth_path = training_cfg.get("pretrained_s2mel", "checkpoints/s2mel.pth")
    reset_cond_projection = training_cfg.get("reset_cond_projection", False)

    if not os.path.exists(pth_path):
        raise FileNotFoundError(
            f"Pretrained s2mel checkpoint not found: {pth_path!r}. "
            "Set training.pretrained_s2mel in config_vc.yaml."
        )

    # 1. Build model (VC config: in_channels=256, f0_condition=True, use_gpt_latent=False)
    args = _make_vc_args(config)
    model = MyModel(args, use_gpt_latent=False)

    # 2. Load TTS weights; shape-mismatched layers auto-skipped by load_checkpoint2
    model, _, _, _ = load_checkpoint2(
        model,
        optimizer=None,
        path=pth_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
        load_ema=False,
    )
    # load_checkpoint2 sets model.eval() — keep it for now
    # The caller (train.py) will call model.train() at training start

    # 3. Identify reset layers
    lr_module = model.models["length_regulator"]
    cfm_module = model.models["cfm"]
    dit = cfm_module.estimator  # DiT instance

    # Collect reset parameter sets (by id for fast lookup)
    reset_param_ids: set = set()

    def _register_reset(module: nn.Module) -> List[nn.Parameter]:
        params = list(module.parameters())
        for p in params:
            reset_param_ids.add(id(p))
        return params

    reset_param_tensors: List[nn.Parameter] = []

    # content_in_proj: Linear(256→512) — always reset (shape mismatch from TTS)
    reset_param_tensors.extend(_register_reset(lr_module.content_in_proj))

    # f0_embedding: Embedding(256, channels) — always reset (n_f0_bins changed)
    reset_param_tensors.extend(_register_reset(lr_module.f0_embedding))

    # f0_mask: Parameter (1, channels) — always reset (untrained in TTS)
    if hasattr(lr_module, "f0_mask") and lr_module.f0_mask is not None:
        reset_param_ids.add(id(lr_module.f0_mask))
        reset_param_tensors.append(lr_module.f0_mask)

    # cond_x_merge_linear: always reset (content semantic space changed)
    reset_param_tensors.extend(_register_reset(dit.cond_x_merge_linear))

    # cond_projection: optional reset (experiment ablation flag)
    if reset_cond_projection:
        reset_param_tensors.extend(_register_reset(dit.cond_projection))

    # 4. Collect finetune params (all model params NOT in reset set)
    finetune_param_tensors: List[nn.Parameter] = []
    for key in model.models:
        for p in model.models[key].parameters():
            if id(p) not in reset_param_ids:
                finetune_param_tensors.append(p)

    # Return as optimizer param group dicts
    reset_params = [{"params": reset_param_tensors}]
    finetune_params = [{"params": finetune_param_tensors}]

    return model, reset_params, finetune_params
