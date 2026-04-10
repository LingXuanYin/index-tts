# indextts/vc — VC Core Modules

This package provides the core, training-framework-agnostic building blocks for
Trained Voice Conversion (VC).  All four modules are designed to be independently
importable and testable without loading any TTS model.

---

## Modules

### `content_encoder.py` — HuBERT-soft Content Encoder

Wraps `bshall/hubert-soft` (loaded via `torch.hub`) to extract speaker-invariant
content features from 16 kHz audio.

**Key class:** `ContentEncoder`

| Member | Description |
|--------|-------------|
| `ContentEncoder(model_name, device)` | Constructor. Downloads HuBERT-soft on first call (~378 MB). |
| `.extract(audio_16k)` | `(B, T_samples) → (B, T_frames, 256)` at ~50 Hz frame rate. |
| `.output_dim` | Always `256` (HuBERT-soft dimensionality). |

The model is **frozen** (eval + `requires_grad_(False)`); it is never fine-tuned.

**Example:**

```python
from indextts.vc.content_encoder import ContentEncoder
enc = ContentEncoder(device="cuda:0")
features = enc.extract(audio_tensor_16k)  # (B, T, 256)
```

---

### `f0_encoder.py` — F0 Extraction and Strategy Functions

RMVPE-based F0 extractor plus four **pure strategy functions** shared between
training (`vc_train/dataset.py`) and inference (`infer_vc_trained.py`).

**Constants (strategy names):**

```python
STRATEGY_SOURCE_CONTOUR    = "source_contour"
STRATEGY_SOURCE_PLUS_SHIFT = "source_plus_shift"
STRATEGY_TARGET_MEDIAN     = "target_median"
STRATEGY_MANUAL            = "manual"
VALID_F0_STRATEGIES        # frozenset of all four names
```

**Pure strategy functions** (numpy → numpy, no side effects):

| Function | Description |
|----------|-------------|
| `apply_source_contour(f0)` | Keep source F0 contour unchanged. |
| `apply_source_plus_shift(f0, src_median_log, tgt_median_log)` | Global pitch shift to align median F0 to target speaker. |
| `apply_target_median(f0, tgt_median_log)` | Replace all voiced frames with target speaker's median F0. |
| `apply_manual(f0, semitone_offset)` | Shift contour by a fixed number of semitones. |

**Utility functions:**

| Function | Description |
|----------|-------------|
| `estimate_speaker_median_log_f0(f0)` | Compute median log-Hz F0 over voiced frames. |
| `align_to_content_framerate(f0_100hz, target_frames)` | Nearest-neighbour resample 100 Hz → content frame count. |

**Key class:** `F0Encoder`

| Member | Description |
|--------|-------------|
| `F0Encoder(model_path, device, is_half)` | Wraps RMVPE. Raises `FileNotFoundError` if checkpoint missing. |
| `.extract(audio_16k, thred)` | `1D numpy (16kHz) → 1D numpy at 100 Hz in Hz`. Unvoiced = 0. |

> **Note:** Never pass bare `"cuda"` as device — RMVPE forces it to `"cuda:0"` internally.
> Always pass `"cuda:0"` (or `"cuda:N"`) explicitly.

**Example:**

```python
from indextts.vc.f0_encoder import F0Encoder, apply_source_plus_shift, estimate_speaker_median_log_f0

f0_enc = F0Encoder(model_path="checkpoints/rmvpe/model.pt", device="cuda:0")
f0 = f0_enc.extract(audio_16k_np)   # 1D numpy at 100 Hz
src_log = estimate_speaker_median_log_f0(f0)
tgt_log = estimate_speaker_median_log_f0(ref_f0)
f0_shifted = apply_source_plus_shift(f0, src_log, tgt_log)
```

---

### `kmeans_quantizer.py` — K-means Content Quantizer

Hard nearest-neighbour quantization that maps HuBERT-soft features to cluster center
vectors (design D3).  This is a second pass of speaker-information removal on top of
HuBERT-soft's soft disentanglement.

**Key class:** `KMeansQuantizer`

| Member | Description |
|--------|-------------|
| `KMeansQuantizer(n_clusters, dim, random_state)` | Default: 200 clusters, dim=256. |
| `.train(features_np)` | Fit `MiniBatchKMeans`. Input: `(N, 256)` or `(B, T, 256)`. |
| `.save(path)` | Save codebook to `.pt` file. |
| `.load(path)` | Load codebook from `.pt` file. |
| `.quantize_to_vector(features)` | `(B, T, 256) → (B, T, 256)` cluster center vectors. |
| `.quantize_hard(features)` | `(B, T, 256) → (B, T)` integer cluster indices. |
| `.bypass(features)` | Pass-through for debug/ablation (config `kmeans.enabled: false`). |

> **Important:** `quantize_to_vector` output dim is `256` (the feature dim), **not**
> `n_clusters`.  Each frame is replaced by the nearest cluster center vector.

**Example:**

```python
from indextts.vc.kmeans_quantizer import KMeansQuantizer

q = KMeansQuantizer(n_clusters=200, dim=256)
q.load("checkpoints/vc/kmeans_200.pt")
quantized = q.quantize_to_vector(features)  # (B, T, 256)
```

---

### `preprocessing.py` — Offline Feature Extraction

Command-line tool and library functions for extracting and saving training features
for each utterance.  For each audio file, four `.npy` files are produced:

| File | Shape | Description |
|------|-------|-------------|
| `<stem>.content.npy` | `(T_content, 256)` | HuBERT-soft features at ~50 Hz |
| `<stem>.f0.npy` | `(T_content,)` | F0 in Hz at 50 Hz (decimated from 100 Hz) |
| `<stem>.mel.npy` | `(80, T_mel)` | Mel spectrogram at 22050 Hz / 80 bands / hop=256 |
| `<stem>.style.npy` | `(192,)` | CAMPPlus speaker embedding |

**Mel spec constants (s2mel spec — must not be changed):**

```python
MEL_SAMPLE_RATE = 22050   # NOT 24000
MEL_N_MELS      = 80      # NOT 100
MEL_HOP_LENGTH  = 256
```

**Key functions:**

| Function | Description |
|----------|-------------|
| `preprocess_utterance(audio_16k, audio_22k, output_stem, ...)` | Process one utterance; returns `True` if saved, `False` if filtered. |
| `compute_mel(audio_22k)` | Compute 22050/80/256 mel via `s2mel.modules.audio`. |
| `compute_voiced_ratio(f0)` | Fraction of voiced frames (f0 > 0). |
| `should_filter_utterance(f0, min_voiced_ratio)` | Returns `True` if utterance should be skipped (voiced ratio < 0.1). |

**CLI usage:**

```bash
python -m indextts.vc.preprocessing \
    --manifest data/vc/aishell3/manifest_train.jsonl \
    --output_dir data/vc/aishell3/preprocessed \
    --content_model_name bshall/hubert-soft \
    --f0_model_path checkpoints/rmvpe/model.pt \
    --device cuda:0
```
