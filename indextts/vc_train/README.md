# VC Training Data Guide

This document describes how to download and prepare datasets for VC (Voice Conversion) finetuning.

## Directory Convention

```
data/
  raw/
    aishell3/                # AISHELL-3 raw wav files
      wav/
        SSB0001/
          SSB0001_001.wav
          ...
        SSB0002/
          ...
    libritts_clean_100/      # LibriTTS clean-100 raw wav files
      train-clean-100/
        84/
          121123/
            84_121123_000001_000000.wav
            ...
  vc/
    aishell3/                # Preprocessed AISHELL-3 features + manifests
      train.jsonl
      val.jsonl
      test.jsonl
      preprocessed/
        SSB0001/
          SSB0001_001.content.npy
          SSB0001_001.f0.npy
          SSB0001_001.mel.npy
          SSB0001_001.style.npy
          ...
    libritts_clean_100/      # Preprocessed LibriTTS features + manifests
      train.jsonl
      val.jsonl
      test.jsonl
      preprocessed/
        ...
    smoke/                   # Mini smoke dataset (5 synthetic utterances for testing)
      manifest.jsonl
      kmeans_codebook.pt
      preprocessed/
        ...
```

## Dataset Download

### AISHELL-3

- **Homepage**: http://www.aishelltech.com/aishell_3
- **License**: CC BY-NC-ND 4.0 (non-commercial research use)
- **Download**: Register and download from the official page or via:
  ```bash
  # Official mirror (requires registration)
  wget https://www.openslr.org/resources/93/data_aishell3.tgz
  tar -xzf data_aishell3.tgz -C data/raw/aishell3/
  ```
- **Expected layout after extraction**: `data/raw/aishell3/wav/<speaker_id>/<utterance>.wav`
- **Size**: ~85,000 utterances, ~218 speakers, ~85 hours

### LibriTTS clean-100

- **Homepage**: https://www.openslr.org/60/
- **License**: CC BY 4.0
- **Download**:
  ```bash
  wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
  tar -xzf train-clean-100.tar.gz -C data/raw/libritts_clean_100/
  ```
- **Expected layout after extraction**: `data/raw/libritts_clean_100/train-clean-100/<speaker_id>/<chapter_id>/<utterance>.wav`
- **Size**: ~53,000 utterances, ~245 speakers, ~100 hours

### Optional: WenetSpeech4TTS Premium Subset

- **Homepage**: https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS
- **License**: CC BY 4.0
- **Download**: Requires HuggingFace account with dataset access approval
  ```bash
  huggingface-cli download Wenetspeech4TTS/WenetSpeech4TTS --repo-type dataset --local-dir data/raw/wenetspeech4tts
  ```
- Filter for `Premium` subset; layout: `data/raw/wenetspeech4tts/premium/<speaker_id>/<utterance>.wav`

## Preprocessing Steps

### 1. Generate Manifest

```bash
# AISHELL-3
python -m indextts.vc_train.manifest \
    --root_dir data/raw/aishell3/wav \
    --language zh \
    --output data/vc/aishell3/manifest.jsonl \
    --split 0.90 0.05 0.05

# LibriTTS clean-100
python -m indextts.vc_train.manifest \
    --root_dir data/raw/libritts_clean_100/train-clean-100 \
    --language en \
    --output data/vc/libritts_clean_100/manifest.jsonl \
    --split 0.90 0.05 0.05
```

This generates `manifest_train.jsonl`, `manifest_val.jsonl`, `manifest_test.jsonl` with speaker-level splits.

### 2. Run Preprocessing

Requires: HuBERT-soft checkpoint, RMVPE model, CAMPPlus model.

```bash
# AISHELL-3
python -m indextts.vc.preprocessing \
    --manifest data/vc/aishell3/manifest_train.jsonl \
    --output_dir data/vc/aishell3/preprocessed \
    --content_model_name bshall/hubert-soft \
    --f0_model_path checkpoints/rmvpe/model.pt \
    --device cuda \
    --num_workers 4

# LibriTTS
python -m indextts.vc.preprocessing \
    --manifest data/vc/libritts_clean_100/manifest_train.jsonl \
    --output_dir data/vc/libritts_clean_100/preprocessed \
    --content_model_name bshall/hubert-soft \
    --f0_model_path checkpoints/rmvpe/model.pt \
    --device cuda \
    --num_workers 4
```

Each utterance produces four files: `.content.npy` (T,256), `.f0.npy` (T,), `.mel.npy` (80,T_mel), `.style.npy` (192,).

### 3. Train K-means Codebook

```bash
python -m indextts.vc.kmeans_quantizer \
    --manifest data/vc/aishell3/manifest_train.jsonl \
    --feature_dir data/vc/aishell3/preprocessed \
    --n_clusters 200 \
    --output checkpoints/vc/kmeans_200.pt
```

### 4. Start Training

```bash
accelerate launch indextts/vc_train/train.py \
    --config indextts/vc_train/config_vc.yaml
```

Or single-GPU without accelerate:

```bash
python indextts/vc_train/train.py \
    --config indextts/vc_train/config_vc.yaml \
    --single_phase
```

## Model Checkpoint Requirements

| Model | Path | Source |
|-------|------|--------|
| IndexTTS S2Mel | `checkpoints/s2mel.pth` | IndexTTS release |
| HuBERT-soft | auto-downloaded via `torch.hub` | https://github.com/bshall/hubert |
| RMVPE | `checkpoints/rmvpe/model.pt` | https://github.com/yxlllc/RMVPE |
| CAMPPlus | auto-downloaded via `modelscope` | ModelScope |

## Notes

- `data/` directory is excluded from git (see root `.gitignore`)
- `checkpoints/vc/` is excluded from git except `.yaml` files
- Preprocessed features (`.npy`) are large binary files; do not commit them
- For smoke testing without real data, use `data/vc/smoke/` (auto-generated by `tools/generate_smoke_data.py`)
