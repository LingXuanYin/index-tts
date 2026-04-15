#!/bin/bash
# Local training on PG199 (GPU 0 only, 32GB)
# Data at L:\DATASET\vc_training\
set -e

cd "J:/index-tts"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="L:/DATASET/vc_training"
AISHELL3_WAV="$DATA_DIR/train/wav"
OUTPUT_DIR="$DATA_DIR/preprocessed_aishell3"
CAMPPLUS_PATH="checkpoints/campplus/campplus_cn_common.bin"

echo "=== Local VC Training Pipeline ==="
echo "GPU: PG199 32GB (CUDA_VISIBLE_DEVICES=0)"
echo "Data: $DATA_DIR"
echo ""

# Step 1: Extract AISHELL-3 if not already done
if [ ! -d "$AISHELL3_WAV" ]; then
    echo "[Step 1] Extracting AISHELL-3..."
    cd "$DATA_DIR"
    tar xzf data_aishell3.tgz
    cd "J:/index-tts"
    echo "  Done. Speakers: $(ls $AISHELL3_WAV | wc -l)"
else
    echo "[Step 1] AISHELL-3 already extracted. Speakers: $(ls $AISHELL3_WAV | wc -l)"
fi

# Step 2-5: Run fix_and_train.py (handles style regen, HP search, training, inference)
.venv/Scripts/python.exe -B -u scripts/fix_and_train.py \
    --aishell3-dir "$AISHELL3_WAV" \
    --output-dir "$OUTPUT_DIR" \
    --campplus-path "$CAMPPLUS_PATH" \
    2>&1 | tee "$DATA_DIR/training.log"
