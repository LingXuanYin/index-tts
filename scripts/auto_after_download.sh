#!/bin/bash
# Auto-start pipeline after AISHELL-3 download completes.
# Run: nohup bash scripts/auto_after_download.sh > /tmp/auto_pipeline.log 2>&1 &

set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp

TGZ="data_aishell3.tgz"
EXPECTED_SIZE=26000000000  # ~26GB, approximate

echo "[$(date)] Waiting for $TGZ download to finish..."

# Poll until wget exits
while pgrep -f "wget.*aishell" > /dev/null 2>&1; do
    SIZE=$(stat -c%s "$TGZ" 2>/dev/null || echo 0)
    SIZE_MB=$((SIZE / 1024 / 1024))
    echo "[$(date)] Downloading... ${SIZE_MB}MB / ~26000MB"
    sleep 60
done

SIZE=$(stat -c%s "$TGZ" 2>/dev/null || echo 0)
SIZE_MB=$((SIZE / 1024 / 1024))
echo "[$(date)] Download finished. File size: ${SIZE_MB}MB"

if [ "$SIZE" -lt 20000000000 ]; then
    echo "[$(date)] ERROR: File too small (${SIZE_MB}MB < 20000MB), download may have failed!"
    exit 1
fi

# Extract
echo "[$(date)] Extracting..."
cd /root/autodl-tmp
tar xzf data_aishell3.tgz
echo "[$(date)] Extraction done."
N_SPEAKERS=$(ls data_aishell3/train/wav/ | wc -l)
echo "[$(date)] Speakers: $N_SPEAKERS"
df -h /root/autodl-tmp/

# Delete tgz to save space
echo "[$(date)] Deleting tgz to save space..."
rm -f data_aishell3.tgz
df -h /root/autodl-tmp/

# Start TensorBoard
cd /root/autodl-tmp/index-tts
pkill -f tensorboard 2>/dev/null || true
nohup tensorboard --logdir runs --port 6006 --bind_all > /tmp/tb.log 2>&1 &
echo "[$(date)] TensorBoard started on 6006"

# Run full pipeline (preprocess + train)
echo "[$(date)] Starting full pipeline..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/full_pipeline.py \
    --steps 50000 \
    --batch-size 2 \
    --lr 2e-4 \
    --kmeans-clusters 200 \
    --aishell3-dir /root/autodl-tmp/data_aishell3/train/wav \
    --output-dir /root/autodl-tmp/index-tts/data/vc/aishell3 \
    --warmup-steps 2000 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --max-duration 8.0

echo "[$(date)] Pipeline complete!"
