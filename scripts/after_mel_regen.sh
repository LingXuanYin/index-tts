#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/index-tts

echo "[$(date)] Waiting for mel regen workers..."
while [ $(ps aux | grep cpu_worker | grep -v grep | wc -l) -gt 0 ]; do
    MEL=$(ls data/vc/aishell3/preprocessed/*.mel.npy 2>/dev/null | wc -l)
    echo "[$(date)] Workers running, mel files: $MEL / ~63000"
    sleep 120
done

echo "[$(date)] Mel regen done!"
MEL=$(ls data/vc/aishell3/preprocessed/*.mel.npy 2>/dev/null | wc -l)
echo "[$(date)] Total mel files: $MEL"

# Rebuild manifest (some files may have been filtered)
echo "[$(date)] Rebuilding manifest..."
python -u scripts/count_features.py

# Start training from scratch (new mel = need fresh training)
echo "[$(date)] Starting training (200k steps, bf16, lr=2e-4)..."
pkill -f tensorboard 2>/dev/null
nohup tensorboard --logdir runs --port 6006 --bind_all > /tmp/tb.log 2>&1 &

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/full_pipeline.py \
    --train-only \
    --steps 200000 \
    --batch-size 2 \
    --lr 2e-4 \
    --kmeans-clusters 200 \
    --aishell3-dir /root/autodl-tmp/train/wav \
    --output-dir data/vc/aishell3 \
    --warmup-steps 2000 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --max-duration 8.0

echo "[$(date)] Training complete!"
