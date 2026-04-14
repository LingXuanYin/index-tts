#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/index-tts

# TensorBoard
pkill -f tensorboard 2>/dev/null
nohup tensorboard --logdir runs --port 6006 --bind_all > /tmp/tb.log 2>&1 &

# Pipeline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/full_pipeline.py \
    --steps 50000 \
    --batch-size 2 \
    --lr 2e-4 \
    --kmeans-clusters 200 \
    --aishell3-dir /root/autodl-tmp/train/wav \
    --output-dir /root/autodl-tmp/index-tts/data/vc/aishell3 \
    --warmup-steps 2000 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --max-duration 8.0
