#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/index-tts
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# TensorBoard
pkill -f tensorboard 2>/dev/null
nohup tensorboard --logdir runs --port 6006 --bind_all > /tmp/tb.log 2>&1 &

python -B -u scripts/train_v2_fresh.py \
    --train-only \
    --steps 300000 \
    --batch-size 10 \
    --lr 2e-4 \
    --grad-clip 10.0 \
    --ema-decay 0.9999 \
    --kmeans-clusters 200 \
    --aishell3-dir /root/autodl-tmp/train/wav \
    --output-dir data/vc/aishell3 \
    --warmup-steps 2000 \
    --save-interval 20000 \
    --eval-interval 20000 \
    --max-duration 6.0 \
    --resume checkpoints/vc/v2_step20000.pth
