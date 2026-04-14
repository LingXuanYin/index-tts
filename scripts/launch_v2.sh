#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/index-tts

pkill -f tensorboard 2>/dev/null
nohup tensorboard --logdir runs --port 6006 --bind_all > /tmp/tb.log 2>&1 &

python -u scripts/full_pipeline_v2.py \
    --train-only \
    --steps 200000 \
    --batch-size 2 \
    --lr 2e-4 \
    --grad-clip 10.0 \
    --ema-decay 0.9999 \
    --kmeans-clusters 200 \
    --aishell3-dir /root/autodl-tmp/train/wav \
    --output-dir data/vc/aishell3 \
    --warmup-steps 2000 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --max-duration 8.0 \
    --resume checkpoints/vc/trained_vc_step20000.pth
