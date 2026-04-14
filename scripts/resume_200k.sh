#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/index-tts

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
    --max-duration 8.0 \
    --resume checkpoints/vc/trained_vc_step50000.pth
