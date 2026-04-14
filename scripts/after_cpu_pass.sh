#!/bin/bash
# Wait for all CPU workers to finish, then run kmeans + manifest + training
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/index-tts

echo "[$(date)] Waiting for CPU workers to finish..."
while [ $(ps aux | grep cpu_worker | grep -v grep | wc -l) -gt 0 ]; do
    F0_COUNT=$(ls data/vc/aishell3/preprocessed/*.f0.npy 2>/dev/null | wc -l)
    echo "[$(date)] Workers running, f0 files: $F0_COUNT / ~62000"
    sleep 120
done

echo "[$(date)] All workers done!"
F0_COUNT=$(ls data/vc/aishell3/preprocessed/*.f0.npy 2>/dev/null | wc -l)
echo "[$(date)] Total f0 files: $F0_COUNT"

# Build manifest + kmeans
echo "[$(date)] Building manifest + kmeans..."
python -u -c "
import glob, os, json, numpy as np, sys
sys.path.insert(0, '.')

prep = 'data/vc/aishell3/preprocessed'
output_dir = 'data/vc/aishell3'

# Find complete utterances (all 4 files exist)
content_files = sorted(glob.glob(os.path.join(prep, '*.content.npy')))
manifest = []
for cf in content_files:
    stem = cf.replace('.content.npy', '')
    base = os.path.basename(stem)
    if not (os.path.exists(stem + '.f0.npy') and os.path.exists(stem + '.mel.npy')):
        continue
    # Find speaker from wav path
    for d in glob.glob('/root/autodl-tmp/train/wav/*/'):
        if os.path.exists(os.path.join(d, base + '.wav')):
            spk = os.path.basename(d.rstrip('/'))
            import librosa
            dur = librosa.get_duration(filename=os.path.join(d, base + '.wav'))
            manifest.append({
                'audio_path': os.path.join(d, base + '.wav'),
                'speaker_id': spk,
                'language': 'zh',
                'duration_s': round(dur, 2),
                'text': base,
            })
            break

mpath = os.path.join(output_dir, 'manifest.jsonl')
with open(mpath, 'w') as f:
    for m in manifest:
        f.write(json.dumps(m, ensure_ascii=False) + '\n')
n_spk = len(set(m['speaker_id'] for m in manifest))
print(f'Manifest: {len(manifest)} utterances, {n_spk} speakers -> {mpath}')

# K-means
from indextts.vc.kmeans_quantizer import KMeansQuantizer
frames = []
for cf in content_files[:5000]:  # Sample from first 5000
    frames.append(np.load(cf))
    if sum(len(x) for x in frames) > 500000:
        break
frames = np.concatenate(frames)
if len(frames) > 500000:
    idx = np.random.choice(len(frames), 500000, replace=False)
    frames = frames[idx]
print(f'K-means on {len(frames)} frames')
kq = KMeansQuantizer(n_clusters=200)
kq.train(frames)
kq.save(os.path.join(output_dir, 'kmeans_codebook.pt'))
print('K-means done')
"

echo "[$(date)] Starting training..."

# TensorBoard
pkill -f tensorboard 2>/dev/null || true
nohup tensorboard --logdir runs --port 6006 --bind_all > /tmp/tb.log 2>&1 &

# Training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/full_pipeline.py \
    --train-only \
    --steps 50000 \
    --batch-size 2 \
    --lr 2e-4 \
    --kmeans-clusters 200 \
    --aishell3-dir /root/autodl-tmp/train/wav \
    --output-dir data/vc/aishell3 \
    --warmup-steps 2000 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --max-duration 8.0

echo "[$(date)] Training complete!"
