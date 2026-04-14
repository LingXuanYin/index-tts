#!/bin/bash
# Kill old stuck preprocessor, split work, launch 8 independent workers
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/index-tts

pkill -9 -f preprocess_parallel 2>/dev/null || true
sleep 2

# Split wav files into 8 chunks
python -u -c "
import glob, os
prep = 'data/vc/aishell3/preprocessed'
content_files = glob.glob(os.path.join(prep, '*.content.npy'))
need = []
for cf in sorted(content_files):
    stem = cf.replace('.content.npy', '')
    # Regenerate if EITHER f0 or mel is missing
    if not os.path.exists(stem + '.f0.npy') or not os.path.exists(stem + '.mel.npy'):
        base = os.path.basename(stem)
        for d in glob.glob('/root/autodl-tmp/train/wav/*/'):
            wp = os.path.join(d, base + '.wav')
            if os.path.exists(wp):
                need.append(wp)
                break
print(f'Need F0+mel for {len(need)} files')
n = 8
chunk_size = (len(need) + n - 1) // n
for i in range(n):
    chunk = need[i*chunk_size:(i+1)*chunk_size]
    with open(f'/tmp/wav_chunk_{i}.txt', 'w') as f:
        for w in chunk:
            f.write(w + '\n')
    print(f'  Chunk {i}: {len(chunk)} files')
"

# Launch 8 workers
for i in $(seq 0 7); do
    nohup python -u scripts/preprocess_cpu_worker.py \
        --wav-list /tmp/wav_chunk_$i.txt \
        --output-dir data/vc/aishell3 \
        --max-duration 8.0 \
        --worker-id $i \
        > /tmp/cpu_worker_$i.log 2>&1 &
done
echo "Launched 8 CPU workers"
sleep 3
ps aux | grep cpu_worker | grep -v grep | wc -l
echo "workers running"
