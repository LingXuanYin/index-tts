#!/bin/bash
# Regenerate all mel files with correct log-mel function
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/index-tts

echo "Finding all wav files that need mel regeneration..."
# Simply list all wav files (content already exists for them)
find /root/autodl-tmp/train/wav/ -name "*.wav" > /tmp/all_wavs.txt
TOTAL=$(wc -l < /tmp/all_wavs.txt)
echo "Total wav files: $TOTAL"

# Split into 8 chunks
split -n l/8 /tmp/all_wavs.txt /tmp/wav_chunk_
# Rename to numbered chunks
i=0
for f in /tmp/wav_chunk_*; do
    mv "$f" "/tmp/wav_chunk_${i}.txt"
    LINES=$(wc -l < "/tmp/wav_chunk_${i}.txt")
    echo "  Chunk $i: $LINES files"
    i=$((i+1))
done

# Launch workers (they skip files that already have mel)
for i in $(seq 0 7); do
    nohup python -u scripts/preprocess_cpu_worker.py \
        --wav-list /tmp/wav_chunk_$i.txt \
        --output-dir data/vc/aishell3 \
        --max-duration 8.0 \
        --worker-id $i \
        > /tmp/cpu_worker_$i.log 2>&1 &
done
echo "Launched 8 CPU workers"
sleep 5
ps aux | grep cpu_worker | grep -v grep | wc -l
echo "workers running"
