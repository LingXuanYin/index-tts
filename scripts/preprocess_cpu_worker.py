"""Single-process CPU worker for F0 + mel preprocessing.

Usage: python preprocess_cpu_worker.py --wav-list files.txt --output-dir ... --worker-id 0
"""
import os, sys, time, argparse
import numpy as np
import torch
import torchaudio
import librosa

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav-list", required=True, help="Text file with one wav path per line")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-duration", type=float, default=8.0)
    p.add_argument("--worker-id", type=int, default=0)
    args = p.parse_args()

    prep_dir = os.path.join(args.output_dir, "preprocessed")
    with open(args.wav_list) as f:
        wav_files = [l.strip() for l in f if l.strip()]

    # Use the SAME mel function as s2mel training (log mel)
    sys.path.insert(0, ".")
    from indextts.s2mel.modules.audio import mel_spectrogram
    mel_fn = lambda x: mel_spectrogram(
        x, n_fft=1024, num_mels=80, sampling_rate=22050,
        hop_size=256, win_size=1024, fmin=0, fmax=None
    )

    done = 0
    skipped = 0
    t0 = time.time()

    for i, wav_path in enumerate(wav_files):
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        fb = os.path.join(prep_dir, stem)

        # Skip if already done
        if os.path.exists(fb + ".f0.npy") and os.path.exists(fb + ".mel.npy"):
            skipped += 1
            continue

        # Skip if no content (GPU pass didn't process this file)
        if not os.path.exists(fb + ".content.npy"):
            skipped += 1
            continue

        try:
            audio_16k, _ = librosa.load(wav_path, sr=16000)
            dur = len(audio_16k) / 16000.0
            if dur < 1.0 or dur > args.max_duration:
                skipped += 1
                continue

            # Load content to get target frame count
            content = np.load(fb + ".content.npy")
            target_len = content.shape[0]

            # F0
            f0, _, _ = librosa.pyin(audio_16k, fmin=65, fmax=2000, sr=16000,
                                     hop_length=16000 // 50)
            f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
            if len(f0) > target_len:
                f0 = f0[:target_len]
            elif len(f0) < target_len:
                f0 = np.pad(f0, (0, target_len - len(f0)))

            if np.mean(f0 > 0) < 0.1:
                skipped += 1
                continue

            np.save(fb + ".f0.npy", f0)

            # Mel — MUST use same log-mel as s2mel training
            audio_22k = librosa.resample(audio_16k, orig_sr=16000, target_sr=22050)
            audio_22k_t = torch.from_numpy(audio_22k).unsqueeze(0)
            mel = mel_fn(audio_22k_t).squeeze(0).numpy()
            np.save(fb + ".mel.npy", mel.astype(np.float32))

            # Style (mock)
            np.save(fb + ".style.npy", np.random.randn(192).astype(np.float32))

            done += 1
        except Exception as e:
            skipped += 1

        if (done + skipped) % 200 == 0 and done > 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = len(wav_files) - i - 1
            eta = remaining / max(rate, 0.01)
            print(f"[W{args.worker_id}] {i+1}/{len(wav_files)} done={done} skip={skipped} "
                  f"{rate:.1f}/s ETA {eta/60:.0f}min", flush=True)

    elapsed = time.time() - t0
    print(f"[W{args.worker_id}] FINISHED: {done} done, {skipped} skipped, {elapsed/60:.1f}min", flush=True)

if __name__ == "__main__":
    main()
