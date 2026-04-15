"""Parallel preprocessing: GPU pass (HuBERT-soft) + CPU multiprocess (F0+mel).

Skips files that already have all outputs. Safe to restart.
"""
import os, sys, glob, json, time, argparse
import numpy as np
import torch
import torchaudio
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, ".")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aishell3-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-duration", type=float, default=8.0)
    p.add_argument("--cpu-workers", type=int, default=8)
    p.add_argument("--kmeans-clusters", type=int, default=200)
    return p.parse_args()

def is_complete(feature_base):
    """Check if all 4 feature files exist."""
    return all(os.path.exists(feature_base + ext)
               for ext in [".content.npy", ".f0.npy", ".mel.npy", ".style.npy"])

def cpu_process_one(wav_path, feature_base, max_duration):
    """CPU-bound: F0 (pyin) + mel + style. Runs in worker process."""
    try:
        audio_16k, _ = librosa.load(wav_path, sr=16000)
        duration = len(audio_16k) / 16000.0
        if duration < 1.0 or duration > max_duration:
            return None

        # F0 via librosa.pyin (~50 Hz)
        f0, _, _ = librosa.pyin(audio_16k, fmin=65, fmax=2000, sr=16000,
                                 hop_length=16000 // 50)
        f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)

        # Voiced ratio check
        if np.mean(f0 > 0) < 0.1:
            return None

        np.save(feature_base + ".f0.npy", f0)

        # Mel (22050/80/256)
        audio_22k = librosa.resample(audio_16k, orig_sr=16000, target_sr=22050)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80)
        mel = mel_transform(torch.from_numpy(audio_22k).unsqueeze(0)).squeeze(0).numpy()
        np.save(feature_base + ".mel.npy", mel.astype(np.float32))

        # Style (mock)
        np.save(feature_base + ".style.npy", np.random.randn(192).astype(np.float32))

        return feature_base
    except Exception as e:
        return None

def main():
    args = parse_args()
    prep_dir = os.path.join(args.output_dir, "preprocessed")
    os.makedirs(prep_dir, exist_ok=True)

    # Scan wav files
    print(f"Scanning {args.aishell3_dir}...")
    wav_files = sorted(glob.glob(os.path.join(args.aishell3_dir, "**", "*.wav"), recursive=True))
    print(f"Found {len(wav_files)} wav files")

    # Build work items
    items = []
    for wav_path in wav_files:
        speaker_id = os.path.basename(os.path.dirname(wav_path))
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        feature_base = os.path.join(prep_dir, stem)
        items.append((wav_path, speaker_id, stem, feature_base))

    # ========== PASS 1: GPU (HuBERT-soft content) ==========
    need_content = [(w, s, st, fb) for w, s, st, fb in items
                    if not os.path.exists(fb + ".content.npy")]
    print(f"\n=== PASS 1: HuBERT-soft (GPU) === {len(need_content)} files to process")

    if need_content:
        print("Loading HuBERT-soft...")
        hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
        hubert = hubert.cuda().eval()
        hubert.requires_grad_(False)

        t0 = time.time()
        for i, (wav_path, _, _, feature_base) in enumerate(need_content):
            try:
                audio_16k, _ = librosa.load(wav_path, sr=16000)
                duration = len(audio_16k) / 16000.0
                if duration < 1.0 or duration > args.max_duration:
                    continue
                audio_t = torch.from_numpy(audio_16k).unsqueeze(0).unsqueeze(0).cuda()
                with torch.no_grad():
                    content = hubert.units(audio_t).cpu().numpy().squeeze(0)
                np.save(feature_base + ".content.npy", content.astype(np.float32))
            except Exception:
                pass

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(need_content) - i - 1) / rate
                print(f"  [{i+1}/{len(need_content)}] {rate:.0f} files/s, ETA {eta/60:.0f}min")

        del hubert
        torch.cuda.empty_cache()
        print(f"  GPU pass done in {(time.time()-t0)/60:.1f}min")
    else:
        print("  All content features exist, skipping.")

    # ========== PASS 2: CPU multiprocess (F0 + mel + style) ==========
    need_cpu = [(w, s, st, fb) for w, s, st, fb in items
                if os.path.exists(fb + ".content.npy")
                and not (os.path.exists(fb + ".f0.npy")
                         and os.path.exists(fb + ".mel.npy"))]
    print(f"\n=== PASS 2: F0+mel (CPU x{args.cpu_workers}) === {len(need_cpu)} files to process")

    if need_cpu:
        t0 = time.time()
        done = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=args.cpu_workers) as pool:
            futures = {
                pool.submit(cpu_process_one, w, fb, args.max_duration): (w, s, st, fb)
                for w, s, st, fb in need_cpu
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    done += 1
                else:
                    failed += 1
                total = done + failed
                if total % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = total / elapsed
                    eta = (len(need_cpu) - total) / max(rate, 0.01)
                    print(f"  [{total}/{len(need_cpu)}] done={done} failed={failed} "
                          f"{rate:.0f} files/s ETA {eta/60:.0f}min")

        print(f"  CPU pass done in {(time.time()-t0)/60:.1f}min (done={done}, failed={failed})")
    else:
        print("  All F0+mel features exist, skipping.")

    # ========== BUILD MANIFEST ==========
    print("\n=== Building manifest ===")
    manifest = []
    for wav_path, speaker_id, stem, feature_base in items:
        if is_complete(feature_base):
            try:
                audio_16k, _ = librosa.load(wav_path, sr=16000, duration=0.1)
                duration = librosa.get_duration(filename=wav_path)
            except Exception:
                continue
            manifest.append({
                "audio_path": wav_path,
                "speaker_id": speaker_id,
                "language": "zh",
                "duration_s": round(duration, 2),
                "text": stem,
            })

    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    n_spk = len(set(m["speaker_id"] for m in manifest))
    print(f"Manifest: {len(manifest)} utterances, {n_spk} speakers → {manifest_path}")

    # ========== K-MEANS ==========
    print(f"\n=== K-means (n_clusters={args.kmeans_clusters}) ===")
    from indextts.vc.kmeans_quantizer import KMeansQuantizer
    content_files = sorted(glob.glob(os.path.join(prep_dir, "*.content.npy")))
    frames = []
    for cf in content_files:
        c = np.load(cf)
        frames.append(c)
        if sum(len(x) for x in frames) > 500000:
            break
    frames = np.concatenate(frames, axis=0)
    if len(frames) > 500000:
        idx = np.random.choice(len(frames), 500000, replace=False)
        frames = frames[idx]
    print(f"  Training on {len(frames)} frames")
    kq = KMeansQuantizer(n_clusters=args.kmeans_clusters)
    kq.train(frames)
    codebook_path = os.path.join(args.output_dir, "kmeans_codebook.pt")
    kq.save(codebook_path)
    print(f"  Saved: {codebook_path}")
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()
