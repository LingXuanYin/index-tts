"""Local parallel preprocessing: GPU pass (HuBERT + CAMPPlus) + CPU pass (F0 + mel).

128 CPU cores available — use 32 workers for F0+mel.
Skips files that already have all outputs. Safe to restart.
"""
import os, sys, glob, json, time, argparse, re
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
    p.add_argument("--campplus-path", default="checkpoints/campplus/campplus_cn_common.bin")
    p.add_argument("--max-duration", type=float, default=6.0)
    p.add_argument("--cpu-workers", type=int, default=32)
    p.add_argument("--kmeans-clusters", type=int, default=200)
    return p.parse_args()

def cpu_worker(wav_path, feature_base, max_duration):
    """CPU: F0 (pyin) + mel (log). Called in worker process."""
    try:
        audio_16k, _ = librosa.load(wav_path, sr=16000)
        dur = len(audio_16k) / 16000.0
        if dur < 1.0 or dur > max_duration:
            return None

        content = np.load(feature_base + ".content.npy")
        target_len = content.shape[0]

        # F0
        f0, _, _ = librosa.pyin(audio_16k, fmin=65, fmax=2000, sr=16000, hop_length=16000 // 50)
        f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
        if len(f0) > target_len:
            f0 = f0[:target_len]
        elif len(f0) < target_len:
            f0 = np.pad(f0, (0, target_len - len(f0)))
        if np.mean(f0 > 0) < 0.1:
            return None
        np.save(feature_base + ".f0.npy", f0)

        # Mel (log mel, same as s2mel)
        from indextts.s2mel.modules.audio import mel_spectrogram
        audio_22k = librosa.resample(audio_16k, orig_sr=16000, target_sr=22050)
        mel = mel_spectrogram(torch.from_numpy(audio_22k).unsqueeze(0),
                              n_fft=1024, num_mels=80, sampling_rate=22050,
                              hop_size=256, win_size=1024, fmin=0, fmax=None)
        np.save(feature_base + ".mel.npy", mel.squeeze(0).numpy().astype(np.float32))

        return feature_base
    except Exception:
        return None

def main():
    args = parse_args()
    prep = os.path.join(args.output_dir, "preprocessed")
    os.makedirs(prep, exist_ok=True)

    wav_files = sorted(glob.glob(os.path.join(args.aishell3_dir, "**", "*.wav"), recursive=True))
    if not wav_files:
        raise RuntimeError(f"No wav files in {args.aishell3_dir}")
    print(f"Found {len(wav_files)} wav files")

    items = []
    for wp in wav_files:
        spk = os.path.basename(os.path.dirname(wp))
        stem = os.path.splitext(os.path.basename(wp))[0]
        items.append((wp, spk, stem, os.path.join(prep, stem)))

    # ========== PASS 1: GPU (HuBERT-soft + CAMPPlus) ==========
    need_gpu = [(w, s, st, fb) for w, s, st, fb in items
                if not os.path.exists(fb + ".content.npy") or not os.path.exists(fb + ".style.npy")]
    print(f"\n=== GPU PASS: HuBERT-soft + CAMPPlus === {len(need_gpu)} files")

    if need_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Loading HuBERT-soft...")
        hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
        hubert = hubert.cuda().eval()
        hubert.requires_grad_(False)

        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus.load_state_dict(torch.load(args.campplus_path, map_location="cpu", weights_only=True))
        campplus = campplus.cuda().eval()
        print("Models loaded.")

        t0 = time.time()
        for i, (wp, spk, stem, fb) in enumerate(need_gpu):
            try:
                audio_16k, _ = librosa.load(wp, sr=16000)
                dur = len(audio_16k) / 16000.0
                if dur < 1.0 or dur > args.max_duration:
                    continue

                audio_t = torch.from_numpy(audio_16k)

                # HuBERT-soft
                if not os.path.exists(fb + ".content.npy"):
                    with torch.no_grad():
                        content = hubert.units(audio_t.unsqueeze(0).unsqueeze(0).cuda())
                    np.save(fb + ".content.npy", content.cpu().numpy().squeeze(0).astype(np.float32))

                # CAMPPlus (kaldi fbank)
                if not os.path.exists(fb + ".style.npy"):
                    feat = torchaudio.compliance.kaldi.fbank(
                        audio_t.unsqueeze(0).cuda(), num_mel_bins=80, dither=0, sample_frequency=16000)
                    feat = feat - feat.mean(dim=0, keepdim=True)
                    with torch.no_grad():
                        style = campplus(feat.unsqueeze(0))
                    np.save(fb + ".style.npy", style.cpu().numpy().squeeze(0).astype(np.float32))

            except Exception:
                pass

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(need_gpu)}] {rate:.0f}/s ETA {(len(need_gpu)-i-1)/rate/60:.0f}min", flush=True)

        del hubert, campplus
        torch.cuda.empty_cache()
        print(f"  GPU pass done in {(time.time()-t0)/60:.1f}min")

    # ========== PASS 2: CPU parallel (F0 + mel) ==========
    need_cpu = [(w, s, st, fb) for w, s, st, fb in items
                if os.path.exists(fb + ".content.npy")
                and (not os.path.exists(fb + ".f0.npy") or not os.path.exists(fb + ".mel.npy"))]
    print(f"\n=== CPU PASS (x{args.cpu_workers} workers): F0 + mel === {len(need_cpu)} files")

    if need_cpu:
        t0 = time.time()
        done = 0
        failed = 0
        with ProcessPoolExecutor(max_workers=args.cpu_workers) as pool:
            futures = {pool.submit(cpu_worker, w, fb, args.max_duration): fb
                       for w, s, st, fb in need_cpu}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    done += 1
                else:
                    failed += 1
                total = done + failed
                if total % 2000 == 0:
                    elapsed = time.time() - t0
                    rate = total / elapsed
                    print(f"  [{total}/{len(need_cpu)}] done={done} fail={failed} "
                          f"{rate:.0f}/s ETA {(len(need_cpu)-total)/rate/60:.0f}min", flush=True)
        print(f"  CPU pass: {done} done, {failed} failed, {(time.time()-t0)/60:.1f}min")

    # ========== MANIFEST ==========
    print("\n=== Building manifest ===")
    manifest = []
    for wp, spk, stem, fb in items:
        if all(os.path.exists(fb + e) for e in [".content.npy", ".f0.npy", ".mel.npy", ".style.npy"]):
            try:
                dur = librosa.get_duration(filename=wp)
                if 1.0 <= dur <= args.max_duration:
                    manifest.append({"audio_path": wp, "speaker_id": spk,
                                     "language": "zh", "duration_s": round(dur, 2), "text": stem})
            except Exception:
                pass
    mpath = os.path.join(args.output_dir, "manifest.jsonl")
    with open(mpath, "w") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    n_spk = len(set(m["speaker_id"] for m in manifest))
    print(f"Manifest: {len(manifest)} utterances, {n_spk} speakers → {mpath}")

    # ========== K-MEANS ==========
    print(f"\n=== K-means ({args.kmeans_clusters} clusters) ===")
    from indextts.vc.kmeans_quantizer import KMeansQuantizer
    content_files = sorted(glob.glob(os.path.join(prep, "*.content.npy")))
    frames = []
    for cf in content_files:
        frames.append(np.load(cf))
        if sum(len(x) for x in frames) > 500000:
            break
    frames = np.concatenate(frames)
    if len(frames) > 500000:
        frames = frames[np.random.choice(len(frames), 500000, replace=False)]
    kq = KMeansQuantizer(n_clusters=args.kmeans_clusters)
    kq.train(frames)
    kq.save(os.path.join(args.output_dir, "kmeans_codebook.pt"))
    print(f"  Saved ({len(frames)} frames)")

    # ========== MINI SUBSET for HP search ==========
    print("\n=== Mini subset for HP search ===")
    mini_dir = os.path.join(args.output_dir, "mini_real")
    mini_prep = os.path.join(mini_dir, "preprocessed")
    os.makedirs(mini_prep, exist_ok=True)
    spk_seen = {}
    mini_entries = []
    for m in manifest:
        sid = m["speaker_id"]
        spk_seen.setdefault(sid, 0)
        if spk_seen[sid] >= 10 or (len(set(spk_seen.keys())) > 5 and sid not in spk_seen):
            continue
        spk_seen[sid] += 1
        stem = os.path.splitext(os.path.basename(m["audio_path"]))[0]
        for ext in [".content.npy", ".f0.npy", ".mel.npy", ".style.npy"]:
            src = os.path.join(prep, stem + ext)
            dst = os.path.join(mini_prep, stem + ext)
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
        mini_entries.append(m)
    with open(os.path.join(mini_dir, "manifest.jsonl"), "w") as f:
        for m in mini_entries:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    mini_kq = KMeansQuantizer(n_clusters=20)
    mini_frames = []
    for m in mini_entries[:30]:
        stem = os.path.splitext(os.path.basename(m["audio_path"]))[0]
        cf = os.path.join(mini_prep, stem + ".content.npy")
        if os.path.exists(cf):
            mini_frames.append(np.load(cf))
    if mini_frames:
        mini_kq.train(np.concatenate(mini_frames))
        mini_kq.save(os.path.join(mini_dir, "kmeans_codebook.pt"))
    print(f"  Mini: {len(mini_entries)} entries")
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()
