import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from indextts.infer_v2 import IndexTTS2


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def release_tts(tts: Any) -> None:
    if tts is not None:
        try:
            tts.reference_conditioning_cache.clear()
            tts.emotion_conditioning_cache.clear()
        except Exception:
            pass
        del tts
    gc.collect()
    torch.cuda.empty_cache()


def build_tts(cfg_path: str, model_dir: str) -> IndexTTS2:
    return IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run speaker fusion manifests with periodic cache clearing/reloads")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cfg-path", default="checkpoints/config.yaml")
    parser.add_argument("--model-dir", default="checkpoints")
    parser.add_argument("--results-path")
    parser.add_argument("--reload-every", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)
    results_path = Path(args.results_path) if args.results_path else manifest_path.with_name("run_results_stable.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    processed_since_reload = 0
    tts = None
    with results_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows, start=1):
            output_path = row["output_path"]
            metadata_output_path = row["metadata_output_path"]
            if not args.force and os.path.exists(output_path) and os.path.exists(metadata_output_path):
                result = {
                    "case_id": row["case_id"],
                    "scheme_id": row["scheme_id"],
                    "status": "skipped",
                    "output_path": output_path,
                    "metadata_output_path": metadata_output_path,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                continue

            if tts is None or processed_since_reload >= args.reload_every:
                release_tts(tts)
                print(f"RELOAD model before case {idx}: {row['case_id']}", flush=True)
                tts = build_tts(args.cfg_path, args.model_dir)
                processed_since_reload = 0

            print(f"[{idx}/{len(rows)}] {row['case_id']}", flush=True)
            try:
                output = tts.infer(
                    spk_audio_prompt=row["spk_audio_prompt"],
                    text=row["text"],
                    output_path=output_path,
                    emo_audio_prompt=row["emo_audio_prompt"],
                    fusion_recipe=row["fusion_recipe"],
                    metadata_output_path=metadata_output_path,
                    return_metadata=True,
                )
                result = {
                    "case_id": row["case_id"],
                    "scheme_id": row["scheme_id"],
                    "status": "done",
                    "output_path": output_path,
                    "metadata_output_path": metadata_output_path,
                    "inference_output": output,
                }
            except Exception as exc:
                result = {
                    "case_id": row["case_id"],
                    "scheme_id": row["scheme_id"],
                    "status": "error",
                    "output_path": output_path,
                    "metadata_output_path": metadata_output_path,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                if not args.continue_on_error:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    release_tts(tts)
                    raise

            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            processed_since_reload += 1
            tts.reference_conditioning_cache.clear()
            tts.emotion_conditioning_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()

    release_tts(tts)
    print(results_path)


if __name__ == "__main__":
    main()
