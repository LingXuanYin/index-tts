import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indextts.infer_v2 import IndexTTS2
from tools import speaker_fusion_experiment as sfe


def read_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["case_id"]] = row
    return rows


def recommendation_candidates(report_rows: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
    candidates = [
        row
        for row in report_rows
        if row["case_kind"] == "fusion"
        and row["pass_rate"] >= 1.0
        and bool(row[field])
    ]
    if candidates:
        return candidates
    return [
        row
        for row in report_rows
        if row["case_kind"] == "fusion" and bool(row[field])
    ]


def logical_scheme_signature(row: Dict[str, Any]) -> str:
    payload = {
        "case_kind": row["case_kind"],
        "profile": row["profile"],
        "recipe_name": row["recipe_name"],
        "experimental_name": row["experimental_name"],
        "supported_levels": row["supported_levels"],
        "experimental_levels": row["experimental_levels"],
        "anchor_mode": row["anchor_mode"],
        "role_assignments": row.get("role_assignments", {}),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def empty_report_row(example: Dict[str, Any], logical_scheme_id: str) -> Dict[str, Any]:
    return {
        "scheme_id": logical_scheme_id,
        "case_kind": example["case_kind"],
        "profile": example["profile"],
        "anchor_mode": example["anchor_mode"],
        "recipe_name": example["recipe_name"],
        "experimental_name": example["experimental_name"],
        "supported_levels": ",".join(example["supported_levels"]),
        "experimental_levels": ",".join(example["experimental_levels"]),
        "count": 0,
        "speaker_similarity_a_mean": None,
        "speaker_similarity_b_mean": None,
        "blend_balance_mean": None,
        "fused_target_similarity_mean": None,
        "semantic_prompt_similarity_mean": None,
        "semantic_stability_mean": None,
        "duration_seconds_mean": None,
        "silence_ratio_mean": None,
        "generation_health_mean": None,
        "total_time_seconds_mean": None,
        "rtf_mean": None,
        "overall_score": 0.0,
        "experimental_case": bool(example["experimental_levels"]),
    }


def lightweight_style_embedding(tts: IndexTTS2, audio_path: str) -> Any:
    _, torchaudio = sfe._require_torch_modules()
    audio_16k, _ = tts._load_and_cut_audio(audio_path, 15, False, sr=16000)
    feat = torchaudio.compliance.kaldi.fbank(
        audio_16k.to(tts.device),
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    return tts.campplus_model(feat.unsqueeze(0)).squeeze(0).detach().cpu()


def configure_tts_for_scoring(tts: IndexTTS2) -> None:
    if hasattr(tts, "gpt"):
        tts.gpt = tts.gpt.to("cpu")
    if hasattr(tts, "semantic_codec"):
        tts.semantic_codec = tts.semantic_codec.to("cpu")
    if hasattr(tts, "bigvgan"):
        tts.bigvgan = tts.bigvgan.to("cpu")
    if hasattr(tts, "emo_matrix"):
        tts.emo_matrix = tuple(chunk.to("cpu") for chunk in tts.emo_matrix)
    if hasattr(tts, "spk_matrix"):
        tts.spk_matrix = tuple(chunk.to("cpu") for chunk in tts.spk_matrix)
    if hasattr(tts, "s2mel"):
        tts.s2mel = tts.s2mel.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score timbre experiments with reliability penalties")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-path")
    parser.add_argument("--cfg-path", default="checkpoints/config.yaml")
    parser.add_argument("--model-dir", default="checkpoints")
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default=None)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()
    use_fp16 = False
    if args.use_fp16:
        use_fp16 = True
    if args.no_fp16:
        use_fp16 = False

    manifest_path = Path(args.manifest)
    results_path = Path(args.results_path) if args.results_path else manifest_path.with_name("run_results_stable.jsonl")
    rows = sfe.read_manifest(manifest_path)
    results_by_case = read_results(results_path)

    tts = IndexTTS2(
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        use_fp16=use_fp16,
        device=args.device,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )
    configure_tts_for_scoring(tts)
    sfe.style_embedding = lightweight_style_embedding
    style_cache: Dict[str, Any] = {}
    semantic_cache: Dict[str, Any] = {}
    batch_baselines: Dict[str, Dict[str, Optional[str]]] = {}
    logical_scheme_by_case: Dict[str, str] = {}
    logical_scheme_examples: Dict[str, Dict[str, Any]] = {}
    attempts_by_scheme: Counter[str] = Counter()
    status_counts_by_scheme: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        logical_scheme_id = logical_scheme_signature(row)
        logical_scheme_by_case[row["case_id"]] = logical_scheme_id
        logical_scheme_examples.setdefault(logical_scheme_id, row)
        attempts_by_scheme[logical_scheme_id] += 1
        result = results_by_case.get(row["case_id"])
        status = result["status"] if result is not None else "missing"
        status_counts_by_scheme[logical_scheme_id][status] += 1
        if row["case_kind"] not in {"baseline_a", "baseline_b"}:
            continue
        batch_baselines.setdefault(row["batch_id"], {"baseline_a": None, "baseline_b": None})
        if os.path.exists(row["output_path"]):
            if row["case_kind"] == "baseline_a":
                batch_baselines[row["batch_id"]]["baseline_a"] = row["output_path"]
            elif row["case_kind"] == "baseline_b":
                batch_baselines[row["batch_id"]]["baseline_b"] = row["output_path"]

    scored_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not os.path.exists(row["output_path"]):
            continue
        logical_row = dict(row)
        logical_row["source_scheme_id"] = row["scheme_id"]
        logical_row["scheme_id"] = logical_scheme_by_case[row["case_id"]]
        metrics = sfe.score_case(tts, logical_row, style_cache, semantic_cache, batch_baselines)
        scored = dict(row)
        scored["source_scheme_id"] = row["scheme_id"]
        scored["scheme_id"] = logical_row["scheme_id"]
        scored.update(metrics)
        scored["run_status"] = results_by_case.get(row["case_id"], {}).get("status", "unknown")
        scored_rows.append(scored)

    scores_path = manifest_path.with_name("scores_stable.jsonl")
    sfe.write_jsonl(scores_path, scored_rows)

    report_rows = sfe.aggregate_scores(scored_rows)
    by_scheme = {row["scheme_id"]: row for row in report_rows}
    merged_rows: List[Dict[str, Any]] = []
    for scheme_id, example in logical_scheme_examples.items():
        report = by_scheme.get(scheme_id, empty_report_row(example, scheme_id))
        attempts = attempts_by_scheme[scheme_id]
        status_counts = status_counts_by_scheme[scheme_id]
        success_count = status_counts["done"] + status_counts["skipped"]
        error_count = status_counts["error"]
        missing_count = attempts - success_count - error_count
        pass_rate = success_count / attempts if attempts else 0.0
        merged = dict(report)
        merged["attempt_count"] = attempts
        merged["success_count"] = success_count
        merged["error_count"] = error_count
        merged["missing_count"] = missing_count
        merged["pass_rate"] = pass_rate
        merged["reliability_adjusted_score"] = merged["overall_score"] * pass_rate
        merged_rows.append(merged)

    merged_rows.sort(
        key=lambda row: (
            row["pass_rate"] >= 1.0,
            row["reliability_adjusted_score"],
            row["overall_score"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(merged_rows, start=1):
        row["rank"] = rank

    supported = recommendation_candidates(
        [row for row in merged_rows if not row["experimental_case"]],
        "supported_levels",
    )
    experimental = [row for row in merged_rows if row["experimental_case"]]
    recommendations = {
        "default_recommendation": supported[0]["scheme_id"] if supported else None,
        "fallback_recommendation": supported[1]["scheme_id"] if len(supported) > 1 else None,
        "top_experimental": experimental[0]["scheme_id"] if experimental else None,
    }

    report_path = manifest_path.with_name("ranked_report_stable.csv")
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(merged_rows[0].keys()) if merged_rows else ["rank"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    summary = {
        "manifest_path": str(manifest_path),
        "results_path": str(results_path),
        "scores_path": str(scores_path),
        "report_path": str(report_path),
        "recommendations": recommendations,
    }
    summary_path = manifest_path.with_name("ranked_summary_stable.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    listening_path = sfe.export_listening_review(scored_rows, recommendations, manifest_path.parent)
    print(json.dumps(
        {
            "scores_path": str(scores_path),
            "report_path": str(report_path),
            "summary_path": str(summary_path),
            "listening_path": str(listening_path),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
