import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from indextts.fusion import coerce_fusion_recipe
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


def recommendation_candidates(report_rows: List[Dict[str, Any]], *, experimental: bool) -> List[Dict[str, Any]]:
    candidates = [
        row
        for row in report_rows
        if row["is_multiref"]
        and row["experimental_case"] == experimental
        and row["pass_rate"] >= 1.0
    ]
    if candidates:
        return candidates
    return [
        row
        for row in report_rows
        if row["is_multiref"] and row["experimental_case"] == experimental
    ]


def emotion_vector(
    tts: IndexTTS2,
    speaker_bundle: Dict[str, torch.Tensor],
    emo_cond_emb: torch.Tensor,
    emo_alpha: float,
) -> torch.Tensor:
    emo_cond_lengths = torch.tensor([emo_cond_emb.shape[-1]], device=emo_cond_emb.device)
    return tts.gpt.merge_emovec(
        speaker_bundle["spk_cond_emb"],
        emo_cond_emb,
        speaker_bundle["cond_length"],
        emo_cond_lengths,
        alpha=emo_alpha,
    ).squeeze(0).detach().cpu()


def score_case(
    tts: IndexTTS2,
    row: Dict[str, Any],
    speaker_bundle_cache: Dict[str, Dict[str, torch.Tensor]],
    style_cache: Dict[str, Any],
    emotion_cond_cache: Dict[str, Any],
) -> Dict[str, Any]:
    _, torchaudio = sfe._require_torch_modules()
    output_path = row["output_path"]
    speaker_a_path = row["speaker_a"]["audio_path"]
    speaker_b_path = row["speaker_b"]["audio_path"]
    recipe = coerce_fusion_recipe(row["fusion_recipe"])
    metadata = sfe.load_metadata(row.get("metadata_output_path"))
    emo_alpha = float(metadata.get("emotion_alpha", 1.0))

    def speaker_bundle(path: str) -> Dict[str, torch.Tensor]:
        if path not in speaker_bundle_cache:
            speaker_bundle_cache[path] = tts._get_reference_conditioning(
                path,
                verbose=False,
                cache_key=f"emotion-score::speaker::{path}",
            )
        return speaker_bundle_cache[path]

    def style(path: str) -> Any:
        if path not in style_cache:
            style_cache[path] = sfe.style_embedding(tts, path)
        return style_cache[path]

    def emotion_cond(path: str) -> torch.Tensor:
        if path not in emotion_cond_cache:
            emotion_cond_cache[path] = tts._get_emotion_conditioning(
                path,
                verbose=False,
                cache_key=f"emotion-score::emo::{path}",
            )
        return emotion_cond_cache[path]

    speaker_a_bundle = speaker_bundle(speaker_a_path)
    target_emo_cond, _ = tts._resolve_emotion_conditioning(
        row["emo_audio_prompt"],
        recipe,
        verbose=False,
    )
    generated_emo_cond = emotion_cond(output_path)
    emo_a_cond = emotion_cond(speaker_a_path)
    emo_b_cond = emotion_cond(speaker_b_path)

    target_vec = emotion_vector(tts, speaker_a_bundle, target_emo_cond, emo_alpha)
    generated_vec = emotion_vector(tts, speaker_a_bundle, generated_emo_cond, emo_alpha)
    emo_a_vec = emotion_vector(tts, speaker_a_bundle, emo_a_cond, emo_alpha)
    emo_b_vec = emotion_vector(tts, speaker_a_bundle, emo_b_cond, emo_alpha)

    waveform, sample_rate = torchaudio.load(output_path)
    duration_seconds = waveform.shape[-1] / sample_rate
    silence_ratio = float((waveform.abs() < 1e-4).float().mean().item())
    style_keep = sfe.cosine_similarity(style(output_path), style(speaker_a_path))
    sim_a = sfe.cosine_similarity(generated_vec, emo_a_vec)
    sim_b = sfe.cosine_similarity(generated_vec, emo_b_vec)

    recipe_name = row["recipe_name"]
    if recipe_name == "emotion_a_only":
        sanity_score = 1.0 if sim_a >= sim_b else 0.0
    elif recipe_name == "emotion_b_only":
        sanity_score = 1.0 if sim_b >= sim_a else 0.0
    elif recipe_name == "emotion_tensor_anchor_a":
        sanity_score = 1.0 if sim_a >= sim_b else 0.0
    else:
        sanity_score = 1.0 - abs(sim_a - sim_b)

    return {
        "case_id": row["case_id"],
        "scheme_id": row["scheme_id"],
        "emo_target_cosine": sfe.cosine_similarity(generated_vec, target_vec),
        "emotion_similarity_a": sim_a,
        "emotion_similarity_b": sim_b,
        "emotion_balance": 1.0 - abs(sim_a - sim_b),
        "sanity_score": sanity_score,
        "speaker_keep_style": style_keep,
        "generation_health": sfe.generation_health(duration_seconds, silence_ratio, metadata),
        "duration_seconds": duration_seconds,
        "silence_ratio": silence_ratio,
        "total_time_seconds": metadata.get("timings", {}).get("total_time"),
        "rtf": metadata.get("timings", {}).get("rtf"),
    }


def aggregate_scores(rows: List[Dict[str, Any]], attempts_by_scheme: Counter[str], status_counts_by_scheme: Dict[str, Counter[str]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["scheme_id"]].append(row)

    report_rows: List[Dict[str, Any]] = []
    for scheme_id, values in grouped.items():
        first = values[0]
        attempts = attempts_by_scheme[scheme_id]
        status_counts = status_counts_by_scheme[scheme_id]
        success_count = status_counts["done"] + status_counts["skipped"]
        error_count = status_counts["error"]
        missing_count = attempts - success_count - error_count
        pass_rate = success_count / attempts if attempts else 0.0
        base_score = (
            0.6 * sum(v["emo_target_cosine"] for v in values) / len(values)
            + 0.2 * sum(v["speaker_keep_style"] for v in values) / len(values)
            + 0.2 * sum(v["generation_health"] for v in values) / len(values)
        )
        sanity_score = sum(v["sanity_score"] for v in values) / len(values)
        rank_score = base_score * pass_rate * (0.5 + 0.5 * sanity_score)
        report_rows.append(
            {
                "scheme_id": scheme_id,
                "recipe_name": first["recipe_name"],
                "profile": first["profile"],
                "anchor_mode": first["anchor_mode"],
                "experimental_levels": ",".join(first["experimental_levels"]),
                "count": len(values),
                "attempt_count": attempts,
                "success_count": success_count,
                "error_count": error_count,
                "missing_count": missing_count,
                "pass_rate": pass_rate,
                "experimental_case": bool(first["experimental_levels"]),
                "is_multiref": "tensor" in first["recipe_name"] or "waveform" in first["recipe_name"],
                "emo_target_cosine_mean": sum(v["emo_target_cosine"] for v in values) / len(values),
                "emotion_similarity_a_mean": sum(v["emotion_similarity_a"] for v in values) / len(values),
                "emotion_similarity_b_mean": sum(v["emotion_similarity_b"] for v in values) / len(values),
                "emotion_balance_mean": sum(v["emotion_balance"] for v in values) / len(values),
                "sanity_score_mean": sanity_score,
                "speaker_keep_style_mean": sum(v["speaker_keep_style"] for v in values) / len(values),
                "generation_health_mean": sum(v["generation_health"] for v in values) / len(values),
                "base_score": base_score,
                "rank_score": rank_score,
            }
        )

    report_rows.sort(
        key=lambda row: (row["pass_rate"] >= 1.0, row["rank_score"], row["base_score"]),
        reverse=True,
    )
    for rank, row in enumerate(report_rows, start=1):
        row["rank"] = rank
    return report_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score emotion-only fusion experiments")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-path")
    parser.add_argument("--cfg-path", default="checkpoints/config.yaml")
    parser.add_argument("--model-dir", default="checkpoints")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    results_path = Path(args.results_path) if args.results_path else manifest_path.with_name("run_results_stable.jsonl")
    manifest_rows = sfe.read_manifest(manifest_path)
    results_by_case = read_results(results_path)

    tts = IndexTTS2(
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )
    attempts_by_scheme: Counter[str] = Counter(row["scheme_id"] for row in manifest_rows)
    status_counts_by_scheme: Dict[str, Counter[str]] = defaultdict(Counter)
    speaker_bundle_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    style_cache: Dict[str, Any] = {}
    emotion_cond_cache: Dict[str, Any] = {}
    scored_rows: List[Dict[str, Any]] = []

    for row in manifest_rows:
        result = results_by_case.get(row["case_id"])
        status = result["status"] if result is not None else "missing"
        status_counts_by_scheme[row["scheme_id"]][status] += 1
        if not os.path.exists(row["output_path"]):
            continue
        metrics = score_case(tts, row, speaker_bundle_cache, style_cache, emotion_cond_cache)
        scored = dict(row)
        scored.update(metrics)
        scored["run_status"] = status
        scored_rows.append(scored)

    scores_path = manifest_path.with_name("scores_emotion.jsonl")
    sfe.write_jsonl(scores_path, scored_rows)

    report_rows = aggregate_scores(scored_rows, attempts_by_scheme, status_counts_by_scheme)
    seen_scheme_ids = {row["scheme_id"] for row in report_rows}
    for row in manifest_rows:
        if row["scheme_id"] in seen_scheme_ids:
            continue
        status_counts = status_counts_by_scheme[row["scheme_id"]]
        success_count = status_counts["done"] + status_counts["skipped"]
        error_count = status_counts["error"]
        missing_count = attempts_by_scheme[row["scheme_id"]] - success_count - error_count
        report_rows.append(
            {
                "scheme_id": row["scheme_id"],
                "recipe_name": row["recipe_name"],
                "profile": row["profile"],
                "anchor_mode": row["anchor_mode"],
                "experimental_levels": ",".join(row["experimental_levels"]),
                "count": 0,
                "attempt_count": attempts_by_scheme[row["scheme_id"]],
                "success_count": success_count,
                "error_count": error_count,
                "missing_count": missing_count,
                "pass_rate": success_count / attempts_by_scheme[row["scheme_id"]] if attempts_by_scheme[row["scheme_id"]] else 0.0,
                "experimental_case": bool(row["experimental_levels"]),
                "is_multiref": "tensor" in row["recipe_name"] or "waveform" in row["recipe_name"],
                "emo_target_cosine_mean": None,
                "emotion_similarity_a_mean": None,
                "emotion_similarity_b_mean": None,
                "emotion_balance_mean": None,
                "sanity_score_mean": None,
                "speaker_keep_style_mean": None,
                "generation_health_mean": None,
                "base_score": 0.0,
                "rank_score": 0.0,
            }
        )
        seen_scheme_ids.add(row["scheme_id"])
    report_rows.sort(
        key=lambda row: (row["pass_rate"] >= 1.0, row["rank_score"], row["base_score"]),
        reverse=True,
    )
    for rank, row in enumerate(report_rows, start=1):
        row["rank"] = rank
    report_path = manifest_path.with_name("ranked_report_emotion.csv")
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(report_rows[0].keys()) if report_rows else ["rank"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    supported_rows = recommendation_candidates(report_rows, experimental=False)
    experimental_rows = recommendation_candidates(report_rows, experimental=True)
    recommendations = {
        "default_multiref": supported_rows[0]["scheme_id"] if supported_rows else None,
        "fallback_multiref": supported_rows[1]["scheme_id"] if len(supported_rows) > 1 else None,
        "top_experimental": experimental_rows[0]["scheme_id"] if experimental_rows else None,
    }
    summary = {
        "manifest_path": str(manifest_path),
        "results_path": str(results_path),
        "scores_path": str(scores_path),
        "report_path": str(report_path),
        "recommendations": recommendations,
    }
    summary_path = manifest_path.with_name("ranked_summary_emotion.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
