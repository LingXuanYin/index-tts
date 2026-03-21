import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IndexTTS2 Command Line")
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument("-v", "--voice", type=str, default=None, help="Primary timbre reference audio")
    parser.add_argument(
        "--voice-ref",
        action="append",
        default=[],
        help="Additional timbre reference audio. Repeat the flag to add more files.",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default=None,
        help="Primary emotion reference audio. Defaults to the primary timbre reference.",
    )
    parser.add_argument(
        "--emotion-ref",
        action="append",
        default=[],
        help="Additional emotion reference audio. Repeat the flag to add more files.",
    )
    parser.add_argument(
        "--speaker-fusion-mode",
        choices=("default", "fallback"),
        default="default",
        help="Supported timbre fusion preset.",
    )
    parser.add_argument(
        "--emotion-fusion-mode",
        choices=("default", "fallback"),
        default="default",
        help="Supported emotion fusion preset.",
    )
    parser.add_argument("--emo-alpha", type=float, default=1.0, help="Emotion reference strength")
    parser.add_argument("-o", "--output_path", type=str, default="gen.wav", help="Output wav path")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="checkpoints/config.yaml",
        help="Config file path",
    )
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model directory")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 when supported")
    parser.add_argument("-f", "--force", action="store_true", default=False, help="Overwrite existing output")
    parser.add_argument("-d", "--device", type=str, default=None, help="Runtime device")
    parser.add_argument("--cuda-kernel", action="store_true", default=False, help="Enable BigVGAN CUDA kernel")
    parser.add_argument("--deepspeed", action="store_true", default=False, help="Enable DeepSpeed when available")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print verbose inference logs",
    )
    return parser


def _resolve_primary_and_extra(primary: str | None, extras: list[str]) -> tuple[str | None, list[str]]:
    refs = [path for path in extras if path]
    if primary is None and refs:
        return refs[0], refs[1:]
    return primary, refs


def _validate_audio_paths(paths: list[str], parser: argparse.ArgumentParser, label: str) -> None:
    for path in paths:
        if not os.path.exists(path):
            print(f"Audio prompt file {path} does not exist for {label}.")
            parser.print_help()
            raise SystemExit(1)


def _resolve_device(args: argparse.Namespace) -> None:
    if args.device is not None:
        if args.device == "cpu":
            args.fp16 = False
        return

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Please install it first.")
        raise SystemExit(1)

    if torch.cuda.is_available():
        args.device = "cuda:0"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        args.device = "xpu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        args.device = "mps"
    else:
        args.device = "cpu"
        args.fp16 = False
        print("WARNING: Running on CPU may be slow.")


def run_from_args(args: argparse.Namespace) -> str:
    parser = build_parser()
    text = args.text.strip()
    if not text:
        print("ERROR: Text is empty.")
        parser.print_help()
        raise SystemExit(1)

    voice, voice_refs = _resolve_primary_and_extra(args.voice, list(args.voice_ref))
    emotion, emotion_refs = _resolve_primary_and_extra(args.emotion, list(args.emotion_ref))
    if voice is None:
        print("ERROR: A primary timbre reference is required.")
        parser.print_help()
        raise SystemExit(1)
    if not os.path.exists(args.config):
        print(f"Config file {args.config} does not exist.")
        parser.print_help()
        raise SystemExit(1)

    _validate_audio_paths([voice] + voice_refs, parser, "voice")
    _validate_audio_paths(([emotion] if emotion else []) + emotion_refs, parser, "emotion")

    output_path = args.output_path
    if os.path.exists(output_path):
        if not args.force:
            print(f"ERROR: Output file {output_path} already exists. Use --force to overwrite.")
            parser.print_help()
            raise SystemExit(1)
        os.remove(output_path)

    _resolve_device(args)

    from indextts.infer_v2 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_fp16=args.fp16,
        device=args.device,
        use_cuda_kernel=args.cuda_kernel,
        use_deepspeed=args.deepspeed,
    )
    return tts.infer(
        spk_audio_prompt=voice,
        text=text,
        output_path=output_path,
        emo_audio_prompt=emotion,
        emo_alpha=args.emo_alpha,
        speaker_references=voice_refs,
        speaker_fusion_mode=args.speaker_fusion_mode,
        emotion_references=emotion_refs,
        emotion_fusion_mode=args.emotion_fusion_mode,
        verbose=args.verbose,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
