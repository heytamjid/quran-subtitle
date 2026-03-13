#!/usr/bin/env python3
"""
QuranSub — Automatic Quran Recitation Video Subtitle Generator

Usage:
    python run.py full input_video.mp4                  # full pipeline
    python run.py full input_video.mp4 --surah 76       # with surah hint
    python run.py extract-audio input_video.mp4         # just extract audio
    python run.py transcribe audio.mp3                  # just transcribe
    python run.py match transcription.json              # just match
    python run.py subtitle matched_segments.json        # just generate subs
    python run.py render input.mp4 --subtitle subs.ass  # just burn subs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml


def _load_config(config_path: str) -> dict:
    """Load YAML config, falling back to defaults if file not found."""
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["_project_root"] = str(p.parent.resolve())
        return cfg
    logging.warning("Config file not found: %s — using defaults", config_path)
    return {
        "_project_root": str(Path.cwd()),
        "transcription": {"model_id": "tarteel-ai/whisper-base-ar-quran",
                          "base_model_id": "openai/whisper-base",
                          "chunk_length_s": 30, "stride_length_s": 5,
                          "language": "ar"},
        "matching": {"surah_hint": None},
        "subtitle": {},
        "video": {},
        "paths": {"quran_csv": "data/ben_quran_with_tafsir.csv",
                  "output_dir": "output"},
    }


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_full(args):
    from src.pipeline import run_pipeline
    cfg = _load_config(args.config)
    if args.surah is not None:
        cfg["matching"]["surah_hint"] = args.surah
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir
    run_pipeline(args.video, cfg)


def cmd_extract_audio(args):
    from src.audio import extract_audio
    out = args.output or str(Path(args.video).with_suffix(".mp3"))
    extract_audio(args.video, out)
    print(f"Audio saved: {out}")


def cmd_transcribe(args):
    from src.transcribe import transcribe_audio, save_transcription
    cfg = _load_config(args.config)
    tc = cfg["transcription"]
    result = transcribe_audio(
        args.audio,
        model_id=tc.get("model_id", "tarteel-ai/whisper-base-ar-quran"),
        base_model_id=tc.get("base_model_id", "openai/whisper-base"),
        chunk_length_s=tc.get("chunk_length_s", 30),
        stride_length_s=tc.get("stride_length_s", 5),
        language=tc.get("language", "ar"),
    )
    out = args.output or "output/transcription.json"
    save_transcription(result, out)
    print(f"Transcription saved: {out}")


def cmd_match(args):
    from src.transcribe import load_transcription
    from src.quran_data import load_quran_data
    from src.match import align_words_to_ayahs, save_segments

    cfg = _load_config(args.config)
    csv_path = Path(cfg["paths"]["quran_csv"])
    if not csv_path.is_absolute():
        csv_path = Path(cfg["_project_root"]) / csv_path

    transcription = load_transcription(args.transcription)
    quran_ayahs = load_quran_data(str(csv_path))

    surah_hint = args.surah if args.surah is not None else cfg["matching"].get("surah_hint")
    segments = align_words_to_ayahs(transcription["words"], quran_ayahs, surah_hint)

    out = args.output or "output/matched_segments.json"
    save_segments(segments, out)
    print(f"Matched {len(segments)} segments -> {out}")


def cmd_subtitle(args):
    from src.match import load_segments
    from src.subtitle import generate_srt, generate_ass

    cfg = _load_config(args.config)
    segments = load_segments(args.segments)

    out_dir = Path(args.output_dir or "output")
    generate_srt(segments, str(out_dir / "subtitles.srt"))
    generate_ass(segments, cfg.get("subtitle", {}), str(out_dir / "subtitles.ass"))
    print(f"Subtitles saved in: {out_dir}")


def cmd_render(args):
    from src.render import render_video

    cfg = _load_config(args.config)
    vc = cfg.get("video", {})
    out = args.output or "output/final_video_with_subtitles.mp4"
    render_video(
        args.video,
        args.subtitle,
        out,
        codec=vc.get("codec", "libx264"),
        crf=vc.get("crf", 18),
        preset=vc.get("preset", "medium"),
    )
    print(f"Rendered: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    _setup_logging()

    parser = argparse.ArgumentParser(
        prog="QuranSub",
        description="Automatic Quran Recitation Video Subtitle Generator",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- full --
    p = sub.add_parser("full", help="Run the full pipeline")
    p.add_argument("video", help="Input video file")
    p.add_argument("--surah", type=int, default=None, help="Surah number hint")
    p.add_argument("--output-dir", default=None, help="Output directory")
    p.set_defaults(func=cmd_full)

    # -- extract-audio --
    p = sub.add_parser("extract-audio", help="Extract audio from video")
    p.add_argument("video", help="Input video file")
    p.add_argument("-o", "--output", default=None, help="Output audio path")
    p.set_defaults(func=cmd_extract_audio)

    # -- transcribe --
    p = sub.add_parser("transcribe", help="Transcribe audio to words+timestamps")
    p.add_argument("audio", help="Input audio file (mp3/wav)")
    p.add_argument("-o", "--output", default=None, help="Output JSON path")
    p.set_defaults(func=cmd_transcribe)

    # -- match --
    p = sub.add_parser("match", help="Match transcription to Quran ayahs")
    p.add_argument("transcription", help="Transcription JSON file")
    p.add_argument("--surah", type=int, default=None, help="Surah number hint")
    p.add_argument("-o", "--output", default=None, help="Output segments JSON path")
    p.set_defaults(func=cmd_match)

    # -- subtitle --
    p = sub.add_parser("subtitle", help="Generate SRT/ASS from matched segments")
    p.add_argument("segments", help="Matched segments JSON file")
    p.add_argument("-o", "--output-dir", default=None, help="Output directory")
    p.set_defaults(func=cmd_subtitle)

    # -- render --
    p = sub.add_parser("render", help="Burn subtitles into video")
    p.add_argument("video", help="Input video file")
    p.add_argument("--subtitle", required=True, help="ASS subtitle file")
    p.add_argument("-o", "--output", default=None, help="Output video path")
    p.set_defaults(func=cmd_render)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
