#!/usr/bin/env python3
"""
QuranSub — Automatic Quran Recitation Video Subtitle Generator

Usage:
    python run.py full taha.mp4 --surah 20              # full pipeline (reads from input/)
    python run.py full taha.mp4 --surah 20              # output -> output/taha/
    python run.py extract-audio input/video.mp4         # just extract audio
    python run.py transcribe audio.mp3                  # just transcribe
    python run.py match transcription.json              # just match
    python run.py subtitle matched_segments.json        # just generate subs
    python run.py render input/video.mp4 --subtitle x   # just burn subs
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
        "word_by_word": {"enabled": True},
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
    if args.mode:
        cfg.setdefault("word_by_word", {})["enabled"] = args.mode == "words"
    if args.recovery is not None:
        cfg.setdefault("word_by_word", {}).setdefault("recovery", {})["enabled"] = args.recovery
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
        fonts_dir=vc.get("fonts_dir"),
    )
    print(f"Rendered: {out}")


def cmd_word_match(args):
    from src.word_pipeline import match_words
    cfg = _load_config(args.config)
    if args.recovery is not None:
        cfg.setdefault("word_by_word", {}).setdefault("recovery", {})["enabled"] = args.recovery
    if args.surah is not None:
        cfg["matching"]["surah_hint"] = args.surah
    transcription = json.loads(Path(args.transcription).read_text(encoding="utf-8"))
    cues = match_words(transcription, cfg, args.output_dir, args.audio)
    print(f"Matched {len(cues)} words -> {args.output_dir}/word_cues.json")


def cmd_word_subtitle(args):
    from src.word_pipeline import word_subtitles
    cfg = _load_config(args.config)
    cues = json.loads(Path(args.cues).read_text(encoding="utf-8"))
    if args.size:
        width, height = args.size.lower().split("x")
        size = (int(width), int(height))
    else:
        size = None
    result = word_subtitles(cues, cfg, args.output_dir, args.video, size)
    print(f"Word overlay saved: {result}")


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
    p.add_argument("video", help="Video filename (looked up in input/ directory)")
    p.add_argument("--surah", type=int, default=None, help="Surah number hint")
    p.add_argument("--output-dir", default=None, help="Output directory")
    p.add_argument("--mode", choices=("words", "ayah"), help="Override word carousel / legacy ayah subtitles")
    p.add_argument("--recovery", action=argparse.BooleanOptionalAction, default=None,
                   help="Enable/disable bounded audio retries for suspicious word spans (default: config)")
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

    p = sub.add_parser("word-match", help="Match saved ASR word timestamps to Arabic/Bangla pairs")
    p.add_argument("transcription", help="Transcription JSON with recording-relative word timestamps")
    p.add_argument("--audio", help="Source audio/video, required when audio recovery is enabled")
    p.add_argument("--recovery", action=argparse.BooleanOptionalAction, default=None,
                   help="Enable/disable audio recovery for this run")
    p.add_argument("--surah", type=int, help="Surah number hint")
    p.add_argument("-o", "--output-dir", default="output")
    p.set_defaults(func=cmd_word_match)

    p = sub.add_parser("word-subtitle", help="Design the word carousel from saved cues, without transcribing again")
    p.add_argument("cues", help="word_cues.json; Arabic, Bangla, start, end per word")
    dimensions = p.add_mutually_exclusive_group(required=True)
    dimensions.add_argument("--video", help="Read dimensions from source video")
    dimensions.add_argument("--size", help="Explicit frame dimensions, e.g. 720x1280")
    p.add_argument("-o", "--output-dir", default="output")
    p.set_defaults(func=cmd_word_subtitle)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
