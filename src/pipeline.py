"""Orchestrate the full subtitle generation pipeline."""

import logging
from pathlib import Path

from .audio import extract_audio
from .transcribe import transcribe_audio, save_transcription
from .quran_data import load_quran_data
from .match import align_words_to_ayahs, save_segments
from .subtitle import generate_srt, generate_ass
from .render import render_video

logger = logging.getLogger(__name__)


def _unique_output_dir(base_output: Path, name: str) -> Path:
    """Return a unique subdirectory under base_output for this run.

    First tries base_output/name. If it exists, tries name_1, name_2, etc.
    """
    candidate = base_output / name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    counter = 1
    while True:
        candidate = base_output / f"{name}_{counter}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        counter += 1


def run_pipeline(video_path: str, config: dict) -> str:
    """Run the full pipeline: video -> subtitled video.

    Args:
        video_path: Path to the input video.
        config: Parsed config dict (from config.yaml).

    Returns:
        Path to the final subtitled video.
    """
    project_root = Path(config.get("_project_root", ".")).resolve()

    # Resolve video: look in input/ directory first, then as-is
    video_path = Path(video_path)
    if not video_path.is_absolute():
        input_candidate = project_root / "input" / video_path
        if input_candidate.exists():
            video_path = input_candidate
        else:
            video_path = (project_root / video_path).resolve()
    video_path = video_path.resolve()

    # Create a unique output subdirectory named after the video
    base_output = Path(config["paths"]["output_dir"])
    if not base_output.is_absolute():
        base_output = project_root / base_output
    base_output = base_output.resolve()

    video_stem = video_path.stem  # e.g. "taha"
    output_dir = _unique_output_dir(base_output, video_stem)

    tc = config["transcription"]
    mc = config["matching"]
    sc = config["subtitle"]
    vc = config.get("video", {})

    # Resolve CSV path relative to project root (where config.yaml lives)
    csv_path = Path(config["paths"]["quran_csv"])
    if not csv_path.is_absolute():
        csv_path = Path(config.get("_project_root", ".")) / csv_path

    # --- Step 1: Extract audio -----------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1 / 6  —  Extract audio")
    audio_path = extract_audio(
        str(video_path),
        str(output_dir / "audio.mp3"),
    )

    # --- Step 2: Transcribe --------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2 / 6  —  Transcribe audio")
    transcription = transcribe_audio(
        audio_path,
        model_id=tc.get("model_id", "tarteel-ai/whisper-base-ar-quran"),
        base_model_id=tc.get("base_model_id", "openai/whisper-base"),
        chunk_length_s=tc.get("chunk_length_s", 30),
        stride_length_s=tc.get("stride_length_s", 5),
        language=tc.get("language", "ar"),
    )
    save_transcription(transcription, str(output_dir / "transcription.json"))

    # --- Step 3: Load Quran data ---------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3 / 6  —  Load Quran data")
    quran_ayahs = load_quran_data(str(csv_path))

    # --- Step 4: Match transcription to ayahs --------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4 / 6  —  Match transcription to Quran")
    segments = align_words_to_ayahs(
        transcription["words"],
        quran_ayahs,
        surah_hint=mc.get("surah_hint"),
    )
    save_segments(segments, str(output_dir / "matched_segments.json"))

    if not segments:
        raise RuntimeError("No matching ayahs found. Check the audio or try a surah hint.")

    # --- Step 5: Generate subtitles ------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5 / 6  —  Generate subtitles")
    generate_srt(segments, str(output_dir / "subtitles.srt"))
    ass_path = generate_ass(segments, sc, str(output_dir / "subtitles.ass"))

    # --- Step 6: Render final video ------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6 / 6  —  Render video with subtitles")
    final_path = render_video(
        str(video_path),
        ass_path,
        str(output_dir / "final_video_with_subtitles.mp4"),
        codec=vc.get("codec", "libx264"),
        crf=vc.get("crf", 18),
        preset=vc.get("preset", "medium"),
    )

    logger.info("=" * 60)
    logger.info("DONE — Output: %s", final_path)
    return final_path
