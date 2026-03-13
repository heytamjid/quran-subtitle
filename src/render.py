"""Burn subtitles into a video using ffmpeg."""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def render_video(
    video_path: str,
    ass_path: str,
    output_path: str,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
) -> str:
    """Burn ASS subtitles into a video.

    Args:
        video_path: Path to the original video.
        ass_path: Path to the .ass subtitle file.
        output_path: Where to save the final video.
        codec: Video codec (default libx264).
        crf: Constant rate factor for quality (lower = better, 18 is visually lossless).
        preset: Encoding speed preset.

    Returns:
        Path to the rendered video.
    """
    video_path = Path(video_path).resolve()
    ass_path = Path(ass_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not ass_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {ass_path}")

    # Run ffmpeg from the subtitle file's directory so we can pass just the
    # filename — this avoids Windows drive-letter colons (D:) breaking the
    # ffmpeg filter option parser.
    work_dir = str(ass_path.parent)
    ass_filename = ass_path.name

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"ass={ass_filename}",
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "copy",
        "-y",
        str(output_path),
    ]

    logger.info("Rendering video (cwd=%s): %s", work_dir, " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg rendering failed:\n{result.stderr}")

    logger.info("Video rendered: %s", output_path)
    return str(output_path)
