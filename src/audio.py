"""Extract audio from video files."""

import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_path: str = None) -> str:
    """Extract audio from a video file using ffmpeg.

    Args:
        video_path: Path to the input video.
        output_path: Where to save the audio. Defaults to same name with .mp3.

    Returns:
        Path to the extracted audio file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = video_path.with_suffix(".mp3")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting audio: %s -> %s", video_path, output_path)

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn",                       # no video
        "-acodec", "libmp3lame",
        "-q:a", "2",                 # high quality VBR
        "-y",                        # overwrite
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")

    logger.info("Audio extracted: %s", output_path)
    return str(output_path)
