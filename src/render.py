"""Burn subtitles into a video using ffmpeg."""

import logging
import json
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def probe_video_size(video_path: str) -> tuple[int, int]:
    """Return display dimensions, accounting for ffmpeg's automatic rotation."""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height:stream_tags=rotate:stream_side_data=rotation", "-of", "json", str(video_path),
    ], capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout)["streams"]
    if not streams:
        raise ValueError(f"No video stream in {video_path}")
    stream = streams[0]
    width, height = stream["width"], stream["height"]
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    for side in stream.get("side_data_list", []):
        rotation = float(side.get("rotation", rotation))
    return (height, width) if abs(rotation) % 180 == 90 else (width, height)


def _filter_path(path: str) -> str:
    return str(path).replace("\\", "/").replace(":", "\\:").replace("'", "'\\''")


def render_video(
    video_path: str,
    ass_path: str,
    output_path: str,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    fonts_dir: str | None = None,
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
    subtitle_filter = f"ass=filename='{_filter_path(ass_filename)}':shaping=complex"
    if fonts_dir:
        subtitle_filter += f":fontsdir='{_filter_path(str(Path(fonts_dir).resolve()))}'"

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", subtitle_filter,
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
