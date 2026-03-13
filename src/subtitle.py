"""Generate SRT and ASS subtitle files from matched ayah segments."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp: H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_srt(segments: list[dict], output_path: str) -> str:
    """Generate an SRT subtitle file.

    Args:
        segments: list of {"translation": str, "start": float, "end": float, ...}
        output_path: Where to write the .srt file.

    Returns:
        Path to the generated SRT file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_format_srt_time(seg['start'])} --> {_format_srt_time(seg['end'])}")
        lines.append(seg["translation"])
        lines.append("")  # blank line between entries

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("SRT generated: %s (%d entries)", output_path, len(segments))
    return str(output_path)


def generate_ass(segments: list[dict], style_cfg: dict, output_path: str) -> str:
    """Generate an ASS subtitle file with styled Bangla text.

    Args:
        segments: list of {"translation": str, "start": float, "end": float, ...}
        style_cfg: subtitle styling dict from config.yaml.
        output_path: Where to write the .ass file.

    Returns:
        Path to the generated ASS file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font = style_cfg.get("font_name", "Nirmala UI")
    size = style_cfg.get("font_size", 22)
    primary = style_cfg.get("primary_color", "&H00FFFFFF")
    outline_c = style_cfg.get("outline_color", "&H00000000")
    back_c = style_cfg.get("back_color", "&H80000000")
    outline_w = style_cfg.get("outline_width", 2)
    shadow = style_cfg.get("shadow_depth", 1)
    bold = -1 if style_cfg.get("bold", True) else 0
    alignment = style_cfg.get("alignment", 2)
    ml = style_cfg.get("margin_left", 40)
    mr = style_cfg.get("margin_right", 40)
    mv = style_cfg.get("margin_vertical", 50)

    header = f"""[Script Info]
Title: Quran Bangla Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{primary},&H000000FF,{outline_c},{back_c},{bold},0,0,0,100,100,0,0,1,{outline_w},{shadow},{alignment},{ml},{mr},{mv},0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""

    dialogue_lines = []
    for seg in segments:
        start = _format_ass_time(seg["start"])
        end = _format_ass_time(seg["end"])
        text = seg["translation"].replace("\n", "\\N")
        dialogue_lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
        )

    content = header + "\n" + "\n".join(dialogue_lines) + "\n"
    output_path.write_text(content, encoding="utf-8-sig")  # BOM for player compat
    logger.info("ASS generated: %s (%d entries)", output_path, len(segments))
    return str(output_path)
