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


def _escape_ass_text(text: str) -> str:
    """Escape text that would otherwise be parsed as ASS override syntax."""
    return (
        str(text)
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", "\\N")
    )


def _ass_text_color(style_color: str) -> str:
    """Convert ASS style color &HAABBGGRR to inline text color &HBBGGRR&."""
    color = str(style_color).strip()
    if color.startswith("&H"):
        hex_part = color[2:]
        if len(hex_part) == 8:
            return f"&H{hex_part[2:]}&"
        if len(hex_part) == 6:
            return f"&H{hex_part}&"
    return color


def _tokenize_display_text(text: str) -> list[str]:
    return [token for token in str(text).split() if token]


def _display_tokens(seg: dict) -> list[str]:
    tokens = seg.get("bangla_tokens")
    if isinstance(tokens, list) and tokens:
        return [str(token).strip() for token in tokens if str(token).strip()]
    return _tokenize_display_text(seg.get("translation", ""))


def _join_display_text(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    if right[0] in ",.;:!?)]}":
        return left + right
    return left + " " + right


def _find_token_range(tokens: list[str], phrase_text: str) -> tuple[int, int]:
    phrase_tokens = _tokenize_display_text(phrase_text)
    if not tokens or not phrase_tokens:
        return 0, max(0, len(tokens) - 1)

    for i in range(0, len(tokens) - len(phrase_tokens) + 1):
        if tokens[i:i + len(phrase_tokens)] == phrase_tokens:
            return i, i + len(phrase_tokens) - 1
    return 0, max(0, len(tokens) - 1)


def _phrase_token_range(phrase: dict, tokens: list[str]) -> tuple[int, int]:
    if not tokens:
        return 0, 0
    if "bangla_start" in phrase and "bangla_end" in phrase:
        start = int(phrase["bangla_start"])
        end = int(phrase["bangla_end"])
    else:
        start, end = _find_token_range(tokens, str(phrase.get("text", "")))
    start = max(0, min(start, len(tokens) - 1))
    end = max(start, min(end, len(tokens) - 1))
    return start, end


def _highlight_ranges(phrases: list[dict], tokens: list[str]) -> list[tuple[int, int]]:
    ranges = []
    for phrase in phrases:
        start, end = _phrase_token_range(phrase, tokens)
        ranges.append((start, end))
    if not ranges:
        return []

    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _build_restyle_text(seg: dict, phrases: list[dict], style_cfg: dict) -> str:
    tokens = _display_tokens(seg)
    if not tokens:
        return _escape_ass_text(seg.get("translation", ""))
    ranges = _highlight_ranges(phrases, tokens)
    if not ranges:
        return _escape_ass_text(" ".join(tokens))

    primary = _ass_text_color(style_cfg.get("primary_color", "&H00FFFFFF"))
    highlight = _ass_text_color(
        style_cfg.get(
            "highlight_color",
            style_cfg.get("karaoke_color", "&H0000FFFF"),
        )
    )
    underline_on = "\\u1" if style_cfg.get("highlight_underline", False) else ""
    underline_off = "\\u0" if style_cfg.get("highlight_underline", False) else ""
    highlight_tag = f"{{\\c{highlight}{underline_on}}}"
    reset_tag = f"{{\\c{primary}{underline_off}}}"

    parts = []
    active = False
    for i, token in enumerate(tokens):
        prefix = ""
        should_highlight = any(start <= i <= end for start, end in ranges)
        if should_highlight and not active:
            prefix += highlight_tag
            active = True
        elif active and not should_highlight:
            prefix += reset_tag
            active = False
        parts.append(prefix + _escape_ass_text(token))
    if active and parts:
        parts[-1] += reset_tag
    return " ".join(parts)


def _restyle_intervals(seg: dict) -> list[tuple[float, float, list[dict]]]:
    phrases = seg.get("translation_phrases") or seg.get("phrases") or []
    start = float(seg["start"])
    end = float(seg["end"])
    if not phrases:
        return [(start, end, [])]

    timed = []
    boundaries = {start, end}
    for phrase in phrases:
        p_start = max(start, float(phrase.get("start", start)))
        p_end = min(end, float(phrase.get("end", p_start)))
        if p_end > p_start:
            timed.append((p_start, p_end, phrase))
            boundaries.add(p_start)
            boundaries.add(p_end)

    if not timed:
        return [(start, end, [])]

    points = sorted(boundaries)
    intervals: list[tuple[float, float, list[dict]]] = []
    for slice_start, slice_end in zip(points, points[1:]):
        if slice_end <= slice_start:
            continue
        active = [
            phrase
            for p_start, p_end, phrase in timed
            if p_start < slice_end and p_end > slice_start
        ]
        intervals.append((slice_start, slice_end, active))
    return intervals


def _build_restyle_dialogue_lines(seg: dict, style_cfg: dict) -> list[str]:
    lines = []
    for start, end, phrases in _restyle_intervals(seg):
        text = _build_restyle_text(seg, phrases, style_cfg)
        lines.append(
            f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},Default,,0,0,0,,{text}"
        )
    return lines


def _build_karaoke_text(seg: dict) -> str:
    """Build ASS karaoke tags from Gemini phrase timing maps."""
    phrases = seg.get("translation_phrases") or seg.get("phrases") or []
    if not phrases:
        return _escape_ass_text(seg["translation"])

    parts = []
    cursor = float(seg["start"])
    rendered_text = ""

    for phrase in phrases:
        text = str(phrase.get("text", "")).strip()
        if not text:
            continue

        start = float(phrase.get("start", cursor))
        end = float(phrase.get("end", start))
        if end < start:
            end = start

        gap_cs = max(0, int(round((start - cursor) * 100)))
        if gap_cs:
            parts.append(f"{{\\k{gap_cs}}}")

        piece = _join_display_text(rendered_text, text)
        if rendered_text:
            piece = piece[len(rendered_text):]
        rendered_text = _join_display_text(rendered_text, text)

        dur_cs = max(1, int(round((end - start) * 100)))
        parts.append(f"{{\\k{dur_cs}}}{_escape_ass_text(piece)}")
        cursor = max(cursor, end)

    if not parts:
        return _escape_ass_text(seg["translation"])
    return "".join(parts)


def _build_karaoke_dialogue_line(seg: dict) -> str:
    start = _format_ass_time(seg["start"])
    end = _format_ass_time(seg["end"])
    text = _build_karaoke_text(seg)
    return f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"


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
    secondary = style_cfg.get("secondary_color", style_cfg.get("karaoke_color", "&H0000FFFF"))
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
Style: Default,{font},{size},{primary},{secondary},{outline_c},{back_c},{bold},0,0,0,100,100,0,0,1,{outline_w},{shadow},{alignment},{ml},{mr},{mv},0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""

    highlight_mode = style_cfg.get("highlight_mode", "restyle")
    dialogue_lines = []
    for seg in segments:
        if highlight_mode == "karaoke":
            dialogue_lines.append(_build_karaoke_dialogue_line(seg))
        elif highlight_mode == "plain":
            start = _format_ass_time(seg["start"])
            end = _format_ass_time(seg["end"])
            text = _escape_ass_text(seg["translation"])
            dialogue_lines.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
            )
        else:
            dialogue_lines.extend(_build_restyle_dialogue_lines(seg, style_cfg))

    content = header + "\n" + "\n".join(dialogue_lines) + "\n"
    output_path.write_text(content, encoding="utf-8-sig")  # BOM for player compat
    logger.info(
        "ASS generated: %s (%d segments, %d dialogue lines)",
        output_path,
        len(segments),
        len(dialogue_lines),
    )
    return str(output_path)
