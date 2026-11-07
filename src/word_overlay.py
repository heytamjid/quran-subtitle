"""ASS presentation of timed Arabic/Bangla pairs, independent of data and ASR.

All coordinates are computed here. Layout, typography, palette and motion have
separate config sections; neither alignment nor translations know about them.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import re
import shutil
import subprocess

from PIL import ImageFont


@dataclass(frozen=True)
class Card:
    center: float
    width: float
    arabic: str
    bangla: str
    arabic_size: float
    bangla_size: float


def _time(t: float) -> str:
    centiseconds = max(0, round(t * 100))
    seconds, cs = divmod(centiseconds, 100)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def _color(value: str) -> str:
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ValueError(f"Use #RRGGBB overlay colors, got {value!r}")
    return f"&H{value[5:7]}{value[3:5]}{value[1:3]}&"


def _text(text: str) -> str:
    # Do not allow custom data to inject ASS override tags.
    return text.replace("\\", "＼").replace("{", "（").replace("}", "）").replace("\n", " ")


def _font(settings: dict, scale: float) -> ImageFont.FreeTypeFont:
    name = settings["font"]
    path = settings.get("font_path")
    if not path and shutil.which("fc-match"):
        result = subprocess.run(["fc-match", "-f", "%{file}\n%{family}", name], capture_output=True, text=True, check=True)
        path, families = result.stdout.strip().split("\n", 1)
        if name.casefold() not in [family.strip().casefold() for family in families.split(",")]:
            raise ValueError(f"Font {name!r} is not installed (system selected {families!r}). Install it or set font_path.")
    try:
        return ImageFont.truetype(path or name, max(1, round(settings["size"] * scale)))
    except OSError as exc:
        raise ValueError(f"Install {name!r} or configure typography font_path (and video.fonts_dir for rendering).") from exc


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: float) -> list[str]:
    lines = []
    for token in text.split():
        if lines and font.getlength(lines[-1] + " " + token) <= width:
            lines[-1] += " " + token
        else:
            lines.append(token)
    return lines or [""]


def _rounded(width: float, height: float, radius: float) -> str:
    r = min(radius, width / 2, height / 2)
    k = r * 0.55228475
    return (f"m {r:.2f} 0 l {width-r:.2f} 0 "
            f"b {width-r+k:.2f} 0 {width:.2f} {r-k:.2f} {width:.2f} {r:.2f} "
            f"l {width:.2f} {height-r:.2f} "
            f"b {width:.2f} {height-r+k:.2f} {width-r+k:.2f} {height:.2f} {width-r:.2f} {height:.2f} "
            f"l {r:.2f} {height:.2f} b {r-k:.2f} {height:.2f} 0 {height-r+k:.2f} 0 {height-r:.2f} "
            f"l 0 {r:.2f} b 0 {r-k:.2f} {r-k:.2f} 0 {r:.2f} 0")


def _settings(config: dict) -> tuple[dict, dict, dict, dict]:
    layout = {"reference_size": 720, "height": 202, "margin_x": 22, "margin_bottom": 24,
              "radius": 24, "padding": 16, "min_card_width": 180, "max_card_width": 300,
              "card_gap": 18, "arabic_top": 14, "bangla_top": 102, "center_x": 0.5}
    typography = {"arabic": {"font": "Noto Naskh Arabic", "size": 48},
                  "bangla": {"font": "Noto Sans Bengali", "size": 24, "max_lines": 2}}
    palette = {"background": "#F4ECE7", "decoration": "#DEDCEC", "decoration_alt": "#D9E8DE",
               "focus": "#FFF9F3", "arabic": "#263D35", "bangla": "#344A42",
               "muted": "#8A918A", "accent": "#719B89", "opacity": 0.95}
    motion = {"duration_ms": 180, "easing": "smoothstep", "fps": 30,
              "pause_threshold": 1.2, "hold_after_word": 0.35}
    layout.update(config.get("layout", {}))
    for language in typography:
        typography[language].update(config.get("typography", {}).get(language, {}))
    palette.update(config.get("palette", {}))
    motion.update(config.get("motion", {}))
    return layout, typography, palette, motion


def generate_word_ass(cues: list[dict], config: dict, output_path: str,
                      width: int, height: int) -> str:
    """Generate an on-video strip; positive word order moves right to left.

    Each cue has canonical Arabic/Bangla text and a recording-relative start/end.
    The strip moves only at a new cue, then stays still. Silence beyond the
    configured threshold hides it. The ASS canvas equals the source video.
    """
    if not cues:
        raise ValueError("No word cues to render")
    layout, typography, palette, motion = _settings(config)
    if width <= 0 or height <= 0 or layout["reference_size"] <= 0:
        raise ValueError("Video dimensions and reference_size must be positive")
    if motion["easing"] not in ("smoothstep", "linear"):
        raise ValueError("motion.easing must be smoothstep or linear")
    if not 1 <= motion["fps"] <= 100 or motion["duration_ms"] < 0:
        raise ValueError("Motion fps must be 1..100 and duration_ms must be nonnegative")
    if not 0 <= palette["opacity"] <= 1 or not 0 < layout["center_x"] < 1:
        raise ValueError("Invalid opacity or center_x")
    colors = {k: _color(v) for k, v in palette.items() if k != "opacity"}
    scale = min(width, height) / layout["reference_size"]
    left = layout["margin_x"] * scale
    panel_width = width - left * 2
    panel_height = layout["height"] * scale
    top = height - layout["margin_bottom"] * scale - panel_height
    center = width * layout["center_x"]
    padding = layout["padding"] * scale
    if left < 0 or top < 0 or panel_height <= 0 or panel_width <= 2 * padding:
        raise ValueError("Overlay panel does not fit inside the video")
    minimum = layout["min_card_width"] * scale
    maximum = min(layout["max_card_width"] * scale, panel_width - padding * 2)
    if not 0 < minimum <= maximum or not left + maximum / 2 <= center <= width - left - maximum / 2:
        raise ValueError("Card widths / center_x do not fit inside the panel")
    ar_font = _font(typography["arabic"], scale)
    bn_font = _font(typography["bangla"], scale)
    cards = []
    cursor = 0.0
    previous_end = 0.0
    for cue in cues:
        start, end = float(cue["start"]), float(cue["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < previous_end or end <= start:
            raise ValueError("Word cues must have finite, positive, non-overlapping times")
        previous_end = end
        arabic, bangla = _text(cue["arabic"]), _text(cue["translation"])
        card_width = min(maximum, max(minimum, ar_font.getlength(arabic) + padding * 2,
                                      bn_font.getlength(bangla) + padding * 2))
        available = card_width - padding * 2
        # Fit long glosses by wrapping and shrinking, never by discarding text.
        font = bn_font
        for size in range(bn_font.size, 0, -1):
            font = bn_font.font_variant(size=size)
            lines = _wrap(bangla, font, available)
            if len(lines) <= typography["bangla"]["max_lines"] and max(map(font.getlength, lines)) <= available:
                break
        ar_size = ar_font.size * min(1, available / max(1, ar_font.getlength(arabic)))
        if layout["bangla_top"] * scale + font.size * 1.6 * len(lines) > panel_height - padding:
            raise ValueError("Bangla text exceeds panel height; increase layout.height or adjust typography/bangla_top")
        cards.append(Card(cursor + card_width / 2, card_width, arabic, "\\N".join(lines), ar_size, font.size))
        cursor += card_width + layout["card_gap"] * scale
    lines = [f"""[Script Info]
Title: QuranSub Arabic and Bangla word carousel
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Arabic,{typography['arabic']['font']},{ar_font.size},&H00263D35,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,8,0,0,0,1
Style: Bangla,{typography['bangla']['font']},{bn_font.size},&H00344A42,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,8,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""]
    clip = f"\\clip({math.ceil(left+padding/2)},{math.ceil(top)},{math.floor(width-left-padding/2)},{math.floor(top+panel_height)})"

    def event(layer, start, end, style, text):
        if round(end * 100) > round(start * 100):
            lines.append(f"Dialogue: {layer},{_time(start)},{_time(end)},{style},,0,0,0,,{text}")

    def shape(layer, start, end, x, y, w, h, radius, color, opacity=1):
        alpha = round(255 * (1 - opacity))
        event(layer, start, end, "Arabic", f"{{\\an7\\pos({x:.2f},{y:.2f})\\bord0\\shad0\\1c{color}\\1a&H{alpha:02X}&\\p1}}{_rounded(w,h,radius)}")

    for i, cue in enumerate(cues):
        start, end = cue["start"], cue["end"]
        if i + 1 < len(cues) and cues[i+1]["start"] - end <= motion["pause_threshold"]:
            end = cues[i+1]["start"]
        else:
            end += motion["hold_after_word"]
            if i + 1 < len(cues):
                end = min(end, cues[i+1]["start"])
        shape(0, start, end, left, top, panel_width, panel_height, layout["radius"]*scale,
              colors["background"], palette["opacity"])
        shape(1, start, end, left+padding, top+padding, 54*scale, 5*scale, 2*scale, colors["decoration"])
        shape(1, start, end, width-left-padding-54*scale, top+panel_height-padding-5*scale,
              54*scale, 5*scale, 2*scale, colors["decoration_alt"])
        focus_width = cards[i].width
        shape(1, start, end, center-focus_width/2, top+8*scale, focus_width, panel_height-16*scale,
              18*scale, colors["focus"], 0.9)
        shape(2, start, end, center-16*scale, top+panel_height-12*scale, 32*scale, 3*scale,
              1.5*scale, colors["accent"])
        target = cards[i].center
        reset = i == 0 or start - cues[i-1]["end"] > motion["pause_threshold"]
        origin = target if reset else cards[i-1].center
        duration = min(motion["duration_ms"] / 1000, (end - start) * 0.45)
        steps = max(1, math.ceil(duration * motion["fps"])) if origin != target and duration else 0
        boundaries = [start + duration * j / steps for j in range(steps + 1)] if steps else [start]
        boundaries.append(end)
        for a, b in zip(boundaries, boundaries[1:]):
            def offset(t):
                progress = min(1.0, (t-start)/duration) if duration else 1.0
                if motion["easing"] == "smoothstep":
                    progress = progress * progress * (3 - 2 * progress)
                return origin + (target-origin) * progress
            offset_a, offset_b = offset(a), offset(b)
            for j, card in enumerate(cards):
                x1, x2 = center + card.center-offset_a, center + card.center-offset_b
                if max(x1, x2) + card.width/2 < left or min(x1, x2) - card.width/2 > width-left:
                    continue
                for style, text, y, size, color in (
                    ("Arabic", card.arabic, top+layout["arabic_top"]*scale, card.arabic_size,
                     colors["arabic"] if i == j else colors["muted"]),
                    ("Bangla", card.bangla, top+layout["bangla_top"]*scale, card.bangla_size,
                     colors["bangla"] if i == j else colors["muted"]),
                ):
                    position = (f"\\pos({x2:.2f},{y:.2f})" if abs(x2-x1) < 0.01 else
                                f"\\move({x1:.2f},{y:.2f},{x2:.2f},{y:.2f})")
                    event(3, a, b, style, f"{{\\an8{clip}{position}\\fs{size:.2f}\\1c{color}}}{text}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return str(output)
