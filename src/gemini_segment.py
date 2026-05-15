"""Use Gemini on Vertex AI to split ayahs into display/timing units."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SCHEMA_VERSION = 3


def _tokenize_translation(text: str) -> list[str]:
    """Tokenize display text for index-based highlighting.

    This intentionally keeps punctuation attached to the adjacent word. It gives
    Gemini stable whitespace token indices while preserving the original Bangla
    word order and making reconstruction predictable for ASS/HTML renderers.
    """
    return [token for token in str(text).split() if token]


def _text_from_tokens(tokens: list[str], start_idx: int, end_idx: int) -> str:
    if not tokens:
        return ""
    start_idx = max(0, min(start_idx, len(tokens) - 1))
    end_idx = max(start_idx, min(end_idx, len(tokens) - 1))
    return " ".join(tokens[start_idx:end_idx + 1])


def _normalized_for_match(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _tokens_match_text(tokens: list[str], start_idx: int, end_idx: int, text: str) -> bool:
    if not text:
        return False
    return _normalized_for_match(_text_from_tokens(tokens, start_idx, end_idx)) == _normalized_for_match(text)


def _read_project_id(credentials_path: str | None) -> str | None:
    if not credentials_path:
        return None
    try:
        with open(credentials_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        project_id = data.get("project_id")
        return str(project_id) if project_id else None
    except OSError:
        return None
    except json.JSONDecodeError:
        return None


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _is_current_schema(segments: list[dict], cfg: dict) -> bool:
    if not isinstance(segments, list):
        return False
    mapping_mode = _mapping_mode(cfg)
    display_mode = _display_mode(cfg)
    return bool(segments) and all(
        seg.get("schema_version") == _SCHEMA_VERSION
        and seg.get("mapping_granularity") == mapping_mode
        and seg.get("display_granularity") == display_mode
        and isinstance(seg.get("bangla_tokens"), list)
        for seg in segments
    )


def _timed_word_span(words: list[dict], start_idx: int, end_idx: int) -> tuple[float, float]:
    start_idx = max(0, min(start_idx, len(words) - 1))
    end_idx = max(start_idx, min(end_idx, len(words) - 1))
    return float(words[start_idx]["start"]), float(words[end_idx]["end"])


def _normalize_phrase_maps(
    phrases: list[dict],
    unit: dict,
    bangla_tokens: list[str],
    display_bangla_start: int,
    display_bangla_end: int,
) -> list[dict]:
    """Normalize phrase maps and derive local Bangla indices for renderers."""
    normalized: list[dict] = []
    min_arabic = int(unit["arabic_start"])
    max_arabic = int(unit["arabic_end"])
    for raw in phrases:
        if not bangla_tokens:
            continue

        text = str(raw.get("text", "")).strip()
        raw_b_start = raw.get("global_bangla_start", raw.get("bangla_start", 0))
        raw_b_end = raw.get("global_bangla_end", raw.get("bangla_end", raw_b_start))
        b_start, b_end, g_start, g_end = _resolve_bangla_range(
            int(raw_b_start),
            int(raw_b_end),
            text,
            bangla_tokens,
            display_bangla_start,
            display_bangla_end,
        )

        a_start = int(raw.get("arabic_start", min_arabic))
        a_end = int(raw.get("arabic_end", a_start))
        a_start = max(min_arabic, min(a_start, max_arabic))
        a_end = max(a_start, min(a_end, max_arabic))

        text = str(text or _text_from_tokens(bangla_tokens, b_start, b_end)).strip()
        if not text:
            continue

        normalized.append({
            "text": text,
            "bangla_start": b_start,
            "bangla_end": b_end,
            "global_bangla_start": g_start,
            "global_bangla_end": g_end,
            "arabic_start": a_start,
            "arabic_end": a_end,
        })

    return normalized


def _resolve_bangla_range(
    raw_start: int,
    raw_end: int,
    text: str,
    bangla_tokens: list[str],
    display_bangla_start: int,
    display_bangla_end: int,
) -> tuple[int, int, int, int]:
    """Resolve Gemini Bangla indices, preferring global coordinates.

    New prompts ask for global_bangla_start/end. Compatibility handling keeps
    old local bangla_start/end outputs usable and avoids clamping invalid
    global values to the final local token.
    """
    max_local = len(bangla_tokens) - 1
    raw_end = max(raw_start, raw_end)

    def from_local(start: int, end: int) -> tuple[int, int, int, int]:
        start = max(0, min(start, max_local))
        end = max(start, min(end, max_local))
        return start, end, display_bangla_start + start, display_bangla_start + end

    def from_global(start: int, end: int) -> tuple[int, int, int, int]:
        start = max(display_bangla_start, min(start, display_bangla_end))
        end = max(start, min(end, display_bangla_end))
        return start - display_bangla_start, end - display_bangla_start, start, end

    global_valid = display_bangla_start <= raw_start <= raw_end <= display_bangla_end
    local_valid = 0 <= raw_start <= raw_end <= max_local

    if global_valid:
        candidate = from_global(raw_start, raw_end)
        if _tokens_match_text(bangla_tokens, candidate[0], candidate[1], text) or not text:
            return candidate

    if local_valid:
        candidate = from_local(raw_start, raw_end)
        if _tokens_match_text(bangla_tokens, candidate[0], candidate[1], text) or not text:
            return candidate

    phrase_tokens = _tokenize_translation(text)
    if phrase_tokens:
        for i in range(0, len(bangla_tokens) - len(phrase_tokens) + 1):
            end = i + len(phrase_tokens) - 1
            if bangla_tokens[i:end + 1] == phrase_tokens:
                return from_local(i, end)

    if global_valid:
        return from_global(raw_start, raw_end)
    if local_valid:
        return from_local(raw_start, raw_end)
    return from_local(raw_start, raw_end)


def _fallback_phrase(text: str, words: list[dict], start_idx: int, end_idx: int) -> list[dict]:
    start, end = _timed_word_span(words, start_idx, end_idx)
    tokens = _tokenize_translation(text)
    return [{
        "text": text.strip(),
        "bangla_start": 0,
        "bangla_end": max(0, len(tokens) - 1),
        "global_bangla_start": 0,
        "global_bangla_end": max(0, len(tokens) - 1),
        "arabic_start": start_idx,
        "arabic_end": end_idx,
        "start": start,
        "end": end,
    }]


def _fallback_segments(ayah: dict, source: str = "fallback", cfg: dict | None = None) -> list[dict]:
    mapping_mode = _mapping_mode(cfg or {})
    display_mode = _display_mode(cfg or {})
    words = ayah.get("words") or []
    if not words:
        return [{
            **ayah,
            "schema_version": _SCHEMA_VERSION,
            "display_granularity": display_mode,
            "mapping_granularity": mapping_mode,
            "source": source,
            "bangla_tokens": _tokenize_translation(ayah.get("translation", "")),
            "layout": {"preset": "bottom_center"},
            "translation_phrases": [],
        }]

    start_idx = 0
    end_idx = len(words) - 1
    start, end = _timed_word_span(words, start_idx, end_idx)
    translation = str(ayah.get("translation", "")).strip()
    bangla_tokens = _tokenize_translation(translation)
    return [{
        "surah": ayah.get("surah"),
        "ayah": ayah.get("ayah"),
        "schema_version": _SCHEMA_VERSION,
        "display_granularity": display_mode,
        "mapping_granularity": mapping_mode,
        "arabic": ayah.get("arabic", ""),
        "translation": translation,
        "start": start,
        "end": end,
        "words": words,
        "source": source,
        "parent_translation": ayah.get("translation", ""),
        "display_arabic_start": start_idx,
        "display_arabic_end": end_idx,
        "arabic_start": start_idx,
        "arabic_end": end_idx,
        "display_bangla_start": 0,
        "display_bangla_end": max(0, len(bangla_tokens) - 1),
        "bangla_tokens": bangla_tokens,
        "layout": {"preset": "bottom_center"},
        "translation_phrases": _fallback_phrase(translation, words, start_idx, end_idx),
    }]


def _mapping_mode(cfg: dict) -> str:
    mode = str(cfg.get("mapping_granularity", "word")).strip().lower()
    return mode if mode in {"phrase", "word"} else "phrase"


def _display_mode(cfg: dict) -> str:
    mode = str(cfg.get("display_granularity", "clause")).strip().lower()
    return mode if mode in {"clause", "compact_phrase"} else "clause"


def _display_mode_instructions(mode: str) -> str:
    if mode == "compact_phrase":
        return """Display granularity mode: COMPACT_PHRASE.
- Split the ayah into shorter consecutive display units suitable for large
  fonts.
- A normal everyday reader should be able to read the full Bangla display unit
  comfortably during the time it stays on screen.
- Prefer compact consecutive Bangla phrases that map to contiguous Arabic word
  ranges.
- The size targets are soft readability hints, not hard limits. Meaning and
  clean Arabic/Bangla synchronization are more important than hitting a number.
- Do not split a meaningful expression only to satisfy a size target."""

    return """Display granularity mode: CLAUSE.
- Split the ayah into clause-level display units.
- Prefer complete clauses or complete expression units over tiny fragments.
- Display units may be longer than compact_phrase mode when that improves
  meaning and readability.
- The size targets are soft readability hints, not hard limits. Meaning and
  clean Arabic/Bangla synchronization are more important than hitting a number."""


def _display_few_shot(mode: str) -> str:
    if mode == "compact_phrase":
        return """Display split example for COMPACT_PHRASE mode:
Translation:
"And Allah invites to the Home of Peace and guides whom He wills to a straight path."
Good display_units:
1. "And Allah invites"
2. "to the Home of Peace"
3. "and guides whom He wills"
4. "to a straight path."
Reason: each unit is a short consecutive translation phrase for large fonts,
and each can be mapped to a contiguous Arabic range. Do not create disconnected
translation snippets inside one display unit."""

    return """Display split example for CLAUSE mode:
Translation:
"And Allah invites to the Home of Peace and guides whom He wills to a straight path."
Good display_units:
1. "And Allah invites to the Home of Peace"
2. "and guides whom He wills to a straight path."
Reason: these are complete clause-level units with shared Arabic/translation
boundaries."""


def _mapping_mode_instructions(mode: str) -> str:
    if mode == "word":
        return """Mapping granularity mode: WORD.
- Prefer the smallest useful Bangla token ranges.
- Map individual Bangla tokens when that is linguistically reasonable.
- Keep multi-token Bangla expressions together when one Arabic word/phrase
  naturally translates to the whole expression, or when splitting would mislead.
- It is acceptable for multiple Bangla token ranges to map to the same Arabic
  word range.
- Usually produce one mapping per Bangla word, except for idioms, helper verb
  expressions, compound terms, or case/postposition expressions."""

    return """Mapping granularity mode: PHRASE.
- Prefer stable semantic phrase chunks rather than isolated function words.
- Use noun phrases, verb phrases, prepositional/case phrases, and compound
  expressions.
- Do not make one giant phrase unless the whole display unit is genuinely one
  indivisible expression.
- Usually produce 2-6 phrase mappings per display unit when the unit has enough
  words.
- It is acceptable for multiple Bangla phrase ranges to map to the same Arabic
  word range."""


def _mapping_few_shot(mode: str) -> str:
    if mode == "word":
        return """Granularity example for WORD mode:
Original Bangla token indices:
0="আর", 1="আল্লাহ", 2="শান্তির", 3="আবাসের", 4="দিকে", 5="আহবান", 6="করেন"
Arabic timing meaning:
0 = "And Allah", 1 = "invites", 2..4 = "to the Home of Peace"
Good translation_phrases:
[
  {"global_bangla_start": 0, "global_bangla_end": 0, "text": "আর", "arabic_start": 0, "arabic_end": 0},
  {"global_bangla_start": 1, "global_bangla_end": 1, "text": "আল্লাহ", "arabic_start": 0, "arabic_end": 0},
  {"global_bangla_start": 5, "global_bangla_end": 6, "text": "আহবান করেন", "arabic_start": 1, "arabic_end": 1},
  {"global_bangla_start": 4, "global_bangla_end": 4, "text": "দিকে", "arabic_start": 2, "arabic_end": 2},
  {"global_bangla_start": 3, "global_bangla_end": 3, "text": "আবাসের", "arabic_start": 3, "arabic_end": 3},
  {"global_bangla_start": 2, "global_bangla_end": 2, "text": "শান্তির", "arabic_start": 4, "arabic_end": 4}
]
Reason: the Bangla sentence order stays unchanged, but highlight timing follows
Arabic recitation order.

Offset example for WORD mode:
If a display unit has display_bangla_start=3 and bangla_tokens=["আহারে", "ও", "ভোগে"],
then "আহারে" must use global_bangla_start=3, not 0."""

    return """Granularity example for PHRASE mode:
Original Bangla token indices:
0="আর", 1="আল্লাহ", 2="শান্তির", 3="আবাসের", 4="দিকে", 5="আহবান", 6="করেন"
Arabic timing meaning:
0 = "And Allah", 1 = "invites", 2..4 = "to the Home of Peace"
Good translation_phrases:
[
  {"global_bangla_start": 0, "global_bangla_end": 1, "text": "আর আল্লাহ", "arabic_start": 0, "arabic_end": 0},
  {"global_bangla_start": 2, "global_bangla_end": 4, "text": "শান্তির আবাসের দিকে", "arabic_start": 2, "arabic_end": 4},
  {"global_bangla_start": 5, "global_bangla_end": 6, "text": "আহবান করেন", "arabic_start": 1, "arabic_end": 1}
]
Reason: these are stable semantic phrases. Do not collapse all seven tokens into
one phrase because useful phrase-level synchronization would be lost.

Offset example for PHRASE mode:
If a display unit has display_bangla_start=9 and bangla_tokens=["এবং", "আশা", "তাদেরকে"],
then "আশা" must use global_bangla_start=10, not 1."""


def _build_prompt(ayah: dict, cfg: dict) -> str:
    max_chars = cfg.get("max_display_chars", 86)
    max_words = cfg.get("max_display_arabic_words", 12)
    max_bangla_tokens = cfg.get("max_display_bangla_tokens", 12)
    display_mode = _display_mode(cfg)
    mapping_mode = _mapping_mode(cfg)
    display_instructions = _display_mode_instructions(display_mode)
    display_example = _display_few_shot(display_mode)
    mode_instructions = _mapping_mode_instructions(mapping_mode)
    mode_example = _mapping_few_shot(mapping_mode)

    original_bangla_tokens = _tokenize_translation(ayah.get("translation", ""))
    payload = {
        "surah": ayah.get("surah"),
        "ayah": ayah.get("ayah"),
        "authoritative_arabic_ayah": ayah.get("arabic", ""),
        "original_bangla_translation": ayah.get("translation", ""),
        "original_bangla_tokens": [
            {"index": i, "token": token}
            for i, token in enumerate(original_bangla_tokens)
        ],
        "timed_arabic_words": [
            {
                "index": i,
                "word": w.get("word", ""),
                "start": w.get("start"),
                "end": w.get("end"),
            }
            for i, w in enumerate(ayah.get("words") or [])
        ],
    }

    return f"""You are preparing synchronized Quran subtitles.

Return JSON only. Do not include Markdown.

Goal:
1. Split the ayah into display units according to display_granularity.
   Prefer units under about {max_chars} Bangla characters, about
   {max_bangla_tokens} Bangla tokens, and about {max_words} Arabic timed words.
   These are soft readability hints, not hard limits. Do not split or merge in
   a way that damages meaning or recitation/translation synchronization.
2. Each display unit must reference a contiguous range of
   original_bangla_tokens using display_bangla_start/display_bangla_end. Keep
   the Bangla word order exactly as the original translation presents it. Do not
   reorder Bangla to match Arabic grammar. Each display unit must also map to a
   contiguous Arabic word range using arabic_start/arabic_end.
3. Map each Bangla phrase to one or more Arabic timed word indices. Some Bangla
   phrases may map to multiple Arabic words and some Arabic particles may be
   absorbed into a larger Bangla phrase.
4. The phrase maps inside a display unit must use global_bangla_start and
   global_bangla_end. These are GLOBAL indices from original_bangla_tokens, not
   local indices from bangla_tokens. Do not output bangla_start or bangla_end.
   Arabic ranges may move forward or backward relative to Bangla order when
   Arabic and Bangla syntax differ. Do not force Bangla into Arabic order.
5. If multiple Bangla phrases correspond to the same Arabic word or the same
   Arabic expression, give those Bangla phrases identical arabic_start and
   arabic_end values so they highlight together. Do not create partial overlaps
   for phrases that should move as one semantic unit.

{display_instructions}

{mode_instructions}

Use timed_arabic_words indices for all arabic_start/arabic_end values.
All start/end indices are inclusive.

Required JSON shape:
{{
  "display_units": [
    {{
      "arabic_start": 0,
      "arabic_end": 5,
      "display_bangla_start": 0,
      "display_bangla_end": 6,
      "bangla_text": "contiguous part of the original Bangla translation",
      "bangla_tokens": ["token0", "token1", "token2"],
      "layout": {{"preset": "bottom_center"}},
      "translation_phrases": [
        {{"global_bangla_start": 0, "global_bangla_end": 0, "text": "Bangla phrase", "arabic_start": 3, "arabic_end": 3}},
        {{"global_bangla_start": 1, "global_bangla_end": 2, "text": "next Bangla phrase", "arabic_start": 0, "arabic_end": 2}}
      ]
    }}
  ]
}}

{display_example}

{mode_example}

Now process this ayah:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def _make_client(cfg: dict):
    credentials_path = cfg.get("credentials_path")
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

    project_id = cfg.get("project_id") or _read_project_id(credentials_path)
    location = cfg.get("location", "global")
    if not project_id:
        raise ValueError(
            "Gemini is enabled, but no project_id was configured and none could "
            "be read from the service account JSON."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "Missing dependency google-genai. Install requirements.txt before "
            "running Gemini segmentation."
        ) from exc

    client = genai.Client(vertexai=True, project=project_id, location=location)
    config = types.GenerateContentConfig(
        temperature=float(cfg.get("temperature", 0.0)),
        response_mime_type="application/json",
    )
    return client, config


def _call_gemini(client, request_config, model: str, prompt: str) -> dict:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=request_config,
    )
    return _extract_json(response.text or "")


def _normalize_response(ayah: dict, response: dict, cfg: dict | None = None) -> list[dict]:
    mapping_mode = _mapping_mode(cfg or {})
    display_mode = _display_mode(cfg or {})
    words = ayah.get("words") or []
    if not words:
        return _fallback_segments(ayah, source="gemini_no_words", cfg=cfg)

    parent_bangla_tokens = _tokenize_translation(ayah.get("translation", ""))
    units = response.get("display_units") or response.get("segments") or []
    if not isinstance(units, list) or not units:
        return _fallback_segments(ayah, source="gemini_empty", cfg=cfg)

    display_segments: list[dict] = []
    word_count = len(words)
    last_end = -1

    for unit in units:
        if not isinstance(unit, dict):
            continue
        a_start = int(unit.get("arabic_start", last_end + 1))
        a_end = int(unit.get("arabic_end", a_start))
        a_start = max(0, min(a_start, word_count - 1))
        a_end = max(a_start, min(a_end, word_count - 1))

        # Avoid overlapping display units. Gaps are allowed when the ASR output
        # contains noise, but overlaps produce unstable subtitle timing.
        if a_start <= last_end:
            a_start = min(last_end + 1, word_count - 1)
            a_end = max(a_start, a_end)
        if a_start >= word_count:
            continue

        bangla_text = str(
            unit.get("bangla_text") or unit.get("translation") or ""
        ).strip()
        display_bangla_start = int(unit.get("display_bangla_start", unit.get("bangla_start", 0)))
        display_bangla_end = int(unit.get("display_bangla_end", unit.get("bangla_end", display_bangla_start)))
        if parent_bangla_tokens:
            max_parent_bangla = len(parent_bangla_tokens) - 1
            display_bangla_start = max(0, min(display_bangla_start, max_parent_bangla))
            display_bangla_end = max(display_bangla_start, min(display_bangla_end, max_parent_bangla))

        bangla_tokens = unit.get("bangla_tokens")
        if not isinstance(bangla_tokens, list) or not bangla_tokens:
            if parent_bangla_tokens:
                bangla_tokens = parent_bangla_tokens[display_bangla_start:display_bangla_end + 1]
            else:
                bangla_tokens = _tokenize_translation(bangla_text)
        bangla_tokens = [str(token).strip() for token in bangla_tokens if str(token).strip()]

        if not bangla_text:
            bangla_text = " ".join(bangla_tokens)
        if not bangla_text or not bangla_tokens:
            continue

        start, end = _timed_word_span(words, a_start, a_end)
        raw_phrases = unit.get("translation_phrases") or unit.get("phrases") or []
        phrase_maps = _normalize_phrase_maps(
            raw_phrases,
            {"arabic_start": a_start, "arabic_end": a_end},
            bangla_tokens,
            display_bangla_start,
            display_bangla_end,
        )
        if not phrase_maps:
            phrase_maps = [{
                "text": bangla_text,
                "bangla_start": 0,
                "bangla_end": len(bangla_tokens) - 1,
                "global_bangla_start": display_bangla_start,
                "global_bangla_end": display_bangla_end,
                "arabic_start": a_start,
                "arabic_end": a_end,
            }]

        timed_phrases = []
        for phrase in phrase_maps:
            phrase_start, phrase_end = _timed_word_span(
                words,
                int(phrase["arabic_start"]),
                int(phrase["arabic_end"]),
            )
            timed_phrases.append({
                **phrase,
                "global_bangla_start": int(phrase["global_bangla_start"]),
                "global_bangla_end": int(phrase["global_bangla_end"]),
                "start": phrase_start,
                "end": phrase_end,
            })

        display_segments.append({
            "surah": ayah.get("surah"),
            "ayah": ayah.get("ayah"),
            "schema_version": _SCHEMA_VERSION,
            "display_granularity": display_mode,
            "mapping_granularity": mapping_mode,
            "arabic": ayah.get("arabic", ""),
            "translation": bangla_text,
            "start": start,
            "end": end,
            "words": words[a_start:a_end + 1],
            "source": "gemini",
            "parent_translation": ayah.get("translation", ""),
            "display_arabic_start": a_start,
            "display_arabic_end": a_end,
            "arabic_start": a_start,
            "arabic_end": a_end,
            "display_bangla_start": display_bangla_start,
            "display_bangla_end": display_bangla_end,
            "bangla_tokens": bangla_tokens,
            "layout": unit.get("layout") or {"preset": "bottom_center"},
            "translation_phrases": timed_phrases,
        })
        last_end = a_end

    return display_segments or _fallback_segments(ayah, source="gemini_invalid", cfg=cfg)


def segment_ayahs_with_gemini(
    ayah_segments: list[dict],
    cfg: dict,
    output_path: str | None = None,
) -> list[dict]:
    """Return display-level subtitle segments with phrase timing maps."""
    output = Path(output_path) if output_path else None
    if output and cfg.get("cache_results", True) and output.exists():
        with open(output, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if _is_current_schema(cached, cfg):
            logger.info("Using cached Gemini segmentation: %s", output)
            return cached
        logger.info("Ignoring stale Gemini segmentation cache: %s", output)

    if not cfg.get("enabled", False):
        logger.info("Gemini segmentation disabled; using ayah-level subtitles")
        enhanced = []
        for ayah in ayah_segments:
            enhanced.extend(_fallback_segments(ayah, source="disabled", cfg=cfg))
        if output:
            save_segments(enhanced, str(output))
        return enhanced

    client, request_config = _make_client(cfg)
    model = cfg.get("model", "gemini-2.5-flash")
    fallback_on_error = cfg.get("fallback_on_error", True)

    enhanced: list[dict] = []
    for ayah in ayah_segments:
        try:
            prompt = _build_prompt(ayah, cfg)
            response = _call_gemini(client, request_config, model, prompt)
            display_segments = _normalize_response(ayah, response, cfg)
            enhanced.extend(display_segments)
            logger.info(
                "Gemini segmented Surah %s Ayah %s into %d display units",
                ayah.get("surah"), ayah.get("ayah"), len(display_segments),
            )
        except Exception:
            if not fallback_on_error:
                raise
            logger.exception(
                "Gemini segmentation failed for Surah %s Ayah %s; using fallback",
                ayah.get("surah"), ayah.get("ayah"),
            )
            enhanced.extend(_fallback_segments(ayah, source="gemini_error", cfg=cfg))

    if output:
        save_segments(enhanced, str(output))
    return enhanced


def save_segments(segments: list[dict], output_path: str) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.info("Gemini segments saved: %s", output)
    return str(output)
