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


def _timed_word_span(words: list[dict], start_idx: int, end_idx: int) -> tuple[float, float]:
    start_idx = max(0, min(start_idx, len(words) - 1))
    end_idx = max(start_idx, min(end_idx, len(words) - 1))
    return float(words[start_idx]["start"]), float(words[end_idx]["end"])


def _merge_phrase_maps(phrases: list[dict], unit: dict) -> list[dict]:
    """Keep phrase order while forcing non-decreasing karaoke timing.

    Arabic and Bangla syntax can diverge. If a fine-grained map would require
    out-of-order highlighting, merge adjacent phrases so ASS karaoke can still
    preserve the original Bangla sequence.
    """
    merged: list[dict] = []
    min_start = int(unit["arabic_start"])
    max_end = int(unit["arabic_end"])

    for raw in phrases:
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        a_start = int(raw.get("arabic_start", min_start))
        a_end = int(raw.get("arabic_end", a_start))
        a_start = max(min_start, min(a_start, max_end))
        a_end = max(a_start, min(a_end, max_end))

        item = {"text": text, "arabic_start": a_start, "arabic_end": a_end}
        if merged and item["arabic_start"] <= merged[-1]["arabic_end"]:
            prev = merged[-1]
            prev["text"] = _join_text(prev["text"], item["text"])
            prev["arabic_start"] = min(prev["arabic_start"], item["arabic_start"])
            prev["arabic_end"] = max(prev["arabic_end"], item["arabic_end"])
        else:
            merged.append(item)

    return merged


def _join_text(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    if right[0] in ",.;:!?)]}":
        return left + right
    return left + " " + right


def _fallback_phrase(text: str, words: list[dict], start_idx: int, end_idx: int) -> list[dict]:
    start, end = _timed_word_span(words, start_idx, end_idx)
    return [{
        "text": text.strip(),
        "arabic_start": start_idx,
        "arabic_end": end_idx,
        "start": start,
        "end": end,
    }]


def _fallback_segments(ayah: dict, source: str = "fallback") -> list[dict]:
    words = ayah.get("words") or []
    if not words:
        return [{
            **ayah,
            "source": source,
            "translation_phrases": [],
        }]

    start_idx = 0
    end_idx = len(words) - 1
    start, end = _timed_word_span(words, start_idx, end_idx)
    translation = str(ayah.get("translation", "")).strip()
    return [{
        "surah": ayah.get("surah"),
        "ayah": ayah.get("ayah"),
        "arabic": ayah.get("arabic", ""),
        "translation": translation,
        "start": start,
        "end": end,
        "words": words,
        "source": source,
        "parent_translation": ayah.get("translation", ""),
        "translation_phrases": _fallback_phrase(translation, words, start_idx, end_idx),
    }]


def _build_prompt(ayah: dict, cfg: dict) -> str:
    max_chars = cfg.get("max_display_chars", 86)
    max_words = cfg.get("max_display_arabic_words", 12)

    payload = {
        "surah": ayah.get("surah"),
        "ayah": ayah.get("ayah"),
        "authoritative_arabic_ayah": ayah.get("arabic", ""),
        "original_bangla_translation": ayah.get("translation", ""),
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
1. Split the ayah into expression-level display units. A unit should be a
   complete understandable phrase/clause that fits on screen, not necessarily
   a full ayah. Prefer units under {max_chars} Bangla characters and about
   {max_words} Arabic timed words, but never split in a way that damages
   meaning or recitation/translation synchronization.
2. For each display unit, split the ORIGINAL Bangla translation into smaller
   contiguous Bangla phrases for karaoke highlighting. Keep the Bangla word
   order exactly as the original translation presents it. Do not reorder Bangla
   to match Arabic grammar.
3. Map each Bangla phrase to one or more Arabic timed word indices. Some Bangla
   phrases may map to multiple Arabic words and some Arabic particles may be
   absorbed into a larger Bangla phrase.
4. The phrase maps inside a display unit must be in Bangla display order and
   their Arabic ranges should be non-decreasing. If fine-grained mapping would
   require non-monotonic timing because Arabic and Bangla syntax differ, merge
   adjacent Bangla phrases into a larger phrase.

Use timed_arabic_words indices for all arabic_start/arabic_end values.
Both arabic_start and arabic_end are inclusive.

Required JSON shape:
{{
  "display_units": [
    {{
      "arabic_start": 0,
      "arabic_end": 5,
      "bangla_text": "contiguous part of the original Bangla translation",
      "translation_phrases": [
        {{"text": "Bangla phrase", "arabic_start": 0, "arabic_end": 1}},
        {{"text": "next Bangla phrase", "arabic_start": 2, "arabic_end": 5}}
      ]
    }}
  ]
}}

Few-shot example:
Input English translation:
"And Allah invites to the Home of Peace [i.e., Paradise] and guides whom He
wills to a straight path."
Good display split:
1. "And Allah invites to the Home of Peace [i.e., Paradise]"
2. "and guides whom He wills to a straight path."
Reason: these are complete expression units with a shared Arabic/translation
boundary.

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
        temperature=float(cfg.get("temperature", 0.15)),
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


def _normalize_response(ayah: dict, response: dict) -> list[dict]:
    words = ayah.get("words") or []
    if not words:
        return _fallback_segments(ayah, source="gemini_no_words")

    units = response.get("display_units") or response.get("segments") or []
    if not isinstance(units, list) or not units:
        return _fallback_segments(ayah, source="gemini_empty")

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
        if not bangla_text:
            continue

        start, end = _timed_word_span(words, a_start, a_end)
        raw_phrases = unit.get("translation_phrases") or unit.get("phrases") or []
        phrase_maps = _merge_phrase_maps(raw_phrases, {"arabic_start": a_start, "arabic_end": a_end})
        if not phrase_maps:
            phrase_maps = [{"text": bangla_text, "arabic_start": a_start, "arabic_end": a_end}]

        timed_phrases = []
        for phrase in phrase_maps:
            phrase_start, phrase_end = _timed_word_span(
                words,
                int(phrase["arabic_start"]),
                int(phrase["arabic_end"]),
            )
            timed_phrases.append({
                **phrase,
                "start": phrase_start,
                "end": phrase_end,
            })

        display_segments.append({
            "surah": ayah.get("surah"),
            "ayah": ayah.get("ayah"),
            "arabic": ayah.get("arabic", ""),
            "translation": bangla_text,
            "start": start,
            "end": end,
            "words": words[a_start:a_end + 1],
            "source": "gemini",
            "parent_translation": ayah.get("translation", ""),
            "arabic_start": a_start,
            "arabic_end": a_end,
            "translation_phrases": timed_phrases,
        })
        last_end = a_end

    return display_segments or _fallback_segments(ayah, source="gemini_invalid")


def segment_ayahs_with_gemini(
    ayah_segments: list[dict],
    cfg: dict,
    output_path: str | None = None,
) -> list[dict]:
    """Return display-level subtitle segments with phrase timing maps."""
    output = Path(output_path) if output_path else None
    if output and cfg.get("cache_results", True) and output.exists():
        logger.info("Using cached Gemini segmentation: %s", output)
        with open(output, "r", encoding="utf-8") as f:
            return json.load(f)

    if not cfg.get("enabled", False):
        logger.info("Gemini segmentation disabled; using ayah-level subtitles")
        enhanced = []
        for ayah in ayah_segments:
            enhanced.extend(_fallback_segments(ayah, source="disabled"))
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
            display_segments = _normalize_response(ayah, response)
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
            enhanced.extend(_fallback_segments(ayah, source="gemini_error"))

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
