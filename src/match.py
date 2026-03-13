"""Match transcribed Arabic words to Quran ayahs via sequence alignment."""

import difflib
import json
import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

# Ayah number markers: ﴿١﴾ style
_AYAH_MARKER_RE = re.compile(r"\uFD3F[^\uFD3E]*\uFD3E")


def normalize_arabic(text: str) -> str:
    """Strip diacritics, ayah markers, and normalize letter variants."""
    # Remove ayah markers  ﴿...﴾
    text = _AYAH_MARKER_RE.sub("", text)
    # Strip all Unicode combining marks (diacritics, tashkeel, etc.)
    text = "".join(
        c for c in text
        if unicodedata.category(c) not in ("Mn", "Mc", "Me")
    )
    # Normalize alef variants  → ا
    text = re.sub(r"[إأآٱ]", "ا", text)
    # Normalize taa marbuta → haa
    text = text.replace("ة", "ه")
    # Normalize Farsi yaa → Arabic yaa
    text = text.replace("\u06CC", "\u064A")
    # Normalize Farsi kaf → Arabic kaf
    text = text.replace("\u06A9", "\u0643")
    # Collapse whitespace
    return " ".join(text.split()).strip()


def _score_surah(trans_words: list[str], surah_words: list[str]) -> float:
    """Score how well transcribed words match a surah's words."""
    surah_set = set(surah_words)
    return sum(1 for w in trans_words if w in surah_set)


def find_matching_surah(trans_words: list[str], quran_ayahs: list[dict]) -> int:
    """Identify which surah the transcription most likely belongs to."""
    # Group words by surah
    surahs: dict[int, list[str]] = {}
    for ayah in quran_ayahs:
        s = ayah["surah"]
        words = normalize_arabic(ayah["text"]).split()
        surahs.setdefault(s, []).extend(words)

    best_surah = None
    best_score = -1
    for surah_num, surah_words in surahs.items():
        score = _score_surah(trans_words, surah_words)
        if score > best_score:
            best_score = score
            best_surah = surah_num

    logger.info(
        "Best matching surah: %d (score %d / %d words)",
        best_surah, best_score, len(trans_words),
    )
    return best_surah


def align_words_to_ayahs(
    transcribed_words: list[dict],
    quran_ayahs: list[dict],
    surah_hint: int | None = None,
) -> list[dict]:
    """Align transcribed words to Quran ayahs using sequence matching.

    Args:
        transcribed_words: list of {"word": str, "start": float, "end": float}
        quran_ayahs: list of {"surah", "ayah", "text", "translation"}
        surah_hint: optional surah number to narrow search

    Returns:
        list of subtitle segments:
        [{"surah": int, "ayah": int, "translation": str,
          "start": float, "end": float}, ...]
    """
    # --- 1. Normalize transcribed words, keep mapping to originals -----------
    trans_norm = []      # normalized word strings
    trans_orig_idx = []  # index into transcribed_words
    for i, w in enumerate(transcribed_words):
        n = normalize_arabic(w["word"])
        if n:
            trans_norm.append(n)
            trans_orig_idx.append(i)

    if not trans_norm:
        logger.warning("No usable transcribed words after normalization")
        return []

    # --- 2. Determine surah --------------------------------------------------
    if surah_hint:
        candidates = [a for a in quran_ayahs if a["surah"] == surah_hint]
        logger.info("Using surah hint: %d (%d ayahs)", surah_hint, len(candidates))
    else:
        best_surah = find_matching_surah(trans_norm, quran_ayahs)
        candidates = [a for a in quran_ayahs if a["surah"] == best_surah]
        logger.info("Auto-detected surah %d (%d ayahs)", best_surah, len(candidates))

    # --- 3. Build flat Quran word list with ayah back-references -------------
    quran_norm = []       # normalized Quran word strings
    quran_ayah_ref = []   # index into candidates
    for ci, ayah in enumerate(candidates):
        words = normalize_arabic(ayah["text"]).split()
        for w in words:
            quran_norm.append(w)
            quran_ayah_ref.append(ci)

    # --- 4. Sequence alignment -----------------------------------------------
    matcher = difflib.SequenceMatcher(None, trans_norm, quran_norm, autojunk=False)
    matching_blocks = matcher.get_matching_blocks()

    # Map each transcribed word to a candidate ayah index (or None)
    trans_ayah_map: list[int | None] = [None] * len(trans_norm)
    for block in matching_blocks:
        for k in range(block.size):
            ti = block.a + k
            qi = block.b + k
            trans_ayah_map[ti] = quran_ayah_ref[qi]

    matched_count = sum(1 for x in trans_ayah_map if x is not None)
    logger.info(
        "Alignment: %d / %d transcribed words matched",
        matched_count, len(trans_norm),
    )

    # --- 5. Fill gaps (assign unmatched words to nearest matched ayah) -------
    # Forward fill
    last = None
    for i in range(len(trans_ayah_map)):
        if trans_ayah_map[i] is not None:
            last = trans_ayah_map[i]
        elif last is not None:
            trans_ayah_map[i] = last
    # Backward fill
    last = None
    for i in range(len(trans_ayah_map) - 1, -1, -1):
        if trans_ayah_map[i] is not None:
            last = trans_ayah_map[i]
        elif last is not None:
            trans_ayah_map[i] = last

    # --- 6. Group by ayah and compute timestamps -----------------------------
    segments = []
    cur_ci = None
    cur_start = None
    cur_end = None

    for ni, ci in enumerate(trans_ayah_map):
        if ci is None:
            continue
        orig = transcribed_words[trans_orig_idx[ni]]

        if ci != cur_ci:
            # Flush previous segment
            if cur_ci is not None:
                ayah = candidates[cur_ci]
                segments.append({
                    "surah": ayah["surah"],
                    "ayah": ayah["ayah"],
                    "translation": ayah["translation"],
                    "start": cur_start,
                    "end": cur_end,
                })
            cur_ci = ci
            cur_start = orig["start"]
            cur_end = orig["end"]
        else:
            cur_end = orig["end"]

    # Flush last segment
    if cur_ci is not None:
        ayah = candidates[cur_ci]
        segments.append({
            "surah": ayah["surah"],
            "ayah": ayah["ayah"],
            "translation": ayah["translation"],
            "start": cur_start,
            "end": cur_end,
        })

    logger.info("Produced %d ayah subtitle segments", len(segments))
    for seg in segments:
        logger.info(
            "  Surah %d Ayah %d  [%.2fs - %.2fs]  %s",
            seg["surah"], seg["ayah"], seg["start"], seg["end"],
            seg["translation"][:60],
        )

    return segments


def save_segments(segments: list[dict], output_path: str) -> str:
    """Save matched segments to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.info("Segments saved: %s", output_path)
    return str(output_path)


def load_segments(json_path: str) -> list[dict]:
    """Load previously saved segments."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
