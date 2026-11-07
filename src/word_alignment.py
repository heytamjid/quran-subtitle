"""Map ASR timestamps to canonical word positions; never invent word meanings."""

from collections import defaultdict
import difflib
import logging
import math
import unicodedata

from .match import normalize_arabic

logger = logging.getLogger(__name__)


def word_key(text: str) -> str:
    text = normalize_arabic(unicodedata.normalize("NFKC", text)).replace("ى", "ي")
    return "".join(c for c in text if unicodedata.category(c).startswith("L")
                   and c != "ـ" and not "\u06d6" <= c <= "\u06ed")


def _script_keys(text: str) -> set[str]:
    # Uthmani spelling sometimes represents Imlaai alif with a dagger alif,
    # and sometimes omits it in the common spelling (e.g. الرحمن). Keep both
    # explicit forms rather than broadly deleting alifs from unrelated words.
    return {word_key(text), word_key(text.replace("ٰ", "ا")),
            word_key(text.replace("وٰ", "ا").replace("ٰ", "ا"))}


def align_word_cues(transcribed: list[dict], words: list[dict], surah_hint: int | None = None) -> tuple[list[dict], dict]:
    """Beam alignment with continuity costs and backward jumps for repetitions.

    Only exact normalized Arabic matches receive meanings. Skipped speech is
    reported, not forward-filled. Timestamps always come from the recording.
    """
    usable = []
    previous_end = 0.0
    for index, item in enumerate(transcribed):
        start, end = float(item["start"]), float(item["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < previous_end or end <= start:
            raise ValueError(f"Invalid/overlapping word timestamps at transcription index {index}")
        previous_end = end
        if word_key(item["word"]):
            usable.append((index, item, word_key(item["word"])))
    if not usable:
        return [], {"matched": 0, "total": len(transcribed), "unmatched": transcribed}
    groups = defaultdict(list)
    for word in words:
        groups[word["surah"]].append(word)
    if surah_hint is not None and surah_hint not in groups:
        raise ValueError(f"No word data for surah {surah_hint}")
    if surah_hint is None:
        tokens = [key for _, _, key in usable]
        def score(surah):
            # Score both Unicode spellings so dagger-alif words contribute to
            # automatic surah selection as well as to word alignment.
            scores = []
            for expand in (False, True):
                keys = [word_key(w["arabic"].replace("ٰ", "ا") if expand else w["arabic"]) for w in groups[surah]]
                blocks = difflib.SequenceMatcher(None, tokens, keys, autojunk=False).get_matching_blocks()
                scores.append(sum(b.size ** 2 for b in blocks))
            return max(scores)
        surah_hint = max(groups, key=score)
    canonical = groups[surah_hint]
    locations = defaultdict(list)
    for index, word in enumerate(canonical):
        for key in _script_keys(word["arabic"]):
            locations[key].append(index)
    # State = score, last canonical position, linked traceback node.
    beam = [(0.0, None, None)]
    for original_index, item, key in usable:
        candidates = {}
        for score, last, node in beam:
            options = [(score - 1.5, last, (node, original_index, None))]
            for pos in locations.get(key, []):
                if last is None:
                    cost = 0.0
                elif pos == last + 1:
                    cost = 0.0
                elif pos > last:
                    cost = 0.3 + 0.2 * (pos - last - 1)
                else:
                    # Repeated words/ayahs can return to an earlier position.
                    cost = 1.0 + 0.08 * (last - pos)
                options.append((score + 2.0 - cost, pos, (node, original_index, pos)))
            for option in options:
                if option[1] not in candidates or option[0] > candidates[option[1]][0]:
                    candidates[option[1]] = option
        beam = sorted(candidates.values(), key=lambda state: state[0], reverse=True)[:24]
    assignments = {}
    node = beam[0][2]
    while node:
        node, original_index, position = node
        if position is not None:
            assignments[original_index] = position
    cues = []
    for index, item in enumerate(transcribed):
        if index in assignments:
            cues.append({**canonical[assignments[index]], "start": float(item["start"]),
                         "end": float(item["end"]), "transcribed": item["word"], "match": "exact_normalized"})
    report = {"surah": surah_hint, "matched": len(cues), "total": len(transcribed),
              "unmatched": [{"index": i, **w} for i, w in enumerate(transcribed) if i not in assignments]}
    logger.info("Word alignment: %d/%d words, surah %d", len(cues), len(transcribed), surah_hint)
    if report["unmatched"]:
        logger.warning("%d words have no meaning assigned; review alignment_report.json", len(report["unmatched"]))
    return cues, report
