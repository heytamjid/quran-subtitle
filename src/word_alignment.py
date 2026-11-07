"""Map ASR timestamps to canonical word positions; never invent word meanings."""

from collections import defaultdict
import difflib
import logging
import math
import unicodedata

from .match import normalize_arabic
from .word_timing import reconcile_word_timestamps

logger = logging.getLogger(__name__)


def word_key(text: str) -> str:
    text = normalize_arabic(unicodedata.normalize("NFKC", text)).replace("ى", "ي")
    return "".join(c for c in text if unicodedata.category(c).startswith("L")
                   and c != "ـ" and not "\u06d6" <= c <= "\u06ed")


def _script_keys(text: str) -> set[str]:
    # Uthmani spelling sometimes represents Imlaai alif with a dagger alif,
    # and sometimes omits it in the common spelling (e.g. الرحمن). Keep both
    # explicit forms rather than broadly deleting alifs from unrelated words.
    keys = {word_key(text), word_key(text.replace("ٰ", "ا")),
            word_key(text.replace("وٰ", "ا").replace("ٰ", "ا")),
            word_key(text.replace("ىٰ", "ا").replace("ٰ", "ا"))}
    # Initial standalone hamza in Uthmani corresponds to a seated hamza in
    # Imlaai. Do not delete every hamza: that would conflate ماء and ما.
    for key in list(keys):
        if key.startswith("ءا"):
            keys.add(key[1:])
        if key.startswith("اء"):
            keys.add("اا" + key[2:])
        # The Uthmani spelling of الليل writes one lam with shadda. Restrict
        # this equivalence to this lexeme; الذين must not acquire another lam.
        if key == "اليل":
            keys.add("الليل")
    return keys


def _expanded_shadda(text: str) -> str:
    result = []
    for char in text:
        if char == "ّ":
            previous = next((c for c in reversed(result) if unicodedata.category(c).startswith("L")), "")
            result.append(previous)
        else:
            result.append(char)
    return "".join(result)


def _distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
        row = new
    return row[-1]


def _similar(raw: str, canonical: str, config: dict) -> bool:
    key = word_key(raw)
    for target in _script_keys(canonical):
        distance = _distance(key, target)
        if distance == 0:
            return True
        length = max(len(key), len(target))
        if min(len(key), len(target)) >= 3 and distance <= config.get("max_edit_distance", 1) and 1 - distance / length >= config.get("min_similarity", 0.6):
            return True
    return False


def align_word_cues(transcribed: list[dict], words: list[dict], surah_hint: int | None = None,
                    alignment_config: dict | None = None) -> tuple[list[dict], dict]:
    """Beam alignment with continuity costs and backward jumps for repetitions.

    Exact spelling aliases and split ASR words are aligned first. Small ASR
    substitutions can then be repaired between exact neighboring anchors.
    Every cue retains its source ASR indices, match reason and timing origin.
    No absent words or estimated timestamps are inserted by this stage.
    """
    config = alignment_config or {}
    beam_width = int(config.get("beam_width", 32))
    if beam_width < 1 or config.get("max_edit_distance", 1) < 0 or not 0 <= config.get("min_similarity", 0.6) <= 1:
        raise ValueError("Invalid word alignment settings")
    transcribed, timestamp_repairs = reconcile_word_timestamps(
        transcribed, config.get("max_timestamp_overlap_s", 0.08))
    for repair in timestamp_repairs:
        logger.warning("Reconciled %.0f ms ASR overlap at word %d (%s / %s); shared boundary %.3fs",
                       repair["overlap_s"]*1000, repair["index"], repair["previous_word"],
                       repair["word"], repair["shared_boundary"])
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
        return [], {"matched": 0, "total": len(transcribed), "unmatched": transcribed,
                    "timestamp_repairs": timestamp_repairs}
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
            token_set = set(tokens)
            for expand in (False, True):
                keys = [(next((key for key in sorted(_script_keys(w["arabic"])) if key in token_set), word_key(w["arabic"])))
                        if expand else word_key(w["arabic"]) for w in groups[surah]]
                blocks = difflib.SequenceMatcher(None, tokens, keys, autojunk=False).get_matching_blocks()
                scores.append(sum(b.size ** 2 for b in blocks))
            return max(scores)
        surah_hint = max(groups, key=score)
    canonical = groups[surah_hint]
    locations, split_locations = defaultdict(list), defaultdict(list)
    for index, word in enumerate(canonical):
        for key in _script_keys(word["arabic"]):
            locations[key].append(index)
        for key in _script_keys(word["arabic"]) | _script_keys(_expanded_shadda(word["arabic"])):
            split_locations[key].append(index)
    # Indexed beams allow two ASR tokens to consume one canonical word (e.g.
    # أم + من -> أمّن). Separate normal words still score better separately.
    beams = [{} for _ in range(len(usable) + 1)]
    beams[0][None] = (0.0, None)

    def add(index, last, score, node):
        previous = beams[index].get(last)
        if previous is None or score > previous[0]:
            beams[index][last] = (score, node)

    for offset, (original_index, item, key) in enumerate(usable):
        states = sorted(beams[offset].items(), key=lambda pair: pair[1][0], reverse=True)[:beam_width]
        options = [(1, locations.get(key, []), "exact_normalized")]
        if offset + 1 < len(usable) and usable[offset+1][0] == original_index + 1:
            combined = word_key(item["word"] + usable[offset+1][1]["word"])
            options.append((2, split_locations.get(combined, []), "split_token"))
        for last, (score, node) in states:
            add(offset+1, last, score-1.5, (node, [original_index], None, "unmatched"))
            for count, positions, kind in options:
                for pos in positions:
                    if last is None:
                        cost = 0.0
                    elif pos == last + 1:
                        cost = 0.0
                    elif pos > last:
                        cost = 0.3 + 0.2 * (pos - last - 1)
                    else:
                        # An isolated common word is insufficient evidence of
                        # a backward jump; require a second word in sequence.
                        following = offset + count
                        if pos < last and not (following < len(usable) and pos+1 < len(canonical)
                                and usable[following][2] in _script_keys(canonical[pos+1]["arabic"])):
                            continue
                        cost = 1.0 + 0.08 * (last - pos)
                    indices = [usable[offset+j][0] for j in range(count)]
                    add(offset+count, pos, score+2.0*count-cost-(0.2 if count == 2 else 0),
                        (node, indices, pos, kind))
        # Earlier beams are no longer needed; tracebacks retain selected nodes.
        beams[offset].clear()
    assignments = {}
    node = max(beams[-1].values(), key=lambda state: state[0])[1]
    while node:
        node, indices, position, kind = node
        if position is not None:
            assignments[indices[0]] = (position, indices, kind)

    if config.get("context_corrections", True):
        anchors = sorted(assignments)
        # Only original anchors justify repairs, never a chain of guesses.
        for left, right in zip(anchors, anchors[1:]):
            left_pos, left_indices, _ = assignments[left]
            right_pos = assignments[right][0]
            missing_indices = list(range(left_indices[-1]+1, right))
            expected = list(range(left_pos+1, right_pos))
            if not missing_indices or len(missing_indices) != len(expected) or len(expected) > 3:
                continue
            if all(_similar(transcribed[i]["word"], canonical[pos]["arabic"], config)
                   for i, pos in zip(missing_indices, expected)):
                for i, pos in zip(missing_indices, expected):
                    assignments[i] = (pos, [i], "context_corrected")
        # A repeated passage can restart with one misrecognized word. Two
        # following exact anchors justify that word without a forward-only
        # interpolation across the backward jump.
        for left, right, following in zip(anchors, anchors[1:], anchors[2:]):
            left_pos, left_indices, _ = assignments[left]
            right_pos = assignments[right][0]
            missing = left_indices[-1]+1
            if (right == missing+1 and following == right+1 and 0 < right_pos < left_pos
                    and assignments[following][0] == right_pos+1
                    and _similar(transcribed[missing]["word"], canonical[right_pos-1]["arabic"], config)):
                assignments[missing] = (right_pos-1, [missing], "repeat_context_corrected")
        # At a cropped audio boundary, require two exact position-consistent anchors
        # and one similar ASR token; never manufacture an unrecited first/last word.
        if len(anchors) >= 2:
            first, second = anchors[:2]
            first_pos = assignments[first][0]
            if (first == 1 and first_pos > 0 and second-first <= 3
                    and assignments[second][0]-first_pos == second-first):
                if _similar(transcribed[0]["word"], canonical[first_pos-1]["arabic"], config):
                    assignments[0] = (first_pos-1, [0], "boundary_context_corrected")
            penultimate, last = anchors[-2:]
            last_pos, last_indices, _ = assignments[last]
            if (last_indices[-1] == len(transcribed)-2 and last_pos+1 < len(canonical)
                    and last-penultimate <= 3 and last_pos-assignments[penultimate][0] == last-penultimate):
                i = len(transcribed)-1
                if _similar(transcribed[i]["word"], canonical[last_pos+1]["arabic"], config):
                    assignments[i] = (last_pos+1, [i], "boundary_context_corrected")
    cues = []
    covered = set()
    for index in sorted(assignments):
        position, indices, kind = assignments[index]
        covered.update(indices)
        cues.append({**canonical[position], "start": float(transcribed[indices[0]]["start"]),
                     "end": float(transcribed[indices[-1]]["end"]),
                     "transcribed": " ".join(transcribed[i]["word"] for i in indices),
                     "transcription_indices": indices, "match": kind, "timing_source": "asr"})
        adjustments = {repair["index"]: repair for i in indices
                       for repair in transcribed[i].get("timestamp_adjustments", [])}
        if adjustments:
            cues[-1]["timing_source"] = "asr_boundary_adjusted"
            cues[-1]["timestamp_adjustments"] = list(adjustments.values())
    positions = {(w["surah"], w["ayah"], w["position"]): i for i, w in enumerate(canonical)}
    gaps = []
    for left, right in zip(cues, cues[1:]):
        a = positions[(left["surah"], left["ayah"], left["position"])]
        b = positions[(right["surah"], right["ayah"], right["position"])]
        if b > a+1:
            gaps.append({"after": left["end"], "before": right["start"],
                         "words": canonical[a+1:b]})
    report = {"surah": surah_hint, "matched": len(covered), "total": len(transcribed), "cue_count": len(cues),
              "unmatched": [{"index": i, **w} for i, w in enumerate(transcribed) if i not in covered],
              "corrections": [cue for cue in cues if cue["match"] != "exact_normalized"],
              "timestamp_repairs": timestamp_repairs,
              "canonical_gaps": gaps}
    logger.info("Word alignment: %d/%d ASR tokens -> %d cues, surah %d", len(covered), len(transcribed), len(cues), surah_hint)
    if report["unmatched"]:
        logger.warning("%d words have no meaning assigned; review alignment_report.json", len(report["unmatched"]))
    return cues, report
