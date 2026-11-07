"""Reconcile small adjacent ASR-boundary overlaps, retaining an audit trail."""

import math


def reconcile_word_timestamps(words: list[dict], max_overlap_s: float = 0.08) -> tuple[list[dict], list[dict]]:
    """Share a midpoint only for small overlaps between otherwise ordered words.

    No words are removed and no missing durations are fabricated. A repair may
    affect at most 25% of either word's duration. Invalid, nested, backwards or
    larger overlaps still fail with the offending values for review.
    """
    if not math.isfinite(max_overlap_s) or max_overlap_s < 0:
        raise ValueError("max_timestamp_overlap_s must be finite and nonnegative")
    result = []
    repairs = []
    for index, source in enumerate(words):
        word = {**source}
        start, end = float(word["start"]), float(word["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Invalid word timestamps at transcription index {index}: start={start}, end={end}")
        word.update(start=start, end=end)
        result.append(word)
        if index == 0:
            continue
        previous = result[-2]
        overlap = previous["end"] - start
        if overlap <= 1e-9:
            continue
        limit = min(max_overlap_s, 0.25 * min(previous["end"]-previous["start"], end-start))
        if start < previous["start"] or end < previous["end"] or overlap > limit + 1e-9:
            raise ValueError(
                f"Invalid/overlapping word timestamps at transcription index {index}: "
                f"previous end={previous['end']:.6f}, start={start:.6f}, end={end:.6f}; "
                f"overlap={overlap:.6f}s exceeds safe repair or timestamps are out of order")
        boundary = round((previous["end"] + start) / 2, 6)
        if not previous["start"] < boundary < end:
            raise ValueError(f"Cannot safely reconcile word timestamps at transcription index {index}")
        repair = {"previous_index": index-1, "index": index,
                  "previous_word": previous["word"], "word": word["word"],
                  "previous_end": previous["end"], "start": start,
                  "overlap_s": round(overlap, 6), "shared_boundary": boundary}
        for item in (previous, word):
            item.setdefault("asr_timestamp", {"start": item["start"], "end": item["end"]})
            item["timestamp_adjustments"] = [*item.get("timestamp_adjustments", []), repair]
        previous["end"] = boundary
        word["start"] = boundary
        repairs.append(repair)
    return result, repairs
