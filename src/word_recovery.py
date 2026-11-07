"""Opt-in acoustic recovery of suspiciously long ASR word spans.

Retries never fill canonical text by elapsed time. A measured replacement must
contain enough exact words, stay inside the original span, follow a consecutive
Quran passage, and end on the original target word. Rejected evidence is saved.
"""

import json
import logging
import math
from pathlib import Path
import subprocess

from .word_alignment import align_word_cues

logger = logging.getLogger(__name__)
DEFAULTS = {"enabled": False, "min_word_duration_s": 5.0, "context_s": 0.4,
            "retry_context_s": 1.0, "max_window_s": 12.0, "max_regions": 6,
            "max_attempts": 2, "min_exact_words": 3, "min_coverage": 0.7,
            "max_lookback_words": 16}


def _location(word):
    return word["surah"], word["ayah"], word["position"]


def recovery_settings(config: dict) -> dict:
    settings = {**DEFAULTS, **config}
    if type(settings["enabled"]) is not bool:
        raise ValueError("recovery.enabled must be true or false")
    for key in ("min_word_duration_s", "max_window_s"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"recovery.{key} must be finite and positive")
    for key in ("context_s", "retry_context_s"):
        if not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"recovery.{key} must be finite and nonnegative")
    for key in ("max_regions", "max_attempts", "min_exact_words", "max_lookback_words"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"recovery.{key} must be a positive integer")
    if settings["max_attempts"] > 2 or not 0 < settings["min_coverage"] <= 1:
        raise ValueError("recovery.max_attempts must be 1..2 and min_coverage must be in (0,1]")
    if settings["max_window_s"] > 28:
        raise ValueError("recovery.max_window_s must be at most 28 seconds to keep retries within one Whisper window")
    return settings


def _extract_window(audio_path: str, path: Path, start: float, end: float):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", str(Path(audio_path).resolve()),
                    "-t", str(end-start), "-vn", "-ac", "1", "-ar", "16000", str(path)],
                   capture_output=True, text=True, check=True)


def _engine(transcription_config: dict):
    # This import and model load occur only when recovery is explicitly enabled
    # and a region actually needs a retry. Styling/matching stays lightweight.
    from .transcribe import create_asr_pipeline, transcribe_audio
    pipe = create_asr_pipeline(
        transcription_config.get("model_id", "tarteel-ai/whisper-base-ar-quran"),
        transcription_config.get("base_model_id", "openai/whisper-base"))
    return lambda path: transcribe_audio(str(path), asr_pipe=pipe, chunk_length_s=30,
                                         stride_length_s=0, language=transcription_config.get("language", "ar"))


def _replacement(cue, candidate, canonical, settings):
    indices = {_location(word): i for i, word in enumerate(canonical)}
    clipped = []
    for word in candidate:
        # Context is used for recognition, but its words cannot replace audio
        # outside the suspicious cue. A minimum retained fraction prevents a
        # cut-off edge word from entering on a few milliseconds of overlap.
        start, end = max(word["start"], cue["start"]), min(word["end"], cue["end"])
        if end-start >= 0.04 and end-start >= (word["end"]-word["start"]) * 0.5:
            clipped.append({**word, "start": start, "end": end, "timing_source": "asr_recovery"})
    if len(clipped) <= 1:
        return None, "retry did not identify additional words"
    if _location(clipped[-1]) != _location(cue):
        return None, "retry did not return to the original target word"
    positions = [indices[_location(word)] for word in clipped]
    if any(b != a+1 for a, b in zip(positions, positions[1:])):
        return None, "retry is not a consecutive Quran sequence"
    if sum(word["match"] == "exact_normalized" for word in clipped) < settings["min_exact_words"]:
        return None, "not enough exact word evidence"
    coverage = sum(word["end"]-word["start"] for word in clipped) / (cue["end"]-cue["start"])
    if coverage < settings["min_coverage"]:
        return None, "retry covers too little of the original audio span"
    return clipped, "accepted measured consecutive sequence"


def recover_word_cues(cues: list[dict], words: list[dict], audio_path: str | None,
                      config: dict, transcription_config: dict, alignment_config: dict,
                      output_dir: str | Path, transcriber=None) -> tuple[list[dict], dict]:
    """Retry bounded regions and return reviewed replacements plus an audit log."""
    settings = recovery_settings(config)
    suspicious = [i for i, cue in enumerate(cues)
                  if cue["end"]-cue["start"] >= settings["min_word_duration_s"]]
    report = {"enabled": settings["enabled"], "settings": settings,
              "suspicious_spans": [{"cue_index": i, "start": cues[i]["start"], "end": cues[i]["end"],
                                     "arabic": cues[i]["arabic"]} for i in suspicious],
              "attempts": [], "accepted_regions": 0, "added_cues": 0}
    if not settings["enabled"] or not suspicious:
        return cues, report
    if not audio_path or not Path(audio_path).is_file():
        raise ValueError("Audio recovery is enabled: provide the source audio/video with --audio, or disable word_by_word.recovery.enabled")
    output = Path(output_dir) / "recovery"
    output.mkdir(parents=True, exist_ok=True)
    canonical = sorted([word for word in words if word["surah"] == cues[0]["surah"]],
                       key=lambda w: (w["ayah"], w["position"]))
    indices = {_location(word): i for i, word in enumerate(canonical)}
    replacements = {}
    engine = transcriber
    for index in suspicious[:settings["max_regions"]]:
        cue = cues[index]
        target = indices[_location(cue)]
        nearby = canonical[max(0, target-settings["max_lookback_words"]):target+3]
        contexts = [settings["context_s"], settings["retry_context_s"]][:settings["max_attempts"]]
        for attempt, context in enumerate(contexts, 1):
            start, end = max(0, cue["start"]-context), cue["end"]+context
            evidence = {"cue_index": index, "attempt": attempt, "start": start, "end": end, "accepted": False}
            report["attempts"].append(evidence)
            if end-start > settings["max_window_s"]:
                evidence["reason"] = "audio span exceeds max_window_s"
                continue
            logger.info("Recovery retry %d: %.2f..%.2fs (%s)", attempt, start, end, cue["arabic"])
            stem = f"cue-{index:04d}-attempt-{attempt}"
            path = output / f"{stem}.wav"
            try:
                _extract_window(audio_path, path, start, end)
                if engine is None:
                    engine = _engine(transcription_config)
                raw = engine(path)
                (output / f"{stem}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                shifted = [{**word, "start": round(word["start"]+start, 3), "end": round(word["end"]+start, 3)}
                           for word in raw["words"]]
                candidate, candidate_report = align_word_cues(shifted, nearby, cue["surah"], alignment_config)
                recovered, reason = _replacement(cue, candidate, canonical, settings)
                evidence.update({"reason": reason, "matched": candidate_report["matched"],
                                 "total": candidate_report["total"], "unmatched": candidate_report["unmatched"],
                                 "timestamp_repairs": candidate_report["timestamp_repairs"],
                                 "transcription": f"recovery/{stem}.json"})
                if recovered is not None:
                    for word in recovered:
                        word["recovery"] = {"original_cue_index": index, "original_word": cue["arabic"],
                                            "window_start": start, "transcription": f"recovery/{stem}.json"}
                    replacements[index] = recovered
                    evidence["accepted"] = True
                    report["accepted_regions"] += 1
                    report["added_cues"] += len(recovered)-1
                    break
            except Exception as exc:
                # This optional external-model boundary preserves the usable
                # original run on malformed output or provider failures. Keep
                # both an audit reason and a traceback for diagnosis.
                evidence["reason"] = f"retry failed: {exc}"
                logger.exception("Recovery retry failed; retaining original cue: %s", exc)
    report["deferred_regions"] = max(0, len(suspicious)-settings["max_regions"])
    result = []
    for index, cue in enumerate(cues):
        result.extend(replacements.get(index, [cue]))
    return result, report
