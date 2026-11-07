"""Small reusable stages for word data, timing and presentation."""

import json
from pathlib import Path

from .word_alignment import align_word_cues
from .word_data import load_word_data
from .render import probe_video_size
from .subtitle import generate_srt


def match_words(transcription: dict, config: dict, output_dir: str | Path) -> list[dict]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = config.get("word_by_word", {})
    words = load_word_data(settings.get("data", {}), config.get("_project_root", "."))
    cues, report = align_word_cues(transcription["words"], words, config.get("matching", {}).get("surah_hint"))
    for filename, data in (("word_cues.json", cues), ("alignment_report.json", report)):
        (output / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if not cues:
        raise RuntimeError("No Quran words matched. Check the transcription and surah hint; see alignment_report.json.")
    return cues


def word_subtitles(cues: list[dict], config: dict, output_dir: str | Path,
                   video_path: str | None = None, size: tuple[int, int] | None = None) -> str:
    from .word_overlay import generate_word_ass

    if size is None:
        if video_path is None:
            raise ValueError("Provide a source video or explicit frame dimensions")
        size = probe_video_size(video_path)
    output = Path(output_dir)
    generate_srt([{**cue, "translation": cue["arabic"] + "\n" + cue["translation"]} for cue in cues],
                 str(output / "subtitles.srt"))
    return generate_word_ass(cues, config.get("word_by_word", {}), str(output / "subtitles.ass"), *size)
