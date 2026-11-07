"""Pinned Quran WBW Arabic/Bangla data, joined by stable word IDs.

Network access is limited to the first load. A custom normalized JSON file can
replace this provider without changing alignment or presentation.
"""

import hashlib
import json
import logging
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)
REVISION = "b17fc7745091497d394f509e6b1fc21bf946732c"
BASE_URL = f"https://raw.githubusercontent.com/tezcane/data-quran/{REVISION}"
FILES = {
    "words.json": ("word/word.json", "1834f15513ec4206c08a845ebb91d4c9b8e835633a458cafbd64810e399ab3b9"),
    "bn.json": ("word-translation/bn-word.json", "49bec45d323c52c2c4080d4e17237460b85c8d24208a395dac3a969bdff1f418"),
    "surahs.json": ("surah/surah.json", "b7f36823208154643489dcc96f245832dd09a6c1a9c92963bf456128beca80c4"),
}


def _read_cached(cache: Path, filename: str) -> dict:
    remote, digest = FILES[filename]
    path = cache / filename
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Quran WBW data: %s", remote)
        try:
            with urlopen(Request(f"{BASE_URL}/{remote}", headers={"User-Agent": "QuranSub/3"}), timeout=60) as response:
                payload = response.read()
        except OSError as exc:
            raise RuntimeError(f"Cannot download {remote}. Connect once or configure word_by_word.data.file for offline data.") from exc
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch for {remote}")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"Invalid cached data: {path}. Remove it and fetch again.")
    return json.loads(payload)


def validate_words(words: list[dict]) -> list[dict]:
    seen = set()
    for word in words:
        key = (int(word["surah"]), int(word["ayah"]), int(word["position"]))
        if not 1 <= key[0] <= 114 or min(key[1:]) < 1 or key in seen:
            raise ValueError(f"Invalid or duplicate Quran word location: {key}")
        if not word.get("arabic", "").strip() or not word.get("translation", "").strip():
            raise ValueError(f"Missing Arabic/Bangla text at {key}")
        seen.add(key)
    if not words:
        raise ValueError("Word dataset is empty")
    return sorted(words, key=lambda w: (w["surah"], w["ayah"], w["position"]))


def load_word_data(config: dict, project_root: str | Path = ".") -> list[dict]:
    """Return {surah, ayah, position, arabic, translation, source} records."""
    root = Path(project_root)
    if config.get("file"):
        path = Path(config["file"])
        if not path.is_absolute():
            path = root / path
        return validate_words(json.loads(path.read_text(encoding="utf-8")))
    cache = Path(config.get("cache_dir", "data/word_cache"))
    if not cache.is_absolute():
        cache = root / cache
    arabic = _read_cached(cache, "words.json")
    bangla = _read_cached(cache, "bn.json")
    surahs = _read_cached(cache, "surahs.json")
    verse_locations = {}
    for number, surah in surahs.items():
        for global_ayah in range(surah["start"], surah["end"] + 1):
            verse_locations[global_ayah] = (int(number), global_ayah - surah["start"] + 1)
    words = []
    for word_id, word in arabic.items():
        surah, ayah = verse_locations[word["ayah"]]
        # This field uses standard Unicode Quranic Arabic, without the private
        # font glyphs and detached prefixes in the other script field.
        words.append({"surah": surah, "ayah": ayah, "position": word["position"],
                      "arabic": word["nastaliq"], "translation": bangla.get(word_id, ""),
                      "source": f"Quran WBW / tezcane/data-quran@{REVISION}", "word_id": word_id})
    result = validate_words(words)
    logger.info("Loaded %d Arabic/Bangla word pairs", len(result))
    return result
