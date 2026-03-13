"""Load and structure Quran verse data from the CSV."""

import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_AYAT_PATTERN = re.compile(r"surah\s+(\d+),\s*ayat\s+(\d+)")


def load_quran_data(csv_path: str) -> list[dict]:
    """Load the Quran CSV and return a list of ayah dicts.

    Each dict: {"surah": int, "ayah": int, "text": str, "translation": str}
    The "text" field uses the simpler (Imlaai) script line when available.
    """
    df = pd.read_csv(csv_path)

    ayahs = []
    for _, row in df.iterrows():
        m = _AYAT_PATTERN.match(str(row["ayat"]))
        if not m:
            continue

        surah = int(m.group(1))
        ayah_num = int(m.group(2))

        raw_text = str(row["text"])
        # The text field often has two lines: Uthmani script then Imlaai script.
        # Use the last line (simpler) for better matching with Whisper output.
        lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
        simple_text = lines[-1] if lines else raw_text

        translation = str(row["আল_বায়ান"]).strip()

        ayahs.append({
            "surah": surah,
            "ayah": ayah_num,
            "text": simple_text,
            "translation": translation,
        })

    logger.info("Loaded %d ayahs from %s", len(ayahs), csv_path)
    return ayahs
