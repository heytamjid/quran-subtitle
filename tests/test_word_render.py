"""Exercise the full pipeline with synthetic ASR, real audio extraction and libass."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from src.pipeline import run_pipeline
from src.render import probe_video_size


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RenderIntegrationTests(unittest.TestCase):
    def test_full_word_mode_preserves_frame_and_audio_without_ayah_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=teal:s=320x568:r=25:d=1",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "libx264",
                "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(source),
            ], check=True, capture_output=True)
            words = [{"surah": 1, "ayah": 1, "position": 1, "arabic": "بِسْمِ", "translation": "নামে"},
                     {"surah": 1, "ayah": 1, "position": 2, "arabic": "اللَّهِ", "translation": "আল্লাহর"}]
            (root / "words.json").write_text(json.dumps(words, ensure_ascii=False))
            fake_asr = ModuleType("src.transcribe")
            fake_asr.transcribe_audio = lambda *a, **k: {"words": [
                {"word": "بسم", "start": .1, "end": .4}, {"word": "الله", "start": .4, "end": .9}]}
            fake_asr.save_transcription = lambda data, path: Path(path).write_text(json.dumps(data))
            cfg = {"_project_root": temporary, "transcription": {}, "matching": {"surah_hint": 1},
                   "subtitle": {}, "video": {"preset": "ultrafast"}, "paths": {"output_dir": "output"},
                   "word_by_word": {"enabled": True, "data": {"file": "words.json"}}}
            with patch.dict("sys.modules", {"src.transcribe": fake_asr}):
                final = run_pipeline(str(source), cfg)
            self.assertEqual(probe_video_size(final), (320, 568))
            output = Path(final).parent
            self.assertEqual(len(json.loads((output / "word_cues.json").read_text())), 2)
            self.assertTrue((output / "transcription.json").exists())
            # Hash the compressed audio packets: the renderer must copy audio.
            def audio_hash(path):
                return subprocess.check_output([
                    "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy",
                    "-f", "hash", "-hash", "sha256", "-"], text=True)
            self.assertEqual(audio_hash(source), audio_hash(final))
            before = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(source),
                "-map", "0:v:0", "-frames:v", "1", "-f", "hash", "-"], text=True)
            after = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", "0.2", "-i", final,
                "-map", "0:v:0", "-frames:v", "1", "-f", "hash", "-"], text=True)
            self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
