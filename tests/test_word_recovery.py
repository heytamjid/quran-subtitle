import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from src.word_recovery import recover_word_cues, recovery_settings


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        texts = ["فَأَنْبَتْنَا", "بِهِ", "حَدَائِقَ", "ذَاتَ", "بَهْجَةٍ", "مَا"]
        self.words = [{"surah": 27, "ayah": 60, "position": i+1, "arabic": text, "translation": "অর্থ"}
                      for i, text in enumerate(texts)]
        self.cues = [{**self.words[-1], "start": 10.0, "end": 16.0, "match": "exact_normalized"}]
        # Includes the actual substitution pattern found in the user's retry.
        self.raw = {"words": [{"word": text, "start": float(i), "end": float(i+1)} for i, text in enumerate(
            ["فأنبسنا", "به", "هدائق", "ذات", "بهدة", "ما"])]}

    def run_recovery(self, settings, raw=None, cues=None):
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "audio.wav"
            audio.touch()
            engine = Mock(return_value=raw or self.raw)
            with patch("src.word_recovery._extract_window") as extract:
                result, report = recover_word_cues(cues or self.cues, self.words, str(audio), settings,
                                                   {}, {}, temporary, transcriber=engine)
            return result, report, engine.call_count, extract.call_count

    def test_default_off_never_extracts_audio_or_loads_asr(self):
        result, report, calls, extracts = self.run_recovery({})
        self.assertEqual(result, self.cues)
        self.assertFalse(report["enabled"])
        self.assertEqual(len(report["suspicious_spans"]), 1)
        self.assertEqual((calls, extracts), (0, 0))

    def test_threshold_and_window_limits_do_not_launch_unnecessary_work(self):
        _, report, calls, _ = self.run_recovery({"enabled": True, "min_word_duration_s": 7})
        self.assertEqual(report["suspicious_spans"], [])
        self.assertEqual(calls, 0)
        _, report, calls, _ = self.run_recovery({"enabled": True, "max_window_s": 5})
        self.assertEqual(calls, 0)
        self.assertTrue(all("max_window_s" in a["reason"] for a in report["attempts"]))

    def test_repetition_restored_with_audio_timestamps_and_provenance(self):
        cues, report, calls, _ = self.run_recovery({"enabled": True, "context_s": 0})
        self.assertEqual([w["position"] for w in cues], [1, 2, 3, 4, 5, 6])
        self.assertEqual(report["added_cues"], 5)
        self.assertEqual(calls, 1)
        self.assertEqual(cues[0]["start"], 10.0)
        self.assertEqual(cues[-1]["end"], 16.0)
        self.assertTrue(all(w["timing_source"] == "asr_recovery" for w in cues))
        self.assertEqual(cues[0]["match"], "boundary_context_corrected")
        self.assertIn("transcription", cues[0]["recovery"])

    def test_held_note_and_missing_target_are_rejected(self):
        held = {"words": [{"word": "ما", "start": 0, "end": 6}]}
        cues, report, calls, _ = self.run_recovery({"enabled": True, "context_s": 0}, held)
        self.assertEqual(cues, self.cues)
        self.assertEqual(report["accepted_regions"], 0)
        self.assertEqual(calls, 2)
        missing = {"words": self.raw["words"][:-1]}
        cues, report, _, _ = self.run_recovery({"enabled": True, "context_s": 0}, missing)
        self.assertEqual(cues, self.cues)
        self.assertIn("target", report["attempts"][0]["reason"])

    def test_invalid_retry_timestamps_fall_back_to_original(self):
        invalid = {"words": [{"word": "ما", "start": 2, "end": 1}]}
        cues, report, _, _ = self.run_recovery({"enabled": True}, invalid)
        self.assertEqual(cues, self.cues)
        self.assertIn("retry failed", report["attempts"][0]["reason"])

    def test_region_and_attempt_budget_and_exact_evidence(self):
        two_spans = self.cues + [{**self.cues[0], "start": 20, "end": 26}]
        _, report, calls, _ = self.run_recovery({"enabled": True, "context_s": 0, "max_regions": 1}, cues=two_spans)
        self.assertEqual(calls, 1)
        self.assertEqual(report["deferred_regions"], 1)
        short = {"words": [{"word": "بهجة", "start": 0, "end": 3}, {"word": "ما", "start": 3, "end": 6}]}
        cues, report, calls, _ = self.run_recovery({"enabled": True, "context_s": 0, "max_attempts": 1}, short)
        self.assertEqual(cues, self.cues)
        self.assertEqual(calls, 1)
        self.assertIn("exact", report["attempts"][0]["reason"])

    def test_configuration_and_missing_audio_errors(self):
        for config in ({"enabled": "false"}, {"min_word_duration_s": -1}, {"max_window_s": 31}, {"max_regions": 0}):
            with self.assertRaises(ValueError):
                recovery_settings(config)
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "--audio"):
            recover_word_cues(self.cues, self.words, None, {"enabled": True}, {}, {}, temporary)


if __name__ == "__main__":
    unittest.main()
