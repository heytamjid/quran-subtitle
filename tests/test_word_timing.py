import copy
import unittest

from src.word_alignment import align_word_cues
from src.word_timing import reconcile_word_timestamps


class WordTimingTests(unittest.TestCase):
    def test_actual_twenty_millisecond_overlap_keeps_both_words_and_original_input(self):
        raw = [{"word": "الله", "start": 44.42, "end": 45.22},
               {"word": "خالق", "start": 45.20, "end": 46.04}]
        original = copy.deepcopy(raw)
        fixed, repairs = reconcile_word_timestamps(raw)
        self.assertEqual(raw, original)
        self.assertEqual(fixed[0]["end"], 45.21)
        self.assertEqual(fixed[1]["start"], 45.21)
        self.assertEqual(fixed[0]["asr_timestamp"]["end"], 45.22)
        self.assertEqual(fixed[1]["asr_timestamp"]["start"], 45.20)
        self.assertEqual(repairs[0]["overlap_s"], .02)
        self.assertEqual(reconcile_word_timestamps(fixed)[1], [])

    def test_large_nested_or_invalid_intervals_still_fail(self):
        cases = [
            [{"word": "a", "start": 0, "end": 1}, {"word": "b", "start": .5, "end": 2}],
            [{"word": "a", "start": 0, "end": 1}, {"word": "b", "start": .95, "end": .99}],
            [{"word": "a", "start": 1, "end": 1}],
            [{"word": "a", "start": float("nan"), "end": 1}],
            [{"word": "a", "start": -.1, "end": 1}],
            # A 60 ms overlap of 80 ms words is not a small boundary repair.
            [{"word": "a", "start": 0, "end": .08}, {"word": "b", "start": .02, "end": .10}],
        ]
        for words in cases:
            with self.subTest(words=words), self.assertRaises(ValueError):
                reconcile_word_timestamps(words)

    def test_tolerance_can_be_lowered_or_disabled(self):
        raw = [{"word": "a", "start": 0, "end": 1}, {"word": "b", "start": .98, "end": 2}]
        for tolerance in (0, .01):
            with self.assertRaises(ValueError):
                reconcile_word_timestamps(raw, tolerance)
        self.assertEqual(len(reconcile_word_timestamps(raw, .03)[1]), 1)

    def test_matcher_audits_adjustment_and_preserves_word_positions(self):
        words = [{"surah": 13, "ayah": 16, "position": i+1, "arabic": text, "translation": "অর্থ"}
                 for i, text in enumerate(["اللَّهُ", "خَالِقُ"])]
        raw = [{"word": "الله", "start": 44.42, "end": 45.22},
               {"word": "خالق", "start": 45.20, "end": 46.04}]
        cues, report = align_word_cues(raw, words, 13)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0]["end"], cues[1]["start"])
        self.assertEqual(len(report["timestamp_repairs"]), 1)
        self.assertTrue(all(cue["timing_source"] == "asr_boundary_adjusted" for cue in cues))


if __name__ == "__main__":
    unittest.main()
