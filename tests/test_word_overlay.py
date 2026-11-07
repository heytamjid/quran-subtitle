import json
from pathlib import Path
import tempfile
import unittest

from src.word_alignment import align_word_cues
from src.word_data import validate_words
from src.word_overlay import generate_word_ass
from src.word_pipeline import match_words, word_subtitles


def word(position, arabic, bangla="অর্থ", ayah=1, surah=1):
    return {"surah": surah, "ayah": ayah, "position": position,
            "arabic": arabic, "translation": bangla}


def transcript(text):
    return [{"word": token, "start": i * 0.7, "end": i * 0.7 + 0.5}
            for i, token in enumerate(text.split())]


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.words = [word(1, "بِسْمِ"), word(2, "اللَّهِ"), word(3, "الرَّحْمَٰنِ"),
                      word(4, "الرَّحِيمِ"), word(1, "الْحَمْدُ", ayah=2), word(2, "لِلَّهِ", ayah=2)]

    def test_repeated_passage_and_unknown_speech(self):
        original = transcript("بسم الله الرحمن الرحيم مرحبا بسم الله الرحمن الرحيم")
        cues, report = align_word_cues(original, self.words, 1)
        self.assertEqual([w["position"] for w in cues], [1, 2, 3, 4, 1, 2, 3, 4])
        self.assertEqual(report["unmatched"][0]["word"], "مرحبا")
        self.assertEqual(cues[4]["start"], original[5]["start"])

    def test_partial_ayah_does_not_invent_unrecited_words(self):
        cues, _ = align_word_cues(transcript("الرحمن الرحيم"), self.words, 1)
        self.assertEqual([w["position"] for w in cues], [3, 4])

    def test_noisy_first_word_of_a_repeat_uses_following_exact_anchors(self):
        raw = transcript("بسم الله الرحمن الرحيم بسن الله الرحمن الرحيم")
        cues, report = align_word_cues(raw, self.words, 1)
        self.assertEqual([w["position"] for w in cues], [1, 2, 3, 4, 1, 2, 3, 4])
        self.assertEqual(cues[4]["match"], "repeat_context_corrected")
        self.assertEqual(report["unmatched"], [])

    def test_dagger_alif_and_automatic_surah(self):
        words = self.words + [word(1, "وَسَلَٰمٌ", surah=27), word(2, "عَلَىٰ", surah=27),
                              word(3, "عِبَادِهِ", surah=27)]
        cues, report = align_word_cues(transcript("وسلام على عباده"), words)
        self.assertEqual(report["surah"], 27)
        self.assertEqual(len(cues), 3)

    def test_invalid_timing_and_surah_fail(self):
        with self.assertRaises(ValueError):
            align_word_cues([{"word": "الله", "start": 2, "end": 1}], self.words, 1)
        with self.assertRaises(ValueError):
            align_word_cues(transcript("الله"), self.words, 115)

    def test_uthmani_hamza_maqsura_and_layl_equivalences(self):
        words = [word(1, "ءَاخِذِينَ"), word(2, "ءَاتَىٰهُمۡ"), word(3, "ٱلَّيۡلِ"), word(4, "ءَايَٰتٞ")]
        cues, report = align_word_cues(transcript("آخذين آتاهم الليل آيات"), words, 1)
        self.assertEqual(report["matched"], 4)
        self.assertEqual([w["position"] for w in cues], [1, 2, 3, 4])
        # Removing every hamza would incorrectly match water to 'what'.
        cues, _ = align_word_cues(transcript("ما"), [word(1, "مَاءٌ")], 1)
        self.assertEqual(cues, [])

    def test_split_asr_word_keeps_combined_span_and_position(self):
        words = [word(1, "يُشْرِكُونَ", ayah=1), word(1, "أَمَّنۡ", ayah=2),
                 word(2, "خَلَقَ", ayah=2), word(3, "السَّمَاوَاتِ", ayah=2), word(4, "مِنَ", ayah=2)]
        raw = transcript("يشركون أم من خلق السماوات")
        cues, report = align_word_cues(raw, words, 1)
        self.assertEqual([(w["ayah"], w["position"]) for w in cues], [(1, 1), (2, 1), (2, 2), (2, 3)])
        self.assertEqual(cues[1]["match"], "split_token")
        self.assertEqual(cues[1]["start"], raw[1]["start"])
        self.assertEqual(cues[1]["end"], raw[2]["end"])
        self.assertEqual(report["matched"], 5)

    def test_context_corrections_can_be_disabled_and_do_not_fill_absent_tokens(self):
        words = [word(1, "لِلَّهِ"), word(2, "وَسَلَٰمٌ"), word(3, "عَلَىٰ")]
        raw = transcript("لله وزلام على")
        cues, _ = align_word_cues(raw, words, 1)
        self.assertEqual(cues[1]["arabic"], "وَسَلَٰمٌ")
        self.assertEqual(cues[1]["match"], "context_corrected")
        self.assertEqual(cues[1]["timing_source"], "asr")
        cues, report = align_word_cues(raw, words, 1, {"context_corrections": False})
        self.assertEqual(len(cues), 2)
        self.assertEqual(len(report["unmatched"]), 1)
        cues, report = align_word_cues(transcript("لله على"), words, 1)
        self.assertEqual(len(cues), 2)
        self.assertEqual(report["canonical_gaps"][0]["words"][0]["position"], 2)


class PresentationTests(unittest.TestCase):
    def test_modular_pipeline_from_offline_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = [word(1, "بِسْمِ", "নামে"), word(2, "اللَّهِ", "আল্লাহর")]
            (root / "data.json").write_text(json.dumps(data, ensure_ascii=False))
            cfg = {"_project_root": temporary, "word_by_word": {"data": {"file": "data.json"}},
                   "matching": {"surah_hint": 1}}
            cues = match_words({"words": transcript("بسم الله")}, cfg, root / "output")
            path = word_subtitles(cues, cfg, root / "output", size=(720, 1280))
            text = Path(path).read_text(encoding="utf-8-sig")
            self.assertIn("PlayResX: 720\nPlayResY: 1280", text)
            self.assertIn("\\move(", text)
            self.assertIn("নামে", text)
            self.assertTrue((root / "output/alignment_report.json").exists())
            # New cues start at center + distance, end at the center: movement
            # follows recitation order from right to left, regardless of script.
            import re
            moves = re.findall(r"\\move\(([\d.-]+),[\d.-]+,([\d.-]+),", text)
            self.assertTrue(moves)
            self.assertTrue(all(float(x2) <= float(x1) for x1, x2 in moves))

    def test_pause_hides_panel_and_restart_does_not_scroll(self):
        cues = [{**word(1, "بسم"), "start": 0, "end": 1},
                {**word(2, "الله"), "start": 5, "end": 6}]
        with tempfile.TemporaryDirectory() as temporary:
            path = generate_word_ass(cues, {}, str(Path(temporary) / "words.ass"), 1920, 1080)
            text = Path(path).read_text(encoding="utf-8-sig")
            self.assertIn("0:00:01.35", text)
            self.assertNotIn("\\move(", text)
            self.assertIn("PlayResX: 1920\nPlayResY: 1080", text)

    def test_duplicate_locations_and_out_of_bounds_layout_rejected(self):
        with self.assertRaises(ValueError):
            validate_words([word(1, "الله"), word(1, "الله")])
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            generate_word_ass([{**word(1, "الله"), "start": 0, "end": 1}],
                              {"layout": {"height": 2000}}, str(Path(temporary) / "bad.ass"), 720, 1280)


if __name__ == "__main__":
    unittest.main()
