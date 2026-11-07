# Arabic / Bangla word carousel

The default full pipeline displays each recited Arabic word above its Bangla
meaning in a pastel panel over the bottom of the source frame. It preserves the
video's dimensions, duration and audio. At each word timestamp the strip moves
from right to left, bringing the active word to the center, then holds still.
It contains no grammatical labels. The next and previous word pairs provide
context around the active pair.

## Run

Install `requirements.txt` in your Python environment and have `ffmpeg` and
`ffprobe` on PATH. Install **Noto Naskh Arabic** and **Noto Sans Bengali** for
the default fonts. Linux font discovery uses `fc-match`. On other systems use
installed font filenames or absolute `font_path` entries under each typography
section, and set `video.fonts_dir` to the folder containing these fonts for
libass. Font family names must match the font's internal names.

```bash
python run.py full input/recitation.mp4 --surah 27
```

The first run downloads the pinned word dataset (about 12 MB) into
`data/word_cache/`; subsequent runs verify checksums and work offline. The first
transcription also needs the configured Whisper models. Those models and
PyTorch are much larger downloads. Model cache behavior is described in
[pipeline.md](pipeline.md).

The ayah mode remains available with `--mode ayah` or
`word_by_word.enabled: false`.

Audio recovery is **enabled in the supplied configuration**, using bounded retries
and conservative acceptance checks. You can also explicitly enable it for a run:

```bash
python run.py full input/recitation.mp4 --surah 27 --recovery
```

`--no-recovery` overrides an enabled configuration. Spelling normalization and
conservative contextual corrections work independently of the audio retry switch.

## Stages and files

```text
video -> audio.mp3 -> Whisper word timestamps (transcription.json)
                                 |
              canonical Arabic / Bangla word pairs
                                 |
          word_alignment.py: recording-relative timing + Quran positions
                                 |
                 word_cues.json + alignment_report.json
                                 |
          word_overlay.py: layout, typography, palette, motion
                                 |
                    subtitles.ass + subtitles.srt
                                 |
              ffmpeg/libass -> final_video_with_subtitles.mp4
```

`src/word_data.py` handles the provider and cache. `src/word_alignment.py` handles
word identities and timing without layout knowledge. `src/word_overlay.py`
handles presentation without network access or transcription. The reusable
stages in `src/word_pipeline.py` connect them. `src/render.py` probes the video's
display dimensions (including rotation) and burns ASS into the existing frame.

Outputs are placed in a unique run directory as before. `word_cues.json` stores
the canonical `surah`, `ayah`, `position`, `arabic`, Bangla `translation`,
`start`, `end`, original `transcribed` token and source attribution. You can edit
this file to correct timings or meanings before rerendering. `subtitles.srt`
contains timed Arabic/Bangla pairs; the animation and panel are in the ASS file.
Each cue also records `match`, `timing_source` and `transcription_indices`.
Split tokens share one combined ASR span; contextual corrections retain the
original word's span. Recovered cues reference their retry JSON in `recovery`.

`alignment_report.json` distinguishes matched ASR-token count from final
`cue_count`, lists corrections and gaps in the identified Quran sequence, and
summarizes audio recovery. An ASR omission can be absent from the ASR-token
count altogether, so a 100% matched count alone does not guarantee full coverage.

To match an existing transcription and iterate on the design without loading
Whisper again:

```bash
python run.py word-match output/recitation/transcription.json --surah 27 -o output/restyle
python run.py word-subtitle output/restyle/word_cues.json --video input/recitation.mp4 -o output/restyle
python run.py render input/recitation.mp4 --subtitle output/restyle/subtitles.ass -o output/restyle/final.mp4
```

To reuse a saved transcription with optional audio retries:

```bash
python run.py word-match output/recitation/transcription.json --surah 27 --audio input/recitation.mp4 --recovery -o output/recovered
python run.py word-subtitle output/recovered/word_cues.json --video input/recitation.mp4 -o output/recovered
python run.py render input/recitation.mp4 --subtitle output/recovered/subtitles.ass -o output/recovered/final.mp4
```

For a design canvas without a video, replace `--video ...` with
`--size 720x1280`. Regenerate ASS with the real source size before rendering.

## Independent design controls

All word settings live under `word_by_word` in `config.yaml`. Legacy `subtitle`
settings apply only to ayah subtitles.

| Section | Controls |
| --- | --- |
| `data` | Cache directory or normalized JSON source override |
| `alignment` | Contextual correction switch, edit-distance and similarity limits, beam width |
| `recovery` | Audio retry switch, duration threshold, context, window and retry budgets, acceptance thresholds |
| `layout` | Panel height, margins, corner radius, card widths, gap, text positions, horizontal center |
| `typography.arabic` | Arabic font family, optional font path, size |
| `typography.bangla` | Bangla font family, optional font path, size, maximum wrapped lines |
| `palette` | Background, lavender/sage accents, focus fill, active and neighboring text, opacity |
| `motion` | Transition duration, easing, animation subdivisions, silence threshold, final-word hold |

Layout distances and font sizes scale with the shorter frame dimension divided
by `layout.reference_size` (default 720). `center_x` is a fraction of frame
width, default 0.5. Bangla glosses wrap and shrink to fit their cards without
discarding text. Out-of-frame panels and insufficient vertical space fail with
a useful error instead of silently cropping the active translation.

`motion.duration_ms` defaults to 180. Each transition starts at the incoming
word's timestamp and settles within the first 45% of the interval at most.
`smoothstep` easing eases in and out; `linear` is also supported. Changing
`motion.fps` changes the ASS easing subdivisions, not the output video frame rate.
Long gaps beyond `pause_threshold` hide the strip after `hold_after_word` seconds
and restart centered on the next word. At shorter pauses the strip stays still.

## Data and attribution

Arabic words, word IDs, Bangla meanings and surah metadata come from
[tezcane/data-quran](https://github.com/tezcane/data-quran), whose
[README](https://github.com/tezcane/data-quran#word-by-word) attributes its word
translations to [Quran WBW](https://quranwbw.com/). The pinned revision is
`b17fc7745091497d394f509e6b1fc21bf946732c`. All 77,429 pairs are joined by word
ID, and global ayah IDs are converted to surah/ayah references using the same
revision's surah metadata. The text field used has standard Unicode Arabic;
the alternate field contains private font glyphs unsuitable for general fonts.

The application does not generate, machine-translate or split ayah translations
to obtain word meanings. It does not bundle or relicense the downloaded source
data. Consult the original publishers for reuse/distribution terms.

To substitute another reviewed source, set `word_by_word.data.file` to a JSON
array in this normalized shape (relative to the configuration directory, or an
absolute path):

```json
[
  {"surah": 1, "ayah": 1, "position": 1,
   "arabic": "بِسْمِ", "translation": "নামে"}
]
```

Positions are one-based and unique per ayah; Arabic and Bangla must be nonempty.
Cache paths also resolve relative to the configuration directory.

## Timing and matching limits

The matcher uses normalized Arabic with explicit Uthmani/dagger-alif spelling
variants. It searches for a coherent sequence with a bounded beam, allowing
backward jumps for repeated words or ayahs. Without a surah hint, exact ordered
sequence evidence selects one surah for the recording. Provide `--surah` when
known, especially for short clips with common words.

Spelling equivalences cover Uthmani initial hamza, dagger alif, alif maqsura and
the Quranic spelling of الليل. They preserve distinctions such as ماء versus ما.
Two adjacent ASR tokens can map to one canonical word (أم + من -> أمّن), using
their combined timestamp span. By default, a small ASR substitution can be
corrected when exact neighboring words identify the missing canonical position.
`alignment.context_corrections: false` turns these corrections off. Defaults
allow one character edit and require similarity of at least 0.6; short ambiguous
words are excluded from approximate repair. Boundary repairs require two
position-consistent anchors, without chaining guessed words into new evidence.

Unmatched speech and canonical gaps remain in `alignment_report.json`; the
matcher does not populate absent words or invent timing. Incorrect ASR words
that happen to match Quran words can still align incorrectly, and word boundaries
are only as accurate as Whisper's output. One run currently selects one surah.
Timestamp records must be finite, ordered, nonnegative and non-overlapping.

Small ASR boundary overlaps are reconciled before matching: by default the
maximum is 80 ms, further limited to 25% of either neighboring word's duration.
The two words share the midpoint of their original overlapping boundaries;
neither word is dropped. `alignment.max_timestamp_overlap_s` sets the tolerance
(0 restores strict rejection). Negative/zero durations, backwards timestamps,
nested intervals and larger overlaps still fail for review. Repairs are logged
in `alignment_report.json.timestamp_repairs`, and affected cues use
`timing_source: asr_boundary_adjusted` with the adjustment details. The saved
raw transcription retains the original model output.

## Optional recovery of missing repetitions

Whisper can collapse a reciter's repeated phrase into one occurrence. In the
investigated `input/1.mp4`, it assigned 35.60–43.22 seconds to a single مَا,
omitting a second فأنبتنا به حدائق ذات بهجة. A short audio retry recovered the
second occurrence using the same cached model. A standalone 30-second retry
still omitted it, so cross-window stitching is not the only source of this error.

The recovery module retries suspiciously long cue spans with short surrounding
audio. It loads one reusable ASR engine only when enabled and needed. It accepts
a replacement only if it contains a consecutive Quran sequence ending at the
original target word, enough exact matches, and sufficient audio coverage.
Replacement timestamps are clipped to the original span to protect surrounding
cues. The retry can also use the same bounded contextual word corrections.

```yaml
word_by_word:
  recovery:
    enabled: true
    min_word_duration_s: 5.0
    context_s: 0.4
    retry_context_s: 1.0
    max_window_s: 12.0
    max_regions: 6
    max_attempts: 2
    min_exact_words: 3
    min_coverage: 0.7
    max_lookback_words: 16
```

Lower `min_word_duration_s` to inspect shorter spans; raise it to reduce retries
on long held notes. Context values control padding on the first and second
attempt. `max_window_s` caps each crop and must be at most 28 seconds to stay
inside one Whisper window. Oversized spans are logged rather than processed.
`max_regions` and `max_attempts` bound runtime. Acceptance thresholds control
how much exact text and span coverage is required. Normal held words that remain
single words on retry are left intact. Set `enabled` to false to skip all audio
extraction and retry-model loading; suspicious spans are still reported.

Every run writes `recovery_report.json`. Enabled retries additionally save the
audio crops and raw model results under `recovery/`, including rejected attempts.
`timing_source: asr_recovery` distinguishes accepted retry timings. Original ASR
token indices refer to the initial transcription; recovered indices refer to
the retry JSON named in the cue's `recovery` object. Failed or ambiguous retries
retain the original usable cue.

This is bounded ASR recovery, not forced acoustic alignment. Its acceptance
checks reduce false replacements but cannot prove every word was spoken or
catch repetitions that have no long-span symptom. No linear interpolation is
used. For higher timing accuracy, the next step is an audio aligner using the
identified Quran text and repetition structure, with confidence-based review.

## Verification and reference preview

```bash
python -m unittest discover -s tests -v
```

Tests cover partial ayahs, repetitions, unmatched speech, spelling equivalences,
split tokens, contextual repairs and their off switch, surah selection, invalid
timestamps, offline data, motion, silence and layout. Recovery tests exercise its
off switch, thresholds, budgets, accepted repeats, rejected held notes and bad
retry output. Real FFmpeg integration verifies rendering and audio preservation.

`output/reference-preview/preview.mp4` is a 9.8-second design sample made from
the supplied reference. The reference's filmed region was cropped to remove
its existing instructional graphics; the production renderer itself does not
crop. Sample timings were set by hand against the reference, not produced by
Whisper. Its editable cues and ASS are in the same output directory.

The installed cached Whisper model was subsequently tested on CPU against the
user's real output files and audio crops. Corrected versions of the two clips
are under `output/1_recovered/` and `output/2_recovered/`; the former includes
the recovered five-word repetition, and the latter matches all 50 ASR words
after spelling fixes. Published-model candidates for a future comparison include
[quran-asr-small](https://huggingface.co/Abdou/quran-asr-small) for a small ONNX
backend and [Quran Whisper Turbo](https://huggingface.co/naazimsnh02/whisper-large-v3-turbo-ar-quran)
for a larger Whisper backend. Neither replacement model has been benchmarked
on these clips; reported metrics use different evaluation sets and are not a
guarantee of improvement here.
