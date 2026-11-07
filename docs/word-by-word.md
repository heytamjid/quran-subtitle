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

To match an existing transcription and iterate on the design without loading
Whisper again:

```bash
python run.py word-match output/recitation/transcription.json --surah 27 -o output/restyle
python run.py word-subtitle output/restyle/word_cues.json --video input/recitation.mp4 -o output/restyle
python run.py render input/recitation.mp4 --subtitle output/restyle/subtitles.ass -o output/restyle/final.mp4
```

For a design canvas without a video, replace `--video ...` with
`--size 720x1280`. Regenerate ASS with the real source size before rendering.

## Independent design controls

All word settings live under `word_by_word` in `config.yaml`. Legacy `subtitle`
settings apply only to ayah subtitles.

| Section | Controls |
| --- | --- |
| `data` | Cache directory or normalized JSON source override |
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

Only matching words receive meanings; unmatched speech is listed in
`alignment_report.json`. No timestamp or missing recitation is interpolated.
Incorrect ASR words that happen to match Quran words can still align
incorrectly, and word boundaries are only as accurate as Whisper's output.
This is not forced audio alignment or verified recitation recognition. Review
the saved cues for a final publication. One run currently selects one surah.
Timestamp records must be finite, ordered, nonnegative and non-overlapping.

## Verification and reference preview

```bash
python -m unittest discover -s tests -v
```

Tests cover partial ayahs, repetitions, unmatched speech, dagger alif,
surah selection, invalid timestamps, an offline source override, right-to-left
motion, silence handling and invalid layout. A local FFmpeg preview was also
rendered and its Arabic/Bangla text and transition frames visually checked.

`output/reference-preview/preview.mp4` is a 9.8-second design sample made from
the supplied reference. The reference's filmed region was cropped to remove
its existing instructional graphics; the production renderer itself does not
crop. Sample timings were set by hand against the reference, not produced by
Whisper. Its editable cues and ASS are in the same output directory. The full
Whisper-to-video path has not been run in this environment because PyTorch and
Transformers are not installed.
