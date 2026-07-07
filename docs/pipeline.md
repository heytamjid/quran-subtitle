# QuranSub Pipeline

This document describes the pipeline as it is currently implemented. QuranSub takes a Quran-recitation video, transcribes the Arabic audio with word timestamps, aligns the transcription to Quran ayahs, uses the corresponding Bangla translations as subtitles, and burns those subtitles into a new video.

## End-to-end data flow

```text
video
  |
  | ffmpeg: discard video, encode audio as MP3
  v
audio.mp3
  |
  | fine-tuned Whisper model: Arabic ASR with word timestamps
  v
transcription.json
  |                         Quran CSV
  |                            |
  |                            | parse ayah identifiers, Arabic text,
  |                            | and আল_বায়ান Bangla translations
  +-------------+--------------+
                |
                | normalize Arabic and run sequence alignment
                v
        matched_segments.json
                |
                | format each matched ayah as a timed subtitle
                v
       subtitles.srt + subtitles.ass
                |
                | ffmpeg ASS filter; re-encode video, copy audio
                v
   final_video_with_subtitles.mp4
```

The full pipeline is orchestrated by `src/pipeline.py` and exposed through `run.py`.

## Running the full pipeline

Install the Python packages from `requirements.txt` and ensure `ffmpeg` is available on `PATH`. A CUDA-capable PyTorch installation is optional; without CUDA, transcription runs on the CPU.

```bash
python run.py full recitation.mp4
```

For a known surah, provide its number to avoid automatic surah detection:

```bash
python run.py full recitation.mp4 --surah 20
```

The global configuration option must appear before the subcommand:

```bash
python run.py --config /path/to/config.yaml full recitation.mp4
```

### Input path resolution

For `full`, an absolute video path is used directly. A relative filename is resolved in this order:

1. `<project-root>/input/<provided-path>`, if that file exists.
2. `<project-root>/<provided-path>` otherwise.

The project root is the directory containing the loaded configuration file. If the configuration file does not exist, it is the current working directory.

### Output directory selection

`paths.output_dir` is resolved relative to the project root unless it is absolute. Each full run creates a new directory based on the input video's filename:

```text
output/recitation/
output/recitation_1/
output/recitation_2/
...
```

An existing run is never reused or overwritten by the full pipeline. The directory is created before processing begins, so a failed run can leave a partially populated output directory.

## Pipeline stages

### 1. Extract audio

`src/audio.py` verifies that the input video exists, creates the output directory, and executes:

```bash
ffmpeg -i <video> -vn -acodec libmp3lame -q:a 2 -y <audio.mp3>
```

This removes the video stream and writes a high-quality variable-bit-rate MP3. Any existing target file is overwritten. A nonzero `ffmpeg` exit status stops the pipeline and includes `ffmpeg`'s standard error in the raised exception.

Output: `audio.mp3`.

### 2. Transcribe the Arabic recitation

`src/transcribe.py` chooses `cuda` with `float16` when CUDA is available; otherwise it chooses `cpu` with `float32`. It loads:

- `transcription.model_id` as the processor and speech-to-text model.
- `transcription.base_model_id` as the generation configuration.

The base generation configuration is patched with the fine-tuned processor's `<|notimestamps|>` token ID. That token is also added to `begin_suppress_tokens`. A Hugging Face automatic-speech-recognition pipeline then processes the audio with:

- word-level timestamps;
- the configured Arabic language code and `transcribe` task;
- `chunk_length_s`-second chunks;
- `stride_length_s`-second overlap between chunks.

The default model settings are:

```yaml
transcription:
  model_id: tarteel-ai/whisper-base-ar-quran
  base_model_id: openai/whisper-base
  chunk_length_s: 30
  stride_length_s: 5
  language: ar
```

Models use the repository-level `models/` directory (the parent of `src/`) as their Hugging Face cache, independently of the selected configuration file. At module import time, if that directory exists and is nonempty, `HF_HUB_OFFLINE=1` is set. If a model or generation configuration cannot be loaded from the cache, offline mode is removed and loading is retried, allowing Hugging Face to download the missing data. `HF_HUB_DISABLE_XET=1` is also set by default.

Each ASR chunk becomes one word record. Leading and trailing whitespace is removed, and timestamps are rounded to three decimal places. If Whisper supplies no end time, the next word's start time is used; for the final word, `0.5` seconds is added.

Output: `transcription.json` with this shape:

```json
{
  "text": "full Arabic transcription",
  "words": [
    {"word": "Arabic token", "start": 0.0, "end": 0.42}
  ]
}
```

### 3. Load Quran source data

`src/quran_data.py` loads `paths.quran_csv` with pandas. A relative CSV path is resolved from the project root.

Only rows whose `ayat` value begins with the pattern `surah <number>, ayat <number>` are retained. From every retained row, the loader creates:

```json
{
  "surah": 1,
  "ayah": 1,
  "text": "Arabic text used for matching",
  "translation": "Bangla subtitle text"
}
```

The CSV's `text` cell may contain Uthmani and simpler Imlaai forms on separate lines. The loader removes blank lines and uses the last nonblank line for matching. The Bangla subtitle is read specifically from the `আল_বায়ান` column; the other translations and tafsir columns are not used by the current pipeline.

### 4. Match transcribed words to ayahs

`src/match.py` first normalizes both the transcription and Quran text. Normalization:

1. Removes ayah markers in the `﴿...﴾` form.
2. Removes Unicode combining marks, including Arabic diacritics.
3. Converts `إ`, `أ`, `آ`, and `ٱ` to `ا`.
4. Converts `ة` to `ه`.
5. Converts Farsi yeh `ی` to Arabic yeh `ي`.
6. Converts Farsi kaf `ک` to Arabic kaf `ك`.
7. Collapses whitespace and trims the result.

Empty normalized transcription entries are discarded, while an index mapping to the original timestamped entries is retained.

#### Surah selection

If `matching.surah_hint` or `--surah` is set, only ayahs from that exact surah number are candidates. The CLI value overrides the configuration value.

Without a hint, the matcher groups all normalized Quran words by surah. For each surah it counts how many normalized transcription entries occur in that surah's word set. The surah with the highest count wins. This is exact normalized-token membership, not a fuzzy score; ties are resolved by the first surah encountered in CSV order.

#### Sequence alignment

All words in the selected surah are flattened into one ordered list, with each word retaining a reference to its source ayah. `difflib.SequenceMatcher` then compares the ordered transcription list with that Quran list using `autojunk=False`.

Only words inside exact matching blocks receive an ayah reference initially. Gaps are filled in two passes:

1. Forward fill assigns each unmatched word to the most recent matched ayah.
2. Backward fill assigns any still-unmatched leading words to the next matched ayah.

If the alignment has no exact matching block, every mapping remains empty and no segments are produced.

Finally, consecutive mapped transcription words with the same ayah reference are grouped. A segment starts at the first original word's start time and ends at the last original word's end time. Its text is the selected ayah's `আল_বায়ান` translation. The Arabic transcription itself is not included in the generated subtitle.

Output: `matched_segments.json`:

```json
[
  {
    "surah": 20,
    "ayah": 1,
    "translation": "Bangla ayah translation",
    "start": 1.24,
    "end": 4.91
  }
]
```

The full pipeline stops with `No matching ayahs found` if this list is empty.

### 5. Generate SRT and ASS subtitles

`src/subtitle.py` writes two subtitle formats from the same segment list:

- `subtitles.srt` contains numbered entries, millisecond timestamps, and plain Bangla translation text.
- `subtitles.ass` contains centisecond timestamps and a configurable `Default` style used for rendering.

The ASS script has a fixed design resolution of `1920x1080`, uses wrap style `0`, and is encoded as UTF-8 with a byte-order mark for player compatibility. Newlines in translation text become ASS `\N` line breaks.

The following `subtitle` settings affect only ASS output:

- font name and size;
- primary, outline, and background colors in ASS color syntax;
- outline width and shadow depth;
- bold state;
- ASS alignment number;
- left, right, and vertical margins.

### 6. Burn subtitles into the video

`src/render.py` verifies the source video and ASS file, then runs `ffmpeg` from the ASS file's directory. Passing only the ASS filename to the filter avoids Windows drive-letter parsing problems.

The effective command is:

```bash
ffmpeg -i <absolute-video> \
  -vf ass=subtitles.ass \
  -c:v <video.codec> \
  -crf <video.crf> \
  -preset <video.preset> \
  -c:a copy -y \
  <absolute-output>
```

The video is re-encoded, by default with `libx264`, CRF `18`, and preset `medium`. The original audio stream is copied without re-encoding. Existing target output is overwritten. Font availability is handled by the operating system and `ffmpeg`/libass; the configured font must be installed for predictable rendering.

Output: `final_video_with_subtitles.mp4`.

## Files produced by a full run

```text
output/<video-name>/
├── audio.mp3
├── transcription.json
├── matched_segments.json
├── subtitles.srt
├── subtitles.ass
└── final_video_with_subtitles.mp4
```

These intermediate files make it possible to inspect transcription and matching results before rendering.

## Running stages independently

Every processing stage also has a CLI command:

```bash
python run.py extract-audio input/video.mp4 -o output/audio.mp3
python run.py transcribe output/audio.mp3 -o output/transcription.json
python run.py match output/transcription.json --surah 20 -o output/matched_segments.json
python run.py subtitle output/matched_segments.json -o output
python run.py render input/video.mp4 --subtitle output/subtitles.ass -o output/final.mp4
```

These commands do not use the full pipeline's input lookup or unique per-video output-directory logic. Their defaults are command-specific:

| Command | Default output |
| --- | --- |
| `extract-audio` | Input path with its suffix changed to `.mp3` |
| `transcribe` | `output/transcription.json` |
| `match` | `output/matched_segments.json` |
| `subtitle` | `output/subtitles.srt` and `output/subtitles.ass` |
| `render` | `output/final_video_with_subtitles.mp4` |

## Configuration behavior

The default configuration path is `config.yaml`. If it exists, it is loaded as-is; the loader does not merge it with defaults or validate its schema. Consequently, the full pipeline expects at least `transcription`, `matching`, `subtitle`, and `paths` mappings with the required path entries. Individual values generally have code-level defaults when retrieved with `dict.get`.

If the configuration file does not exist, `run.py` supplies a complete fallback configuration with default transcription, matching, path, subtitle, and video mappings. Command-line overrides are applied in memory and do not modify the YAML file.

## Current matching constraints

The current implementation is deterministic but intentionally simple:

- Surah detection uses exact normalized word membership, so common Quranic words can influence the result and word frequency within a candidate surah is not considered.
- Ayah alignment uses exact normalized sequence blocks. It does not use phonetic similarity, edit distance per token, semantic similarity, or an alignment confidence threshold.
- Gap filling assigns ASR errors and extra speech to the nearest matched context even when those words did not match Quran text.
- Timing boundaries come entirely from Whisper word timestamps; there is no silence detection or boundary refinement.
- Only one surah is selected for an entire transcription, so a video spanning multiple surahs is not modeled explicitly.
- Only the Bangla `আল_বায়ান` translation is emitted; Arabic text and ayah labels are not rendered.
- A wrong or nonexistent surah hint is not validated explicitly and normally leads to an empty match result.

These constraints are important when interpreting `matched_segments.json`: it is an alignment result derived from exact token sequences and gap propagation, not a separately verified recitation index.
