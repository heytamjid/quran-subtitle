# QuranSub Project Documentation

## Overview
QuranSub generates Bangla subtitles for Quran recitation videos and burns them into the final video.

Pipeline flow:
1. Extract audio from video
2. Transcribe Arabic recitation with word timestamps
3. Load Quran ayah + translation dataset
4. Match transcribed words to ayahs
5. Generate subtitle files (`.srt` and `.ass`)
6. Render final subtitled video

## Project Structure
```text
QuranSub/
|-- run.py
|-- config.yaml
|-- requirements.txt
|-- src/
|   |-- audio.py
|   |-- transcribe.py
|   |-- quran_data.py
|   |-- match.py
|   |-- subtitle.py
|   |-- render.py
|   |-- pipeline.py
|   `-- __init__.py
|-- data/
|   `-- ben_quran_with_tafsir.csv
|-- input/
|-- output/
`-- docs/
```

## Module Responsibilities

### `run.py`
CLI entry point. Defines all subcommands and dispatches to module functions.

Responsibilities:
- Load configuration (`config.yaml`)
- Parse command arguments
- Run full pipeline or specific stage commands

### `src/pipeline.py`
Orchestrates the complete end-to-end workflow.

Responsibilities:
- Resolve input video path (prefers `input/<video_name>`)
- Create unique output folder per run
- Execute all six processing stages in sequence
- Save stage artifacts (`audio.mp3`, `transcription.json`, `matched_segments.json`, subtitles, final video)

### `src/audio.py`
Audio extraction layer.

Responsibilities:
- Extract audio from input video using `ffmpeg`
- Save audio as MP3 (default)

### `src/transcribe.py`
Speech-to-text layer for Quran recitation.

Responsibilities:
- Load Whisper-based Quran model from Hugging Face
- Support offline cache behavior when models already exist locally
- Produce word-level timestamps
- Save/load transcription JSON

### `src/quran_data.py`
Quran dataset loader and parser.

Responsibilities:
- Read Quran CSV using pandas
- Parse surah/ayah metadata
- Choose simplified Arabic line (Imlaai) for matching
- Attach Bangla translation text per ayah

### `src/match.py`
Text normalization and alignment engine.

Responsibilities:
- Normalize Arabic text (remove diacritics and variant forms)
- Auto-detect best matching surah (or use hint)
- Align transcribed words to Quran words via sequence matching
- Build timed subtitle segments by ayah
- Save/load matched segment JSON

### `src/subtitle.py`
Subtitle file generation layer.

Responsibilities:
- Generate `.srt` subtitles
- Generate styled `.ass` subtitles based on `config.yaml`

### `src/render.py`
Final video rendering layer.

Responsibilities:
- Burn `.ass` subtitles into video using `ffmpeg`
- Preserve source audio stream
- Write final encoded video output

### `src/__init__.py`
Package marker file for the `src` module.

## Configuration
Main configuration file: `config.yaml`

Key sections:
- `subtitle`: font, size, color, alignment, margins, outline/shadow
- `transcription`: model IDs, chunk length, stride, language
- `matching`: optional `surah_hint`
- `video`: codec, CRF, preset
- `paths`: Quran CSV path and output directory

## Commands

### Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Full Pipeline
Run full pipeline for a video:
```powershell
python run.py full taha.mp4
```

Run with surah hint:
```powershell
python run.py full taha.mp4 --surah 20
```

Choose custom output directory:
```powershell
python run.py full taha.mp4 --output-dir output
```

### Stage-by-Stage Commands
Extract audio only:
```powershell
python run.py extract-audio input\video.mp4
```

Transcribe audio only:
```powershell
python run.py transcribe output\audio.mp3
```

Match transcription to ayahs:
```powershell
python run.py match output\transcription.json
python run.py match output\transcription.json --surah 20
```

Generate subtitles from matched segments:
```powershell
python run.py subtitle output\matched_segments.json
```

Burn subtitles into video:
```powershell
python run.py render input\video.mp4 --subtitle output\subtitles.ass
```

### CLI Help
```powershell
python run.py --help
python run.py <subcommand> --help
```

## Inputs and Outputs

Input:
- Place source videos in `input/` (or pass absolute/relative file path)

Output:
- Generated outputs are written under `output/`
- Full pipeline creates a unique subfolder per video (for example `output/taha/`, `output/taha_1/`, etc.)

## External Requirements
- `ffmpeg` must be installed and available in system `PATH`
- Python dependencies from `requirements.txt`
- Optional CUDA GPU for faster transcription (`CPU` fallback is supported)
