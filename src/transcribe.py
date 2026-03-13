"""Transcribe Quran recitation audio with word-level timestamps."""

import json
import logging
import os
from pathlib import Path

# --- Must be set BEFORE importing huggingface_hub / transformers -----------
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # xet protocol stalls on Windows

# If models are already cached locally, go fully offline so huggingface_hub
# never makes a single HTTP request (cache validation, model cards, etc.).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_CACHE = _PROJECT_ROOT / "models"
if _MODEL_CACHE.exists() and any(_MODEL_CACHE.iterdir()):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
# ---------------------------------------------------------------------------

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    GenerationConfig,
    pipeline,
)

logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path: str,
    model_id: str = "tarteel-ai/whisper-base-ar-quran",
    base_model_id: str = "openai/whisper-base",
    chunk_length_s: int = 30,
    stride_length_s: int = 5,
    language: str = "ar",
) -> dict:
    """Run Whisper-based ASR and return word-level timestamps.

    Args:
        audio_path: Path to the audio file (mp3/wav).
        model_id: HuggingFace model ID for the fine-tuned Quran Whisper model.
        base_model_id: Base Whisper model (used for generation config).
        chunk_length_s: Audio chunk length for long-form transcription.
        stride_length_s: Overlap between chunks.
        language: Language code for transcription.

    Returns:
        dict with keys:
            "text": full transcription string
            "words": list of {"word": str, "start": float, "end": float}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    local_cache = str(_MODEL_CACHE)
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    logger.info("Device: %s | Model: %s | Offline: %s", device, model_id, offline)

    try:
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=local_cache)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, cache_dir=local_cache)
    except OSError:
        # Cache missing or incomplete — go online and download
        os.environ.pop("HF_HUB_OFFLINE", None)
        logger.info("Model not in local cache — downloading...")
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=local_cache)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, cache_dir=local_cache)
    model = model.to(device).eval()

    # Patch generation config (required for this fine-tuned checkpoint)
    try:
        gen_cfg = GenerationConfig.from_pretrained(base_model_id, cache_dir=local_cache)
    except OSError:
        os.environ.pop("HF_HUB_OFFLINE", None)
        gen_cfg = GenerationConfig.from_pretrained(base_model_id, cache_dir=local_cache)
    no_ts_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    gen_cfg.no_timestamps_token_id = no_ts_id
    gen_cfg.begin_suppress_tokens = [no_ts_id]
    model.generation_config = gen_cfg

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=device,
    )

    logger.info("Transcribing: %s", audio_path)

    result = asr_pipe(
        audio_path,
        return_timestamps="word",
        generate_kwargs={"language": language, "task": "transcribe"},
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
    )

    words = []
    for i, chunk in enumerate(result["chunks"]):
        start, end = chunk["timestamp"]
        if end is None:
            # Fallback: use next word's start or add 0.5s
            if i + 1 < len(result["chunks"]):
                end = result["chunks"][i + 1]["timestamp"][0]
            else:
                end = start + 0.5
        words.append({
            "word": chunk["text"].strip(),
            "start": round(start, 3),
            "end": round(end, 3),
        })

    logger.info("Transcribed %d words", len(words))
    return {"text": result["text"], "words": words}


def save_transcription(transcription: dict, output_path: str) -> str:
    """Save transcription result to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcription, f, ensure_ascii=False, indent=2)
    logger.info("Transcription saved: %s", output_path)
    return str(output_path)


def load_transcription(json_path: str) -> dict:
    """Load a previously saved transcription JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
