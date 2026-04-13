import io
import os

import numpy as np
import soundfile as sf
import streamlit as st
from faster_whisper import WhisperModel

_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")


@st.cache_resource(show_spinner="Loading Whisper model...")
def _load_model() -> WhisperModel:
    return WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8")


def _audio_to_float32(audio_file: io.IOBase) -> np.ndarray:
    audio_file.seek(0)
    data, _ = sf.read(audio_file, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


def compute_audio_hash(audio_bytes: bytes) -> int:
    """Return a hash of raw audio bytes for deduplication."""
    return hash(audio_bytes)


def transcribe_audio(audio_file: io.IOBase, language: str = "en") -> str:
    """Transcribe an audio file-like object to text.

    Raises
    ------
    ValueError
        If no speech is detected in the audio.
    RuntimeError
        Wraps any transcription error with a user-friendly message.
    """
    try:
        audio_array = _audio_to_float32(audio_file)
        model = _load_model()
        segments, _ = model.transcribe(
            audio_array,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(seg.text for seg in segments).strip()
        if not text:
            raise ValueError("No speech detected. Please try again.")
        return text
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Transcription failed: {exc}") from exc