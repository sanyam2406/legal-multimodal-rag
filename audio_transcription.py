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


def transcribe_audio(audio_file: io.IOBase, language: str = "en") -> str:
    """Transcribe an audio file-like object to text.

    Parameters
    ----------
    audio_file : file-like object
        Accepts the UploadedFile returned by st.audio_input or st.file_uploader.
    language : str
        BCP-47 language code. Pass None to auto-detect.

    Returns
    -------
    str
        Transcribed text. Empty string if no speech detected.

    Raises
    ------
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
        return " ".join(seg.text for seg in segments).strip()
    except Exception as exc:
        raise RuntimeError(f"Transcription failed: {exc}") from exc

# ***REMOVED***