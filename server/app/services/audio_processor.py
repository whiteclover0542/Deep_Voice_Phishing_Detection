import subprocess
import numpy as np
from app.core.config import settings


def convert_to_pcm(raw_audio: bytes) -> bytes:
    """FFmpeg으로 입력 오디오를 PCM 16kHz mono 16-bit로 변환"""
    command = [
        "ffmpeg",
        "-f", "s16le",
        "-ar", str(settings.sample_rate),
        "-ac", "1",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ar", str(settings.sample_rate),
        "-ac", "1",
        "pipe:1",
    ]

    result = subprocess.run(
        command,
        input=raw_audio,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

    return result.stdout


def pcm_to_numpy(pcm_data: bytes) -> np.ndarray:
    """PCM bytes → float32 numpy array (faster-whisper 입력 형식)"""
    audio = np.frombuffer(pcm_data, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0
