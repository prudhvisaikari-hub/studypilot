"""
ASR (speech-to-text) module for StudyPilot.

Two main backends:
  - MockASR:      reads a pre-existing transcript (or does silence->text stub).
                  Zero downloads, zero dependencies beyond stdlib. Use this to
                  develop/test the rest of the pipeline (summarizer, RAG, UI)
                  without needing model weights or a microphone.
  - WhisperASR:   real on-device transcription using faster-whisper (CTranslate2).
"""

from __future__ import annotations
import os
import time
import json
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


class ASRInitError(Exception):
    """Raised when an ASR backend fails to initialize."""
    pass


class ASRTranscriptionError(Exception):
    """Raised when transcription fails during execution."""
    pass


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str = "Speaker 1"


class BaseASR:
    def transcribe(self, audio_path: str, enable_diarization: bool = False) -> List[TranscriptSegment]:
        raise NotImplementedError


class MockASR(BaseASR):
    """No-download stand-in. If `audio_path` points at a .txt file, treats each
    line as one segment with fabricated timestamps (5s apart). This lets you
    exercise the full app today with a sample lecture transcript instead of
    a real audio file + model weights."""

    def transcribe(self, audio_path: str, enable_diarization: bool = False) -> List[TranscriptSegment]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        segments = []
        t = 0.0
        current_speaker = "Speaker 1"
        with open(audio_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if enable_diarization and idx > 0 and idx % 4 == 0:
                    current_speaker = "Speaker 2" if current_speaker == "Speaker 1" else "Speaker 1"
                segments.append(
                    TranscriptSegment(
                        start=round(t, 2),
                        end=round(t + 5.0, 2),
                        text=line,
                        speaker=current_speaker if enable_diarization else "Speaker 1"
                    )
                )
                t += 5.0
        return segments


class WhisperASR(BaseASR):
    """Real transcription via faster-whisper.
    Includes robust error handling for missing models, empty audio, short/long clips,
    and heuristic speaker diarization.
    """

    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading WhisperModel({model_size}, device={device}, compute_type={compute_type})...")
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info("WhisperModel initialized successfully.")
        except Exception as e:
            raise ASRInitError(
                f"Failed to load Whisper ASR model '{model_size}'. "
                f"Ensure internet access for initial download or valid model name. Error: {str(e)}"
            ) from e

    def transcribe(self, audio_path: str, enable_diarization: bool = False) -> List[TranscriptSegment]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file or transcript not found: {audio_path}")

        # If audio_path is a text file (e.g., sample transcript used with whisper mode), return graceful parse
        if audio_path.endswith(".txt"):
            mock = MockASR()
            return mock.transcribe(audio_path, enable_diarization=enable_diarization)

        # Check file size (0 bytes = empty audio)
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.warning("Empty audio file provided (0 bytes). Returning empty transcript.")
            return []

        start_time = time.time()
        try:
            segments_raw, info = self.model.transcribe(audio_path, beam_size=3, vad_filter=True)
            segment_list = list(segments_raw)
        except Exception as e:
            logger.error(f"Whisper transcription failed for {audio_path}: {e}")
            raise ASRTranscriptionError(f"Transcription error: {str(e)}") from e

        elapsed = time.time() - start_time
        audio_duration = getattr(info, "duration", 0.0)
        logger.info(
            f"Transcribed {audio_duration:.2f}s audio in {elapsed:.2f}s "
            f"(speedup: {audio_duration / max(elapsed, 0.001):.2f}x)"
        )

        if not segment_list:
            logger.info("No speech detected in audio clip.")
            return []

        # Process segments & optional heuristic speaker diarization
        results: List[TranscriptSegment] = []
        current_speaker_idx = 1
        prev_end = 0.0

        for seg in segment_list:
            text = seg.text.strip()
            if not text:
                continue

            # Simple pause-based turn taking heuristic for speaker attribution
            if enable_diarization:
                pause_duration = seg.start - prev_end
                if prev_end > 0 and pause_duration > 1.5:
                    current_speaker_idx = 2 if current_speaker_idx == 1 else 1

            speaker_label = f"Speaker {current_speaker_idx}"
            results.append(
                TranscriptSegment(
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    text=text,
                    speaker=speaker_label,
                )
            )
            prev_end = seg.end

        return results


class SnapdragonASR(BaseASR):
    """Placeholder for the QAIRT/Hexagon-NPU-compiled model."""

    def __init__(self, compiled_model_path: str = ""):
        self.compiled_model_path = compiled_model_path
        raise NotImplementedError(
            "Snapdragon NPU compilation requires physical Snapdragon hardware "
            "and Qualcomm AI Hub runtime."
        )

    def transcribe(self, audio_path: str, enable_diarization: bool = False) -> List[TranscriptSegment]:
        raise NotImplementedError


def get_asr_backend(name: str = "mock", **kwargs) -> BaseASR:
    name = name.lower()
    if name == "mock":
        return MockASR()
    if name == "whisper":
        model_size = kwargs.get("model_size", "tiny")
        return WhisperASR(model_size=model_size)
    if name == "snapdragon":
        return SnapdragonASR(**kwargs)
    raise ValueError(f"Unknown ASR backend: {name}")


def segments_to_transcript_json(segments: List[TranscriptSegment]) -> str:
    return json.dumps([s.__dict__ for s in segments], indent=2)
