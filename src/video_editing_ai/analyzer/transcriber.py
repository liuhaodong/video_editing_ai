"""Timestamped dialogue and lyrics extraction using faster-whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..models.schema import DialogueSegment, MediaCatalog

_WHISPER_MODEL_CACHE: dict[str, object] = {}


def _get_whisper_model(model_size: str = "small"):
    if model_size not in _WHISPER_MODEL_CACHE:
        from faster_whisper import WhisperModel

        _WHISPER_MODEL_CACHE[model_size] = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )
    return _WHISPER_MODEL_CACHE[model_size]


def transcribe_video_dialogues(
    video_path: str | Path,
    model_size: str = "small",
    language: Optional[str] = None,
) -> list[DialogueSegment]:
    """Transcribe spoken dialogue and singing in a video file with exact timestamps."""
    vpath = Path(video_path)
    if not vpath.exists():
        return []

    try:
        model = _get_whisper_model(model_size)
        segments, info = model.transcribe(
            str(vpath),
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            beam_size=5,
        )
        results: list[DialogueSegment] = []
        for idx, seg in enumerate(segments):
            txt = (seg.text or "").strip()
            if not txt:
                continue
            # Skip common Whisper silence hallucinations
            if txt.lower() in {"you", "thank you.", "thanks for watching!", "字幕由amara.org社区提供"}:
                continue
            prob = round(float(getattr(seg, "avg_logprob", -0.25)), 3)
            conf = round(min(1.0, max(0.3, 1.0 + prob * 0.4)), 2)
            results.append(
                DialogueSegment(
                    start_time=round(float(seg.start), 2),
                    end_time=round(float(seg.end), 2),
                    speaker=f"Speaker ({info.language.upper()})" if getattr(info, "language", None) else "Speaker",
                    text=txt,
                    confidence=conf,
                )
            )
        return results
    except Exception:
        return []


def transcribe_catalog_videos(
    catalog: MediaCatalog,
    model_size: str = "small",
    language: Optional[str] = None,
) -> MediaCatalog:
    """Populate timestamped dialogues for all videos with audio in the MediaCatalog."""
    for video in catalog.videos:
        if not video.has_audio:
            continue
        dialogues = transcribe_video_dialogues(
            video.path,
            model_size=model_size,
            language=language,
        )
        video.dialogues = dialogues
        if dialogues:
            preview_quotes = " / ".join(f'"{d.text}" ({d.start_time:.1f}s-{d.end_time:.1f}s)' for d in dialogues[:3])
            video.overall_summary = (
                f"{video.filename} ({video.duration:.1f}s): Includes spoken dialogue/audio: {preview_quotes}"
            )
    return catalog
