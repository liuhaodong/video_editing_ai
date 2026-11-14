"""Timestamped dialogue and lyrics extraction using faster-whisper."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Optional

_WHISPER_MODEL_CACHE: dict[str, object] = {}
_HALLUCINATIONS = {"you", "thank you.", "thanks for watching!", "字幕由amara.org社区提供"}
_UNSUPPORTED_AUDIO_CODECS = {"apac", "apple_apac"}


def _decode_audio_pcm_16k(video_path: Path):
    """
    Decode the first non-APAC audio stream via FFmpeg into a 16kHz mono float32 NumPy waveform.
    Avoids PyAV `metadata_errors` incompatibility inside faster-whisper and skips undecodable
    Apple Spatial Audio (`apple_apac`) primary streams on iPhone MOV files.
    """
    import numpy as np

    if not video_path.exists():
        return None
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    a_pos: Optional[int] = None
    if probe.returncode == 0 and probe.stdout.strip():
        try:
            astreams = json.loads(probe.stdout).get("streams", [])
            for idx, st in enumerate(astreams):
                codec = str(st.get("codec_name") or "").strip().lower()
                if codec not in _UNSUPPORTED_AUDIO_CODECS:
                    a_pos = idx
                    break
        except Exception:
            a_pos = None
    if a_pos is None:
        return None

    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-map",
            f"0:a:{a_pos}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or len(proc.stdout) < 3200:
        return None
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _get_whisper_model(model_size: str = "small"):
    if model_size not in _WHISPER_MODEL_CACHE:
        from faster_whisper import WhisperModel

        for compute_type in ("int8", "default", "float32"):
            try:
                _WHISPER_MODEL_CACHE[model_size] = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type=compute_type,
                )
                break
            except Exception:
                if compute_type == "float32":
                    raise
    return _WHISPER_MODEL_CACHE[model_size]


def _transcribe_file_raw(
    video_path: Path,
    model_size: str = "small",
    language: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Run faster-whisper in-process and return plain dict records (safe for isolated worker)."""
    if not video_path.exists():
        return []
    try:
        pcm = _decode_audio_pcm_16k(video_path)
        if pcm is None or pcm.size < 1600:
            return []
        model = _get_whisper_model(model_size)
        segments, info = model.transcribe(
            pcm,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            beam_size=5,
        )
        results: list[dict[str, Any]] = []
        speaker = f"Speaker ({info.language.upper()})" if getattr(info, "language", None) else "Speaker"
        for seg in segments:
            txt = (seg.text or "").strip()
            if not txt or txt.lower() in _HALLUCINATIONS:
                continue
            prob = round(float(getattr(seg, "avg_logprob", -0.25)), 3)
            conf = round(min(1.0, max(0.3, 1.0 + prob * 0.4)), 2)
            results.append(
                {
                    "start_time": round(float(seg.start), 2),
                    "end_time": round(float(seg.end), 2),
                    "speaker": speaker,
                    "text": txt,
                    "confidence": conf,
                }
            )
        return results
    except Exception as exc:
        print(f"[transcriber] Warning: failed to transcribe {video_path.name}: {exc}", file=sys.stderr)
        return []


def _transcribe_videos_in_subprocess(
    video_paths: list[Path],
    model_size: str = "small",
    language: Optional[str] = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Isolate ctranslate2/faster-whisper in a clean Python subprocess on macOS so its bundled
    libiomp5.dylib never collides with PyTorch/sentence-transformers OpenMP in the main process.
    """
    existing = [str(p) for p in video_paths if p.exists()]
    if not existing:
        return {}
    with tempfile.TemporaryDirectory(prefix="whisper_worker_") as tmp_dir:
        tmp = Path(tmp_dir)
        req_path = tmp / "request.json"
        out_path = tmp / "output.json"
        req_path.write_text(
            json.dumps({"videos": existing, "model_size": model_size, "language": language}),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["_VIDEO_EDITING_AI_WHISPER_WORKER"] = "1"
        env.setdefault("OMP_NUM_THREADS", "2")
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-input",
                str(req_path),
                "--worker-output",
                str(out_path),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and out_path.exists():
            try:
                data = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return {}


def _should_isolate_whisper_subprocess() -> bool:
    return (
        sys.platform == "darwin"
        and os.environ.get("_VIDEO_EDITING_AI_WHISPER_WORKER") != "1"
    )


def transcribe_video_dialogues(
    video_path: str | Path,
    model_size: str = "small",
    language: Optional[str] = None,
) -> list[DialogueSegment]:
    """Transcribe spoken dialogue and singing in a video file with exact timestamps."""
    from ..models.schema import DialogueSegment

    vpath = Path(video_path)
    if not vpath.exists():
        return []

    if _should_isolate_whisper_subprocess():
        raw_map = _transcribe_videos_in_subprocess([vpath], model_size=model_size, language=language)
        raw_list = raw_map.get(str(vpath), [])
    else:
        raw_list = _transcribe_file_raw(vpath, model_size=model_size, language=language)

    return [DialogueSegment(**row) for row in raw_list]


def transcribe_catalog_videos(
    catalog: MediaCatalog,
    model_size: str = "small",
    language: Optional[str] = None,
) -> MediaCatalog:
    """Populate timestamped dialogues for all videos with audio in the MediaCatalog."""
    from ..models.schema import DialogueSegment

    audio_videos = [v for v in catalog.videos if v.has_audio]
    if not audio_videos:
        return catalog

    raw_by_path: Optional[dict[str, list[dict[str, Any]]]] = None
    if _should_isolate_whisper_subprocess():
        raw_by_path = _transcribe_videos_in_subprocess(
            [Path(v.path) for v in audio_videos],
            model_size=model_size,
            language=language,
        )

    for video in audio_videos:
        if raw_by_path is not None:
            dialogues = [DialogueSegment(**row) for row in raw_by_path.get(str(Path(video.path)), [])]
        else:
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


def _run_worker_cli() -> int:
    parser = argparse.ArgumentParser(description="Isolated faster-whisper transcription worker")
    parser.add_argument("--worker-input", required=True)
    parser.add_argument("--worker-output", required=True)
    args = parser.parse_args()

    req = json.loads(Path(args.worker_input).read_text(encoding="utf-8"))
    model_size = str(req.get("model_size") or "small")
    language = req.get("language")
    out: dict[str, list[dict[str, Any]]] = {}
    for vp in req.get("videos", []):
        out[str(vp)] = _transcribe_file_raw(Path(vp), model_size=model_size, language=language)
    Path(args.worker_output).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_worker_cli())
