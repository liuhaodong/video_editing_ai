"""Stage 10: Audio & Vibe Sub-Agent (audio_vibe)."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Optional

import numpy as np

from ..analyzer.media_probe import probe_video_file
from ..models.schema import BeatGrid, EnergySection
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


def _analyze_music_waveform(audio_path: Path, max_duration: float = 180.0) -> BeatGrid:
    """Decode audio via FFmpeg into 8kHz mono PCM to detect BPM, beats, downbeats, and energy sections."""
    probe = probe_video_file(audio_path)
    duration = round(min(max_duration, float(probe.get("duration") or 120.0)), 2)
    if duration <= 1.0:
        duration = 120.0

    sr = 8000
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-t",
        f"{duration:.2f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "s16le",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, timeout=25)
        pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        pcm = np.array([], dtype=np.float32)

    if pcm.size < sr * 2:
        bpm = 116.0
        beat_step = 60.0 / bpm
        beats = [round(float(t), 3) for t in np.arange(0.0, duration, beat_step)]
        downbeats = beats[::4]
        sections = [
            EnergySection(start=0.0, end=round(duration * 0.2, 2), label="intro", rms_energy=0.45),
            EnergySection(start=round(duration * 0.2, 2), end=round(duration * 0.5, 2), label="verse", rms_energy=0.65),
            EnergySection(start=round(duration * 0.5, 2), end=round(duration * 0.85, 2), label="chorus_drop", rms_energy=0.88),
            EnergySection(start=round(duration * 0.85, 2), end=duration, label="outro", rms_energy=0.50),
        ]
        return BeatGrid(
            source_audio=str(audio_path),
            duration=duration,
            tempo_bpm=bpm,
            beats=beats,
            downbeats=downbeats,
            energy_sections=sections,
        )

    hop = 200
    env_fps = sr / hop
    n_frames = pcm.size // hop
    framed = pcm[: n_frames * hop].reshape((n_frames, hop))
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-9)

    onset = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    if float(np.max(onset)) > 0:
        onset = onset / float(np.max(onset))

    min_lag = int(round((60.0 / 148.0) * env_fps))
    max_lag = int(round((60.0 / 85.0) * env_fps))
    best_lag = int(round((60.0 / 116.0) * env_fps))
    if n_frames > max_lag * 2:
        centered = onset - float(np.mean(onset))
        ac = np.correlate(centered, centered, mode="full")[n_frames - 1 :]
        window = ac[min_lag : max_lag + 1]
        if window.size > 0:
            best_lag = min_lag + int(np.argmax(window))

    beat_step = best_lag / env_fps
    bpm = round(60.0 / max(0.35, beat_step), 1)

    phase_samples = min(best_lag, n_frames)
    best_phase = 0
    if phase_samples > 1:
        phase_scores = [float(np.mean(onset[p::best_lag])) for p in range(phase_samples)]
        best_phase = int(np.argmax(phase_scores))

    phase_sec = best_phase / env_fps
    beats = [round(float(t), 3) for t in np.arange(phase_sec, duration, beat_step)]
    if not beats or beats[0] > 0.35:
        beats.insert(0, 0.0)
    downbeats = beats[::4]

    q1 = round(duration * 0.20, 2)
    q2 = round(duration * 0.50, 2)
    q3 = round(duration * 0.85, 2)
    max_rms = float(np.percentile(rms, 95)) + 1e-6

    def _sec_rms(t0: float, t1: float) -> float:
        i0 = max(0, int(t0 * env_fps))
        i1 = min(n_frames, max(i0 + 1, int(t1 * env_fps)))
        return round(min(1.0, float(np.mean(rms[i0:i1]) / max_rms)), 3)

    sections = [
        EnergySection(start=0.0, end=q1, label="intro", rms_energy=_sec_rms(0.0, q1)),
        EnergySection(start=q1, end=q2, label="verse", rms_energy=_sec_rms(q1, q2)),
        EnergySection(start=q2, end=q3, label="chorus_drop", rms_energy=_sec_rms(q2, q3)),
        EnergySection(start=q3, end=duration, label="outro", rms_energy=_sec_rms(q3, duration)),
    ]

    return BeatGrid(
        source_audio=str(audio_path),
        duration=duration,
        tempo_bpm=bpm,
        beats=beats,
        downbeats=downbeats,
        energy_sections=sections,
    )


@register_agent
class AudioVibeAgent(BaseEditingAgent):
    """Agent 4 (Stage 10): Detects music beat/downbeat grid and configures 3-layer audio mix rules."""

    role_id = "audio_vibe"
    display_name = "1b. Audio & Vibe Specialist"
    stage_order = 10
    description = (
        "Analyzes background music for BPM, beats, downbeats, and energy drops into beat_grid.json, "
        "and sets -20dB Nat-Sound & -14dB auto-ducking envelopes."
    )
    enabled_by_default = True
    requires_artifacts = ["creative_brief.json"]
    produces_artifacts = ["beat_grid.json"]
    skill_reference_file = "04_audio_vibe_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        audio_candidate: Optional[Path] = None
        if blackboard.extra_audio_paths:
            audio_candidate = blackboard.extra_audio_paths[0]
        elif blackboard.catalog and blackboard.catalog.audio_files:
            p = Path(blackboard.catalog.audio_files[0].path)
            if p.exists():
                audio_candidate = p

        target_dur = 110.0
        if blackboard.creative_brief is not None:
            target_dur = blackboard.creative_brief.target_duration_sec

        if audio_candidate is not None and audio_candidate.exists():
            beat_grid = _analyze_music_waveform(audio_candidate, max_duration=max(140.0, target_dur + 30.0))
        else:
            bpm = 120.0
            beat_step = 0.5
            beats = [round(float(t), 3) for t in np.arange(0.0, target_dur + 20.0, beat_step)]
            downbeats = beats[::4]
            beat_grid = BeatGrid(
                source_audio="",
                duration=round(target_dur + 20.0, 2),
                tempo_bpm=bpm,
                beats=beats,
                downbeats=downbeats,
                energy_sections=[
                    EnergySection(start=0.0, end=round(target_dur * 0.25, 2), label="intro", rms_energy=0.5),
                    EnergySection(start=round(target_dur * 0.25, 2), end=round(target_dur * 0.55, 2), label="verse", rms_energy=0.7),
                    EnergySection(start=round(target_dur * 0.55, 2), end=round(target_dur * 0.85, 2), label="chorus_drop", rms_energy=0.9),
                    EnergySection(start=round(target_dur * 0.85, 2), end=round(target_dur, 2), label="outro", rms_energy=0.55),
                ],
            )

        out_path = blackboard.project_dir / "beat_grid.json"
        beat_grid.to_json_file(out_path)
        blackboard.beat_grid = beat_grid

        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="montage_editor",
            message_type="handoff",
            subject=f"Beat grid ready ({beat_grid.tempo_bpm} BPM, {len(beat_grid.downbeats)} downbeats)",
            payload={
                "tempo_bpm": beat_grid.tempo_bpm,
                "downbeat_count": len(beat_grid.downbeats),
                "source_audio": beat_grid.source_audio,
                "nat_sound_db": -20.0,
                "music_duck_db": -14.0,
                "duck_ramp_sec": 1.0,
            },
        )

        return {
            "beat_grid_path": str(out_path),
            "tempo_bpm": beat_grid.tempo_bpm,
            "total_beats": len(beat_grid.beats),
            "total_downbeats": len(beat_grid.downbeats),
        }
