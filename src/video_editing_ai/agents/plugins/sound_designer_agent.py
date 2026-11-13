"""Stage 40 Specialist: Supervising Sound Designer & Loudness/Ducking Mixer (sound_mixer)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional
import wave

import numpy as np

from ...models.schema import (
    AudioMixManifest,
    AudioTrackItem,
    ClipAudioMixEntry,
    CulledSubClip,
    DuckingEnvelopeSegment,
    EditTimeline,
    MediaIndex,
    SoundDesignCue,
    SoundDesignReport,
)
from ..base import BaseEditingAgent, PipelineBlackboard, register_agent


def compute_dialogue_normalization_gain(
    measured_lufs: float,
    target_lufs: float = -14.0,
    min_gain_db: float = -12.0,
    max_gain_db: float = 12.0,
) -> tuple[float, float]:
    """
    Compute (applied_gain_db, linear_volume) normalizing spoken dialogue toward target_lufs (-14 LUFS).
    Boosts quiet children's speech (e.g. -25 LUFS) and attenuates loud camera-operator speech (e.g. -9 LUFS).
    """
    if measured_lufs <= -65.0:
        return 0.0, 1.0
    raw_delta_db = float(target_lufs) - float(measured_lufs)
    applied_db = round(max(min_gain_db, min(max_gain_db, raw_delta_db)), 3)
    linear_vol = round(max(0.15, min(3.9811, 10.0 ** (applied_db / 20.0))), 4)
    return applied_db, linear_vol


def build_audio_mix_plan(
    timeline: EditTimeline,
    media_index: Optional[MediaIndex] = None,
    target_dialogue_lufs: float = -14.0,
    nat_sound_bed_db: float = -20.0,
    bgm_speech_duck_db: float = -14.0,
    bgm_diegetic_duck_db: float = -30.0,
    crossfade_sec: float = 0.33,
) -> tuple[list[ClipAudioMixEntry], list[DuckingEnvelopeSegment]]:
    """
    Build per-clip dialogue loudness normalization entries and frame-accurate BGM ducking segments.
    Enforces <= -28 dB (default -30 dB) BGM muting during clips flagged with has_diegetic_music.
    """
    sc_by_path: dict[str, list[CulledSubClip]] = {}
    if media_index is not None:
        for sc in media_index.subclips:
            sc_by_path.setdefault(sc.source_path, []).append(sc)

    nat_linear = round(10.0 ** (nat_sound_bed_db / 20.0), 4)  # 0.1000 (-20 dB)
    speech_duck_linear = round(10.0 ** (bgm_speech_duck_db / 20.0), 4)  # 0.1995 (-14 dB)
    diegetic_duck_linear = round(10.0 ** (bgm_diegetic_duck_db / 20.0), 5)  # 0.03162 (-30 dB <= -28 dB)

    clip_entries: list[ClipAudioMixEntry] = []
    envelope: list[DuckingEnvelopeSegment] = []

    for clip in timeline.visual_track:
        cands = sc_by_path.get(clip.source, [])
        best_sc = min(cands, key=lambda c: abs(c.golden_in - clip.source_in)) if cands else None

        measured_lufs = float(best_sc.integrated_lufs) if best_sc is not None else -20.0
        has_diegetic = bool(best_sc.has_diegetic_music) if best_sc is not None else False
        st_sec = round(clip.timeline_start, 3)
        en_sec = round(clip.timeline_start + clip.duration, 3)

        if clip.type == "photo":
            clip.volume = 0.0
            clip.audio_crossfade_sec = 0.0
            clip_entries.append(
                ClipAudioMixEntry(
                    clip_id=clip.id,
                    role_type="still",
                    measured_lufs=-70.0,
                    target_lufs=-70.0,
                    applied_gain_db=0.0,
                    linear_volume=0.0,
                    has_diegetic_music=False,
                    crossfade_sec=0.0,
                )
            )
            envelope.append(
                DuckingEnvelopeSegment(
                    start_sec=st_sec,
                    end_sec=en_sec,
                    mode="full_bgm",
                    bgm_attenuation_db=0.0,
                    bgm_linear_gain=1.0,
                    trigger_clip_id=clip.id,
                )
            )
        elif has_diegetic:
            gain_db, lin_vol = compute_dialogue_normalization_gain(measured_lufs, target_lufs=target_dialogue_lufs)
            clip.volume = lin_vol
            clip.audio_crossfade_sec = max(crossfade_sec, clip.audio_crossfade_sec)
            clip_entries.append(
                ClipAudioMixEntry(
                    clip_id=clip.id,
                    role_type="diegetic_music",
                    measured_lufs=measured_lufs,
                    target_lufs=target_dialogue_lufs,
                    applied_gain_db=gain_db,
                    linear_volume=lin_vol,
                    has_diegetic_music=True,
                    crossfade_sec=clip.audio_crossfade_sec,
                )
            )
            envelope.append(
                DuckingEnvelopeSegment(
                    start_sec=st_sec,
                    end_sec=en_sec,
                    mode="diegetic_music_mute",
                    bgm_attenuation_db=bgm_diegetic_duck_db,
                    bgm_linear_gain=diegetic_duck_linear,
                    trigger_clip_id=clip.id,
                )
            )
        elif clip.roll_type == "a_roll":
            gain_db, lin_vol = compute_dialogue_normalization_gain(measured_lufs, target_lufs=target_dialogue_lufs)
            clip.volume = lin_vol
            clip.audio_crossfade_sec = max(crossfade_sec, clip.audio_crossfade_sec)
            clip_entries.append(
                ClipAudioMixEntry(
                    clip_id=clip.id,
                    role_type="dialogue",
                    measured_lufs=measured_lufs,
                    target_lufs=target_dialogue_lufs,
                    applied_gain_db=gain_db,
                    linear_volume=lin_vol,
                    has_diegetic_music=False,
                    crossfade_sec=clip.audio_crossfade_sec,
                )
            )
            envelope.append(
                DuckingEnvelopeSegment(
                    start_sec=st_sec,
                    end_sec=en_sec,
                    mode="speech_duck",
                    bgm_attenuation_db=bgm_speech_duck_db,
                    bgm_linear_gain=speech_duck_linear,
                    trigger_clip_id=clip.id,
                )
            )
        else:
            clip.volume = nat_linear
            clip.audio_crossfade_sec = max(crossfade_sec, clip.audio_crossfade_sec)
            clip_entries.append(
                ClipAudioMixEntry(
                    clip_id=clip.id,
                    role_type="b_roll_nat",
                    measured_lufs=measured_lufs,
                    target_lufs=nat_sound_bed_db,
                    applied_gain_db=nat_sound_bed_db,
                    linear_volume=nat_linear,
                    has_diegetic_music=False,
                    crossfade_sec=clip.audio_crossfade_sec,
                )
            )
            envelope.append(
                DuckingEnvelopeSegment(
                    start_sec=st_sec,
                    end_sec=en_sec,
                    mode="nat_sound_bed",
                    bgm_attenuation_db=0.0,
                    bgm_linear_gain=1.0,
                    trigger_clip_id=clip.id,
                )
            )

    # Also add speech_duck windows for any Track A1 voiceover cues
    for a_item in timeline.audio_tracks:
        if a_item.role == "voiceover" and a_item.duration:
            envelope.append(
                DuckingEnvelopeSegment(
                    start_sec=round(a_item.timeline_start, 3),
                    end_sec=round(a_item.timeline_start + a_item.duration, 3),
                    mode="speech_duck",
                    bgm_attenuation_db=bgm_speech_duck_db,
                    bgm_linear_gain=speech_duck_linear,
                    trigger_clip_id=a_item.id,
                )
            )

    envelope.sort(key=lambda seg: (seg.start_sec, 0 if seg.mode == "diegetic_music_mute" else 1))
    return clip_entries, envelope


def _synthesize_sound_design_wav(
    output_wav: Path,
    cue_type: str,
    duration_sec: float = 4.0,
    sample_rate: int = 48000,
) -> Path:
    """Mechanically synthesize a 48kHz stereo spatial ambience bed or subtle transition foley cue."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.5, min(30.0, float(duration_sec)))
    n = int(round(dur * sample_rate))
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(seed=abs(hash(cue_type)) % 10000 + 7)

    fi = min(n // 3, int(0.35 * sample_rate))
    fo = min(n // 3, int(0.45 * sample_rate))
    env = np.ones(n, dtype=np.float32)
    if fi > 0:
        env[:fi] = np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        env[-fo:] = np.linspace(1.0, 0.0, fo, dtype=np.float32)

    white_l = rng.normal(0.0, 1.0, n).astype(np.float32)
    white_r = rng.normal(0.0, 1.0, n).astype(np.float32)
    kernel_len = 64 if "air" in cue_type or "room" in cue_type else 32
    kernel = np.ones(kernel_len, dtype=np.float32) / float(kernel_len)
    warm_l = np.convolve(white_l, kernel, mode="same")
    warm_r = np.convolve(white_r, kernel, mode="same")

    if cue_type == "polaroid_shutter":
        click_env = np.exp(-t * 28.0) + 0.45 * np.exp(-np.maximum(0.0, t - 0.06) * 22.0)
        sig_l = (white_l * click_env * 0.25).astype(np.float32)
        sig_r = (white_r * click_env * 0.25).astype(np.float32)
    elif cue_type == "optical_loupe_chime":
        chime_env = np.exp(-t * 2.8) * env
        sig_l = (0.14 * np.sin(2.0 * math.pi * 528.0 * t) + 0.07 * np.sin(2.0 * math.pi * 1056.0 * t)) * chime_env
        sig_r = (0.14 * np.sin(2.0 * math.pi * 530.0 * t) + 0.07 * np.sin(2.0 * math.pi * 1060.0 * t)) * chime_env
    elif cue_type == "velvet_whoosh":
        swell = np.exp(-((t - dur * 0.45) ** 2) / (2.0 * (dur * 0.18) ** 2))
        sub_tone = np.sin(2.0 * math.pi * (65.0 + 35.0 * (t / dur)) * t)
        sig_l = (warm_l * 0.4 + sub_tone * 0.2) * swell * 0.22
        sig_r = (warm_r * 0.4 + sub_tone * 0.2) * swell * 0.22
    else:
        mod_l = 0.75 + 0.25 * np.sin(2.0 * math.pi * 0.22 * t)
        mod_r = 0.75 + 0.25 * np.cos(2.0 * math.pi * 0.22 * t)
        harmonic = 0.04 * np.sin(2.0 * math.pi * 110.0 * t) + 0.025 * np.sin(2.0 * math.pi * 165.0 * t)
        sig_l = (warm_l * mod_l * 0.35 + harmonic) * env * 0.18
        sig_r = (warm_r * mod_r * 0.35 + harmonic) * env * 0.18

    stereo = np.stack([sig_l, sig_r], axis=-1)
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return output_wav


@register_agent
class SoundDesignerMixerAgent(BaseEditingAgent):
    """Stage 40 Agent: Normalizes dialogue to -14 LUFS, ducks BGM (-14 dB speech / -30 dB diegetic music), and mixes foley."""

    role_id = "sound_mixer"
    legacy_role_aliases = ["sound_designer"]
    display_name = "4d. Supervising Sound Designer & Re-Recording Mixer"
    stage_order = 40
    substage_priority = 3
    description = (
        "Post-assembly re-recording mixer that normalizes dialogue clips toward -14 LUFS, generates "
        "frame-accurate BGM ducking (-14 dB under speech, -20 dB Nat-Sound under pure B-roll, and "
        "<=-28 dB deep muting during has_diegetic_music clips), applies 0.33s audio crossfades, and "
        "synthesizes spatial foley/ambience stems."
    )
    enabled_by_default = True
    requires_artifacts = ["media_index.json", "paper_edit.json", "beat_grid.json", "timeline.yaml"]
    produces_artifacts = ["audio_mix_manifest.json", "sound_design_plan.json"]
    skill_reference_file = "14_sound_designer_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline
        if timeline is None:
            return {"ambient_beds_count": 0, "foley_accents_count": 0}

        # 1. Compute post-assembly dialogue loudness normalization (-14 LUFS), -20 dB Nat-Sound, -30 dB diegetic mute & 0.33s crossfades
        clip_mix_entries, ducking_envelope = build_audio_mix_plan(
            timeline=timeline,
            media_index=blackboard.media_index,
            target_dialogue_lufs=-14.0,
            nat_sound_bed_db=-20.0,
            bgm_speech_duck_db=-14.0,
            bgm_diegetic_duck_db=-30.0,
            crossfade_sec=0.33,
        )

        # Calibrate Track A3 BGM duck_volume to -14 dB relative to base_vol (0.28 * 10^(-14/20) = 0.0559)
        for a_trk in timeline.audio_tracks:
            if a_trk.role == "music":
                a_trk.duck_under_dialogue = True
                a_trk.duck_volume = round(a_trk.volume * (10.0 ** (-14.0 / 20.0)), 4)
                a_trk.duration = round(timeline.total_duration, 2)

        # 2. Build spatial ambience & foley cues for photo plates
        cues: list[SoundDesignCue] = []
        ambient_count = 0
        foley_count = 0

        for idx, clip in enumerate(timeline.visual_track):
            if clip.type == "photo" and clip.ken_burns is not None:
                mode = clip.ken_burns.mode
                note_low = f"{clip.caption_title or ''} {clip.narrative_note or ''}".lower()
                if mode == "polaroid_scrapbook" and foley_count < 4:
                    foley_count += 1
                    cues.append(
                        SoundDesignCue(
                            cue_id=f"sfx_foley_{len(cues) + 1:02d}",
                            cue_type="polaroid_shutter",
                            timeline_start=round(clip.timeline_start + 0.12, 2),
                            duration=0.85,
                            volume=0.09,
                            rationale=f"Tactile vintage camera shutter foley on Polaroid reveal ({clip.id}).",
                        )
                    )
                elif mode == "curator_loupe" and foley_count < 4:
                    foley_count += 1
                    cues.append(
                        SoundDesignCue(
                            cue_id=f"sfx_foley_{len(cues) + 1:02d}",
                            cue_type="optical_loupe_chime",
                            timeline_start=round(clip.timeline_start + 0.20, 2),
                            duration=1.6,
                            volume=0.07,
                            rationale=f"Subtle optical glass harmonic shimmer for Curator's Loupe ({clip.id}).",
                        )
                    )
                elif ambient_count < 4 and idx % 3 == 0:
                    ambient_count += 1
                    amb_type = (
                        "outdoor_breeze_water"
                        if any(k in note_low for k in ("river", "garden", "park", "courtyard", "tower", "bridge"))
                        else (
                            "cafe_porcelain_murmur"
                            if any(k in note_low for k in ("culinary", "dining", "bakery", "menu", "cafe"))
                            else "gallery_hall_air"
                        )
                    )
                    cues.append(
                        SoundDesignCue(
                            cue_id=f"sfx_amb_{len(cues) + 1:02d}",
                            cue_type=amb_type,  # type: ignore[arg-type]
                            timeline_start=round(clip.timeline_start, 2),
                            duration=round(min(6.0, clip.duration), 2),
                            volume=0.07,
                            rationale=f"Diegetic '{amb_type}' spatial acoustic bed under still plate ({clip.id}).",
                        )
                    )

        ai_resp = self.ask_ai_creative_json(
            blackboard=blackboard,
            prompt=(
                f"Timeline has {len(timeline.visual_track)} clips; proposed {len(cues)} sound design cues.\n"
                "Optionally return JSON {\"cues\": [...], \"target_loudness_lufs\": -14.0} to refine the soundscape."
            ),
            decision_key=self.role_id,
            timeout_sec=20.0,
        )
        if isinstance(ai_resp, dict) and isinstance(ai_resp.get("cues"), list) and ai_resp["cues"]:
            try:
                cues = [SoundDesignCue.model_validate(c) for c in ai_resp["cues"]]
                ambient_count = sum(
                    1
                    for c in cues
                    if "air" in c.cue_type or "breeze" in c.cue_type or "murmur" in c.cue_type or "room" in c.cue_type
                )
                foley_count = len(cues) - ambient_count
            except Exception:
                pass

        sfx_dir = blackboard.project_dir / "sound_design"
        timeline.audio_tracks = [a for a in timeline.audio_tracks if a.role != "sfx"]

        for cue in cues:
            wav_path = sfx_dir / f"{cue.cue_id}_{cue.cue_type}.wav"
            _synthesize_sound_design_wav(
                output_wav=wav_path,
                cue_type=cue.cue_type,
                duration_sec=cue.duration,
            )
            cue.audio_path = str(wav_path.resolve())
            timeline.audio_tracks.append(
                AudioTrackItem(
                    id=cue.cue_id,
                    role="sfx",
                    source=cue.audio_path,
                    timeline_start=cue.timeline_start,
                    duration=cue.duration,
                    volume=cue.volume,
                    duck_under_dialogue=True,
                    duck_volume=round(cue.volume * 0.4, 3),
                    fade_in=0.25,
                    fade_out=0.35,
                )
            )

        proj_title = blackboard.creative_brief.project_title if blackboard.creative_brief else timeline.project_name
        diegetic_muted = sum(1 for c in clip_mix_entries if c.has_diegetic_music)

        mix_manifest = AudioMixManifest(
            project_title=proj_title,
            target_dialogue_lufs=-14.0,
            nat_sound_bed_db=-20.0,
            bgm_speech_duck_db=-14.0,
            bgm_diegetic_duck_db=-30.0,
            default_crossfade_sec=0.33,
            diegetic_music_clips_muted=diegetic_muted,
            clip_mix=clip_mix_entries,
            ducking_envelope=ducking_envelope,
        )
        mix_path = blackboard.project_dir / "audio_mix_manifest.json"
        mix_manifest.to_json_file(mix_path)
        blackboard.audio_mix_manifest = mix_manifest

        sd_report = SoundDesignReport(
            project_title=proj_title,
            target_loudness_lufs=-14.0,
            j_l_cut_crossfade_sec=0.40,
            ambient_beds_count=ambient_count,
            foley_accents_count=foley_count,
            cues=cues,
        )
        sd_path = blackboard.project_dir / "sound_design_plan.json"
        sd_report.to_json_file(sd_path)
        blackboard.set_custom_artifact("sound_design_plan.json", sd_report.model_dump(mode="json"))
        timeline.to_yaml(blackboard.project_dir / "timeline.yaml")

        blackboard.emit_activity(
            self.role_id,
            f"Normalized {len(clip_mix_entries)} clips (-14 LUFS dialogue, -20 dB Nat-Sound, {diegetic_muted} diegetic BGM mutes) & {len(cues)} foley stems",
            detail="Saved audio_mix_manifest.json (0.33s crossfades, -30 dB diegetic ducking) + sound_design_plan.json",
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="showrunner",
            message_type="handoff",
            subject="Post-assembly audio loudness normalization, ducking envelope, and foley stems complete",
            payload={
                "clips_mixed": len(clip_mix_entries),
                "diegetic_music_clips_muted": diegetic_muted,
                "ambient_beds": ambient_count,
                "foley_accents": foley_count,
            },
        )
        return {
            "clips_mixed_count": len(clip_mix_entries),
            "diegetic_music_clips_muted": diegetic_muted,
            "ambient_beds_count": ambient_count,
            "foley_accents_count": foley_count,
            "audio_mix_manifest_path": str(mix_path),
            "sound_design_plan_path": str(sd_path),
        }


# Backward-compatible class alias
SoundDesignerAgent = SoundDesignerMixerAgent
