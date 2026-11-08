"""Stage 45 Sub-Agent: Supervising Sound Designer & Foley/Ambience Mixer (sound_designer)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import wave

import numpy as np

from ...models.schema import AudioTrackItem, SoundDesignCue, SoundDesignReport
from ..base import BaseEditingAgent, PipelineBlackboard, register_agent


def _synthesize_sound_design_wav(
    output_wav: Path,
    cue_type: str,
    duration_sec: float = 4.0,
    sample_rate: int = 48000,
) -> Path:
    """
    Mechanically synthesize a 48kHz stereo spatial ambience bed or subtle transition foley cue.
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.5, min(30.0, float(duration_sec)))
    n = int(round(dur * sample_rate))
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(seed=abs(hash(cue_type)) % 10000 + 7)

    # Smooth fade-in and fade-out envelope
    fi = min(n // 3, int(0.35 * sample_rate))
    fo = min(n // 3, int(0.45 * sample_rate))
    env = np.ones(n, dtype=np.float32)
    if fi > 0:
        env[:fi] = np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        env[-fo:] = np.linspace(1.0, 0.0, fo, dtype=np.float32)

    white_l = rng.normal(0.0, 1.0, n).astype(np.float32)
    white_r = rng.normal(0.0, 1.0, n).astype(np.float32)
    # Moving-average low-pass filter to create warm brown/pink room air
    kernel_len = 64 if "air" in cue_type or "room" in cue_type else 32
    kernel = np.ones(kernel_len, dtype=np.float32) / float(kernel_len)
    warm_l = np.convolve(white_l, kernel, mode="same")
    warm_r = np.convolve(white_r, kernel, mode="same")

    if cue_type == "polaroid_shutter":
        # Tactile mechanical shutter click + film advance whirr
        click_env = np.exp(-t * 28.0) + 0.45 * np.exp(-np.maximum(0.0, t - 0.06) * 22.0)
        sig_l = (white_l * click_env * 0.25).astype(np.float32)
        sig_r = (white_r * click_env * 0.25).astype(np.float32)
    elif cue_type == "optical_loupe_chime":
        # Delicate crystal glass harmonic shimmer
        chime_env = np.exp(-t * 2.8) * env
        sig_l = (0.14 * np.sin(2.0 * math.pi * 528.0 * t) + 0.07 * np.sin(2.0 * math.pi * 1056.0 * t)) * chime_env
        sig_r = (0.14 * np.sin(2.0 * math.pi * 530.0 * t) + 0.07 * np.sin(2.0 * math.pi * 1060.0 * t)) * chime_env
    elif cue_type == "velvet_whoosh":
        # Soft sub-bass cinematic transition swell
        swell = np.exp(-((t - dur * 0.45) ** 2) / (2.0 * (dur * 0.18) ** 2))
        sub_tone = np.sin(2.0 * math.pi * (65.0 + 35.0 * (t / dur)) * t)
        sig_l = (warm_l * 0.4 + sub_tone * 0.2) * swell * 0.22
        sig_r = (warm_r * 0.4 + sub_tone * 0.2) * swell * 0.22
    else:
        # Spatial room tone / breeze / gallery hall acoustic bed with gentle stereo modulation
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
class SoundDesignerAgent(BaseEditingAgent):
    """Stage 45 Agent: Synthesizes diegetic spatial ambience beds, foley accents, and J/L-cut audio polish."""

    role_id = "sound_designer"
    display_name = "4c. Supervising Sound Designer & Foley Mixer"
    stage_order = 45
    description = (
        "Big-budget Supervising Sound Designer that layers subtle spatial room-tone ambience under "
        "silent photo galleries, adds tactile foley accents on Polaroid/Loupe reveals, and enforces "
        "smooth 0.40s J/L-cut audio crossfades across all video cuts."
    )
    enabled_by_default = True
    requires_artifacts = ["paper_edit.json"]
    produces_artifacts = ["sound_design_plan.json"]
    skill_reference_file = "14_sound_designer_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline
        if timeline is None:
            return {"ambient_beds_count": 0, "foley_accents_count": 0}

        # 1. Mechanical J/L-cut audio crossfade polish across all Track V1 video clips
        for clip in timeline.visual_track:
            if clip.type == "video":
                clip.audio_crossfade_sec = max(0.40, clip.audio_crossfade_sec)

        # 2. Build spatial ambience & foley cues for photo plates and chapter transitions
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

        # Allow AI Agent creative refinement via ai_decisions/sound_designer.json
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
                ambient_count = sum(1 for c in cues if "air" in c.cue_type or "breeze" in c.cue_type or "murmur" in c.cue_type or "room" in c.cue_type)
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
        report = SoundDesignReport(
            project_title=proj_title,
            target_loudness_lufs=-14.0,
            j_l_cut_crossfade_sec=0.40,
            ambient_beds_count=ambient_count,
            foley_accents_count=foley_count,
            cues=cues,
        )
        out_path = blackboard.project_dir / "sound_design_plan.json"
        report.to_json_file(out_path)
        blackboard.set_custom_artifact("sound_design_plan.json", report.model_dump(mode="json"))
        timeline.to_yaml(blackboard.project_dir / "timeline.yaml")

        blackboard.emit_activity(
            self.role_id,
            f"Synthesized {ambient_count} spatial ambience beds & {foley_count} foley accents (-14 LUFS master)",
            detail=f"Saved sound_design_plan.json and {len(cues)} 48kHz stereo WAV stems in sound_design/",
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="showrunner",
            message_type="handoff",
            subject=f"Mixed {len(cues)} spatial sound design stems and enforced 0.40s J/L-cut crossfades",
            payload={"ambient_beds": ambient_count, "foley_accents": foley_count},
        )
        return {
            "ambient_beds_count": ambient_count,
            "foley_accents_count": foley_count,
            "sound_design_plan_path": str(out_path),
        }
