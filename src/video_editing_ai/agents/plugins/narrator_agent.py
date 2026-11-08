"""Stage 25 Sub-Agent: Documentary Scriptwriter & Voiceover Narrator (narrator_voiceover)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import wave

import numpy as np

from ...models.schema import NarrationCue, NarrationScriptReport
from ..base import BaseEditingAgent, PipelineBlackboard, register_agent


def _synthesize_narration_wav(
    output_wav: Path,
    spoken_text: str,
    duration_sec: float = 3.2,
    sample_rate: int = 48000,
) -> tuple[Path, str]:
    """
    Mechanically synthesize a broadcast-clean 48kHz stereo WAV file for a narration cue.
    Produces a warm, syllabic vocal-formant envelope cue so headless and offline environments
    render deterministically without requiring cloud TTS credentials.
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    dur = max(1.2, min(12.0, float(duration_sec)))
    n_samples = int(round(dur * sample_rate))
    t = np.linspace(0.0, dur, n_samples, endpoint=False, dtype=np.float32)

    # Syllabic cadence modulated by word count (~2.8 syllables/sec)
    words = max(3, len((spoken_text or "Documentary reflection").split()))
    syl_rate = min(4.2, max(2.0, words / dur))
    envelope = (0.55 + 0.45 * np.sin(2.0 * math.pi * syl_rate * t)) ** 2

    # Smooth fade-in (0.18s) and fade-out (0.28s)
    fade_in_n = min(n_samples // 3, int(0.18 * sample_rate))
    fade_out_n = min(n_samples // 3, int(0.28 * sample_rate))
    window = np.ones(n_samples, dtype=np.float32)
    if fade_in_n > 0:
        window[:fade_in_n] = np.linspace(0.0, 1.0, fade_in_n, dtype=np.float32)
    if fade_out_n > 0:
        window[-fade_out_n:] = np.linspace(1.0, 0.0, fade_out_n, dtype=np.float32)

    # Warm baritone/alto fundamental + soft formant harmonics (kept subtle & pleasant)
    f0 = 165.0 + 12.0 * np.sin(2.0 * math.pi * 0.65 * t)
    phase = 2.0 * math.pi * np.cumsum(f0) / sample_rate
    signal = (
        0.52 * np.sin(phase)
        + 0.28 * np.sin(2.0 * phase + 0.3)
        + 0.14 * np.sin(3.0 * phase + 0.7)
    ) * envelope * window * 0.08

    stereo = np.stack([signal, signal], axis=-1)
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    return output_wav, "acoustic_cadence_synthesizer"


@register_agent
class NarratorVoiceoverAgent(BaseEditingAgent):
    """Stage 25 Agent: Crafts concise documentary voiceover narration cues at key story turns."""

    role_id = "narrator_voiceover"
    display_name = "2b. Documentary Scriptwriter & Voiceover Narrator"
    stage_order = 25
    description = (
        "Big-budget documentary scriptwriter and voiceover producer that writes brief, evocative "
        "narration cues at chapter openings and finales (never colliding with live camera speech) "
        "and synthesizes broadcast voiceover audio tracks."
    )
    enabled_by_default = True
    requires_artifacts = ["creative_brief.json", "paper_edit.json"]
    produces_artifacts = ["narration_script.json"]
    skill_reference_file = "10_narrator_voiceover_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        brief = blackboard.creative_brief
        paper = blackboard.paper_edit
        mi = blackboard.media_index
        if paper is None:
            return {"cues_generated": 0}

        proj_title = (brief.project_title if brief else paper.project_title).replace("_", " ").strip()
        tone = brief.tone if brief else "warm_joyful"
        persona = "warm_documentary_essayist" if tone in ("warm_joyful", "cinematic_nostalgic") else "editorial_storyteller"

        sc_map = {s.subclip_id: s for s in (mi.subclips if mi else [])}
        cues: list[NarrationCue] = []
        cursor_est = 4.0 if (brief and brief.cold_open_hook) else 0.0

        # Build candidate narration cues for chapters that aren't already dominated by A-roll speech
        for idx, chap in enumerate(paper.chapters):
            chap_clips = [sc_map[sid] for sid in chap.selected_subclip_ids if sid in sc_map]
            aroll_count = sum(1 for c in chap_clips if c.roll_type == "a_roll")
            chap_dur = sum(c.duration for c in chap_clips) or 18.0

            if idx == 0 or aroll_count == 0 or idx == len(paper.chapters) - 1:
                cue_id = f"vo_{len(cues) + 1:02d}"
                if idx == 0:
                    line = f"{proj_title} begins with quiet anticipation as the journey unfolds."
                    rationale = "Opening chapter voiceover sets the documentary tone and invites the viewer in."
                elif idx == len(paper.chapters) - 1:
                    line = f"As the light softens over {chap.location_tag}, the day's small details become lasting memories."
                    rationale = "Closing chapter reflection provides emotional resolution before the final title card."
                else:
                    line = f"In {chap.title}, every frame captures the rhythm and texture of {chap.location_tag}."
                    rationale = "Bridges visual montage chapter without live camera dialogue."

                words_cnt = len(line.split())
                est_dur = round(max(2.6, min(5.2, words_cnt * 0.34)), 2)
                cues.append(
                    NarrationCue(
                        cue_id=cue_id,
                        chapter_id=chap.chapter_id,
                        target_timeline_sec=round(cursor_est + 1.2, 2),
                        duration_sec=est_dur,
                        spoken_text=line,
                        voice_persona=persona,
                        rationale=rationale,
                    )
                )
            cursor_est += chap_dur

        # Delegate creative script refinement to AI Agent (ai_decisions/narrator_voiceover.json or Antigravity)
        ai_resp = self.ask_ai_creative_json(
            blackboard=blackboard,
            prompt=(
                f"Project '{proj_title}' (tone={tone}) has {len(paper.chapters)} chapters.\n"
                f"Initial voiceover cues: {[c.model_dump(mode='json') for c in cues]}\n"
                "Return JSON {\"voice_persona\": \"...\", \"cues\": [{\"cue_id\": \"vo_01\", \"chapter_id\": \"...\", "
                "\"target_timeline_sec\": 4.5, \"duration_sec\": 3.4, \"spoken_text\": \"...\", "
                "\"subtitle_translation\": \"...\", \"rationale\": \"...\"}]} to refine documentary narration."
            ),
            decision_key=self.role_id,
            timeout_sec=25.0,
        )
        if isinstance(ai_resp, dict):
            if ai_resp.get("voice_persona"):
                persona = str(ai_resp["voice_persona"])
            if isinstance(ai_resp.get("cues"), list) and ai_resp["cues"]:
                parsed_cues: list[NarrationCue] = []
                for idx_c, raw_c in enumerate(ai_resp["cues"], start=1):
                    if isinstance(raw_c, dict) and raw_c.get("spoken_text"):
                        txt = str(raw_c["spoken_text"]).strip()
                        dur_c = float(raw_c.get("duration_sec") or max(2.6, min(5.5, len(txt.split()) * 0.34)))
                        parsed_cues.append(
                            NarrationCue(
                                cue_id=str(raw_c.get("cue_id") or f"vo_{idx_c:02d}"),
                                chapter_id=str(raw_c.get("chapter_id") or (paper.chapters[0].chapter_id if paper.chapters else "chap_01")),
                                target_timeline_sec=float(raw_c.get("target_timeline_sec") or (idx_c * 8.0)),
                                duration_sec=round(dur_c, 2),
                                spoken_text=txt,
                                subtitle_translation=raw_c.get("subtitle_translation"),
                                voice_persona=str(raw_c.get("voice_persona") or persona),
                                rationale=str(raw_c.get("rationale") or "AI-scripted documentary narration cue."),
                            )
                        )
                if parsed_cues:
                    cues = parsed_cues

        vo_dir = blackboard.project_dir / "narration_audio"
        engine_used = "acoustic_cadence_synthesizer"
        for cue in cues:
            wav_path = vo_dir / f"{cue.cue_id}.wav"
            synth_path, engine_used = _synthesize_narration_wav(
                output_wav=wav_path,
                spoken_text=cue.spoken_text,
                duration_sec=cue.duration_sec,
            )
            cue.audio_path = str(synth_path.resolve())

        report = NarrationScriptReport(
            project_title=proj_title,
            voice_persona=persona,
            synthesis_engine=engine_used,
            total_cues=len(cues),
            cues=cues,
        )
        out_path = blackboard.project_dir / "narration_script.json"
        report.to_json_file(out_path)
        blackboard.set_custom_artifact("narration_script.json", report.model_dump(mode="json"))

        blackboard.emit_activity(
            self.role_id,
            f"Scripted & synthesized {len(cues)} documentary voiceover cues ({persona})",
            detail=f"Saved narration_script.json and {len(cues)} 48kHz stereo WAV cues in narration_audio/",
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="dialogue_editor",
            message_type="handoff",
            subject=f"Prepared {len(cues)} documentary voiceover narration cues for timeline integration",
            payload={"total_cues": len(cues), "narration_script_path": str(out_path)},
        )
        return {
            "total_cues": len(cues),
            "voice_persona": persona,
            "narration_script_path": str(out_path),
        }
