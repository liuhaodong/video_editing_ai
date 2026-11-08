"""Stage 40 Pluggable Sub-Agent: VFX & Motion Polish Designer (vfx_designer)."""

from __future__ import annotations

from typing import Any

from ...models.schema import EffectTrackItem
from ..base import BaseEditingAgent, PipelineBlackboard, register_agent


@register_agent
class VFXAgent(BaseEditingAgent):
    """Pluggable Stage-40 Agent: Attaches non-destructive visual polish & beat-accent VFX to timeline.effects_track."""

    role_id = "vfx_designer"
    display_name = "4. VFX & Motion Polish Designer (Plugin)"
    stage_order = 40
    description = (
        "AI-guided Stage-40 specialist that adds warm golden glow, subtle portrait vignettes, "
        "and beat-drop contrast pops to timeline.effects_track and writes vfx_manifest.json."
    )
    enabled_by_default = True
    requires_artifacts = ["paper_edit.json", "beat_grid.json"]
    produces_artifacts = ["vfx_manifest.json"]
    skill_reference_file = "06_vfx_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline
        if timeline is None:
            return {"effects_added": 0}

        effects: list[EffectTrackItem] = []
        fx_counter = 1

        for idx, item in enumerate(timeline.visual_track):
            note_low = (item.narrative_note or "").lower()
            if item.roll_type == "a_roll" or (
                item.type == "photo" and item.ken_burns and item.ken_burns.mode in ("curator_loupe", "cinematic_depth_3d")
            ):
                # Subtle vignette focus on spoken A-roll hero takes and 3D depth / loupe focal plates
                effects.append(
                    EffectTrackItem(
                        id=f"fx_{fx_counter:03d}",
                        effect_type="vignette_focus",
                        target_clip_id=item.id,
                        timeline_start=item.timeline_start,
                        duration=item.duration,
                        intensity=0.35,
                        parameters={"angle_rad": 0.42},
                        created_by_agent=self.role_id,
                    )
                )
                fx_counter += 1
            elif item.type == "photo" and (
                idx % 2 == 0 or any(k in note_low for k in ("portrait", "warm", "culinary", "dining", "family"))
            ):
                # Warm golden glow on keepsake photos (calibrated <=0.45 for 10-bit HLG HDR)
                effects.append(
                    EffectTrackItem(
                        id=f"fx_{fx_counter:03d}",
                        effect_type="warm_glow",
                        target_clip_id=item.id,
                        timeline_start=item.timeline_start,
                        duration=item.duration,
                        intensity=0.42,
                        parameters={"saturation": 1.07, "brightness": 0.018},
                        created_by_agent=self.role_id,
                    )
                )
                fx_counter += 1
            elif item.roll_type == "b_roll":
                # Crisp contrast pop on B-roll action & architectural windows (calibrated <=0.45 for HDR)
                effects.append(
                    EffectTrackItem(
                        id=f"fx_{fx_counter:03d}",
                        effect_type="contrast_pop",
                        target_clip_id=item.id,
                        timeline_start=item.timeline_start,
                        duration=item.duration,
                        intensity=0.44,
                        parameters={"contrast": 1.06, "saturation": 1.05},
                        created_by_agent=self.role_id,
                    )
                )
                fx_counter += 1

        ai_vfx_resp = self.ask_ai_creative_json(
            blackboard=blackboard,
            prompt=(
                f"Timeline has {len(timeline.visual_track)} clips and {len(effects)} initial HDR-safe VFX items.\n"
                "Optionally return JSON {\"effects\": [...], \"ai_rationale\": \"...\"} to refine effect assignments."
            ),
            decision_key="vfx_designer",
            timeout_sec=25.0,
        )
        if isinstance(ai_vfx_resp, dict) and isinstance(ai_vfx_resp.get("effects"), list):
            try:
                effects = [EffectTrackItem.model_validate(e) for e in ai_vfx_resp["effects"]]
            except Exception:
                pass

        timeline.effects_track = [
            e for e in timeline.effects_track if e.created_by_agent != self.role_id
        ] + effects
        manifest = {
            "agent_role": self.role_id,
            "effects_count": len(effects),
            "supported_ffmpeg_effects": ["warm_glow", "vignette_focus", "contrast_pop"],
            "effects": [e.model_dump(mode="json") for e in effects],
        }
        manifest_path = blackboard.set_custom_artifact("vfx_manifest.json", manifest)
        blackboard.emit_activity(
            self.role_id,
            f"Applied {len(effects)} non-destructive VFX polish items (warm_glow, vignette_focus, contrast_pop)",
            detail=f"Saved vfx_manifest.json ({len(effects)} track effects)",
        )

        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="showrunner",
            message_type="handoff",
            subject=f"Attached {len(effects)} non-destructive VFX items to timeline.effects_track",
            payload={"effects_count": len(effects), "vfx_manifest_path": str(manifest_path)},
        )

        return {
            "effects_count": len(effects),
            "vfx_manifest_path": str(manifest_path),
        }
