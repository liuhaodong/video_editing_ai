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
        "Optional plug-and-play Stage-40 specialist that adds warm golden glow, subtle portrait vignettes, "
        "and beat-drop contrast pops to timeline.effects_track and writes vfx_manifest.json."
    )
    enabled_by_default = False
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
            if item.roll_type == "a_roll":
                # Subtle vignette focus on spoken A-roll hero takes
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
            elif item.type == "photo" and idx % 2 == 0:
                # Warm golden glow on keepsake photos
                effects.append(
                    EffectTrackItem(
                        id=f"fx_{fx_counter:03d}",
                        effect_type="warm_glow",
                        target_clip_id=item.id,
                        timeline_start=item.timeline_start,
                        duration=item.duration,
                        intensity=0.45,
                        parameters={"saturation": 1.08, "brightness": 0.02},
                        created_by_agent=self.role_id,
                    )
                )
                fx_counter += 1
            elif item.roll_type == "b_roll":
                # Crisp contrast pop on B-roll action windows
                effects.append(
                    EffectTrackItem(
                        id=f"fx_{fx_counter:03d}",
                        effect_type="contrast_pop",
                        target_clip_id=item.id,
                        timeline_start=item.timeline_start,
                        duration=item.duration,
                        intensity=0.50,
                        parameters={"contrast": 1.07, "saturation": 1.06},
                        created_by_agent=self.role_id,
                    )
                )
                fx_counter += 1

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
