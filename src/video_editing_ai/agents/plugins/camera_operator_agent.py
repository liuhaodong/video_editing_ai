"""Stage 32 Sub-Agent: Cinematography, Stabilization & Smart-Reframe Operator (camera_operator)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...analyzer.media_probe import probe_video_file
from ...models.schema import CinematographyReport, ClipCameraOperation, EffectTrackItem
from ..base import BaseEditingAgent, PipelineBlackboard, register_agent


@register_agent
class CameraOperatorAgent(BaseEditingAgent):
    """Stage 32 Agent: Pairs portrait 9:16 videos into Live Video+Photo Diptychs and stabilizes handheld B-roll."""

    role_id = "camera_operator"
    display_name = "3b. Cinematography, Stabilization & Video Diptych Operator"
    stage_order = 32
    description = (
        "Big-budget camera operator and reframing specialist that pairs vertical 9:16 video clips with "
        "complementary vertical photos into Live Video+Photo Diptychs and applies optical stabilization "
        "to handheld walking B-roll."
    )
    enabled_by_default = True
    requires_artifacts = ["media_index.json", "paper_edit.json"]
    produces_artifacts = ["cinematography_report.json"]
    skill_reference_file = "11_camera_operator_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline
        mi = blackboard.media_index
        if timeline is None:
            return {"portrait_video_diptychs_created": 0, "stabilized_clips_count": 0}

        # Collect candidate portrait photos from MediaIndex or visual_track for pairing with portrait 9:16 videos
        portrait_photos: list[str] = []
        if mi is not None:
            for sc in mi.subclips:
                if sc.media_type == "photo" and not sc.is_discarded and sc.orientation == "portrait":
                    if Path(sc.source_path).exists():
                        portrait_photos.append(sc.source_path)
        if not portrait_photos:
            for v_item in timeline.visual_track:
                if v_item.type == "photo" and Path(v_item.source).exists():
                    portrait_photos.append(v_item.source)

        sc_by_path = {s.source_path: s for s in (mi.subclips if mi else [])}
        operations: list[ClipCameraOperation] = []
        stabilize_effects: list[EffectTrackItem] = []
        diptych_count = 0
        stabilized_count = 0
        photo_cursor = 0

        for item in timeline.visual_track:
            if item.type != "video" or not Path(item.source).exists():
                continue

            info = probe_video_file(Path(item.source))
            w = int(info.get("width", 1920) or 1920)
            h = int(info.get("height", 1080) or 1080)
            is_portrait_video = w < h
            sc_meta = sc_by_path.get(item.source)
            cam_move = sc_meta.camera_movement if sc_meta else "static"

            if is_portrait_video and portrait_photos:
                paired_photo = item.paired_visual_source or portrait_photos[photo_cursor % len(portrait_photos)]
                photo_cursor += 1
                item.paired_visual_source = paired_photo
                diptych_count += 1
                operations.append(
                    ClipCameraOperation(
                        clip_id=item.id,
                        operation_type="video_diptych_pair",
                        paired_visual_source=paired_photo,
                        stabilize_strength=0.0,
                        rationale=(
                            f"Paired vertical 9:16 video ({w}x{h}) with complementary gallery plate "
                            f"'{Path(paired_photo).name}' for balanced 16:9 widescreen composition."
                        ),
                    )
                )
            elif cam_move in ("walk_forward", "pan_left", "pan_right", "tilt_up") or item.roll_type == "b_roll":
                stabilized_count += 1
                operations.append(
                    ClipCameraOperation(
                        clip_id=item.id,
                        operation_type="optical_stabilize",
                        stabilize_strength=0.35,
                        rationale=f"Applied smooth handheld motion damping for '{cam_move}' camera movement.",
                    )
                )
                stabilize_effects.append(
                    EffectTrackItem(
                        id=f"fx_stab_{item.id}",
                        effect_type="video_stabilize",
                        target_clip_id=item.id,
                        timeline_start=item.timeline_start,
                        duration=item.duration,
                        intensity=0.35,
                        parameters={"rx": 16, "ry": 16},
                        created_by_agent=self.role_id,
                    )
                )
            else:
                operations.append(
                    ClipCameraOperation(
                        clip_id=item.id,
                        operation_type="keep_native",
                        rationale="Locked tripod/clean framing preserved natively.",
                    )
                )

        # Allow AI Agent creative overrides via ai_decisions/camera_operator.json
        ai_resp = self.ask_ai_creative_json(
            blackboard=blackboard,
            prompt=(
                f"Evaluated {len(operations)} video clips on Track V1 ({diptych_count} portrait video diptychs, "
                f"{stabilized_count} stabilized handheld clips).\n"
                "Optionally return JSON {\"operations\": [{\"clip_id\": \"...\", \"operation_type\": "
                "\"video_diptych_pair|optical_stabilize|keep_native\", \"paired_visual_source\": \"...\", "
                "\"rationale\": \"...\"}]} to refine camera framing."
            ),
            decision_key=self.role_id,
            timeout_sec=20.0,
        )
        if isinstance(ai_resp, dict) and isinstance(ai_resp.get("operations"), list):
            op_map = {str(o.get("clip_id")): o for o in ai_resp["operations"] if isinstance(o, dict)}
            for op in operations:
                ov = op_map.get(op.clip_id)
                if ov:
                    if ov.get("operation_type") in ("video_diptych_pair", "optical_stabilize", "subtle_push_in", "keep_native"):
                        op.operation_type = ov["operation_type"]
                    if ov.get("paired_visual_source"):
                        op.paired_visual_source = str(ov["paired_visual_source"])
                    if ov.get("rationale"):
                        op.rationale = str(ov["rationale"])
            # Sync overrides back to timeline items
            clip_lookup = {c.id: c for c in timeline.visual_track}
            for op in operations:
                c_obj = clip_lookup.get(op.clip_id)
                if c_obj and op.operation_type == "video_diptych_pair" and op.paired_visual_source:
                    c_obj.paired_visual_source = op.paired_visual_source

        timeline.effects_track = [
            e for e in timeline.effects_track if e.created_by_agent != self.role_id
        ] + stabilize_effects

        # Inject voiceover audio tracks from Stage 25 if present and not yet on timeline.audio_tracks
        narration_data = blackboard.custom_artifacts.get("narration_script.json")
        if narration_data is None and (blackboard.project_dir / "narration_script.json").exists():
            try:
                from ...models.schema import NarrationScriptReport

                narration_data = NarrationScriptReport.from_json_file(
                    blackboard.project_dir / "narration_script.json"
                ).model_dump(mode="json")
            except Exception:
                narration_data = None
        if isinstance(narration_data, dict) and isinstance(narration_data.get("cues"), list):
            from ...models.schema import AudioTrackItem

            existing_vo_ids = {a.id for a in timeline.audio_tracks if a.role == "voiceover"}
            for cue in narration_data["cues"]:
                cid = str(cue.get("cue_id") or "")
                apath = str(cue.get("audio_path") or "")
                if cid and apath and Path(apath).exists() and cid not in existing_vo_ids:
                    st_t = min(max(0.0, float(cue.get("target_timeline_sec", 0.0))), max(0.0, timeline.total_duration - 2.0))
                    timeline.audio_tracks.append(
                        AudioTrackItem(
                            id=cid,
                            role="voiceover",
                            source=apath,
                            timeline_start=round(st_t, 2),
                            duration=float(cue.get("duration_sec", 3.2)),
                            volume=0.92,
                            duck_under_dialogue=False,
                            fade_in=0.15,
                            fade_out=0.25,
                        )
                    )

        proj_title = blackboard.creative_brief.project_title if blackboard.creative_brief else timeline.project_name
        report = CinematographyReport(
            project_title=proj_title,
            portrait_video_diptychs_created=diptych_count,
            stabilized_clips_count=stabilized_count,
            operations=operations,
        )
        out_path = blackboard.project_dir / "cinematography_report.json"
        report.to_json_file(out_path)
        blackboard.set_custom_artifact("cinematography_report.json", report.model_dump(mode="json"))
        timeline.to_yaml(blackboard.project_dir / "timeline.yaml")

        blackboard.emit_activity(
            self.role_id,
            f"Framed {diptych_count} Live Video+Photo Diptychs & stabilized {stabilized_count} handheld clips",
            detail=f"Saved cinematography_report.json ({len(operations)} camera operations)",
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="motion_graphics",
            message_type="handoff",
            subject=f"Completed camera stabilization ({stabilized_count}) and portrait video diptychs ({diptych_count})",
            payload={"diptychs": diptych_count, "stabilized": stabilized_count},
        )
        return {
            "portrait_video_diptychs_created": diptych_count,
            "stabilized_clips_count": stabilized_count,
            "cinematography_report_path": str(out_path),
        }
