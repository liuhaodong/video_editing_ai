"""Stage 35: Lead Assembly Editor Agent (lead_editor)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models.schema import AudioTrackItem, CulledSubClip, TimelineMarker, VisualTrackItem
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


@register_agent
class LeadAssemblyEditorAgent(BaseEditingAgent):
    """Stage 35 Agent: Stitches chapter cuts, inserts J/L-cut chapter bridges, keeps A3 BGM continuous, and clamps trim handles."""

    role_id = "lead_editor"
    display_name = "3c. Lead Assembly & Continuity Editor"
    stage_order = 35
    substage_priority = 0
    description = (
        "Stitches chapter timelines into one continuous EditTimeline, inserts 0.35-0.75s J/L-cut audio bridges "
        "across chapter transitions, keeps Track A3 BGM continuous across chapters, and clamps V1/V2 trims "
        "within usable_in/usable_out handles."
    )
    enabled_by_default = True
    requires_artifacts = ["paper_edit.json", "media_index.json", "beat_grid.json", "timeline.yaml"]
    produces_artifacts = ["assembly_report.json", "timeline.yaml"]
    skill_reference_file = "16_lead_editor_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline
        if timeline is None and (blackboard.project_dir / "timeline.yaml").exists():
            from ..models.schema import EditTimeline

            timeline = EditTimeline.from_yaml(blackboard.project_dir / "timeline.yaml")
            blackboard.timeline = timeline

        if timeline is None:
            return {"stitched_clips": 0, "chapter_bridges_inserted": 0, "clamped_handles": 0}

        media_index = blackboard.media_index
        subclips_by_path: dict[str, list[CulledSubClip]] = {}
        if media_index is not None:
            for sc in media_index.subclips:
                subclips_by_path.setdefault(sc.source_path, []).append(sc)

        # 1. Stitch Track V1 sequentially with zero gaps/overlaps (preserving beat-snapped clip durations)
        timeline.recalculate_sequential_starts()

        # 2. Clamp V1 & V2 video source_in/source_out inside usable_in/usable_out without altering timeline durations
        clamped_handles = 0

        def _clamp_item_handles(item: VisualTrackItem) -> None:
            nonlocal clamped_handles
            if item.type != "video":
                return
            cands = subclips_by_path.get(item.source, [])
            if not cands:
                return
            # Match closest subclip window
            best_sc = min(cands, key=lambda c: abs(c.golden_in - item.source_in))
            u_in = float(best_sc.usable_in)
            u_out = float(best_sc.usable_out)
            dur = float(item.duration)

            if item.source_in < u_in:
                item.source_in = round(u_in, 3)
                item.source_out = round(min(u_out, item.source_in + dur), 3)
                clamped_handles += 1
            elif item.source_in + dur > u_out and (u_out - dur) >= u_in:
                # Slide window backward inside usable margins while keeping beat-snapped duration untouched
                item.source_in = round(u_out - dur, 3)
                item.source_out = round(u_out, 3)
                clamped_handles += 1
            else:
                item.source_out = round(min(max(u_out, item.source_in + 0.5), item.source_in + dur), 3)

            # Clamp J/L-cut audio handle extensions inside [u_in, u_out]
            jl = float(item.j_l_cut_audio_offset or 0.0)
            if jl < 0.0 and (item.source_in + jl) < u_in:
                item.j_l_cut_audio_offset = round(min(0.0, u_in - item.source_in), 3)
                clamped_handles += 1
            elif jl > 0.0 and (item.source_out + jl) > u_out:
                item.j_l_cut_audio_offset = round(max(0.0, u_out - item.source_out), 3)
                clamped_handles += 1

        for v_item in timeline.visual_track:
            _clamp_item_handles(v_item)
        for b_item in timeline.broll_track:
            _clamp_item_handles(b_item)
            # Clamp V2 timeline window strictly inside [0, total_duration]
            if b_item.timeline_start + b_item.duration > timeline.total_duration:
                b_item.timeline_start = round(max(0.0, timeline.total_duration - b_item.duration), 3)

        # 3. Insert 0.35-0.75s J/L-cut audio bridges across chapter transitions
        chapter_bridges = 0
        v1 = timeline.visual_track
        for idx in range(1, len(v1)):
            prev_clip = v1[idx - 1]
            curr_clip = v1[idx]
            if prev_clip.act == curr_clip.act or prev_clip.id == "clip_000_hook":
                continue

            # Entering a new chapter on a video clip: apply a -0.45s J-cut audio pre-lap when usable_in allows
            if curr_clip.type == "video":
                cands = subclips_by_path.get(curr_clip.source, [])
                u_in = min((c.usable_in for c in cands), default=0.0)
                avail_pre = max(0.0, curr_clip.source_in - u_in)
                if avail_pre >= 0.35:
                    bridge_sec = round(min(0.55, max(0.35, avail_pre)), 3)
                    curr_clip.j_l_cut_audio_offset = -bridge_sec
                    curr_clip.audio_crossfade_sec = max(curr_clip.audio_crossfade_sec, 0.35)
                    chapter_bridges += 1
            # Or trailing audio L-cut (+0.45s) from outgoing chapter video into a chapter-opening still
            elif prev_clip.type == "video" and curr_clip.type == "photo":
                cands_prev = subclips_by_path.get(prev_clip.source, [])
                u_out = max((c.usable_out for c in cands_prev), default=prev_clip.source_out)
                avail_post = max(0.0, u_out - prev_clip.source_out)
                if avail_post >= 0.35 and abs(prev_clip.j_l_cut_audio_offset) < 0.05:
                    bridge_sec = round(min(0.60, max(0.35, avail_post)), 3)
                    prev_clip.j_l_cut_audio_offset = bridge_sec
                    prev_clip.audio_crossfade_sec = max(prev_clip.audio_crossfade_sec, 0.35)
                    chapter_bridges += 1

        # 4. Sync chapter markers to exact first-clip start timestamps
        if blackboard.paper_edit is not None:
            marker_colors = ["Green", "Yellow", "Sand", "Purple", "Blue"]
            new_markers: list[TimelineMarker] = []
            if v1 and v1[0].id == "clip_000_hook":
                new_markers.append(
                    TimelineMarker(
                        time_sec=0.0,
                        name="Cold Open Teaser Hook",
                        color="Blue",
                        note="First-6-seconds peak teaser hook",
                    )
                )
            for c_idx, chap in enumerate(blackboard.paper_edit.chapters):
                first_in_chap = next((c for c in v1 if c.act == chap.title), None)
                if first_in_chap is not None:
                    new_markers.append(
                        TimelineMarker(
                            time_sec=round(first_in_chap.timeline_start, 3),
                            name=chap.title,
                            color=marker_colors[c_idx % len(marker_colors)],
                            note=chap.narrative_goal,
                        )
                    )
            if new_markers:
                timeline.markers = new_markers

        # 5. Keep Track A3 BGM continuous across all adjacent chapters (single seamless music bed)
        music_tracks = [a for a in timeline.audio_tracks if a.role == "music"]
        other_audio = [a for a in timeline.audio_tracks if a.role != "music"]
        bgm_continuous = False
        if music_tracks:
            primary_bgm = music_tracks[0]
            primary_bgm.id = "bgm_main"
            primary_bgm.timeline_start = 0.0
            primary_bgm.duration = round(timeline.total_duration, 2)
            timeline.audio_tracks = [primary_bgm, *other_audio]
            bgm_continuous = True
        else:
            # Populate from bgm_selection or beat_grid if missing
            bgm_src = ""
            if blackboard.beat_grid and blackboard.beat_grid.source_audio and Path(blackboard.beat_grid.source_audio).exists():
                bgm_src = blackboard.beat_grid.source_audio
            elif blackboard.bgm_selection and blackboard.bgm_selection.selected_track_path and Path(blackboard.bgm_selection.selected_track_path).exists():
                bgm_src = blackboard.bgm_selection.selected_track_path
            if bgm_src:
                timeline.audio_tracks = [
                    AudioTrackItem(
                        id="bgm_main",
                        role="music",
                        source=bgm_src,
                        timeline_start=0.0,
                        duration=round(timeline.total_duration, 2),
                        source_in=0.0,
                        volume=0.28,
                        duck_under_dialogue=True,
                        duck_volume=0.056,
                        duck_ramp_sec=0.8,
                        fade_in=1.0,
                        fade_out=2.5,
                    ),
                    *other_audio,
                ]
                bgm_continuous = True

        if blackboard.pending_revision_directives(self.role_id):
            blackboard.mark_directives_resolved(self.role_id)

        assembly_summary = {
            "agent_role": self.role_id,
            "stitched_v1_clips": len(timeline.visual_track),
            "connected_v2_cutaways": len(timeline.broll_track),
            "chapter_bridges_inserted": chapter_bridges,
            "clamped_handles_count": clamped_handles,
            "bgm_continuous": bgm_continuous,
            "total_duration_sec": round(timeline.total_duration, 3),
        }
        report_path = blackboard.set_custom_artifact("assembly_report.json", assembly_summary)
        tl_path = blackboard.project_dir / "timeline.yaml"
        timeline.to_yaml(tl_path)

        blackboard.emit_activity(
            self.role_id,
            f"Stitched {len(timeline.visual_track)} V1 cuts ({timeline.total_duration:.1f}s) with {chapter_bridges} J/L chapter bridges & continuous A3 BGM",
            detail=f"Clamped {clamped_handles} trim handles inside usable_in/usable_out",
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="title_designer",
            message_type="handoff",
            subject="Continuous master timeline assembled and handles verified",
            payload=assembly_summary,
        )

        return {
            **assembly_summary,
            "assembly_report_path": str(report_path),
            "timeline_path": str(tl_path),
        }
