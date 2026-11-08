"""Stage 30: Dual Picture Editors — DialogueEditorAgent (3a) & MontageEditorAgent (3b)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..models.schema import (
    AudioTrackItem,
    CulledSubClip,
    EditTimeline,
    KenBurnsConfig,
    SubtitleConfig,
    SubtitleItem,
    TimelineMarker,
    TimelineSettings,
    TitleOverlayItem,
    TransitionConfig,
    VisualTrackItem,
)
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


def _ensure_timeline_initialized(blackboard: PipelineBlackboard) -> EditTimeline:
    """Create or reset the shared EditTimeline on the blackboard before Stage 30 assembly."""
    brief = blackboard.creative_brief
    paper = blackboard.paper_edit
    media_index = blackboard.media_index
    if paper is None or media_index is None:
        raise RuntimeError("Stage 30 Picture Editors require paper_edit and media_index on the blackboard.")

    subclip_map: dict[str, CulledSubClip] = {s.subclip_id: s for s in media_index.subclips}
    photo_mode = brief.photo_mode if brief else "gallery_frame"

    visual_track: list[VisualTrackItem] = []
    broll_track: list[VisualTrackItem] = []
    title_track: list[TitleOverlayItem] = []
    markers: list[TimelineMarker] = []

    kb_presets = [
        "floating_gallery",
        "zoom_in",
        "pan_left_to_right",
        "zoom_out",
        "pan_right_to_left",
        "tilt_up",
    ]
    marker_colors = ["Green", "Yellow", "Sand", "Purple", "Blue"]

    cursor = 0.0
    clip_counter = 1
    broll_counter = 1

    # 1. Optional Cold-Open Teaser Hook (0.0s - ~3.4s) before the Main Opening Title Card
    hook_sc = subclip_map.get(paper.cold_open_subclip_id) if paper.cold_open_subclip_id else None
    if hook_sc is not None and hook_sc.media_type == "video":
        hook_in = paper.cold_open_in_tc if paper.cold_open_in_tc is not None else hook_sc.in_tc
        hook_out = paper.cold_open_out_tc if paper.cold_open_out_tc is not None else min(hook_sc.out_tc, hook_in + 3.2)
        hook_dur = round(max(2.0, min(3.8, hook_out - hook_in)), 2)
        markers.append(
            TimelineMarker(
                time_sec=0.0,
                name="Cold Open Teaser Hook",
                color="Blue",
                note=paper.cold_open_rationale or "First-5-seconds high-retention teaser hook",
            )
        )
        visual_track.append(
            VisualTrackItem(
                id="clip_000_hook",
                type="video",
                source=hook_sc.source_path,
                track_lane=1,
                roll_type=hook_sc.roll_type,
                shot_scale=hook_sc.shot_scale,
                act="Cold Open Teaser",
                timeline_start=0.0,
                duration=hook_dur,
                source_in=round(hook_in, 2),
                source_out=round(hook_in + hook_dur, 2),
                speed=1.0,
                volume=1.0 if hook_sc.roll_type == "a_roll" else 0.25,
                j_l_cut_audio_offset=0.0,
                audio_crossfade_sec=0.25,
                ken_burns=None,
                transition_in=TransitionConfig(type="fade_black", duration=0.25),
                transition_out=TransitionConfig(type="fade_black", duration=0.25),
                narrative_note=f"[COLD OPEN HOOK] {paper.cold_open_hook_caption or hook_sc.summary}",
            )
        )
        cursor = round(hook_dur, 3)

    # 2. Main Opening Title Card on Track V3 (drops right after Cold-Open Hook if present)
    clean_proj_title = paper.project_title.replace("_", " ").strip() or "Featured Story"
    if len(clean_proj_title) > 42:
        clean_proj_title = " ".join(clean_proj_title.split()[:5]).title()
    else:
        clean_proj_title = clean_proj_title.title()
    opening_sub = "A Curated Documentary Film"

    opening_start = round(cursor + 0.35, 2)
    title_track.append(
        TitleOverlayItem(
            id="title_opening",
            style="opening_card",
            text=clean_proj_title,
            subtitle=opening_sub,
            timeline_start=opening_start,
            duration=3.6,
            fade_in=0.45,
            fade_out=0.55,
            accent_color="#E8E4DC",
        )
    )

    for c_idx, chapter in enumerate(paper.chapters):
        chapter_start = round(cursor, 3)
        markers.append(
            TimelineMarker(
                time_sec=chapter_start,
                name=chapter.title,
                color=marker_colors[c_idx % len(marker_colors)],
                note=chapter.narrative_goal,
            )
        )

        for s_pos, sid in enumerate(chapter.selected_subclip_ids):
            sc = subclip_map.get(sid)
            if sc is None:
                continue

            clip_id = f"clip_{clip_counter:03d}"
            clip_counter += 1

            # Choose semantic transition based on narrative boundary and clip content
            is_chap_opener = s_pos == 0
            raw_trans = sc.recommended_transition or "crossfade"
            in_trans_type = "fade_black" if is_chap_opener else raw_trans
            out_trans_type = "fade_black" if (s_pos == len(chapter.selected_subclip_ids) - 1) else raw_trans
            in_trans_dur = 0.0 if in_trans_type in ("cut", "none") else (0.40 if in_trans_type == "fade_black" else 0.32)
            out_trans_dur = 0.0 if out_trans_type in ("cut", "none") else (0.40 if out_trans_type == "fade_black" else 0.32)

            if sc.media_type == "video":
                vol = 1.0 if sc.roll_type == "a_roll" else 0.10
                jl_offset = -0.35 if sc.roll_type == "a_roll" else 0.0
                v_item = VisualTrackItem(
                    id=clip_id,
                    type="video",
                    source=sc.source_path,
                    track_lane=1,
                    roll_type=sc.roll_type,
                    shot_scale=sc.shot_scale,
                    act=chapter.title,
                    timeline_start=round(cursor, 3),
                    duration=round(sc.duration, 2),
                    source_in=round(sc.in_tc, 2),
                    source_out=round(sc.out_tc, 2),
                    speed=1.0,
                    volume=vol,
                    j_l_cut_audio_offset=jl_offset,
                    audio_crossfade_sec=0.33,
                    ken_burns=None,
                    transition_in=TransitionConfig(type=in_trans_type, duration=in_trans_dur),  # type: ignore[arg-type]
                    transition_out=TransitionConfig(type=out_trans_type, duration=out_trans_dur),  # type: ignore[arg-type]
                    narrative_note=f"[{sc.roll_type.upper()}] {sc.annotation_title or sc.summary}",
                )
                visual_track.append(v_item)

                # Add contextual bilingual lower_third scene caption for B-roll video clips only when a curated non-boilerplate title is present
                is_mechanical_vid_title = bool(
                    sc.annotation_title
                    and ("— Scene " in sc.annotation_title or "Highlight from " in sc.annotation_title)
                )
                if sc.annotation_title and not is_mechanical_vid_title and sc.roll_type == "b_roll" and v_item.duration >= 3.6:
                    lt_start = round(v_item.timeline_start + 0.5, 2)
                    lt_dur = round(min(3.2, max(2.0, v_item.duration - 1.0)), 2)
                    if not any(abs(t.timeline_start - lt_start) < 4.0 for t in title_track):
                        title_track.append(
                            TitleOverlayItem(
                                id=f"title_lt_{clip_id}",
                                style="lower_third",
                                text=sc.annotation_title,
                                subtitle=sc.annotation_subtitle or "",
                                timeline_start=lt_start,
                                duration=lt_dur,
                                fade_in=0.35,
                                fade_out=0.35,
                                accent_color="#E8E4DC",
                            )
                        )

                # Check if this A-Roll clip has Track V2 B-Roll cutaways mapped in broll_anchor_map
                for cutaway_sid in chapter.broll_anchor_map.get(sc.subclip_id, []):
                    cut_sc = subclip_map.get(cutaway_sid)
                    if cut_sc is None:
                        continue
                    cut_dur = round(min(3.2, max(2.2, v_item.duration * 0.35)), 2)
                    cut_offset = round(max(1.8, (v_item.duration - cut_dur) * 0.5), 2)
                    cut_start = round(v_item.timeline_start + cut_offset, 3)
                    preset = kb_presets[broll_counter % len(kb_presets)]
                    broll_track.append(
                        VisualTrackItem(
                            id=f"broll_{broll_counter:03d}",
                            type=cut_sc.media_type,
                            source=cut_sc.source_path,
                            track_lane=2,
                            roll_type=cut_sc.roll_type,
                            shot_scale=cut_sc.shot_scale,
                            act=chapter.title,
                            timeline_start=cut_start,
                            duration=cut_dur,
                            source_in=cut_sc.in_tc,
                            source_out=round(cut_sc.in_tc + cut_dur, 2),
                            speed=1.0,
                            volume=0.0,
                            j_l_cut_audio_offset=0.0,
                            audio_crossfade_sec=0.33,
                            ken_burns=(
                                KenBurnsConfig(
                                    mode=(
                                        "diptych_frame"
                                        if cut_sc.secondary_source_path and cut_sc.recommended_photo_mode == "parallax_gallery"
                                        else (cut_sc.recommended_photo_mode or photo_mode)
                                    ),  # type: ignore[arg-type]
                                    overlay_fx=cut_sc.overlay_fx,
                                    preset=preset,  # type: ignore[arg-type]
                                    start_zoom=1.0,
                                    end_zoom=1.08,
                                    start_center=cut_sc.focal_point,
                                    end_center=(cut_sc.focal_point[0], max(0.2, cut_sc.focal_point[1] - 0.03)),
                                    secondary_source=cut_sc.secondary_source_path,
                                    tertiary_source=cut_sc.tertiary_source_path,
                                    caption_title=cut_sc.annotation_title,
                                    caption_subtitle=cut_sc.annotation_subtitle,
                                    camera_badge=cut_sc.camera_badge,
                                )
                                if cut_sc.media_type == "photo"
                                else None
                            ),
                            transition_in=TransitionConfig(type="crossfade", duration=0.30),
                            transition_out=TransitionConfig(type="crossfade", duration=0.30),
                            narrative_note=f"[Track V2 Cutaway over {clip_id}] {cut_sc.summary}",
                        )
                    )
                    broll_counter += 1

                cursor += v_item.duration
            else:
                preset = kb_presets[clip_counter % len(kb_presets)]
                p_item = VisualTrackItem(
                    id=clip_id,
                    type="photo",
                    source=sc.source_path,
                    track_lane=1,
                    roll_type="still",
                    shot_scale=sc.shot_scale,
                    act=chapter.title,
                    timeline_start=round(cursor, 3),
                    duration=round(sc.duration, 2),
                    source_in=0.0,
                    source_out=round(sc.duration, 2),
                    speed=1.0,
                    volume=0.0,
                    j_l_cut_audio_offset=0.0,
                    audio_crossfade_sec=0.0,
                    ken_burns=KenBurnsConfig(
                        mode=(
                            "diptych_frame"
                            if sc.secondary_source_path and sc.recommended_photo_mode == "parallax_gallery"
                            else (sc.recommended_photo_mode or photo_mode)
                        ),  # type: ignore[arg-type]
                        overlay_fx=sc.overlay_fx,
                        preset=preset,  # type: ignore[arg-type]
                        start_zoom=1.0,
                        end_zoom=1.09,
                        start_center=sc.focal_point,
                        end_center=(sc.focal_point[0], max(0.2, sc.focal_point[1] - 0.03)),
                        secondary_source=sc.secondary_source_path,
                        tertiary_source=sc.tertiary_source_path,
                        caption_title=sc.annotation_title,
                        caption_subtitle=sc.annotation_subtitle,
                        camera_badge=sc.camera_badge,
                    ),
                    transition_in=TransitionConfig(type=in_trans_type, duration=in_trans_dur),  # type: ignore[arg-type]
                    transition_out=TransitionConfig(type=out_trans_type, duration=out_trans_dur),  # type: ignore[arg-type]
                    narrative_note=(
                        f"[STILL · {sc.annotation_title}] {sc.annotation_subtitle}"
                        if sc.annotation_title
                        else f"[STILL] {sc.summary}"
                    ),
                )
                visual_track.append(p_item)
                cursor += p_item.duration

    timeline = EditTimeline(
        project_name=paper.project_title,
        user_prompt=brief.raw_user_prompt if brief else blackboard.user_prompt,
        story_arc_summary=paper.story_arc_summary,
        settings=TimelineSettings(
            width=blackboard.width,
            height=blackboard.height,
            fps=blackboard.fps,
            hdr=blackboard.hdr,
        ),
        markers=markers,
        visual_track=visual_track,
        broll_track=broll_track,
        title_track=title_track,
        effects_track=[],
        audio_tracks=[],
        subtitles=SubtitleConfig(enabled=True, burn_in=True, items=[]),
    )
    blackboard.timeline = timeline
    return timeline


@register_agent
class DialogueEditorAgent(BaseEditingAgent):
    """Agent 3a (Stage 30): Trims A-Roll on word boundaries, applies J/L cuts, and anchors Track V2 B-Roll."""

    role_id = "dialogue_editor"
    display_name = "3a. Dialogue & Narrative Editor"
    stage_order = 30
    description = (
        "Trims A-roll clips on word timestamps, sets -0.35s J-cut audio lead-ins, populates bilingual "
        "subtitles, and places Track V2 B-roll cutaways over long dialogue takes."
    )
    enabled_by_default = True
    requires_artifacts = ["paper_edit.json", "media_index.json"]
    produces_artifacts = []
    skill_reference_file = "03_picture_editors.md"

    def run(self, blackboard: PipelineBlackboard, realign_only: bool = False) -> dict[str, Any]:
        timeline = (
            (blackboard.timeline or _ensure_timeline_initialized(blackboard))
            if realign_only
            else _ensure_timeline_initialized(blackboard)
        )
        catalog = blackboard.catalog

        # Map source path -> VideoAsset for exact dialogue segment lookup
        video_by_path: dict[str, Any] = {}
        if catalog is not None:
            for v in catalog.videos:
                video_by_path[str(Path(v.path).resolve())] = v

        sub_items: list[SubtitleItem] = []
        sub_counter = 1
        aroll_clips_processed = 0

        for item in timeline.visual_track:
            if item.type == "video" and item.roll_type == "a_roll":
                aroll_clips_processed += 1
                item.volume = 1.0
                item.j_l_cut_audio_offset = -0.35
                item.audio_crossfade_sec = 0.33

                v_asset = video_by_path.get(str(Path(item.source).resolve()))
                if v_asset and v_asset.dialogues:
                    for d in v_asset.dialogues:
                        if d.end_time <= item.source_in or d.start_time >= (item.source_in + item.duration):
                            continue
                        rel_start = max(0.0, d.start_time - item.source_in)
                        rel_end = min(item.duration, d.end_time - item.source_in)
                        if rel_end - rel_start < 0.25:
                            continue
                        sub_items.append(
                            SubtitleItem(
                                id=f"sub_{sub_counter:03d}",
                                start=round(item.timeline_start + rel_start, 2),
                                end=round(item.timeline_start + rel_end, 2),
                                speaker=d.speaker,
                                text=d.text,
                                secondary_text=d.translation,
                            )
                        )
                        sub_counter += 1

        timeline.subtitles.items = sub_items
        blackboard.emit_activity(
            self.role_id,
            f"Cut {aroll_clips_processed} A-Roll dialogue clips with -0.35s J-cuts, {len(sub_items)} bilingual subtitles & {len(timeline.broll_track)} V2 cutaways",
            detail="Applied 0.33s room-tone audio crossfades across all dialogue boundaries",
        )

        # Check if ProductionDirectorAgent directed a dialogue highlight badge on Track V3
        inject_badge = any(
            msg.message_type == "director_directive" and msg.payload.get("inject_dialogue_highlight_badge")
            for msg in blackboard.get_messages_for(self.role_id)
        )
        if inject_badge and sub_items and not any(t.style == "dialogue_badge" for t in timeline.title_track):
            best_sub = max(sub_items, key=lambda s: len(s.text))
            badge_start = round(best_sub.start + 0.2, 2)
            # Ensure badge does not collide with existing title_track items
            if not any(abs(t.timeline_start - badge_start) < 3.2 for t in timeline.title_track):
                timeline.title_track.append(
                    TitleOverlayItem(
                        id="title_director_quote_badge",
                        style="dialogue_badge",
                        text=f'"{best_sub.text[:36]}"',
                        subtitle=best_sub.secondary_text or "Candid Spoken Moment",
                        timeline_start=badge_start,
                        duration=min(3.2, max(1.8, best_sub.end - best_sub.start)),
                        fade_in=0.3,
                        fade_out=0.3,
                        accent_color="#E8E4DC",
                    )
                )

        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="montage_editor",
            message_type="handoff",
            subject="A-Roll dialogue trims, J-cuts, subtitles, and V2 cutaway anchors placed",
            payload={
                "aroll_clips": aroll_clips_processed,
                "subtitle_count": len(sub_items),
                "v2_broll_cutaways": len(timeline.broll_track),
            },
        )

        return {
            "aroll_clips_processed": aroll_clips_processed,
            "subtitle_count": len(sub_items),
            "v2_broll_cutaways": len(timeline.broll_track),
        }


@register_agent
class MontageEditorAgent(BaseEditingAgent):
    """Agent 3b (Stage 30): Beat-snaps B-roll & stills, sequences scale/motion, and configures 3-layer audio."""

    role_id = "montage_editor"
    display_name = "3b. Rhythm & Montage Editor"
    stage_order = 30
    description = (
        "Snaps B-roll and photo durations to the musical beat grid, enforces Wide -> Medium -> Close-up "
        "visual progression, sets -20dB Nat-Sound with 0.33s crossfades, and configures -14dB auto-ducking."
    )
    enabled_by_default = True
    requires_artifacts = ["paper_edit.json", "beat_grid.json"]
    produces_artifacts = []
    skill_reference_file = "03_picture_editors.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline or _ensure_timeline_initialized(blackboard)
        beat_grid = blackboard.beat_grid
        brief = blackboard.creative_brief
        max_broll = brief.max_broll_duration_sec if brief else 4.5

        # Record parent A-roll clip IDs for V2 B-roll cutaways before snapping V1 durations
        v2_parent_info: list[tuple[VisualTrackItem, Optional[VisualTrackItem], float]] = []
        for b_item in timeline.broll_track:
            parent = next(
                (
                    v
                    for v in timeline.visual_track
                    if v.timeline_start <= b_item.timeline_start < (v.timeline_start + v.duration)
                ),
                None,
            )
            rel_offset = (b_item.timeline_start - parent.timeline_start) if parent else 2.0
            v2_parent_info.append((b_item, parent, rel_offset))

        # Snap B-roll and photo durations on Track V1 to musical beat intervals,
        # giving multi-panel layouts and dense bilingual captions comfortable reading hold time
        v1_with_v2_cutaway = {p.id for _, p, _ in v2_parent_info if p is not None}
        snapped_count = 0
        scale_cycle = ["wide", "medium", "closeup"]
        is_extended_cut = (brief.target_duration_sec if brief else 110.0) >= 135.0
        for idx, item in enumerate(timeline.visual_track):
            if item.id == "clip_000_hook":
                continue
            if item.roll_type in ("b_roll", "still"):
                cap = max(max_broll, 7.0) if item.id in v1_with_v2_cutaway else max_broll
                target_d = min(cap, item.duration)
                min_hold = 2.4
                if item.type == "photo" and item.ken_burns is not None and is_extended_cut:
                    kb_mode = item.ken_burns.mode
                    cap_len = len(item.ken_burns.caption_title or "") + len(item.ken_burns.caption_subtitle or "")
                    if kb_mode in ("diptych_frame", "triptych_cinematic"):
                        min_hold = min(cap, 4.8)
                        target_d = max(target_d, min_hold)
                    elif cap_len >= 42:
                        min_hold = min(cap, 4.2)
                        target_d = max(target_d, min_hold)

                if beat_grid is not None:
                    snapped_d = beat_grid.snap_duration_to_beat(target_d, min_dur=min(min_hold, cap), max_dur=cap)
                else:
                    snapped_d = round(min(cap, max(min_hold, target_d)), 2)

                item.duration = snapped_d
                item.source_out = round(item.source_in + snapped_d, 2)
                if item.roll_type == "b_roll":
                    item.volume = 0.10  # -20 dB Nat-Sound bed
                    item.audio_crossfade_sec = 0.33
                else:
                    item.volume = 0.0
                if item.shot_scale == "medium":
                    item.shot_scale = scale_cycle[idx % len(scale_cycle)]  # type: ignore[assignment]
                snapped_count += 1

        # Recalculate sequential Track V1 timeline_start positions and update subtitles + V2 cutaways + lower_thirds
        timeline.recalculate_sequential_starts()
        new_starts = {item.id: item.timeline_start for item in timeline.visual_track}
        clip_by_id = {item.id: item for item in timeline.visual_track}
        kept_titles: list[TitleOverlayItem] = []
        for t_item in timeline.title_track:
            if t_item.style == "act_banner":
                continue
            if t_item.id.startswith("title_lt_clip_"):
                cid_ref = t_item.id.replace("title_lt_", "")
                parent_v = clip_by_id.get(cid_ref)
                if parent_v is None or parent_v.type != "video" or parent_v.duration < 3.2:
                    continue
                t_item.timeline_start = round(parent_v.timeline_start + 0.45, 2)
                t_item.duration = round(min(t_item.duration, max(1.8, parent_v.duration - 0.9)), 2)
            kept_titles.append(t_item)
        timeline.title_track = kept_titles

        # Re-align subtitles to updated V1 A-roll start times
        if blackboard.catalog is not None:
            DialogueEditorAgent().run(blackboard, realign_only=True)

        # Re-align Track V2 B-Roll cutaways to their parent A-roll clips and beat-snap their durations
        for b_item, parent, rel_offset in v2_parent_info:
            if beat_grid is not None:
                b_item.duration = beat_grid.snap_duration_to_beat(b_item.duration, min_dur=2.0, max_dur=3.5)
                b_item.source_out = round(b_item.source_in + b_item.duration, 2)
            if parent is not None and parent.id in new_starts:
                b_item.timeline_start = round(new_starts[parent.id] + rel_offset, 3)

        # Honor any ProductionDirectorAgent directives for Ken Burns motion variety and title collision spacing
        kb_cycle = [
            "floating_gallery",
            "zoom_in",
            "pan_left_to_right",
            "zoom_out",
            "pan_right_to_left",
            "tilt_up",
        ]
        prev_preset = None
        for idx, item in enumerate(timeline.visual_track):
            if item.type == "photo" and item.ken_burns is not None:
                if item.ken_burns.preset == prev_preset:
                    item.ken_burns.preset = kb_cycle[(idx + 1) % len(kb_cycle)]  # type: ignore[assignment]
                prev_preset = item.ken_burns.preset

        # Sort title_track and drop any non-opening/closing lower-third that collides with an existing title or lands on a photo plate
        timeline.title_track.sort(key=lambda t: t.timeline_start)
        non_colliding_titles: list[TitleOverlayItem] = []
        for cur_t in timeline.title_track:
            if cur_t.style not in ("opening_card", "closing_card"):
                mid_t = cur_t.timeline_start + cur_t.duration * 0.5
                under_clip = next(
                    (c for c in timeline.visual_track if c.timeline_start <= mid_t <= (c.timeline_start + c.duration)),
                    None,
                )
                if under_clip is not None and under_clip.type == "photo" and under_clip.caption_title:
                    continue
            if non_colliding_titles:
                prev_t = non_colliding_titles[-1]
                if cur_t.timeline_start < prev_t.timeline_start + prev_t.duration + 1.0:
                    if cur_t.style in ("opening_card", "closing_card"):
                        non_colliding_titles[-1] = cur_t
                    continue
            non_colliding_titles.append(cur_t)
        timeline.title_track = non_colliding_titles

        # Configure Track A3 Background Music with -14dB auto-ducking & 1.0s smooth ramps
        audio_source = ""
        if beat_grid and beat_grid.source_audio and Path(beat_grid.source_audio).exists():
            audio_source = beat_grid.source_audio
        elif blackboard.extra_audio_paths:
            audio_source = str(blackboard.extra_audio_paths[0])
        elif blackboard.catalog and blackboard.catalog.audio_files:
            audio_source = blackboard.catalog.audio_files[0].path

        if audio_source and Path(audio_source).exists():
            timeline.audio_tracks = [
                AudioTrackItem(
                    id="bgm_main",
                    role="music",
                    source=audio_source,
                    timeline_start=0.0,
                    duration=round(timeline.total_duration, 2),
                    source_in=0.0,
                    volume=0.28,
                    duck_under_dialogue=True,
                    duck_volume=0.07,
                    duck_ramp_sec=1.0,
                    fade_in=1.0,
                    fade_out=2.5,
                )
            ]

        # Optional AI Creative Intelligence refinement of transitions, hold durations, and lower-third captions
        ai_montage_resp = self.ask_ai_creative_json(
            blackboard=blackboard,
            prompt=(
                f"Assembled Timeline ({len(timeline.visual_track)} V1 cuts, {timeline.total_duration:.1f}s).\n"
                "Optionally refine clip transitions ('cut', 'crossfade', 'dissolve', 'fade_black'), photo modes, or add "
                "Track V3 lower_third scene captions. Return JSON compatible with apply_timeline_intelligence_overrides."
            ),
            decision_key="montage_editor",
            timeout_sec=30.0,
        )
        if isinstance(ai_montage_resp, dict) and (
            ai_montage_resp.get("clips") or ai_montage_resp.get("lower_thirds")
        ):
            _apply_timeline_intelligence_to_obj(timeline, ai_montage_resp)

        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="showrunner",
            message_type="handoff",
            subject="Montage beat-snapping & 3-layer audio mix complete",
            payload={
                "snapped_clips": snapped_count,
                "total_duration": round(timeline.total_duration, 2),
                "has_bgm": bool(timeline.audio_tracks),
            },
        )
        blackboard.emit_activity(
            self.role_id,
            f"Beat-snapped {snapped_count} B-Roll & 2.5D photo plates -> Total timeline runtime: {timeline.total_duration:.1f}s",
            detail="Alternated Wide -> Medium -> Close-up shot scales, semantic transitions, and Track V3 lower-thirds",
        )

        return {
            "snapped_clips": snapped_count,
            "total_duration": round(timeline.total_duration, 2),
        }


def _apply_timeline_intelligence_to_obj(timeline: EditTimeline, overrides: dict[str, Any]) -> int:
    """Apply AI transition, duration, photo mode, and lower-third overrides directly to an EditTimeline."""
    clips_ov = overrides.get("clips") or {}
    if isinstance(clips_ov, list):
        clips_ov = {str(c.get("id") or c.get("clip_id")): c for c in clips_ov if isinstance(c, dict)}

    updated = 0
    for item in timeline.visual_track:
        u = clips_ov.get(item.id)
        if not isinstance(u, dict):
            continue
        if u.get("transition_in") in ("none", "cut", "crossfade", "dissolve", "fade_black", "dip_to_white"):
            t_type = u["transition_in"]
            item.transition_in = TransitionConfig(
                type=t_type,  # type: ignore[arg-type]
                duration=0.0 if t_type in ("none", "cut") else float(u.get("transition_duration", 0.35)),
            )
        if u.get("transition_out") in ("none", "cut", "crossfade", "dissolve", "fade_black", "dip_to_white"):
            t_type = u["transition_out"]
            item.transition_out = TransitionConfig(
                type=t_type,  # type: ignore[arg-type]
                duration=0.0 if t_type in ("none", "cut") else float(u.get("transition_duration", 0.35)),
            )
        if isinstance(u.get("duration"), (int, float)):
            new_d = max(1.8, min(12.0, float(u["duration"])))
            item.duration = round(new_d, 2)
            item.source_out = round(item.source_in + new_d, 2)
        if item.ken_burns is not None:
            if u.get("photo_mode"):
                item.ken_burns.mode = u["photo_mode"]
            if u.get("overlay_fx"):
                item.ken_burns.overlay_fx = u["overlay_fx"]
            if u.get("caption_title"):
                item.ken_burns.caption_title = str(u["caption_title"])
            if u.get("caption_subtitle"):
                item.ken_burns.caption_subtitle = str(u["caption_subtitle"])
        updated += 1

    if updated > 0:
        timeline.recalculate_sequential_starts()
        for a_item in timeline.audio_tracks:
            if a_item.role == "music":
                a_item.duration = round(timeline.total_duration, 2)

    lt_list = overrides.get("lower_thirds")
    if isinstance(lt_list, list):
        clip_by_id = {c.id: c for c in timeline.visual_track}
        for idx_lt, lt in enumerate(lt_list):
            if not isinstance(lt, dict) or not lt.get("text"):
                continue
            target_cid = str(lt.get("clip_id") or "")
            start_t = float(lt.get("timeline_start", 0.0))
            if target_cid in clip_by_id:
                start_t = round(clip_by_id[target_cid].timeline_start + 0.5, 2)
            lt_id = str(lt.get("id") or f"title_lt_ai_{idx_lt + 1:02d}")
            existing = next((t for t in timeline.title_track if t.id == lt_id), None)
            if existing:
                existing.text = str(lt["text"])
                existing.subtitle = str(lt.get("subtitle") or "")
                existing.timeline_start = start_t
            else:
                timeline.title_track.append(
                    TitleOverlayItem(
                        id=lt_id,
                        style="lower_third",
                        text=str(lt["text"]),
                        subtitle=str(lt.get("subtitle") or ""),
                        timeline_start=start_t,
                        duration=float(lt.get("duration", 3.2)),
                        fade_in=0.30,
                        fade_out=0.30,
                        accent_color=str(lt.get("accent_color") or "#E8E4DC"),
                    )
                )
            updated += 1

    return updated


def apply_timeline_intelligence_overrides(
    project_dir: str | Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Persist AI Agent transition, hold duration, photo mode, and lower-third caption overrides to timeline.yaml."""
    import json as _json

    proj_path = Path(project_dir).resolve()
    tl_path = proj_path / "timeline.yaml"
    if not tl_path.exists():
        raise FileNotFoundError(f"timeline.yaml not found in {proj_path}")

    timeline = EditTimeline.from_yaml(tl_path)
    updated = _apply_timeline_intelligence_to_obj(timeline, overrides)
    timeline.to_yaml(tl_path)

    decisions_dir = proj_path / "ai_decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    (decisions_dir / "montage_editor.json").write_text(
        _json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "updated_items": updated,
        "total_duration": round(timeline.total_duration, 2),
        "timeline_path": str(tl_path),
    }
