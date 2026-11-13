"""Stage 30: Dual Picture Editors — DialogueEditorAgent (3a) & MontageEditorAgent (3b)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..models.schema import (
    AudioTrackItem,
    BeatGrid,
    CastManifest,
    ChapterPlan,
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
from .cast_continuity_agent import compute_face_safe_ken_burns_centers


KB_PRESETS = [
    "floating_gallery",
    "zoom_in",
    "pan_left_to_right",
    "zoom_out",
    "pan_right_to_left",
    "tilt_up",
]
MARKER_COLORS = ["Green", "Yellow", "Sand", "Purple", "Blue"]


def _get_bar_duration(beat_grid: Optional[BeatGrid]) -> float:
    """Return nominal bar / downbeat interval in seconds from BeatGrid."""
    if beat_grid is None:
        return 2.0
    if len(beat_grid.downbeats) >= 2:
        diffs = [
            beat_grid.downbeats[i + 1] - beat_grid.downbeats[i]
            for i in range(min(8, len(beat_grid.downbeats) - 1))
            if beat_grid.downbeats[i + 1] > beat_grid.downbeats[i] + 0.2
        ]
        if diffs:
            return round(sum(diffs) / len(diffs), 4)
    bpm = max(40.0, float(beat_grid.tempo_bpm or 120.0))
    return round(4.0 * 60.0 / bpm, 4)


def _snap_montage_duration(
    target_d: float,
    min_dur: float,
    max_dur: float,
    beat_grid: Optional[BeatGrid],
    prefer_downbeat: bool = False,
) -> float:
    """Snap a clip hold duration to downbeat bar multiples (for music_montage) or beat intervals."""
    lo = min(min_dur, max_dur)
    hi = max(lo, max_dur)
    if beat_grid is None:
        return round(min(hi, max(lo, target_d)), 2)

    if prefer_downbeat:
        bar_sec = _get_bar_duration(beat_grid)
        if bar_sec > 0.25:
            candidates = [
                round(k * bar_sec, 2)
                for k in range(1, 12)
                if (lo - 1e-3) <= round(k * bar_sec, 2) <= (hi + 1e-3)
            ]
            if candidates:
                return min(candidates, key=lambda c: abs(c - target_d))

    return beat_grid.snap_duration_to_beat(target_d, min_dur=lo, max_dur=hi)


def _nearest_downbeat_time(cursor: float, beat_grid: Optional[BeatGrid], min_time: float = 0.0) -> float:
    """Find the nearest musical downbeat timestamp >= min_time."""
    if beat_grid is None:
        return round(cursor, 3)
    bar_sec = _get_bar_duration(beat_grid)
    candidates: list[float] = [d for d in beat_grid.downbeats if d >= min_time - 1e-4]
    if bar_sec > 0.25:
        base = beat_grid.downbeats[0] if beat_grid.downbeats else 0.0
        k_approx = max(0, int(round((cursor - base) / bar_sec)))
        for k in range(max(0, k_approx - 3), k_approx + 5):
            cand = round(base + k * bar_sec, 3)
            if cand >= min_time - 1e-4:
                candidates.append(cand)
    if not candidates:
        return round(cursor, 3)
    return round(min(candidates, key=lambda d: abs(d - cursor)), 3)


def _align_tail_clips_to_downbeat(
    preceding_clips: list[VisualTrackItem],
    current_cursor: float,
    beat_grid: Optional[BeatGrid],
    subclip_by_source: dict[str, list[CulledSubClip]],
    max_broll: float = 4.5,
) -> float:
    """Adjust the tail clip(s) before a music_montage chapter so the chapter starts on an exact downbeat."""
    if not preceding_clips or beat_grid is None:
        return round(current_cursor, 3)

    prefix_sum = round(sum(c.duration for c in preceding_clips[:-1]), 3)
    last_clip = preceding_clips[-1]

    # Determine safe min/max duration bounds for last_clip
    min_d = 1.6
    max_d = 8.0
    if last_clip.roll_type == "b_roll":
        max_d = max_broll
    elif last_clip.type == "video":
        # Respect elastic usable_out handle when available
        matches = subclip_by_source.get(last_clip.source, [])
        usable_hi = max((m.usable_out for m in matches), default=last_clip.source_in + max_d)
        max_d = max(min_d, min(10.0, usable_hi - last_clip.source_in))

    min_target_cursor = prefix_sum + min_d
    max_target_cursor = prefix_sum + max_d

    bar_sec = _get_bar_duration(beat_grid)
    candidates: list[float] = [
        round(d, 3)
        for d in beat_grid.downbeats
        if (min_target_cursor - 1e-3) <= d <= (max_target_cursor + 1e-3)
    ]
    if bar_sec > 0.25:
        base = beat_grid.downbeats[0] if beat_grid.downbeats else 0.0
        for k in range(0, 200):
            cand = round(base + k * bar_sec, 3)
            if cand > max_target_cursor + bar_sec:
                break
            if (min_target_cursor - 1e-3) <= cand <= (max_target_cursor + 1e-3):
                candidates.append(cand)

    if candidates:
        best_cursor = min(candidates, key=lambda d: abs(d - current_cursor))
        new_dur = round(best_cursor - prefix_sum, 3)
        last_clip.duration = new_dur
        last_clip.source_out = round(last_clip.source_in + new_dur, 3)
        return round(prefix_sum + new_dur, 3)

    # If a single tail clip cannot span the entire delta (e.g. tight B-roll cap), distribute across last 2 clips
    if len(preceding_clips) >= 2:
        prev_clip = preceding_clips[-2]
        prefix2 = round(sum(c.duration for c in preceding_clips[:-2]), 3)
        prev_max = max_broll if prev_clip.roll_type == "b_roll" else 8.0
        min_c2 = prefix2 + 1.6 + min_d
        max_c2 = prefix2 + prev_max + max_d
        cands2 = [
            round(d, 3)
            for d in beat_grid.downbeats
            if (min_c2 - 1e-3) <= d <= (max_c2 + 1e-3)
        ]
        if not cands2 and bar_sec > 0.25:
            base = beat_grid.downbeats[0] if beat_grid.downbeats else 0.0
            for k in range(0, 200):
                cand = round(base + k * bar_sec, 3)
                if (min_c2 - 1e-3) <= cand <= (max_c2 + 1e-3):
                    cands2.append(cand)
        if cands2:
            best_cursor = min(cands2, key=lambda d: abs(d - current_cursor))
            total_pair = round(best_cursor - prefix2, 3)
            d1 = round(min(prev_max, max(1.6, total_pair * 0.5)), 3)
            d2 = round(total_pair - d1, 3)
            if d2 > max_d:
                d2 = round(max_d, 3)
                d1 = round(total_pair - d2, 3)
            prev_clip.duration = d1
            prev_clip.source_out = round(prev_clip.source_in + d1, 3)
            last_clip.duration = d2
            last_clip.source_out = round(last_clip.source_in + d2, 3)
            return round(prefix2 + d1 + d2, 3)

    return round(current_cursor, 3)


def _build_ken_burns_for_subclip(
    sc: CulledSubClip,
    preset: str,
    photo_mode: str,
    cast_manifest: Optional[CastManifest],
    start_zoom: float = 1.0,
    end_zoom: float = 1.09,
) -> KenBurnsConfig:
    """Construct a KenBurnsConfig with CastContinuityAgent face-safe framing bias."""
    bbox: Optional[list[float]] = None
    if cast_manifest is not None:
        det = cast_manifest.subclip_detections.get(sc.subclip_id)
        if det is not None and det.bbox and len(det.bbox) == 4:
            bbox = det.bbox

    start_center, end_center = compute_face_safe_ken_burns_centers(
        bbox,
        orientation=sc.orientation,
        start_zoom=start_zoom,
        end_zoom=end_zoom,
        fallback_focal=sc.focal_point,
    )
    resolved_mode = (
        "diptych_frame"
        if sc.secondary_source_path and sc.recommended_photo_mode == "parallax_gallery"
        else (sc.recommended_photo_mode or photo_mode)
    )
    return KenBurnsConfig(
        mode=resolved_mode,  # type: ignore[arg-type]
        overlay_fx=sc.overlay_fx,
        preset=preset,  # type: ignore[arg-type]
        start_zoom=start_zoom,
        end_zoom=end_zoom,
        start_center=start_center,
        end_center=end_center,
        secondary_source=sc.secondary_source_path,
        tertiary_source=sc.tertiary_source_path,
        caption_title=sc.annotation_title,
        caption_subtitle=sc.annotation_subtitle,
        camera_badge=sc.camera_badge,
    )


def _build_chapter_items(
    chapter: ChapterPlan,
    subclip_map: dict[str, CulledSubClip],
    photo_mode: str,
    cast_manifest: Optional[CastManifest],
    clip_counter_start: int,
    broll_counter_start: int,
    cursor_start: float,
) -> tuple[list[VisualTrackItem], list[VisualTrackItem], int, int, float]:
    """Build initial V1 and V2 VisualTrackItems for a single chapter."""
    v1_items: list[VisualTrackItem] = []
    v2_items: list[VisualTrackItem] = []
    clip_counter = clip_counter_start
    broll_counter = broll_counter_start
    cursor = cursor_start
    candid_set = set(chapter.candid_reaction_ids or [])

    for s_pos, sid in enumerate(chapter.selected_subclip_ids):
        sc = subclip_map.get(sid)
        if sc is None:
            continue

        clip_id = f"clip_{clip_counter:03d}"
        clip_counter += 1

        is_chap_opener = s_pos == 0
        raw_trans = sc.recommended_transition or "crossfade"
        in_trans_type = "fade_black" if is_chap_opener else raw_trans
        out_trans_type = "fade_black" if (s_pos == len(chapter.selected_subclip_ids) - 1) else raw_trans
        in_trans_dur = 0.0 if in_trans_type in ("cut", "none") else (0.40 if in_trans_type == "fade_black" else 0.32)
        out_trans_dur = 0.0 if out_trans_type in ("cut", "none") else (0.40 if out_trans_type == "fade_black" else 0.32)

        in_tc = sc.golden_in if sc.golden_in is not None else sc.in_tc
        out_tc = sc.golden_out if sc.golden_out is not None else sc.out_tc
        clip_dur = round(max(1.5, out_tc - in_tc), 2)

        if sc.media_type == "video":
            is_hybrid_candid = (
                chapter.editing_mode == "hybrid"
                and (
                    sid in candid_set
                    or (bool(sc.transcript_excerpt.strip()) and 1.5 <= clip_dur <= 4.0)
                )
            )
            eff_roll_type = "a_roll" if (sc.roll_type == "a_roll" or is_hybrid_candid) else sc.roll_type
            if is_hybrid_candid:
                clip_dur = round(min(4.0, max(1.5, clip_dur)), 2)

            vol = 1.0 if eff_roll_type == "a_roll" else 0.10
            jl_offset = -0.35 if eff_roll_type == "a_roll" else 0.0
            # Clamp J-cut offset within usable_in handle
            usable_lo = sc.usable_in if sc.usable_in is not None else 0.0
            if in_tc + jl_offset < usable_lo:
                jl_offset = round(max(-0.35, usable_lo - in_tc), 3)

            v_item = VisualTrackItem(
                id=clip_id,
                type="video",
                source=sc.source_path,
                track_lane=1,
                roll_type=eff_roll_type,
                shot_scale=sc.shot_scale,
                act=chapter.title,
                timeline_start=round(cursor, 3),
                duration=clip_dur,
                source_in=round(in_tc, 3),
                source_out=round(in_tc + clip_dur, 3),
                speed=1.0,
                volume=vol,
                j_l_cut_audio_offset=jl_offset,
                audio_crossfade_sec=0.33,
                ken_burns=None,
                transition_in=TransitionConfig(type=in_trans_type, duration=in_trans_dur),  # type: ignore[arg-type]
                transition_out=TransitionConfig(type=out_trans_type, duration=out_trans_dur),  # type: ignore[arg-type]
                narrative_note=(
                    f"[HYBRID CANDID] {sc.annotation_title or sc.summary}"
                    if is_hybrid_candid
                    else f"[{eff_roll_type.upper()}] {sc.annotation_title or sc.summary}"
                ),
            )
            v1_items.append(v_item)

            for cutaway_sid in chapter.broll_anchor_map.get(sc.subclip_id, []):
                cut_sc = subclip_map.get(cutaway_sid)
                if cut_sc is None:
                    continue
                cut_dur = round(min(3.2, max(2.2, v_item.duration * 0.35)), 2)
                cut_offset = round(max(1.8, (v_item.duration - cut_dur) * 0.5), 2)
                cut_start = round(v_item.timeline_start + cut_offset, 3)
                preset = KB_PRESETS[broll_counter % len(KB_PRESETS)]
                cut_in = cut_sc.golden_in if cut_sc.golden_in is not None else cut_sc.in_tc
                v2_items.append(
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
                        source_in=round(cut_in, 3),
                        source_out=round(cut_in + cut_dur, 3),
                        speed=1.0,
                        volume=0.0,
                        j_l_cut_audio_offset=0.0,
                        audio_crossfade_sec=0.33,
                        ken_burns=(
                            _build_ken_burns_for_subclip(
                                cut_sc,
                                preset=preset,
                                photo_mode=photo_mode,
                                cast_manifest=cast_manifest,
                                start_zoom=1.0,
                                end_zoom=1.08,
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

            cursor = round(cursor + v_item.duration, 3)
        else:
            preset = KB_PRESETS[clip_counter % len(KB_PRESETS)]
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
                ken_burns=_build_ken_burns_for_subclip(
                    sc,
                    preset=preset,
                    photo_mode=photo_mode,
                    cast_manifest=cast_manifest,
                    start_zoom=1.0,
                    end_zoom=1.09,
                ),
                transition_in=TransitionConfig(type=in_trans_type, duration=in_trans_dur),  # type: ignore[arg-type]
                transition_out=TransitionConfig(type=out_trans_type, duration=out_trans_dur),  # type: ignore[arg-type]
                narrative_note=(
                    f"[STILL · {sc.annotation_title}] {sc.annotation_subtitle}"
                    if sc.annotation_title
                    else f"[STILL] {sc.summary}"
                ),
            )
            v1_items.append(p_item)
            cursor = round(cursor + p_item.duration, 3)

    return v1_items, v2_items, clip_counter, broll_counter, cursor


def _ensure_timeline_initialized(blackboard: PipelineBlackboard) -> EditTimeline:
    """Create or reset the shared EditTimeline on the blackboard before Stage 30 assembly."""
    brief = blackboard.creative_brief
    paper = blackboard.paper_edit
    media_index = blackboard.media_index
    if paper is None or media_index is None:
        raise RuntimeError("Stage 30 Picture Editors require paper_edit and media_index on the blackboard.")

    subclip_map: dict[str, CulledSubClip] = {s.subclip_id: s for s in media_index.subclips}
    photo_mode = brief.photo_mode if brief else "gallery_frame"
    cast_manifest = blackboard.cast_manifest

    visual_track: list[VisualTrackItem] = []
    broll_track: list[VisualTrackItem] = []
    title_track: list[TitleOverlayItem] = []
    markers: list[TimelineMarker] = []

    cursor = 0.0
    clip_counter = 1
    broll_counter = 1

    # 1. Optional Cold-Open Teaser Hook (00:00-00:06) before the Main Opening Title Card
    hook_slot = paper.cold_open_hook
    hook_sid = hook_slot.subclip_id if hook_slot else paper.cold_open_subclip_id
    hook_sc = subclip_map.get(hook_sid) if hook_sid else None
    if hook_sc is not None and hook_sc.media_type == "video":
        hook_in = (
            hook_slot.in_tc
            if hook_slot is not None
            else (paper.cold_open_in_tc if paper.cold_open_in_tc is not None else hook_sc.in_tc)
        )
        hook_out_raw = (
            hook_slot.out_tc
            if hook_slot is not None
            else (paper.cold_open_out_tc if paper.cold_open_out_tc is not None else min(hook_sc.out_tc, hook_in + 3.2))
        )
        hook_dur = round(max(2.0, min(6.0, hook_out_raw - hook_in)), 2)
        hook_rationale = (hook_slot.rationale if hook_slot else paper.cold_open_rationale) or "First-6-seconds peak teaser hook"
        hook_caption = (hook_slot.caption if hook_slot else paper.cold_open_hook_caption) or hook_sc.summary
        markers.append(
            TimelineMarker(
                time_sec=0.0,
                name="Cold Open Teaser Hook",
                color="Blue",
                note=hook_rationale,
            )
        )
        hook_item = VisualTrackItem(
            id="clip_000_hook",
            type="video",
            source=hook_sc.source_path,
            track_lane=1,
            roll_type=hook_sc.roll_type,
            shot_scale=hook_sc.shot_scale,
            act="Cold Open Teaser",
            timeline_start=0.0,
            duration=hook_dur,
            source_in=round(hook_in, 3),
            source_out=round(hook_in + hook_dur, 3),
            speed=1.0,
            volume=1.0 if hook_sc.roll_type == "a_roll" else 0.25,
            j_l_cut_audio_offset=0.0,
            audio_crossfade_sec=0.25,
            ken_burns=None,
            transition_in=TransitionConfig(type="fade_black", duration=0.25),
            transition_out=TransitionConfig(type="fade_black", duration=0.25),
            narrative_note=f"[COLD OPEN HOOK] {hook_caption}",
        )
        visual_track.append(hook_item)
        blackboard.chapter_timeline_cache["__cold_open__"] = [hook_item.model_dump(mode="json")]
        cursor = round(hook_dur, 3)
    else:
        blackboard.chapter_timeline_cache.pop("__cold_open__", None)

    # 2. Seed Opening Title Card placeholder (MotionGraphicsTitleAgent at Stage 40 consolidates all V3 overlays)
    clean_proj_title = paper.project_title.replace("_", " ").strip() or "Featured Story"
    if len(clean_proj_title) > 42:
        clean_proj_title = " ".join(clean_proj_title.split()[:5]).title()
    else:
        clean_proj_title = clean_proj_title.title()

    title_track.append(
        TitleOverlayItem(
            id="title_opening",
            style="opening_card",
            text=clean_proj_title,
            subtitle="A Curated Documentary Film",
            timeline_start=round(cursor + 0.35, 2),
            duration=3.6,
            fade_in=0.45,
            fade_out=0.55,
            accent_color="#E8E4DC",
            hdr_safe=bool(blackboard.hdr),
        )
    )

    for c_idx, chapter in enumerate(paper.chapters):
        chapter_start = round(cursor, 3)
        markers.append(
            TimelineMarker(
                time_sec=chapter_start,
                name=chapter.title,
                color=MARKER_COLORS[c_idx % len(MARKER_COLORS)],
                note=chapter.narrative_goal,
            )
        )
        ch_v1, ch_v2, clip_counter, broll_counter, cursor = _build_chapter_items(
            chapter=chapter,
            subclip_map=subclip_map,
            photo_mode=photo_mode,
            cast_manifest=cast_manifest,
            clip_counter_start=clip_counter,
            broll_counter_start=broll_counter,
            cursor_start=cursor,
        )
        visual_track.extend(ch_v1)
        broll_track.extend(ch_v2)
        blackboard.chapter_timeline_cache[chapter.chapter_id] = [it.model_dump(mode="json") for it in ch_v1]
        blackboard.chapter_timeline_cache[f"__broll__:{chapter.chapter_id}"] = [
            it.model_dump(mode="json") for it in ch_v2
        ]

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


def _enforce_chapter_revision_constraints(
    ch_clips: list[VisualTrackItem],
    constraints: dict[str, Any],
    beat_grid: Optional[BeatGrid],
    prefer_downbeat: bool,
    media_index_subclips: list[CulledSubClip],
) -> list[VisualTrackItem]:
    """Apply chapter-scoped RevisionDirective constraints while keeping cuts snapped on-grid."""
    max_broll_c = float(constraints.get("max_broll_duration_sec", 4.5))
    max_stills_c = int(constraints.get("max_consecutive_stills", 3))

    # 1. Trim/snap any pure B-roll clip exceeding max_broll_duration_sec onto the beat/downbeat grid
    for item in ch_clips:
        if item.roll_type == "b_roll" and item.duration > max_broll_c + 1e-3:
            new_d = _snap_montage_duration(
                max_broll_c,
                min_dur=min(2.0, max_broll_c),
                max_dur=max_broll_c,
                beat_grid=beat_grid,
                prefer_downbeat=prefer_downbeat,
            )
            item.duration = new_d
            item.source_out = round(item.source_in + new_d, 3)

    # 2. Break >max_stills_c consecutive static stills by interleaving a video B-roll or trimming excess stills
    used_sources = {c.source for c in ch_clips}
    spare_videos = [
        sc
        for sc in media_index_subclips
        if sc.media_type == "video" and not sc.is_discarded and sc.source_path not in used_sources
    ]
    revised: list[VisualTrackItem] = []
    still_run = 0
    spare_idx = 0
    for item in ch_clips:
        if item.type == "photo":
            still_run += 1
            if still_run > max_stills_c:
                if spare_idx < len(spare_videos):
                    sv = spare_videos[spare_idx]
                    spare_idx += 1
                    sv_in = sv.golden_in if sv.golden_in is not None else sv.in_tc
                    sv_dur = _snap_montage_duration(
                        min(max_broll_c, sv.duration),
                        min_dur=2.0,
                        max_dur=max_broll_c,
                        beat_grid=beat_grid,
                        prefer_downbeat=prefer_downbeat,
                    )
                    revised.append(
                        VisualTrackItem(
                            id=f"{item.id}_vid_break",
                            type="video",
                            source=sv.source_path,
                            track_lane=1,
                            roll_type="b_roll",
                            shot_scale=sv.shot_scale,
                            act=item.act,
                            timeline_start=item.timeline_start,
                            duration=sv_dur,
                            source_in=round(sv_in, 3),
                            source_out=round(sv_in + sv_dur, 3),
                            speed=1.0,
                            volume=0.10,
                            j_l_cut_audio_offset=0.0,
                            audio_crossfade_sec=0.33,
                            transition_in=TransitionConfig(type="crossfade", duration=0.30),
                            transition_out=TransitionConfig(type="crossfade", duration=0.30),
                            narrative_note=f"[REVISION INTERLEAVE] {sv.summary}",
                        )
                    )
                    revised.append(item)
                    still_run = 1
                else:
                    # Drop excess consecutive still when no video cutaway exists so chapter honors <=max_stills_c
                    continue
            else:
                revised.append(item)
        else:
            still_run = 0
            revised.append(item)

    return revised if revised else ch_clips[:1]


@register_agent
class DialogueEditorAgent(BaseEditingAgent):
    """Agent 3a (Stage 30): Trims A-Roll on word boundaries, applies J/L cuts, and anchors Track V2 B-Roll."""

    role_id = "dialogue_editor"
    display_name = "3a. Dialogue & Narrative Editor"
    stage_order = 30
    substage_priority = 0
    description = (
        "Trims A-roll clips on word timestamps within usable_in/usable_out handles, sets -0.35s J-cut audio "
        "lead-ins, populates bilingual subtitles, and places Track V2 B-roll cutaways over spoken takes."
    )
    enabled_by_default = True
    requires_artifacts = ["paper_edit.json", "media_index.json"]
    produces_artifacts = ["dialogue_selects.json"]
    skill_reference_file = "03_picture_editors.md"

    def run(self, blackboard: PipelineBlackboard, realign_only: bool = False) -> dict[str, Any]:
        pending_dialogue_dirs = blackboard.pending_revision_directives(self.role_id)
        has_montage_revisions = bool(blackboard.pending_revision_directives("montage_editor"))

        if realign_only or ((pending_dialogue_dirs or has_montage_revisions) and blackboard.timeline is not None):
            timeline = blackboard.timeline or _ensure_timeline_initialized(blackboard)
        else:
            timeline = _ensure_timeline_initialized(blackboard)

        catalog = blackboard.catalog
        media_index = blackboard.media_index
        subclips_by_path: dict[str, list[CulledSubClip]] = {}
        if media_index is not None:
            for sc in media_index.subclips:
                subclips_by_path.setdefault(sc.source_path, []).append(sc)

        video_by_path: dict[str, Any] = {}
        if catalog is not None:
            for v in catalog.videos:
                video_by_path[str(Path(v.path).resolve())] = v

        sub_items: list[SubtitleItem] = []
        sub_counter = 1
        aroll_clips_processed = 0
        dialogue_selects_summary: list[dict[str, Any]] = []

        for item in timeline.visual_track:
            if item.type == "video" and item.roll_type == "a_roll":
                aroll_clips_processed += 1
                item.volume = 1.0
                # Respect elastic usable_in handle for -0.35s J-cut pre-lap
                sc_candidates = subclips_by_path.get(item.source, [])
                usable_lo = min((c.usable_in for c in sc_candidates), default=0.0)
                desired_jl = -0.35 if item.id != "clip_000_hook" else 0.0
                if item.source_in + desired_jl < usable_lo:
                    desired_jl = round(max(-0.35, usable_lo - item.source_in), 3)
                item.j_l_cut_audio_offset = desired_jl
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

                dialogue_selects_summary.append(
                    {
                        "clip_id": item.id,
                        "act": item.act,
                        "source_in": item.source_in,
                        "source_out": item.source_out,
                        "j_l_cut_audio_offset": item.j_l_cut_audio_offset,
                    }
                )

        timeline.subtitles.items = sub_items
        if pending_dialogue_dirs:
            blackboard.mark_directives_resolved(self.role_id)

        if not realign_only:
            blackboard.set_custom_artifact(
                "dialogue_selects.json",
                {
                    "aroll_clips_processed": aroll_clips_processed,
                    "subtitle_count": len(sub_items),
                    "v2_broll_cutaways": len(timeline.broll_track),
                    "selects": dialogue_selects_summary,
                },
            )
            blackboard.emit_activity(
                self.role_id,
                f"Cut {aroll_clips_processed} A-Roll dialogue clips with elastic J-cuts, {len(sub_items)} bilingual subtitles & {len(timeline.broll_track)} V2 cutaways",
                detail="Verified J-cut pre-laps against usable_in handles and wrote dialogue_selects.json",
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
    """Agent 3b (Stage 30): Beat-snaps B-roll & stills on-grid, applies face-safe Ken Burns, and handles chapter revisions."""

    role_id = "montage_editor"
    display_name = "3b. Rhythm & Montage Editor"
    stage_order = 30
    substage_priority = 1
    description = (
        "Snaps B-roll and photo durations to beat_grid.json beats/downbeats, biases Ken Burns framing toward "
        "CastContinuityAgent subject bboxes, supports hybrid candid-reaction chapters, and executes chapter-scoped "
        "beat-preserving revisions."
    )
    enabled_by_default = True
    requires_artifacts = ["paper_edit.json", "beat_grid.json", "media_index.json"]
    produces_artifacts = ["timeline.yaml"]
    skill_reference_file = "03_picture_editors.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline or _ensure_timeline_initialized(blackboard)
        beat_grid = blackboard.beat_grid
        brief = blackboard.creative_brief
        paper = blackboard.paper_edit
        media_index = blackboard.media_index
        max_broll = brief.max_broll_duration_sec if brief else 4.5
        is_extended_cut = (brief.target_duration_sec if brief else 110.0) >= 135.0

        subclips_by_path: dict[str, list[CulledSubClip]] = {}
        if media_index is not None:
            for sc in media_index.subclips:
                subclips_by_path.setdefault(sc.source_path, []).append(sc)

        pending_directives = blackboard.pending_revision_directives(self.role_id)

        # Record V2 cutaway parent mapping before adjusting V1 durations
        v2_parent_info: list[tuple[VisualTrackItem, Optional[str], float]] = []
        for b_item in timeline.broll_track:
            parent = next(
                (
                    v
                    for v in timeline.visual_track
                    if v.timeline_start <= b_item.timeline_start < (v.timeline_start + v.duration + 0.05)
                ),
                None,
            )
            rel_offset = (b_item.timeline_start - parent.timeline_start) if parent else 2.0
            v2_parent_info.append((b_item, parent.id if parent else None, rel_offset))
        v1_with_v2_cutaway = {pid for _, pid, _ in v2_parent_info if pid is not None}

        snapped_count = 0
        scale_cycle = ["wide", "medium", "closeup"]

        # Branch A: Chapter-scoped RevisionDirective re-cut (preserves untargeted music_montage chapters on downbeats!)
        if pending_directives and paper is not None and blackboard.chapter_timeline_cache:
            directives_by_chap: dict[str, dict[str, Any]] = {}
            global_constraints: dict[str, Any] = {}
            for d in pending_directives:
                if d.chapter_id:
                    merged = directives_by_chap.setdefault(d.chapter_id, {})
                    merged.update(d.constraints)
                else:
                    global_constraints.update(d.constraints)

            rebuilt_v1: list[VisualTrackItem] = []
            rebuilt_markers: list[TimelineMarker] = []
            cursor = 0.0

            hook_cached = blackboard.chapter_timeline_cache.get("__cold_open__")
            if hook_cached:
                hook_item = VisualTrackItem.model_validate(hook_cached[0])
                hook_item.timeline_start = 0.0
                rebuilt_v1.append(hook_item)
                rebuilt_markers.append(
                    TimelineMarker(
                        time_sec=0.0,
                        name="Cold Open Teaser Hook",
                        color="Blue",
                        note="First-6-seconds peak teaser hook",
                    )
                )
                cursor = round(hook_item.duration, 3)

            for c_idx, chap in enumerate(paper.chapters):
                # Before entering any music_montage chapter, align preceding chapter tail so montage starts on a downbeat
                if chap.editing_mode == "music_montage" and rebuilt_v1 and beat_grid is not None:
                    cursor = _align_tail_clips_to_downbeat(
                        rebuilt_v1,
                        current_cursor=cursor,
                        beat_grid=beat_grid,
                        subclip_by_source=subclips_by_path,
                        max_broll=max_broll,
                    )
                    # Sync sequential starts of rebuilt_v1 after tail alignment
                    run_t = 0.0
                    for prev_it in rebuilt_v1:
                        prev_it.timeline_start = round(run_t, 3)
                        run_t = round(run_t + prev_it.duration, 3)
                    cursor = run_t

                raw_list = blackboard.chapter_timeline_cache.get(chap.chapter_id)
                if raw_list is not None:
                    ch_clips = [VisualTrackItem.model_validate(x) for x in raw_list]
                else:
                    ch_clips = [c for c in timeline.visual_track if c.act == chap.title]

                ch_constraints = {**global_constraints, **directives_by_chap.get(chap.chapter_id, {})}
                is_targeted = (chap.chapter_id in directives_by_chap) or bool(global_constraints)
                if is_targeted and ch_clips:
                    ch_clips = _enforce_chapter_revision_constraints(
                        ch_clips=ch_clips,
                        constraints=ch_constraints,
                        beat_grid=beat_grid,
                        prefer_downbeat=(chap.editing_mode == "music_montage"),
                        media_index_subclips=(media_index.subclips if media_index else []),
                    )
                    snapped_count += len(ch_clips)

                rebuilt_markers.append(
                    TimelineMarker(
                        time_sec=round(cursor, 3),
                        name=chap.title,
                        color=MARKER_COLORS[c_idx % len(MARKER_COLORS)],
                        note=chap.narrative_goal,
                    )
                )
                for item in ch_clips:
                    item.timeline_start = round(cursor, 3)
                    cursor = round(cursor + item.duration, 3)
                    rebuilt_v1.append(item)

                blackboard.chapter_timeline_cache[chap.chapter_id] = [it.model_dump(mode="json") for it in ch_clips]

            timeline.visual_track = rebuilt_v1
            timeline.markers = rebuilt_markers
            blackboard.mark_directives_resolved(self.role_id, chapter_ids=list(directives_by_chap.keys()) or None)
        else:
            # Branch B: Initial full Stage-30 montage rhythm pass across chapters
            chap_mode_by_title: dict[str, str] = {}
            if paper is not None:
                for chap in paper.chapters:
                    chap_mode_by_title[chap.title] = chap.editing_mode

            for idx, item in enumerate(timeline.visual_track):
                if item.id == "clip_000_hook":
                    continue
                chap_mode = chap_mode_by_title.get(item.act or "", "music_montage")
                prefer_db = chap_mode == "music_montage"

                if item.roll_type in ("b_roll", "still"):
                    cap = max(max_broll, 7.0) if item.id in v1_with_v2_cutaway else max_broll
                    target_d = min(cap, item.duration)
                    min_hold = 2.0 if prefer_db else 2.4
                    if item.type == "photo" and item.ken_burns is not None and is_extended_cut:
                        kb_mode = item.ken_burns.mode
                        cap_len = len(item.ken_burns.caption_title or "") + len(item.ken_burns.caption_subtitle or "")
                        if kb_mode in ("diptych_frame", "triptych_cinematic"):
                            min_hold = min(cap, 4.8)
                            target_d = max(target_d, min_hold)
                        elif cap_len >= 42:
                            min_hold = min(cap, 4.2)
                            target_d = max(target_d, min_hold)

                    snapped_d = _snap_montage_duration(
                        target_d,
                        min_dur=min(min_hold, cap),
                        max_dur=cap,
                        beat_grid=beat_grid,
                        prefer_downbeat=prefer_db,
                    )
                    # Clamp video source_out against elastic usable_out handle
                    if item.type == "video":
                        sc_cands = subclips_by_path.get(item.source, [])
                        if sc_cands:
                            usable_hi = max(c.usable_out for c in sc_cands)
                            if item.source_in + snapped_d > usable_hi and usable_hi > item.source_in + 1.5:
                                snapped_d = _snap_montage_duration(
                                    usable_hi - item.source_in,
                                    min_dur=1.5,
                                    max_dur=usable_hi - item.source_in,
                                    beat_grid=beat_grid,
                                    prefer_downbeat=prefer_db,
                                )

                    item.duration = snapped_d
                    item.source_out = round(item.source_in + snapped_d, 3)
                    if item.roll_type == "b_roll":
                        item.volume = 0.10
                        item.audio_crossfade_sec = 0.33
                    else:
                        item.volume = 0.0
                    if item.shot_scale == "medium":
                        item.shot_scale = scale_cycle[idx % len(scale_cycle)]  # type: ignore[assignment]
                    snapped_count += 1
                elif prefer_db and item.type == "video":
                    # In pure music_montage chapters, snap any remaining video clips to downbeat multiples too
                    usable_hi = max(
                        (c.usable_out for c in subclips_by_path.get(item.source, [])),
                        default=item.source_in + item.duration,
                    )
                    avail_max = max(2.0, min(6.0, usable_hi - item.source_in))
                    snapped_d = _snap_montage_duration(
                        item.duration,
                        min_dur=2.0,
                        max_dur=avail_max,
                        beat_grid=beat_grid,
                        prefer_downbeat=True,
                    )
                    item.duration = snapped_d
                    item.source_out = round(item.source_in + snapped_d, 3)

            # Align chapter boundaries so every music_montage chapter starts on an exact beat_grid downbeat
            if paper is not None:
                assembled_v1: list[VisualTrackItem] = []
                updated_markers: list[TimelineMarker] = []
                cursor = 0.0
                hook_clips = [c for c in timeline.visual_track if c.id == "clip_000_hook"]
                if hook_clips:
                    hook_clips[0].timeline_start = 0.0
                    assembled_v1.append(hook_clips[0])
                    updated_markers.append(
                        TimelineMarker(
                            time_sec=0.0,
                            name="Cold Open Teaser Hook",
                            color="Blue",
                            note="First-6-seconds peak teaser hook",
                        )
                    )
                    cursor = round(hook_clips[0].duration, 3)

                for c_idx, chap in enumerate(paper.chapters):
                    if chap.editing_mode == "music_montage" and assembled_v1 and beat_grid is not None:
                        cursor = _align_tail_clips_to_downbeat(
                            assembled_v1,
                            current_cursor=cursor,
                            beat_grid=beat_grid,
                            subclip_by_source=subclips_by_path,
                            max_broll=max_broll,
                        )
                        run_t = 0.0
                        for prev_it in assembled_v1:
                            prev_it.timeline_start = round(run_t, 3)
                            run_t = round(run_t + prev_it.duration, 3)
                        cursor = run_t

                    ch_clips = [
                        c for c in timeline.visual_track if c.act == chap.title and c.id != "clip_000_hook"
                    ]
                    updated_markers.append(
                        TimelineMarker(
                            time_sec=round(cursor, 3),
                            name=chap.title,
                            color=MARKER_COLORS[c_idx % len(MARKER_COLORS)],
                            note=chap.narrative_goal,
                        )
                    )
                    for ch_it in ch_clips:
                        ch_it.timeline_start = round(cursor, 3)
                        cursor = round(cursor + ch_it.duration, 3)
                        assembled_v1.append(ch_it)
                    blackboard.chapter_timeline_cache[chap.chapter_id] = [
                        it.model_dump(mode="json") for it in ch_clips
                    ]

                if assembled_v1:
                    timeline.visual_track = assembled_v1
                    timeline.markers = updated_markers

        timeline.recalculate_sequential_starts()
        new_starts = {item.id: item.timeline_start for item in timeline.visual_track}

        # Re-align subtitles to updated V1 A-roll start times
        if blackboard.catalog is not None:
            DialogueEditorAgent().run(blackboard, realign_only=True)

        # Re-align Track V2 B-Roll cutaways to their parent A-roll clips and beat-snap their durations
        kept_broll: list[VisualTrackItem] = []
        for b_item, parent_id, rel_offset in v2_parent_info:
            if beat_grid is not None:
                b_item.duration = beat_grid.snap_duration_to_beat(b_item.duration, min_dur=2.0, max_dur=3.5)
                b_item.source_out = round(b_item.source_in + b_item.duration, 3)
            if parent_id is not None and parent_id in new_starts:
                b_item.timeline_start = round(new_starts[parent_id] + rel_offset, 3)
                kept_broll.append(b_item)
            elif parent_id is None:
                kept_broll.append(b_item)
        timeline.broll_track = kept_broll

        # Diversify adjacent Ken Burns presets and apply CastContinuityAgent face-safe centers if cast_manifest updated
        prev_preset = None
        for idx, item in enumerate(timeline.visual_track):
            if item.type == "photo" and item.ken_burns is not None:
                if item.ken_burns.preset == prev_preset:
                    item.ken_burns.preset = KB_PRESETS[(idx + 1) % len(KB_PRESETS)]  # type: ignore[assignment]
                prev_preset = item.ken_burns.preset

        # Seed continuous Track A3 Background Music bed (Stage 35 lead_editor & Stage 40 sound_mixer refine envelopes)
        audio_source = ""
        if beat_grid and beat_grid.source_audio and Path(beat_grid.source_audio).exists():
            audio_source = beat_grid.source_audio
        elif blackboard.bgm_selection and blackboard.bgm_selection.selected_track_path and Path(blackboard.bgm_selection.selected_track_path).exists():
            audio_source = blackboard.bgm_selection.selected_track_path
        elif blackboard.extra_audio_paths:
            audio_source = str(blackboard.extra_audio_paths[0])
        elif blackboard.catalog and blackboard.catalog.audio_files:
            audio_source = blackboard.catalog.audio_files[0].path

        if audio_source and Path(audio_source).exists():
            non_music = [a for a in timeline.audio_tracks if a.role != "music"]
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
                    duck_volume=0.056,
                    duck_ramp_sec=0.8,
                    fade_in=1.0,
                    fade_out=2.5,
                ),
                *non_music,
            ]

        # Optional AI Creative Intelligence refinement (only when not executing a deterministic revision constraint pass)
        if not pending_directives:
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

        tl_path = blackboard.project_dir / "timeline.yaml"
        timeline.to_yaml(tl_path)

        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="lead_editor",
            message_type="handoff",
            subject="Montage beat-snapping & face-safe Ken Burns framing complete",
            payload={
                "snapped_clips": snapped_count,
                "total_duration": round(timeline.total_duration, 2),
                "has_bgm": bool(timeline.audio_tracks),
            },
        )
        blackboard.emit_activity(
            self.role_id,
            f"Beat-snapped {snapped_count} clips on-grid -> Total timeline runtime: {timeline.total_duration:.1f}s",
            detail="Aligned music_montage chapters to beat_grid downbeats with face-safe Ken Burns trajectories",
        )

        return {
            "snapped_clips": snapped_count,
            "total_duration": round(timeline.total_duration, 2),
            "timeline_path": str(tl_path),
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
                        hdr_safe=bool(timeline.settings.hdr),
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
