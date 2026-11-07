"""Mechanical, Ripple-Aware Surgical Timeline Editor for AI Director & Creator Revisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..analyzer.media_probe import PHOTO_EXTENSIONS
from ..exporters.nle_export import export_all_nle_formats
from ..models.schema import (
    CulledSubClip,
    EditTimeline,
    KenBurnsConfig,
    MediaIndex,
    TransitionConfig,
    VisualTrackItem,
)
from .story_director import generate_storyboard_markdown


def _ripple_shift_downstream(timeline: EditTimeline, pivot_time: float, delta: float) -> None:
    """Mechanically shift all timeline items starting at or after pivot_time by delta seconds."""
    if abs(delta) < 1e-4:
        return

    timeline.recalculate_sequential_starts()

    for b in timeline.broll_track:
        if b.timeline_start >= pivot_time - 0.05:
            b.timeline_start = round(max(0.0, b.timeline_start + delta), 3)

    for t in timeline.title_track:
        if t.timeline_start >= pivot_time - 0.05:
            t.timeline_start = round(max(0.0, t.timeline_start + delta), 3)

    for fx in timeline.effects_track:
        if fx.timeline_start >= pivot_time - 0.05:
            fx.timeline_start = round(max(0.0, fx.timeline_start + delta), 3)

    for m in timeline.markers:
        if m.time_sec >= pivot_time - 0.05:
            m.time_sec = round(max(0.0, m.time_sec + delta), 3)

    for sub in timeline.subtitles.items:
        if sub.start >= pivot_time - 0.05:
            sub.start = round(max(0.0, sub.start + delta), 2)
            sub.end = round(max(sub.start + 0.2, sub.end + delta), 2)


def _resolve_media_source(
    project_dir: Path,
    raw_ref: str,
) -> tuple[Optional[Path], Optional[CulledSubClip]]:
    """Resolve a subclip_id, filename, or file path against media_index.json and disk."""
    mi_path = project_dir / "media_index.json"
    if mi_path.exists():
        try:
            mi = MediaIndex.from_json_file(mi_path)
            for sc in mi.subclips:
                if sc.subclip_id == raw_ref or sc.filename.lower() == raw_ref.lower():
                    p = Path(sc.source_path)
                    if p.exists():
                        return p.resolve(), sc
        except Exception:
            pass

    cand = Path(raw_ref)
    if cand.exists():
        return cand.resolve(), None
    return None, None


def apply_surgical_edit(
    project_dir: str | Path,
    operations: list[dict[str, Any]] | dict[str, Any],
) -> dict[str, Any]:
    """Execute one or more ripple-aware surgical edits on <project_dir>/timeline.yaml."""
    proj = Path(project_dir).resolve()
    tl_path = proj / "timeline.yaml"
    if not tl_path.exists():
        raise FileNotFoundError(f"timeline.yaml not found in {proj}")

    timeline = EditTimeline.from_yaml(tl_path)
    old_total_dur = round(timeline.total_duration, 2)
    ops_list = [operations] if isinstance(operations, dict) else list(operations)
    applied_logs: list[str] = []

    for op in ops_list:
        action = str(op.get("action") or "").strip().lower()
        clip_id = str(op.get("clip_id") or "").strip()

        # 1. UPDATE-PLAQUE: Update museum/editorial plaque or Ken Burns motion preset on a photo clip
        if action in ("update-plaque", "update_plaque", "plaque"):
            target = next(
                (c for c in (timeline.visual_track + timeline.broll_track) if c.id == clip_id),
                None,
            )
            if target is None:
                continue
            if target.ken_burns is None:
                target.ken_burns = KenBurnsConfig(mode="parallax_gallery", preset="floating_gallery")
            if op.get("caption_title") is not None:
                target.ken_burns.caption_title = str(op["caption_title"])
            if op.get("caption_subtitle") is not None:
                target.ken_burns.caption_subtitle = str(op["caption_subtitle"])
            if op.get("camera_badge") is not None:
                target.ken_burns.camera_badge = str(op["camera_badge"])
            if op.get("mode") in ("parallax_gallery", "diptych_frame", "gallery_frame", "full_bleed"):
                target.ken_burns.mode = op["mode"]
            if op.get("preset"):
                target.ken_burns.preset = op["preset"]
            if op.get("secondary_source") is not None:
                target.ken_burns.secondary_source = str(op["secondary_source"]) if op["secondary_source"] else None
            target.narrative_note = (
                f"[STILL · {target.ken_burns.caption_title}] {target.ken_burns.caption_subtitle or ''}".strip()
            )
            applied_logs.append(f"Updated plaque on {clip_id} -> '{target.ken_burns.caption_title}'")

        # 2. TRIM: Change duration / source_in of a clip and ripple-shift all downstream tracks
        elif action == "trim":
            v1_target = next((c for c in timeline.visual_track if c.id == clip_id), None)
            v2_target = next((c for c in timeline.broll_track if c.id == clip_id), None)
            target = v1_target or v2_target
            if target is None:
                continue

            old_dur = target.duration
            old_end = target.timeline_start + old_dur
            if op.get("new_source_in") is not None:
                target.source_in = max(0.0, round(float(op["new_source_in"]), 2))
            if op.get("new_duration") is not None:
                new_dur = max(0.8, round(float(op["new_duration"]), 2))
            elif op.get("delta_sec") is not None:
                new_dur = max(0.8, round(old_dur + float(op["delta_sec"]), 2))
            else:
                new_dur = old_dur

            target.duration = new_dur
            target.source_out = round(target.source_in + new_dur, 2)
            for fx in timeline.effects_track:
                if fx.target_clip_id == target.id:
                    fx.duration = new_dur

            if v1_target is not None:
                delta = round(new_dur - old_dur, 3)
                _ripple_shift_downstream(timeline, old_end, delta)
            applied_logs.append(f"Trimmed {clip_id} from {old_dur:.2f}s to {new_dur:.2f}s (rippled downstream)")

        # 3. DROP: Remove a clip and ripple-close the timeline gap
        elif action in ("drop", "delete", "remove"):
            v1_idx = next((i for i, c in enumerate(timeline.visual_track) if c.id == clip_id), None)
            if v1_idx is not None:
                dropped = timeline.visual_track.pop(v1_idx)
                old_end = dropped.timeline_start + dropped.duration
                timeline.effects_track = [fx for fx in timeline.effects_track if fx.target_clip_id != clip_id]
                _ripple_shift_downstream(timeline, old_end, -dropped.duration)
                applied_logs.append(f"Dropped V1 clip {clip_id} ({dropped.duration:.2f}s) and closed gap")
            else:
                before_len = len(timeline.broll_track)
                timeline.broll_track = [b for b in timeline.broll_track if b.id != clip_id]
                if len(timeline.broll_track) < before_len:
                    applied_logs.append(f"Dropped V2 cutaway {clip_id}")

        # 4. SWAP: Replace a clip's source asset (by path, filename, or subclip_id)
        elif action == "swap":
            target = next(
                (c for c in (timeline.visual_track + timeline.broll_track) if c.id == clip_id),
                None,
            )
            new_ref = str(op.get("new_source") or op.get("subclip_id") or "").strip()
            if target is None or not new_ref:
                continue
            resolved_path, sc = _resolve_media_source(proj, new_ref)
            if resolved_path is None:
                continue

            is_photo = resolved_path.suffix.lower() in PHOTO_EXTENSIONS
            target.source = str(resolved_path)
            target.type = "photo" if is_photo else "video"
            if is_photo:
                target.roll_type = "still"
                if target.ken_burns is None:
                    target.ken_burns = KenBurnsConfig(mode="parallax_gallery", preset="zoom_in")
                if sc is not None:
                    target.ken_burns.caption_title = sc.annotation_title or target.ken_burns.caption_title
                    target.ken_burns.caption_subtitle = sc.annotation_subtitle or target.ken_burns.caption_subtitle
                    target.ken_burns.camera_badge = sc.camera_badge or target.ken_burns.camera_badge
                if op.get("caption_title") is not None:
                    target.ken_burns.caption_title = str(op["caption_title"])
                if op.get("caption_subtitle") is not None:
                    target.ken_burns.caption_subtitle = str(op["caption_subtitle"])
                target.narrative_note = f"[STILL · {target.ken_burns.caption_title or resolved_path.name}]"
            else:
                target.ken_burns = None
                if sc is not None:
                    target.source_in = round(sc.in_tc, 2)
                    target.source_out = round(sc.in_tc + target.duration, 2)
                    target.roll_type = sc.roll_type
                target.narrative_note = f"[{target.roll_type.upper()}] {resolved_path.name}"
            applied_logs.append(f"Swapped {clip_id} source -> {resolved_path.name}")

        # 5. MOVE: Reorder a V1 clip before or after another V1 clip
        elif action == "move":
            src_idx = next((i for i, c in enumerate(timeline.visual_track) if c.id == clip_id), None)
            if src_idx is None:
                continue
            moving_clip = timeline.visual_track.pop(src_idx)
            before_id = op.get("before_clip_id")
            after_id = op.get("after_clip_id")
            if before_id:
                dst_idx = next((i for i, c in enumerate(timeline.visual_track) if c.id == before_id), 0)
                timeline.visual_track.insert(dst_idx, moving_clip)
            elif after_id:
                dst_idx = next(
                    (i + 1 for i, c in enumerate(timeline.visual_track) if c.id == after_id),
                    len(timeline.visual_track),
                )
                timeline.visual_track.insert(dst_idx, moving_clip)
            else:
                timeline.visual_track.insert(0, moving_clip)
            timeline.recalculate_sequential_starts()
            applied_logs.append(f"Moved {clip_id} and recalculated Track V1 start times")

        # 6. SET-COLD-OPEN: Insert or update clip_000_hook at 0.0s
        elif action in ("set-cold-open", "set_cold_open", "cold-open"):
            ref = str(op.get("subclip_id") or op.get("new_source") or "").strip()
            resolved_path, sc = _resolve_media_source(proj, ref)
            if resolved_path is None:
                continue
            hook_dur = round(float(op.get("duration") or 3.2), 2)
            hook_in = round(float(op.get("source_in") if op.get("source_in") is not None else (sc.in_tc if sc else 0.0)), 2)
            hook_caption = str(op.get("hook_caption") or (sc.spoken_text if sc else "") or "Cold Open Hook")

            existing_hook = next((c for c in timeline.visual_track if c.id == "clip_000_hook"), None)
            if existing_hook is not None:
                old_dur = existing_hook.duration
                existing_hook.source = str(resolved_path)
                existing_hook.source_in = hook_in
                existing_hook.duration = hook_dur
                existing_hook.source_out = round(hook_in + hook_dur, 2)
                existing_hook.narrative_note = f"[COLD OPEN HOOK] {hook_caption}"
                _ripple_shift_downstream(timeline, old_dur, round(hook_dur - old_dur, 3))
            else:
                new_hook = VisualTrackItem(
                    id="clip_000_hook",
                    type="video",
                    source=str(resolved_path),
                    track_lane=1,
                    roll_type=sc.roll_type if sc else "a_roll",
                    shot_scale=sc.shot_scale if sc else "medium",
                    act="Cold Open Teaser",
                    timeline_start=0.0,
                    duration=hook_dur,
                    source_in=hook_in,
                    source_out=round(hook_in + hook_dur, 2),
                    speed=1.0,
                    volume=1.0,
                    j_l_cut_audio_offset=0.0,
                    audio_crossfade_sec=0.25,
                    ken_burns=None,
                    transition_in=TransitionConfig(type="fade_black", duration=0.25),
                    transition_out=TransitionConfig(type="fade_black", duration=0.25),
                    narrative_note=f"[COLD OPEN HOOK] {hook_caption}",
                )
                timeline.visual_track.insert(0, new_hook)
                _ripple_shift_downstream(timeline, 0.0, hook_dur)
            applied_logs.append(f"Set Cold-Open Hook clip_000_hook ({resolved_path.name}, {hook_dur}s)")

        # 7. AUDIO-MIX: Adjust BGM volume, ducking level, or B-Roll Nat-Sound volume
        elif action in ("audio-mix", "audio_mix", "audio"):
            for a in timeline.audio_tracks:
                if op.get("bgm_volume") is not None:
                    a.volume = float(op["bgm_volume"])
                if op.get("duck_volume") is not None:
                    a.duck_volume = float(op["duck_volume"])
            if op.get("nat_sound_volume") is not None:
                nat_v = float(op["nat_sound_volume"])
                for v in timeline.visual_track:
                    if v.type == "video" and v.roll_type == "b_roll":
                        v.volume = nat_v
            applied_logs.append("Updated timeline audio mix parameters")

    # Sync BGM duration with updated total timeline duration
    total_dur = round(timeline.total_duration, 2)
    for a in timeline.audio_tracks:
        a.duration = total_dur

    timeline.to_yaml(tl_path)
    sb_md = generate_storyboard_markdown(timeline)
    (proj / "storyboard.md").write_text(sb_md, encoding="utf-8")
    export_all_nle_formats(timeline, proj)

    return {
        "timeline_path": str(tl_path),
        "operations_count": len(applied_logs),
        "old_duration_sec": old_total_dur,
        "new_duration_sec": total_dur,
        "total_duration_sec": total_dur,
        "v1_cuts": len(timeline.visual_track),
        "v2_cutaways": len(timeline.broll_track),
        "applied_operations": applied_logs,
    }
