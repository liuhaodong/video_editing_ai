"""Mechanical, Ripple-Aware Surgical Timeline Editor for AI Director & Creator Revisions."""

from __future__ import annotations

from pathlib import Path
import os
import re
from typing import Any, Optional

from ..agents.plugins.color_grading_agent import apply_color_grade_overrides
from ..analyzer.media_probe import PHOTO_EXTENSIONS
from ..exporters.nle_export import export_all_nle_formats
from ..models.schema import (
    VALID_PHOTO_MODES,
    VALID_PHOTO_OVERLAY_FX,
    VALID_TRAVEL_MAP_STYLES,
    CulledSubClip,
    EditTimeline,
    KenBurnsConfig,
    MapWaypointSpec,
    MediaIndex,
    TitleOverlayItem,
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

    for a in timeline.audio_tracks:
        if a.role != "music" and a.timeline_start >= pivot_time - 0.05:
            a.timeline_start = round(max(0.0, a.timeline_start + delta), 3)


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

        # 1. UPDATE-PLAQUE / SET-PHOTO-EFFECT: Update museum/editorial plaque, photo presentation mode, or atmospheric finisher
        if action in ("update-plaque", "update_plaque", "plaque", "set-photo-effect", "photo-effect"):
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
            req_mode = op.get("mode") or op.get("photo_mode")
            if req_mode in VALID_PHOTO_MODES:
                target.ken_burns.mode = req_mode
            req_fx = op.get("overlay_fx")
            if req_fx in VALID_PHOTO_OVERLAY_FX:
                target.ken_burns.overlay_fx = req_fx
            if op.get("preset"):
                target.ken_burns.preset = op["preset"]
            if op.get("secondary_source") is not None:
                target.ken_burns.secondary_source = str(op["secondary_source"]) if op["secondary_source"] else None
            if op.get("tertiary_source") is not None:
                target.ken_burns.tertiary_source = str(op["tertiary_source"]) if op["tertiary_source"] else None
            target.narrative_note = (
                f"[STILL · {target.ken_burns.caption_title}] {target.ken_burns.caption_subtitle or ''}".strip()
            )
            applied_logs.append(
                f"Updated {clip_id} -> mode={target.ken_burns.mode}, overlay_fx={target.ken_burns.overlay_fx}, title='{target.ken_burns.caption_title}'"
            )

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
                target.source_in = max(0.0, round(float(op["new_source_in"]), 3))
            if op.get("new_duration") is not None:
                new_dur = max(0.8, round(float(op["new_duration"]), 3))
            elif op.get("delta_sec") is not None:
                new_dur = max(0.8, round(old_dur + float(op["delta_sec"]), 3))
            else:
                new_dur = old_dur

            target.duration = new_dur
            target.source_out = round(target.source_in + new_dur, 3)
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
                timeline.effects_track = [
                    fx
                    for fx in timeline.effects_track
                    if not (
                        fx.target_clip_id == clip_id
                        or (
                            fx.target_clip_id is None
                            and abs(fx.timeline_start - dropped.timeline_start) < 0.05
                            and abs(fx.duration - dropped.duration) < 0.05
                        )
                    )
                ]
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
                if a.role == "music":
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

        elif action in ("insert-travel-map", "insert_travel_map", "travel-map"):
            from ..renderer.travel_map import (
                haversine_km,
                infer_transport_mode,
                lookup_gazetteer_place,
                render_travel_map_clip,
                resolve_travel_map_style,
            )

            def _parse_geo_stop(raw_place: Any, default_lat: float, default_lon: float, default_lbl: str) -> MapWaypointSpec:
                txt = str(raw_place or "").strip()
                m_coord = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*(?:,\s*(.*))?$", txt)
                if m_coord:
                    return MapWaypointSpec(
                        waypoint_id="wp",
                        label=(m_coord.group(3) or default_lbl)[:28],
                        lat=float(m_coord.group(1)),
                        lon=float(m_coord.group(2)),
                        alt_m=40.0,
                        source="exif_gps",
                    )
                if txt:
                    hit = lookup_gazetteer_place(txt)
                    if hit is not None:
                        return MapWaypointSpec(
                            waypoint_id="wp",
                            label=hit[3][:28],
                            lat=hit[0],
                            lon=hit[1],
                            alt_m=hit[2],
                            source="gazetteer",
                        )
                return MapWaypointSpec(
                    waypoint_id="wp",
                    label=(txt or default_lbl)[:28],
                    lat=default_lat,
                    lon=default_lon,
                    alt_m=40.0,
                    source="synthetic",
                )

            req_wps_raw = op.get("waypoints")
            wps_list: list[MapWaypointSpec] = []
            if isinstance(req_wps_raw, list) and len(req_wps_raw) >= 2:
                for idx_w, rw in enumerate(req_wps_raw):
                    if isinstance(rw, MapWaypointSpec):
                        wps_list.append(rw)
                    elif isinstance(rw, dict):
                        wps_list.append(
                            MapWaypointSpec(
                                waypoint_id=str(rw.get("waypoint_id") or f"wp_{idx_w + 1:02d}"),
                                chapter_id=str(rw.get("chapter_id") or ""),
                                label=str(rw.get("label") or rw.get("name") or f"Stop {idx_w + 1}")[:28],
                                subtitle=str(rw.get("subtitle") or ""),
                                lat=float(rw.get("lat", 35.6762 + idx_w * 0.25)),
                                lon=float(rw.get("lon", 139.6503 - idx_w * 0.35)),
                                alt_m=float(rw.get("alt_m", 40.0)),
                                source=rw.get("source", "exif_gps") if rw.get("source") in ("exif_gps", "gazetteer", "synthetic") else "exif_gps",
                                hero_thumbnail_path=rw.get("hero_thumbnail_path"),
                            )
                        )
            if len(wps_list) < 2:
                wp_orig = _parse_geo_stop(op.get("origin"), 35.6762, 139.6503, "Departure")
                wp_orig.waypoint_id = "wp_01"
                wp_dest = _parse_geo_stop(op.get("destination"), 35.0116, 135.7681, "Destination")
                wp_dest.waypoint_id = "wp_02"
                wps_list = [wp_orig, wp_dest]

            raw_leg_i = op.get("leg_index")
            use_leg_idx: Optional[int] = int(raw_leg_i) if raw_leg_i is not None else (0 if len(wps_list) == 2 else 0)
            focus_i = max(0, min(len(wps_list) - 2, int(use_leg_idx or 0)))
            wp_orig = wps_list[focus_i]
            wp_dest = wps_list[focus_i + 1]

            allow_tiles = os.environ.get("AI_EDITOR_OFFLINE_MAPS", "0") != "1"
            req_style = str(op.get("style") or op.get("map_style") or "auto").strip().lower()
            dist_km = haversine_km(wp_orig.lat, wp_orig.lon, wp_dest.lat, wp_dest.lon)
            t_mode = infer_transport_mode(dist_km, wp_orig.label, wp_dest.label, hint=str(op.get("transport_mode") or "auto"))
            op_leg_modes = op.get("leg_modes") if isinstance(op.get("leg_modes"), list) else None
            if req_style in VALID_TRAVEL_MAP_STYLES and req_style != "auto":
                tm_style = req_style
            else:
                ctx_blob = " ".join(
                    part for part in (
                        str(timeline.project_title or ""),
                        wp_orig.label,
                        wp_dest.label,
                        wp_orig.subtitle,
                        wp_dest.subtitle,
                    )
                    if part
                )
                tm_style = resolve_travel_map_style(
                    "auto",
                    timeline.subtitles.theme or "apple_tv_glass",
                    ctx_blob,
                    transport_mode=t_mode,
                    leg_km=dist_km,
                )
            placement = str(op.get("placement") or "v2_flyover").strip().lower()
            req_dur = max(1.4, min(12.0, float(op.get("duration") or op.get("new_duration") or (4.8 if placement in ("v1_interlude", "v1", "interlude") else 3.8))))

            host_v1 = None
            if clip_id:
                host_v1 = next((v for v in timeline.visual_track if v.id == clip_id), None)
            elif op.get("timeline_start") is not None:
                tgt_t = float(op["timeline_start"])
                host_v1 = min(timeline.visual_track, key=lambda v: abs(float(v.timeline_start) - tgt_t), default=None)
            elif timeline.visual_track:
                cands = [v for v in timeline.visual_track if v.id != "clip_000_hook" and float(v.duration) >= 1.6]
                host_v1 = cands[0] if cands else timeline.visual_track[0]

            if host_v1 is None:
                raise ValueError("Cannot insert travel map onto an empty timeline")

            if placement == "v3_pip":
                st = round(max(float(host_v1.timeline_start) + 0.05, float(op.get("timeline_start") or (host_v1.timeline_start + 0.15))), 2)
                dur = round(min(req_dur, max(1.2, float(host_v1.timeline_start) + float(host_v1.duration) - st - 0.05)), 2)
                km_lbl = f"{dist_km:.0f} km" if dist_km >= 2.0 else f"{dist_km * 1000.0:.0f} m"
                pip_id = f"title_map_pip_{len(timeline.title_track) + 1:02d}"
                timeline.title_track.append(
                    TitleOverlayItem(
                        id=pip_id,
                        style="travel_map_pip_card",
                        theme=timeline.subtitles.theme or "apple_tv_glass",  # type: ignore[arg-type]
                        motion_preset="masked_rise",
                        eyebrow="ROUTE TELEMETRY",
                        overline="ROUTE TELEMETRY",
                        text=f"{wp_orig.label} -> {wp_dest.label}"[:28],
                        subtitle=f"{km_lbl} | {t_mode.upper()}",
                        timeline_start=st,
                        duration=dur,
                        fade_in=0.35,
                        fade_out=0.35,
                        backdrop_blur=True,
                        map_waypoints=[
                            {**wp_orig.model_dump(mode="json"), "active": False, "completed": True},
                            {**wp_dest.model_dump(mode="json"), "active": True, "completed": True},
                        ],
                        hdr_safe=bool(timeline.settings.hdr),
                    )
                )
                timeline.title_track.sort(key=lambda t: t.timeline_start)
                applied_logs.append(f"Inserted travel_map_pip_card '{pip_id}' ({wp_orig.label} -> {wp_dest.label}) at {st:.2f}s")
            elif placement in ("v1_interlude", "v1", "interlude"):
                host_idx = timeline.visual_track.index(host_v1)
                insert_idx = host_idx + (1 if str(op.get("position") or "before").lower() == "after" else 0)
                pivot_t = float(host_v1.timeline_start) if insert_idx == host_idx else float(host_v1.timeline_start + host_v1.duration)
                dur = round(req_dur, 2)
                map_mp4 = proj / "cache" / "travel_maps" / f"surgical_leg_{len(timeline.visual_track) + 1:03d}.mp4"
                render_travel_map_clip(
                    waypoints=wps_list,
                    output_mp4=map_mp4,
                    style=tm_style,
                    duration=dur,
                    width=max(640, min(int(timeline.settings.width), 1920)),
                    height=max(360, min(int(timeline.settings.height), 1080)),
                    fps=int(timeline.settings.fps),
                    leg_index=use_leg_idx,
                    transport_mode=t_mode,
                    hdr_safe=bool(timeline.settings.hdr),
                    allow_tile_fetch=allow_tiles,
                    with_audio=True,
                    cache_dir=proj / "cache" / "maps" / "tiles",
                    leg_modes=op_leg_modes,
                    dynamic_zoom=bool(allow_tiles and t_mode == "flight" and dist_km >= 250.0),
                )
                existing_ids = {v.id for v in timeline.visual_track}
                new_id = f"clip_map_{len(timeline.visual_track) + 1:03d}"
                while new_id in existing_ids:
                    new_id = f"{new_id}_m"
                map_item = VisualTrackItem(
                    id=new_id,
                    type="video",
                    track_lane=1,
                    roll_type="b_roll",
                    shot_scale="wide",
                    act=host_v1.act,
                    timeline_start=round(pivot_t, 3),
                    duration=dur,
                    source=str(map_mp4),
                    source_in=0.0,
                    source_out=dur,
                    speed=1.0,
                    volume=0.70,
                    audio_crossfade_sec=0.30,
                    ken_burns=KenBurnsConfig(mode="full_bleed", overlay_fx="none", preset="zoom_in", caption_title="", caption_subtitle=""),
                    transition_in=TransitionConfig(type="dissolve", duration=0.35),
                    transition_out=TransitionConfig(type="dissolve", duration=0.35),
                    narrative_note=f"[2.5D TRAVEL MAP LEG] {wp_orig.label} -> {wp_dest.label} ({dist_km:.1f} km, {t_mode})",
                )
                timeline.visual_track.insert(insert_idx, map_item)
                _ripple_shift_downstream(timeline, pivot_time=pivot_t, delta=dur)
                applied_logs.append(f"Inserted standalone V1 travel map leg '{new_id}' ({wp_orig.label} -> {wp_dest.label}) at {pivot_t:.2f}s ({dur:.2f}s)")
            else:
                dur = round(min(req_dur, max(1.2, float(host_v1.duration) - 0.10)), 2)
                raw_st = float(op.get("timeline_start") if op.get("timeline_start") is not None else (host_v1.timeline_start + 0.05))
                st = round(max(float(host_v1.timeline_start) + 0.04, min(raw_st, float(host_v1.timeline_start) + float(host_v1.duration) - dur - 0.04)), 2)
                map_mp4 = proj / "cache" / "maps" / f"surgical_flyover_{len(timeline.broll_track) + 1:02d}.mp4"
                render_travel_map_clip(
                    waypoints=wps_list,
                    output_mp4=map_mp4,
                    style=tm_style,
                    duration=dur,
                    width=max(640, min(int(timeline.settings.width), 1920)),
                    height=max(360, min(int(timeline.settings.height), 1080)),
                    fps=int(timeline.settings.fps),
                    leg_index=use_leg_idx,
                    transport_mode=t_mode,
                    hdr_safe=bool(timeline.settings.hdr),
                    allow_tile_fetch=allow_tiles,
                    with_audio=False,
                    cache_dir=proj / "cache" / "maps" / "tiles",
                    leg_modes=op_leg_modes,
                    dynamic_zoom=bool(allow_tiles and t_mode == "flight" and dist_km >= 250.0),
                )
                broll_id = f"broll_map_{len(timeline.broll_track) + 1:02d}"
                timeline.broll_track.append(
                    VisualTrackItem(
                        id=broll_id,
                        type="video",
                        track_lane=2,
                        roll_type="b_roll",
                        shot_scale="wide",
                        act=host_v1.act,
                        timeline_start=st,
                        duration=dur,
                        source=str(map_mp4),
                        source_in=0.0,
                        source_out=dur,
                        volume=0.0,
                        audio_crossfade_sec=0.33,
                        narrative_note=f"Surgical 2.5D Travel Map Flyover ({wp_orig.label} -> {wp_dest.label}, {dist_km:.1f} km)",
                    )
                )
                timeline.broll_track.sort(key=lambda c: float(c.timeline_start))
                applied_logs.append(f"Inserted 2.5D V2 travel map '{broll_id}' ({wp_orig.label} -> {wp_dest.label}) at {st:.2f}s ({dur:.2f}s)")

        elif action == "set-color-grade":
            c_rep = apply_color_grade_overrides(
                timeline=timeline,
                project_dir=proj,
                look_name=op.get("look") or op.get("chapter_look"),
                clip_id=op.get("clip_id"),
                intensity=float(op["intensity"]) if op.get("intensity") is not None else None,
                exposure_ev=float(op["exposure_ev"]) if op.get("exposure_ev") is not None else None,
                wb_red=float(op["wb_red"]) if op.get("wb_red") is not None else None,
                wb_blue=float(op["wb_blue"]) if op.get("wb_blue") is not None else None,
                saturation=float(op["saturation"]) if op.get("saturation") is not None else None,
                contrast=float(op["contrast"]) if op.get("contrast") is not None else None,
                skin_protection=float(op["skin_protection"]) if op.get("skin_protection") is not None else None,
                strategy=op.get("strategy") or op.get("grading_strategy"),
            )
            tgt_desc = op.get("clip_id") or "all V1/V2 clips"
            applied_logs.append(
                f"Updated DI color grade on {tgt_desc}: look='{c_rep.primary_film_stock_look}', "
                f"strategy='{c_rep.grading_strategy}', shots={c_rep.shots_matched_count}"
            )

    # Sync BGM duration and closing cards with updated total timeline duration
    total_dur = round(timeline.total_duration, 3)
    for t_ov in timeline.title_track:
        if t_ov.id == "title_closing" or t_ov.style == "closing_card":
            t_ov.timeline_start = round(max(0.0, total_dur - t_ov.duration - 0.2), 2)
    for a in timeline.audio_tracks:
        if a.role == "music":
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
