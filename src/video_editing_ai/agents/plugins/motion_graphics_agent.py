"""Stage 40 Specialist: Motion Graphics, Lower-Thirds, Bilingual Subtitles & Closing Card Designer (title_designer)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageOps

from ...models.schema import (
    EditTimeline,
    GraphicsManifest,
    GraphicsOverlaySpec,
    MapWaypointSpec,
    MotionGraphicItem,
    MotionGraphicsReport,
    NarrationScriptReport,
    TitleOverlayItem,
    TravelMapManifest,
    TravelMapSegmentSpec,
    VisualTrackItem,
    WordTimestamp,
)
from ...renderer.subtitles import resolve_subtitle_font_stack
from ...renderer.titles import (
    HDR_DIFFUSE_WHITE_PEAK_RGB,
    compute_overlay_bbox_norm,
    get_theme_spec,
    sample_shot_accent_hex,
)
from ...renderer.travel_map import (
    CART_THEME_SPECS,
    haversine_km,
    infer_transport_mode,
    initial_bearing_deg,
    render_travel_map_clip,
    resolve_journey_waypoints,
    resolve_travel_map_style,
)
from ..base import BaseEditingAgent, PipelineBlackboard, register_agent


_ROMAN_NUMERALS = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII")


def _bbox_intersection_area(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _resolve_sample_image_for_time(t_sec: float, timeline: Any, catalog: Any) -> Optional[str]:
    """Find a representative still/keyframe image file near timeline time t_sec for accent sampling."""
    video_kf_by_name: dict[str, str] = {}
    first_photo: Optional[str] = None
    if catalog is not None:
        for p in getattr(catalog, "photos", []):
            cand = p.preview_path or p.path
            if cand and Path(cand).exists():
                first_photo = cand
                break
        for v in getattr(catalog, "videos", []):
            for sc in getattr(v, "scenes", []):
                if getattr(sc, "keyframe_path", None) and Path(sc.keyframe_path).exists():
                    video_kf_by_name[Path(v.path).name] = sc.keyframe_path
                    break

    for vc in getattr(timeline, "visual_track", []):
        if vc.timeline_start <= t_sec + 1.5 and (vc.timeline_start + vc.duration) >= max(0.0, t_sec - 0.5):
            if vc.type == "photo" and Path(vc.source).exists():
                return vc.source
            kf = video_kf_by_name.get(Path(vc.source).name)
            if kf:
                return kf
    return first_photo


def _sync_word_timestamps_onto_subtitles(timeline: Any, media_index: Any) -> None:
    """Propagate word-level ASR timestamps from CulledSubClips onto master SubtitleItems."""
    if media_index is None or not getattr(timeline.subtitles, "items", None):
        return
    subclips = getattr(media_index, "subclips", []) or []
    if not subclips:
        return

    by_filename: dict[str, list[Any]] = {}
    for sc in subclips:
        if getattr(sc, "words", None):
            by_filename.setdefault(Path(sc.source_path).name, []).append(sc)

    if not by_filename:
        return

    for sub in timeline.subtitles.items:
        if sub.words:
            continue
        for clip in timeline.visual_track:
            if clip.type != "video":
                continue
            clip_end = clip.timeline_start + clip.duration
            if clip.timeline_start - 0.5 > sub.end or clip_end + 0.5 < sub.start:
                continue
            spd = max(0.1, float(clip.speed or 1.0))
            matched_words: list[WordTimestamp] = []
            for sc in by_filename.get(Path(clip.source).name, []):
                for w in sc.words:
                    tl_w_start = clip.timeline_start + (float(w.start) - float(clip.source_in)) / spd
                    tl_w_end = clip.timeline_start + (float(w.end) - float(clip.source_in)) / spd
                    if tl_w_end < sub.start - 0.15 or tl_w_start > sub.end + 0.15:
                        continue
                    w_start = round(max(sub.start, min(sub.end, tl_w_start)), 3)
                    w_end = round(max(w_start + 0.04, min(sub.end + 0.05, tl_w_end)), 3)
                    matched_words.append(
                        WordTimestamp(
                            word=w.word,
                            start=w_start,
                            end=w_end,
                            confidence=getattr(w, "confidence", 0.9),
                        )
                    )
            if matched_words:
                sub.words = matched_words
                break


def _format_exif_date(raw_ct: Any) -> str:
    """Format a datetime/ISO/EXIF creation_time value as YYYY.MM.DD."""
    if raw_ct is None:
        return ""
    if hasattr(raw_ct, "strftime"):
        return raw_ct.strftime("%Y.%m.%d")
    match = re.search(r"(\d{4})[-:/](\d{2})[-:/](\d{2})", str(raw_ct).strip())
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return ""


_MINOR_TITLE_WORDS = {"a", "an", "the", "in", "of", "at", "on", "and", "to", "for"}


def _smart_title_case(raw: str) -> str:
    """Title-case words while preserving uppercase acronyms/numerals and lowercasing minor connectors."""
    tokens = raw.split()
    if not tokens:
        return ""
    out: list[str] = []
    cap_next = True
    for i, tok in enumerate(tokens):
        if tok in {"—", ":", "-", "·", "/"}:
            out.append(tok)
            cap_next = tok in {"—", ":", "-"}
            continue
        stripped = tok.strip(".,:;!?()[]\"'")
        is_minor = stripped.lower() in _MINOR_TITLE_WORDS
        if len(stripped) >= 2 and stripped.isupper() and any(ch.isalpha() for ch in stripped):
            out.append(tok)
        elif is_minor and i > 0 and not cap_next:
            out.append(tok.lower())
        else:
            out.append(tok[0].upper() + tok[1:].lower() if len(tok) > 1 else tok.upper())
        cap_next = tok.endswith((":", "—", "-"))
    return " ".join(out)


_GENERIC_LOCATION_TAGS = {
    "keepsake",
    "architectural",
    "living",
    "general",
    "scene",
    "location",
}


def _strip_chapter_prefix(raw_title: str) -> str:
    cleaned = re.sub(
        r"^(?:Act\s+[IVX0-9]+|Chapter\s+\d+)\s*[-–—:]\s*",
        "",
        (raw_title or "").strip(),
        flags=re.IGNORECASE,
    ).strip()
    return cleaned or (raw_title or "").strip()


def _clean_waypoint_label(chap: Any) -> str:
    """Return a viewer-facing chapter/location waypoint label, filtering internal placeholder tags."""
    raw_loc = (getattr(chap, "location_tag", None) or "").strip()
    if raw_loc and raw_loc.lower() not in _GENERIC_LOCATION_TAGS:
        return raw_loc[:22]
    stripped = _strip_chapter_prefix(getattr(chap, "title", "") or "")
    return (stripped or "Story Chapter")[:22]


def _clamp_to_v1_segment(
    timeline: EditTimeline,
    start_t: float,
    dur_t: float,
) -> tuple[float, float]:
    """
    Nudge & clamp an overlay window strictly inside a single V1 clip when possible so
    Step 2 bakes Track V3 overlays directly and Step 4 final mux uses fast -c:v copy.
    """
    if not timeline.visual_track:
        return round(max(0.0, start_t), 2), round(max(0.5, dur_t), 2)

    target_vc = None
    for vc in timeline.visual_track:
        if vc.timeline_start - 0.05 <= start_t <= (vc.timeline_start + vc.duration + 0.05):
            target_vc = vc
            break
    if target_vc is None:
        target_vc = min(
            timeline.visual_track,
            key=lambda vc: abs(vc.timeline_start - start_t),
        )

    c_st = float(target_vc.timeline_start)
    c_en = c_st + float(target_vc.duration)
    if target_vc.duration >= 2.2:
        margin_in = min(0.25, target_vc.duration * 0.08)
        margin_out = min(0.20, target_vc.duration * 0.08)
        max_avail = max(1.2, (c_en - margin_out) - (c_st + margin_in))
        clamped_dur = min(float(dur_t), max_avail)
        clamped_st = max(c_st + margin_in, min(float(start_t), c_en - margin_out - clamped_dur))
        return round(clamped_st, 2), round(max(0.5, clamped_dur), 2)

    clamped_st = max(0.0, min(float(start_t), max(0.0, timeline.total_duration - 1.0)))
    clamped_dur = min(float(dur_t), max(0.5, timeline.total_duration - clamped_st))
    return round(clamped_st, 2), round(clamped_dur, 2)


def _format_gps_or_camera(
    lat: float | None,
    lon: float | None,
    fallback_cam: str | None = None,
) -> str:
    """Format GPS coordinates with N/S and E/W hemispheres, falling back to camera metadata."""
    if lat is not None and lon is not None:
        lat_hem = "N" if lat >= 0 else "S"
        lon_hem = "E" if lon >= 0 else "W"
        return f"{abs(lat):.2f}{lat_hem}, {abs(lon):.2f}{lon_hem}"
    return (fallback_cam or "").strip()


@register_agent
class MotionGraphicsTitleAgent(BaseEditingAgent):
    """Stage 40 Agent: Consolidates Track V3 overlays, route maps, quote badges, EXIF stamps, closing card & HDR-safe subtitles."""

    role_id = "title_designer"
    legacy_role_aliases = ["motion_graphics"]
    display_name = "4a. Motion Graphics, Typography & Title Designer"
    stage_order = 40
    substage_priority = 0
    description = (
        "Art-directs and consolidates Track V3 overlays across 6 Cinema Title Themes and 12 kinetic qtrle/card archetypes "
        "(knockout stencil opening, magazine chapter plates, curved GPS ribbons, travel map PiP telemetry cards, "
        "editorial wall-notes, credits colophon, frosted glass lower-thirds & quote badges with live backdrop blur), "
        "renders Cinema 2.5D Travel Map flyover cutaways onto Track V2 when multi-stop journeys exist, "
        "configures bilingual/word-highlighted subtitles, and clamps RGBA peak white to ~203-nit HDR diffuse white."
    )
    enabled_by_default = True
    requires_artifacts = ["creative_brief.json", "paper_edit.json", "timeline.yaml"]
    produces_artifacts = ["graphics_manifest.json", "motion_graphics_plan.json"]
    skill_reference_file = "12_motion_graphics_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline
        paper = blackboard.paper_edit
        brief = blackboard.creative_brief
        catalog = blackboard.catalog
        if timeline is None or paper is None:
            return {"graphics_injected": 0}

        is_hdr = bool(timeline.settings.hdr)
        theme = brief.title_theme if brief is not None else (timeline.subtitles.theme or "apple_tv_glass")
        theme_spec = get_theme_spec(theme)
        use_glass = bool(theme_spec.get("glass_default", False))

        primary_font, secondary_font = resolve_subtitle_font_stack(timeline.subtitles.font_name)
        timeline.subtitles.font_name = primary_font
        timeline.subtitles.theme = theme  # type: ignore[assignment]
        if brief is not None:
            timeline.subtitles.style_preset = brief.subtitle_style
            timeline.subtitles.burn_in = bool(brief.burn_in_subtitles)
        _sync_word_timestamps_onto_subtitles(timeline, blackboard.media_index)

        gps_assets = 0
        first_exif_stamp_meta: tuple[str, str] | None = None
        if catalog is not None:
            for asset in (*catalog.photos, *catalog.videos):
                if asset.gps_lat is not None and asset.gps_lon is not None:
                    gps_assets += 1
                if first_exif_stamp_meta is None and asset.creation_time:
                    date_str = _format_exif_date(asset.creation_time)
                    if date_str:
                        cam = getattr(asset, "camera_make_model", None) or getattr(asset, "camera_model", None)
                        loc_str = _format_gps_or_camera(asset.gps_lat, asset.gps_lon, cam)
                        first_exif_stamp_meta = (date_str, loc_str)

        # Build chapter route waypoints across the story arc (filtering placeholder tags)
        chapter_stops: list[dict[str, Any]] = []
        chap_by_id = {c.chapter_id: c for c in paper.chapters}
        for idx, chap in enumerate(paper.chapters):
            clean_lbl = _clean_waypoint_label(chap)
            chapter_stops.append(
                {
                    "index": idx + 1,
                    "chapter_id": chap.chapter_id,
                    "label": clean_lbl[:24],
                    "time_of_day": chap.time_of_day,
                }
            )

        act_start_times: dict[str, float] = {}
        for clip in timeline.visual_track:
            if clip.act and clip.act not in act_start_times:
                act_start_times[clip.act] = clip.timeline_start

        # Load Stage 25 narrative script to check delivery_mode (visual_editorial_cards vs spoken_voiceover vs none)
        narration_data = blackboard.custom_artifacts.get("narration_script.json")
        if narration_data is None and (blackboard.project_dir / "narration_script.json").exists():
            try:
                narration_data = NarrationScriptReport.from_json_file(
                    blackboard.project_dir / "narration_script.json"
                ).model_dump(mode="json")
            except Exception:
                narration_data = None
        delivery_mode = (
            str(narration_data.get("delivery_mode") or "visual_editorial_cards")
            if isinstance(narration_data, dict)
            else (getattr(brief, "narration_delivery_mode", "visual_editorial_cards") if brief else "visual_editorial_cards")
        )

        # 1. Opening card right after any cold-open hook
        hook_end = 0.0
        if timeline.visual_track and timeline.visual_track[0].id == "clip_000_hook":
            hook_end = timeline.visual_track[0].duration

        clean_proj_title = paper.project_title.replace("_", " ").strip() or "Featured Story"
        if len(clean_proj_title) > 42:
            clean_proj_title = _smart_title_case(" ".join(clean_proj_title.split()[:5]))
        else:
            clean_proj_title = _smart_title_case(clean_proj_title)

        open_st, open_dur = _clamp_to_v1_segment(timeline, round(hook_end + 0.35, 2), 3.4)
        open_accent = sample_shot_accent_hex(
            _resolve_sample_image_for_time(open_st, timeline, catalog),
            fallback_hex=str(theme_spec["accent_hex"]),
        )
        use_stencil_opening = hook_end > 0.0 and theme in ("a24_minimal", "apple_tv_glass", "kinetic_travelogue")
        opening_style = "knockout_stencil_opening" if use_stencil_opening else "opening_card"
        opening_motion = (
            "stencil_reveal"
            if use_stencil_opening
            else ("tracking_expand" if theme in ("a24_minimal", "criterion_atelier") else "masked_rise")
        )
        open_eyebrow = _clean_waypoint_label(paper.chapters[0]) if paper.chapters else "CURATED DOCUMENTARY"

        consolidated_overlays: list[TitleOverlayItem] = [
            TitleOverlayItem(
                id="title_opening",
                style=opening_style,  # type: ignore[arg-type]
                theme=theme,  # type: ignore[arg-type]
                motion_preset=opening_motion,  # type: ignore[arg-type]
                eyebrow=open_eyebrow,
                overline=open_eyebrow,
                text=clean_proj_title,
                subtitle="A Curated Documentary Film",
                timeline_start=open_st,
                duration=open_dur,
                fade_in=0.45,
                fade_out=0.55,
                accent_color=open_accent,
                backdrop_blur=(not use_stencil_opening and use_glass),
                hdr_safe=is_hdr,
            )
        ]

        # 1b. Cinema 2.5D Travel Map & Cartography Engine (V2 Flyover Cutaways + V3 PiP Telemetry Cards)
        tm_mode = getattr(brief, "travel_map_mode", "auto") if brief is not None else "auto"
        tm_raw_style = getattr(brief, "travel_map_style", "auto") if brief is not None else "auto"
        tm_style = resolve_travel_map_style(
            requested_style=tm_raw_style,
            title_theme=theme,
            prompt_text=(getattr(brief, "raw_user_prompt", "") if brief is not None else blackboard.user_prompt),
        )
        subclip_by_id = (
            {sc.subclip_id: sc for sc in blackboard.media_index.subclips}
            if blackboard.media_index is not None
            else {}
        )
        force_syn = tm_mode in ("flyover_interlude", "picture_in_picture", "hybrid")
        waypoints = resolve_journey_waypoints(
            chapters=paper.chapters,
            subclip_by_id=subclip_by_id,
            catalog=catalog,
            user_prompt=blackboard.user_prompt,
            force_synthetic_if_missing=force_syn,
        )

        # Populate per-waypoint destination hero thumbnails for polaroid/badge map callouts
        kf_by_vid_path: dict[str, str] = {}
        photo_by_asset_id: dict[str, str] = {}
        selected_photos: list[str] = []
        if catalog is not None:
            for p in getattr(catalog, "photos", []):
                cand_p = p.preview_path or p.path
                if cand_p and Path(cand_p).exists():
                    photo_by_asset_id[p.id] = str(cand_p)
            for v in getattr(catalog, "videos", []):
                for sc_obj in getattr(v, "scenes", []):
                    kp = getattr(sc_obj, "keyframe_path", None)
                    if kp and Path(kp).exists():
                        kf_by_vid_path[str(v.path)] = str(kp)
                        break
        if blackboard.photo_curation is not None:
            for pc_it in getattr(blackboard.photo_curation, "items", []):
                if getattr(pc_it, "is_selected", False) and Path(pc_it.source_path).exists():
                    selected_photos.append(str(pc_it.source_path))

        thumbs_dir = blackboard.project_dir / "cache" / "maps" / "thumbs"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        for wp_idx, wp in enumerate(waypoints):
            hero_src: Optional[str] = None
            ch_obj = chap_by_id.get(wp.chapter_id)
            if ch_obj is not None:
                for sid in ch_obj.selected_subclip_ids:
                    sc = subclip_by_id.get(sid)
                    if sc is None:
                        continue
                    if sc.media_type == "photo" and Path(sc.source_path).exists():
                        hero_src = str(sc.source_path)
                        break
                    if sc.source_asset_id in photo_by_asset_id:
                        hero_src = photo_by_asset_id[sc.source_asset_id]
                        break
                    if str(sc.source_path) in kf_by_vid_path:
                        hero_src = kf_by_vid_path[str(sc.source_path)]
                        break
                if hero_src is None:
                    for vc in timeline.visual_track:
                        if vc.act == ch_obj.title:
                            if vc.type == "photo" and Path(vc.source).exists():
                                hero_src = str(vc.source)
                                break
                            if str(vc.source) in kf_by_vid_path:
                                hero_src = kf_by_vid_path[str(vc.source)]
                                break
            if hero_src is None and wp_idx < len(selected_photos):
                hero_src = selected_photos[wp_idx]
            if hero_src:
                try:
                    with Image.open(hero_src) as raw_im:
                        fit_im = ImageOps.fit(
                            ImageOps.exif_transpose(raw_im).convert("RGB"),
                            (320, 200),
                            method=Image.Resampling.LANCZOS,
                        )
                        thumb_p = thumbs_dir / f"{wp.waypoint_id}.jpg"
                        fit_im.save(thumb_p, "JPEG", quality=88)
                        wp.hero_thumbnail_path = str(thumb_p)
                except Exception:
                    pass

        leg_kms = [
            haversine_km(waypoints[i].lat, waypoints[i].lon, waypoints[i + 1].lat, waypoints[i + 1].lon)
            for i in range(max(0, len(waypoints) - 1))
        ]
        total_route_km = round(sum(leg_kms), 2)
        max_leg_km = max(leg_kms, default=0.0)
        map_active = (tm_mode != "off") and (len(waypoints) >= 2) and (force_syn or max_leg_km >= 0.35)

        map_segments: list[TravelMapSegmentSpec] = []
        v2_map_windows: list[tuple[float, float]] = []
        pip_windows: list[tuple[float, float]] = []
        busy_v2: list[tuple[float, float]] = [
            (float(b.timeline_start), float(b.timeline_start) + float(b.duration))
            for b in timeline.broll_track
        ]
        act_start_by_cid = {c.chapter_id: act_start_times.get(c.title) for c in paper.chapters}

        def _overlaps_intervals(st_val: float, en_val: float, ivs: list[tuple[float, float]]) -> bool:
            return any(st_val < (iv_en - 0.04) and en_val > (iv_st + 0.04) for iv_st, iv_en in ivs)

        if map_active and timeline.visual_track:
            n_total_legs = len(waypoints) - 1
            if n_total_legs <= 8:
                chosen_legs = list(range(n_total_legs))
            else:
                step = (n_total_legs - 1) / 7.0
                chosen_legs = sorted({int(round(i * step)) for i in range(8)})

            allow_tiles = os.environ.get("AI_EDITOR_OFFLINE_MAPS", "0") != "1"
            maps_dir = blackboard.project_dir / "cache" / "maps"
            tiles_cache_dir = maps_dir / "tiles"
            max_map_w = 3840 if os.environ.get("AI_EDITOR_MAP_4K") == "1" else 2560
            max_map_h = 2160 if os.environ.get("AI_EDITOR_MAP_4K") == "1" else 1440
            map_w = max(640, min(int(timeline.settings.width), max_map_w))
            map_h = max(360, min(int(timeline.settings.height), max_map_h))
            user_prompt_blob = str(getattr(brief, "raw_user_prompt", "") if brief is not None else blackboard.user_prompt)

            for leg_pos, leg_i in enumerate(chosen_legs):
                orig_wp = waypoints[leg_i]
                dest_wp = waypoints[leg_i + 1]
                leg_km = leg_kms[leg_i]
                bearing = round(initial_bearing_deg(orig_wp.lat, orig_wp.lon, dest_wp.lat, dest_wp.lon), 1)
                ch_dest = chap_by_id.get(dest_wp.chapter_id)
                mode_hint = getattr(ch_dest, "transport_mode", "auto") if ch_dest is not None else "auto"
                t_mode = infer_transport_mode(leg_km, orig_wp.label, dest_wp.label, hint=mode_hint)
                leg_style = (
                    resolve_travel_map_style("auto", theme, user_prompt_blob, transport_mode=t_mode)
                    if str(tm_raw_style or "auto").strip().lower() == "auto"
                    else tm_style
                )

                if tm_mode == "picture_in_picture":
                    want_place = "v3_pip"
                elif tm_mode == "flyover_interlude":
                    want_place = "v2_flyover"
                elif tm_mode == "hybrid":
                    want_place = "v2_flyover" if leg_pos == 0 else "v3_pip"
                else:
                    want_place = "v2_flyover" if max_leg_km >= 1.0 else "v3_pip"

                dest_t = act_start_by_cid.get(dest_wp.chapter_id)
                if dest_t is None:
                    dest_t = ((leg_i + 1) / float(max(2, len(waypoints)))) * timeline.total_duration
                dest_t = max(open_st + open_dur + 0.35 if timeline.total_duration > 10.0 else hook_end + 0.2, float(dest_t))

                placed_v2 = False
                if want_place == "v2_flyover":
                    cands = [
                        vc
                        for vc in timeline.visual_track
                        if vc.id != "clip_000_hook"
                        and float(vc.duration) >= 2.0
                        and not _overlaps_intervals(float(vc.timeline_start), float(vc.timeline_start) + float(vc.duration), busy_v2)
                    ]
                    # Prefer candidates away from opening card and preferring B-roll / photo plates first
                    non_open = [
                        vc for vc in cands if (float(vc.timeline_start) + float(vc.duration) <= open_st) or (float(vc.timeline_start) >= open_st + open_dur)
                    ]
                    pool = non_open if non_open else cands
                    if pool:
                        host_v1 = min(
                            pool,
                            key=lambda vc: (
                                0 if (vc.type == "video" and vc.roll_type == "b_roll") else (1 if vc.type == "video" else 2),
                                abs(float(vc.timeline_start) - dest_t),
                            ),
                        )
                        seg_dur = round(min(5.6, max(1.8, float(host_v1.duration) - 0.12)), 2)
                        seg_start = round(float(host_v1.timeline_start) + 0.06, 2)
                        map_mp4 = maps_dir / f"flyover_leg_{leg_i + 1:02d}.mp4"
                        render_travel_map_clip(
                            waypoints=waypoints,
                            output_mp4=map_mp4,
                            style=leg_style,
                            duration=seg_dur,
                            width=map_w,
                            height=map_h,
                            fps=int(timeline.settings.fps),
                            leg_index=leg_i,
                            transport_mode=t_mode,
                            hdr_safe=is_hdr,
                            allow_tile_fetch=allow_tiles,
                            with_audio=False,
                            cache_dir=tiles_cache_dir,
                        )
                        broll_id = f"broll_map_{len(v2_map_windows) + 1:02d}"
                        timeline.broll_track.append(
                            VisualTrackItem(
                                id=broll_id,
                                type="video",
                                track_lane=2,
                                roll_type="b_roll",
                                shot_scale="wide",
                                act=host_v1.act,
                                timeline_start=seg_start,
                                duration=seg_dur,
                                source=str(map_mp4),
                                source_in=0.0,
                                source_out=seg_dur,
                                volume=0.0,
                                audio_crossfade_sec=0.33,
                                narrative_note=f"2.5D Travel Map Flyover ({orig_wp.label} -> {dest_wp.label}, {leg_km:.1f} km, {t_mode})",
                            )
                        )
                        busy_v2.append((seg_start, seg_start + seg_dur))
                        v2_map_windows.append((seg_start, seg_start + seg_dur))
                        map_segments.append(
                            TravelMapSegmentSpec(
                                segment_id=f"tm_seg_{len(map_segments) + 1:02d}",
                                leg_index=leg_i,
                                origin=orig_wp,
                                destination=dest_wp,
                                transport_mode=t_mode,  # type: ignore[arg-type]
                                distance_km=round(leg_km, 2),
                                bearing_deg=bearing,
                                style=leg_style,
                                placement="v2_flyover",
                                timeline_start=seg_start,
                                duration=seg_dur,
                                rendered_clip_path=str(map_mp4),
                                overlay_id=broll_id,
                            )
                        )
                        placed_v2 = True

                if not placed_v2:
                    # Place non-colliding upper-right travel_map_pip_card on Track V3
                    pip_cands = sorted(timeline.visual_track, key=lambda vc: abs(float(vc.timeline_start) - dest_t))
                    chosen_st: Optional[float] = None
                    chosen_dur: Optional[float] = None
                    for vc in pip_cands:
                        cand_st, cand_dur = _clamp_to_v1_segment(timeline, max(float(vc.timeline_start), dest_t), 3.4)
                        if _overlaps_intervals(cand_st, cand_st + cand_dur, [*v2_map_windows, *pip_windows, (open_st, open_st + open_dur)]):
                            cand_st2, cand_dur2 = _clamp_to_v1_segment(timeline, float(vc.timeline_start) + 0.25, 3.2)
                            if _overlaps_intervals(cand_st2, cand_st2 + cand_dur2, [*v2_map_windows, *pip_windows, (open_st, open_st + open_dur)]):
                                continue
                            cand_st, cand_dur = cand_st2, cand_dur2
                        chosen_st, chosen_dur = cand_st, cand_dur
                        break
                    if chosen_st is not None and chosen_dur is not None:
                        pip_id = f"title_map_pip_{leg_i + 1:02d}"
                        km_txt = f"{leg_km:.0f} km" if leg_km >= 2.0 else f"{leg_km * 1000.0:.0f} m"
                        wp_dicts = [
                            {
                                "waypoint_id": w.waypoint_id,
                                "chapter_id": w.chapter_id,
                                "label": w.label,
                                "subtitle": w.subtitle,
                                "lat": w.lat,
                                "lon": w.lon,
                                "alt_m": w.alt_m,
                                "source": w.source,
                                "active": (idx_w == (leg_i + 1)),
                                "completed": (idx_w <= (leg_i + 1)),
                            }
                            for idx_w, w in enumerate(waypoints)
                        ]
                        consolidated_overlays.append(
                            TitleOverlayItem(
                                id=pip_id,
                                style="travel_map_pip_card",
                                theme=theme,  # type: ignore[arg-type]
                                motion_preset="masked_rise",
                                eyebrow="ROUTE TELEMETRY",
                                overline="ROUTE TELEMETRY",
                                text=f"{orig_wp.label} -> {dest_wp.label}"[:28],
                                subtitle=f"{km_txt} | {t_mode.upper()}",
                                timeline_start=chosen_st,
                                duration=chosen_dur,
                                fade_in=0.35,
                                fade_out=0.35,
                                accent_color=open_accent,
                                backdrop_blur=True,
                                map_waypoints=wp_dicts,
                                hdr_safe=is_hdr,
                            )
                        )
                        pip_windows.append((chosen_st, chosen_st + chosen_dur))
                        map_segments.append(
                            TravelMapSegmentSpec(
                                segment_id=f"tm_seg_{len(map_segments) + 1:02d}",
                                leg_index=leg_i,
                                origin=orig_wp,
                                destination=dest_wp,
                                transport_mode=t_mode,  # type: ignore[arg-type]
                                distance_km=round(leg_km, 2),
                                bearing_deg=bearing,
                                style=leg_style,
                                placement="v3_pip",
                                timeline_start=chosen_st,
                                duration=chosen_dur,
                                overlay_id=pip_id,
                            )
                        )

        timeline.broll_track.sort(key=lambda c: float(c.timeline_start))
        tm_manifest = TravelMapManifest(
            project_title=clean_proj_title,
            enabled=bool(map_segments),
            travel_map_mode=tm_mode,  # type: ignore[arg-type]
            resolved_style=tm_style,
            total_distance_km=total_route_km,
            waypoints=waypoints,
            segments=map_segments,
        )
        blackboard.travel_map_manifest = tm_manifest
        blackboard.set_custom_artifact("travel_map_manifest.json", tm_manifest.model_dump(mode="json"))
        tm_manifest.to_json_file(blackboard.project_dir / "travel_map_manifest.json")

        # 2. Route progression cards on alternate chapter transitions + alternating chapter plates / frosted lower-thirds
        mg_items: list[MotionGraphicItem] = []
        if len(paper.chapters) >= 3:
            target_chapters = [c for c_idx, c in enumerate(paper.chapters) if c_idx >= 1 and c_idx % 2 == 1][:3]
        elif len(paper.chapters) > 1:
            target_chapters = paper.chapters[1:2]
        else:
            target_chapters = paper.chapters[:1]
        route_chap_ids = {c.chapter_id for c in target_chapters}

        for c_idx, chap in enumerate(target_chapters, start=1):
            st_sec = act_start_times.get(chap.title, c_idx * (timeline.total_duration / max(2, len(paper.chapters))))
            st_sec = min(max(4.5, st_sec + 0.4), max(4.5, timeline.total_duration - 4.5))
            chap_lbl = _clean_waypoint_label(chap)
            waypoints_for_card = [
                {
                    "label": stop["label"],
                    "active": stop["chapter_id"] == chap.chapter_id,
                    "completed": stop["index"] <= (c_idx + (1 if len(paper.chapters) > 1 else 0)),
                }
                for stop in chapter_stops[:4]
            ]
            mg_items.append(
                MotionGraphicItem(
                    graphic_id=f"mg_route_{c_idx:02d}",
                    chapter_id=chap.chapter_id,
                    title=chap.title,
                    subtitle=f"Chapter {min(len(chapter_stops), c_idx + 1)} of {len(chapter_stops)} · {chap_lbl}",
                    timeline_start=round(st_sec, 2),
                    duration=3.2,
                    waypoints=waypoints_for_card,
                    rationale=f"Spatial route progression bridge into '{chap.title}' ({chap_lbl}).",
                )
            )

        ai_resp = self.ask_ai_creative_json(
            blackboard=blackboard,
            prompt=(
                f"Project has {len(paper.chapters)} chapters and {gps_assets} GPS-tagged assets.\n"
                f"Initial route progression graphics: {[m.model_dump(mode='json') for m in mg_items]}\n"
                "Optionally return JSON {\"items\": [...]} to customize waypoint labels, titles, and timing."
            ),
            decision_key=self.role_id,
            timeout_sec=20.0,
        )
        if isinstance(ai_resp, dict) and isinstance(ai_resp.get("items"), list) and ai_resp["items"]:
            try:
                mg_items = [MotionGraphicItem.model_validate(it) for it in ai_resp["items"]]
            except Exception:
                pass

        # Helper to check if a bottom-third card would collide with a burned-in photo plaque or active subtitle
        def _has_bottom_collision(start_t: float, dur_t: float, style_name: str) -> bool:
            end_t = start_t + dur_t
            # Check photo burned-in caption plaque (top-level or nested ken_burns caption)
            for vc in timeline.visual_track:
                if vc.timeline_start < end_t and (vc.timeline_start + vc.duration) > start_t:
                    has_plaque = bool(
                        vc.caption_title
                        or (vc.ken_burns and (vc.ken_burns.caption_title or vc.ken_burns.caption_subtitle))
                    )
                    if vc.type == "photo" and has_plaque:
                        return True
            # Route map cards & curved ribbons sit at y=[0.80..0.96], so avoid overlapping subtitles at y=[0.85..0.96]
            sub_box = [0.10, 0.85, 0.90, 0.96]
            if _bbox_intersection_area(sub_box, compute_overlay_bbox_norm(style_name)) > 0.015:
                for sub in timeline.subtitles.items:
                    if sub.start < end_t and sub.end > start_t:
                        return True
            return False

        for mg in mg_items:
            st_t, dur_t = _clamp_to_v1_segment(timeline, mg.timeline_start, mg.duration)
            route_style = "curved_gps_ribbon" if len(mg.waypoints) >= 2 else "route_map_card"
            ch_accent = sample_shot_accent_hex(
                _resolve_sample_image_for_time(st_t, timeline, catalog),
                fallback_hex=str(theme_spec["accent_hex"]),
            )
            if _has_bottom_collision(st_t, dur_t, route_style):
                has_photo_plaque = any(
                    vc.type == "photo" and vc.caption_title and vc.timeline_start < (st_t + dur_t) and (vc.timeline_start + vc.duration) > st_t
                    for vc in timeline.visual_track
                )
                if not has_photo_plaque:
                    consolidated_overlays.append(
                        TitleOverlayItem(
                            id=mg.graphic_id,
                            style="lower_third",
                            theme=theme,  # type: ignore[arg-type]
                            motion_preset="masked_rise",
                            eyebrow="ROUTE PROGRESSION",
                            overline="ROUTE PROGRESSION",
                            text=mg.title,
                            subtitle=mg.subtitle,
                            timeline_start=st_t,
                            duration=dur_t,
                            fade_in=0.35,
                            fade_out=0.35,
                            accent_color=ch_accent,
                            backdrop_blur=True,
                            hdr_safe=is_hdr,
                        )
                    )
                continue

            consolidated_overlays.append(
                TitleOverlayItem(
                    id=mg.graphic_id,
                    style=route_style,  # type: ignore[arg-type]
                    theme=theme,  # type: ignore[arg-type]
                    motion_preset="bezier_route_draw" if route_style == "curved_gps_ribbon" else "masked_rise",
                    eyebrow="JOURNEY ROUTE" if route_style == "curved_gps_ribbon" else None,
                    overline="JOURNEY ROUTE" if route_style == "curved_gps_ribbon" else "",
                    text=mg.title,
                    subtitle=mg.subtitle,
                    timeline_start=st_t,
                    duration=dur_t,
                    fade_in=0.35,
                    fade_out=0.35,
                    accent_color=ch_accent,
                    backdrop_blur=use_glass,
                    map_waypoints=mg.waypoints,
                    hdr_safe=is_hdr,
                )
            )

        # Alternate magazine_chapter_plate (with Roman numerals I..VIII) & frosted lower_third on non-route chapters
        plate_turn = 0
        for chap_idx, chap in enumerate(paper.chapters):
            if chap.chapter_id in route_chap_ids or chap == paper.chapters[0]:
                continue
            st_sec = act_start_times.get(chap.title)
            if st_sec is None or st_sec >= timeline.total_duration - 5.0:
                continue
            lt_start, lt_dur = _clamp_to_v1_segment(timeline, round(st_sec + 0.45, 2), 3.0)
            roman = _ROMAN_NUMERALS[chap_idx] if chap_idx < len(_ROMAN_NUMERALS) else f"{chap_idx + 1:02d}"
            use_mag_plate = (plate_turn % 2 == 0)
            card_style = "magazine_chapter_plate" if use_mag_plate else "lower_third"
            plate_turn += 1
            if _has_bottom_collision(lt_start, lt_dur, card_style):
                # Relocate onto first non-plaque clip in the chapter, else promote to top-left editorial_caption
                clean_vc = next(
                    (
                        vc
                        for vc in timeline.visual_track
                        if vc.act == chap.title
                        and vc.duration >= 2.4
                        and not _has_bottom_collision(
                            round(vc.timeline_start + 0.35, 2),
                            round(min(3.0, vc.duration - 0.5), 2),
                            card_style,
                        )
                    ),
                    None,
                )
                if clean_vc is not None:
                    lt_start, lt_dur = _clamp_to_v1_segment(timeline, round(clean_vc.timeline_start + 0.35, 2), 3.0)
                else:
                    card_style = "editorial_caption"
                    use_mag_plate = False
            chap_lbl = _clean_waypoint_label(chap)
            plate_accent = sample_shot_accent_hex(
                _resolve_sample_image_for_time(lt_start, timeline, catalog),
                fallback_hex=str(theme_spec["accent_hex"]),
            )
            kicker = f"CHAPTER {roman}" if use_mag_plate else chap_lbl
            consolidated_overlays.append(
                TitleOverlayItem(
                    id=f"title_lt_{chap.chapter_id}",
                    style=card_style,  # type: ignore[arg-type]
                    theme=theme,  # type: ignore[arg-type]
                    motion_preset="masked_rise",
                    eyebrow=kicker,
                    overline=kicker,
                    chapter_numeral=roman if use_mag_plate else "",
                    text=chap.title,
                    subtitle=chap.narrative_goal[:56] if chap.narrative_goal else chap_lbl,
                    timeline_start=lt_start,
                    duration=lt_dur,
                    fade_in=0.30,
                    fade_out=0.35,
                    accent_color=plate_accent,
                    backdrop_blur=(not use_mag_plate) or use_glass,
                    hdr_safe=is_hdr,
                )
            )

        # 2b. Convert Stage 25 narrative cues into upper-left curatorial editorial captions (zero plaque/subtitle overlap)
        if delivery_mode == "visual_editorial_cards" and isinstance(narration_data, dict):
            raw_cues = narration_data.get("cues") or []
            for idx_cue, cue in enumerate(raw_cues, start=1):
                if not isinstance(cue, dict):
                    continue
                body = str(cue.get("spoken_text") or "").strip()
                if not body:
                    continue
                raw_t = float(cue.get("target_timeline_sec") or 0.0)
                if timeline.total_duration > 10.0:
                    raw_t = max(raw_t, open_st + open_dur + 0.45)
                dur_pref = min(5.2, max(3.4, float(cue.get("duration_sec") or 4.2)))
                ed_st, ed_dur = _clamp_to_v1_segment(timeline, raw_t, dur_pref)
                chap_obj = chap_by_id.get(str(cue.get("chapter_id") or ""))
                chap_lbl = _clean_waypoint_label(chap_obj) if chap_obj else ""
                headline = str(cue.get("headline") or "").strip() or (chap_lbl or "Story Note")
                eyebrow = f"CURATORIAL NOTE · {chap_lbl}" if chap_lbl else "CURATORIAL NOTE"
                consolidated_overlays.append(
                    TitleOverlayItem(
                        id=f"title_ed_{idx_cue:02d}",
                        style="editorial_caption",
                        theme=theme,  # type: ignore[arg-type]
                        motion_preset="masked_rise",
                        eyebrow=eyebrow,
                        overline=eyebrow,
                        text=headline,
                        subtitle=body,
                        timeline_start=ed_st,
                        duration=ed_dur,
                        fade_in=0.35,
                        fade_out=0.35,
                        accent_color=open_accent,
                        backdrop_blur=use_glass,
                        hdr_safe=is_hdr,
                    )
                )

        # 2c. Add clean lower-third titles for uncaptioned B-roll video segments when metadata exists
        if delivery_mode != "none":
            sc_by_src: dict[str, list[Any]] = {}
            if blackboard.media_index is not None:
                for sc in blackboard.media_index.subclips:
                    if not sc.is_discarded:
                        sc_by_src.setdefault(str(sc.source_path), []).append(sc)
            for vc in timeline.visual_track:
                if vc.type != "video" or vc.roll_type != "b_roll" or vc.id == "clip_000_hook":
                    continue
                if vc.ken_burns and vc.ken_burns.caption_title:
                    continue
                has_existing_bottom = any(
                    compute_overlay_bbox_norm(ov.style)[1] >= 0.60
                    and ov.timeline_start < (vc.timeline_start + vc.duration - 0.1)
                    and (ov.timeline_start + ov.duration) > (vc.timeline_start + 0.1)
                    for ov in consolidated_overlays
                )
                if has_existing_bottom:
                    continue
                v_title = ""
                v_sub = ""
                for sc in sc_by_src.get(str(vc.source), []):
                    if sc.in_tc <= (vc.source_in + 0.6) and sc.out_tc >= (vc.source_in + 0.2):
                        v_title = (sc.annotation_title or "").strip()
                        v_sub = (sc.annotation_subtitle or "").strip()
                        if v_title:
                            break
                if not v_title and vc.narrative_note:
                    v_title = re.sub(r"^\[[^\]]+\]\s*", "", vc.narrative_note).strip()
                is_generic_auto = v_title.startswith(
                    ("Living Tableau", "Steady Frame", "Cinematic Scene", "Opening Establishing Vista", "Closing Finale Vista")
                ) or bool(re.search(r"— Scene \d+$", v_title))
                if len(v_title) >= 4 and not is_generic_auto:
                    lt_st, lt_dur = _clamp_to_v1_segment(timeline, vc.timeline_start + 0.35, min(3.8, max(2.2, vc.duration - 0.6)))
                    kicker = _strip_chapter_prefix(vc.act or "GALLERY MOMENT")[:26]
                    consolidated_overlays.append(
                        TitleOverlayItem(
                            id=f"title_vid_{vc.id}",
                            style="lower_third",
                            theme=theme,  # type: ignore[arg-type]
                            motion_preset="masked_rise",
                            eyebrow=kicker,
                            overline=kicker,
                            text=v_title,
                            subtitle=v_sub or None,
                            timeline_start=lt_st,
                            duration=lt_dur,
                            fade_in=0.30,
                            fade_out=0.35,
                            accent_color=open_accent,
                            backdrop_blur=True,
                            hdr_safe=is_hdr,
                        )
                    )

        # 3. Dialogue quote badge (upper-left [0.06, 0.10, 0.58, 0.25] -> zero collision with bottom subtitles!)
        if timeline.subtitles.items:
            best_sub = max(timeline.subtitles.items, key=lambda s: len(s.text))
            if len(best_sub.text.strip()) >= 6 and best_sub.start > open_st + open_dur + 0.3:
                qb_st, qb_dur = _clamp_to_v1_segment(
                    timeline,
                    round(best_sub.start + 0.15, 2),
                    round(min(3.2, max(1.8, best_sub.end - best_sub.start)), 2),
                )
                consolidated_overlays.append(
                    TitleOverlayItem(
                        id="title_quote_badge_01",
                        style="dialogue_badge",
                        theme=theme,  # type: ignore[arg-type]
                        motion_preset="masked_rise",
                        eyebrow="CANDID SPOKEN MOMENT",
                        overline="CANDID SPOKEN MOMENT",
                        text=f'"{best_sub.text[:36]}"',
                        subtitle=best_sub.secondary_text or "Family Dialogue",
                        timeline_start=qb_st,
                        duration=qb_dur,
                        fade_in=0.30,
                        fade_out=0.30,
                        accent_color=open_accent,
                        backdrop_blur=use_glass,
                        hdr_safe=is_hdr,
                    )
                )

        # 4. Optional EXIF archival date/location stamp (upper-right) when GPS/timestamp metadata is available
        prompt_low = (brief.raw_user_prompt if brief else blackboard.user_prompt).lower()
        if first_exif_stamp_meta is not None and (gps_assets > 0 or "exif" in prompt_low or "stamp" in prompt_low):
            stamp_date, stamp_loc = first_exif_stamp_meta
            raw_stamp_start = round(min(max(open_st + open_dur + 0.5, 4.4), max(0.5, timeline.total_duration * 0.25)), 2)
            stamp_start, stamp_dur = _clamp_to_v1_segment(timeline, raw_stamp_start, 2.6)
            if stamp_start + stamp_dur < timeline.total_duration - 3.6:
                consolidated_overlays.append(
                    TitleOverlayItem(
                        id="title_exif_stamp_01",
                        style="exif_stamp",
                        theme=theme,  # type: ignore[arg-type]
                        motion_preset="typewriter_telemetry",
                        eyebrow="EXIF TELEMETRY",
                        overline="EXIF TELEMETRY",
                        text=stamp_date,
                        subtitle=stamp_loc,
                        timeline_start=stamp_start,
                        duration=stamp_dur,
                        fade_in=0.30,
                        fade_out=0.35,
                        accent_color=open_accent,
                        backdrop_blur=use_glass,
                        hdr_safe=is_hdr,
                    )
                )

        # 5. Closing card / Editorial Colophon at the tail of the film (moved out of ShowrunnerAgent into Stage 40)
        if timeline.total_duration >= 8.0:
            raw_closing_dur = 3.2
            raw_closing_start = round(max(open_st + open_dur + 0.4, timeline.total_duration - raw_closing_dur - 0.2), 2)
            closing_start, closing_dur = _clamp_to_v1_segment(timeline, raw_closing_start, raw_closing_dur)
            use_colophon = len(paper.chapters) >= 2
            cams: list[str] = []
            if catalog is not None:
                cams = sorted(
                    {
                        str(getattr(a, "camera_model", "") or "").strip()
                        for a in [*getattr(catalog, "videos", []), *getattr(catalog, "photos", [])]
                        if str(getattr(a, "camera_model", "") or "").strip()
                    }
                )
            colophon_lines = (
                [
                    f"{len(paper.chapters)} CHAPTERS · {len(timeline.visual_track)} TIMELINE CUTS",
                    f"OPTICS · {', '.join(cams[:2])}" if cams else "MULTI-CAM ARCHIVE SELECTION",
                    f"GRADE · {timeline.settings.width}x{timeline.settings.height} {'BT.2020 HLG' if is_hdr else 'REC.709'}",
                ]
                if use_colophon
                else []
            )
            # Drop any non-opening bottom/center overlay that overlaps the closing card window
            consolidated_overlays = [
                t
                for t in consolidated_overlays
                if t.id == "title_opening"
                or compute_overlay_bbox_norm(t.style)[1] < 0.27
                or (t.timeline_start + t.duration) <= closing_start - 0.35
            ]
            closing_kicker = "END CREDITS & COLOPHON" if use_colophon else "MEMORIES PRESERVED"
            consolidated_overlays.append(
                TitleOverlayItem(
                    id="title_closing",
                    style="credits_colophon_card" if use_colophon else "closing_card",
                    theme=theme,  # type: ignore[arg-type]
                    motion_preset="tracking_expand",
                    eyebrow=closing_kicker,
                    overline=closing_kicker,
                    colophon_lines=colophon_lines,
                    text="Fin · Memories Preserved",
                    subtitle=clean_proj_title,
                    timeline_start=closing_start,
                    duration=closing_dur,
                    fade_in=0.55,
                    fade_out=0.65,
                    accent_color=open_accent,
                    hdr_safe=is_hdr,
                )
            )

        # Ensure Track V3 localized plaques/ribbons never cover active Track V2 travel map flyovers
        if v2_map_windows:
            consolidated_overlays = [
                ov
                for ov in consolidated_overlays
                if ov.style in ("opening_card", "knockout_stencil_opening", "closing_card", "credits_colophon_card")
                or not any(
                    ov.timeline_start < (w_en + 0.20) and (ov.timeline_start + ov.duration) > (w_st - 0.20)
                    for w_st, w_en in v2_map_windows
                )
            ]

        # De-conflict overlays: never overlap full-canvas knockout stencil opening, and keep 3-zone spatial discipline
        def _zone_for(style_name: str) -> str:
            x1, y1, x2, y2 = compute_overlay_bbox_norm(style_name)
            if y2 < 0.32 and x2 < 0.65:
                return "upper_left"
            if y2 <= 0.44 and x1 >= 0.55:
                return "upper_right"
            return "bottom_center"

        consolidated_overlays.sort(key=lambda t: t.timeline_start)
        deduped_overlays: list[TitleOverlayItem] = []
        last_by_zone: dict[str, int] = {}
        for ov in consolidated_overlays:
            ov.hdr_safe = is_hdr
            if (
                opening_style == "knockout_stencil_opening"
                and ov.id != "title_opening"
                and ov.timeline_start < (open_st + open_dur + 0.25)
                and (ov.timeline_start + ov.duration) > max(0.0, open_st - 0.15)
            ):
                continue
            z = _zone_for(ov.style)
            if z in last_by_zone:
                prev_idx = last_by_zone[z]
                prev = deduped_overlays[prev_idx]
                if ov.timeline_start < (prev.timeline_start + prev.duration + 0.45):
                    if ov.style in (
                        "opening_card",
                        "closing_card",
                        "knockout_stencil_opening",
                        "credits_colophon_card",
                        "dialogue_badge",
                    ):
                        deduped_overlays[prev_idx] = ov
                    continue
            last_by_zone[z] = len(deduped_overlays)
            deduped_overlays.append(ov)

        timeline.title_track = deduped_overlays

        if blackboard.pending_revision_directives(self.role_id):
            blackboard.mark_directives_resolved(self.role_id)

        proj_title = brief.project_title if brief else timeline.project_name
        peak_rgb = HDR_DIFFUSE_WHITE_PEAK_RGB if is_hdr else 255
        overlay_specs = [
            GraphicsOverlaySpec(
                overlay_id=t.id,
                style=t.style,
                theme=t.theme,
                motion_preset=t.motion_preset,
                backdrop_blur=t.backdrop_blur,
                overlay_kind=t.style,
                text=t.text,
                subtitle=t.subtitle or "",
                timeline_start=t.timeline_start,
                duration=t.duration,
                bbox_norm=compute_overlay_bbox_norm(t.style),
                hdr_safe_clamped=is_hdr,
                peak_rgb_limit=peak_rgb,
            )
            for t in timeline.title_track
        ]
        gfx_manifest = GraphicsManifest(
            project_title=proj_title,
            title_theme=theme,  # type: ignore[arg-type]
            hdr_safe_mode=is_hdr,
            primary_font_family=primary_font,
            secondary_font_family=secondary_font,
            subtitle_count=len(timeline.subtitles.items),
            overlays_count=len(overlay_specs),
            overlays=overlay_specs,
        )
        gfx_path = blackboard.project_dir / "graphics_manifest.json"
        gfx_manifest.to_json_file(gfx_path)
        blackboard.graphics_manifest = gfx_manifest

        # Also persist backward-compatible motion_graphics_plan.json
        mg_report = MotionGraphicsReport(
            project_title=proj_title,
            gps_Tagged_assets_found=gps_assets,
            graphics_injected=len(overlay_specs),
            items=mg_items,
        )
        mg_out_path = blackboard.project_dir / "motion_graphics_plan.json"
        mg_report.to_json_file(mg_out_path)
        blackboard.set_custom_artifact("motion_graphics_plan.json", mg_report.model_dump(mode="json"))
        timeline.to_yaml(blackboard.project_dir / "timeline.yaml")

        blackboard.emit_activity(
            self.role_id,
            f"Populated {len(overlay_specs)} Track V3 overlays & {len(timeline.subtitles.items)} bilingual subtitles ({primary_font}/{secondary_font})",
            detail=f"HDR-safe diffuse white clamping={'ON (<=192 RGB)' if is_hdr else 'OFF (SDR)'}",
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="vfx_designer",
            message_type="handoff",
            subject=f"Consolidated {len(overlay_specs)} Track V3 titles, route cards, badges & closing card",
            payload={"overlays_count": len(overlay_specs), "hdr_safe_mode": is_hdr},
        )
        return {
            "graphics_injected": len(overlay_specs),
            "gps_assets_found": gps_assets,
            "hdr_safe_mode": is_hdr,
            "graphics_manifest_path": str(gfx_path),
            "motion_graphics_plan_path": str(mg_out_path),
        }


# Backward-compatible class alias
MotionGraphicsAgent = MotionGraphicsTitleAgent
