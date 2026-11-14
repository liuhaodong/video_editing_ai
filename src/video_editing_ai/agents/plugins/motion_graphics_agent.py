"""Stage 40 Specialist: Motion Graphics, Lower-Thirds, Bilingual Subtitles & Closing Card Designer (title_designer)."""

from __future__ import annotations

import re
from typing import Any

from ...models.schema import (
    EditTimeline,
    GraphicsManifest,
    GraphicsOverlaySpec,
    MotionGraphicItem,
    MotionGraphicsReport,
    NarrationScriptReport,
    TitleOverlayItem,
)
from ...renderer.subtitles import resolve_subtitle_font_stack
from ...renderer.titles import HDR_DIFFUSE_WHITE_PEAK_RGB, compute_overlay_bbox_norm
from ..base import BaseEditingAgent, PipelineBlackboard, register_agent


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
        "Consolidates all Track V3 graphics (opening card, chapter lower-thirds, GPS route progression cards, "
        "dialogue quote badges, optional EXIF stamps, and closing card), configures bilingual subtitles "
        "(PingFang SC / Avenir Next / platform fallbacks), and clamps RGBA peak white to HDR-safe diffuse white."
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
        primary_font, secondary_font = resolve_subtitle_font_stack(timeline.subtitles.font_name)
        timeline.subtitles.font_name = primary_font

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
        consolidated_overlays: list[TitleOverlayItem] = [
            TitleOverlayItem(
                id="title_opening",
                style="opening_card",
                eyebrow=_clean_waypoint_label(paper.chapters[0]) if paper.chapters else "OPENING",
                text=clean_proj_title,
                subtitle="A Curated Documentary Film",
                timeline_start=open_st,
                duration=open_dur,
                fade_in=0.45,
                fade_out=0.55,
                accent_color="#E8E4DC",
                hdr_safe=is_hdr,
            )
        ]

        # 2. Route progression cards on major location transitions (up to 3) + clean chapter lower-thirds
        mg_items: list[MotionGraphicItem] = []
        target_chapters = paper.chapters[1:4] if len(paper.chapters) > 1 else paper.chapters[:1]
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
            # Check photo burned-in caption plaque
            for vc in timeline.visual_track:
                if vc.timeline_start < end_t and (vc.timeline_start + vc.duration) > start_t:
                    if vc.type == "photo" and vc.caption_title:
                        return True
            # Route map cards sit at y=[0.79..0.96], so avoid overlapping active subtitles at y=[0.85..0.96]
            if style_name == "route_map_card":
                for sub in timeline.subtitles.items:
                    if sub.start < end_t and sub.end > start_t:
                        return True
            return False

        for mg in mg_items:
            st_t, dur_t = _clamp_to_v1_segment(timeline, mg.timeline_start, mg.duration)
            if _has_bottom_collision(st_t, dur_t, "route_map_card"):
                has_photo_plaque = any(
                    vc.type == "photo" and vc.caption_title and vc.timeline_start < (st_t + dur_t) and (vc.timeline_start + vc.duration) > st_t
                    for vc in timeline.visual_track
                )
                if not has_photo_plaque:
                    consolidated_overlays.append(
                        TitleOverlayItem(
                            id=mg.graphic_id,
                            style="lower_third",
                            eyebrow="CHAPTER",
                            text=mg.title,
                            subtitle=mg.subtitle,
                            timeline_start=st_t,
                            duration=dur_t,
                            fade_in=0.35,
                            fade_out=0.35,
                            accent_color="#E8E4DC",
                            hdr_safe=is_hdr,
                        )
                    )
                continue

            consolidated_overlays.append(
                TitleOverlayItem(
                    id=mg.graphic_id,
                    style="route_map_card",
                    text=mg.title,
                    subtitle=mg.subtitle,
                    timeline_start=st_t,
                    duration=dur_t,
                    fade_in=0.35,
                    fade_out=0.35,
                    accent_color="#E8E4DC",
                    map_waypoints=mg.waypoints,
                    hdr_safe=is_hdr,
                )
            )

        # Add chapter lower-thirds for remaining chapters (>4) that do not have route cards or photo plaques
        for chap in paper.chapters:
            if chap.chapter_id in route_chap_ids or chap == paper.chapters[0]:
                continue
            st_sec = act_start_times.get(chap.title)
            if st_sec is None or st_sec >= timeline.total_duration - 5.0:
                continue
            lt_start, lt_dur = _clamp_to_v1_segment(timeline, round(st_sec + 0.45, 2), 3.0)
            if not _has_bottom_collision(lt_start, lt_dur, "lower_third"):
                chap_lbl = _clean_waypoint_label(chap)
                consolidated_overlays.append(
                    TitleOverlayItem(
                        id=f"title_lt_{chap.chapter_id}",
                        style="lower_third",
                        eyebrow=chap_lbl,
                        text=chap.title,
                        subtitle=chap.narrative_goal[:56] if chap.narrative_goal else chap_lbl,
                        timeline_start=lt_start,
                        duration=lt_dur,
                        fade_in=0.30,
                        fade_out=0.35,
                        accent_color="#E8E4DC",
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
                    raw_t = max(raw_t, open_st + open_dur + 0.4)
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
                        eyebrow=eyebrow,
                        text=headline,
                        subtitle=body,
                        timeline_start=ed_st,
                        duration=ed_dur,
                        fade_in=0.35,
                        fade_out=0.35,
                        accent_color="#E8E4DC",
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
                if len(v_title) >= 4:
                    lt_st, lt_dur = _clamp_to_v1_segment(timeline, vc.timeline_start + 0.35, min(3.8, max(2.2, vc.duration - 0.6)))
                    consolidated_overlays.append(
                        TitleOverlayItem(
                            id=f"title_vid_{vc.id}",
                            style="lower_third",
                            eyebrow=_strip_chapter_prefix(vc.act or "GALLERY MOMENT")[:26],
                            text=v_title,
                            subtitle=v_sub or None,
                            timeline_start=lt_st,
                            duration=lt_dur,
                            fade_in=0.30,
                            fade_out=0.35,
                            accent_color="#E8E4DC",
                            hdr_safe=is_hdr,
                        )
                    )

        # 3. Dialogue quote badge (upper-left [0.06, 0.09, 0.58, 0.24] -> zero collision with bottom subtitles!)
        if timeline.subtitles.items:
            best_sub = max(timeline.subtitles.items, key=lambda s: len(s.text))
            if len(best_sub.text.strip()) >= 6 and best_sub.start > hook_end + 4.2:
                qb_st, qb_dur = _clamp_to_v1_segment(
                    timeline,
                    round(best_sub.start + 0.15, 2),
                    round(min(3.2, max(1.8, best_sub.end - best_sub.start)), 2),
                )
                consolidated_overlays.append(
                    TitleOverlayItem(
                        id="title_quote_badge_01",
                        style="dialogue_badge",
                        eyebrow="CANDID SPOKEN MOMENT",
                        text=f'"{best_sub.text[:36]}"',
                        subtitle=best_sub.secondary_text or "Family Dialogue",
                        timeline_start=qb_st,
                        duration=qb_dur,
                        fade_in=0.30,
                        fade_out=0.30,
                        accent_color="#E8E4DC",
                        hdr_safe=is_hdr,
                    )
                )

        # 4. Optional EXIF archival date/location stamp (upper-right) when GPS/timestamp metadata is available
        prompt_low = (brief.raw_user_prompt if brief else blackboard.user_prompt).lower()
        if first_exif_stamp_meta is not None and (gps_assets > 0 or "exif" in prompt_low or "stamp" in prompt_low):
            stamp_date, stamp_loc = first_exif_stamp_meta
            raw_stamp_start = round(min(max(hook_end + 4.4, 4.4), max(0.5, timeline.total_duration * 0.25)), 2)
            stamp_start, stamp_dur = _clamp_to_v1_segment(timeline, raw_stamp_start, 2.6)
            if stamp_start + stamp_dur < timeline.total_duration - 3.6:
                consolidated_overlays.append(
                    TitleOverlayItem(
                        id="title_exif_stamp_01",
                        style="exif_stamp",
                        eyebrow="LOCATION ARCHIVE",
                        text=stamp_date,
                        subtitle=stamp_loc,
                        timeline_start=stamp_start,
                        duration=stamp_dur,
                        fade_in=0.30,
                        fade_out=0.35,
                        accent_color="#E8E4DC",
                        hdr_safe=is_hdr,
                    )
                )

        # 5. Closing card at the tail of the film (moved out of ShowrunnerAgent into Stage 40 title_designer)
        if timeline.total_duration >= 8.0:
            raw_closing_dur = 3.2
            raw_closing_start = round(max(open_st + open_dur + 0.4, timeline.total_duration - raw_closing_dur - 0.2), 2)
            closing_start, closing_dur = _clamp_to_v1_segment(timeline, raw_closing_start, raw_closing_dur)
            # Drop any non-opening bottom/center overlay that overlaps the closing card window
            consolidated_overlays = [
                t
                for t in consolidated_overlays
                if t.id == "title_opening"
                or compute_overlay_bbox_norm(t.style)[1] < 0.30
                or (t.timeline_start + t.duration) <= closing_start - 0.35
            ]
            consolidated_overlays.append(
                TitleOverlayItem(
                    id="title_closing",
                    style="closing_card",
                    eyebrow="MEMORIES PRESERVED",
                    text="Fin · Memories Preserved",
                    subtitle=clean_proj_title,
                    timeline_start=closing_start,
                    duration=closing_dur,
                    fade_in=0.55,
                    fade_out=0.65,
                    accent_color="#E8E4DC",
                    hdr_safe=is_hdr,
                )
            )

        # De-conflict sequential overlays in the same spatial zone (upper_left / upper_right / bottom_center)
        def _zone_for(style_name: str) -> str:
            x1, y1, x2, y2 = compute_overlay_bbox_norm(style_name)
            if y2 < 0.32 and x2 < 0.65:
                return "upper_left"
            if y2 < 0.32 and x1 >= 0.55:
                return "upper_right"
            return "bottom_center"

        consolidated_overlays.sort(key=lambda t: t.timeline_start)
        deduped_overlays: list[TitleOverlayItem] = []
        last_by_zone: dict[str, int] = {}
        for ov in consolidated_overlays:
            ov.hdr_safe = is_hdr
            z = _zone_for(ov.style)
            if z in last_by_zone:
                prev_idx = last_by_zone[z]
                prev = deduped_overlays[prev_idx]
                if ov.timeline_start < (prev.timeline_start + prev.duration + 0.45):
                    if ov.style in ("opening_card", "closing_card", "dialogue_badge"):
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
