"""Stage 40 Specialist: Motion Graphics, Lower-Thirds, Bilingual Subtitles & Closing Card Designer (title_designer)."""

from __future__ import annotations

from typing import Any

from ...models.schema import (
    GraphicsManifest,
    GraphicsOverlaySpec,
    MotionGraphicItem,
    MotionGraphicsReport,
    TitleOverlayItem,
)
from ...renderer.subtitles import resolve_subtitle_font_stack
from ...renderer.titles import HDR_DIFFUSE_WHITE_PEAK_RGB, compute_overlay_bbox_norm
from ..base import BaseEditingAgent, PipelineBlackboard, register_agent


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
            for p in catalog.photos:
                if p.gps_lat is not None and p.gps_lon is not None:
                    gps_assets += 1
                if first_exif_stamp_meta is None and p.creation_time:
                    loc_str = f"{p.gps_lat:.2f}N, {p.gps_lon:.2f}E" if (p.gps_lat and p.gps_lon) else (p.camera_model or "")
                    first_exif_stamp_meta = (p.creation_time.strftime("%Y.%m.%d"), loc_str)
            for v in catalog.videos:
                if v.gps_lat is not None and v.gps_lon is not None:
                    gps_assets += 1
                if first_exif_stamp_meta is None and v.creation_time:
                    loc_str = f"{v.gps_lat:.2f}N, {v.gps_lon:.2f}E" if (v.gps_lat and v.gps_lon) else (v.camera_model or "")
                    first_exif_stamp_meta = (v.creation_time.strftime("%Y.%m.%d"), loc_str)

        # Build chapter route waypoints across the story arc
        chapter_stops: list[dict[str, Any]] = []
        for idx, chap in enumerate(paper.chapters):
            clean_lbl = chap.location_tag or chap.title.split(":")[-1].strip()
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

        # 1. Opening card right after any cold-open hook
        hook_end = 0.0
        if timeline.visual_track and timeline.visual_track[0].id == "clip_000_hook":
            hook_end = timeline.visual_track[0].duration

        clean_proj_title = paper.project_title.replace("_", " ").strip() or "Featured Story"
        if len(clean_proj_title) > 42:
            clean_proj_title = " ".join(clean_proj_title.split()[:5]).title()
        else:
            clean_proj_title = clean_proj_title.title()

        consolidated_overlays: list[TitleOverlayItem] = [
            TitleOverlayItem(
                id="title_opening",
                style="opening_card",
                text=clean_proj_title,
                subtitle="A Curated Documentary Film",
                timeline_start=round(hook_end + 0.35, 2),
                duration=3.4,
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
                    subtitle=f"Journey Waypoint {min(len(chapter_stops), c_idx + 1)} of {len(chapter_stops)} · {chap.location_tag}",
                    timeline_start=round(st_sec, 2),
                    duration=3.0,
                    waypoints=waypoints_for_card,
                    rationale=f"Spatial route progression bridge into '{chap.title}' ({chap.location_tag}).",
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
            # Route map cards sit at y=[0.80..0.96], so avoid overlapping active subtitles at y=[0.85..0.96]
            if style_name == "route_map_card":
                for sub in timeline.subtitles.items:
                    if sub.start < end_t and sub.end > start_t:
                        return True
            return False

        for mg in mg_items:
            st_t = mg.timeline_start
            # Nudge route card slightly if an active subtitle sits right on st_t
            if _has_bottom_collision(st_t, mg.duration, "route_map_card"):
                # Convert to upper-safe lower_third above subtitle lane if subtitles are spoken during transition
                has_photo_plaque = any(
                    vc.type == "photo" and vc.caption_title and vc.timeline_start < (st_t + mg.duration) and (vc.timeline_start + vc.duration) > st_t
                    for vc in timeline.visual_track
                )
                if not has_photo_plaque:
                    consolidated_overlays.append(
                        TitleOverlayItem(
                            id=mg.graphic_id,
                            style="lower_third",
                            text=mg.title,
                            subtitle=mg.subtitle,
                            timeline_start=st_t,
                            duration=mg.duration,
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
                    duration=mg.duration,
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
            lt_start = round(st_sec + 0.45, 2)
            if not _has_bottom_collision(lt_start, 2.8, "lower_third"):
                consolidated_overlays.append(
                    TitleOverlayItem(
                        id=f"title_lt_{chap.chapter_id}",
                        style="lower_third",
                        text=chap.title,
                        subtitle=chap.location_tag or chap.narrative_goal[:48],
                        timeline_start=lt_start,
                        duration=2.8,
                        fade_in=0.30,
                        fade_out=0.35,
                        accent_color="#E8E4DC",
                        hdr_safe=is_hdr,
                    )
                )

        # 3. Dialogue quote badge (upper-left [0.06, 0.10, 0.56, 0.24] -> zero collision with bottom subtitles!)
        if timeline.subtitles.items:
            best_sub = max(timeline.subtitles.items, key=lambda s: len(s.text))
            if len(best_sub.text.strip()) >= 6 and best_sub.start > hook_end + 4.2:
                consolidated_overlays.append(
                    TitleOverlayItem(
                        id="title_quote_badge_01",
                        style="dialogue_badge",
                        text=f'"{best_sub.text[:36]}"',
                        subtitle=best_sub.secondary_text or "Candid Spoken Moment",
                        timeline_start=round(best_sub.start + 0.15, 2),
                        duration=round(min(3.0, max(1.8, best_sub.end - best_sub.start)), 2),
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
            stamp_start = round(min(max(hook_end + 4.4, 4.4), max(0.5, timeline.total_duration * 0.25)), 2)
            if stamp_start + 2.6 < timeline.total_duration - 4.0:
                consolidated_overlays.append(
                    TitleOverlayItem(
                        id="title_exif_stamp_01",
                        style="exif_stamp",
                        text=stamp_date,
                        subtitle=stamp_loc,
                        timeline_start=stamp_start,
                        duration=2.6,
                        fade_in=0.30,
                        fade_out=0.35,
                        accent_color="#E8E4DC",
                        hdr_safe=is_hdr,
                    )
                )

        # 5. Closing card at the tail of the film (moved out of ShowrunnerAgent into Stage 40 title_designer)
        if timeline.total_duration >= 8.0:
            closing_dur = 3.2
            closing_start = round(max(hook_end + 4.2, timeline.total_duration - closing_dur - 0.2), 2)
            # Drop any non-opening overlay that overlaps the closing card window
            consolidated_overlays = [
                t
                for t in consolidated_overlays
                if t.id == "title_opening" or (t.timeline_start + t.duration) <= closing_start - 0.4
            ]
            consolidated_overlays.append(
                TitleOverlayItem(
                    id="title_closing",
                    style="closing_card",
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

        # De-conflict sequential overlays in the same screen region
        consolidated_overlays.sort(key=lambda t: t.timeline_start)
        deduped_overlays: list[TitleOverlayItem] = []
        for ov in consolidated_overlays:
            ov.hdr_safe = is_hdr
            if deduped_overlays:
                prev = deduped_overlays[-1]
                same_zone = (
                    compute_overlay_bbox_norm(prev.style)[1] >= 0.60
                    and compute_overlay_bbox_norm(ov.style)[1] >= 0.60
                )
                if same_zone and ov.timeline_start < (prev.timeline_start + prev.duration + 0.6):
                    if ov.style in ("opening_card", "closing_card"):
                        deduped_overlays[-1] = ov
                    continue
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
                subtitle=t.subtitle,
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
