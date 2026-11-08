"""Stage 38 Sub-Agent: Motion Graphics & GPS Journey Map Animator (motion_graphics)."""

from __future__ import annotations

from typing import Any

from ...models.schema import MotionGraphicItem, MotionGraphicsReport, TitleOverlayItem
from ..base import BaseEditingAgent, PipelineBlackboard, register_agent


@register_agent
class MotionGraphicsAgent(BaseEditingAgent):
    """Stage 38 Agent: Generates sleek GPS route map & chapter progression cards on Track V3."""

    role_id = "motion_graphics"
    display_name = "3c. Motion Graphics & GPS Journey Map Animator"
    stage_order = 38
    description = (
        "National Geographic / Apple TV+ style motion graphics artist that extracts EXIF GPS waypoints "
        "and chapter milestones to render sleek smoked-glass route map progression cards at key journey transitions."
    )
    enabled_by_default = True
    requires_artifacts = ["paper_edit.json"]
    produces_artifacts = ["motion_graphics_plan.json"]
    skill_reference_file = "12_motion_graphics_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline
        paper = blackboard.paper_edit
        catalog = blackboard.catalog
        if timeline is None or paper is None:
            return {"graphics_injected": 0}

        gps_assets = 0
        if catalog is not None:
            gps_assets = sum(1 for p in catalog.photos if p.gps_lat is not None and p.gps_lon is not None)
            gps_assets += sum(1 for v in catalog.videos if v.gps_lat is not None and v.gps_lon is not None)

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

        # Map each chapter's start time on the assembled timeline
        act_start_times: dict[str, float] = {}
        for clip in timeline.visual_track:
            if clip.act and clip.act not in act_start_times:
                act_start_times[clip.act] = clip.timeline_start

        mg_items: list[MotionGraphicItem] = []
        # Create a route progression card at major location transitions (up to 3 across the film)
        target_chapters = paper.chapters[1:4] if len(paper.chapters) > 1 else paper.chapters[:1]
        for c_idx, chap in enumerate(target_chapters, start=1):
            st_sec = act_start_times.get(chap.title, c_idx * (timeline.total_duration / max(2, len(paper.chapters))))
            # Offset slightly after chapter start so it never collides with opening card
            st_sec = min(max(4.5, st_sec + 0.4), max(4.5, timeline.total_duration - 4.0))
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
                    duration=3.2,
                    waypoints=waypoints_for_card,
                    rationale=f"Spatial route progression bridge into '{chap.title}' ({chap.location_tag}).",
                )
            )

        # Delegate creative waypoint naming & selection to AI Agent
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

        # Inject route_map_card overlays onto timeline.title_track without overlapping existing lower_thirds
        timeline.title_track = [t for t in timeline.title_track if t.style != "route_map_card"]
        for mg in mg_items:
            # Remove any duplicate act_banner at the same timestamp and replace with the richer route_map_card
            timeline.title_track.append(
                TitleOverlayItem(
                    id=mg.graphic_id,
                    style="route_map_card",
                    text=mg.title,
                    subtitle=mg.subtitle,
                    timeline_start=mg.timeline_start,
                    duration=mg.duration,
                    fade_in=0.35,
                    fade_out=0.35,
                    accent_color="#E8E4DC",
                    map_waypoints=mg.waypoints,
                )
            )
        timeline.title_track.sort(key=lambda t: t.timeline_start)

        proj_title = blackboard.creative_brief.project_title if blackboard.creative_brief else timeline.project_name
        report = MotionGraphicsReport(
            project_title=proj_title,
            gps_Tagged_assets_found=gps_assets,
            graphics_injected=len(mg_items),
            items=mg_items,
        )
        out_path = blackboard.project_dir / "motion_graphics_plan.json"
        report.to_json_file(out_path)
        blackboard.set_custom_artifact("motion_graphics_plan.json", report.model_dump(mode="json"))
        timeline.to_yaml(blackboard.project_dir / "timeline.yaml")

        blackboard.emit_activity(
            self.role_id,
            f"Designed {len(mg_items)} smoked-glass GPS/Chapter Route Map progression overlays",
            detail=f"Saved motion_graphics_plan.json ({gps_assets} GPS assets inspected)",
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="vfx_designer",
            message_type="handoff",
            subject=f"Injected {len(mg_items)} route map chapter progression overlays onto Track V3",
            payload={"graphics_injected": len(mg_items)},
        )
        return {
            "graphics_injected": len(mg_items),
            "gps_assets_found": gps_assets,
            "motion_graphics_plan_path": str(out_path),
        }
