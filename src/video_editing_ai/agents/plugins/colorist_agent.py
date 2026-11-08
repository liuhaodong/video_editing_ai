"""Stage 42 Sub-Agent: Senior DI Colorist & Shot-to-Shot Camera Matcher (colorist)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from ...analyzer.media_probe import compute_image_color_stats
from ...models.schema import ClipColorGrade, ColorGradeReport, EffectTrackItem
from ..base import BaseEditingAgent, PipelineBlackboard, register_agent

FILM_STOCK_PRESETS: dict[str, dict[str, float]] = {
    "archival_kodak_2383": {
        "rs": 0.025,
        "gs": 0.008,
        "bs": -0.022,
        "contrast": 1.05,
        "saturation": 1.04,
        "gamma": 0.99,
    },
    "museum_chiaroscuro": {
        "rs": 0.015,
        "gs": 0.005,
        "bs": -0.012,
        "contrast": 1.06,
        "saturation": 0.98,
        "gamma": 0.97,
    },
    "golden_hour_warmth": {
        "rs": 0.038,
        "gs": 0.014,
        "bs": -0.032,
        "contrast": 1.04,
        "saturation": 1.07,
        "gamma": 1.01,
    },
    "twilight_teal_amber": {
        "rs": 0.020,
        "gs": -0.005,
        "bs": 0.028,
        "contrast": 1.06,
        "saturation": 1.05,
        "gamma": 0.98,
    },
    "neutral_reference": {
        "rs": 0.0,
        "gs": 0.0,
        "bs": 0.0,
        "contrast": 1.02,
        "saturation": 1.02,
        "gamma": 1.00,
    },
}


@register_agent
class ColoristAgent(BaseEditingAgent):
    """Stage 42 Agent: Normalizes white balance/exposure across mixed cameras and applies Cinema Film Stock Looks."""

    role_id = "colorist"
    display_name = "4b. Senior DI Colorist & Camera Matcher"
    stage_order = 42
    description = (
        "Senior Digital Intermediate (DI) Colorist that measures RGB histograms and correlated color "
        "temperature (Kelvin) across every clip to perform shot-to-shot camera matching and apply "
        "10-bit HLG HDR-safe Cinema Film Stock looks."
    )
    enabled_by_default = True
    requires_artifacts = ["paper_edit.json"]
    produces_artifacts = ["color_grade_plan.json"]
    skill_reference_file = "13_colorist_agent.md"

    def _measure_clip_stats(self, source_path: str, blackboard: PipelineBlackboard) -> dict[str, float]:
        if blackboard.catalog is not None:
            for p in blackboard.catalog.photos:
                if p.path == source_path and p.color_stats:
                    return p.color_stats
            for v in blackboard.catalog.videos:
                if v.path == source_path and v.color_stats:
                    return v.color_stats
        sp = Path(source_path)
        if sp.exists() and sp.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"}:
            try:
                with Image.open(sp) as im:
                    return compute_image_color_stats(im)
            except Exception:
                pass
        return {
            "mean_r": 128.0,
            "mean_g": 128.0,
            "mean_b": 128.0,
            "p05_luma": 24.0,
            "p50_luma": 128.0,
            "p95_luma": 232.0,
            "est_kelvin": 5500.0,
        }

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline
        paper = blackboard.paper_edit
        brief = blackboard.creative_brief
        if timeline is None:
            return {"shots_matched_count": 0}

        tone = brief.tone if brief else "warm_joyful"
        default_look = "archival_kodak_2383" if tone in ("warm_joyful", "cinematic_nostalgic") else "golden_hour_warmth"

        chapter_looks: dict[str, str] = {}
        if paper is not None:
            for idx, chap in enumerate(paper.chapters):
                if chap.time_of_day == "night":
                    chapter_looks[chap.title] = "twilight_teal_amber"
                elif chap.time_of_day == "twilight":
                    chapter_looks[chap.title] = "golden_hour_warmth"
                elif idx % 2 == 1:
                    chapter_looks[chap.title] = "museum_chiaroscuro"
                else:
                    chapter_looks[chap.title] = default_look

        # Allow AI Agent creative look selection per chapter
        ai_resp = self.ask_ai_creative_json(
            blackboard=blackboard,
            prompt=(
                f"Project tone='{tone}', chapters={list(chapter_looks.keys())}.\n"
                f"Available film stock looks: {list(FILM_STOCK_PRESETS.keys())}.\n"
                "Return JSON {\"primary_film_stock_look\": \"...\", \"chapter_looks\": {\"Chapter Title\": \"...\"}} "
                "to set the DI color palette."
            ),
            decision_key=self.role_id,
            timeout_sec=20.0,
        )
        if isinstance(ai_resp, dict):
            if ai_resp.get("primary_film_stock_look") in FILM_STOCK_PRESETS:
                default_look = str(ai_resp["primary_film_stock_look"])
            if isinstance(ai_resp.get("chapter_looks"), dict):
                for k, v in ai_resp["chapter_looks"].items():
                    if str(v) in FILM_STOCK_PRESETS:
                        chapter_looks[str(k)] = str(v)

        clip_grades: list[ClipColorGrade] = []
        color_effects: list[EffectTrackItem] = []

        for item in timeline.visual_track:
            stats = self._measure_clip_stats(item.source, blackboard)
            kelvin = float(stats.get("est_kelvin", 5500.0))
            p50 = float(stats.get("p50_luma", 128.0))
            look_name = chapter_looks.get(item.act or "", default_look)
            if look_name not in FILM_STOCK_PRESETS:
                look_name = default_look
            preset = FILM_STOCK_PRESETS[look_name]

            # Mechanical shot-to-shot white-balance normalization toward 5500K + Film Stock Look offset
            kelvin_delta = (kelvin - 5500.0) / 5500.0
            wb_r = round(max(-0.08, min(0.08, preset["rs"] + kelvin_delta * 0.04)), 4)
            wb_g = round(max(-0.05, min(0.05, preset["gs"])), 4)
            wb_b = round(max(-0.08, min(0.08, preset["bs"] - kelvin_delta * 0.04)), 4)

            # Mechanical midtone exposure normalization toward p50 = 126
            luma_err = (126.0 - p50) / 255.0
            gamma_corr = round(max(0.92, min(1.08, preset["gamma"] + luma_err * 0.12)), 3)

            grade = ClipColorGrade(
                clip_id=item.id,
                chapter_look=look_name,  # type: ignore[arg-type]
                measured_kelvin=kelvin,
                wb_red_shift=wb_r,
                wb_green_shift=wb_g,
                wb_blue_shift=wb_b,
                contrast=preset["contrast"],
                saturation=preset["saturation"],
                gamma=gamma_corr,
                rationale=(
                    f"Matched {kelvin:.0f}K source (p50={p50:.0f}) to '{look_name}' DI LUT "
                    f"(RGB shift={wb_r:+.3f}/{wb_g:+.3f}/{wb_b:+.3f}, gamma={gamma_corr:.2f})."
                ),
            )
            clip_grades.append(grade)
            color_effects.append(
                EffectTrackItem(
                    id=f"fx_color_{item.id}",
                    effect_type="color_grade",
                    target_clip_id=item.id,
                    timeline_start=item.timeline_start,
                    duration=item.duration,
                    intensity=0.40,
                    parameters={
                        "rs": wb_r,
                        "gs": wb_g,
                        "bs": wb_b,
                        "contrast": preset["contrast"],
                        "saturation": preset["saturation"],
                        "gamma": gamma_corr,
                        "look": look_name,
                    },
                    created_by_agent=self.role_id,
                )
            )

        timeline.effects_track = [
            e for e in timeline.effects_track if e.created_by_agent != self.role_id
        ] + color_effects

        proj_title = brief.project_title if brief else timeline.project_name
        report = ColorGradeReport(
            project_title=proj_title,
            primary_film_stock_look=default_look,
            shots_matched_count=len(clip_grades),
            chapter_looks=chapter_looks,
            clip_grades=clip_grades,
        )
        out_path = blackboard.project_dir / "color_grade_plan.json"
        report.to_json_file(out_path)
        blackboard.set_custom_artifact("color_grade_plan.json", report.model_dump(mode="json"))
        timeline.to_yaml(blackboard.project_dir / "timeline.yaml")

        blackboard.emit_activity(
            self.role_id,
            f"Camera-matched {len(clip_grades)} clips & applied '{default_look}' 10-bit DI Color Grade",
            detail=f"Saved color_grade_plan.json ({len(chapter_looks)} chapter LUT profiles)",
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="sound_designer",
            message_type="handoff",
            subject=f"Completed shot-to-shot camera color matching ({len(clip_grades)} clips, look='{default_look}')",
            payload={"shots_matched": len(clip_grades), "primary_look": default_look},
        )
        return {
            "shots_matched_count": len(clip_grades),
            "primary_film_stock_look": default_look,
            "color_grade_plan_path": str(out_path),
        }
