"""Stage 40 Specialist: HDR Colorist & Per-Clip Color Pipeline Classifier (hdr_colorist)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ...analyzer.media_probe import compute_image_color_stats, probe_video_file
from ...models.schema import (
    ClipColorGrade,
    ClipColorTransform,
    ColorGradeReport,
    ColorTransformManifest,
    CulledSubClip,
    EffectTrackItem,
    VisualTrackItem,
)
from ...renderer.ken_burns import build_hdr_color_filters, has_ffmpeg_filter
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


def classify_clip_color_pipeline(
    item: VisualTrackItem,
    subclip_meta: Optional[CulledSubClip] = None,
    hdr_target: bool = True,
    diffuse_white_nits: float = 203.0,
    use_zscale: Optional[bool] = None,
) -> ClipColorTransform:
    """Classify a timeline clip's input color tags and build its FFmpeg SDR/Log/HLG filter chain."""
    if item.type == "photo":
        src_prim = "bt709"
        src_trc = "iec61966-2-1"
        src_space = "bt709"
        pix_fmt = "rgb24"
    elif subclip_meta is not None:
        src_prim = subclip_meta.color_primaries or "bt709"
        src_trc = subclip_meta.color_trc or "bt709"
        src_space = subclip_meta.color_space or "bt709"
        pix_fmt = subclip_meta.pix_fmt or "yuv420p"
    else:
        info = probe_video_file(Path(item.source)) if Path(item.source).exists() else {}
        src_prim = str(info.get("color_primaries") or "bt709")
        src_trc = str(info.get("color_transfer") or info.get("color_trc") or "bt709")
        src_space = str(info.get("color_space") or "bt709")
        pix_fmt = str(info.get("pix_fmt") or "yuv420p")

    trc_low = src_trc.lower()
    prim_low = src_prim.lower()
    is_log = "log" in trc_low or "slog" in Path(item.source).stem.lower() or "dlog" in Path(item.source).stem.lower()

    if item.type == "photo" or trc_low in ("iec61966-2-1", "srgb"):
        input_class = "srgb_still"
    elif is_log:
        input_class = "camera_log"
    elif trc_low == "arib-std-b67" and ("2020" in prim_low or "10" in pix_fmt):
        input_class = "bt2020_hlg_10bit"
    elif trc_low in ("smpte2084", "pq"):
        input_class = "bt2020_pq_10bit"
    else:
        input_class = "sdr_bt709_8bit"

    zscale_ok = has_ffmpeg_filter("zscale") if use_zscale is None else bool(use_zscale)

    if not hdr_target:
        ff_filters = ["format=yuv420p"]
        transform_mode = "sdr_bt709"
    else:
        ff_filters = build_hdr_color_filters(
            src_trc=src_trc,
            src_primaries=src_prim,
            src_space=src_space,
            is_log=is_log,
            diffuse_white_nits=diffuse_white_nits,
            use_zscale=zscale_ok,
        )
        if input_class == "bt2020_hlg_10bit" and not is_log:
            transform_mode = "passthrough_hlg"
        elif is_log:
            transform_mode = "log_to_hlg_203nits"
        elif zscale_ok:
            transform_mode = "zscale_sdr_to_hlg_203nits"
        else:
            transform_mode = "curve_setparams_fallback_203nits"

    return ClipColorTransform(
        clip_id=item.id,
        source_path=item.source,
        input_classification=input_class,  # type: ignore[arg-type]
        color_primaries=src_prim,
        color_trc=src_trc,
        color_space=src_space,
        pix_fmt=pix_fmt,
        target_diffuse_white_nits=diffuse_white_nits,
        transform_mode=transform_mode,  # type: ignore[arg-type]
        ffmpeg_filters=ff_filters,
        rationale=(
            f"Classified {Path(item.source).name} as '{input_class}' ({src_prim}/{src_trc}/{pix_fmt}) -> "
            f"'{transform_mode}' ({diffuse_white_nits:.0f} nits diffuse white)."
        ),
    )


@register_agent
class HDRColoristAgent(BaseEditingAgent):
    """Stage 40 Agent: Classifies per-clip HDR/Log/SDR color pipelines and matches shot-to-shot camera white balance."""

    role_id = "hdr_colorist"
    legacy_role_aliases = ["colorist"]
    display_name = "4c. Senior HDR DI Colorist & Pipeline Classifier"
    stage_order = 40
    substage_priority = 2
    description = (
        "Inspects per-clip color tags (10-bit BT.2020 HLG vs. Camera Log vs. 8-bit BT.709/sRGB stills) "
        "from media_index.json, maps SDR clips & stills to ~203-nit HLG diffuse white (~75% HLG signal), "
        "and writes color_transform_manifest.json."
    )
    enabled_by_default = True
    requires_artifacts = ["media_index.json", "paper_edit.json", "timeline.yaml"]
    produces_artifacts = ["color_transform_manifest.json", "color_grade_plan.json"]
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
        mi = blackboard.media_index
        if timeline is None:
            return {"shots_matched_count": 0}

        hdr_active = bool(timeline.settings.hdr)
        zscale_avail = has_ffmpeg_filter("zscale")

        sc_by_path: dict[str, CulledSubClip] = {}
        if mi is not None:
            for sc in mi.subclips:
                sc_by_path.setdefault(sc.source_path, sc)

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

        ai_resp = self.ask_ai_creative_json(
            blackboard=blackboard,
            prompt=(
                f"Project tone='{tone}', hdr={hdr_active}, chapters={list(chapter_looks.keys())}.\n"
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
        color_transforms: list[ClipColorTransform] = []
        color_effects: list[EffectTrackItem] = []

        all_visual_items = list(timeline.visual_track) + list(timeline.broll_track)
        for item in all_visual_items:
            ct = classify_clip_color_pipeline(
                item=item,
                subclip_meta=sc_by_path.get(item.source),
                hdr_target=hdr_active,
                diffuse_white_nits=203.0,
                use_zscale=zscale_avail,
            )
            color_transforms.append(ct)

            if item.track_lane != 1:
                continue

            stats = self._measure_clip_stats(item.source, blackboard)
            kelvin = float(stats.get("est_kelvin", 5500.0))
            p50 = float(stats.get("p50_luma", 128.0))
            look_name = chapter_looks.get(item.act or "", default_look)
            if look_name not in FILM_STOCK_PRESETS:
                look_name = default_look
            preset = FILM_STOCK_PRESETS[look_name]

            kelvin_delta = (kelvin - 5500.0) / 5500.0
            wb_r = round(max(-0.08, min(0.08, preset["rs"] + kelvin_delta * 0.04)), 4)
            wb_g = round(max(-0.05, min(0.05, preset["gs"])), 4)
            wb_b = round(max(-0.08, min(0.08, preset["bs"] - kelvin_delta * 0.04)), 4)
            luma_err = (126.0 - p50) / 255.0
            gamma_corr = round(max(0.92, min(1.08, preset["gamma"] + luma_err * 0.12)), 3)

            clip_grades.append(
                ClipColorGrade(
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
                        f"{ct.input_classification} -> {ct.transform_mode}; matched {kelvin:.0f}K source "
                        f"to '{look_name}' DI look."
                    ),
                )
            )
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
                        "src_trc": ct.color_trc,
                        "src_primaries": ct.color_primaries,
                        "src_space": ct.color_space,
                        "is_log": ct.input_classification == "camera_log",
                        "diffuse_white_nits": 203.0,
                    },
                    created_by_agent=self.role_id,
                )
            )

        timeline.effects_track = [
            e for e in timeline.effects_track if e.created_by_agent not in (self.role_id, "colorist")
        ] + color_effects

        proj_title = brief.project_title if brief else timeline.project_name
        transform_manifest = ColorTransformManifest(
            project_title=proj_title,
            hdr_delivery=hdr_active,
            target_transfer="arib-std-b67" if hdr_active else "bt709",
            target_primaries="bt2020" if hdr_active else "bt709",
            diffuse_white_nits=203.0,
            zscale_available=zscale_avail,
            primary_film_stock_look=default_look,
            clips=color_transforms,
        )
        manifest_path = blackboard.project_dir / "color_transform_manifest.json"
        transform_manifest.to_json_file(manifest_path)
        blackboard.color_transform_manifest = transform_manifest

        grade_report = ColorGradeReport(
            project_title=proj_title,
            primary_film_stock_look=default_look,
            shots_matched_count=len(clip_grades),
            chapter_looks=chapter_looks,
            clip_grades=clip_grades,
        )
        grade_path = blackboard.project_dir / "color_grade_plan.json"
        grade_report.to_json_file(grade_path)
        blackboard.set_custom_artifact("color_grade_plan.json", grade_report.model_dump(mode="json"))
        timeline.to_yaml(blackboard.project_dir / "timeline.yaml")

        blackboard.emit_activity(
            self.role_id,
            f"Classified {len(color_transforms)} HDR/SDR/sRGB color pipelines & matched {len(clip_grades)} V1 shots ('{default_look}')",
            detail=f"203-nit diffuse white mapping via {'zscale' if zscale_avail else 'calibrated curve+setparams'}",
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="sound_mixer",
            message_type="handoff",
            subject=f"Completed per-clip HDR color classification ({len(color_transforms)} clips)",
            payload={"clips_classified": len(color_transforms), "hdr_delivery": hdr_active},
        )
        return {
            "shots_matched_count": len(clip_grades),
            "clips_classified_count": len(color_transforms),
            "primary_film_stock_look": default_look,
            "color_transform_manifest_path": str(manifest_path),
            "color_grade_plan_path": str(grade_path),
        }


# Backward-compatible class alias
ColoristAgent = HDRColoristAgent
