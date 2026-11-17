"""Stage 42 Senior DI Cinema Color Grading Specialist Sub-Agent (color_grader)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import median
from typing import Any, Optional

from ...models.schema import (
    VALID_FILM_STOCK_LOOKS,
    ClipColorGrade,
    ColorGradeRecommendation,
    ColorGradeReport,
    EditTimeline,
    EffectTrackItem,
)
from ...renderer.color_grading import (
    CINEMA_FILM_STOCK_SPECS,
    CREATIVE_FILM_STOCK_ORDER,
    ensure_project_luts,
    is_synthetic_graphic_clip,
    measure_media_colorimetry,
    render_color_grading_scopes_sheet,
    render_color_lookbook_sheet,
)
from ..base import BaseEditingAgent, PipelineBlackboard, register_agent

logger = logging.getLogger(__name__)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _infer_primary_film_stock(blackboard: PipelineBlackboard) -> str:
    """Select an aesthetic film-print spine from CreativeBrief and user intent."""
    text = " ".join(
        [
            str(blackboard.user_prompt or ""),
            str(getattr(blackboard.creative_brief, "visual_style", "") or ""),
            str(getattr(blackboard.creative_brief, "color_palette_direction", "") or ""),
            str(getattr(blackboard.creative_brief, "overall_mood", "") or ""),
        ]
    ).lower()

    if any(k in text for k in ("fuji", "eterna", "pastel", "soft")):
        return "fuji_eterna_soft"
    if any(k in text for k in ("arri", "natural", "neutral", "clean documentary")):
        return "arri_natural_cinema"
    if any(k in text for k in ("coastal", "ocean", "beach", "ektachrome", "pacific")):
        return "pacific_coastal_chrome"
    if any(k in text for k in ("golden", "sunset", "nostalg", "warm")):
        return "golden_hour_warmth"
    if any(k in text for k in ("museum", "gallery", "chiaroscuro", "art")):
        return "museum_chiaroscuro"
    if any(k in text for k in ("twilight", "dusk", "neon", "teal")):
        return "twilight_teal_amber"
    if any(k in text for k in ("a24", "matte", "indie", "film grain")):
        return "a24_indie_matte"
    return "archival_kodak_2383"


def _resolve_clip_chapter_look(
    clip_note: str,
    source_path: str,
    telemetry: dict[str, Any],
    primary_look: str,
    strategy: str,
    chapter_looks: dict[str, str],
    clip_act: str = "",
) -> tuple[str, str]:
    """Resolve clip look and chapter key under 'chapter_adaptive' | 'uniform_stock' | 'natural_neutral'."""
    if strategy == "natural_neutral":
        return "neutral_reference", "Neutral"
    if strategy == "uniform_stock":
        return primary_look, "Master Spine"

    if clip_act:
        for ch_key, ch_look in chapter_looks.items():
            if ch_key and ch_key.lower() == clip_act.lower() and ch_look in VALID_FILM_STOCK_LOOKS:
                return ch_look, clip_act

    combined = f"{clip_act} {clip_note} {Path(source_path).stem}".lower()
    for ch_key, ch_look in chapter_looks.items():
        if ch_key and ch_key.lower() in combined and ch_look in VALID_FILM_STOCK_LOOKS:
            return ch_look, ch_key

    p50 = float(telemetry.get("p50_luma", 0.45))
    kelvin = float(telemetry.get("est_kelvin", 5500.0))

    if any(w in combined for w in ("museum", "gallery", "exhibit", "hall", "indoor", "library", "science center")):
        return "museum_chiaroscuro", "Indoor / Gallery"
    if any(w in combined for w in ("sunset", "dusk", "evening", "twilight", "night")):
        return "twilight_teal_amber", "Dusk / Evening"
    if any(w in combined for w in ("golden", "late afternoon", "amber", "cactus", "desert")):
        return "golden_hour_warmth", "Golden Hour"
    if any(w in combined for w in ("beach", "coast", "ocean", "bay", "harbor", "sea", "whale", "pacific")):
        return "pacific_coastal_chrome", "Coastal Daylight"
    if p50 < 0.30 and kelvin < 4300.0:
        return "museum_chiaroscuro", "Low-Light Interior"
    if p50 < 0.34 and kelvin > 6400.0:
        return "twilight_teal_amber", "Blue-Hour Dusk"
    return primary_look, clip_act or "Master Spine"


def _measure_timeline_clips(timeline: EditTimeline) -> tuple[list[tuple[int, Any, dict[str, Any]]], int]:
    """Measure non-synthetic V1 and V2 visual clips once via multi-frame Planckian colorimetry."""
    all_items = list(timeline.visual_track) + list(timeline.broll_track)
    measured: list[tuple[int, Any, dict[str, Any]]] = []
    skipped_synthetic = 0
    for idx, item in enumerate(all_items):
        if is_synthetic_graphic_clip(item.source, item.id, getattr(item, "narrative_note", "")):
            skipped_synthetic += 1
            continue
        tel = measure_media_colorimetry(
            item.source,
            media_type=item.type,
            duration=item.duration,
            start_offset=item.source_in,
        )
        measured.append((idx, item, tel))
    return measured, skipped_synthetic


def _summarize_tel_group(tels: list[dict[str, Any]]) -> dict[str, Any]:
    if not tels:
        return {
            "count": 0,
            "kelvin_median": 5600.0,
            "kelvin_spread": 0.0,
            "p50_luma_median": 0.45,
            "p95_luma_median": 0.88,
            "skin_shot_ratio": 0.0,
            "low_light_ratio": 0.0,
            "bright_daylight_ratio": 0.0,
            "warm_golden_ratio": 0.0,
            "blue_hour_ratio": 0.0,
        }
    kelvins = [float(t.get("est_kelvin", 5500.0)) for t in tels]
    p50s = [float(t.get("p50_luma", 0.45)) for t in tels]
    p95s = [float(t.get("p95_luma", 0.88)) for t in tels]
    n = float(len(tels))
    return {
        "count": len(tels),
        "kelvin_median": round(float(median(kelvins)), 1),
        "kelvin_min": round(min(kelvins), 1),
        "kelvin_max": round(max(kelvins), 1),
        "kelvin_spread": round(max(kelvins) - min(kelvins), 1) if len(kelvins) >= 2 else 0.0,
        "p50_luma_median": round(float(median(p50s)), 3),
        "p95_luma_median": round(float(median(p95s)), 3),
        "skin_shot_ratio": round(sum(1 for t in tels if float(t.get("skin_pixel_ratio", 0.0)) >= 0.04) / n, 3),
        "low_light_ratio": round(sum(1 for t in tels if float(t.get("p50_luma", 0.45)) < 0.32) / n, 3),
        "bright_daylight_ratio": round(
            sum(1 for t in tels if float(t.get("p50_luma", 0.45)) >= 0.50 and float(t.get("est_kelvin", 5500.0)) >= 5300.0) / n,
            3,
        ),
        "warm_golden_ratio": round(
            sum(1 for t in tels if float(t.get("est_kelvin", 5500.0)) < 4900.0 and float(t.get("p50_luma", 0.45)) >= 0.32) / n,
            3,
        ),
        "blue_hour_ratio": round(
            sum(1 for t in tels if float(t.get("est_kelvin", 5500.0)) >= 6500.0 and float(t.get("p50_luma", 0.45)) < 0.38) / n,
            3,
        ),
    }


def _build_photometry_digest(
    timeline: EditTimeline,
    measured: list[tuple[int, Any, dict[str, Any]]],
) -> dict[str, Any]:
    """Summarize global and per-chapter photometry & skin presence across measured timeline clips."""
    global_tels = [t for _, _, t in measured]
    digest = _summarize_tel_group(global_tels)
    digest["video_count"] = sum(1 for _, it, _ in measured if it.type != "photo")
    digest["photo_count"] = sum(1 for _, it, _ in measured if it.type == "photo")

    by_chapter: dict[str, list[dict[str, Any]]] = {}
    for _, item, tel in measured:
        act_label = str(getattr(item, "act", "") or "Main").strip() or "Main"
        by_chapter.setdefault(act_label, []).append(tel)
    digest["chapters"] = {ch: _summarize_tel_group(tels) for ch, tels in by_chapter.items()}
    return digest


def _score_film_stocks(
    digest: dict[str, Any],
    brief_text: str,
    preferred_hint: str = "",
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Score all 8 curated film stocks and infer per-chapter adaptations from photometry & brief context."""
    text = (brief_text or "").lower()
    skin_r = float(digest.get("skin_shot_ratio", 0.0))
    low_r = float(digest.get("low_light_ratio", 0.0))
    day_r = float(digest.get("bright_daylight_ratio", 0.0))
    warm_r = float(digest.get("warm_golden_ratio", 0.0))
    blue_r = float(digest.get("blue_hour_ratio", 0.0))
    k_spread = float(digest.get("kelvin_spread", 0.0))
    k_med = float(digest.get("kelvin_median", 5600.0))

    scores: dict[str, float] = {s: 50.0 for s in CREATIVE_FILM_STOCK_ORDER}
    reasons: dict[str, list[str]] = {s: [] for s in CREATIVE_FILM_STOCK_ORDER}

    # Global versatility + skin-tone & family documentary synergy
    if skin_r >= 0.15:
        scores["archival_kodak_2383"] += 16.0
        scores["fuji_eterna_soft"] += 18.0
        reasons["archival_kodak_2383"].append(f"protects natural warmth across {int(round(skin_r*100))}% people shots")
        reasons["fuji_eterna_soft"].append(f"flattering pastel roll-off on {int(round(skin_r*100))}% portrait clips")
    if k_spread >= 1500.0:
        scores["archival_kodak_2383"] += 12.0
        scores["arri_natural_cinema"] += 11.0
        reasons["archival_kodak_2383"].append(f"unifies wide {int(round(k_spread))}K mixed-lighting spread")
    if any(k in text for k in ("family", "holiday", "road trip", "keepsake", "documentary", "memories")):
        scores["archival_kodak_2383"] += 14.0
        scores["fuji_eterna_soft"] += 8.0
        reasons["archival_kodak_2383"].append("classic warm family keepsake print spine")
    if any(k in text for k in ("coast", "ocean", "beach", "whale", "dolphin", "bay", "pacific")) or day_r >= 0.35:
        scores["pacific_coastal_chrome"] += 15.0 + (8.0 if day_r >= 0.35 else 0.0)
        reasons["pacific_coastal_chrome"].append("crisp marine blues & open daylight skies")
    if any(k in text for k in ("sunset", "golden", "desert", "cactus", "warm", "christmas")) or warm_r >= 0.25:
        scores["golden_hour_warmth"] += 14.0 + (7.0 if k_med < 5300 else 0.0)
        reasons["golden_hour_warmth"].append("amber highlight bloom for golden late-afternoon scenery")
    if any(k in text for k in ("museum", "gallery", "science", "indoor", "exhibit", "art")) or low_r >= 0.25:
        scores["museum_chiaroscuro"] += 14.0 + (6.0 if low_r >= 0.25 else 0.0)
        reasons["museum_chiaroscuro"].append("sculpted indoor contrast with neutral architectural whites")
    if any(k in text for k in ("twilight", "dusk", "night", "evening", "lights")) or blue_r >= 0.20:
        scores["twilight_teal_amber"] += 13.0 + (7.0 if blue_r >= 0.20 else 0.0)
        reasons["twilight_teal_amber"].append("cinematic dusk shadow separation with skin guard")
    if any(k in text for k in ("indie", "a24", "matte", "minimal", "analog")):
        scores["a24_indie_matte"] += 16.0
        reasons["a24_indie_matte"].append("editorial lifted shadow toe")
    if any(k in text for k in ("neutral", "arri", "clean", "technical", "natural")):
        scores["arri_natural_cinema"] += 14.0
        reasons["arri_natural_cinema"].append("transparent 10-bit dynamic range")

    if preferred_hint in scores:
        scores[preferred_hint] += 10.0

    ranked: list[dict[str, Any]] = []
    for stock_id in sorted(scores.keys(), key=lambda s: (-scores[s], CREATIVE_FILM_STOCK_ORDER.index(s))):
        spec = CINEMA_FILM_STOCK_SPECS[stock_id]
        r_list = reasons[stock_id] or [spec["subtitle"]]
        ranked.append(
            {
                "look": stock_id,
                "display_name": spec["display_name"],
                "score": round(scores[stock_id], 1),
                "recommended_intensity": round(float(spec["lut_intensity"]), 2),
                "reasons": r_list,
            }
        )

    primary = ranked[0]["look"] if ranked else "archival_kodak_2383"
    chapter_looks: dict[str, str] = {}
    for ch_name, ch_stats in (digest.get("chapters") or {}).items():
        ch_low = ch_name.lower()
        c_low_r = float(ch_stats.get("low_light_ratio", 0.0))
        c_day_r = float(ch_stats.get("bright_daylight_ratio", 0.0))
        c_blue_r = float(ch_stats.get("blue_hour_ratio", 0.0))
        c_warm_r = float(ch_stats.get("warm_golden_ratio", 0.0))
        c_skin_r = float(ch_stats.get("skin_shot_ratio", 0.0))

        if any(w in ch_low for w in ("museum", "gallery", "science", "indoor", "library", "hall")):
            chapter_looks[ch_name] = "museum_chiaroscuro"
        elif any(w in ch_low for w in ("beach", "coast", "ocean", "bay", "pacific", "whale", "harbor", "sea")):
            chapter_looks[ch_name] = "pacific_coastal_chrome"
        elif any(w in ch_low for w in ("sunset", "golden", "cactus", "desert", "joshua")):
            chapter_looks[ch_name] = "golden_hour_warmth"
        elif any(w in ch_low for w in ("christmas eve", "dusk", "evening", "twilight", "night")):
            chapter_looks[ch_name] = "twilight_teal_amber" if c_blue_r >= 0.15 or c_low_r >= 0.3 else "golden_hour_warmth"
        elif c_blue_r >= 0.35:
            chapter_looks[ch_name] = "twilight_teal_amber"
        elif c_low_r >= 0.45 and c_skin_r < 0.25:
            chapter_looks[ch_name] = "museum_chiaroscuro"
        elif c_warm_r >= 0.45:
            chapter_looks[ch_name] = "golden_hour_warmth"
        elif c_day_r >= 0.55 and c_skin_r < 0.25:
            chapter_looks[ch_name] = "pacific_coastal_chrome"
        else:
            chapter_looks[ch_name] = primary

    return ranked, chapter_looks


def _solve_clip_color_grade(
    clip_id: str,
    media_type: str,
    source: str,
    telemetry: dict[str, Any],
    ref_kelvin: float,
    look_name: str,
    lut_paths: dict[str, str],
    hlg_rolloff: bool = False,
    cst_meta: Optional[dict[str, Any]] = None,
    override: Optional[dict[str, Any]] = None,
) -> tuple[ClipColorGrade, dict[str, Any]]:
    """Compute 3-way Lift/Gamma/Gain primary balance + LUT binding for a single clip."""
    ov = dict(override or {})
    chosen_look = str(ov.get("chapter_look") or ov.get("look") or look_name)
    if chosen_look not in CINEMA_FILM_STOCK_SPECS:
        chosen_look = look_name
    spec = CINEMA_FILM_STOCK_SPECS[chosen_look]

    kelvin = float(telemetry.get("est_kelvin", 5500.0))
    tint = float(telemetry.get("est_tint", 0.0))
    p05 = float(telemetry.get("p05_luma", 0.05))
    p50 = float(telemetry.get("p50_luma", 0.45))
    p95 = float(telemetry.get("p95_luma", 0.90))
    skin_ratio = float(telemetry.get("skin_pixel_ratio", 0.0))

    # Proportional primary WB matching toward neutral daylight/scene median (mild filmic nudge)
    kelvin_diff = ref_kelvin - kelvin
    if abs(kelvin_diff) <= 220.0:
        norm_shift = 0.0
    else:
        adj_diff = kelvin_diff - (220.0 if kelvin_diff > 0 else -220.0)
        norm_shift = _clamp(adj_diff / 30000.0, -0.075, 0.075)
    wb_red = float(ov.get("wb_red_shift", round(-norm_shift * 0.72, 4)))
    wb_blue = float(ov.get("wb_blue_shift", round(norm_shift * 0.78, 4)))
    wb_green = float(ov.get("wb_green_shift", round(_clamp(-tint * 0.35, -0.04, 0.04), 4)))

    # Gentle exposure & black-point normalization targeting ~0.44 mid-gray without crushing night/sky mood
    luma_err = 0.44 - p50
    raw_ev = 0.0 if abs(luma_err) < 0.08 else (luma_err - (0.08 if luma_err > 0 else -0.08)) * 0.65
    exposure_ev = float(ov.get("exposure_ev", round(_clamp(raw_ev, -0.25, 0.25), 3)))

    raw_toe = -(p05 - 0.06) * 0.35 if p05 > 0.09 else (0.012 if p05 < 0.025 else 0.0)
    black_lift = float(ov.get("black_lift", round(_clamp(raw_toe, -0.03, 0.02), 4)))

    lift_rgb = tuple(float(x) for x in ov.get("lift_rgb", spec["lift"]))
    gamma_rgb = tuple(float(x) for x in ov.get("gamma_rgb", spec["gamma"]))
    gain_rgb = tuple(float(x) for x in ov.get("gain_rgb", spec["gain"]))
    contrast = float(ov.get("contrast", spec["contrast"]))
    saturation = float(ov.get("saturation", spec["saturation"]))
    vibrance = float(ov.get("vibrance", spec["vibrance"]))
    gamma_scalar = float(ov.get("gamma", 1.00))

    skin_prot = float(ov.get("skin_protection", spec["skin_protection"]))
    if skin_ratio >= 0.04:
        skin_prot = max(skin_prot, 0.78)
    lut_int = float(ov.get("lut_intensity", spec["lut_intensity"]))
    lut_cube = lut_paths.get(chosen_look, "")

    grade_obj = ClipColorGrade(
        clip_id=clip_id,
        media_type=media_type,
        chapter_look=chosen_look,  # type: ignore[arg-type]
        measured_kelvin=round(kelvin, 1),
        measured_tint=round(tint, 4),
        p05_luma=round(p05, 4),
        p50_luma=round(p50, 4),
        p95_luma=round(p95, 4),
        skin_pixel_ratio=round(skin_ratio, 4),
        wb_red_shift=round(wb_red, 4),
        wb_green_shift=round(wb_green, 4),
        wb_blue_shift=round(wb_blue, 4),
        lift_rgb=(lift_rgb[0], lift_rgb[1], lift_rgb[2]),
        gamma_rgb=(gamma_rgb[0], gamma_rgb[1], gamma_rgb[2]),
        gain_rgb=(gain_rgb[0], gain_rgb[1], gain_rgb[2]),
        exposure_ev=round(exposure_ev, 3),
        black_lift=round(black_lift, 4),
        contrast=round(contrast, 3),
        saturation=round(saturation, 3),
        vibrance=round(vibrance, 3),
        gamma=round(gamma_scalar, 3),
        skin_protection=round(skin_prot, 3),
        lut_intensity=round(lut_int, 3),
        lut_cube_path=str(lut_cube),
        rationale=(
            f"Matched {kelvin:.0f}K ({media_type}) toward {ref_kelvin:.0f}K baseline; "
            f"look={chosen_look} ({int(round(lut_int*100))}% LUT)"
            + (f", skin_prot={int(round(skin_prot*100))}%" if skin_ratio >= 0.03 else "")
        ),
    )

    fx_params: dict[str, Any] = {
        "chapter_look": chosen_look,
        "look": chosen_look,
        "measured_kelvin": grade_obj.measured_kelvin,
        "wb_red_shift": grade_obj.wb_red_shift,
        "wb_green_shift": grade_obj.wb_green_shift,
        "wb_blue_shift": grade_obj.wb_blue_shift,
        "rs": grade_obj.wb_red_shift,
        "gs": grade_obj.wb_green_shift,
        "bs": grade_obj.wb_blue_shift,
        "lift_rgb": list(grade_obj.lift_rgb),
        "gamma_rgb": list(grade_obj.gamma_rgb),
        "gain_rgb": list(grade_obj.gain_rgb),
        "exposure_ev": grade_obj.exposure_ev,
        "black_lift": grade_obj.black_lift,
        "contrast": grade_obj.contrast,
        "saturation": grade_obj.saturation,
        "vibrance": grade_obj.vibrance,
        "gamma": grade_obj.gamma,
        "skin_protection": grade_obj.skin_protection,
        "skin_pixel_ratio": grade_obj.skin_pixel_ratio,
        "lut_intensity": grade_obj.lut_intensity,
        "lut_cube_path": str(lut_cube),
        "hlg_rolloff": bool(hlg_rolloff or p95 >= 0.89),
    }
    for k in ("src_trc", "src_primaries", "src_space", "is_log", "diffuse_white_nits", "hdr"):
        if cst_meta and k in cst_meta:
            fx_params[k] = cst_meta[k]
    return grade_obj, fx_params


def apply_color_grade_overrides(
    timeline: EditTimeline,
    project_dir: str | Path,
    look_name: Optional[str] = None,
    clip_id: Optional[str] = None,
    intensity: Optional[float] = None,
    exposure_ev: Optional[float] = None,
    wb_red: Optional[float] = None,
    wb_blue: Optional[float] = None,
    saturation: Optional[float] = None,
    contrast: Optional[float] = None,
    skin_protection: Optional[float] = None,
    strategy: Optional[str] = None,
) -> ColorGradeReport:
    """Apply surgical / interactive DI color grade overrides to a project's timeline and report."""
    proj = Path(project_dir)
    plan_path = proj / "color_grade_plan.json"
    if plan_path.exists():
        try:
            report = ColorGradeReport.from_json_file(plan_path)
        except Exception:
            report = ColorGradeReport(project_title=timeline.project_name)
    else:
        report = ColorGradeReport(project_title=timeline.project_name)

    chosen_look = look_name if (look_name and look_name in VALID_FILM_STOCK_LOOKS) else report.primary_film_stock_look
    if chosen_look not in VALID_FILM_STOCK_LOOKS:
        chosen_look = "archival_kodak_2383"
    if look_name and not clip_id:
        report.primary_film_stock_look = chosen_look
    if strategy:
        report.grading_strategy = strategy

    lut_paths = ensure_project_luts(proj)
    report.generated_luts = dict(lut_paths)

    existing_by_clip: dict[str, ClipColorGrade] = {g.clip_id: g for g in report.clip_grades}
    updated_grades: list[ClipColorGrade] = []
    target_ids = {clip_id} if clip_id else None
    preserved_fx = [
        fx
        for fx in timeline.effects_track
        if fx.effect_type != "color_grade" or (target_ids is not None and fx.target_clip_id not in target_ids)
    ]

    all_items = list(timeline.visual_track) + list(timeline.broll_track)
    for idx, item in enumerate(all_items):
        if is_synthetic_graphic_clip(item.source, item.id, getattr(item, "narrative_note", "")):
            continue
        prev = existing_by_clip.get(item.id)
        if target_ids is not None and item.id not in target_ids:
            if prev is not None:
                updated_grades.append(prev)
            continue

        tel = (
            {
                "est_kelvin": prev.measured_kelvin,
                "est_tint": prev.measured_tint,
                "p05_luma": prev.p05_luma,
                "p50_luma": prev.p50_luma,
                "p95_luma": prev.p95_luma,
                "skin_pixel_ratio": prev.skin_pixel_ratio,
            }
            if prev is not None
            else measure_media_colorimetry(
                item.source, media_type=item.type, duration=item.duration, start_offset=item.source_in
            )
        )

        ov: dict[str, Any] = {}
        eff_look = chosen_look if (look_name or prev is None) else prev.chapter_look
        ov["chapter_look"] = eff_look
        if intensity is not None:
            ov["lut_intensity"] = _clamp(intensity, 0.0, 1.0)
        elif prev is not None and look_name is None:
            ov["lut_intensity"] = prev.lut_intensity
        if exposure_ev is not None:
            ov["exposure_ev"] = _clamp(exposure_ev, -1.5, 1.5)
        elif prev is not None:
            ov["exposure_ev"] = prev.exposure_ev
        if wb_red is not None:
            ov["wb_red_shift"] = _clamp(wb_red, -0.35, 0.35)
        elif prev is not None:
            ov["wb_red_shift"] = prev.wb_red_shift
        if wb_blue is not None:
            ov["wb_blue_shift"] = _clamp(wb_blue, -0.35, 0.35)
        elif prev is not None:
            ov["wb_blue_shift"] = prev.wb_blue_shift
        if saturation is not None:
            ov["saturation"] = _clamp(saturation, 0.5, 1.6)
        if contrast is not None:
            ov["contrast"] = _clamp(contrast, 0.7, 1.45)
        if skin_protection is not None:
            ov["skin_protection"] = _clamp(skin_protection, 0.0, 1.0)

        prev_cst_by_clip: dict[str, dict[str, Any]] = {}
        for fx in timeline.effects_track:
            if fx.effect_type == "color_grade" and fx.target_clip_id:
                prev_cst_by_clip[fx.target_clip_id] = dict(fx.parameters or {})

        hlg_flag = bool(prev_cst_by_clip.get(item.id, {}).get("hlg_rolloff", False))
        grade_obj, fx_params = _solve_clip_color_grade(
            clip_id=item.id,
            media_type=item.type,
            source=item.source,
            telemetry=tel,
            ref_kelvin=5600.0,
            look_name=eff_look,
            lut_paths=lut_paths,
            hlg_rolloff=hlg_flag,
            cst_meta=prev_cst_by_clip.get(item.id),
            override=ov,
        )
        updated_grades.append(grade_obj)
        preserved_fx.append(
            EffectTrackItem(
                id=f"fx_color_{idx}_{item.id}",
                effect_type="color_grade",
                target_clip_id=item.id,
                timeline_start=item.timeline_start,
                duration=item.duration,
                intensity=grade_obj.lut_intensity,
                parameters=fx_params,
                created_by_agent="color_grader",
            )
        )

    timeline.effects_track = preserved_fx
    report.clip_grades = updated_grades
    report.shots_matched_count = len(updated_grades)
    report.to_json_file(plan_path)
    return report


def recommend_color_grading(
    blackboard: PipelineBlackboard,
    measured: Optional[list[tuple[int, Any, dict[str, Any]]]] = None,
    lookbook_path: str = "",
    agent: Optional["CinemaColorGradingAgent"] = None,
) -> ColorGradeRecommendation:
    """Analyze timeline photometry + visual lookbook and produce an AI-backed ColorGradeRecommendation."""
    timeline = blackboard.timeline
    if timeline is None:
        raise ValueError("timeline.yaml must exist on blackboard before recommending color grading.")

    if measured is None:
        measured, _ = _measure_timeline_clips(timeline)

    digest = _build_photometry_digest(timeline, measured)
    existing_sources = [str(it.source) for _, it, _ in measured if Path(str(it.source)).exists()]
    if not lookbook_path and existing_sources:
        try:
            lb_out = render_color_lookbook_sheet(
                existing_sources[:4],
                blackboard.project_dir / "_color_lookbook_8stocks.jpg",
                project_title=timeline.project_name,
            )
            lookbook_path = str(lb_out)
        except Exception as exc:
            logger.debug("Lookbook render skipped during recommendation: %s", exc)

    brief_text = " ".join(
        [
            str(blackboard.user_prompt or ""),
            str(getattr(blackboard.creative_brief, "visual_style", "") or ""),
            str(getattr(blackboard.creative_brief, "color_palette_direction", "") or ""),
            str(getattr(blackboard.creative_brief, "overall_mood", "") or ""),
            str(getattr(blackboard.creative_brief, "narrative_style", "") or ""),
        ]
    )
    brief_hint = str(getattr(blackboard.creative_brief, "color_grade_look", "") or "")
    ranked, base_chapter_looks = _score_film_stocks(digest, brief_text=brief_text, preferred_hint=brief_hint)

    image_paths: list[str | Path] = []
    if lookbook_path and Path(lookbook_path).exists():
        image_paths.append(lookbook_path)
    for scr_cand in (
        blackboard.project_dir / "cache" / "curation" / "director_screening_sheet.jpg",
        blackboard.project_dir / "director_screening_sheet.jpg",
    ):
        if scr_cand.exists():
            image_paths.append(scr_cand)
            break

    decision_file = blackboard.project_dir / "ai_decisions" / "color_grader.json"
    had_override_file = decision_file.exists()

    stock_menu = {
        k: {
            "display_name": CINEMA_FILM_STOCK_SPECS[k]["display_name"],
            "subtitle": CINEMA_FILM_STOCK_SPECS[k]["subtitle"],
            "default_lut_intensity": CINEMA_FILM_STOCK_SPECS[k]["lut_intensity"],
        }
        for k in CREATIVE_FILM_STOCK_ORDER
    }
    prompt = (
        f"Act as Senior DI Cinema Colorist for '{timeline.project_name}'.\n"
        f"Project narrative context: {brief_text.strip() or timeline.project_name}\n"
        f"Global & chapter photometry telemetry:\n{json.dumps(digest, indent=2)}\n\n"
        f"Available 8 curated 33x33x33 .cube film stocks:\n{json.dumps(stock_menu, indent=2)}\n\n"
        f"Baseline photometric ranking top 3: {[r['look'] for r in ranked[:3]]}\n"
        "Inspect the attached 8-stock DI Color Lookbook sheet (and screening sheet if attached) and return strict JSON:\n"
        "{\n"
        '  "primary_film_stock_look": "<one of VALID_FILM_STOCK_LOOKS>",\n'
        '  "runner_up_look": "<2nd best look>",\n'
        '  "grading_strategy": "chapter_adaptive",\n'
        '  "recommended_lut_intensity": 0.80,\n'
        '  "chapter_looks": {"<chapter_name>": "<film_stock_look>"},\n'
        '  "ai_rationale": "<concise cinema colorist explanation citing lighting/skin/chapter cohesion>"\n'
        "}"
    )

    color_agent = agent or CinemaColorGradingAgent()
    ai_resp = color_agent.ask_ai_creative_json(
        blackboard,
        prompt=prompt,
        image_paths=image_paths or None,
        decision_key="color_grader",
    )

    rec_source = "photometric_ranker"
    if isinstance(ai_resp, dict) and ai_resp:
        rec_source = "ai_override_packet" if had_override_file else "ai_vision"
    ai_doc = dict(ai_resp or {})

    primary_look = str(ai_doc.get("primary_film_stock_look") or ranked[0]["look"])
    if primary_look not in VALID_FILM_STOCK_LOOKS:
        primary_look = ranked[0]["look"]

    default_runner_up = next((r["look"] for r in ranked if r["look"] != primary_look), "fuji_eterna_soft")
    runner_up = str(ai_doc.get("runner_up_look") or default_runner_up)
    if runner_up not in VALID_FILM_STOCK_LOOKS or runner_up == primary_look:
        runner_up = default_runner_up

    strategy = str(ai_doc.get("grading_strategy") or "chapter_adaptive")
    if strategy not in ("chapter_adaptive", "uniform_stock", "natural_neutral"):
        strategy = "chapter_adaptive"

    spec_p = CINEMA_FILM_STOCK_SPECS.get(primary_look, CINEMA_FILM_STOCK_SPECS["archival_kodak_2383"])
    try:
        rec_intensity = _clamp(
            float(ai_doc.get("recommended_lut_intensity") or ai_doc.get("lut_intensity") or spec_p["lut_intensity"]),
            0.0,
            1.0,
        )
    except Exception:
        rec_intensity = float(spec_p["lut_intensity"])

    merged_chapter_looks = dict(base_chapter_looks)
    for k, v in (ai_doc.get("chapter_looks") or {}).items():
        if str(v) in VALID_FILM_STOCK_LOOKS:
            merged_chapter_looks[str(k)] = str(v)

    top_reasons = "; ".join(ranked[0]["reasons"]) if ranked else spec_p["subtitle"]
    default_rationale = (
        f"Recommended '{primary_look}' ({spec_p['display_name']} @ {int(round(rec_intensity * 100))}% LUT): "
        f"{top_reasons} (median {digest.get('kelvin_median', 5600):.0f}K, "
        f"spread {digest.get('kelvin_spread', 0):.0f}K across {digest.get('count', 0)} shots)."
    )
    ai_rationale = str(ai_doc.get("ai_rationale") or default_rationale)

    alt_candidates = [primary_look, runner_up] + [r["look"] for r in ranked if r["look"] not in (primary_look, runner_up)][:2]
    options: list[str] = []
    for idx_opt, opt_look in enumerate(alt_candidates):
        sp = CINEMA_FILM_STOCK_SPECS[opt_look]
        prefix = "(Recommended) " if idx_opt == 0 else ""
        desc = ai_rationale[:110] if idx_opt == 0 else sp["subtitle"]
        options.append(f"{prefix}{opt_look} — {sp['display_name']}: {desc}")
    options.append("neutral_reference — Neutral Reference: Primary WB balance only (no film LUT look)")

    ask_payload = [
        {
            "id": "color_grade_look",
            "question": f"Select DI Film-Stock Grade for '{timeline.project_name}' (Stage 42 3D LUT + Primary Camera Match)",
            "options": options,
            "is_multi_select": False,
        }
    ]

    rec = ColorGradeRecommendation(
        project_title=timeline.project_name,
        recommended_primary_look=primary_look,  # type: ignore[arg-type]
        runner_up_look=runner_up,  # type: ignore[arg-type]
        recommended_strategy=strategy,  # type: ignore[arg-type]
        recommended_lut_intensity=round(rec_intensity, 2),
        recommended_chapter_looks=merged_chapter_looks,
        stock_rankings=ranked,
        photometry_digest=digest,
        recommendation_source=rec_source,  # type: ignore[arg-type]
        ai_rationale=ai_rationale,
        lookbook_sheet_path=lookbook_path,
        ask_question_tool_payload=ask_payload,
    )
    blackboard.set_custom_artifact("color_grade_recommendation.json", rec.model_dump(mode="json"))
    return rec


@register_agent
class CinemaColorGradingAgent(BaseEditingAgent):
    """Stage 42 DI Color Grader: perceptual colorimetry, 3-way LGG matching & 3D LUT print synthesis."""

    role_id = "color_grader"
    legacy_role_aliases = ["di_colorist"]
    display_name = "Cinema DI Color Grader"
    stage_order = 42
    substage_priority = 0
    description = (
        "Measures multi-frame Planckian colorimetry across videos and still photos, solves 3-way "
        "Lift/Gamma/Gain camera matching, synthesizes 33x33x33 .cube 3D LUTs, and emits lookbook "
        "and RGB Parade scopes sheets while keeping synthetic graphics color-bypassed."
    )
    enabled_by_default = True
    requires_artifacts = ["timeline.yaml"]
    produces_artifacts = ["color_grade_plan.json"]
    skill_reference_file = "19_color_grading_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline
        if timeline is None:
            return {"shots_matched_count": 0, "skipped": True}

        lut_paths = ensure_project_luts(blackboard.project_dir)

        hlg_hints: dict[str, bool] = {}
        cst_hints: dict[str, dict[str, Any]] = {}
        for fx in timeline.effects_track:
            if fx.effect_type == "color_grade" and fx.target_clip_id:
                hlg_hints[fx.target_clip_id] = bool(fx.parameters.get("hlg_rolloff", False))
                cst_hints[fx.target_clip_id] = dict(fx.parameters or {})
        if blackboard.color_transform_manifest is not None:
            for ct in blackboard.color_transform_manifest.clips:
                cst_hints.setdefault(
                    ct.clip_id,
                    {
                        "src_trc": ct.color_trc,
                        "src_primaries": ct.color_primaries,
                        "src_space": ct.color_space,
                        "diffuse_white_nits": 203.0,
                        "hdr": bool(blackboard.hdr),
                    },
                )
        timeline.effects_track = [fx for fx in timeline.effects_track if fx.effect_type != "color_grade"]

        measured, skipped_synthetic = _measure_timeline_clips(timeline)

        lookbook_path = ""
        existing_sources = [str(it.source) for _, it, _ in measured if Path(str(it.source)).exists()]
        if existing_sources:
            try:
                lb = render_color_lookbook_sheet(
                    existing_sources[:4],
                    blackboard.project_dir / "_color_lookbook_8stocks.jpg",
                    project_title=timeline.project_name,
                )
                lookbook_path = str(lb)
            except Exception as exc:
                logger.debug("Lookbook generation skipped: %s", exc)

        rec = recommend_color_grading(
            blackboard,
            measured=measured,
            lookbook_path=lookbook_path,
            agent=self,
        )
        primary_look = rec.recommended_primary_look
        strategy = rec.recommended_strategy
        chapter_looks_override = dict(rec.recommended_chapter_looks)

        decision_file = blackboard.project_dir / "ai_decisions" / "color_grader.json"
        ai_override_doc: dict[str, Any] = {}
        if decision_file.exists():
            try:
                ai_override_doc = json.loads(decision_file.read_text(encoding="utf-8"))
            except Exception:
                ai_override_doc = {}
        clip_overrides: dict[str, dict[str, Any]] = dict(ai_override_doc.get("clip_overrides") or {})

        kelvins = [float(t["est_kelvin"]) for _, _, t in measured]
        ref_kelvin = float(_clamp(median(kelvins) if kelvins else 5600.0, 5200.0, 5800.0))

        clip_grades: list[ClipColorGrade] = []
        resolved_chapter_looks: dict[str, str] = {}
        scope_samples: list[dict[str, Any]] = []
        post_wb_kelvins: list[float] = []

        for idx, item, tel in measured:
            clip_act = str(getattr(item, "act", "") or "").strip()
            note = str(getattr(item, "narrative_note", "") or "")
            look_name, ch_label = _resolve_clip_chapter_look(
                clip_note=note,
                source_path=item.source,
                telemetry=tel,
                primary_look=primary_look,
                strategy=strategy,
                chapter_looks=chapter_looks_override,
                clip_act=clip_act,
            )
            resolved_chapter_looks[ch_label] = look_name
            cst_meta = dict(cst_hints.get(item.id) or {})
            cst_meta.setdefault("hdr", bool(blackboard.hdr))
            clip_ov = dict(clip_overrides.get(item.id) or {})
            if "lut_intensity" not in clip_ov and look_name == primary_look and rec.recommended_lut_intensity > 0.0:
                clip_ov["lut_intensity"] = rec.recommended_lut_intensity
            grade_obj, fx_params = _solve_clip_color_grade(
                clip_id=item.id,
                media_type=item.type,
                source=item.source,
                telemetry=tel,
                ref_kelvin=ref_kelvin,
                look_name=look_name,
                lut_paths=lut_paths,
                hlg_rolloff=hlg_hints.get(item.id, False),
                cst_meta=cst_meta,
                override=clip_ov,
            )
            clip_grades.append(grade_obj)
            residual_k = grade_obj.measured_kelvin + (ref_kelvin - grade_obj.measured_kelvin) * 0.72
            post_wb_kelvins.append(residual_k)

            timeline.effects_track.append(
                EffectTrackItem(
                    id=f"fx_color_{idx}_{item.id}",
                    effect_type="color_grade",
                    target_clip_id=item.id,
                    timeline_start=item.timeline_start,
                    duration=item.duration,
                    intensity=grade_obj.lut_intensity,
                    parameters=fx_params,
                    created_by_agent=self.role_id,
                )
            )
            scope_samples.append(
                {
                    "clip_id": item.id,
                    "media_type": item.type,
                    "source": item.source,
                    "start": item.source_in,
                    "duration": item.duration,
                    "grade": grade_obj,
                }
            )

        spread_before = round((max(kelvins) - min(kelvins)) if len(kelvins) >= 2 else 0.0, 1)
        spread_after = round((max(post_wb_kelvins) - min(post_wb_kelvins)) if len(post_wb_kelvins) >= 2 else 0.0, 1)

        scopes_path = ""
        if existing_sources:
            try:
                photo_samples = [s for s in scope_samples if s["media_type"] == "photo" and Path(s["source"]).exists()]
                video_samples = [s for s in scope_samples if s["media_type"] != "photo" and Path(s["source"]).exists()]
                picked: list[dict[str, Any]] = []
                for p_i in range(max(len(photo_samples), len(video_samples))):
                    if p_i < len(video_samples) and len(picked) < 6:
                        picked.append(video_samples[p_i])
                    if p_i < len(photo_samples) and len(picked) < 6:
                        picked.append(photo_samples[p_i])
                    if len(picked) >= 6:
                        break
                if picked:
                    report_preview = ColorGradeReport(
                        project_title=timeline.project_name,
                        primary_film_stock_look=primary_look,
                        grading_strategy=strategy,
                        kelvin_spread_before=spread_before,
                        kelvin_spread_after=spread_after,
                        recommendation_source=rec.recommendation_source,
                        ai_rationale=rec.ai_rationale,
                    )
                    sc = render_color_grading_scopes_sheet(
                        picked,
                        blackboard.project_dir / "_color_grading_scopes.jpg",
                        report=report_preview,
                    )
                    scopes_path = str(sc)
            except Exception as exc:
                logger.debug("Scopes sheet generation skipped: %s", exc)

        report = ColorGradeReport(
            project_title=timeline.project_name,
            primary_film_stock_look=primary_look,
            grading_strategy=strategy,
            shots_matched_count=len(clip_grades),
            chapter_looks=resolved_chapter_looks,
            generated_luts=lut_paths,
            scopes_sheet_path=scopes_path,
            lookbook_sheet_path=lookbook_path,
            kelvin_spread_before=spread_before,
            kelvin_spread_after=spread_after,
            recommendation_source=rec.recommendation_source,
            ai_rationale=rec.ai_rationale,
            colorist_notes=(
                f"Balanced {len(clip_grades)} real camera clips toward {ref_kelvin:.0f}K ({strategy}, "
                f"spine='{primary_look}', source={rec.recommendation_source}); bypassed {skipped_synthetic} synthetic graphics."
            ),
            clip_grades=clip_grades,
        )

        out_path = blackboard.set_custom_artifact("color_grade_plan.json", report.model_dump(mode="json"))
        timeline.to_yaml(blackboard.project_dir / "timeline.yaml")

        blackboard.emit_activity(
            self.role_id,
            f"Graded {len(clip_grades)} V1/V2 clips with '{primary_look}' (Kelvin spread {spread_before:.0f}K -> {spread_after:.0f}K)",
            detail=f"Source={rec.recommendation_source}; synthesized {len(lut_paths)} .cube LUTs; bypassed {skipped_synthetic} graphics",
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="executive_director",
            subject="DI Color Grade Plan & 3D LUTs ready",
            payload={
                "primary_film_stock_look": primary_look,
                "grading_strategy": strategy,
                "recommendation_source": rec.recommendation_source,
                "shots_matched_count": len(clip_grades),
                "kelvin_spread_before": spread_before,
                "kelvin_spread_after": spread_after,
            },
        )

        return {
            "shots_matched_count": len(clip_grades),
            "primary_film_stock_look": primary_look,
            "grading_strategy": strategy,
            "recommendation_source": rec.recommendation_source,
            "ai_rationale": rec.ai_rationale,
            "skipped_synthetic_graphics": skipped_synthetic,
            "color_grade_plan_path": str(out_path),
            "color_grade_recommendation_path": str(blackboard.project_dir / "color_grade_recommendation.json"),
            "scopes_sheet_path": scopes_path,
            "lookbook_sheet_path": lookbook_path,
        }
