"""Tests for Stage 42 Cinema DI Color Grading Subagent, 3D LUTs, FFmpeg/Pillow Rendering, and Skill CLI."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import pytest

from video_editing_ai.agents.base import PipelineBlackboard
import video_editing_ai.agents.orchestrator  # noqa: F401
from video_editing_ai.agents.plugins.color_grading_agent import (
    CinemaColorGradingAgent,
    apply_color_grade_overrides,
    classify_clip_grading_suitability,
    recommend_color_grading,
)
from video_editing_ai.models.schema import (
    ClipColorGrade,
    ColorGradeRecommendation,
    ColorGradeReport,
    EditTimeline,
    EffectTrackItem,
    KenBurnsConfig,
    VisualTrackItem,
)
from video_editing_ai.renderer.color_grading import (
    analyze_rgb_sample_array,
    apply_color_grade_to_rgb,
    build_film_stock_lut_table,
    measure_media_colorimetry,
    write_film_stock_cube_lut,
)
from video_editing_ai.renderer.ffmpeg_composer import _build_color_grade_filter, _render_video_segment
from video_editing_ai.renderer.ken_burns import render_photo_clip


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        pytest.skip("ffmpeg binary not installed")
    return exe


def _make_synthetic_video(out_mp4: Path, hex_color: str, duration: float = 1.2) -> Path:
    ffmpeg = _ffmpeg_bin()
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={hex_color}:s=320x180:d={duration:.2f}:r=24",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    return out_mp4


def _make_test_photo(out_jpg: Path, bg_rgb: tuple[int, int, int], with_skin: bool = False) -> Path:
    im = Image.new("RGB", (640, 480), bg_rgb)
    if with_skin:
        draw = ImageDraw.Draw(im)
        # Central skin-tone subject patch
        draw.ellipse([210, 120, 430, 360], fill=(208, 152, 122))
    im.save(out_jpg, quality=95)
    return out_jpg


def test_planckian_colorimetry_detects_warm_tungsten_vs_cool_shade_and_skin(tmp_path: Path) -> None:
    warm_arr = np.full((64, 64, 3), [245, 185, 115], dtype=np.uint8)
    cool_arr = np.full((64, 64, 3), [125, 170, 245], dtype=np.uint8)
    neutral_arr = np.full((64, 64, 3), [180, 180, 180], dtype=np.uint8)

    warm_stats = analyze_rgb_sample_array(warm_arr)
    cool_stats = analyze_rgb_sample_array(cool_arr)
    neu_stats = analyze_rgb_sample_array(neutral_arr)

    assert warm_stats["est_kelvin"] < 4200.0
    assert cool_stats["est_kelvin"] > 7200.0
    assert warm_stats["est_kelvin"] < neu_stats["est_kelvin"] < cool_stats["est_kelvin"]
    assert cool_stats["skin_pixel_ratio"] < 0.02

    portrait_path = _make_test_photo(tmp_path / "portrait.jpg", (110, 155, 225), with_skin=True)
    p_tel = measure_media_colorimetry(portrait_path, media_type="photo")
    # Ellipse occupies ~13.5% of the 640x480 canvas
    assert p_tel["skin_pixel_ratio"] > 0.10


def test_cube_3d_lut_synthesis_subtractive_rolloff_and_skin_protection(tmp_path: Path) -> None:
    cube_full = write_film_stock_cube_lut("archival_kodak_2383", tmp_path / "kodak2383.cube", lut_size=33)
    lines = cube_full.read_text(encoding="utf-8").splitlines()
    assert any(ln == "LUT_3D_SIZE 33" for ln in lines)
    assert any(ln == "DOMAIN_MIN 0.0 0.0 0.0" for ln in lines)
    assert any(ln == "DOMAIN_MAX 1.0 1.0 1.0" for ln in lines)

    data_lines = [ln for ln in lines if ln.strip() and ln[0].isdigit()]
    assert len(data_lines) == 33 ** 3

    table = build_film_stock_lut_table("archival_kodak_2383", lut_size=33)
    # R-fastest check: index 1 in flattened file corresponds to (b=0, g=0, r=1)
    r1_vals = [float(x) for x in data_lines[1].split()]
    np.testing.assert_allclose(r1_vals, table[0, 0, 1], atol=1e-4)

    # Neutral reference must be exact unity
    neutral_tbl = build_film_stock_lut_table("neutral_reference", lut_size=33)
    assert abs(float(neutral_tbl[16, 16, 16, 0]) - 0.5) < 1e-6

    # Skin protection should attenuate extreme stylized twilight shift relative to non-skin cyan shadow
    img = np.zeros((16, 32, 3), dtype=np.uint8)
    img[:, :16, :] = [208, 152, 122]  # skin
    img[:, 16:, :] = [70, 95, 115]    # neutral/cool background
    grade_prot = ClipColorGrade(
        clip_id="skin_test",
        chapter_look="twilight_teal_amber",
        lift_rgb=(-0.024, 0.006, 0.026),
        gamma_rgb=(0.010, 0.002, -0.012),
        gain_rgb=(0.028, 0.008, -0.026),
        contrast=1.06,
        saturation=1.05,
        skin_protection=0.85,
        lut_intensity=0.90,
    )
    grade_unprot = grade_prot.model_copy(update={"skin_protection": 0.0})
    out_prot = apply_color_grade_to_rgb(img, grade_prot)
    out_unprot = apply_color_grade_to_rgb(img, grade_unprot)

    # Skin shift distance from original must be smaller when skin_protection=0.85
    delta_prot = float(np.mean(np.abs(out_prot[:, :16].astype(float) - img[:, :16].astype(float))))
    delta_unprot = float(np.mean(np.abs(out_unprot[:, :16].astype(float) - img[:, :16].astype(float))))
    assert delta_prot < delta_unprot


def test_stage42_color_grader_matches_mixed_cameras_and_skips_motion_graphics(tmp_path: Path) -> None:
    warm_mp4 = _make_synthetic_video(tmp_path / "warm_tungsten.mp4", "#F2B66D", duration=1.0)
    cool_jpg = _make_test_photo(tmp_path / "cool_coastal.jpg", (128, 172, 244))
    broll_jpg = _make_test_photo(tmp_path / "broll_gallery.jpg", (185, 175, 165))
    title_mp4 = _make_synthetic_video(tmp_path / "sec_title_day1.mp4", "#101216", duration=1.0)
    map_mp4 = _make_synthetic_video(tmp_path / "flyover_leg_01.mp4", "#182436", duration=1.0)

    tl = EditTimeline(
        project_name="Color Mix Test",
        visual_track=[
            VisualTrackItem(id="clip_title", type="video", source=str(title_mp4), duration=1.0, timeline_start=0.0, act="Opening"),
            VisualTrackItem(id="clip_warm", type="video", source=str(warm_mp4), duration=1.0, timeline_start=1.0, act="Evening"),
            VisualTrackItem(id="clip_cool", type="photo", source=str(cool_jpg), duration=1.0, timeline_start=2.0, act="Beach Coast"),
        ],
        broll_track=[
            VisualTrackItem(id="broll_photo", type="photo", source=str(broll_jpg), duration=0.8, timeline_start=1.1, track_lane=2, narrative_note="Museum Gallery"),
            VisualTrackItem(id="broll_map", type="video", source=str(map_mp4), duration=0.8, timeline_start=2.1, track_lane=2, narrative_note="Travel Map Flyover"),
        ],
    )
    tl.to_yaml(tmp_path / "timeline.yaml")

    bb = PipelineBlackboard(media_dir=tmp_path, project_dir=tmp_path, use_antigravity=False)
    bb.timeline = tl
    res = CinemaColorGradingAgent().run(bb)

    assert res["shots_matched_count"] == 3
    assert res["skipped_synthetic_graphics"] == 2
    assert (tmp_path / "color_grade_plan.json").exists()
    assert (tmp_path / "_luts" / "archival_kodak_2383.cube").exists()

    report = ColorGradeReport.from_json_file(tmp_path / "color_grade_plan.json")
    assert report.kelvin_spread_after < report.kelvin_spread_before
    by_id = {g.clip_id: g for g in report.clip_grades}
    assert by_id["clip_warm"].measured_kelvin < by_id["clip_cool"].measured_kelvin
    # Warm tungsten clip gets positive blue WB shift (cools down), cool clip gets positive red shift (warms up)
    assert by_id["clip_warm"].wb_blue_shift > 0.0
    assert by_id["clip_cool"].wb_red_shift > 0.0

    fx_targets = {fx.target_clip_id for fx in tl.effects_track if fx.effect_type == "color_grade"}
    assert fx_targets == {"clip_warm", "clip_cool", "broll_photo"}


def test_photo_color_grading_preserves_pure_white_gallery_matte_and_renders_hdr_video(tmp_path: Path) -> None:
    ffmpeg = _ffmpeg_bin()
    cube_path = write_film_stock_cube_lut("golden_hour_warmth", tmp_path / "golden.cube", lut_size=33, intensity=0.85)
    photo_path = _make_test_photo(tmp_path / "art.jpg", (90, 145, 220), with_skin=True)

    # Strong warm grade to prove inner photo shifts while gallery matte border remains pure neutral white
    strong_fx = EffectTrackItem(
        id="fx_photo_1",
        effect_type="color_grade",
        target_clip_id="clip_art",
        timeline_start=0.0,
        duration=0.4,
        intensity=0.90,
        parameters={
            "chapter_look": "golden_hour_warmth",
            "wb_red_shift": 0.16,
            "wb_blue_shift": -0.14,
            "contrast": 1.08,
            "saturation": 1.08,
            "lut_intensity": 0.90,
            "lut_cube_path": str(cube_path),
        },
    )
    ungraded_photo_mp4 = render_photo_clip(
        photo_path=photo_path,
        output_mp4=tmp_path / "ungraded_gallery.mp4",
        duration=0.4,
        width=640,
        height=360,
        fps=15,
        hdr=False,
        ken_burns=KenBurnsConfig(mode="gallery_frame", caption_title="Gallery Exhibit", border_px=14),
        attached_effects=None,
    )
    graded_photo_mp4 = render_photo_clip(
        photo_path=photo_path,
        output_mp4=tmp_path / "graded_gallery.mp4",
        duration=0.4,
        width=640,
        height=360,
        fps=15,
        hdr=False,
        ken_burns=KenBurnsConfig(mode="gallery_frame", caption_title="Gallery Exhibit", border_px=14),
        attached_effects=[strong_fx],
    )
    assert graded_photo_mp4.exists() and graded_photo_mp4.stat().st_size > 2000

    # Decode mid-frame from both clips and verify matte stays identical archival ivory while photo interior shifts
    png_ungraded = tmp_path / "frame_ungraded.png"
    png_graded = tmp_path / "frame_graded.png"
    for mp4_p, png_p in ((ungraded_photo_mp4, png_ungraded), (graded_photo_mp4, png_graded)):
        subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-ss", "0.15", "-i", str(mp4_p), "-frames:v", "1", str(png_p)],
            check=True,
        )
    arr_u = np.asarray(Image.open(png_ungraded).convert("RGB"), dtype=np.int16)
    arr_g = np.asarray(Image.open(png_graded).convert("RGB"), dtype=np.int16)
    matte_mask = (arr_u[..., 0] >= 238) & (arr_u[..., 1] >= 238) & (arr_u[..., 2] >= 232)
    assert int(np.sum(matte_mask)) >= 500, "Expected archival gallery matte border pixels"
    matte_drift = np.abs(arr_g[matte_mask] - arr_u[matte_mask])
    assert float(np.mean(matte_drift)) <= 2.5

    h, w, _ = arr_u.shape
    inner_u = arr_u[h // 3 : (2 * h) // 3, w // 3 : (2 * w) // 3]
    inner_g = arr_g[h // 3 : (2 * h) // 3, w // 3 : (2 * w) // 3]
    assert float(np.mean(np.abs(inner_g - inner_u))) > 8.0

    # Render an actual HLG HDR video clip through _render_video_segment using lut3d + zscale (no code 3074 crash!)
    src_mp4 = _make_synthetic_video(tmp_path / "vid_in.mp4", "#70A0D8", duration=0.8)
    v_item = VisualTrackItem(id="clip_vid", type="video", source=str(src_mp4), timeline_start=0.0, duration=0.6, source_in=0.0, source_out=0.6)
    vid_fx = strong_fx.model_copy(
        update={
            "target_clip_id": "clip_vid",
            "parameters": {**strong_fx.parameters, "hlg_rolloff": True, "hdr": True, "src_trc": "bt709", "src_space": "bt709"},
        }
    )
    flt_str = _build_color_grade_filter(vid_fx, hdr=True)
    assert "lutyuv=" in flt_str and "lut3d=" in flt_str and "scale=out_color_matrix=bt709" in flt_str

    hdr_out = _render_video_segment(
        item=v_item,
        output_mp4=tmp_path / "vid_hdr_graded.mp4",
        width=640,
        height=360,
        fps=15,
        hdr=True,
        attached_effects=[vid_fx],
    )
    assert hdr_out.exists() and hdr_out.stat().st_size > 2000

    # Regression check: 10-bit HLG/BT.2020 video with negative black_lift (colorlevels dehaze)
    # must prepend format=rgb48le so FFmpeg 7.1+ does not clamp 10-bit inputs to solid black.
    hlg_src_mp4 = tmp_path / "vid_hlg_10bit_in.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=#7892B8:s=320x180:d=0.60:r=24",
            "-pix_fmt",
            "yuv420p10le",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "arib-std-b67",
            "-colorspace",
            "bt2020nc",
            "-c:v",
            "libx265",
            "-x265-params",
            "log-level=error",
            str(hlg_src_mp4),
        ],
        check=True,
    )
    hlg_item = VisualTrackItem(
        id="clip_hlg_10bit",
        type="video",
        source=str(hlg_src_mp4),
        timeline_start=0.0,
        duration=0.5,
        source_in=0.0,
        source_out=0.5,
    )
    pedestal_fx = EffectTrackItem(
        id="fx_hlg_pedestal",
        effect_type="color_grade",
        target_clip_id="clip_hlg_10bit",
        timeline_start=0.0,
        duration=0.5,
        intensity=0.55,
        parameters={
            "chapter_look": "cinematic_warm_film",
            "black_lift": -0.11,
            "lut_intensity": 0.45,
            "lut_cube_path": str(cube_path),
            "hlg_rolloff": True,
            "hdr": True,
            "src_trc": "arib-std-b67",
            "src_space": "bt2020nc",
        },
    )
    pedestal_flt = _build_color_grade_filter(pedestal_fx, hdr=True)
    assert "format=rgb48le,colorlevels=" in pedestal_flt
    hlg_graded_mp4 = _render_video_segment(
        item=hlg_item,
        output_mp4=tmp_path / "vid_hlg_10bit_graded.mp4",
        width=640,
        height=360,
        fps=15,
        hdr=True,
        attached_effects=[pedestal_fx],
    )
    assert hlg_graded_mp4.exists() and hlg_graded_mp4.stat().st_size > 2000
    hlg_png = tmp_path / "frame_hlg_pedestal.png"
    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-ss", "0.2", "-i", str(hlg_graded_mp4), "-frames:v", "1", str(hlg_png)],
        check=True,
    )
    hlg_arr = np.asarray(Image.open(hlg_png).convert("RGB"), dtype=np.float32)
    assert float(hlg_arr.mean()) > 30.0, f"10-bit colorlevels clamped to black: mean={hlg_arr.mean():.2f}"


def test_surgical_set_color_grade_and_skill_cli_lookbook_and_scopes(tmp_path: Path) -> None:
    p1 = _make_test_photo(tmp_path / "p1.jpg", (215, 175, 130), with_skin=True)
    p2 = _make_test_photo(tmp_path / "p2.jpg", (120, 165, 225))
    tl = EditTimeline(
        project_name="Skill CLI Color Test",
        visual_track=[
            VisualTrackItem(id="clip_01", type="photo", source=str(p1), duration=2.0, timeline_start=0.0),
            VisualTrackItem(id="clip_02", type="photo", source=str(p2), duration=2.0, timeline_start=2.0),
        ],
    )
    tl.to_yaml(tmp_path / "timeline.yaml")

    apply_color_grade_overrides(tl, tmp_path, look_name="archival_kodak_2383", intensity=0.80)
    tl.to_yaml(tmp_path / "timeline.yaml")

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / ".agent" / "skills" / "ai-video-editor" / "scripts" / "ai_video_studio_skill.py"
    env = dict(os.environ, AI_VIDEO_EDITOR_ROOT=str(repo_root))

    lookbook_jpg = tmp_path / "lookbook.jpg"
    scopes_jpg = tmp_path / "scopes.jpg"

    for cmd_args in (
        ["preview-color-grades", "--project-dir", str(tmp_path), "--output", str(lookbook_jpg)],
        ["render-color-scopes", "--project-dir", str(tmp_path), "--output", str(scopes_jpg)],
        [
            "edit-cut",
            "--project-dir",
            str(tmp_path),
            "--action",
            "set-color-grade",
            "--clip-id",
            "clip_01",
            "--look",
            "golden_hour_warmth",
            "--lut-intensity",
            "0.74",
            "--exposure-ev",
            "0.18",
        ],
    ):
        proc = subprocess.run(
            [sys.executable, str(script_path), *cmd_args],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"CLI failed for {cmd_args}: {proc.stderr}\n{proc.stdout}"

    assert lookbook_jpg.exists() and Image.open(lookbook_jpg).size == (2560, 1440)
    assert scopes_jpg.exists() and Image.open(scopes_jpg).size == (2560, 1440)

    updated_report = ColorGradeReport.from_json_file(tmp_path / "color_grade_plan.json")
    c1 = next(g for g in updated_report.clip_grades if g.clip_id == "clip_01")
    assert c1.chapter_look == "golden_hour_warmth"
    assert abs(c1.exposure_ev - 0.18) < 1e-3
    assert abs(c1.lut_intensity - 0.74) < 1e-3


def test_ai_color_grading_recommendation_multimodal_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coastal_jpg = _make_test_photo(tmp_path / "coastal.jpg", (120, 175, 240))
    dinner_jpg = _make_test_photo(tmp_path / "dinner.jpg", (210, 175, 130), with_skin=True)

    tl = EditTimeline(
        project_name="AI Color Recommend Test",
        visual_track=[
            VisualTrackItem(id="clip_coast", type="photo", source=str(coastal_jpg), duration=2.0, timeline_start=0.0, act="Chapter I · Coastal Day"),
            VisualTrackItem(id="clip_dinner", type="photo", source=str(dinner_jpg), duration=2.0, timeline_start=2.0, act="Chapter II · Indoor Dinner"),
        ],
    )
    tl.to_yaml(tmp_path / "timeline.yaml")

    captured_calls: list[dict[str, object]] = []

    def _fake_ask_ai(self, blackboard, prompt, image_paths=None, decision_key=None, timeout_sec=45.0):
        captured_calls.append(
            {
                "prompt": prompt,
                "image_paths": [str(p) for p in (image_paths or [])],
                "decision_key": decision_key,
            }
        )
        return {
            "primary_film_stock_look": "fuji_eterna_soft",
            "runner_up_look": "archival_kodak_2383",
            "grading_strategy": "chapter_adaptive",
            "recommended_lut_intensity": 0.77,
            "chapter_looks": {
                "Chapter I · Coastal Day": "pacific_coastal_chrome",
                "Chapter II · Indoor Dinner": "fuji_eterna_soft",
            },
            "ai_rationale": "Visual VLM audit paired Eterna portraits with Ektachrome coastal daylight.",
        }

    monkeypatch.setattr(CinemaColorGradingAgent, "ask_ai_creative_json", _fake_ask_ai)

    bb = PipelineBlackboard(media_dir=tmp_path, project_dir=tmp_path, use_antigravity=True)
    bb.timeline = tl
    rec = recommend_color_grading(bb)

    assert len(captured_calls) == 1
    assert any(p.endswith("_color_lookbook_8stocks.jpg") for p in captured_calls[0]["image_paths"])  # type: ignore[index]
    assert rec.recommendation_source == "ai_vision"
    assert rec.recommended_primary_look == "fuji_eterna_soft"
    assert rec.recommended_chapter_looks["Chapter I · Coastal Day"] == "pacific_coastal_chrome"
    assert rec.ask_question_tool_payload[0]["options"][0].startswith("(Recommended) fuji_eterna_soft")
    assert (tmp_path / "color_grade_recommendation.json").exists()

    # Verify Skill CLI offline photometric recommend + --apply works end to end cleanly
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / ".agent" / "skills" / "ai-video-editor" / "scripts" / "ai_video_studio_skill.py"
    env = dict(os.environ, AI_VIDEO_EDITOR_ROOT=str(repo_root))
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "recommend-color-grade",
            "--project-dir",
            str(tmp_path),
            "--no-ai",
            "--apply",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"recommend-color-grade CLI failed: {proc.stderr}\n{proc.stdout}"
    cli_rec = ColorGradeRecommendation.from_json_file(tmp_path / "color_grade_recommendation.json")
    assert cli_rec.recommendation_source == "photometric_ranker"
    assert (tmp_path / "color_grade_plan.json").exists()


def test_color_scientist_suitability_regimes_and_natural_bypass(tmp_path: Path) -> None:
    # 1. Pure optical regime classifier checks
    assert classify_clip_grading_suitability({"est_kelvin": 6800.0, "est_tint": 0.004, "p05_luma": 0.25, "p50_luma": 0.72, "p95_luma": 0.90, "skin_pixel_ratio": 0.0}) == "preserve_natural"
    assert classify_clip_grading_suitability({"est_kelvin": 5600.0, "est_tint": 0.002, "p05_luma": 0.07, "p50_luma": 0.49, "p95_luma": 0.81, "skin_pixel_ratio": 0.12}) == "preserve_natural"
    assert classify_clip_grading_suitability({"est_kelvin": 7100.0, "est_tint": -0.004, "p05_luma": 0.22, "p50_luma": 0.38, "p95_luma": 0.52, "skin_pixel_ratio": 0.0}) == "log_dehaze_restore"
    assert classify_clip_grading_suitability({"est_kelvin": 5200.0, "est_tint": 0.003, "p05_luma": 0.03, "p50_luma": 0.24, "p95_luma": 0.68, "skin_pixel_ratio": 0.20}) == "shadow_recovery"
    assert classify_clip_grading_suitability({"est_kelvin": 4900.0, "est_tint": 0.042, "p05_luma": 0.08, "p50_luma": 0.48, "p95_luma": 0.85, "skin_pixel_ratio": 0.05}) == "wb_tint_balance"
    assert classify_clip_grading_suitability({"est_kelvin": 5200.0, "est_tint": 0.004, "p05_luma": 0.08, "p50_luma": 0.44, "p95_luma": 0.60, "skin_pixel_ratio": 0.18}) == "creative_film_print"

    # 2. Timeline integration: optimal high-key sky bypassed vs hazy fog & dark room & green cast graded
    # High-key sky with natural clouds gradient [155..238]
    hk_arr = np.linspace(155, 238, 128, dtype=np.uint8)[None, :, None].repeat(96, axis=0).repeat(3, axis=2)
    hk_path = tmp_path / "highkey.jpg"
    Image.fromarray(hk_arr, mode="RGB").save(hk_path, quality=95)

    # Hazy flat marine image [65..128] (high black pedestal p05~0.25, narrow span~0.25)
    hz_arr = np.linspace(65, 128, 128, dtype=np.uint8)[None, :, None].repeat(96, axis=0).repeat(3, axis=2)
    hz_path = tmp_path / "haze.jpg"
    Image.fromarray(hz_arr, mode="RGB").save(hz_path, quality=95)

    # Dark underexposed interior [8..175] with mostly dark area (80% @ 30, 20% @ 175)
    dk_arr = np.full((96, 128, 3), 30, dtype=np.uint8)
    dk_arr[:20, :32, :] = 175
    dk_path = tmp_path / "dark.jpg"
    Image.fromarray(dk_arr, mode="RGB").save(dk_path, quality=95)

    # Fluorescent green-tinted indoor shot [R=105, G=155, B=110] with highlight stripe
    gr_arr = np.full((96, 128, 3), [105, 155, 110], dtype=np.uint8)
    gr_arr[:24, :, :] = [210, 238, 212]
    gr_arr[72:, :, :] = [28, 52, 30]
    gr_path = tmp_path / "green.jpg"
    Image.fromarray(gr_arr, mode="RGB").save(gr_path, quality=95)

    tl = EditTimeline(
        project_name="Suitability Test",
        visual_track=[
            VisualTrackItem(id="clip_hk", type="photo", source=str(hk_path), duration=1.0, timeline_start=0.0, act="Day Sky"),
            VisualTrackItem(id="clip_hz", type="photo", source=str(hz_path), duration=1.0, timeline_start=1.0, act="Ocean Mist"),
            VisualTrackItem(id="clip_dk", type="photo", source=str(dk_path), duration=1.0, timeline_start=2.0, act="Night Room"),
            VisualTrackItem(id="clip_gr", type="photo", source=str(gr_path), duration=1.0, timeline_start=3.0, act="Hallway"),
        ],
    )
    tl.to_yaml(tmp_path / "timeline.yaml")

    bb = PipelineBlackboard(media_dir=tmp_path, project_dir=tmp_path, use_antigravity=False)
    bb.timeline = tl
    res = CinemaColorGradingAgent().run(bb)

    assert res["bypassed_natural_count"] >= 1
    report = ColorGradeReport.from_json_file(tmp_path / "color_grade_plan.json")
    by_id = {g.clip_id: g for g in report.clip_grades}

    assert by_id["clip_hk"].grading_regime == "preserve_natural" and by_id["clip_hk"].bypass is True
    assert by_id["clip_hz"].grading_regime == "log_dehaze_restore" and by_id["clip_hz"].black_lift < 0.0
    assert by_id["clip_dk"].grading_regime == "shadow_recovery" and by_id["clip_dk"].gamma >= 1.05
    assert by_id["clip_gr"].grading_regime == "wb_tint_balance" and by_id["clip_gr"].wb_green_shift < 0.0

    fx_by_id = {fx.target_clip_id: fx for fx in tl.effects_track if fx.effect_type == "color_grade"}
    assert "clip_hk" not in fx_by_id
    assert {"clip_hz", "clip_dk", "clip_gr"}.issubset(set(fx_by_id.keys()))

    flt_haze = _build_color_grade_filter(fx_by_id["clip_hz"], hdr=True)
    assert "colorlevels=" in flt_haze and "scale=out_color_matrix=" in flt_haze
