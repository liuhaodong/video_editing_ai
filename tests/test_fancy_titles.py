"""Verification tests for the 6-theme Cinema Kinetic Title & Typography Engine."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
import pytest

from video_editing_ai.agents.base import PipelineBlackboard
from video_editing_ai.agents.director_agent import ProductionDirectorAgent
from video_editing_ai.agents.intake_agent import ClientIntakeAgent, generate_intake_questions
from video_editing_ai.agents.plugins.motion_graphics_agent import MotionGraphicsTitleAgent
from video_editing_ai.agents.visual_qc_agent import ProxyWatchdownQCAgent
from video_editing_ai.exporters.nle_export import export_fcpxml
from video_editing_ai.models.schema import (
    ChapterPlan,
    CreativeBrief,
    CulledSubClip,
    EditTimeline,
    MediaCatalog,
    MediaIndex,
    PacingAuditReport,
    PaperEdit,
    PhotoAsset,
    SubtitleConfig,
    SubtitleItem,
    TimelineSettings,
    TitleOverlayItem,
    VideoAsset,
    VisualTrackItem,
    WordTimestamp,
    VALID_TITLE_STYLES,
    VALID_TITLE_THEMES,
)
from video_editing_ai.renderer.ffmpeg_composer import render_timeline
from video_editing_ai.renderer.subtitles import resolve_subtitle_font_stack
from video_editing_ai.renderer.titles import (
    HDR_DIFFUSE_WHITE_PEAK_RGB,
    HDR_MAX_SHADOW_ALPHA,
    _load_font_tier,
    get_theme_spec,
    measure_tracked_text,
    render_title_frame,
    render_title_png,
    sample_shot_accent_hex,
)

FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"


def _make_synthetic_video(out_path: Path, duration: float = 2.4, size: str = "640x360", fps: int = 15) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}:rate={fps}:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=44100:duration={duration}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def _make_warm_test_photo(out_path: Path, width: int = 320, height: int = 240) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), (225, 160, 95))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, width - 40, height - 40), fill=(210, 140, 80))
    img.save(out_path, quality=92)
    return out_path


def _assert_hdr_rgba_clamped(rgba_img: Image.Image) -> None:
    arr = np.asarray(rgba_img.convert("RGBA"), dtype=np.uint8)
    alpha = arr[:, :, 3]
    vis_mask = alpha > 16
    assert np.any(vis_mask), "Expected visible title pixels"
    peak_rgb = int(np.max(arr[:, :, :3][vis_mask]))
    assert peak_rgb <= HDR_DIFFUSE_WHITE_PEAK_RGB, f"Peak RGB {peak_rgb} exceeded HDR limit {HDR_DIFFUSE_WHITE_PEAK_RGB}"

    rgb_sum = arr[:, :, 0].astype(np.int16) + arr[:, :, 1].astype(np.int16) + arr[:, :, 2].astype(np.int16)
    dark_shadow = (rgb_sum <= 60) & (alpha > 0)
    if np.any(dark_shadow):
        max_shadow_a = int(np.max(alpha[dark_shadow]))
        assert max_shadow_a <= HDR_MAX_SHADOW_ALPHA, f"Dark shadow alpha {max_shadow_a} exceeded {HDR_MAX_SHADOW_ALPHA}"


def test_all_themes_and_archetypes_render_and_hdr_clamp(tmp_path: Path) -> None:
    """Every combination of the 6 Cinema Themes x 11 Archetypes renders cleanly and obeys HDR luminance/shadow caps."""
    for theme in VALID_TITLE_THEMES:
        for style in VALID_TITLE_STYLES:
            item = TitleOverlayItem(
                id=f"ov_{theme}_{style}",
                style=style,  # type: ignore[arg-type]
                theme=theme,  # type: ignore[arg-type]
                motion_preset="stencil_reveal" if style == "knockout_stencil_opening" else "masked_rise",
                text="Golden Hour Courtyard",
                subtitle="Curated Story Chapter",
                overline="CHAPTER I",
                chapter_numeral="III",
                colophon_lines=["4 CHAPTERS · 12 CUTS", "OPTICS · MULTI-CAM 4K"],
                timeline_start=0.5,
                duration=3.0,
                map_waypoints=[
                    {"label": "Gate", "completed": True, "active": False},
                    {"label": "Gallery", "completed": True, "active": True},
                    {"label": "Garden", "completed": False, "active": False},
                ],
                backdrop_blur=(theme == "apple_tv_glass"),
                hdr_safe=True,
            )
            out_png = tmp_path / f"{theme}_{style}.png"
            render_title_png(item, out_png, width=320, height=180, hdr_safe=True)
            assert out_png.exists() and out_png.stat().st_size > 100
            with Image.open(out_png) as im:
                _assert_hdr_rgba_clamped(im)

    # Verify kinetic keyframe progression & HDR clamping across t_norm = (0.15, 0.55, 0.90)
    for style, preset in (
        ("knockout_stencil_opening", "stencil_reveal"),
        ("curved_gps_ribbon", "bezier_route_draw"),
        ("magazine_chapter_plate", "masked_rise"),
        ("credits_colophon_card", "tracking_expand"),
        ("exif_stamp", "typewriter_telemetry"),
    ):
        ov = TitleOverlayItem(
            id=f"kin_{style}",
            style=style,  # type: ignore[arg-type]
            theme="a24_minimal",
            motion_preset=preset,  # type: ignore[arg-type]
            text="Morning Promenade",
            subtitle="Light & Architecture",
            overline="ACT II",
            chapter_numeral="II",
            colophon_lines=["CHAPTERS · 04", "MASTER · BT.2020 HLG"],
            map_waypoints=[
                {"label": "Arrival", "completed": True, "active": False},
                {"label": "Atrium", "completed": True, "active": True},
                {"label": "Finale", "completed": False, "active": False},
            ],
            timeline_start=0.0,
            duration=3.0,
            hdr_safe=True,
        )
        frames = [render_title_frame(ov, width=480, height=270, hdr_safe=True, t_norm=tn) for tn in (0.15, 0.55, 0.90)]
        for f in frames:
            _assert_hdr_rgba_clamped(f)
        arr_early = np.asarray(frames[0], dtype=np.int16)
        arr_mid = np.asarray(frames[1], dtype=np.int16)
        assert float(np.mean(np.abs(arr_mid - arr_early))) > 0.1, f"{style} did not animate between t=0.15 and t=0.55"


def test_optical_tracking_mixed_cjk_latin_and_knockout_stencil_alpha(tmp_path: Path) -> None:
    """Verify monotonic tracking widths, dynamic CJK+Latin rendering, shot accent sampling, and stencil alpha cutouts."""
    dummy = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    font = _load_font_tier(32, tier="sans", bold=True)
    w0, _ = measure_tracked_text(draw, "ATELIER", font, letter_spacing_px=0.0)
    w6, _ = measure_tracked_text(draw, "ATELIER", font, letter_spacing_px=6.0)
    w12, _ = measure_tracked_text(draw, "ATELIER", font, letter_spacing_px=12.0)
    assert w0 < w6 < w12

    cjk_phrase = "".join(chr(c) for c in (0x65C5, 0x884C, 0x8BB0, 0x5FC6))
    mixed_text = f"{cjk_phrase} · Cinema Title"
    wm, hm = measure_tracked_text(draw, mixed_text, font, letter_spacing_px=2.5)
    assert wm > 40 and hm > 10

    warm_photo = _make_warm_test_photo(tmp_path / "warm_sunset.jpg")
    sampled_hex = sample_shot_accent_hex(str(warm_photo), fallback_hex="#DCE6F2")
    assert len(sampled_hex) == 7 and sampled_hex.startswith("#")
    assert sampled_hex != "#DCE6F2"

    stencil = TitleOverlayItem(
        id="stencil_check",
        style="knockout_stencil_opening",
        theme="a24_minimal",
        motion_preset="stencil_reveal",
        text=mixed_text,
        subtitle="Curated Story",
        overline="OPENING",
        timeline_start=0.0,
        duration=3.0,
        hdr_safe=True,
    )
    frame = render_title_frame(stencil, width=640, height=360, hdr_safe=True, t_norm=0.55)
    arr = np.asarray(frame, dtype=np.uint8)
    _assert_hdr_rgba_clamped(frame)

    alpha = arr[:, :, 3]
    center_band = alpha[int(360 * 0.25) : int(360 * 0.72), int(640 * 0.12) : int(640 * 0.88)]
    assert np.any(center_band == 0), "Expected transparent knocked-out glyph pixels inside matte"
    assert np.any(alpha >= 180), "Expected opaque stencil matte surrounding knocked-out title"


def test_kinetic_mov_live_frosted_glass_and_karaoke_ass_render_end_to_end(tmp_path: Path) -> None:
    """Full render_timeline with kinetic qtrle overlay, live Gaussian frosted glass, and burned-in \\kf ASS subtitles."""
    clip_path = _make_synthetic_video(tmp_path / "src_colorful.mp4", duration=2.4, size="640x360", fps=15)
    cjk_sub = "".join(chr(c) for c in (0x6E05, 0x6668, 0x91D1, 0x8272, 0x9633, 0x5149))

    tl = EditTimeline(
        project_name="Kinetic Render Test",
        settings=TimelineSettings(width=640, height=360, fps=15, hdr=False),
        visual_track=[
            VisualTrackItem(
                id="v1_01",
                type="video",
                roll_type="a_roll",
                source=str(clip_path),
                source_in=0.0,
                source_out=2.2,
                duration=2.2,
                timeline_start=0.0,
                volume=0.8,
            )
        ],
        title_track=[
            TitleOverlayItem(
                id="ov_glass",
                style="lower_third",
                theme="apple_tv_glass",
                motion_preset="masked_rise",
                text="Chapter I: Conservatory",
                subtitle="Morning Courtyard",
                overline="ACT I",
                timeline_start=0.15,
                duration=1.8,
                backdrop_blur=True,
                backdrop_blur_sigma=16.0,
                hdr_safe=False,
            )
        ],
        subtitles=SubtitleConfig(
            enabled=True,
            burn_in=True,
            style_preset="kinetic_word_highlight",
            theme="kinetic_travelogue",
            font_name=resolve_subtitle_font_stack()[0],
            items=[
                SubtitleItem(
                    id="sub_01",
                    start=0.25,
                    end=1.85,
                    text="Golden morning light",
                    secondary_text=cjk_sub,
                    words=[
                        WordTimestamp(word="Golden", start=0.25, end=0.75),
                        WordTimestamp(word="morning", start=0.75, end=1.30),
                        WordTimestamp(word="light", start=1.30, end=1.85),
                    ],
                )
            ],
        ),
    )

    out_mp4 = tmp_path / "kinetic_test_out.mp4"
    rendered = render_timeline(tl, out_mp4)
    assert isinstance(rendered, dict)
    assert out_mp4.exists() and out_mp4.stat().st_size > 1000

    ass_path = out_mp4.with_suffix(".ass")
    srt_path = out_mp4.with_suffix(".srt")
    assert ass_path.exists() and srt_path.exists()
    ass_text = ass_path.read_text(encoding="utf-8")
    assert r"\kf" in ass_text

    # Decode two timestamps inside the overlay + subtitle window and confirm live motion
    png_early = tmp_path / "frame_early.png"
    png_mid = tmp_path / "frame_mid.png"
    subprocess.run(
        [FFMPEG_BIN, "-y", "-ss", "0.28", "-i", str(out_mp4), "-frames:v", "1", str(png_early)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [FFMPEG_BIN, "-y", "-ss", "1.15", "-i", str(out_mp4), "-frames:v", "1", str(png_mid)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    arr1 = np.asarray(Image.open(png_early).convert("RGB"), dtype=np.int16)
    arr2 = np.asarray(Image.open(png_mid).convert("RGB"), dtype=np.int16)
    assert arr1.shape == (360, 640, 3) and arr2.shape == (360, 640, 3)
    assert float(np.mean(np.abs(arr2 - arr1))) > 0.2


def test_motion_graphics_agent_qc_intake_and_fcpxml_integration(tmp_path: Path) -> None:
    """Verify intake recommendation (still 8 questions), MotionGraphicsTitleAgent archetypes, VisualQC, Director & FCPXML."""
    media_dir = tmp_path / "media"
    proj_dir = tmp_path / "project"
    media_dir.mkdir(parents=True, exist_ok=True)
    proj_dir.mkdir(parents=True, exist_ok=True)

    vid1 = _make_synthetic_video(media_dir / "arrival.mp4", duration=3.0)
    vid2 = _make_synthetic_video(media_dir / "dialogue.mp4", duration=3.0)
    photo1 = _make_warm_test_photo(media_dir / "courtyard.jpg", width=640, height=480)

    questions = generate_intake_questions(media_dir)
    assert len(questions) == 8

    bb_intake = PipelineBlackboard(
        media_dir=media_dir,
        project_dir=proj_dir,
        user_prompt="Create a warm Criterion Atelier documentary film and burn in subtitles",
        interview_answers={},
    )
    ClientIntakeAgent().run(bb_intake)
    assert bb_intake.creative_brief is not None
    assert bb_intake.creative_brief.title_theme == "criterion_atelier"
    assert bb_intake.creative_brief.burn_in_subtitles is True
    assert "recommended_title_theme" in bb_intake.creative_brief.director_recommendations

    # Now verify full Track V3 orchestration on apple_tv_glass + hook + 4 chapters
    brief = CreativeBrief(
        project_title="Afternoon_Botanical_Walk",
        raw_user_prompt="Modern frosted glass travel diary",
        title_theme="apple_tv_glass",
        burn_in_subtitles=False,
        subtitle_style="minimal_glass",
    )
    catalog = MediaCatalog(
        source_directory=str(media_dir),
        created_at="2026-06-01T12:00:00Z",
        videos=[
            VideoAsset(id="v1", path=str(vid1), filename="arrival.mp4", duration=3.0, gps_lat=48.85, gps_lon=2.35),
            VideoAsset(id="v2", path=str(vid2), filename="dialogue.mp4", duration=3.0),
        ],
        photos=[
            PhotoAsset(
                id="p1",
                path=str(photo1),
                preview_path=str(photo1),
                filename="courtyard.jpg",
                camera_model="SONY ILCE",
            )
        ],
    )
    media_index = MediaIndex(
        project_title=brief.project_title,
        source_directory=str(media_dir),
        created_at="2026-06-01T12:00:00Z",
        total_raw_assets=3,
        subclips=[
            CulledSubClip(
                subclip_id="sc_v2",
                source_asset_id="v2",
                source_path=str(vid2),
                filename="dialogue.mp4",
                media_type="video",
                normalized_timestamp="2026-06-01T12:05:00Z",
                in_tc=0.0,
                out_tc=3.0,
                duration=3.0,
                roll_type="a_roll",
                spoken_text="Look at the sunlight",
                words=[
                    WordTimestamp(word="Look", start=0.2, end=0.5),
                    WordTimestamp(word="at", start=0.5, end=0.7),
                    WordTimestamp(word="the", start=0.7, end=0.9),
                    WordTimestamp(word="sunlight", start=0.9, end=1.4),
                ],
            )
        ],
    )
    paper = PaperEdit(
        project_title=brief.project_title,
        story_arc_summary="Four-chapter chronological botanical walk.",
        chapters=[
            ChapterPlan(chapter_id="c1", title="Act I: Gatehouse", narrative_goal="Arrival", location_tag="South Gate"),
            ChapterPlan(chapter_id="c2", title="Act II: Glasshouse", narrative_goal="Discovery", location_tag="Palm Dome"),
            ChapterPlan(chapter_id="c3", title="Act III: Terrace", narrative_goal="Midday Pause", location_tag="Stone Steps"),
            ChapterPlan(chapter_id="c4", title="Act IV: Evening Light", narrative_goal="Finale", location_tag="West Lawn"),
        ],
    )
    tl = EditTimeline(
        project_name=brief.project_title,
        settings=TimelineSettings(width=1920, height=1080, fps=30, hdr=True),
        visual_track=[
            VisualTrackItem(id="clip_000_hook", type="video", act="Act I: Gatehouse", source=str(vid1), timeline_start=0.0, duration=3.0),
            VisualTrackItem(id="v1_01", type="photo", roll_type="b_roll", act="Act I: Gatehouse", source=str(photo1), timeline_start=3.0, duration=4.0),
            VisualTrackItem(id="v1_02a", type="photo", roll_type="b_roll", act="Act II: Glasshouse", source=str(photo1), timeline_start=7.0, duration=4.0),
            VisualTrackItem(id="v1_02b", type="video", roll_type="a_roll", act="Act II: Glasshouse", source=str(vid2), timeline_start=11.0, duration=3.0, source_in=0.0, source_out=3.0),
            VisualTrackItem(id="v1_03", type="photo", roll_type="b_roll", act="Act III: Terrace", source=str(photo1), timeline_start=14.0, duration=7.0),
            VisualTrackItem(id="v1_04", type="video", roll_type="b_roll", act="Act IV: Evening Light", source=str(vid1), timeline_start=21.0, duration=7.0),
        ],
        subtitles=SubtitleConfig(
            enabled=True,
            items=[
                # Aligned with v1_02b (11.0..14.0) words at source 0.2..1.4s -> master 11.2..12.4s
                SubtitleItem(id="sub_dialogue", start=11.2, end=12.4, text="Look at the sunlight", speaker="Narrator")
            ],
        ),
    )

    bb = PipelineBlackboard(media_dir=media_dir, project_dir=proj_dir)
    bb.creative_brief = brief
    bb.catalog = catalog
    bb.media_index = media_index
    bb.paper_edit = paper
    bb.timeline = tl
    bb.pacing_audit = PacingAuditReport(project_title=brief.project_title, passed_compliance=True)

    MotionGraphicsTitleAgent().run(bb)

    styles_emitted = {ov.style for ov in tl.title_track}
    assert "knockout_stencil_opening" in styles_emitted
    assert "curved_gps_ribbon" in styles_emitted
    assert "magazine_chapter_plate" in styles_emitted
    assert "credits_colophon_card" in styles_emitted
    assert tl.subtitles.theme == "apple_tv_glass"
    assert tl.subtitles.style_preset == "minimal_glass"
    assert tl.subtitles.burn_in is False
    assert bb.graphics_manifest is not None
    assert bb.graphics_manifest.title_theme == "apple_tv_glass"
    assert len(tl.subtitles.items[0].words) >= 2

    # Stage 55 Visual QC must pass with zero overlay overlaps
    qc_summary = ProxyWatchdownQCAgent().run(bb)
    assert qc_summary["overlay_overlap_count"] == 0

    # Stage 60 Production Director must recognize knockout_stencil_opening + credits_colophon_card
    ProductionDirectorAgent().run(bb)
    assert bb.director_review is not None
    assert bb.director_review.approved_for_master_render is True
    assert bb.director_review.rubric_scores.get("story_arc", 0.0) >= 8.0

    # FCPXML export carries theme font and structured colophon/overline metadata
    fcpxml_out = proj_dir / "timeline.fcpxml"
    export_fcpxml(tl, fcpxml_out)
    xml_text = fcpxml_out.read_text(encoding="utf-8")
    ET.fromstring(xml_text)
    expected_font = str(get_theme_spec("apple_tv_glass")["fcpxml_font"])
    assert expected_font in xml_text
    assert "CHAPTER III" in xml_text
    assert "END CREDITS" in xml_text
