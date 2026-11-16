"""Tests for Live Photo video filtering and jitter-free high-contrast Track V3 overlays over video."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from video_editing_ai.agents.base import PipelineBlackboard
from video_editing_ai.agents.showrunner_agent import ShowrunnerAgent
from video_editing_ai.analyzer.live_photo_filter import is_live_photo_video
from video_editing_ai.models.schema import (
    CreativeBrief,
    EditTimeline,
    EffectTrackItem,
    PaperEdit,
    ChapterPlan,
    TimelineSettings,
    TitleOverlayItem,
    VisualTrackItem,
)
from video_editing_ai.agents.plugins.colorist_agent import HDRColoristAgent
from video_editing_ai.renderer.ffmpeg_composer import _build_color_grade_filter, render_timeline
from video_editing_ai.renderer.titles import THEME_SPECS, render_title_frame

FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"


def _make_moving_video(out_path: Path, duration: float = 3.2, size: str = "640x360", fps: int = 15) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            FFMPEG_BIN,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
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
        ],
        check=True,
    )
    return out_path


def test_live_photo_and_motion_photo_videos_rejected(tmp_path: Path) -> None:
    """Reject extracted Motion Photo MP4s & paired short Live Photo MOVs while keeping normal clips & pano pans."""
    mp_dir = tmp_path / "cache" / "motion_photos"
    mp_dir.mkdir(parents=True)
    extracted = mp_dir / "mp_092022_mirrors.mp4"
    extracted.write_bytes(b"fake")
    pano_clip = mp_dir / "pano_etoile_boulevards_4k.mp4"
    pano_clip.write_bytes(b"fake")

    assert is_live_photo_video(extracted) is True
    assert is_live_photo_video(tmp_path / "mp_123456.mp4") is True
    assert is_live_photo_video(tmp_path / "MVIMG_20260421_1103.mp4") is True
    assert is_live_photo_video(pano_clip) is False

    # Same-stem iPhone/Android Live Photo companion MOV paired with HEIC/JPG
    heic = tmp_path / "IMG_1024.HEIC"
    live_mov = tmp_path / "IMG_1024.MOV"
    heic.write_bytes(b"fake")
    live_mov.write_bytes(b"fake")
    assert is_live_photo_video(live_mov, sibling_filenames={"IMG_1024.HEIC", "IMG_1024.MOV"}, duration_sec=2.4) is True
    assert is_live_photo_video(tmp_path / "A7C2_0673.MP4", sibling_filenames={"A7C2_0673.MP4"}, duration_sec=28.0) is False


def test_render_timeline_and_showrunner_block_live_photo_video(tmp_path: Path) -> None:
    """Both ShowrunnerAgent and render_timeline reject Live Photo / Motion Photo videos on the timeline."""
    lp_vid = _make_moving_video(tmp_path / "motion_photos" / "mp_092022_blurry.mp4", duration=2.0)
    tl = EditTimeline(
        project_name="live_photo_guard",
        user_prompt="test",
        story_arc_summary="test",
        settings=TimelineSettings(width=640, height=360, fps=15, hdr=False),
        visual_track=[
            VisualTrackItem(
                id="clip_live_bad",
                type="video",
                source=str(lp_vid),
                track_lane=1,
                roll_type="b_roll",
                act="Act I",
                timeline_start=0.0,
                duration=2.0,
                source_in=0.0,
                source_out=2.0,
            )
        ],
    )

    with pytest.raises(ValueError, match="Live Photo / Motion Photo"):
        render_timeline(tl, tmp_path / "out.mp4", work_dir=tmp_path / "cache")

    proj_dir = tmp_path / "proj"
    proj_dir.mkdir(parents=True)
    tl.to_yaml(proj_dir / "timeline.yaml")
    bb = PipelineBlackboard(project_dir=proj_dir, media_dir=tmp_path, user_prompt="test", use_antigravity=False)
    bb.timeline = tl
    bb.creative_brief = CreativeBrief(
        target_duration_sec=30.0,
        story_theme="test",
        narrative_tone="calm",
        must_include_topics=[],
    )
    bb.paper_edit = PaperEdit(
        project_title="test",
        story_arc_summary="test",
        estimated_total_duration=2.0,
        chapters=[
            ChapterPlan(
                chapter_id="ch1",
                title="Act I",
                narrative_goal="test",
                target_duration_sec=2.0,
                selected_subclip_ids=[],
            )
        ],
    )
    res = ShowrunnerAgent().run(bb)
    assert res["passed_brief_compliance"] is False
    assert any(e.rule_name == "prohibit_live_photo_video" for e in bb.pacing_audit.entries)


def test_title_overlay_zero_steady_state_jitter_and_dark_glass_contrast() -> None:
    """Steady-state Track V3 frames are bit-identical with deep smoked-glass alpha for crisp contrast."""
    for th in THEME_SPECS:
        assert THEME_SPECS[th]["glass_tint"][3] >= 145, f"Theme {th} glass_tint too translucent over video"

    ov = TitleOverlayItem(
        id="title_lt_07",
        style="lower_third",
        theme="criterion_atelier",
        motion_preset="masked_rise",
        overline="PROMENADE ROYALE",
        text="Allée Royale · Bosquets",
        subtitle="Gliding past manicured hedges & marble vases along the central walk",
        timeline_start=0.2,
        duration=3.2,
        fade_in=0.30,
        fade_out=0.30,
        backdrop_blur=True,
        hdr_safe=True,
    )
    f_hold_1 = render_title_frame(ov, width=1280, height=720, t_norm=0.42, hdr_safe=True, transparent_glass_body=True)
    f_hold_2 = render_title_frame(ov, width=1280, height=720, t_norm=0.55, hdr_safe=True, transparent_glass_body=True)
    f_hold_3 = render_title_frame(ov, width=1280, height=720, t_norm=0.72, hdr_safe=True, transparent_glass_body=True)
    a1 = np.asarray(f_hold_1, dtype=np.int16)
    a2 = np.asarray(f_hold_2, dtype=np.int16)
    a3 = np.asarray(f_hold_3, dtype=np.int16)
    assert float(np.max(np.abs(a2 - a1))) == 0.0
    assert float(np.max(np.abs(a3 - a2))) == 0.0


@pytest.mark.parametrize("hdr_mode", [True, False])
def test_video_lower_third_yuv444_backdrop_blur_renders_cleanly(tmp_path: Path, hdr_mode: bool) -> None:
    """Track V3 lower_third with frosted glass renders cleanly onto video in both HDR and SDR modes."""
    vid = _make_moving_video(tmp_path / f"src_{'hdr' if hdr_mode else 'sdr'}.mp4", duration=3.0)
    ov = TitleOverlayItem(
        id="title_lt_test",
        style="lower_third",
        theme="criterion_atelier",
        motion_preset="masked_rise",
        overline="PROMENADE ROYALE",
        text="Allée Royale · Bosquets",
        subtitle="Crisp typography over moving footage",
        timeline_start=0.2,
        duration=2.5,
        fade_in=0.25,
        fade_out=0.25,
        backdrop_blur=True,
        hdr_safe=hdr_mode,
    )
    tl = EditTimeline(
        project_name="overlay_qc",
        user_prompt="test",
        story_arc_summary="test",
        settings=TimelineSettings(width=640, height=360, fps=15, hdr=hdr_mode),
        visual_track=[
            VisualTrackItem(
                id="clip_001_vid",
                type="video",
                source=str(vid),
                track_lane=1,
                roll_type="b_roll",
                act="Act III",
                timeline_start=0.0,
                duration=3.0,
                source_in=0.0,
                source_out=3.0,
            )
        ],
        title_track=[ov],
    )
    out_mp4 = tmp_path / f"rendered_{'hdr' if hdr_mode else 'sdr'}.mp4"
    res = render_timeline(tl, out_mp4, work_dir=tmp_path / "cache")
    assert res["video"].exists() and res["video"].stat().st_size > 20_000


def test_hlg_highlight_rolloff_targets_hot_hlg_only() -> None:
    """Verify 10-bit HLG highlight roll-off activates for blown-out HLG clips and spares normal clips."""
    fx_hot = EffectTrackItem(
        id="fx_hot",
        effect_type="color_grade",
        target_clip_id="c1",
        timeline_start=0.0,
        duration=2.0,
        parameters={"rs": 0.02, "gs": 0.0, "bs": -0.02, "contrast": 1.0, "saturation": 1.04, "gamma": 1.02, "hlg_rolloff": True},
    )
    fx_normal = EffectTrackItem(
        id="fx_norm",
        effect_type="color_grade",
        target_clip_id="c2",
        timeline_start=2.0,
        duration=2.0,
        parameters={"rs": 0.02, "gs": 0.0, "bs": -0.02, "contrast": 1.05, "saturation": 1.04, "gamma": 0.99, "hlg_rolloff": False},
    )
    assert "lutyuv=" in _build_color_grade_filter(fx_hot)
    assert "lutyuv=" not in _build_color_grade_filter(fx_normal)
    assert HDRColoristAgent._measure_clip_stats is not None
