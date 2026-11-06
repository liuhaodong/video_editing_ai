"""Unit and integration tests for the AI Photo & Video Editing Tool."""

from __future__ import annotations

from pathlib import Path
import subprocess

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from video_editing_ai.analyzer.ag_annotator import analyze_media_directory
from video_editing_ai.exporters.nle_export import export_all_nle_formats
from video_editing_ai.models.schema import EditTimeline, MediaCatalog, TitleOverlayItem
from video_editing_ai.planner.story_director import generate_edit_plan
from video_editing_ai.renderer.ffmpeg_composer import render_timeline
from video_editing_ai.renderer.ken_burns import build_gallery_frame_image, render_photo_clip
from video_editing_ai.renderer.titles import render_title_png
from video_editing_ai.web.app import create_app


@pytest.fixture()
def sample_media_dir(tmp_path: Path) -> Path:
    """Create a synthetic media folder with 1 test video, 2 test photos, and 1 BGM audio file."""
    media = tmp_path / "raw_media"
    media.mkdir(parents=True, exist_ok=True)

    # Generate a 3-second test video with audio tone using FFmpeg
    vid_path = media / "VID_001.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "3.0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(vid_path),
        ],
        check=True,
    )

    # Create 1 portrait photo and 1 landscape photo
    p1 = Image.new("RGB", (600, 900), (40, 120, 200))
    d1 = ImageDraw.Draw(p1)
    d1.rectangle([150, 250, 450, 650], fill=(255, 210, 60))
    p1.save(media / "IMG_002.jpg", "JPEG")

    p2 = Image.new("RGB", (960, 540), (180, 70, 90))
    d2 = ImageDraw.Draw(p2)
    d2.ellipse([300, 120, 660, 420], fill=(255, 240, 200))
    p2.save(media / "IMG_003.jpg", "JPEG")

    # Create a 6-second background audio file
    bgm_path = media / "bgm.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:sample_rate=48000",
            "-t",
            "6.0",
            str(bgm_path),
        ],
        check=True,
    )
    return media


def test_gallery_frame_and_photo_render(tmp_path: Path, sample_media_dir: Path) -> None:
    photo = sample_media_dir / "IMG_002.jpg"
    framed = build_gallery_frame_image(photo, width=640, height=360)
    assert framed.size == (640, 360)

    out_clip = tmp_path / "photo_clip.mp4"
    render_photo_clip(photo, out_clip, duration=1.5, width=640, height=360, fps=24)
    assert out_clip.exists() and out_clip.stat().st_size > 1000


def test_title_png_rendering(tmp_path: Path) -> None:
    for style in ("opening_card", "act_banner", "lower_third", "dialogue_badge", "closing_card"):
        item = TitleOverlayItem(
            id=f"t_{style}",
            style=style,  # type: ignore[arg-type]
            text="Gilroy Gardens 万圣节奇妙之旅",
            subtitle="THE GREAT BIG BOO • HIGHLIGHTS",
            timeline_start=0.0,
            duration=3.0,
        )
        out = render_title_png(item, tmp_path / f"{style}.png", width=640, height=360)
        assert out.exists() and out.stat().st_size > 500


def test_end_to_end_pipeline_and_web_api(tmp_path: Path, sample_media_dir: Path) -> None:
    proj_dir = tmp_path / "project"
    catalog = analyze_media_directory(
        media_dir=sample_media_dir,
        project_dir=proj_dir,
        whisper_model="base",
        use_antigravity=False,
    )
    assert len(catalog.videos) == 1
    assert len(catalog.photos) == 2
    assert (proj_dir / "annotations.yaml").exists()

    timeline, tl_path, sb_path = generate_edit_plan(
        catalog=catalog,
        user_prompt="Test adventure story",
        project_dir=proj_dir,
        width=640,
        height=360,
        fps=24,
        use_antigravity=False,
    )
    assert tl_path.exists()
    assert sb_path.exists()
    assert len(timeline.visual_track) == 3

    nle = export_all_nle_formats(timeline, proj_dir)
    assert nle["fcpxml"].exists()
    assert nle["edl"].exists()
    assert nle["otio"].exists()

    out_mp4 = proj_dir / "final_story.mp4"
    outputs = render_timeline(timeline, out_mp4)
    assert outputs["video"].exists() and outputs["video"].stat().st_size > 2000
    assert outputs["srt"].exists()
    assert outputs["ass"].exists()

    # Verify FastAPI Web Studio endpoints
    app = create_app(default_project_dir=proj_dir)
    client = TestClient(app)
    res = client.get("/api/state")
    assert res.status_code == 200
    body = res.json()
    assert body["has_annotations"] is True
    assert body["has_timeline"] is True
    assert body["has_rendered_video"] is True
