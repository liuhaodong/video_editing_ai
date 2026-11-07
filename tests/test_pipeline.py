"""Unit and integration tests for the AI Photo & Video Editing Tool and Modular Multi-Agent Architecture."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from video_editing_ai.agents import (
    AgentRegistry,
    BaseEditingAgent,
    PipelineBlackboard,
    register_agent,
    run_multi_agent_pipeline,
)
from video_editing_ai.analyzer.ag_annotator import analyze_media_directory
from video_editing_ai.exporters.nle_export import export_all_nle_formats
from video_editing_ai.models.schema import EffectTrackItem, TitleOverlayItem
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


def test_modular_agent_registry_and_custom_plugin(tmp_path: Path, sample_media_dir: Path) -> None:
    """Verify that future sub-agent roles can be dynamically registered and plugged into Stage 40."""
    role_ids = [cls.role_id for cls in AgentRegistry.list_all()]
    for expected in (
        "client_intake",
        "music_supervisor",
        "culler",
        "photo_curator",
        "audio_vibe",
        "story_producer",
        "dialogue_editor",
        "montage_editor",
        "vfx_designer",
        "showrunner",
        "production_director",
    ):
        assert expected in role_ids

    @register_agent
    class CustomColoristPluginAgent(BaseEditingAgent):
        role_id = "custom_colorist"
        display_name = "4b. Custom HDR Colorist (Test Plugin)"
        stage_order = 40
        description = "Test pluggable color grading agent."
        enabled_by_default = False
        requires_artifacts = ["paper_edit.json"]
        produces_artifacts = ["color_lut_manifest.json"]

        def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
            if blackboard.timeline and blackboard.timeline.visual_track:
                first_clip = blackboard.timeline.visual_track[0]
                blackboard.timeline.effects_track.append(
                    EffectTrackItem(
                        id="fx_custom_color_01",
                        effect_type="warm_glow",
                        target_clip_id=first_clip.id,
                        timeline_start=first_clip.timeline_start,
                        duration=first_clip.duration,
                        intensity=0.5,
                        parameters={"saturation": 1.10, "brightness": 0.01},
                        created_by_agent=self.role_id,
                    )
                )
            p = blackboard.set_custom_artifact("color_lut_manifest.json", {"lut": "Rec2020_Warm_Golden"})
            blackboard.send_message(self.role_id, "showrunner", "Applied custom HDR look", {"path": str(p)})
            return {"color_manifest": str(p)}

    try:
        proj_dir = tmp_path / "plugin_test_proj"
        bb = run_multi_agent_pipeline(
            media_dir=sample_media_dir,
            project_dir=proj_dir,
            user_prompt="Warm family adventure with custom color grading",
            enabled_optional_roles=["vfx_designer", "custom_colorist"],
            width=640,
            height=360,
            fps=24,
            hdr=False,
            use_antigravity=False,
            whisper_model="base",
            render_video=False,
        )
        executed_ids = [a["role_id"] for a in bb.executed_agents]
        assert "vfx_designer" in executed_ids
        assert "custom_colorist" in executed_ids
        assert (proj_dir / "vfx_manifest.json").exists()
        assert (proj_dir / "color_lut_manifest.json").exists()
        assert bb.timeline is not None
        assert any(fx.created_by_agent == "custom_colorist" for fx in bb.timeline.effects_track)
    finally:
        AgentRegistry.unregister("custom_colorist")


def test_six_agent_coordinated_vlog_pipeline(tmp_path: Path, sample_media_dir: Path) -> None:
    """Verify full Stage 0 -> Stage 50 multi-agent coordination, JSON contracts, and Web API endpoints."""
    proj_dir = tmp_path / "multi_agent_proj"
    bb = run_multi_agent_pipeline(
        media_dir=sample_media_dir,
        project_dir=proj_dir,
        user_prompt="Create a heartwarming family vlog with rides and portraits",
        extra_audio_paths=[sample_media_dir / "bgm.wav"],
        interview_answers={
            "narrative_style": "family_keepsake_4act",
            "dialogue_preference": "keep_full_sentences_with_broll_cutaways",
            "photo_mode": "gallery_frame",
            "target_duration_sec": "60",
            "must_include_topics": "rides, portraits",
        },
        enabled_optional_roles=["vfx_designer"],
        width=640,
        height=360,
        fps=24,
        hdr=False,
        use_antigravity=False,
        whisper_model="base",
        render_video=True,
    )

    for contract_name in (
        "creative_brief.json",
        "bgm_selection.json",
        "media_index.json",
        "photo_curation.json",
        "beat_grid.json",
        "paper_edit.json",
        "vfx_manifest.json",
        "pacing_audit.json",
        "director_review.json",
        "blackboard.json",
        "timeline.yaml",
        "storyboard.md",
        "final_story.mp4",
    ):
        assert (proj_dir / contract_name).exists(), f"Missing {contract_name}"

    assert bb.creative_brief is not None
    assert bb.bgm_selection is not None
    assert bb.media_index is not None
    assert bb.photo_curation is not None
    assert bb.beat_grid is not None
    assert bb.paper_edit is not None
    assert bb.pacing_audit is not None
    assert bb.director_review is not None
    assert bb.director_review.approved_for_master_render is True
    assert bb.director_review.overall_quality_score >= 9.0
    assert bb.timeline is not None
    assert len(bb.messages) >= 5

    # Test Web Studio multi-agent endpoints
    app = create_app(default_project_dir=proj_dir)
    client = TestClient(app)

    r_agents = client.get("/api/agents")
    assert r_agents.status_code == 200
    assert len(r_agents.json()["agents"]) >= 8

    r_q = client.get("/api/intake-questions", params={"media_dir": str(sample_media_dir)})
    assert r_q.status_code == 200
    q_body = r_q.json()
    assert len(q_body["questions"]) >= 5
    assert q_body["library"]["video_count"] == 1
    assert q_body["library"]["photo_count"] == 2

    r_libs = client.get("/api/fs/libraries")
    assert r_libs.status_code == 200
    assert "libraries" in r_libs.json()

    r_insp = client.get("/api/fs/inspect", params={"media_dir": str(sample_media_dir)})
    assert r_insp.status_code == 200
    assert r_insp.json()["video_count"] == 1

    r_browse = client.get("/api/fs/browse", params={"path": str(sample_media_dir.parent)})
    assert r_browse.status_code == 200
    assert any(sub["name"] == "raw_media" for sub in r_browse.json()["subdirectories"])

    # Verify live interactive intake refinement endpoint (POST /api/intake/refine) and >2m duration support
    r_refine = client.post(
        "/api/intake/refine",
        json={
            "media_dir": str(sample_media_dir),
            "user_message": "Make it a 3 minutes extended documentary with classical piano music and 2.5D parallax",
            "interview_answers": {},
            "chat_history": [],
        },
    )
    assert r_refine.status_code == 200
    ref_body = r_refine.json()
    assert "agent_reply" in ref_body
    assert "director_note" in ref_body
    assert ref_body["draft_brief"]["target_duration_sec"] == 180.0
    assert ref_body["updated_answers"]["bgm_mood"] == "classical_chamber"
    assert len(ref_body["suggested_chips"]) >= 3

    # Verify live job start + status tracking endpoint with agent_activities & director_thoughts
    job_proj = tmp_path / "async_job_proj"
    r_job = client.post(
        "/api/jobs/start?wait=true",
        json={
            "media_dir": str(sample_media_dir),
            "project_dir": str(job_proj),
            "user_prompt": "Quick family highlight reel",
            "interview_answers": {
                "narrative_style": "family_keepsake_4act",
                "photo_mode": "parallax_gallery",
                "target_duration_sec": "155",
            },
            "enabled_optional_roles": ["vfx_designer"],
            "width": 640,
            "height": 360,
            "fps": 24,
            "hdr": False,
            "use_antigravity": False,
            "render_video": False,
        },
    )
    assert r_job.status_code == 200
    job_data = r_job.json()
    assert job_data["status"] == "completed"
    assert job_data["progress_pct"] == 100
    assert len(job_data["stages"]) >= 10
    assert len(job_data["activity_log"]) >= 5
    assert len(job_data["agent_activities"]) >= 10
    assert len(job_data["director_thoughts"]) >= 8
    assert any(th["observation"] and th["thinking"] for th in job_data["director_thoughts"])

    r_job_status = client.get("/api/jobs/status")
    assert r_job_status.status_code == 200
    assert r_job_status.json()["status"] == "completed"

    r_state = client.get("/api/state", params={"project_dir": str(proj_dir)})
    assert r_state.status_code == 200
    st = r_state.json()
    assert st["agent_contracts"]["creative_brief"] is not None
    assert st["agent_contracts"]["pacing_audit"] is not None
    assert st["agent_contracts"]["director_review"] is not None
    assert st["agent_contracts"]["vfx_manifest"] is not None
    assert st["project_summary"]["has_rendered_video"] is True

    # Verify dedicated newly-rendered video download endpoint
    r_dl = client.get("/api/projects/download-video", params={"project_dir": str(proj_dir)})
    assert r_dl.status_code == 200
    assert "attachment" in r_dl.headers.get("content-disposition", "").lower()
    assert "no-store" in r_dl.headers.get("cache-control", "").lower()
    assert len(r_dl.content) > 1000

    # Verify project management endpoints (list, create, load, reset)
    r_projs = client.get("/api/projects")
    assert r_projs.status_code == 200
    assert "projects" in r_projs.json()

    custom_proj_dir = tmp_path / "custom_trip_project"
    r_create = client.post(
        "/api/projects/create",
        json={
            "project_name": "Custom Trip",
            "media_dir": str(sample_media_dir),
            "project_dir": str(custom_proj_dir),
        },
    )
    assert r_create.status_code == 200
    assert (custom_proj_dir / "project_meta.json").exists()

    r_load = client.post("/api/projects/load", json={"project_dir": str(proj_dir)})
    assert r_load.status_code == 200
    assert r_load.json()["project"]["has_rendered_video"] is True

    assert (job_proj / "timeline.yaml").exists()
    r_reset = client.post("/api/projects/reset", json={"project_dir": str(job_proj)})
    assert r_reset.status_code == 200
    assert r_reset.json()["removed_files_count"] >= 5
    assert not (job_proj / "timeline.yaml").exists()
    assert (job_proj / "project_meta.json").exists()
