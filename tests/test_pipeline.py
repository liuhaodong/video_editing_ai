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


def test_antigravity_skill_cli_and_global_sync(tmp_path: Path, sample_media_dir: Path) -> None:
    """Verify the Antigravity Skill package (SKILL.md, references, scripts/ai_video_studio_skill.py, and global sync)."""
    import json
    import sys
    from video_editing_ai.cli import sync_skill_to_global
    from video_editing_ai.models.schema import EditTimeline

    installed_skill = sync_skill_to_global(dest_root=tmp_path / "global_skills")
    assert (installed_skill / "SKILL.md").exists()
    script_path = installed_skill / "scripts" / "ai_video_studio_skill.py"
    assert script_path.exists()
    assert len(list((installed_skill / "references").glob("*.md"))) >= 11

    skill_proj = tmp_path / "skill_cli_proj"

    # 1. inspect-library
    res_insp = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "inspect-library",
            "--media-dir",
            str(sample_media_dir),
            "--project-dir",
            str(skill_proj),
            "--prompt",
            "Family museum trip",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Director Recommends:" in res_insp.stdout
    assert (skill_proj / "library_inspection.json").exists()

    # 2. intake-refine
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "intake-refine",
            "--media-dir",
            str(sample_media_dir),
            "--project-dir",
            str(skill_proj),
            "--prompt",
            "Cinematic 3 minute museum keepsake with classical piano",
            "--duration",
            "180",
            "--photo-mode",
            "parallax_gallery",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (skill_proj / "creative_brief.json").exists()
    assert (skill_proj / "intake_consultation.json").exists()

    # 3. Modular run-stage (Part 1: Stages 00..15)
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "run-stage",
            "--stage",
            "client_intake,music_supervisor,culler,audio_vibe,photo_curator",
            "--project-dir",
            str(skill_proj),
            "--media-dir",
            str(sample_media_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (skill_proj / "bgm_selection.json").exists()
    assert (skill_proj / "photo_curation.json").exists()
    assert (skill_proj / "contact_sheets" / "contact_sheets_manifest.json").exists()

    # 3b. render-contact-sheets & apply-vision-curation (Feedback 1: Mechanical Sheets -> AI Curation -> Mechanical Persistence)
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "render-contact-sheets",
            "--project-dir",
            str(skill_proj),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cs_manifest = json.loads((skill_proj / "contact_sheets" / "contact_sheets_manifest.json").read_text(encoding="utf-8"))
    assert len(cs_manifest["photo_contact_sheets"]) >= 1
    assert Path(cs_manifest["photo_contact_sheets"][0]["sheet_path"]).exists()

    # 4. Modular run-stage (Part 2: Stages 20..60)
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "run-stage",
            "--stage",
            "story_producer,dialogue_editor,montage_editor,vfx_designer,showrunner,production_director",
            "--project-dir",
            str(skill_proj),
            "--media-dir",
            str(sample_media_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (skill_proj / "timeline.yaml").exists()
    assert (skill_proj / "director_review.json").exists()
    assert (skill_proj / "candidate_story_digest.json").exists()

    # Verify Feedback 2: Cold-Open Teaser Hook (clip_000_hook at 0.0s before title_opening)
    tl_before = EditTimeline.from_yaml(skill_proj / "timeline.yaml")
    assert tl_before.visual_track[0].clip_id == "clip_000_hook"
    assert tl_before.visual_track[0].timeline_start == 0.0
    assert tl_before.visual_track[0].act == "Cold Open Teaser"

    # Apply AI Agent vision curation override & verify persistence into photo_curation.json & timeline.yaml
    first_photo_name = cs_manifest["photo_contact_sheets"][0]["items"][0]["filename"]
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "apply-vision-curation",
            "--project-dir",
            str(skill_proj),
            "--overrides-json",
            json.dumps(
                [
                    {
                        "filename": first_photo_name,
                        "selected": True,
                        "hero_pick": True,
                        "caption_title": "Winged Victory of Samothrace",
                        "caption_subtitle": "Daru Staircase · Denon Wing, Louvre",
                    }
                ]
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (skill_proj / "vision_curation_applied.json").exists()

    # Apply AI Agent Story Plan override (Feedback 2: apply-story-plan)
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "apply-story-plan",
            "--project-dir",
            str(skill_proj),
            "--overrides-json",
            json.dumps(
                {
                    "cold_open_hook_caption": "Inside the Louvre · A Family Day in Paris",
                    "cold_open_rationale": "AI Agent selected opening kinetic walk toward the Pyramid.",
                }
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    pe_doc = json.loads((skill_proj / "paper_edit.json").read_text(encoding="utf-8"))
    assert pe_doc["cold_open_hook_caption"] == "Inside the Louvre · A Family Day in Paris"

    # 4b. Surgical Timeline Edit (Feedback 3: edit-cut with ripple shifting)
    old_dur = tl_before.total_duration
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "edit-cut",
            "--project-dir",
            str(skill_proj),
            "--ops-json",
            json.dumps(
                [
                    {
                        "action": "update-plaque",
                        "clip_id": "clip_000_hook",
                        "caption_title": "Cold Open · Louvre Courtyard",
                        "caption_subtitle": "Paris · 4K HDR Master",
                    },
                    {
                        "action": "trim",
                        "clip_id": "clip_000_hook",
                        "delta_sec": -0.5,
                    },
                ]
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    tl_after = EditTimeline.from_yaml(skill_proj / "timeline.yaml")
    assert tl_after.visual_track[0].caption_title == "Cold Open · Louvre Courtyard"
    assert round(tl_after.total_duration, 2) == round(old_dur - 0.5, 2)

    # 4c. Fast Segment Preview Renderer (Feedback 3: render-segment)
    seg_mp4 = skill_proj / "segments" / "preview_0_4s.mp4"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "render-segment",
            "--project-dir",
            str(skill_proj),
            "--start",
            "0",
            "--end",
            "4",
            "--fast-preview",
            "--output-mp4",
            str(seg_mp4),
            "--extract-frames",
            "1,2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert seg_mp4.exists() and seg_mp4.stat().st_size > 1000

    # 5. project reset
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "project",
            "--action",
            "reset",
            "--project-dir",
            str(skill_proj),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert not (skill_proj / "timeline.yaml").exists()
    assert (skill_proj / "project_meta.json").exists()


def test_cross_platform_macbook_pro_2019_hardware_encoding_and_filters(monkeypatch) -> None:
    """Verify 2019 MacBook Pro (Intel + AMD Radeon Pro + Apple T2 VideoToolbox / Boot Camp AMF) support."""
    from video_editing_ai.renderer import ken_burns

    # 1. Simulate 2019 MacBook Pro on macOS (hevc_videotoolbox)
    monkeypatch.setattr(ken_burns, "detect_best_video_encoder", lambda hdr=True: "hevc_videotoolbox" if hdr else "h264_videotoolbox")
    vt_hdr_args = ken_burns.get_video_encoder_args(hdr=True)
    assert "hevc_videotoolbox" in vt_hdr_args
    assert "main10" in vt_hdr_args
    assert "p010le" in vt_hdr_args

    vt_sdr_args = ken_burns.get_video_encoder_args(hdr=False)
    assert "h264_videotoolbox" in vt_sdr_args

    # 2. Simulate 2019 MacBook Pro on Windows Boot Camp (AMD Radeon Pro 5500M -> hevc_amf)
    monkeypatch.setattr(ken_burns, "detect_best_video_encoder", lambda hdr=True: "hevc_amf" if hdr else "h264_amf")
    amf_hdr_args = ken_burns.get_video_encoder_args(hdr=True)
    assert "hevc_amf" in amf_hdr_args
    assert "p010le" in amf_hdr_args

    # 3. Simulate macOS FFmpeg build without libzimg (zscale unavailable)
    monkeypatch.setattr(ken_burns, "has_ffmpeg_filter", lambda name: False)
    hdr_filters_no_zscale = ken_burns.build_hdr_color_filters("709")
    assert all("zscale" not in f for f in hdr_filters_no_zscale)
    assert "format=p010le" in hdr_filters_no_zscale
    assert any("arib-std-b67" in f for f in hdr_filters_no_zscale)

    # 4. Simulate FFmpeg build with zscale available
    monkeypatch.setattr(ken_burns, "has_ffmpeg_filter", lambda name: True)
    hdr_filters_with_zscale = ken_burns.build_hdr_color_filters("709")
    assert any("zscale" in f for f in hdr_filters_with_zscale)


def test_premium_photo_effects_modes_and_finishers(tmp_path: Path) -> None:
    """Verify all 6 new Premium Photo Presentation Modes + 4 Atmospheric Finishers and showcase sheet generator."""
    from PIL import Image, ImageDraw
    from video_editing_ai.models.schema import KenBurnsConfig
    from video_editing_ai.renderer.ken_burns import (
        _prepare_photo_scene_context,
        _render_photo_frame_from_context,
        render_photo_clip,
        render_photo_effects_showcase_sheet,
    )

    p1 = tmp_path / "photo_1.jpg"
    p2 = tmp_path / "photo_2.jpg"
    p3 = tmp_path / "photo_3.jpg"
    for idx, p in enumerate((p1, p2, p3), start=1):
        img = Image.new("RGB", (640, 480), (40 * idx, 80, 140))
        d = ImageDraw.Draw(img)
        d.rectangle([120, 90, 520, 390], fill=(235, 195, 95), outline=(255, 255, 255), width=6)
        img.save(p, quality=90)

    new_modes_and_finishers = [
        ("cinematic_depth_3d", "golden_light_leak"),
        ("triptych_cinematic", "none"),
        ("polaroid_scrapbook", "nostalgic_film_grain"),
        ("curator_loupe", "museum_spotlight"),
        ("editorial_magazine", "none"),
        ("cinema_anamorphic", "champagne_bokeh"),
    ]
    for mode_name, fx_name in new_modes_and_finishers:
        kb = KenBurnsConfig(
            mode=mode_name,  # type: ignore[arg-type]
            overlay_fx=fx_name,  # type: ignore[arg-type]
            secondary_source=str(p2),
            tertiary_source=str(p3),
            caption_title=f"Title for {mode_name}",
            caption_subtitle=f"Subtitle with {fx_name} finisher",
            camera_badge="PARIS 4K HDR",
        )
        ctx = _prepare_photo_scene_context(p1, width=480, height=270, kb=kb)
        frame = _render_photo_frame_from_context(ctx, t_norm=0.5, f_idx=15, duration=2.0, fps=24)
        assert frame.size == (480, 270)

    # Render an actual MP4 clip with cinematic_depth_3d + golden_light_leak
    out_mp4 = tmp_path / "depth_3d_clip.mp4"
    render_photo_clip(
        photo_path=p1,
        output_mp4=out_mp4,
        duration=1.0,
        width=480,
        height=270,
        fps=12,
        hdr=False,
        ken_burns=KenBurnsConfig(
            mode="cinematic_depth_3d",
            overlay_fx="golden_light_leak",
            caption_title="Eiffel Tower 3D Depth",
            caption_subtitle="Champ de Mars · Paris",
        ),
    )
    assert out_mp4.exists() and out_mp4.stat().st_size > 1000

    # Render the 10-mode showcase sheet
    showcase_jpg = tmp_path / "showcase.jpg"
    render_photo_effects_showcase_sheet(
        photo_path=p1,
        output_jpg=showcase_jpg,
        secondary_path=p2,
        tertiary_path=p3,
        tile_width=320,
        tile_height=180,
    )
    assert showcase_jpg.exists() and showcase_jpg.stat().st_size > 5000


def test_ai_creative_intelligence_and_interactive_multi_choice_intake(
    tmp_path: Path, sample_media_dir: Path
) -> None:
    """Verify interactive multi-select intake, ask_question payload, video/timeline AI overrides, and Director screening sheet."""
    from video_editing_ai.agents import (
        apply_timeline_intelligence_overrides,
        apply_video_curation_overrides,
        build_ask_question_tool_payload,
        generate_intake_questions,
        refine_intake_interactively,
        render_director_screening_sheet,
        render_video_culling_contact_sheets,
        run_multi_agent_pipeline,
    )
    from video_editing_ai.models.schema import EditTimeline, MediaIndex

    # 1. Interactive Intake Questions + Multi-Select + Recommendations + ask_question payload
    questions = generate_intake_questions(sample_media_dir, "Paris Louvre museum & family trip")
    assert len(questions) == 8
    q_by_id = {q.id: q for q in questions}
    assert q_by_id["photo_effects_suite"].is_multi_select is True
    assert q_by_id["photo_overlay_finishers"].is_multi_select is True
    assert q_by_id["must_include_topics"].is_multi_select is True
    assert len(q_by_id["photo_effects_suite"].recommended_options) >= 3
    assert q_by_id["target_duration_sec"].recommendation_rationale != ""

    ask_payload = build_ask_question_tool_payload(questions, max_questions=4)
    assert len(ask_payload) == 4
    assert any(item["is_multi_select"] for item in ask_payload)
    assert all(any(opt.startswith("(Recommended)") for opt in item["options"]) for item in ask_payload)

    # 2. Interactive Refinement with Multi-Select Photo Effects & Finishers
    ref = refine_intake_interactively(
        media_dir=sample_media_dir,
        user_message="Apply all Director recommendations with 3D depth parallax and curator loupe",
        interview_answers={
            "photo_effects_suite": "cinematic_depth_3d, curator_loupe, polaroid_scrapbook",
            "photo_overlay_finishers": "golden_light_leak, museum_spotlight",
        },
    )
    draft = ref["draft_brief"]
    assert "cinematic_depth_3d" in draft["enabled_photo_modes"]
    assert "curator_loupe" in draft["enabled_photo_modes"]
    assert "golden_light_leak" in draft["enabled_overlay_fx"]
    assert "recommended_duration_sec" in draft["director_recommendations"]
    assert len(ref["ask_question_payload"]) == 4

    # 3. Run multi-agent pipeline and verify AI Video Culling Sheet, Timeline Intelligence & Director Screening Sheet
    proj_dir = tmp_path / "ai_intel_proj"
    bb = run_multi_agent_pipeline(
        media_dir=sample_media_dir,
        project_dir=proj_dir,
        user_prompt="Paris family adventure with rich captions and transitions",
        interview_answers={
            "photo_effects_suite": "cinematic_depth_3d, curator_loupe, polaroid_scrapbook, editorial_magazine",
            "photo_overlay_finishers": "golden_light_leak, museum_spotlight",
        },
        width=640,
        height=360,
        fps=24,
        hdr=False,
        use_antigravity=False,
        whisper_model="base",
        render_video=False,
    )
    assert bb.timeline is not None
    assert bb.director_review is not None
    assert bb.director_review.screening_sheet_path is not None
    assert Path(bb.director_review.screening_sheet_path).exists()

    # Verify video culling contact sheet & AI video curation override helper
    vcs = render_video_culling_contact_sheets(proj_dir)
    assert vcs["video_clips_count"] >= 1
    assert len(vcs["contact_sheets"]) >= 1
    assert Path(vcs["contact_sheets"][0]).exists()

    mi = MediaIndex.from_json_file(proj_dir / "media_index.json")
    first_vid = next(s for s in mi.subclips if s.media_type == "video")
    vc_res = apply_video_curation_overrides(
        proj_dir,
        {
            "videos": {
                first_vid.subclip_id: {
                    "recommended_transition": "dissolve",
                    "annotation_title": "AI Curated Opening Vista",
                    "annotation_subtitle": "AI 精选开篇实景镜头",
                }
            }
        },
    )
    assert vc_res["updated_video_clips"] == 1

    # Verify AI timeline intelligence override helper (transitions, hold duration, Track V3 lower-third captions)
    first_clip_id = bb.timeline.visual_track[0].id
    ti_res = apply_timeline_intelligence_overrides(
        proj_dir,
        {
            "clips": {
                first_clip_id: {
                    "transition_in": "dissolve",
                    "transition_out": "crossfade",
                    "duration": 4.2,
                }
            },
            "lower_thirds": [
                {
                    "id": "title_lt_custom_01",
                    "clip_id": first_clip_id,
                    "text": "Pont Alexandre III & Eiffel View",
                    "subtitle": "亚历山大三世桥与埃菲尔铁塔远眺",
                    "duration": 3.0,
                }
            ],
        },
    )
    assert ti_res["updated_items"] >= 2
    tl_updated = EditTimeline.from_yaml(proj_dir / "timeline.yaml")
    assert any(t.id == "title_lt_custom_01" and t.style == "lower_third" for t in tl_updated.title_track)

    screening_jpg = render_director_screening_sheet(proj_dir)
    assert screening_jpg.exists() and screening_jpg.stat().st_size > 5000



