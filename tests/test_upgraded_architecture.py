"""Verification suite for the 21-agent upgraded post-production architecture (Gates 1-7)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
import pytest

from video_editing_ai.agents import (
    AgentRegistry,
    BaseEditingAgent,
    CastContinuityAgent,
    CullerAgent,
    DITIngestAgent,
    DialogueEditorAgent,
    HDRColoristAgent,
    LeadAssemblyEditorAgent,
    MontageEditorAgent,
    MotionGraphicsTitleAgent,
    MultiAgentOrchestrator,
    NLEExportAgent,
    PipelineBlackboard,
    ProxyWatchdownQCAgent,
    ShowrunnerAgent,
    SoundDesignerMixerAgent,
    build_audio_mix_plan,
    classify_clip_color_pipeline,
    classify_utterance_language,
    compute_dialogue_normalization_gain,
    compute_face_safe_ken_burns_centers,
    register_agent,
)
from video_editing_ai.models.schema import (
    AudioTrackItem,
    BeatGrid,
    ChapterPlan,
    CreativeBrief,
    CulledSubClip,
    EditTimeline,
    EnergySection,
    KenBurnsConfig,
    MediaCatalog,
    MediaIndex,
    PaperEdit,
    PhotoAsset,
    SubtitleConfig,
    SubtitleItem,
    TimelineMarker,
    TimelineSettings,
    TitleOverlayItem,
    VideoAsset,
    VisualTrackItem,
)
from video_editing_ai.agents.plugins.motion_graphics_agent import (
    _format_exif_date,
    _format_gps_or_camera,
)
from video_editing_ai.renderer.ffmpeg_composer import _build_ducking_volume_expr
from video_editing_ai.renderer.ken_burns import build_hdr_color_filters
from video_editing_ai.renderer.titles import render_title_png


def _parse_fcpxml_seconds(expr: str) -> Fraction:
    """Parse FCPXML rational time string like '150/30s' or '0s' into exact Fraction seconds."""
    cleaned = expr.strip()
    assert cleaned.endswith("s"), f"Invalid FCPXML rational time: {expr}"
    body = cleaned[:-1]
    if "/" in body:
        num_s, den_s = body.split("/", 1)
        return Fraction(int(num_s), int(den_s))
    return Fraction(int(body), 1)


def _make_synthetic_video(path: Path, duration: float = 6.0, color: str = "steelblue", freq: int = 440) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            f"color=c={color}:s=640x360:r=30:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:sample_rate=48000:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )
    return path


def _make_portrait_photo_with_top_face(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (900, 1200), (28, 36, 48))
    draw = ImageDraw.Draw(img)
    # Warm skin-tone oval near the upper portrait third [0.35, 0.10, 0.65, 0.30]
    draw.ellipse((315, 120, 585, 360), fill=(228, 182, 152))
    img.save(path, "JPEG", quality=92)
    return path


# ---------------------------------------------------------------------------
# Gate 1: AgentRegistry deterministic stage ordering, uniqueness & DAG check
# ---------------------------------------------------------------------------
def test_gate1_agent_registry_ordering_uniqueness_and_artifact_dag() -> None:
    roster = AgentRegistry.list_all()
    assert len(roster) >= 21

    # Verify deterministic (stage_order, substage_priority, role_id) ordering
    keys = [(c.stage_order, getattr(c, "substage_priority", 0), c.role_id) for c in roster]
    assert keys == sorted(keys), "AgentRegistry.list_all() must be deterministically sorted"

    # Verify backward-compatible role_id aliases map to upgraded classes in place
    assert AgentRegistry.get("hdr_colorist") is HDRColoristAgent
    assert AgentRegistry.get("colorist") is HDRColoristAgent
    assert AgentRegistry.get("sound_mixer") is SoundDesignerMixerAgent
    assert AgentRegistry.get("sound_designer") is SoundDesignerMixerAgent
    assert AgentRegistry.get("title_designer") is MotionGraphicsTitleAgent
    assert AgentRegistry.get("motion_graphics") is MotionGraphicsTitleAgent

    # Verify artifact dependency graph has zero unsatisfied requires_artifacts
    errors_default = AgentRegistry.validate_artifact_dependencies()
    assert errors_default == [], f"Unsatisfied artifact dependencies: {errors_default}"

    # Verify duplicate role_id registration raises ValueError cleanly
    saved_agents = dict(AgentRegistry._agents)
    saved_aliases = dict(AgentRegistry._aliases)
    try:
        with pytest.raises(ValueError, match="Duplicate role_id"):

            @register_agent
            class _DuplicateCullerAgent(BaseEditingAgent):
                role_id = "culler"
                display_name = "Duplicate Culler"
                stage_order = 10

                def run(self, blackboard: PipelineBlackboard) -> dict:
                    return {}
    finally:
        AgentRegistry._agents = saved_agents
        AgentRegistry._aliases = saved_aliases


# ---------------------------------------------------------------------------
# Gate 2: DIT cache, CullerAgent elastic handles & CastContinuity face-safe crop
# ---------------------------------------------------------------------------
def test_gate2_dit_cache_culler_elastic_handles_and_cast_face_safe_crop(tmp_path: Path) -> None:
    media_dir = tmp_path / "raw_media"
    proj_dir = tmp_path / "proj"
    vid_a = _make_synthetic_video(media_dir / "clip_01.mp4", duration=7.5, color="darkslateblue", freq=520)
    photo_p = _make_portrait_photo_with_top_face(media_dir / "portrait_01.jpg")
    assert vid_a.exists() and photo_p.exists()

    bb = PipelineBlackboard(media_dir=media_dir, project_dir=proj_dir, user_prompt="Family trip highlight")
    bb.creative_brief = CreativeBrief(project_title="Trip Test", raw_user_prompt="Family trip highlight")
    bb.creative_brief.to_json_file(proj_dir / "creative_brief.json")

    dit = DITIngestAgent()
    res1 = dit.run(bb)
    assert res1["cache_misses_count"] >= 2
    assert (proj_dir / "dit_manifest.json").exists()

    # Second run on unchanged files must hit the hash+mtime cache
    res2 = dit.run(bb)
    assert res2["cache_hits_count"] >= 2
    assert res2["cache_misses_count"] == 0

    culler = CullerAgent()
    culler.run(bb)
    assert bb.media_index is not None
    video_subclips = [sc for sc in bb.media_index.subclips if sc.media_type == "video"]
    assert len(video_subclips) >= 1
    for sc in video_subclips:
        assert sc.usable_in is not None and sc.golden_in is not None
        assert sc.golden_out is not None and sc.usable_out is not None
        assert sc.usable_in <= sc.golden_in < sc.golden_out <= sc.usable_out
        assert (sc.usable_out - sc.usable_in) >= (sc.golden_out - sc.golden_in)
        assert sc.pix_fmt != ""
        assert sc.color_space != ""

    # Language classification & CastContinuityAgent face-safe crop calculation
    assert classify_utterance_language("\u592a\u597d\u770b\u4e86") == "zh"
    assert classify_utterance_language("Look at that fountain over there!") == "en"
    assert classify_utterance_language("\u770b Look over there") == "mixed"

    cast_agent = CastContinuityAgent()
    cast_summary = cast_agent.run(bb)
    assert cast_summary["detections_count"] >= 1
    assert (proj_dir / "cast_manifest.json").exists()

    # Verify face-safe Ken Burns trajectory keeps upper portrait head [0.35, 0.08, 0.65, 0.28] uncropped
    head_bbox = (0.35, 0.08, 0.65, 0.28)
    start_zoom, end_zoom = 1.04, 1.14
    start_c, end_c = compute_face_safe_ken_burns_centers(
        bbox=head_bbox,
        orientation="portrait",
        start_zoom=start_zoom,
        end_zoom=end_zoom,
        pan_sign=1.0,
    )
    max_zoom = max(start_zoom, end_zoom)
    half_vis_h = min(0.5, 0.28 / max_zoom)
    for center in (start_c, end_c):
        crop_top = max(0.0, center[1] - half_vis_h)
        crop_bottom = min(1.0, center[1] + half_vis_h)
        # Crown of head (y1=0.08) must never be sliced off by top crop edge
        assert crop_top <= head_bbox[1] + 0.02, f"Head crown cropped: crop_top={crop_top} > y1={head_bbox[1]}"
        assert crop_bottom >= head_bbox[3] - 0.04


# ---------------------------------------------------------------------------
# Gate 3: Beat-sync invariance on chapter revision & max_revisions <= 2 cap
# ---------------------------------------------------------------------------
def test_gate3_beat_sync_invariance_on_chapter_revision_and_max_revisions_cap(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir(parents=True, exist_ok=True)

    v1 = _make_synthetic_video(media_dir / "c1.mp4", duration=8.0, color="navy")
    v2 = _make_synthetic_video(media_dir / "c2.mp4", duration=8.0, color="teal")
    v3 = _make_synthetic_video(media_dir / "c3.mp4", duration=8.0, color="maroon")
    v4 = _make_synthetic_video(media_dir / "c4.mp4", duration=8.0, color="olive")

    downbeats = [round(i * 2.0, 2) for i in range(25)]  # Every 2.0s is a musical bar downbeat
    beats = [round(i * 0.5, 2) for i in range(100)]

    bb = PipelineBlackboard(media_dir=media_dir, project_dir=proj_dir, user_prompt="3-chapter beat test")
    bb.creative_brief = CreativeBrief(
        project_title="Beat Sync Test",
        raw_user_prompt="3-chapter beat test",
        max_broll_duration_sec=4.5,
        cold_open_hook=False,
    )
    bb.creative_brief.to_json_file(proj_dir / "creative_brief.json")

    bb.beat_grid = BeatGrid(
        source_audio=str(v1),
        duration=40.0,
        tempo_bpm=120.0,
        beats=beats,
        downbeats=downbeats,
        energy_sections=[EnergySection(start=0.0, end=40.0, label="chorus_drop", rms_energy=0.9)],
    )
    bb.beat_grid.to_json_file(proj_dir / "beat_grid.json")

    bb.media_index = MediaIndex(
        source_directory=str(media_dir),
        created_at="2026-01-01T10:00:00Z",
        total_raw_videos=4,
        kept_selects_count=4,
        subclips=[
            CulledSubClip(
                subclip_id="sc_1",
                source_asset_id="asset_1",
                filename="c1.mp4",
                source_path=str(v1),
                media_type="video",
                normalized_timestamp="2026-01-01T10:00:00Z",
                roll_type="b_roll",
                in_tc=0.5,
                out_tc=4.5,
                usable_in=0.0,
                golden_in=0.5,
                golden_out=4.5,
                usable_out=7.5,
                duration=4.0,
                summary="Opening fountain wide",
            ),
            CulledSubClip(
                subclip_id="sc_2",
                source_asset_id="asset_2",
                filename="c2.mp4",
                source_path=str(v2),
                media_type="video",
                normalized_timestamp="2026-01-01T10:05:00Z",
                roll_type="b_roll",
                in_tc=0.5,
                out_tc=4.5,
                usable_in=0.0,
                golden_in=0.5,
                golden_out=4.5,
                usable_out=7.5,
                duration=4.0,
                summary="Garden walk medium",
            ),
            CulledSubClip(
                subclip_id="sc_3",
                source_asset_id="asset_3",
                filename="c3.mp4",
                source_path=str(v3),
                media_type="video",
                normalized_timestamp="2026-01-01T10:10:00Z",
                roll_type="b_roll",
                in_tc=0.5,
                out_tc=4.5,
                usable_in=0.0,
                golden_in=0.5,
                golden_out=4.5,
                usable_out=7.5,
                duration=4.0,
                summary="Carousel lights closeup",
            ),
            CulledSubClip(
                subclip_id="sc_4",
                source_asset_id="asset_4",
                filename="c4.mp4",
                source_path=str(v4),
                media_type="video",
                normalized_timestamp="2026-01-01T10:15:00Z",
                roll_type="b_roll",
                in_tc=0.5,
                out_tc=4.5,
                usable_in=0.0,
                golden_in=0.5,
                golden_out=4.5,
                usable_out=7.5,
                duration=4.0,
                summary="Evening finale wide",
            ),
        ],
    )
    bb.media_index.to_json_file(proj_dir / "media_index.json")

    bb.paper_edit = PaperEdit(
        project_title="Beat Sync Test",
        story_arc_summary="Three-act beat sync invariance test",
        chapters=[
            ChapterPlan(
                chapter_id="chap_01",
                title="Chapter 1",
                location_tag="Plaza",
                time_of_day="morning",
                editing_mode="music_montage",
                selected_subclip_ids=["sc_1"],
            ),
            ChapterPlan(
                chapter_id="chap_02",
                title="Chapter 2",
                location_tag="Garden",
                time_of_day="afternoon",
                editing_mode="music_montage",
                selected_subclip_ids=["sc_2"],
            ),
            ChapterPlan(
                chapter_id="chap_03",
                title="Chapter 3",
                location_tag="Carousel",
                time_of_day="twilight",
                editing_mode="music_montage",
                selected_subclip_ids=["sc_3", "sc_4"],
            ),
        ],
    )
    bb.paper_edit.to_json_file(proj_dir / "paper_edit.json")

    # Initial Stage 30 + 35 assembly
    dlg_ed = DialogueEditorAgent()
    montage = MontageEditorAgent()
    lead = LeadAssemblyEditorAgent()
    dlg_ed.run(bb)
    montage.run(bb)
    lead.run(bb)
    assert bb.timeline is not None

    def _chap3_clips(tl: EditTimeline) -> list[VisualTrackItem]:
        return [c for c in tl.visual_track if c.source in (str(v3), str(v4))]

    for c in _chap3_clips(bb.timeline):
        assert any(abs(c.timeline_start - db) <= 0.06 for db in downbeats), (
            f"Pre-revision chap_03 clip {c.id} start {c.timeline_start} not on downbeat"
        )

    # Deliberately corrupt Chapter 2 on the timeline with an overlong 6.3s B-roll (> 4.5s cap)
    # that shifts Chapter 3 off the 2.0s downbeat grid
    for c in bb.timeline.visual_track:
        if c.source == str(v2):
            c.duration = 6.3
            c.source_out = round(c.source_in + 6.3, 3)
    bb.timeline.recalculate_sequential_starts()

    # Stage 50 Showrunner must NOT mutate timeline.yaml directly; it must emit a RevisionDirective on chap_02
    showrunner = ShowrunnerAgent()
    sr_out = showrunner.run(bb)
    assert sr_out["revision_directives_emitted"] >= 1
    pending = bb.pending_revision_directives()
    assert any(d.chapter_id == "chap_02" and d.target_role_id == "montage_editor" for d in pending)

    # Execute the targeted revision round via MontageEditorAgent -> LeadAssemblyEditorAgent -> ShowrunnerAgent
    bb.revision_round = 1
    montage.run(bb)
    lead.run(bb)
    showrunner.run(bb)

    # Chapter 2's B-roll duration must now comply (<= 4.5s) AND Chapter 3 cuts must sit on beat_grid downbeats!
    chap2_clips = [c for c in bb.timeline.visual_track if c.source == str(v2)]
    assert all(c.duration <= 4.5 + 1e-3 for c in chap2_clips)
    chap3_after = _chap3_clips(bb.timeline)
    assert len(chap3_after) == 2
    for c in chap3_after:
        assert any(abs(c.timeline_start - db) <= 0.06 for db in downbeats), (
            f"Post-revision chap_03 clip {c.id} start {c.timeline_start} drifted off beat_grid downbeats!"
        )
        end_tc = round(c.timeline_start + c.duration, 2)
        assert any(abs(end_tc - db) <= 0.06 for db in downbeats), (
            f"Post-revision chap_03 clip {c.id} end {end_tc} drifted off beat_grid downbeats!"
        )

    # Verify hard max_revisions <= 2 termination guarantee on PipelineBlackboard & MultiAgentOrchestrator
    saved_agents = dict(AgentRegistry._agents)
    saved_aliases = dict(AgentRegistry._aliases)
    try:

        @register_agent
        class _PersistentCriticAgent(BaseEditingAgent):
            role_id = "persistent_test_critic"
            display_name = "Persistent Test Critic"
            stage_order = 52
            enabled_by_default = False

            def run(self, blackboard: PipelineBlackboard) -> dict:
                blackboard.add_revision_directive(
                    issued_by=self.role_id,
                    target_role_id="montage_editor",
                    chapter_id="chap_01",
                    reason="Synthetic unresolvable test directive",
                    constraints={"max_clip_duration_sec": 3.5},
                )
                return {"emitted": True}

        loop_dir = tmp_path / "loop_proj"
        loop_dir.mkdir(parents=True, exist_ok=True)
        bb_loop = PipelineBlackboard(media_dir=media_dir, project_dir=loop_dir, max_revisions=2)
        bb_loop.creative_brief = bb.creative_brief
        bb_loop.media_index = bb.media_index
        bb_loop.beat_grid = bb.beat_grid
        bb_loop.paper_edit = bb.paper_edit
        bb.creative_brief.to_json_file(loop_dir / "creative_brief.json")
        bb.media_index.to_json_file(loop_dir / "media_index.json")
        bb.beat_grid.to_json_file(loop_dir / "beat_grid.json")
        bb.paper_edit.to_json_file(loop_dir / "paper_edit.json")

        orch = MultiAgentOrchestrator(
            enabled_optional_roles=["persistent_test_critic"],
            disabled_roles=[
                "client_intake",
                "dit_ingest",
                "music_supervisor",
                "culler",
                "audio_vibe",
                "photo_curator",
                "cast_tracker",
                "story_producer",
                "narrator_voiceover",
                "camera_operator",
                "title_designer",
                "vfx_designer",
                "hdr_colorist",
                "sound_mixer",
                "visual_qc",
                "production_director",
                "nle_exporter",
            ],
        )
        orch.run(bb_loop, render_video=False, export_fcpxml=False)
        assert bb_loop.revision_round == 2
        assert bb_loop.revision_round <= bb_loop.max_revisions
    finally:
        AgentRegistry._agents = saved_agents
        AgentRegistry._aliases = saved_aliases


# ---------------------------------------------------------------------------
# Gate 4: HDRColoristAgent SDR-to-HLG filters & MotionGraphics HDR white clamp
# ---------------------------------------------------------------------------
def test_gate4_hdr_colorist_sdr_to_hlg_filters_and_hdr_safe_overlay_clamping(tmp_path: Path) -> None:
    # 1. zscale path: maps 8-bit SDR BT.709 to ~203-nit HLG diffuse white (npl=203)
    zscale_chain = build_hdr_color_filters(
        src_trc="bt709",
        src_primaries="bt709",
        src_space="bt709",
        is_log=False,
        diffuse_white_nits=203.0,
        use_zscale=True,
    )
    zscale_joined = ",".join(zscale_chain)
    assert "zscale=" in zscale_joined
    assert "transfer=arib-std-b67" in zscale_joined
    assert "primaries=2020" in zscale_joined
    assert "npl=203" in zscale_joined
    assert "format=p010le" in zscale_joined

    # 2. setparams fallback path: scales peak white to ~75% HLG signal prior to setparams
    fallback_chain = build_hdr_color_filters(
        src_trc="bt709",
        src_primaries="bt709",
        src_space="bt709",
        is_log=False,
        diffuse_white_nits=203.0,
        use_zscale=False,
    )
    fallback_joined = ",".join(fallback_chain)
    assert "colorlevels=" in fallback_joined
    assert "romax=0.75" in fallback_joined
    assert "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc" in fallback_joined
    assert fallback_joined.index("colorlevels=") < fallback_joined.index("setparams=")

    # 3. Native 10-bit HLG passthrough check via classify_clip_color_pipeline
    hlg_clip = VisualTrackItem(
        id="clip_hlg",
        type="video",
        source="cam_hlg.mov",
        duration=3.0,
        timeline_start=0.0,
    )
    hlg_sc = CulledSubClip(
        subclip_id="sc_hlg",
        source_asset_id="asset_hlg",
        filename="cam_hlg.mov",
        source_path="cam_hlg.mov",
        media_type="video",
        normalized_timestamp="2026-01-01T10:00:00Z",
        in_tc=0.0,
        out_tc=3.0,
        usable_in=0.0,
        golden_in=0.0,
        golden_out=3.0,
        usable_out=3.0,
        duration=3.0,
        color_primaries="bt2020",
        color_trc="arib-std-b67",
        color_space="bt2020nc",
        pix_fmt="p010le",
    )
    hlg_transform = classify_clip_color_pipeline(hlg_clip, subclip_meta=hlg_sc, hdr_target=True, use_zscale=True)
    assert hlg_transform.input_classification == "bt2020_hlg_10bit"
    assert "passthrough" in hlg_transform.transform_mode

    # 4. MotionGraphicsTitleAgent HDR-safe white & shadow alpha clamping on Pillow RGBA overlays
    hdr_png = tmp_path / "lower_third_hdr.png"
    sdr_png = tmp_path / "lower_third_sdr.png"
    base_kwargs = dict(
        id="title_test",
        style="lower_third",
        text="Chapter 1: Morning Conservatory",
        subtitle="Warm Sunlight",
        timeline_start=1.0,
        duration=3.0,
    )
    render_title_png(TitleOverlayItem(**base_kwargs, hdr_safe=True), hdr_png, width=960, height=540)
    render_title_png(TitleOverlayItem(**base_kwargs, hdr_safe=False), sdr_png, width=960, height=540)

    hdr_arr = np.asarray(Image.open(hdr_png).convert("RGBA"))
    sdr_arr = np.asarray(Image.open(sdr_png).convert("RGBA"))

    visible_hdr = hdr_arr[hdr_arr[:, :, 3] > 0]
    visible_sdr = sdr_arr[sdr_arr[:, :, 3] > 0]
    assert visible_hdr.size > 0 and visible_sdr.size > 0
    assert int(visible_hdr[:, :3].max()) <= 192, "HDR-safe overlay peak RGB must be clamped <= 192 (~203 nits)"
    assert int(visible_sdr[:, :3].max()) > 200, "SDR overlay should retain full 8-bit peak white"

    dark_shadow_hdr = hdr_arr[(hdr_arr[:, :, 0] < 45) & (hdr_arr[:, :, 1] < 45) & (hdr_arr[:, :, 2] < 45)]
    if dark_shadow_hdr.size > 0:
        assert int(dark_shadow_hdr[:, 3].max()) <= 152, "HDR drop-shadow alpha must be clamped <= 152"


# ---------------------------------------------------------------------------
# Gate 5: SoundDesignerMixerAgent diegetic muting & -14 LUFS normalization math
# ---------------------------------------------------------------------------
def test_gate5_sound_mixer_diegetic_music_muting_and_lufs_normalization() -> None:
    # Quiet child utterance (-24 LUFS) boosted toward -14 LUFS; loud operator (-8 LUFS) attenuated
    child_db, child_lin = compute_dialogue_normalization_gain(-24.0, target_lufs=-14.0)
    adult_db, adult_lin = compute_dialogue_normalization_gain(-8.0, target_lufs=-14.0)
    assert child_db == pytest.approx(10.0, abs=0.01)
    assert child_lin > 3.0
    assert adult_db == pytest.approx(-6.0, abs=0.01)
    assert adult_lin < 0.55

    tl = EditTimeline(
        project_name="Audio Mix Gate",
        visual_track=[
            VisualTrackItem(
                id="c_speech",
                type="video",
                roll_type="a_roll",
                source="speech.mp4",
                source_in=0.0,
                source_out=4.0,
                duration=4.0,
                timeline_start=0.0,
            ),
            VisualTrackItem(
                id="c_stage",
                type="video",
                roll_type="b_roll",
                source="stage_show.mp4",
                source_in=0.0,
                source_out=5.0,
                duration=5.0,
                timeline_start=4.0,
            ),
            VisualTrackItem(
                id="c_broll",
                type="video",
                roll_type="b_roll",
                source="scenery.mp4",
                source_in=0.0,
                source_out=3.5,
                duration=3.5,
                timeline_start=9.0,
            ),
        ],
    )
    mi = MediaIndex(
        source_directory=".",
        created_at="2026-01-01T10:00:00Z",
        subclips=[
            CulledSubClip(
                subclip_id="sc_speech",
                source_asset_id="a_sp",
                filename="speech.mp4",
                source_path="speech.mp4",
                media_type="video",
                normalized_timestamp="2026-01-01T10:00:00Z",
                roll_type="a_roll",
                in_tc=0.0,
                out_tc=4.0,
                usable_in=0.0,
                golden_in=0.0,
                golden_out=4.0,
                usable_out=4.0,
                duration=4.0,
                integrated_lufs=-22.0,
                has_diegetic_music=False,
            ),
            CulledSubClip(
                subclip_id="sc_stage",
                source_asset_id="a_st",
                filename="stage_show.mp4",
                source_path="stage_show.mp4",
                media_type="video",
                normalized_timestamp="2026-01-01T10:01:00Z",
                roll_type="b_roll",
                in_tc=0.0,
                out_tc=5.0,
                usable_in=0.0,
                golden_in=0.0,
                golden_out=5.0,
                usable_out=5.0,
                duration=5.0,
                integrated_lufs=-15.0,
                has_diegetic_music=True,
            ),
            CulledSubClip(
                subclip_id="sc_broll",
                source_asset_id="a_br",
                filename="scenery.mp4",
                source_path="scenery.mp4",
                media_type="video",
                normalized_timestamp="2026-01-01T10:02:00Z",
                roll_type="b_roll",
                in_tc=0.0,
                out_tc=3.5,
                usable_in=0.0,
                golden_in=0.0,
                golden_out=3.5,
                usable_out=3.5,
                duration=3.5,
                integrated_lufs=-28.0,
                has_diegetic_music=False,
            ),
        ],
    )

    entries, env = build_audio_mix_plan(tl, media_index=mi)
    by_id = {e.clip_id: e for e in entries}
    env_by_id = {s.trigger_clip_id: s for s in env}

    assert by_id["c_speech"].applied_gain_db == pytest.approx(8.0, abs=0.05)
    assert by_id["c_speech"].crossfade_sec == pytest.approx(0.33, abs=0.01)
    assert env_by_id["c_speech"].mode == "speech_duck"
    assert env_by_id["c_speech"].bgm_attenuation_db == pytest.approx(-14.0, abs=0.05)

    # Diegetic live music clip must duck BGM by <= -28 dB (-30 dB)
    assert by_id["c_stage"].has_diegetic_music is True
    assert env_by_id["c_stage"].mode == "diegetic_music_mute"
    assert env_by_id["c_stage"].bgm_attenuation_db <= -28.0

    # Pure B-roll clip gets -20 dB (0.10 linear) Nat-Sound bed
    assert by_id["c_broll"].linear_volume == pytest.approx(0.10, abs=0.005)

    # Verify FFmpeg volume expression incorporates the -30 dB diegetic ducking window
    vol_expr = _build_ducking_volume_expr(tl, base_vol=0.28, duck_vol=0.056, diegetic_windows=[(4.0, 9.0)])
    assert "between(t,4.00,9.00)" in vol_expr
    assert "0.00885" in vol_expr  # 0.28 * 10^(-30/20) = 0.00885


# ---------------------------------------------------------------------------
# Gate 6 & Gate 7: FCPXML 1.11 timing integrity + docs-sync CLI verification
# ---------------------------------------------------------------------------
def test_gate6_and_gate7_fcpxml_integrity_and_docs_sync_cli(tmp_path: Path) -> None:
    proj_dir = tmp_path / "fcpxml_proj"
    proj_dir.mkdir(parents=True, exist_ok=True)

    vid1 = _make_synthetic_video(proj_dir / "media" / "shot_a.mp4", duration=4.0, color="darkgreen")
    vid2 = _make_synthetic_video(proj_dir / "media" / "shot_b.mp4", duration=4.0, color="darkorange")
    photo1 = _make_portrait_photo_with_top_face(proj_dir / "media" / "still_c.jpg")

    tl = EditTimeline(
        project_name="FCPXML Integrity Check",
        settings=TimelineSettings(width=3840, height=2160, fps=30, hdr=True),
        visual_track=[
            VisualTrackItem(
                id="v1_01",
                type="video",
                roll_type="a_roll",
                source=str(vid1),
                source_in=0.5,
                source_out=3.5,
                duration=3.0,
                timeline_start=0.0,
                volume=1.0,
                audio_crossfade_sec=0.33,
            ),
            VisualTrackItem(
                id="v1_02",
                type="photo",
                roll_type="b_roll",
                source=str(photo1),
                duration=2.5,
                timeline_start=3.0,
                ken_burns=KenBurnsConfig(mode="gallery_frame"),
            ),
        ],
        broll_track=[
            VisualTrackItem(
                id="v2_01",
                type="video",
                track_lane=2,
                roll_type="b_roll",
                source=str(vid2),
                source_in=0.2,
                source_out=1.7,
                duration=1.5,
                timeline_start=1.0,
                volume=0.0,
            )
        ],
        title_track=[
            TitleOverlayItem(
                id="v3_01",
                style="lower_third",
                text="Chapter 1: Conservatory",
                subtitle="Morning Walk",
                timeline_start=0.5,
                duration=2.0,
                hdr_safe=True,
            )
        ],
        audio_tracks=[
            AudioTrackItem(
                id="a2_sfx",
                source=str(vid2),
                role="sfx",
                timeline_start=0.5,
                duration=1.5,
                volume=0.12,
            ),
            AudioTrackItem(
                id="a3_bgm",
                source=str(vid1),
                role="music",
                timeline_start=0.0,
                duration=5.5,
                volume=0.28,
            ),
        ],
        markers=[TimelineMarker(name="Act I", time_sec=0.0)],
        subtitles=SubtitleConfig(
            items=[SubtitleItem(id="sub_1", start=3.2, end=4.8, text="Morning walk", speaker="Cast 1")]
        ),
    )
    tl_yaml = proj_dir / "timeline.yaml"
    tl.to_yaml(tl_yaml)

    # Also verify Stage 55 ProxyWatchdownQCAgent passes cleanly on non-overlapping layout
    bb = PipelineBlackboard(media_dir=proj_dir / "media", project_dir=proj_dir)
    bb.timeline = tl
    qc_agent = ProxyWatchdownQCAgent()
    qc_summary = qc_agent.run(bb)
    assert qc_summary["overlay_overlap_count"] == 0
    assert (proj_dir / "visual_qc_report.json").exists()

    # Run Stage 65 NLEExportAgent AND standalone CLI `export-fcpxml`
    nle_agent = NLEExportAgent()
    nle_agent.run(bb)
    cli_fcpxml = proj_dir / "cli_exported.fcpxml"
    proc_fcpxml = subprocess.run(
        [
            sys.executable,
            "-m",
            "video_editing_ai.cli",
            "export-fcpxml",
            str(proj_dir),
            "-o",
            str(cli_fcpxml),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc_fcpxml.returncode == 0
    assert (proj_dir / "timeline.fcpxml").exists()
    assert cli_fcpxml.exists()

    # Parse XML and validate FCPXML 1.11 resources, spine continuity, lanes V2/V3/A1-A3, and markers
    root = ET.fromstring((proj_dir / "timeline.fcpxml").read_text(encoding="utf-8"))
    assert root.tag == "fcpxml"
    assert root.attrib["version"] == "1.11"

    resources = root.find("resources")
    assert resources is not None
    resource_ids = {el.attrib["id"] for el in resources if "id" in el.attrib}
    style_def_ids = {el.attrib["id"] for el in root.iter("text-style-def") if "id" in el.attrib}

    for ref_el in root.iter():
        if "ref" in ref_el.attrib:
            if ref_el.tag == "text-style":
                assert ref_el.attrib["ref"] in style_def_ids
            else:
                assert ref_el.attrib["ref"] in resource_ids, f"Missing resource ref={ref_el.attrib['ref']}"

    sequence = root.find("./library/event/project/sequence")
    assert sequence is not None
    seq_dur = _parse_fcpxml_seconds(sequence.attrib["duration"])

    spine = sequence.find("spine")
    assert spine is not None
    primary_clips = [el for el in list(spine) if el.tag in ("asset-clip", "video")]
    assert len(primary_clips) == 2

    running = Fraction(0, 1)
    for pc in primary_clips:
        off = _parse_fcpxml_seconds(pc.attrib["offset"])
        dur = _parse_fcpxml_seconds(pc.attrib["duration"])
        assert off == running, f"Spine gap/overlap: expected offset {running}, got {off}"
        running += dur
    assert running == seq_dur

    # Verify V2 (lane="1"), V3 (<title lane="2">), A2/A3 negative audio lanes, and chapter <marker>
    lanes_found = {el.attrib.get("lane") for el in spine.iter() if "lane" in el.attrib}
    assert "1" in lanes_found
    assert "2" in lanes_found
    assert "-2" in lanes_found
    assert "-3" in lanes_found
    assert len(list(spine.iter("marker"))) >= 1

    # Gate 7: Run `python -m video_editing_ai.cli docs-sync --no-sync-global` and verify README & SKILL match AgentRegistry
    repo_root = Path(__file__).resolve().parents[1]
    proc_sync = subprocess.run(
        [sys.executable, "-m", "video_editing_ai.cli", "docs-sync", "--no-sync-global"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc_sync.returncode == 0

    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    skill_text = (repo_root / ".agent" / "skills" / "ai-video-editor" / "SKILL.md").read_text(encoding="utf-8")
    refs_dir = repo_root / ".agent" / "skills" / "ai-video-editor" / "references"

    for cls in AgentRegistry.list_all():
        assert f"`{cls.role_id}`" in readme_text, f"{cls.role_id} missing from README.md"
        assert f"`{cls.__name__}`" in readme_text, f"{cls.__name__} missing from README.md"
        assert f"`{cls.role_id}`" in skill_text, f"{cls.role_id} missing from SKILL.md"
        assert f"`{cls.__name__}`" in skill_text, f"{cls.__name__} missing from SKILL.md"
        assert (refs_dir / cls.skill_reference_file).exists(), f"Missing playbook {cls.skill_reference_file}"


def test_motion_graphics_exif_stamp_string_dates_and_west_hemisphere(tmp_path: Path) -> None:
    """Verify MotionGraphicsTitleAgent handles string creation_time, W/S coords, and camera_make_model fallback."""
    assert _format_exif_date("2026-10-25T11:15:30-07:00") == "2026.10.25"
    assert _format_exif_date("2026:10:25 14:20:00") == "2026.10.25"
    assert _format_gps_or_camera(37.7857, -122.4011, "iPhone") == "37.79N, 122.40W"
    assert _format_gps_or_camera(0.0, 0.0, "iPhone") == "0.00N, 0.00E"
    assert _format_gps_or_camera(None, None, "Apple iPhone 18 Pro") == "Apple iPhone 18 Pro"

    def _run_case(name: str, catalog: MediaCatalog) -> list[str]:
        proj_dir = tmp_path / name
        media_dir = tmp_path / f"{name}_media"
        media_dir.mkdir(parents=True, exist_ok=True)
        bb = PipelineBlackboard(media_dir=media_dir, project_dir=proj_dir)
        bb.creative_brief = CreativeBrief(project_title="SFMOMA Test", raw_user_prompt="SFMOMA test")
        bb.paper_edit = PaperEdit(project_title="SFMOMA Test", story_arc_summary="Day at SFMOMA")
        bb.catalog = catalog
        bb.timeline = EditTimeline(
            project_name="SFMOMA Test",
            settings=TimelineSettings(width=1920, height=1080, fps=30, hdr=False),
            visual_track=[
                VisualTrackItem(
                    id="v1_01",
                    type="video",
                    roll_type="a_roll",
                    source=str(media_dir / "clip.mov"),
                    source_in=0.0,
                    source_out=30.0,
                    duration=30.0,
                    timeline_start=0.0,
                )
            ],
        )
        res = MotionGraphicsTitleAgent().run(bb)
        assert res["graphics_injected"] >= 1
        assert (proj_dir / "graphics_manifest.json").exists()
        return [f"{t.text} | {t.subtitle}" for t in bb.timeline.title_track]

    sf_photo_titles = _run_case(
        "sf_photo",
        MediaCatalog(
            source_directory=str(tmp_path),
            created_at="2026-10-25T18:00:00Z",
            photos=[
                PhotoAsset(
                    id="p1",
                    path=str(tmp_path / "p1.heic"),
                    filename="p1.heic",
                    width=4032,
                    height=3024,
                    creation_time="2026-10-25T11:15:30-07:00",
                    gps_lat=37.7857,
                    gps_lon=-122.4011,
                    camera_make_model="Apple iPhone 18 Pro",
                )
            ],
        ),
    )
    assert any("2026.10.25" in s and "37.79N, 122.40W" in s for s in sf_photo_titles)

    no_gps_first_photo_titles = _run_case(
        "no_gps_first_photo",
        MediaCatalog(
            source_directory=str(tmp_path),
            created_at="2026-10-25T18:00:00Z",
            photos=[
                PhotoAsset(
                    id="p2",
                    path=str(tmp_path / "p2.heic"),
                    filename="p2.heic",
                    width=3024,
                    height=4032,
                    creation_time="2026:10:25 12:00:00",
                    camera_make_model="Apple iPhone 18 Pro",
                ),
                PhotoAsset(
                    id="p3",
                    path=str(tmp_path / "p3.heic"),
                    filename="p3.heic",
                    width=4032,
                    height=3024,
                    creation_time="2026:10:25 12:05:00",
                    gps_lat=37.7857,
                    gps_lon=-122.4011,
                ),
            ],
        ),
    )
    assert any("2026.10.25" in s and "Apple iPhone 18 Pro" in s for s in no_gps_first_photo_titles)

    video_only_titles = _run_case(
        "video_only",
        MediaCatalog(
            source_directory=str(tmp_path),
            created_at="2026-10-25T18:00:00Z",
            videos=[
                VideoAsset(
                    id="v1",
                    path=str(tmp_path / "v1.mov"),
                    filename="v1.mov",
                    duration=8.0,
                    width=3840,
                    height=2160,
                    fps=30.0,
                    creation_time="2026-10-25T14:05:00Z",
                ),
                VideoAsset(
                    id="v2",
                    path=str(tmp_path / "v2.mov"),
                    filename="v2.mov",
                    duration=8.0,
                    width=3840,
                    height=2160,
                    fps=30.0,
                    creation_time="2026-10-25T14:10:00Z",
                    gps_lat=37.7857,
                    gps_lon=-122.4011,
                ),
            ],
        ),
    )
    assert any("2026.10.25" in s for s in video_only_titles)

    no_gps_anywhere_titles = _run_case(
        "no_gps_anywhere",
        MediaCatalog(
            source_directory=str(tmp_path),
            created_at="2026-10-25T18:00:00Z",
            videos=[
                VideoAsset(
                    id="v3",
                    path=str(tmp_path / "v3.mov"),
                    filename="v3.mov",
                    duration=8.0,
                    width=3840,
                    height=2160,
                    fps=30.0,
                    creation_time="2026-10-25T15:00:00Z",
                )
            ],
        ),
    )
    assert any("SFMOMA Test" in s for s in no_gps_anywhere_titles)


def test_audio_stream_selection_and_whisper_pcm_decode(tmp_path: Path) -> None:
    """Verify non-APAC audio stream selection and 16kHz PCM decoding for faster-whisper."""
    from video_editing_ai.analyzer.media_probe import probe_video_file, select_decodable_audio_stream_pos
    from video_editing_ai.analyzer.transcriber import _decode_audio_pcm_16k

    assert (
        select_decodable_audio_stream_pos(
            [
                {"codec_type": "video", "codec_name": "hevc"},
                {"codec_type": "audio", "codec_name": "apple_apac"},
                {"codec_type": "audio", "codec_name": "aac"},
            ]
        )
        == 1
    )
    assert select_decodable_audio_stream_pos([{"codec_type": "audio", "codec_name": "apac"}]) is None
    assert select_decodable_audio_stream_pos([{"codec_type": "audio", "codec_name": "aac"}]) == 0

    synth = _make_synthetic_video(tmp_path / "synth_pcm.mp4", duration=2.0, freq=440)
    info = probe_video_file(synth)
    assert info["has_audio"] is True
    assert info["audio_stream_pos"] == 0

    pcm = _decode_audio_pcm_16k(synth)
    assert pcm is not None
    assert pcm.dtype == np.float32
    assert 24000 <= pcm.size <= 36000
    assert float(np.max(np.abs(pcm))) > 0.01


def test_timeline_yaml_gain_roundtrip_cold_open_sync_ripple_and_cjk_title(tmp_path: Path) -> None:
    """Verify +12dB YAML round-trip, cold_open_hook override sync, ripple duration edits, and CJK title font."""
    from video_editing_ai.agents.picture_editors import _apply_timeline_intelligence_to_obj
    from video_editing_ai.agents.story_producer_agent import apply_story_plan_overrides
    from video_editing_ai.models.schema import ColdOpenHookSlot, EffectTrackItem
    from video_editing_ai.renderer.titles import _load_font, render_title_png

    # 1. VisualTrackItem.volume up to 4.0 (+12dB dialogue boost) round-trips via EditTimeline YAML
    tl = EditTimeline(
        project_name="SFMOMA — A Day in Modern Art",
        visual_track=[
            VisualTrackItem(
                id="clip_001",
                type="video",
                roll_type="a_roll",
                source=str(tmp_path / "a.mov"),
                timeline_start=0.0,
                duration=4.0,
                source_in=0.0,
                source_out=4.0,
                volume=3.9811,
            ),
            VisualTrackItem(
                id="clip_002",
                type="photo",
                roll_type="still",
                source=str(tmp_path / "p.heic"),
                timeline_start=4.0,
                duration=3.0,
                source_in=0.0,
                source_out=3.0,
                volume=0.0,
                ken_burns=KenBurnsConfig(mode="parallax_gallery", caption_title="Gallery", caption_subtitle="Test"),
            ),
        ],
        markers=[
            TimelineMarker(time_sec=0.0, name="Act I"),
            TimelineMarker(time_sec=4.0, name="Act II"),
        ],
        title_track=[
            TitleOverlayItem(id="title_opening", style="opening_card", text="Open", timeline_start=0.2, duration=2.0),
            TitleOverlayItem(id="title_quote", style="dialogue_badge", text="看出来3D了吗", timeline_start=4.5, duration=2.0),
            TitleOverlayItem(id="title_closing", style="closing_card", text="Fin", timeline_start=4.8, duration=2.0),
        ],
        effects_track=[
            EffectTrackItem(id="fx_1", effect_type="vignette_focus", target_clip_id="clip_001", timeline_start=0.0, duration=4.0),
            EffectTrackItem(id="fx_2", effect_type="vignette_focus", target_clip_id="clip_002", timeline_start=4.0, duration=3.0),
        ],
        audio_tracks=[
            AudioTrackItem(id="bgm_main", role="music", source=str(tmp_path / "bgm.mp3"), timeline_start=0.0, duration=7.0),
            AudioTrackItem(id="sfx_01", role="sfx", source=str(tmp_path / "sfx.wav"), timeline_start=4.2, duration=1.0),
        ],
        subtitles=SubtitleConfig(
            items=[
                SubtitleItem(id="sub_1", start=0.5, end=3.5, text="这次我们去博物馆", speaker="Speaker"),
                SubtitleItem(id="sub_2", start=4.3, end=6.0, text="看到3D了吗", speaker="Speaker"),
            ]
        ),
    )
    tl_path = tmp_path / "timeline.yaml"
    tl.to_yaml(tl_path)
    reloaded = EditTimeline.from_yaml(tl_path)
    assert reloaded.visual_track[0].volume == pytest.approx(3.9811, rel=1e-4)

    # 2. _apply_timeline_intelligence_to_obj ripples downstream tracks accurately when clip durations expand
    _apply_timeline_intelligence_to_obj(
        reloaded,
        {"clips": {"clip_001": {"duration": 6.0}, "clip_002": {"duration": 5.5}}},
    )
    assert reloaded.total_duration == pytest.approx(11.5, abs=1e-2)
    assert reloaded.visual_track[1].timeline_start == pytest.approx(6.0, abs=1e-2)
    assert reloaded.markers[1].time_sec == pytest.approx(6.0, abs=1e-2)
    assert reloaded.effects_track[0].duration == pytest.approx(6.0, abs=1e-2)
    assert reloaded.effects_track[1].timeline_start == pytest.approx(6.0, abs=1e-2)
    assert reloaded.effects_track[1].duration == pytest.approx(5.5, abs=1e-2)
    assert reloaded.subtitles.items[0].start == pytest.approx(0.5, abs=1e-2)
    assert reloaded.subtitles.items[1].start == pytest.approx(6.3, abs=1e-2)
    assert reloaded.title_track[1].timeline_start == pytest.approx(6.5, abs=1e-2)
    assert reloaded.title_track[2].timeline_start == pytest.approx(9.3, abs=1e-2)
    assert reloaded.audio_tracks[0].duration == pytest.approx(11.5, abs=1e-2)
    assert reloaded.audio_tracks[1].timeline_start == pytest.approx(6.2, abs=1e-2)

    # 3. apply_story_plan_overrides syncs cold_open_hook when cold_open_subclip_id overrides are supplied
    proj_story = tmp_path / "story_proj"
    proj_story.mkdir(parents=True, exist_ok=True)
    PaperEdit(
        project_title="Old Title",
        story_arc_summary="Summary",
        cold_open_hook=ColdOpenHookSlot(enabled=True, subclip_id="stale_hook", in_tc=1.0, out_tc=3.0),
        cold_open_subclip_id="stale_hook",
    ).to_json_file(proj_story / "paper_edit.json")
    apply_story_plan_overrides(
        proj_story,
        {
            "project_title": "SFMOMA — A Day in Modern Art",
            "cold_open_subclip_id": "new_leap_hook",
            "cold_open_in_tc": 7.5,
            "cold_open_out_tc": 11.2,
            "cold_open_hook_caption": "Three Leaps Inside SFMOMA",
        },
    )
    pe_updated = PaperEdit.from_json_file(proj_story / "paper_edit.json")
    assert pe_updated.cold_open_hook is not None
    assert pe_updated.cold_open_hook.subclip_id == "new_leap_hook"
    assert pe_updated.cold_open_hook.in_tc == pytest.approx(7.5)
    assert pe_updated.cold_open_hook.out_tc == pytest.approx(11.2)
    assert pe_updated.cold_open_hook.caption == "Three Leaps Inside SFMOMA"

    # 4. CJK serif & dialogue_badge rendering uses CJK-capable font on macOS
    cjk_sample = "看到 看出来3D了吗"
    cjk_font = _load_font(28, bold=True, serif=True, text=cjk_sample)
    if sys.platform == "darwin":
        family_name = " ".join(cjk_font.getname()).lower()
        assert any(k in family_name for k in ("songti", "pingfang", "heiti", "hiragino", "arial unicode"))
    badge_png = tmp_path / "cjk_badge.png"
    render_title_png(
        TitleOverlayItem(
            id="t_cjk",
            style="dialogue_badge",
            text=f'"{cjk_sample}"',
            subtitle="Candid Spoken Moment",
            timeline_start=0.0,
            duration=3.0,
            hdr_safe=True,
        ),
        badge_png,
        width=1280,
        height=720,
        hdr_safe=True,
    )
    assert badge_png.exists() and badge_png.stat().st_size > 1000


def test_surgical_drop_ripple_audio_role_and_revision_constraints(tmp_path: Path) -> None:
    """Verify surgical drop ripples non-music audio & closing card, audio-mix guards non-music, and revision constraints exclude A-roll."""
    from video_editing_ai.agents.picture_editors import (
        _apply_timeline_intelligence_to_obj,
        _enforce_chapter_revision_constraints,
    )
    from video_editing_ai.models.schema import CulledSubClip, EffectTrackItem
    from video_editing_ai.planner.timeline_surgeon import apply_surgical_edit

    proj = tmp_path / "surgeon_proj"
    proj.mkdir(parents=True, exist_ok=True)
    tl = EditTimeline(
        project_name="Test Surgical Drop",
        visual_track=[
            VisualTrackItem(
                id="c1",
                type="video",
                roll_type="a_roll",
                source=str(tmp_path / "c1.mov"),
                timeline_start=0.0,
                duration=3.0,
                source_in=0.0,
                source_out=3.0,
                volume=1.0,
            ),
            VisualTrackItem(
                id="c2_break",
                type="video",
                roll_type="b_roll",
                source=str(tmp_path / "c2.mov"),
                timeline_start=3.0,
                duration=2.5,
                source_in=0.0,
                source_out=2.5,
                volume=0.1,
            ),
            VisualTrackItem(
                id="c3",
                type="photo",
                roll_type="still",
                source=str(tmp_path / "c3.heic"),
                timeline_start=5.5,
                duration=4.5,
                source_in=0.0,
                source_out=4.5,
                volume=0.0,
                ken_burns=KenBurnsConfig(mode="parallax_gallery", caption_title="Gallery Plate"),
            ),
        ],
        title_track=[
            TitleOverlayItem(id="title_closing", style="closing_card", text="Fin", timeline_start=7.8, duration=2.0),
        ],
        effects_track=[
            EffectTrackItem(id="fx_break", effect_type="vignette_focus", target_clip_id="c2_break", timeline_start=3.0, duration=2.5),
            EffectTrackItem(id="fx_c3", effect_type="vignette_focus", target_clip_id="c3", timeline_start=5.5, duration=4.5),
        ],
        audio_tracks=[
            AudioTrackItem(id="bgm_main", role="music", source=str(tmp_path / "bgm.mp3"), timeline_start=0.0, duration=10.0, volume=0.25),
            AudioTrackItem(id="vo_late", role="voiceover", source=str(tmp_path / "vo.wav"), timeline_start=6.0, duration=2.0, volume=0.92),
        ],
    )
    tl.to_yaml(proj / "timeline.yaml")

    apply_surgical_edit(proj, {"action": "drop", "clip_id": "c2_break"})
    apply_surgical_edit(proj, {"action": "audio-mix", "bgm_volume": 0.35, "duck_volume": 0.08})

    after = EditTimeline.from_yaml(proj / "timeline.yaml")
    assert [c.id for c in after.visual_track] == ["c1", "c3"]
    assert after.total_duration == pytest.approx(7.5, abs=1e-2)
    assert [fx.id for fx in after.effects_track] == ["fx_c3"]
    assert after.effects_track[0].timeline_start == pytest.approx(3.0, abs=1e-2)
    assert after.title_track[0].timeline_start == pytest.approx(5.3, abs=1e-2)
    vo = next(a for a in after.audio_tracks if a.id == "vo_late")
    bgm = next(a for a in after.audio_tracks if a.id == "bgm_main")
    assert vo.timeline_start == pytest.approx(3.5, abs=1e-2)
    assert vo.volume == pytest.approx(0.92, abs=1e-3)
    assert bgm.volume == pytest.approx(0.35, abs=1e-3)
    assert bgm.duration == pytest.approx(7.5, abs=1e-2)

    # Verify _apply_timeline_intelligence_to_obj sets clip volume cleanly
    _apply_timeline_intelligence_to_obj(after, {"clips": {"c1": {"volume": 1.35}}})
    assert after.visual_track[0].volume == pytest.approx(1.35, abs=1e-3)

    # Verify _enforce_chapter_revision_constraints excludes a_roll & globally used sources and keeps captioned stills
    stills = [
        VisualTrackItem(
            id=f"s_{i}",
            type="photo",
            roll_type="still",
            source=str(tmp_path / f"s_{i}.jpg"),
            timeline_start=float(i * 3),
            duration=3.0,
            source_in=0.0,
            source_out=3.0,
            ken_burns=KenBurnsConfig(mode="parallax_gallery", caption_title=f"Plate {i}"),
        )
        for i in range(5)
    ]
    subclips = [
        CulledSubClip(
            subclip_id="aroll_1",
            source_asset_id="a1",
            source_path=str(tmp_path / "aroll_1.mov"),
            filename="aroll_1.mov",
            media_type="video",
            normalized_timestamp="2026-10-25T12:35:00Z",
            in_tc=0.0,
            out_tc=3.0,
            usable_in=0.0,
            usable_out=3.0,
            duration=3.0,
            roll_type="a_roll",
        ),
        CulledSubClip(
            subclip_id="broll_used",
            source_asset_id="b1",
            source_path=str(tmp_path / "broll_used.mov"),
            filename="broll_used.mov",
            media_type="video",
            normalized_timestamp="2026-10-25T12:40:00Z",
            in_tc=0.0,
            out_tc=3.0,
            usable_in=0.0,
            usable_out=3.0,
            duration=3.0,
            roll_type="b_roll",
        ),
    ]
    revised = _enforce_chapter_revision_constraints(
        ch_clips=stills,
        constraints={"max_consecutive_stills": 3},
        beat_grid=None,
        prefer_downbeat=False,
        media_index_subclips=subclips,
        global_used_sources={str(tmp_path / "broll_used.mov")},
    )
    assert [c.id for c in revised] == [f"s_{i}" for i in range(5)]
    assert all("_vid_break" not in c.id for c in revised)

    # Verify verified SSL context uses CA trust (fixes macOS Python CA bundle errors)
    import ssl
    from video_editing_ai.agents.base import _build_verified_ssl_context

    ctx = _build_verified_ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_mixed_hevc_extradata_concat_preserves_all_frames(tmp_path: Path) -> None:
    """Ensure HDR concat preserves all frames and produces zero HEVC decode errors when hvcC extradata differs."""
    from video_editing_ai.renderer.ffmpeg_composer import (
        _concat_segments_safe,
        _render_video_segment,
    )
    from video_editing_ai.renderer.ken_burns import render_photo_clip

    src_vid = tmp_path / "src_vid.mp4"
    gen_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x180:rate=15:duration=1.5",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=1.5",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(src_vid),
    ]
    gen_proc = subprocess.run(gen_cmd, capture_output=True, text=True, check=False)
    assert gen_proc.returncode == 0, gen_proc.stderr

    src_jpg = tmp_path / "src_photo.jpg"
    img = Image.new("RGB", (320, 180), (42, 74, 118))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 30, 280, 150), fill=(215, 165, 92))
    img.save(src_jpg, quality=92)

    seg_vid = tmp_path / "seg_vid.mp4"
    _render_video_segment(
        item=VisualTrackItem(
            id="v1",
            type="video",
            source=str(src_vid),
            timeline_start=0.0,
            duration=1.0,
            source_in=0.0,
            source_out=1.0,
        ),
        output_mp4=seg_vid,
        width=320,
        height=180,
        fps=15,
        hdr=True,
    )

    seg_ph = tmp_path / "seg_ph.mp4"
    render_photo_clip(
        photo_path=src_jpg,
        output_mp4=seg_ph,
        duration=1.0,
        width=320,
        height=180,
        fps=15,
        hdr=True,
        ken_burns=KenBurnsConfig(mode="parallax_gallery", caption_title="Test Plate"),
    )

    out_mp4 = tmp_path / "concat_out.mp4"
    _concat_segments_safe([seg_vid, seg_ph], out_mp4, w_dir=tmp_path, fps=15, hdr=True)
    assert out_mp4.exists()

    dec_proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(out_mp4), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert dec_proc.returncode == 0, dec_proc.stderr
    assert dec_proc.stderr.strip() == "", f"Unexpected HEVC decode warnings: {dec_proc.stderr}"

    probe_proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nw=1:nk=1",
            str(out_mp4),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe_proc.returncode == 0, probe_proc.stderr
    nb_read = int(probe_proc.stdout.strip())
    assert nb_read >= 29 and nb_read <= 30, f"Expected ~30 frames, got {nb_read}"


def test_gate8_hardware_profile_accel_policy_and_amd_opencl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate 8: Unified HardwareProfile caching, AI_EDITOR_ACCEL policies, AMD OpenCL Apple:GPU:N binding, sleep guard."""
    import os
    import sys
    import cv2

    from video_editing_ai import hardware

    try:
        # 1. CPU override forces all GPU paths off cleanly
        monkeypatch.setenv("AI_EDITOR_ACCEL", "cpu")
        monkeypatch.setenv("AI_EDITOR_MAX_WORKERS", "3")
        hardware.reset_hardware_cache()
        cpu_prof = hardware.get_hardware_profile()
        assert cpu_prof.accel_policy == "cpu"
        assert cpu_prof.has_mps is False
        assert cpu_prof.torch_device == "cpu"
        assert cpu_prof.has_amd_opencl is False
        assert cpu_prof.has_videotoolbox_hwaccel is False
        assert cpu_prof.best_hdr_encoder == "libx265"
        assert cpu_prof.best_sdr_encoder == "libx264"
        assert cpu_prof.vt_supports_qscale_hdr is False
        assert cpu_prof.vt_supports_qscale_sdr is False
        assert hardware.get_io_worker_count(8) == 3

        # 2. MPS-only policy keeps OpenCV OpenCL off orthogonally
        monkeypatch.setenv("AI_EDITOR_ACCEL", "mps")
        hardware.reset_hardware_cache()
        assert hardware.configure_opencv_amd_opencl() is False

        # 3. Mocked dual-GPU macOS system_profiler selects Apple:GPU:1 (never "AMD:GPU:")
        monkeypatch.setenv("AI_EDITOR_ACCEL", "opencl")
        monkeypatch.delenv("OPENCV_OPENCL_DEVICE", raising=False)
        hardware.reset_hardware_cache()
        monkeypatch.setattr(
            hardware,
            "_detect_macos_gpus",
            lambda: [
                {"name": "Intel UHD Graphics 630", "vendor": "0x8086", "vram": "1536 MB"},
                {"name": "AMD Radeon Pro 5500M", "vendor": "0x1002", "vram": "8176 MB"},
            ],
        )
        monkeypatch.setattr(hardware.sys, "platform", "darwin")
        idx, name = hardware._find_amd_gpu_index_and_name(hardware._detect_macos_gpus())
        assert idx == 1 and "AMD" in name
        if cv2.ocl.haveOpenCL():
            assert hardware.configure_opencv_amd_opencl() is True
            assert os.environ.get("OPENCV_OPENCL_DEVICE") == "Apple:GPU:1"

        # 4. Caffeinate guard runs safely without raising on any platform
        with hardware.caffeinate_guard():
            pass
    finally:
        monkeypatch.undo()
        hardware.reset_hardware_cache()

    # 5. Live auto-profile sanity check on Apple Mac hardware
    if sys.platform == "darwin":
        auto_prof = hardware.get_hardware_profile()
        assert auto_prof.has_videotoolbox_hwaccel is True
        assert auto_prof.best_hdr_encoder == "hevc_videotoolbox"
        assert auto_prof.has_mps is True
        assert auto_prof.torch_device == "mps"


def test_gate9_mps_photo_compositor_parity_clamping_and_broken_pipe(tmp_path: Path) -> None:
    """Gate 9: MPSPhotoCompositor parity (<6.0 MAE vs CPU Pillow), edge clamping, and BrokenPipeError shutdown."""
    import time
    from PIL import Image, ImageDraw

    from video_editing_ai.models.schema import KenBurnsConfig
    from video_editing_ai.renderer.ken_burns import _prepare_photo_scene_context, _render_photo_frame_from_context
    from video_editing_ai.renderer.mps_photo_compositor import MPSPhotoCompositor, stream_photo_frames_to_pipe

    photo_path = tmp_path / "gallery_sample.jpg"
    img = Image.new("RGB", (960, 720), (36, 48, 72))
    draw = ImageDraw.Draw(img)
    for y in range(0, 720, 4):
        c = int(40 + 170 * (y / 720.0))
        draw.line([(0, y), (960, y)], fill=(c, 80 + c // 3, 180 - c // 3), width=4)
    draw.rectangle([220, 140, 740, 560], fill=(215, 198, 172), outline=(45, 38, 30), width=8)
    draw.ellipse([370, 210, 590, 430], fill=(195, 76, 62))
    img.save(photo_path, quality=95)

    configs = [
        KenBurnsConfig(
            mode="parallax_gallery",
            start_zoom=1.0,
            end_zoom=1.18,
            focal_x=0.52,
            focal_y=0.45,
            overlay_fx="museum_spotlight",
            caption_title="SFMOMA Gallery",
            caption_subtitle="Fourth Floor Terrace",
            date_badge="2026-10-18",
        ),
        KenBurnsConfig(
            mode="full_bleed",
            start_zoom=1.0,
            end_zoom=1.35,
            focal_x=0.99,
            focal_y=0.99,
            overlay_fx="golden_light_leak",
        ),
    ]

    for kb in configs:
        ctx = _prepare_photo_scene_context(photo_path=photo_path, width=640, height=360, kb=kb)
        total_frames = 16
        f_idx = 6
        duration = 2.0
        fps = 8
        t_norm = f_idx / (total_frames - 1)

        cpu_frame = _render_photo_frame_from_context(
            ctx=ctx, t_norm=t_norm, f_idx=f_idx, duration=duration, fps=fps
        )
        # Test both auto device (MPS on Mac) and explicit CPU PyTorch fallback
        for dev_ovr in (None, "cpu"):
            comp = MPSPhotoCompositor(ctx=ctx, device_override=dev_ovr)
            try:
                batch = comp.render_batch_rgb(
                    frame_indices=[f_idx],
                    total_frames=total_frames,
                    duration=duration,
                    fps=fps,
                )
            finally:
                comp.close()
            assert batch.shape == (1, 360, 640, 3)
            mps_frame = batch[0].numpy()
            cpu_arr = np.asarray(cpu_frame, dtype=np.float32)
            mae = float(np.mean(np.abs(mps_frame.astype(np.float32) - cpu_arr)))
            assert mae < 6.0, f"Parity MAE too high ({mae:.2f}) for mode={kb.mode}, fx={kb.overlay_fx}, dev={dev_ovr}"

    # Verify BrokenPipeError tears down async pipe queue cleanly without deadlocking
    class _BrokenPipe:
        def write(self, data: object) -> int:
            raise BrokenPipeError("simulated downstream ffmpeg exit")

    ctx_pipe = _prepare_photo_scene_context(photo_path=photo_path, width=320, height=180, kb=configs[1])
    t0 = time.monotonic()
    stream_photo_frames_to_pipe(
        ctx=ctx_pipe,
        total_frames=16,
        duration=2.0,
        fps=8,
        pipe_stdin=_BrokenPipe(),  # type: ignore[arg-type]
        batch_size=4,
    )
    assert (time.monotonic() - t0) < 5.0


def test_gate10_uniform_sei_stripping_segment_baking_and_stream_copy_mux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 10: Uniform SEI stripping, segment overlay baking filtergraph, Step 3 -c copy & Step 4 -c:v copy."""
    import subprocess
    from PIL import Image

    from video_editing_ai import hardware
    from video_editing_ai.models.schema import (
        AudioTrackItem,
        EditTimeline,
        KenBurnsConfig,
        TimelineSettings,
        TitleOverlayItem,
        TransitionConfig,
        VisualTrackItem,
    )
    from video_editing_ai.renderer import ffmpeg_composer
    from video_editing_ai.renderer.ken_burns import build_segment_overlay_filtergraph, build_uniform_mux_args

    mux_args = build_uniform_mux_args(fps=30, hdr=True, v_enc_args=["-c:v", "hevc_videotoolbox"])
    assert "-map_metadata" in mux_args and "-1" in mux_args
    assert "-r" in mux_args and "30" in mux_args
    if hardware.has_ffmpeg_bsf("filter_units"):
        assert "-bsf:v" in mux_args and "filter_units=remove_types=39-40" in mux_args

    b_dummy = tmp_path / "b.mp4"
    t_dummy = tmp_path / "t.png"
    extra_in, f_parts, out_lbl = build_segment_overlay_filtergraph(
        base_label="[vbase]",
        hdr=True,
        next_input_idx=2,
        segment_broll=[(b_dummy, 0.25, 0.75)],
        segment_titles=[(t_dummy, 0.5, 1.0, 0.15, 0.15)],
    )
    assert extra_in[:2] == ["-i", str(b_dummy)]
    assert "-loop" in extra_in and str(t_dummy) in extra_in
    joined_fc = ";".join(f_parts)
    assert "overlay=0:0:format=yuv420p10" in joined_fc
    assert "between(t,0.250,1.000)" in joined_fc
    assert "between(t,0.500,1.500)" in joined_fc
    assert out_lbl == "[svhdr]"

    # Create real micro assets: 2s H.264 clip, photo, broll photo, BGM wav
    vid_src = tmp_path / "clip1.mov"
    broll_src = tmp_path / "broll_card.jpg"
    photo_src = tmp_path / "photo2.jpg"
    bgm_src = tmp_path / "bgm.wav"

    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=s=640x360:r=24:d=2.0",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2.0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(vid_src),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=4.0",
            "-ac", "2", str(bgm_src),
        ],
        check=True,
    )
    Image.new("RGB", (640, 360), (24, 90, 140)).save(photo_src, quality=92)
    Image.new("RGB", (640, 360), (140, 60, 40)).save(broll_src, quality=92)

    real_run = subprocess.run
    captured_cmds: list[list[str]] = []

    def _spy_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(cmd, list) and cmd and str(cmd[0]) == "ffmpeg":
            captured_cmds.append([str(c) for c in cmd])
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(ffmpeg_composer.subprocess, "run", _spy_run)

    for hdr_flag in (False, True):
        captured_cmds.clear()
        out_mp4 = tmp_path / f"final_baked_{'hdr' if hdr_flag else 'sdr'}.mp4"
        timeline = EditTimeline(
            project_name=f"Gate10 Baked Overlays ({'HDR' if hdr_flag else 'SDR'})",
            settings=TimelineSettings(width=640, height=360, fps=24, hdr=hdr_flag),
            visual_track=[
                VisualTrackItem(
                    id="v1_01",
                    track="V1",
                    type="video",
                    asset_id="vid_clip1",
                    source=str(vid_src),
                    source_in=0.0,
                    source_out=1.5,
                    timeline_start=0.0,
                    duration=1.5,
                    transition_out=TransitionConfig(type="none", duration=0.0),
                ),
                VisualTrackItem(
                    id="v1_02",
                    track="V1",
                    type="photo",
                    asset_id="img_photo2",
                    source=str(photo_src),
                    timeline_start=1.5,
                    duration=1.5,
                    ken_burns=KenBurnsConfig(mode="gallery_frame", start_zoom=1.0, end_zoom=1.08),
                    transition_in=TransitionConfig(type="none", duration=0.0),
                    transition_out=TransitionConfig(type="none", duration=0.0),
                ),
            ],
            broll_track=[
                VisualTrackItem(
                    id="v2_01",
                    track="V2",
                    type="photo",
                    asset_id="img_broll",
                    source=str(broll_src),
                    timeline_start=0.25,
                    duration=0.75,
                    ken_burns=KenBurnsConfig(mode="full_bleed"),
                    transition_in=TransitionConfig(type="none", duration=0.0),
                    transition_out=TransitionConfig(type="none", duration=0.0),
                )
            ],
            title_track=[
                TitleOverlayItem(
                    id="t3_01",
                    text="Baked Photo Title",
                    style="lower_third",
                    timeline_start=1.6,
                    duration=1.2,
                    fade_in=0.15,
                    fade_out=0.15,
                )
            ],
            audio_tracks=[
                AudioTrackItem(
                    id="bgm_01",
                    role="music",
                    source=str(bgm_src),
                    timeline_start=0.0,
                    duration=3.0,
                    volume=0.35,
                    duck_under_dialogue=False,
                )
            ],
        )

        rendered = ffmpeg_composer.render_timeline(
            timeline=timeline,
            output_mp4=out_mp4,
            work_dir=tmp_path / f"work_{'hdr' if hdr_flag else 'sdr'}",
        )
        assert rendered["video"].exists() and rendered["video"].stat().st_size > 2048

        concat_cmds = [c for c in captured_cmds if "-f" in c and "concat" in c]
        assert concat_cmds and "-c" in concat_cmds[0] and "copy" in concat_cmds[0], (
            f"Expected Step 3 stream-copy concat (hdr={hdr_flag}), got {concat_cmds}"
        )
        final_cmds = [c for c in captured_cmds if str(out_mp4.resolve()) in c]
        assert final_cmds and "-c:v" in final_cmds[-1] and "copy" in final_cmds[-1], (
            f"Expected Step 4 -c:v copy mux (hdr={hdr_flag}), got {final_cmds}"
        )
        if hdr_flag:
            assert "-tag:v" in final_cmds[-1] and "hvc1" in final_cmds[-1]

        err_check = real_run(
            ["ffmpeg", "-v", "error", "-i", str(out_mp4), "-f", "null", "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert err_check.returncode == 0
        assert err_check.stderr.strip() == "", f"FFmpeg decode errors on final output (hdr={hdr_flag}): {err_check.stderr}"


def test_gate11_parallel_dit_ingest_and_catalog_dedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 11: Parallel DITIngestAgent ordering, preview caching, and 0-reprobe catalog deduplication."""
    import subprocess
    from PIL import Image

    from video_editing_ai.agents.base import PipelineBlackboard
    from video_editing_ai.agents.dit_ingest_agent import DITIngestAgent
    from video_editing_ai.analyzer import media_probe

    media_dir = tmp_path / "media"
    proj_dir = tmp_path / "proj"
    media_dir.mkdir(parents=True, exist_ok=True)
    proj_dir.mkdir(parents=True, exist_ok=True)

    # Create deterministically named files: a_vid, b_photo, c_vid, d_photo, e_audio
    for v_name in ("a_vid.mov", "c_vid.mov"):
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=navy:s=320x180:r=24:d=0.6",
                "-f", "lavfi", "-i", "sine=frequency=520:sample_rate=48000:duration=0.6",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                str(media_dir / v_name),
            ],
            check=True,
        )
    Image.new("RGB", (640, 480), (200, 140, 90)).save(media_dir / "b_photo.jpg", quality=90)
    Image.new("RGB", (480, 640), (70, 120, 175)).save(media_dir / "d_photo.jpg", quality=90)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000:duration=1.0",
            "-ac", "2", str(media_dir / "e_audio.wav"),
        ],
        check=True,
    )

    bb = PipelineBlackboard(media_dir=media_dir, project_dir=proj_dir)
    res1 = DITIngestAgent().run(bb)
    assert bb.dit_manifest is not None
    assert [e.filename for e in bb.dit_manifest.entries] == [
        "a_vid.mov",
        "b_photo.jpg",
        "c_vid.mov",
        "d_photo.jpg",
        "e_audio.wav",
    ]
    assert res1["cache_hits_count"] == 0 and res1["cache_misses_count"] == 5
    photo_entries = [e for e in bb.dit_manifest.entries if e.media_type == "photo"]
    assert all(pe.preview_path and Path(pe.preview_path).exists() for pe in photo_entries)

    # Second run hits cache across all 5 files including photos
    bb2 = PipelineBlackboard(media_dir=media_dir, project_dir=proj_dir)
    res2 = DITIngestAgent().run(bb2)
    assert res2["cache_hits_count"] == 5 and res2["cache_misses_count"] == 0

    # Zero-reprobe verification in build_initial_catalog when dit_manifest is fresh
    probe_calls = {"video": 0, "photo": 0, "audio": 0}
    real_pv = media_probe.probe_video_file
    real_ip = media_probe.inspect_photo_file
    real_pa = media_probe.probe_audio_file

    def _spy_pv(*args: Any, **kwargs: Any) -> Any:
        probe_calls["video"] += 1
        return real_pv(*args, **kwargs)

    def _spy_ip(*args: Any, **kwargs: Any) -> Any:
        probe_calls["photo"] += 1
        return real_ip(*args, **kwargs)

    def _spy_pa(*args: Any, **kwargs: Any) -> Any:
        probe_calls["audio"] += 1
        return real_pa(*args, **kwargs)

    monkeypatch.setattr(media_probe, "probe_video_file", _spy_pv)
    monkeypatch.setattr(media_probe, "inspect_photo_file", _spy_ip)
    monkeypatch.setattr(media_probe, "probe_audio_file", _spy_pa)

    catalog = media_probe.build_initial_catalog(
        media_dir=media_dir,
        cache_dir=proj_dir / "cache",
        dit_manifest=bb2.dit_manifest,
    )
    assert probe_calls == {"video": 0, "photo": 0, "audio": 0}, f"Redundant probes occurred: {probe_calls}"
    assert [v.filename for v in catalog.videos] == ["a_vid.mov", "c_vid.mov"]
    assert [p.filename for p in catalog.photos] == ["b_photo.jpg", "d_photo.jpg"]
    assert [a.filename for a in catalog.audio_files] == ["e_audio.wav"]
    assert all(p.preview_path and Path(p.preview_path).exists() for p in catalog.photos)

    # Stale photo triggers re-inspection ONLY for that modified file
    Image.new("RGB", (642, 480), (10, 200, 120)).save(media_dir / "b_photo.jpg", quality=92)
    catalog_stale = media_probe.build_initial_catalog(
        media_dir=media_dir,
        cache_dir=proj_dir / "cache",
        dit_manifest=bb2.dit_manifest,
    )
    assert probe_calls == {"video": 0, "photo": 1, "audio": 0}
    assert catalog_stale.photos[0].width == 642
