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
    MediaIndex,
    PaperEdit,
    SubtitleConfig,
    SubtitleItem,
    TimelineMarker,
    TimelineSettings,
    TitleOverlayItem,
    VisualTrackItem,
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
