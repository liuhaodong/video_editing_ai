"""Regression tests for generalized editorial, companion pairing, travel-map, BGM extension, and title fixes."""

from __future__ import annotations

from pathlib import Path
import subprocess

from PIL import Image

from video_editing_ai.agents.base import PipelineBlackboard
from video_editing_ai.agents.music_supervisor_agent import probe_audio_diagnostics
from video_editing_ai.agents.picture_editors import MontageEditorAgent
from video_editing_ai.agents.plugins.camera_operator_agent import CameraOperatorAgent
from video_editing_ai.agents.plugins.sound_designer_agent import SoundDesignerMixerAgent
from video_editing_ai.models.schema import (
    AudioTrackItem,
    BeatGrid,
    ChapterPlan,
    CreativeBrief,
    CulledSubClip,
    EditTimeline,
    KenBurnsConfig,
    MediaIndex,
    PaperEdit,
    TimelineSettings,
    VisualTrackItem,
)
from video_editing_ai.renderer.titles import _sanitize_display_text
from video_editing_ai.renderer.travel_map import infer_transport_mode


def _make_portrait_video(out_path: Path, dur: float = 4.0) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=navy:s=270x480:r=15:d={dur}",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:d={dur}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out_path),
        ],
        check=True,
    )
    return out_path


def _make_photo(out_path: Path, size: tuple[int, int] = (300, 400)) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (80, 120, 160)).save(out_path, "JPEG", quality=90)
    return out_path


def _make_audio_mp3(out_path: Path, dur: float = 12.0, freq: int = 330) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=48000:d={dur}",
            "-c:a", "libmp3lame", "-b:a", "192k", str(out_path),
        ],
        check=True,
    )
    return out_path


def test_camera_operator_scopes_diptych_to_same_chapter(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    v_ch1 = _make_portrait_video(media_dir / "day1_clip.mp4")
    p_ch2 = _make_photo(media_dir / "day2_photo.jpg")

    mi = MediaIndex(
        project_title="Test Trip",
        source_directory=str(media_dir),
        created_at="2026-07-01T00:00:00Z",
        subclips=[
            CulledSubClip(
                subclip_id="vid_d1", source_asset_id="v1", source_path=str(v_ch1), filename="day1_clip.mp4",
                media_type="video", normalized_timestamp="2026-07-01T10:00:00",
                in_tc=0.0, out_tc=4.0, golden_in=0.0, golden_out=4.0, usable_in=0.0, usable_out=4.0, duration=4.0,
                orientation="portrait",
            ),
            CulledSubClip(
                subclip_id="ph_d2", source_asset_id="p2", source_path=str(p_ch2), filename="day2_photo.jpg",
                media_type="photo", normalized_timestamp="2026-07-03T15:00:00",
                in_tc=0.0, out_tc=4.0, golden_in=0.0, golden_out=4.0, usable_in=0.0, usable_out=4.0, duration=4.0,
                orientation="portrait",
            ),
        ],
    )
    paper = PaperEdit(
        project_title="Test Trip",
        story_arc_summary="Day 1 and Day 3",
        chapters=[
            ChapterPlan(chapter_id="ch1", title="Day 1", subtitle="", selected_subclip_ids=["vid_d1"]),
            ChapterPlan(chapter_id="ch2", title="Day 3", subtitle="", selected_subclip_ids=["ph_d2"]),
        ],
    )
    tl = EditTimeline(
        project_name="Test Trip",
        visual_track=[
            VisualTrackItem(id="c1", type="video", source=str(v_ch1), roll_type="b_roll", act="Day 1", timeline_start=0.0, duration=4.0, source_in=0.0, source_out=4.0),
            VisualTrackItem(id="c2", type="photo", source=str(p_ch2), roll_type="still", act="Day 3", timeline_start=4.0, duration=4.0, source_in=0.0, source_out=4.0),
        ],
    )
    bb = PipelineBlackboard(media_dir=media_dir, project_dir=tmp_path)
    bb.creative_brief = CreativeBrief(project_title="Test Trip", target_duration_sec=120.0)
    bb.media_index = mi
    bb.paper_edit = paper
    bb.timeline = tl

    CameraOperatorAgent().run(bb)
    # Must NOT borrow Day 3 photo into Day 1 portrait video
    assert bb.timeline is not None
    assert bb.timeline.visual_track[0].paired_visual_source is None


def test_extended_cut_preserves_longer_broll_videos(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    vid_p = _make_portrait_video(media_dir / "scenic_drive.mp4", dur=14.0)
    mi = MediaIndex(
        project_title="Long Feature",
        source_directory=str(media_dir),
        created_at="2026-07-01T00:00:00Z",
        subclips=[
            CulledSubClip(
                subclip_id="v_drive", source_asset_id="v1", source_path=str(vid_p), filename="scenic_drive.mp4",
                media_type="video", normalized_timestamp="2026-07-01T11:00:00",
                in_tc=0.0, out_tc=12.0, golden_in=0.0, golden_out=12.0, usable_in=0.0, usable_out=14.0, duration=12.0,
                roll_type="b_roll",
            )
        ],
    )
    paper = PaperEdit(
        project_title="Long Feature",
        story_arc_summary="Long drive",
        chapters=[ChapterPlan(chapter_id="ch1", title="Act I", subtitle="", editing_mode="dialogue_with_broll", selected_subclip_ids=["v_drive"])],
    )
    tl = EditTimeline(
        project_name="Long Feature",
        visual_track=[VisualTrackItem(id="clip_001", type="video", source=str(vid_p), roll_type="b_roll", act="Act I", timeline_start=0.0, duration=12.0, source_in=0.0, source_out=12.0)],
    )
    bb = PipelineBlackboard(media_dir=media_dir, project_dir=tmp_path)
    bb.creative_brief = CreativeBrief(project_title="Long Feature", target_duration_sec=540.0, max_broll_duration_sec=4.5)
    bb.media_index = mi
    bb.paper_edit = paper
    bgm_p = _make_audio_mp3(tmp_path / "bgm.mp3", dur=24.0)
    bb.beat_grid = BeatGrid(source_audio=str(bgm_p), duration=24.0, bpm=100.0, beat_times_sec=[i * 0.6 for i in range(40)], downbeat_times_sec=[i * 2.4 for i in range(10)])
    bb.timeline = tl

    MontageEditorAgent().run(bb)
    assert bb.timeline is not None
    assert bb.timeline.visual_track[0].duration >= 9.0


def test_sound_designer_remasters_short_bgm_to_full_timeline(tmp_path: Path) -> None:
    bgm_dir = tmp_path / "bgm_library"
    short_src = _make_audio_mp3(bgm_dir / "track_a.mp3", dur=10.0)
    short_master = _make_audio_mp3(bgm_dir / "mastered_soundtrack.mp3", dur=10.0)
    ph = _make_photo(tmp_path / "media" / "img.jpg", size=(640, 480))

    tl = EditTimeline(
        project_name="Long Audio Test",
        settings=TimelineSettings(width=1280, height=720, fps=30),
        visual_track=[
            VisualTrackItem(id="c1", type="photo", source=str(ph), roll_type="still", timeline_start=0.0, duration=28.0, ken_burns=KenBurnsConfig(mode="cinematic_depth_3d")),
        ],
        audio_tracks=[AudioTrackItem(id="bgm_main", role="music", source=str(short_master), timeline_start=0.0, duration=10.0, volume=0.28)],
    )
    bb = PipelineBlackboard(media_dir=tmp_path / "media", project_dir=tmp_path)
    bb.creative_brief = CreativeBrief(project_title="Long Audio Test", target_duration_sec=28.0)
    bb.timeline = tl

    SoundDesignerMixerAgent().run(bb)
    assert short_src.exists()
    remastered = Path(bb.timeline.audio_tracks[0].source)
    assert remastered.exists()
    assert probe_audio_diagnostics(remastered)["duration_sec"] >= 27.5


def test_music_supervisor_ai_driven_selection_and_mix_styling(tmp_path: Path, monkeypatch) -> None:
    from unittest.mock import MagicMock
    from video_editing_ai.agents import music_supervisor_agent as msa

    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True)
    proj_dir = tmp_path / "proj"

    track_a = _make_audio_mp3(tmp_path / "src_a.mp3", dur=12.0)
    track_b = _make_audio_mp3(tmp_path / "src_b.mp3", dur=12.0)

    def _fake_search(query_keywords, max_results=5, timeout_sec=6.0):
        kw = ",".join(query_keywords)
        if "tropical" in kw:
            return [{"track_id": "t_trop", "title": "Coral Sun", "composer_or_artist": "Artist A", "genre_mood": "Tropical", "license_info": "CC BY", "source_url": "https://example.com/a.mp3", "tempo_bpm_estimate": 102.0, "mood_match_score": 9.8, "notes": "Sunny"}]
        return [{"track_id": "t_warm", "title": "Candlelight", "composer_or_artist": "Artist B", "genre_mood": "Warm", "license_info": "CC BY", "source_url": "https://example.com/b.mp3", "tempo_bpm_estimate": 88.0, "mood_match_score": 9.6, "notes": "Dinner"}]

    def _fake_dl(url, dest_path, timeout_sec=15.0):
        src = track_a if url.endswith("a.mp3") else track_b
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(src.read_bytes())
        return True

    monkeypatch.setattr(msa, "search_online_royalty_free_catalog", _fake_search)
    monkeypatch.setattr(msa, "_try_download_free_bgm", _fake_dl)

    agent = msa.MusicSupervisorAgent()
    ai_mock = MagicMock(
        side_effect=[
            {
                "mood_profile": "sunlit_caribbean_anniversary",
                "chapter_search_queries": [["tropical", "steel drum"], ["warm", "acoustic guitar"]],
                "recommended_bgm_volume": 0.68,
                "recommended_speech_duck_db": -9.0,
                "direction_notes": "Breezy island groove into intimate anniversary strings.",
            },
            {
                "selected_track_ids": ["t_trop", "t_warm"],
                "ai_rationale": "Contrasts bright beach groove with warm acoustic evening.",
            },
        ]
    )
    monkeypatch.setattr(agent, "ask_ai_creative_json", ai_mock)

    bb = PipelineBlackboard(media_dir=media_dir, project_dir=proj_dir)
    bb.creative_brief = CreativeBrief(project_title="Cancun Test", target_duration_sec=20.0)
    res = agent.run(bb)

    assert Path(res["selected_track_path"]).exists()
    assert bb.bgm_selection is not None
    assert bb.bgm_selection.target_mood_profile == "sunlit_caribbean_anniversary"
    assert abs(bb.bgm_selection.recommended_bgm_volume - 0.68) < 1e-4
    assert abs(bb.bgm_selection.recommended_speech_duck_db - (-9.0)) < 1e-4
    assert [c.track_id for c in bb.bgm_selection.candidates][:2] == ["t_trop", "t_warm"]


def test_transport_inference_and_title_sanitization() -> None:
    assert infer_transport_mode(46.9, "San Diego Safari Park", "San Diego Bay") == "drive"
    assert infer_transport_mode(28.0, "Ferry Terminal", "Harbor Bay") == "boat"
    assert infer_transport_mode(19.0, "Palo Alto", "Half Moon Bay") == "drive"
    assert "★" not in _sanitize_display_text("★FAV Airborne Dolphin ☆")
    assert "FAV · " in _sanitize_display_text("★FAV Airborne Dolphin")
