"""Gate 14: Cinema 2.5D Travel Map & Cartography Engine Verification."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from PIL.TiffImagePlugin import IFDRational
import pytest

from video_editing_ai.agents.orchestrator import run_multi_agent_pipeline
from video_editing_ai.analyzer.media_probe import _extract_exif_gps_and_alt, inspect_photo_file, probe_video_file
from video_editing_ai.exporters.nle_export import export_fcpxml
from video_editing_ai.models.schema import (
    ChapterPlan,
    EditTimeline,
    SoundDesignReport,
    TimelineSettings,
    TitleOverlayItem,
    TravelMapManifest,
    VisualTrackItem,
)
from video_editing_ai.planner.timeline_surgeon import apply_surgical_edit
from video_editing_ai.renderer.titles import (
    HDR_DIFFUSE_WHITE_PEAK_RGB,
    compute_overlay_bbox_norm,
    render_title_frame,
)
from video_editing_ai.renderer.travel_map import (
    CART_THEME_SPECS,
    find_gazetteer_sequence,
    haversine_km,
    infer_transport_mode,
    initial_bearing_deg,
    lookup_gazetteer_place,
    render_travel_map_clip,
    render_travel_map_frame,
    render_travel_map_showcase_sheet,
    resolve_journey_waypoints,
    resolve_travel_map_style,
)


def _make_test_video(out_path: Path, duration: float = 4.0, color: str = "steelblue", freq: int = 440, iso6709_loc: str | None = None) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=640x360:r=15:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={freq}:sample_rate=48000:d={duration}",
    ]
    if iso6709_loc:
        cmd.extend(["-metadata", f"location={iso6709_loc}"])
    cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out_path)])
    subprocess.run(cmd, check=True)
    return out_path


def _make_test_photo(out_path: Path, lat_deg: float | None = None, lon_deg: float | None = None, alt_m: float | None = None) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (480, 320), (36, 62, 94))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 450, 290], fill=(190, 140, 88))
    draw.ellipse([160, 80, 320, 240], fill=(235, 200, 145))

    if lat_deg is not None and lon_deg is not None:
        exif = Image.Exif()
        gps_ifd = {
            1: "N" if lat_deg >= 0 else "S",
            2: (
                IFDRational(int(abs(lat_deg)), 1),
                IFDRational(int((abs(lat_deg) % 1) * 60), 1),
                IFDRational(int((((abs(lat_deg) * 60) % 1) * 60) * 100), 100),
            ),
            3: "E" if lon_deg >= 0 else "W",
            4: (
                IFDRational(int(abs(lon_deg)), 1),
                IFDRational(int((abs(lon_deg) % 1) * 60), 1),
                IFDRational(int((((abs(lon_deg) * 60) % 1) * 60) * 100), 100),
            ),
        }
        if alt_m is not None:
            gps_ifd[5] = b"\x00" if alt_m >= 0 else b"\x01"
            gps_ifd[6] = IFDRational(int(round(abs(alt_m) * 10)), 10)
        exif[0x8825] = gps_ifd
        img.save(out_path, "JPEG", quality=92, exif=exif)
    else:
        img.save(out_path, "JPEG", quality=92)
    return out_path


def test_exif_and_quicktime_altitude_extraction(tmp_path: Path) -> None:
    photo_p = _make_test_photo(tmp_path / "mt_fuji_stop.jpg", lat_deg=35.3606, lon_deg=138.7274, alt_m=2305.5)
    with Image.open(photo_p) as im:
        lat, lon, alt = _extract_exif_gps_and_alt(im)
    assert lat is not None and abs(lat - 35.3606) < 0.01
    assert lon is not None and abs(lon - 138.7274) < 0.01
    assert alt is not None and abs(alt - 2305.5) < 0.5

    photo_asset = inspect_photo_file(photo_p, previews_dir=tmp_path / "previews")
    assert photo_asset.gps_alt is not None and abs(photo_asset.gps_alt - 2305.5) < 0.5

    vid_p = _make_test_video(tmp_path / "tokyo_clip.mp4", duration=2.0, iso6709_loc="+35.6762+139.6503+042.5/")
    vid_meta = probe_video_file(vid_p)
    assert vid_meta["gps_lat"] is not None and abs(float(vid_meta["gps_lat"]) - 35.6762) < 0.01
    assert vid_meta["gps_lon"] is not None and abs(float(vid_meta["gps_lon"]) - 139.6503) < 0.01
    assert vid_meta["gps_alt"] is not None and abs(float(vid_meta["gps_alt"]) - 42.5) < 0.5


def test_gazetteer_geodesic_and_transport_inference() -> None:
    tk = lookup_gazetteer_place("Arrival in Tokyo")
    ky = lookup_gazetteer_place("Kyoto Temples")
    assert tk is not None and ky is not None
    dist = haversine_km(tk[0], tk[1], ky[0], ky[1])
    assert 340.0 <= dist <= 400.0
    brg = initial_bearing_deg(tk[0], tk[1], ky[0], ky[1])
    assert 0.0 <= brg < 360.0

    # Generic non-geographic labels must NOT hallucinate coordinates
    assert lookup_gazetteer_place("Chapter 1") is None
    assert lookup_gazetteer_place("Overview") is None
    assert lookup_gazetteer_place("Family Dinner") is None

    seq = find_gazetteer_sequence("Road trip from San Francisco to Monterey and Big Sur")
    assert [lbl for _, _, _, lbl in seq] == ["San Francisco", "Monterey", "Big Sur"]

    assert infer_transport_mode(9500.0, "San Francisco", "Tokyo") == "flight"
    assert infer_transport_mode(1.2, "Studio", "Garden") == "walk"
    assert infer_transport_mode(28.0, "Ferry Terminal", "Harbor Bay") == "boat"
    assert infer_transport_mode(370.0, "Tokyo Shinkansen", "Kyoto Station") == "train"

    # Untaxed untagged chapters stay < 2 stops in auto mode
    untagged = [
        ChapterPlan(chapter_id="ch_01", title="Opening Moments", subtitle="", location_tag="Featured Highlight"),
        ChapterPlan(chapter_id="ch_02", title="Afternoon Laughs", subtitle="", location_tag="Overview"),
    ]
    assert resolve_journey_waypoints(untagged, force_synthetic_if_missing=False) == []
    assert len(resolve_journey_waypoints(untagged, force_synthetic_if_missing=True)) == 2


def test_render_travel_map_styles_motion_callout_and_hdr(tmp_path: Path) -> None:
    hero_jpg = _make_test_photo(tmp_path / "kyoto_hero.jpg")
    waypoints = [
        {"label": "Tokyo", "lat": 35.6762, "lon": 139.6503, "alt_m": 40.0},
        {"label": "Kyoto", "lat": 35.0116, "lon": 135.7681, "alt_m": 120.0, "hero_thumbnail_path": str(hero_jpg)},
    ]

    for style_name in CART_THEME_SPECS:
        f_early = render_travel_map_frame(waypoints=waypoints, style=style_name, width=640, height=360, leg_index=0, t_norm=0.15, hdr_safe=True)
        f_late = render_travel_map_frame(waypoints=waypoints, style=style_name, width=640, height=360, leg_index=0, t_norm=0.88, hdr_safe=True)
        arr0 = np.asarray(f_early, dtype=np.uint8)
        arr1 = np.asarray(f_late, dtype=np.uint8)
        assert arr0.shape == (360, 640, 3)
        assert int(arr0.max()) <= HDR_DIFFUSE_WHITE_PEAK_RGB
        assert int(arr1.max()) <= HDR_DIFFUSE_WHITE_PEAK_RGB
        assert float(np.mean(np.abs(arr1.astype(np.float32) - arr0.astype(np.float32)))) > 1.5

    mp4_path = render_travel_map_clip(
        waypoints=waypoints,
        output_mp4=tmp_path / "flyover.mp4",
        style="archival_expedition_parchment",
        duration=1.2,
        width=640,
        height=360,
        fps=12,
        with_audio=True,
    )
    assert mp4_path.exists() and mp4_path.stat().st_size > 2000

    sheet_path = render_travel_map_showcase_sheet(tmp_path / "showcase.jpg", waypoints=waypoints, cell_w=480, cell_h=270)
    assert sheet_path.exists() and sheet_path.stat().st_size > 5000

    assert resolve_travel_map_style("auto", title_theme="archival_35mm") == "archival_expedition_parchment"
    assert resolve_travel_map_style("auto", title_theme="kinfolk_magazine") == "kinfolk_minimal_monograph"


def test_travel_map_pip_card_title_frame_and_hdr() -> None:
    bbox = compute_overlay_bbox_norm("travel_map_pip_card")
    assert bbox == [0.62, 0.06, 0.95, 0.40]

    item = TitleOverlayItem(
        id="title_pip_01",
        style="travel_map_pip_card",
        theme="apple_tv_glass",
        eyebrow="ROUTE TELEMETRY",
        overline="ROUTE TELEMETRY",
        text="Tokyo -> Kyoto",
        subtitle="363 km | TRAIN",
        timeline_start=1.0,
        duration=3.0,
        hdr_safe=True,
        map_waypoints=[
            {"label": "Tokyo", "lat": 35.6762, "lon": 139.6503, "alt_m": 40.0, "completed": True, "active": False},
            {"label": "Kyoto", "lat": 35.0116, "lon": 135.7681, "alt_m": 120.0, "completed": True, "active": True},
        ],
    )
    frame = render_title_frame(item, width=640, height=360, t_norm=0.55, hdr_safe=True)
    arr = np.asarray(frame, dtype=np.uint8)
    assert arr.shape == (360, 640, 4)
    assert int(arr[:, :, :3].max()) <= HDR_DIFFUSE_WHITE_PEAK_RGB
    assert int((arr[:, :, 3] > 20).sum()) > 1000


def test_pipeline_travel_map_v2_qc_render_and_fcpxml(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    proj_dir = tmp_path / "proj"

    _make_test_video(media_dir / "01_tokyo_departure.mp4", duration=4.2, color="steelblue", freq=330, iso6709_loc="+35.6762+139.6503+042.0/")
    _make_test_video(media_dir / "02_hakone_mountain.mp4", duration=4.2, color="seagreen", freq=440, iso6709_loc="+35.2324+139.1069+725.0/")
    _make_test_video(media_dir / "03_kyoto_arrival.mp4", duration=4.2, color="sienna", freq=520, iso6709_loc="+35.0116+135.7681+110.0/")
    _make_test_photo(media_dir / "04_kyoto_shrine.jpg", lat_deg=35.0116, lon_deg=135.7681, alt_m=120.0)

    bb = run_multi_agent_pipeline(
        media_dir=media_dir,
        project_dir=proj_dir,
        user_prompt="Create a scenic journey highlight reel traveling from Tokyo through Hakone to Kyoto",
        interview_answers={
            "target_duration_sec": "30",
            "story_arc_mode": "chronological_cinema",
            "travel_map_mode": "flyover_interlude",
            "travel_map_style": "midnight_vox_atlas",
            "photo_mode": "parallax_gallery",
        },
        width=640,
        height=360,
        fps=15,
        hdr=False,
        render_video=True,
        use_antigravity=False,
        skip_whisper=True,
        skip_auto_bgm=True,
    )

    tm_json = proj_dir / "travel_map_manifest.json"
    assert tm_json.exists()
    manifest = TravelMapManifest.from_json_file(tm_json)
    assert manifest.enabled is True
    assert len(manifest.waypoints) >= 2
    assert any(seg.placement == "v2_flyover" for seg in manifest.segments)

    assert bb.timeline is not None
    v2_maps = [b for b in bb.timeline.broll_track if b.id.startswith("broll_map_")]
    assert v2_maps
    for b in v2_maps:
        assert Path(b.source).exists()
        assert any(
            (v.timeline_start - 0.02) <= b.timeline_start and (b.timeline_start + b.duration) <= (v.timeline_start + v.duration + 0.02)
            for v in bb.timeline.visual_track
        )

    sd_report = SoundDesignReport.from_json_file(proj_dir / "sound_design_plan.json")
    assert any(c.cue_type == "cartography_flyover_whoosh" for c in sd_report.cues)

    assert bb.visual_qc_report is not None
    assert bb.visual_qc_report.overlay_overlaps_detected == 0

    final_mp4 = proj_dir / "final_story.mp4"
    assert final_mp4.exists() and final_mp4.stat().st_size > 10000

    fcpxml_path = proj_dir / "timeline.fcpxml"
    if not fcpxml_path.exists():
        export_fcpxml(bb.timeline, fcpxml_path)
    xml_text = fcpxml_path.read_text(encoding="utf-8")
    assert 'lane="1"' in xml_text


def test_timeline_surgeon_and_skill_cli_travel_map(tmp_path: Path) -> None:
    proj = tmp_path / "surgical_proj"
    vid = _make_test_video(tmp_path / "clip.mp4", duration=4.5, color="navy")
    tl = EditTimeline(
        project_title="Map Test",
        settings=TimelineSettings(width=640, height=360, fps=15, hdr=False),
        visual_track=[
            VisualTrackItem(
                id="clip_001",
                type="video",
                roll_type="b_roll",
                source=str(vid),
                timeline_start=0.0,
                duration=4.5,
                source_in=0.0,
                source_out=4.5,
            )
        ],
    )
    proj.mkdir(parents=True, exist_ok=True)
    tl.to_yaml(proj / "timeline.yaml")

    res_v2 = apply_surgical_edit(
        proj,
        [
            {
                "action": "insert-travel-map",
                "origin": "Tokyo",
                "destination": "Kyoto",
                "style": "aerial_relief_hybrid",
                "placement": "v2_flyover",
                "duration": 1.8,
            }
        ],
    )
    assert res_v2["operations_count"] == 1
    tl2 = EditTimeline.from_yaml(proj / "timeline.yaml")
    assert len(tl2.broll_track) == 1 and tl2.broll_track[0].id.startswith("broll_map_")
    assert Path(tl2.broll_track[0].source).exists()

    res_v3 = apply_surgical_edit(
        proj,
        [
            {
                "action": "insert-travel-map",
                "origin": "San Francisco",
                "destination": "Monterey",
                "style": "kinfolk_minimal_monograph",
                "placement": "v3_pip",
                "clip_id": "clip_001",
                "duration": 2.0,
            }
        ],
    )
    assert res_v3["operations_count"] == 1
    tl3 = EditTimeline.from_yaml(proj / "timeline.yaml")
    assert any(t.style == "travel_map_pip_card" for t in tl3.title_track)

    repo_root = Path(__file__).resolve().parent.parent
    skill_script = repo_root / ".agent" / "skills" / "ai-video-editor" / "scripts" / "ai_video_studio_skill.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    out_sheet = tmp_path / "cli_sheet.jpg"
    subprocess.run(
        [sys.executable, str(skill_script), "preview-travel-maps", "--project-dir", str(proj), "--output", str(out_sheet)],
        env=env,
        check=True,
    )
    assert out_sheet.exists() and out_sheet.stat().st_size > 5000

    out_mp4 = tmp_path / "cli_flyover.mp4"
    subprocess.run(
        [
            sys.executable,
            str(skill_script),
            "render-travel-map",
            "--project-dir",
            str(proj),
            "--origin",
            "Zurich",
            "--destination",
            "Zermatt",
            "--style",
            "midnight_vox_atlas",
            "--duration",
            "1.0",
            "--output",
            str(out_mp4),
        ],
        env=env,
        check=True,
    )
    assert out_mp4.exists() and out_mp4.stat().st_size > 2000


def test_arclength_constant_speed_and_great_circle_geodesic() -> None:
    from video_editing_ai.renderer.travel_map import _ArcLengthPath, _great_circle_latlon_pts

    path = _ArcLengthPath([(0.0, 0.0), (100.0, 0.0), (101.0, 0.0), (102.0, 0.0), (200.0, 0.0)])
    assert abs(path.total_len - 200.0) < 1e-4
    for u, expected_x in ((0.25, 50.0), (0.50, 100.0), (0.75, 150.0)):
        sampled = path.sample_prefix(u)
        assert abs(sampled[-1][0] - expected_x) < 0.5
        assert abs(sampled[-1][1]) < 1e-4

    heading, bank = path.tangent_and_curvature(0.5)
    assert isinstance(heading, float) and isinstance(bank, float)

    gc_pts = _great_circle_latlon_pts(40.64, -73.78, 51.47, -0.45)
    assert len(gc_pts) >= 3
    mid_lat, _mid_lon = gc_pts[len(gc_pts) // 2]
    assert mid_lat > 0.5 * (40.64 + 51.47) + 2.5

    trans_pacific = [
        {"label": "San Francisco", "lat": 37.62, "lon": -122.38, "alt_m": 4.0},
        {"label": "Tokyo", "lat": 35.55, "lon": 139.78, "alt_m": 6.0},
    ]
    frame = render_travel_map_frame(
        waypoints=trans_pacific,
        style="satellite_hybrid_4k",
        width=640,
        height=360,
        leg_index=0,
        t_norm=0.55,
        transport_mode="flight",
        hdr_safe=True,
        allow_tile_fetch=False,
    )
    arr = np.asarray(frame, dtype=np.uint8)
    assert arr.shape == (360, 640, 3)
    assert int(arr.max()) <= HDR_DIFFUSE_WHITE_PEAK_RGB


def test_all_vehicles_and_satellite_hybrid_4k_hdr(tmp_path: Path) -> None:
    assert resolve_travel_map_style("auto", transport_mode="drive") == "midnight_vox_atlas"
    assert resolve_travel_map_style("auto", transport_mode="train") == "midnight_vox_atlas"
    assert resolve_travel_map_style("auto", transport_mode="walk") == "midnight_vox_atlas"
    assert resolve_travel_map_style("auto", transport_mode="flight") == "satellite_hybrid_4k"
    assert resolve_travel_map_style("auto", transport_mode="boat") == "satellite_hybrid_4k"
    assert "satellite_hybrid_4k" in CART_THEME_SPECS

    waypoints = [
        {"label": "Palm Springs", "lat": 33.8303, "lon": -116.5453, "alt_m": 146.0},
        {"label": "San Diego", "lat": 32.7157, "lon": -117.1611, "alt_m": 19.0},
    ]
    styles = ("google_road_navigator", "aerial_relief_hybrid", "satellite_hybrid_4k", "midnight_vox_atlas")
    modes = ("flight", "drive", "train", "boat", "walk")

    for mode in modes:
        for style in styles:
            frame_hdr = render_travel_map_frame(
                waypoints=waypoints,
                style=style,
                width=640,
                height=360,
                leg_index=0,
                t_norm=0.58,
                transport_mode=mode,
                hdr_safe=True,
                allow_tile_fetch=False,
                cache_dir=tmp_path / "map_cache",
            )
            arr_hdr = np.asarray(frame_hdr, dtype=np.uint8)
            assert arr_hdr.shape == (360, 640, 3)
            assert int(arr_hdr.max()) <= HDR_DIFFUSE_WHITE_PEAK_RGB

    frame_sdr = render_travel_map_frame(
        waypoints=waypoints,
        style="satellite_hybrid_4k",
        width=640,
        height=360,
        leg_index=0,
        t_norm=0.58,
        transport_mode="drive",
        hdr_safe=False,
        allow_tile_fetch=False,
        cache_dir=tmp_path / "map_cache",
    )
    arr_sdr = np.asarray(frame_sdr, dtype=np.uint8)
    assert arr_sdr.shape == (360, 640, 3)
    assert int(arr_sdr.max()) > 0


def test_google_earth_dynamic_zoom_and_multimodal_leg_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_editing_ai.renderer.travel_map import _draw_aa_polyline, render_google_earth_flyover_frame

    # Ensure NaN/Inf off-horizon points never raise warnings or errors
    buf = np.zeros((100, 100, 4), dtype=np.uint8)
    _draw_aa_polyline(buf, [(10.0, 10.0), (float("nan"), 20.0), (50.0, 50.0)], (255, 255, 255, 255), 2)

    # Verify offline-safe dynamic zoom frame & 'name' fallback label
    monkeypatch.setenv("AI_EDITOR_OFFLINE_MAPS", "1")
    wps = [
        {"name": "Lucerne Pier", "lat": 47.0502, "lon": 8.3093, "alt_m": 434.0},
        {"name": "Vitznau", "lat": 47.0094, "lon": 8.4836, "alt_m": 436.0},
        {"name": "Rigi Kulm", "lat": 47.0567, "lon": 8.4855, "alt_m": 1798.0},
    ]
    f_boat = render_google_earth_flyover_frame(
        waypoints=wps, style="satellite_hybrid_4k", width=640, height=360,
        global_t=0.25, leg_modes=["boat", "train"], hdr_safe=True, allow_tile_fetch=False, cache_dir=tmp_path / "tiles",
    )
    f_train = render_google_earth_flyover_frame(
        waypoints=wps, style="satellite_hybrid_4k", width=640, height=360,
        global_t=0.80, leg_index=1, leg_modes=["boat", "train"], hdr_safe=True, allow_tile_fetch=False, cache_dir=tmp_path / "tiles",
    )
    assert np.asarray(f_boat).shape == (360, 640, 3)
    assert np.asarray(f_train).shape == (360, 640, 3)
    assert int(np.asarray(f_boat).max()) <= HDR_DIFFUSE_WHITE_PEAK_RGB
    assert int(np.asarray(f_train).max()) <= HDR_DIFFUSE_WHITE_PEAK_RGB

    # Focused per-stop street walk leg over multi-stop itinerary
    zurich_walk = [
        {"label": "Ruby Mimi HB", "lat": 47.3768, "lon": 8.5398, "alt_m": 408.0},
        {"label": "Landesmuseum", "lat": 47.3790, "lon": 8.5406, "alt_m": 409.0},
        {"label": "ETH Zurich", "lat": 47.3763, "lon": 8.5477, "alt_m": 460.0},
    ]
    f_walk = render_google_earth_flyover_frame(
        waypoints=zurich_walk, style="google_road_navigator", width=640, height=360,
        global_t=0.85, leg_index=0, leg_modes=["walk", "walk"], hdr_safe=True, allow_tile_fetch=False, cache_dir=tmp_path / "tiles",
    )
    assert np.asarray(f_walk).shape == (360, 640, 3)
    assert int(np.asarray(f_walk).max()) <= HDR_DIFFUSE_WHITE_PEAK_RGB

    # Surgical v1_interlude placement splices standalone leg before stop clip and ripple-shifts downstream
    proj = tmp_path / "interlude_proj"
    vid = _make_test_video(tmp_path / "stop1.mp4", duration=4.0, color="teal")
    tl = EditTimeline(
        project_title="Zurich Playbook Test",
        settings=TimelineSettings(width=640, height=360, fps=15, hdr=False),
        visual_track=[
            VisualTrackItem(
                id="clip_stop_01",
                type="video",
                roll_type="b_roll",
                source=str(vid),
                timeline_start=0.0,
                duration=4.0,
                source_in=0.0,
                source_out=4.0,
            )
        ],
    )
    proj.mkdir(parents=True, exist_ok=True)
    tl.to_yaml(proj / "timeline.yaml")
    res_v1 = apply_surgical_edit(
        proj,
        {
            "action": "insert-travel-map",
            "clip_id": "clip_stop_01",
            "placement": "v1_interlude",
            "waypoints": zurich_walk,
            "leg_index": 0,
            "leg_modes": ["walk", "walk"],
            "transport_mode": "walk",
            "style": "google_road_navigator",
            "duration": 2.0,
        },
    )
    assert res_v1["operations_count"] == 1
    tl_after = EditTimeline.from_yaml(proj / "timeline.yaml")
    assert len(tl_after.visual_track) == 2
    assert tl_after.visual_track[0].id.startswith("clip_map_")
    assert abs(tl_after.visual_track[0].duration - 2.0) < 0.05
    assert tl_after.visual_track[1].id == "clip_stop_01"
    assert abs(tl_after.visual_track[1].timeline_start - 2.0) < 0.05


def test_cuda_travel_map_warp_parity_and_ssl_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CUDA-accelerated flat & 3D curved globe warps match CPU Lanczos4 (<1.5 MAE) and SSL context is cached."""
    import torch
    from video_editing_ai.renderer import travel_map as tm

    assert tm._build_ssl_ctx() is tm._build_ssl_ctx()
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")

    arr = np.zeros((1080, 1920, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(20, 220, 1920, dtype=np.uint8)[None, :]
    arr[:, :, 1] = np.linspace(30, 190, 1080, dtype=np.uint8)[:, None]
    arr[:, :, 2] = 110
    im = Image.fromarray(arr)

    for g_str in (0.0, 0.85):
        monkeypatch.setenv("AI_EDITOR_ACCEL", "cuda")
        gpu_out, _, _ = tm._apply_camera_homography(
            im, 1280, 720, (960.0, 540.0), zoom=1.15, pitch_deg=22.0, bank_deg=0.0, globe_strength=g_str
        )
        monkeypatch.setenv("AI_EDITOR_ACCEL", "cpu")
        cpu_out, _, _ = tm._apply_camera_homography(
            im, 1280, 720, (960.0, 540.0), zoom=1.15, pitch_deg=22.0, bank_deg=0.0, globe_strength=g_str
        )
        mae = float(np.abs(gpu_out.astype(np.float32) - cpu_out.astype(np.float32)).mean())
        assert mae < 1.5, f"CUDA/CPU map warp parity drift for g_str={g_str}: MAE={mae:.3f}"


def test_glow_beacon_dot_default_for_all_ground_modes_and_vfx_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify glowing beacon dot is used by default for train/walk/drive across both renderers and VFX skips map clips."""
    from video_editing_ai.agents.base import PipelineBlackboard
    from video_editing_ai.agents.plugins.vfx_agent import VFXAgent
    from video_editing_ai.renderer import travel_map as tm

    beacon_calls: list[tuple[float, float]] = []
    orig_beacon = tm._draw_glow_beacon_dot

    def _spy_beacon(*args, **kwargs):
        beacon_calls.append((float(kwargs.get("cx", 0.0)), float(kwargs.get("cy", 0.0))))
        return orig_beacon(*args, **kwargs)

    monkeypatch.setattr(tm, "_draw_glow_beacon_dot", _spy_beacon)
    wps = [
        {"label": "Zurich HB", "lat": 47.3779, "lon": 8.5402, "alt_m": 408.0},
        {"label": "ETH Zurich", "lat": 47.3764, "lon": 8.5476, "alt_m": 446.0},
    ]
    for mode in ("train", "walk", "drive"):
        beacon_calls.clear()
        tm.render_travel_map_frame(
            waypoints=wps, style="midnight_vox_atlas", width=320, height=180,
            leg_index=0, t_norm=0.5, transport_mode=mode, allow_tile_fetch=False,
        )
        assert len(beacon_calls) == 1, f"Expected beacon dot in render_travel_map_frame for mode={mode}"

        beacon_calls.clear()
        tm.render_google_earth_flyover_frame(
            waypoints=wps, style="midnight_vox_atlas", width=320, height=180,
            global_t=0.5, leg_index=0, leg_modes=[mode], transport_mode=mode, allow_tile_fetch=False,
        )
        assert len(beacon_calls) == 1, f"Expected beacon dot in render_google_earth_flyover_frame for mode={mode}"

    # Verify VFXAgent leaves travel-map clips untouched (no contrast_pop / brightness blowout)
    proj = tmp_path / "vfx_map_proj"
    proj.mkdir(parents=True, exist_ok=True)
    tl = EditTimeline(
        project_name="vfx_map_test",
        visual_track=[
            VisualTrackItem(
                id="clip_map_01",
                type="video",
                roll_type="b_roll",
                source=str(proj / "cache" / "travel_maps" / "map_01.mp4"),
                timeline_start=0.0,
                duration=6.0,
                source_in=0.0,
                source_out=6.0,
                narrative_note="2.5D Travel Map (midnight_vox_atlas)",
            ),
            VisualTrackItem(
                id="clip_broll_01",
                type="video",
                roll_type="b_roll",
                source=str(proj / "broll.mp4"),
                timeline_start=6.0,
                duration=5.0,
                source_in=0.0,
                source_out=5.0,
                narrative_note="Scenic harbor panning shot",
            ),
        ],
    )
    bb = PipelineBlackboard(media_dir=proj, project_dir=proj, user_prompt="test", use_antigravity=False)
    bb.timeline = tl
    VFXAgent().run(bb)
    target_ids = {fx.target_clip_id for fx in bb.timeline.effects_track}
    assert "clip_map_01" not in target_ids
    assert "clip_broll_01" in target_ids




