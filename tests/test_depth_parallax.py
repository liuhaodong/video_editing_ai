"""Tests for neural monocular depth MPI parallax, disocclusion inpainting, and optical VFX modes."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import pytest


def _make_layered_scene(w: int = 640, h: int = 360) -> Image.Image:
    """Create a rich high-contrast scene with clear horizon, midground buildings, and foreground portrait."""
    img = Image.new("RGB", (w, h), (35, 125, 205))
    draw = ImageDraw.Draw(img)
    # Sky gradient band + ocean horizon
    draw.rectangle([0, int(h * 0.45), w, h], fill=(22, 162, 188))
    # Sand beach foreground
    draw.rectangle([0, int(h * 0.72), w, h], fill=(224, 196, 146))
    # Architectural midground block
    draw.rectangle([int(w * 0.12), int(h * 0.30), int(w * 0.34), int(h * 0.70)], fill=(235, 228, 215))
    # Bright sun reflection specular highlight
    draw.ellipse([int(w * 0.72), int(h * 0.10), int(w * 0.84), int(h * 0.22)], fill=(255, 246, 220))
    # Centered foreground subject
    draw.ellipse(
        [int(w * 0.42), int(h * 0.26), int(w * 0.58), int(h * 0.84)],
        fill=(195, 118, 92),
        outline=(250, 245, 238),
        width=4,
    )
    return img


def test_offline_depth_map_and_mpi_layers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify offline depth estimator, suitability gate, 4-layer MPI construction, and disk cache."""
    monkeypatch.setenv("AI_EDITOR_OFFLINE_DEPTH", "1")
    from video_editing_ai.renderer.depth_parallax import (
        build_mpi_depth_layers,
        build_rack_focus_assets,
        compute_depth_suitability,
        estimate_depth_map,
    )

    scene = _make_layered_scene(640, 360)
    cache_dir = tmp_path / "depth_cache"

    d_map = estimate_depth_map(
        scene,
        focal_point=(0.5, 0.55),
        cache_dir=cache_dir,
        photo_key="test_scene_01",
        target_size=(640, 360),
    )
    assert d_map.shape == (360, 640)
    assert d_map.dtype == np.float32
    assert float(d_map.min()) >= 0.0 and float(d_map.max()) <= 1.0
    assert (cache_dir / "depth_test_scene_01_640x360.png").exists()

    # Second call should read cached uint16 map instantaneously
    d_cached = estimate_depth_map(
        scene,
        focal_point=(0.5, 0.55),
        cache_dir=cache_dir,
        photo_key="test_scene_01",
        target_size=(640, 360),
    )
    assert np.allclose(d_map, d_cached, atol=1.0 / 65535.0 * 2.0)

    suit = compute_depth_suitability(d_map)
    assert 0.35 <= suit <= 1.0

    # Flat depth map should get low suitability amplitude
    flat_suit = compute_depth_suitability(np.full((120, 160), 0.5, dtype=np.float32))
    assert flat_suit < suit

    mpi = build_mpi_depth_layers(bg_sharp=scene, depth=d_map, scale_factor=1.0)
    assert len(mpi["layers"]) == 4
    assert all(layer.mode == "RGBA" and layer.size == (640, 360) for layer in mpi["layers"])
    t1, t2, t3 = mpi["thresholds"]
    assert 0.10 <= t1 < t2 < t3 <= 0.92

    rf = build_rack_focus_assets(
        raw_primary=scene,
        width=480,
        height=270,
        focal_point=(0.5, 0.5),
        scale_factor=0.5,
    )
    assert rf["sharp"].size == rf["bokeh"].size
    assert rf["depth_q"].ndim == 2
    assert 0.05 <= rf["focus_start"] <= 0.95
    assert 0.05 <= rf["focus_end"] <= 0.95


def test_cinematic_depth_and_new_vfx_modes_no_dark_tint_or_glitches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify MPI depth parallax preserves natural lighting (no dark navy tint) and all new VFX modes render cleanly."""
    monkeypatch.setenv("AI_EDITOR_OFFLINE_DEPTH", "1")
    from video_editing_ai.models.schema import KenBurnsConfig
    from video_editing_ai.renderer.ken_burns import _prepare_photo_scene_context, _render_photo_frame_from_context

    p = tmp_path / "beach_hero.jpg"
    scene = _make_layered_scene(800, 450)
    scene.save(p, quality=95)

    # Check natural luminance preservation in upgraded cinematic_depth_3d vs input mean
    kb_depth = KenBurnsConfig(mode="cinematic_depth_3d", overlay_fx="none")
    ctx_d = _prepare_photo_scene_context(p, width=480, height=270, kb=kb_depth)
    f0 = _render_photo_frame_from_context(ctx_d, t_norm=0.0, f_idx=0, duration=4.0, fps=24)
    f1 = _render_photo_frame_from_context(ctx_d, t_norm=1.0, f_idx=95, duration=4.0, fps=24)

    mean_src = float(np.asarray(scene, dtype=np.float32).mean())
    mean_f0 = float(np.asarray(f0, dtype=np.float32).mean())
    assert abs(mean_f0 - mean_src) < 12.0, f"Unexpected exposure/mud tint shift: src={mean_src:.1f}, out={mean_f0:.1f}"
    # Start & end frames must visibly differ due to stereo orbital motion + dolly
    assert float(np.abs(np.asarray(f1, dtype=np.float32) - np.asarray(f0, dtype=np.float32)).mean()) > 0.4

    # Archival color bloom must transition from low-saturation monochrome -> full color
    kb_bloom = KenBurnsConfig(mode="archival_color_bloom", overlay_fx="halation_bloom_35mm")
    ctx_b = _prepare_photo_scene_context(p, width=480, height=270, kb=kb_bloom)
    b0 = np.asarray(_render_photo_frame_from_context(ctx_b, t_norm=0.0, f_idx=0, duration=4.0, fps=24), dtype=np.float32)
    b1 = np.asarray(_render_photo_frame_from_context(ctx_b, t_norm=1.0, f_idx=95, duration=4.0, fps=24), dtype=np.float32)
    chroma_0 = float(np.abs(b0[:, :, 0] - b0[:, :, 2]).mean())
    chroma_1 = float(np.abs(b1[:, :, 0] - b1[:, :, 2]).mean())
    assert chroma_1 > chroma_0 + 15.0

    # Aperture rack focus & dolly zoom vertigo + caustic shimmer
    for m_name, fx_name in (
        ("aperture_rack_focus", "museum_spotlight"),
        ("dolly_zoom_vertigo", "caustic_water_shimmer"),
    ):
        kb = KenBurnsConfig(mode=m_name, overlay_fx=fx_name)  # type: ignore[arg-type]
        ctx = _prepare_photo_scene_context(p, width=480, height=270, kb=kb)
        fr = _render_photo_frame_from_context(ctx, t_norm=0.5, f_idx=12, duration=2.0, fps=24)
        assert fr.size == (480, 270)
        assert fr.getpixel((0, 135)) != (0, 0, 0)
        assert fr.getpixel((479, 135)) != (0, 0, 0)


def test_neural_depth_anything_v2_runs_if_available(tmp_path: Path) -> None:
    """Smoke-test Depth-Anything-V2 inference latency and shape when model weights are cached."""
    from video_editing_ai.renderer import depth_parallax

    bundle = depth_parallax._load_depth_anything_model()
    if bundle is None:
        pytest.skip("Depth-Anything-V2 model not cached locally")

    scene = _make_layered_scene(640, 360)
    t0 = time.perf_counter()
    d = depth_parallax.estimate_depth_map(scene, target_size=(640, 360))
    elapsed = time.perf_counter() - t0
    assert d.shape == (360, 640)
    assert elapsed < 10.0
