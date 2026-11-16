"""Tests for NVIDIA CUDA / NVDEC / NVENC acceleration, parallel SHA1-cached kinetic titles, and uniform HLG concat."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch

from video_editing_ai import hardware
from video_editing_ai.models.schema import (
    EditTimeline,
    KenBurnsConfig,
    TimelineSettings,
    TitleOverlayItem,
    TransitionConfig,
    VisualTrackItem,
)
from video_editing_ai.renderer import ffmpeg_composer
from video_editing_ai.renderer.ken_burns import _prepare_photo_scene_context, _render_photo_frame_from_context
from video_editing_ai.renderer.kinetic_titles import compute_kinetic_roi
from video_editing_ai.renderer.mps_photo_compositor import MPSPhotoCompositor
from video_editing_ai.renderer.titles import render_title_frame


def _make_sample_photo(path: Path, width: int = 800, height: int = 600) -> Path:
    img = Image.new("RGB", (width, height), (34, 48, 72))
    draw = ImageDraw.Draw(img)
    for y in range(0, height, 4):
        c = int(40 + 180 * (y / max(1, height)))
        draw.line([(0, y), (width, y)], fill=(c, 85 + c // 4, 210 - c // 3), width=4)
    draw.rectangle([width // 5, height // 5, (width * 4) // 5, (height * 4) // 5], fill=(212, 188, 152))
    draw.ellipse([width // 3, height // 4, (width * 2) // 3, (height * 3) // 4], fill=(184, 68, 56))
    img.save(path, quality=94)
    return path


def test_hardware_policy_cuda_detection_and_render_worker_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify AI_EDITOR_ACCEL policies (auto/cuda/mps/cpu) and platform-aware render worker scaling."""
    try:
        monkeypatch.delenv("AI_EDITOR_MAX_WORKERS", raising=False)
        monkeypatch.setenv("AI_EDITOR_ACCEL", "cpu")
        hardware.reset_hardware_cache()
        cpu_prof = hardware.get_hardware_profile()
        assert cpu_prof.torch_device == "cpu"
        assert cpu_prof.has_cuda is False
        assert cpu_prof.has_mps is False
        assert cpu_prof.has_cuda_hwaccel is False
        assert cpu_prof.cuda_vram_gb == 0.0
        assert hardware.select_torch_device("cpu") == "cpu"

        if torch.cuda.is_available():
            monkeypatch.setenv("AI_EDITOR_ACCEL", "auto")
            hardware.reset_hardware_cache()
            auto_prof = hardware.get_hardware_profile()
            assert auto_prof.torch_device == "cuda"
            assert auto_prof.has_cuda is True
            assert auto_prof.has_mps is False
            assert auto_prof.cuda_vram_gb > 0.0
            assert auto_prof.discrete_gpu_name != ""
            assert hardware.select_torch_device("cuda") == "cuda"
            assert hardware.get_render_worker_count(auto_prof) >= 4
        else:
            assert hardware.select_torch_device("cuda") in ("cpu", "cuda")

        # Synthetic profile check for render worker scaling
        mac_prof = hardware.HardwareProfile(
            platform="darwin",
            arch="x86_64",
            cpu_count=16,
            accel_policy="auto",
            has_mps=True,
            torch_device="mps",
            has_amd_opencl=True,
            opencl_device_spec="Apple:GPU:1",
            has_videotoolbox_hwaccel=True,
            has_scale_vt=True,
            has_zscale=True,
            has_filter_units_bsf=True,
            best_hdr_encoder="hevc_videotoolbox",
            best_sdr_encoder="h264_videotoolbox",
            vt_supports_qscale_hdr=False,
            vt_supports_qscale_sdr=False,
            discrete_gpu_name="AMD Radeon Pro",
        )
        assert hardware.get_render_worker_count(mac_prof) == 4

        cuda_prof = hardware.HardwareProfile(
            platform="win32",
            arch="AMD64",
            cpu_count=32,
            accel_policy="auto",
            has_mps=False,
            torch_device="cuda",
            has_amd_opencl=False,
            opencl_device_spec="",
            has_videotoolbox_hwaccel=False,
            has_scale_vt=False,
            has_zscale=True,
            has_filter_units_bsf=True,
            best_hdr_encoder="hevc_nvenc",
            best_sdr_encoder="h264_nvenc",
            vt_supports_qscale_hdr=False,
            vt_supports_qscale_sdr=False,
            discrete_gpu_name="NVIDIA GeForce RTX 4080 SUPER",
            has_cuda=True,
            has_cuda_hwaccel=True,
            cuda_vram_gb=16.0,
        )
        assert hardware.get_render_worker_count(cuda_prof) == 8
    finally:
        monkeypatch.undo()
        hardware.reset_hardware_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_cuda_photo_compositor_concurrent_fp16_parity(tmp_path: Path) -> None:
    """Verify multi-stream FP16 CUDA photo compositing across concurrent threads matches CPU within <6.0 MAE."""
    photo_path = _make_sample_photo(tmp_path / "cuda_parity.jpg")
    modes = ["floating_tilt_plate", "cinematic_depth_3d", "glassmorphic_pedestal"]

    def _run_mode(mode: str) -> tuple[str, float]:
        kb = KenBurnsConfig(
            mode=mode,  # type: ignore[arg-type]
            start_zoom=1.0,
            end_zoom=1.14,
            focal_x=0.52,
            focal_y=0.44,
            overlay_fx="none",
            caption_title="CUDA Parity Check",
            caption_subtitle="FP16 Multi-Stream",
            date_badge="2026-04-21",
        )
        ctx = _prepare_photo_scene_context(photo_path=photo_path, width=640, height=360, kb=kb)
        comp = MPSPhotoCompositor(ctx=ctx, device_override="cuda")
        try:
            assert comp.device.type == "cuda"
            assert comp.work_dtype == torch.float16
            frame_indices = [4, 18, 28]
            batch = comp.render_batch_rgb(frame_indices=frame_indices, total_frames=36, duration=1.5, fps=24)
            assert batch.shape == (len(frame_indices), 360, 640, 3)
            max_mae = 0.0
            for b_i, f_idx in enumerate(frame_indices):
                ref = _render_photo_frame_from_context(
                    ctx=ctx,
                    t_norm=f_idx / 35.0,
                    f_idx=f_idx,
                    duration=1.5,
                    fps=24,
                )
                mae = float(np.mean(np.abs(batch[b_i].numpy().astype(np.float32) - np.asarray(ref, dtype=np.float32))))
                max_mae = max(max_mae, mae)
            return mode, max_mae
        finally:
            comp.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_run_mode, [*modes, "floating_tilt_plate"]))

    for mode, mae in results:
        assert mae < 6.0, f"CUDA FP16 parity MAE {mae:.2f} >= 6.0 for mode={mode}"


def test_render_title_frame_crop_roi_exact_equality() -> None:
    """Cropping before HDR NumPy clamp must be bit-for-bit identical to cropping afterwards."""
    item = TitleOverlayItem(
        id="roi_eq",
        style="editorial_caption",
        theme="apple_tv_glass",
        motion_preset="masked_rise",
        text="Gallery of Mirrors · 1678",
        subtitle="357 mirrors catching 17 gardens windows along a 73-meter vault",
        overline="ACT II · CURATORIAL NOTE",
        timeline_start=0.0,
        duration=3.0,
        backdrop_blur=True,
        hdr_safe=True,
    )
    w, h = 960, 540
    roi = compute_kinetic_roi(item, width=w, height=h)
    rx, ry, rw, rh = roi
    full_crop = render_title_frame(item, width=w, height=h, t_norm=0.45, hdr_safe=True).crop((rx, ry, rx + rw, ry + rh))
    direct_roi = render_title_frame(item, width=w, height=h, t_norm=0.45, hdr_safe=True, crop_roi=roi)
    assert direct_roi.size == (rw, rh)
    assert np.array_equal(np.asarray(full_crop), np.asarray(direct_roi))


def test_kinetic_title_parallel_sha1_cache_reuse(tmp_path: Path) -> None:
    """Second render_timeline reuses SHA1-cached kinetic .mov files with identical mtime."""
    vid = tmp_path / "src.mp4"
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
            "color=c=darkslategray:s=640x360:r=15:d=3.0",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3.0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(vid),
        ],
        check=True,
    )

    tl = EditTimeline(
        project_name="Title Cache Reuse",
        settings=TimelineSettings(width=640, height=360, fps=15, hdr=False),
        visual_track=[
            VisualTrackItem(
                id="v1",
                type="video",
                source=str(vid),
                source_in=0.0,
                source_out=2.6,
                duration=2.6,
                timeline_start=0.0,
                transition_in=TransitionConfig(type="none", duration=0.0),
                transition_out=TransitionConfig(type="none", duration=0.0),
            )
        ],
        title_track=[
            TitleOverlayItem(
                id="lt_1",
                style="lower_third",
                theme="apple_tv_glass",
                motion_preset="masked_rise",
                text="Courtyard Arrival",
                timeline_start=0.1,
                duration=1.0,
                backdrop_blur=True,
                hdr_safe=False,
            ),
            TitleOverlayItem(
                id="lt_2",
                style="exif_stamp",
                theme="a24_minimal",
                motion_preset="typewriter_telemetry",
                text="SONY A7C II · 24mm",
                timeline_start=1.3,
                duration=1.0,
                backdrop_blur=False,
                hdr_safe=False,
            ),
        ],
    )

    work_dir = tmp_path / "cache"
    out_1 = tmp_path / "out_1.mp4"
    out_2 = tmp_path / "out_2.mp4"
    ffmpeg_composer.render_timeline(tl, out_1, work_dir=work_dir)
    titles_dir = work_dir / "titles"
    movs_1 = sorted(titles_dir.glob("kin_title_*.mov"))
    assert len(movs_1) == 2
    mtimes_1 = [p.stat().st_mtime_ns for p in movs_1]

    ffmpeg_composer.render_timeline(tl, out_2, work_dir=work_dir)
    movs_2 = sorted(titles_dir.glob("kin_title_*.mov"))
    assert [p.stat().st_mtime_ns for p in movs_2] == mtimes_1


def test_hdr_uniform_tv_range_and_zero_displaymatrix_across_mixed_sources(tmp_path: Path) -> None:
    """
    Rotated source (-display_rotation 90 at source_in=0.0), full-range yuvj420p source, and photo plate
    must produce identical color_range='tv', zero DisplayMatrix side_data, and 100% uniform hvcC for -c copy.
    """
    rot_vid = tmp_path / "rot_portrait.mp4"
    pc_vid = tmp_path / "motion_fullrange.mp4"
    photo = _make_sample_photo(tmp_path / "plate.jpg", width=640, height=360)

    # Synthetic stream tagged with -90 Display Matrix rotation (simulating smartphone portrait MP4 at t=0.0)
    raw_rot = tmp_path / "raw_rot.mp4"
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
            "color=c=navy:s=640x360:r=24:d=1.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=520:sample_rate=48000:duration=1.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(raw_rot),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-display_rotation:v:0",
            "-90",
            "-i",
            str(raw_rot),
            "-c",
            "copy",
            str(rot_vid),
        ],
        check=True,
    )
    # Synthetic full-range (pc / yuvj420p) motion photo clip
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
            "color=c=olive:s=640x360:r=24:d=1.2,format=yuvj420p",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuvj420p",
            "-color_range",
            "pc",
            "-an",
            str(pc_vid),
        ],
        check=True,
    )

    work_dir = tmp_path / "hdr_mix_cache"
    out_mp4 = tmp_path / "hdr_mix_final.mp4"
    tl = EditTimeline(
        project_name="HDR Uniformity",
        settings=TimelineSettings(width=640, height=360, fps=24, hdr=True),
        visual_track=[
            VisualTrackItem(
                id="clip_rot",
                type="video",
                source=str(rot_vid),
                source_in=0.0,
                source_out=1.0,
                duration=1.0,
                timeline_start=0.0,
                transition_in=TransitionConfig(type="none", duration=0.0),
                transition_out=TransitionConfig(type="none", duration=0.0),
            ),
            VisualTrackItem(
                id="clip_pc",
                type="video",
                source=str(pc_vid),
                source_in=0.0,
                source_out=1.0,
                duration=1.0,
                timeline_start=1.0,
                transition_in=TransitionConfig(type="none", duration=0.0),
                transition_out=TransitionConfig(type="none", duration=0.0),
            ),
            VisualTrackItem(
                id="clip_photo",
                type="photo",
                source=str(photo),
                duration=1.0,
                timeline_start=2.0,
                ken_burns=KenBurnsConfig(mode="full_bleed", overlay_fx="none"),
                transition_in=TransitionConfig(type="none", duration=0.0),
                transition_out=TransitionConfig(type="none", duration=0.0),
            ),
        ],
    )

    ffmpeg_composer.render_timeline(tl, out_mp4, work_dir=work_dir)
    assert out_mp4.exists() and out_mp4.stat().st_size > 2048

    seg_paths = sorted((work_dir / "segments").glob("seg_*.mp4"))
    assert len(seg_paths) == 3

    extradata_sigs = []
    for sp in seg_paths:
        pr = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_data",
                "-of",
                "json",
                str(sp),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        s = json.loads(pr.stdout)["streams"][0]
        assert s.get("width") == 640 and s.get("height") == 360
        assert s.get("color_range") == "tv", f"{sp.name} had color_range={s.get('color_range')}"
        assert not s.get("side_data_list"), f"{sp.name} retained side_data_list={s.get('side_data_list')}"
        extradata_sigs.append(s.get("extradata"))

    assert len(set(extradata_sigs)) == 1, "All mixed-source HDR segments must share identical hvcC extradata"
