"""Kinetic frame-animated RGBA (.mov qtrle) overlay generator & frosted-glass pill masks."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Optional

from PIL import Image, ImageDraw

from ..models.schema import TitleOverlayItem
from .titles import (
    compute_glass_pill_rect,
    compute_overlay_bbox_norm,
    render_title_frame,
)

FULL_CANVAS_STYLES: tuple[str, ...] = (
    "opening_card",
    "closing_card",
    "knockout_stencil_opening",
    "credits_colophon_card",
    "act_banner",
)

BOTTOM_SCRIM_STYLES: tuple[str, ...] = (
    "lower_third",
    "route_map_card",
    "curved_gps_ribbon",
    "magazine_chapter_plate",
)


def ease_out_expo(x: float) -> float:
    """Exponential ease-out curve for crisp editorial settlements."""
    t = max(0.0, min(1.0, float(x)))
    return 1.0 if t >= 1.0 else 1.0 - math.pow(2.0, -10.0 * t)


def ease_out_cubic(x: float) -> float:
    """Cubic ease-out curve for masked baseline rises."""
    t = max(0.0, min(1.0, float(x)))
    return 1.0 - (1.0 - t) ** 3


def ease_in_cubic(x: float) -> float:
    """Cubic ease-in curve."""
    t = max(0.0, min(1.0, float(x)))
    return t * t * t


def ease_in_out_cubic(x: float) -> float:
    """Smooth S-curve easing."""
    t = max(0.0, min(1.0, float(x)))
    return 4.0 * t * t * t if t < 0.5 else 1.0 - math.pow(-2.0 * t + 2.0, 3) / 2.0


def _even_down(val: int, minimum: int = 0) -> int:
    return max(minimum, int(val) - (int(val) % 2))


def compute_kinetic_roi(
    item: TitleOverlayItem,
    width: int = 1920,
    height: int = 1080,
) -> tuple[int, int, int, int]:
    """
    Compute even-aligned pixel ROI (rx, ry, rw, rh) for localized RGBA kinetic overlays.
    Styles with full-width bottom gradient scrims span full canvas width cleanly with vertical crop.
    """
    w_even = _even_down(width, 16)
    h_even = _even_down(height, 16)
    if item.style in FULL_CANVAS_STYLES:
        return (0, 0, w_even, h_even)

    bbox = compute_overlay_bbox_norm(item.style)
    pill = compute_glass_pill_rect(item, width, height)
    pad_x = max(16, int(round(width * 0.04)))
    pad_y = max(16, int(round(height * 0.04)))

    if item.style in BOTTOM_SCRIM_STYLES:
        # Preserve seamless full-width gradient scrim while cropping away empty upper 65-75%
        y_top_bbox = int(math.floor(bbox[1] * height)) - pad_y
        y_top_scrim = int(math.floor(height * 0.72))
        y_top = min(y_top_bbox, y_top_scrim, (pill[1] - pad_y) if pill else y_top_bbox)
        ry = max(0, _even_down(y_top, 0))
        rh = _even_down(height - ry, 16)
        return (0, ry, w_even, rh)

    x1 = min(int(math.floor(bbox[0] * width)), pill[0] if pill else width) - pad_x
    y1 = min(int(math.floor(bbox[1] * height)), pill[1] if pill else height) - pad_y
    x2 = max(int(math.ceil(bbox[2] * width)), (pill[0] + pill[2]) if pill else 0) + pad_x
    y2 = max(int(math.ceil(bbox[3] * height)), (pill[1] + pill[3]) if pill else 0) + pad_y

    rx = max(0, min(w_even - 16, _even_down(x1, 0)))
    ry = max(0, min(h_even - 16, _even_down(y1, 0)))
    rw = min(w_even - rx, _even_down(max(16, x2 - rx + 1), 16))
    rh = min(h_even - ry, _even_down(max(16, y2 - ry + 1), 16))
    return (rx, ry, rw, rh)


def render_glass_mask_png(
    rect: tuple[int, int, int, int, int],
    output_png: str | Path,
) -> Path:
    """Render a 2x-supersampled rounded-pill alpha mask PNG ('L' mode, size rw x rh) for FFmpeg alphamerge."""
    _, _, rw, rh, rad = rect
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rw_e = _even_down(rw, 16)
    rh_e = _even_down(rh, 16)
    ss_mask = Image.new("L", (rw_e * 2, rh_e * 2), 0)
    draw = ImageDraw.Draw(ss_mask)
    draw.rounded_rectangle(
        [0, 0, rw_e * 2 - 1, rh_e * 2 - 1],
        radius=max(4, int(rad) * 2),
        fill=255,
    )
    mask = ss_mask.resize((rw_e, rh_e), Image.Resampling.LANCZOS)
    mask.save(out_path, "PNG")
    return out_path


def render_kinetic_title_mov(
    item: TitleOverlayItem,
    output_mov: str | Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    hdr_safe: Optional[bool] = None,
    transparent_glass_body: bool = False,
) -> tuple[Path, tuple[int, int, int, int]]:
    """
    Generate an ROI-cropped lossless RGBA kinetic title clip (.mov using QuickTime qtrle)
    by streaming per-frame Pillow renders to FFmpeg over stdin.
    """
    out_path = Path(output_mov)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fps_safe = max(1, int(fps))
    n_frames = max(2, int(round(max(0.25, float(item.duration)) * fps_safe)))
    rx, ry, rw, rh = compute_kinetic_roi(item, width=width, height=height)
    crop_needed = (rx != 0) or (ry != 0) or (rw != width) or (rh != height)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{rw}x{rh}",
        "-r",
        str(fps_safe),
        "-i",
        "-",
        "-c:v",
        "qtrle",
        "-pix_fmt",
        "argb",
        str(out_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stderr is not None

    crop_arg = (rx, ry, rw, rh) if crop_needed else None
    dur = max(0.25, float(item.duration))
    fi = min(dur * 0.42, max(0.04, float(item.fade_in)))
    fo = min(dur * 0.42, max(0.04, float(item.fade_out)))
    entry_span = min(dur * 0.44, max(0.72, fi * 1.8))
    is_static_hold_style = (
        getattr(item, "style", "") != "travel_map_pip_card"
        and getattr(item, "motion_preset", "masked_rise") != "static"
    )
    settled_bytes: Optional[bytes] = None
    try:
        for f_idx in range(n_frames):
            t_norm = float(f_idx) / float(max(1, n_frames - 1))
            t_sec = t_norm * dur
            is_settled = (
                is_static_hold_style
                and (t_sec >= entry_span + 0.02)
                and (t_sec <= dur - fo - 0.02)
                and not (getattr(item, "motion_preset", "") == "typewriter_telemetry" and t_norm < 0.45)
            )
            if is_settled and settled_bytes is not None:
                proc.stdin.write(settled_bytes)
                continue
            frame = render_title_frame(
                item=item,
                width=width,
                height=height,
                t_norm=t_norm,
                hdr_safe=hdr_safe,
                transparent_glass_body=transparent_glass_body,
                crop_roi=crop_arg,
            )
            if frame.size != (rw, rh):
                frame = frame.resize((rw, rh), Image.Resampling.BILINEAR)
            raw_bytes = frame.tobytes()
            if is_settled:
                settled_bytes = raw_bytes
            proc.stdin.write(raw_bytes)
        proc.stdin.close()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        raise

    stderr_bytes = proc.stderr.read()
    rc = proc.wait()
    if rc != 0:
        err_msg = stderr_bytes.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg qtrle kinetic title encode failed (rc={rc}): {err_msg}")

    return out_path, (rx, ry, rw, rh)
