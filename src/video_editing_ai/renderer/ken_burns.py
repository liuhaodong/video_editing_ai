"""Sub-pixel Ken Burns and Gallery Frame photo-to-video renderer using Pillow + FFmpeg."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

from ..models.schema import KenBurnsConfig, TransitionConfig


def _ease_in_out_cubic(t: float) -> float:
    """Smooth cubic easing for natural camera motion."""
    if t < 0.5:
        return 4.0 * t * t * t
    p = 2.0 * t - 2.0
    return 0.5 * p * p * p + 1.0


def build_gallery_frame_image(
    src_path: str | Path,
    width: int = 1920,
    height: int = 1080,
    border_px: Optional[int] = None,
) -> Image.Image:
    """Create a MrAlexTech-style uncropped gallery frame over a Gaussian-blurred ambient background."""
    img = Image.open(src_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    scale_factor = width / 1920.0
    b_px = border_px if border_px is not None else max(8, int(round(14 * scale_factor)))
    blur_radius = max(18, int(round(28 * scale_factor)))
    shadow_margin = max(20, int(round(32 * scale_factor)))
    shadow_offset = max(4, int(round(7 * scale_factor)))
    shadow_blur = max(10, int(round(18 * scale_factor)))

    # 1. Ambient blurred & darkened background filling the 16:9 canvas
    bg = ImageOps.fit(img, (width, height), method=Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    dimmer = Image.new("RGB", (width, height), (0, 0, 0))
    bg = Image.blend(bg, dimmer, 0.22)

    # 2. Uncropped proportional foreground
    aspect = img.width / max(1, img.height)
    if aspect < 1.0:
        max_w, max_h = int(width * 0.70), int(height * 0.84)
    else:
        max_w, max_h = int(width * 0.84), int(height * 0.82)

    fg = img.copy()
    fg.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

    # 3. Crisp white photo border
    fw, fh = fg.width + b_px * 2, fg.height + b_px * 2
    framed = Image.new("RGBA", (fw, fh), (255, 255, 255, 255))
    framed.paste(fg, (b_px, b_px))

    # 4. Soft floating 3D drop shadow
    sw, sh = fw + shadow_margin * 2, fh + shadow_margin * 2
    shadow = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow)
    s_draw.rectangle(
        [
            shadow_margin,
            shadow_margin + shadow_offset,
            shadow_margin + fw,
            shadow_margin + fh + shadow_offset,
        ],
        fill=(0, 0, 0, 165),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
    shadow.paste(framed, (shadow_margin, shadow_margin), framed)

    # 5. Composite centered on ambient background
    px, py = (width - sw) // 2, (height - sh) // 2
    bg_rgba = bg.convert("RGBA")
    bg_rgba.paste(shadow, (px, py), shadow)
    return bg_rgba.convert("RGB")


def _resolve_kb_endpoints(
    kb: KenBurnsConfig,
) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    """Resolve preset names into concrete (start_zoom, end_zoom, start_center, end_center)."""
    preset = kb.preset
    if preset == "static":
        return (1.0, 1.0, (0.5, 0.5), (0.5, 0.5))
    if preset == "floating_gallery":
        return (1.0, 1.06, (0.5, 0.5), (0.5, 0.485))
    if preset == "zoom_in":
        return (1.0, max(1.12, kb.end_zoom), (0.5, 0.5), kb.end_center)
    if preset == "zoom_out":
        return (max(1.14, kb.start_zoom), 1.0, kb.start_center, (0.5, 0.5))
    if preset == "pan_left_to_right":
        return (1.12, 1.12, (0.44, 0.5), (0.56, 0.5))
    if preset == "pan_right_to_left":
        return (1.12, 1.12, (0.56, 0.5), (0.44, 0.5))
    if preset == "tilt_up":
        return (1.12, 1.12, (0.5, 0.56), (0.5, 0.44))
    return (kb.start_zoom, kb.end_zoom, kb.start_center, kb.end_center)


def render_photo_clip(
    photo_path: str | Path,
    output_mp4: str | Path,
    duration: float = 3.0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    ken_burns: Optional[KenBurnsConfig] = None,
    transition_in: Optional[TransitionConfig] = None,
    transition_out: Optional[TransitionConfig] = None,
) -> Path:
    """Render a photo into an MP4 video clip with sub-pixel Ken Burns or Gallery Frame animation."""
    kb = ken_burns or KenBurnsConfig()
    out_path = Path(output_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare oversampled base canvas (1.25x target resolution) for razor-sharp sub-pixel resampling
    oversample = 1.25
    work_w = int(round(width * oversample))
    work_h = int(round(height * oversample))

    if kb.mode == "gallery_frame":
        base_img = build_gallery_frame_image(photo_path, width=work_w, height=work_h, border_px=kb.border_px)
    else:
        raw = Image.open(photo_path)
        raw = ImageOps.exif_transpose(raw).convert("RGB")
        base_img = ImageOps.fit(raw, (work_w, work_h), method=Image.Resampling.LANCZOS)

    s_zoom, e_zoom, (sx, sy), (ex, ey) = _resolve_kb_endpoints(kb)
    total_frames = max(1, int(round(duration * fps)))

    # Build FFmpeg fade filter for smooth head/tail transitions
    vf_filters: list[str] = ["format=yuv420p", "setsar=1:1"]
    if transition_in and transition_in.type == "fade_black" and transition_in.duration > 0:
        vf_filters.append(f"fade=t=in:st=0:d={min(duration * 0.45, transition_in.duration):.3f}")
    if transition_out and transition_out.type == "fade_black" and transition_out.duration > 0:
        fade_d = min(duration * 0.45, transition_out.duration)
        st = max(0.0, duration - fade_d)
        vf_filters.append(f"fade=t=out:st={st:.3f}:d={fade_d:.3f}")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-t",
        f"{duration:.3f}",
        "-vf",
        ",".join(vf_filters),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-video_track_timescale",
        "30000",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    try:
        for f_idx in range(total_frames):
            t_norm = f_idx / max(1, total_frames - 1)
            eased = _ease_in_out_cubic(t_norm)

            cur_zoom = s_zoom + (e_zoom - s_zoom) * eased
            cx = sx + (ex - sx) * eased
            cy = sy + (ey - sy) * eased

            crop_w = work_w / cur_zoom
            crop_h = work_h / cur_zoom

            # Clamp center so crop window stays strictly inside [0, work_w] x [0, work_h]
            half_w = crop_w / 2.0
            half_h = crop_h / 2.0
            center_px = min(max(cx * work_w, half_w), work_w - half_w)
            center_py = min(max(cy * work_h, half_h), work_h - half_h)

            x0 = center_px - half_w
            y0 = center_py - half_h
            x1 = center_px + half_w
            y1 = center_py + half_h

            # Floating-point EXTENT transform performs sub-pixel resampling without integer stepping jitter
            frame = base_img.transform(
                (width, height),
                Image.Transform.EXTENT,
                (x0, y0, x1, y1),
                resample=Image.Resampling.BICUBIC,
            )
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        stderr_bytes = proc.stderr.read() if proc.stderr else b""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"FFmpeg photo clip render failed: {stderr_bytes.decode('utf-8', errors='replace')}")

    return out_path
