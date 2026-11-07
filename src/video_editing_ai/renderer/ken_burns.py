"""Sub-pixel 2.5D Parallax, Diptych Split-Screen, and Museum Gallery Frame renderer using Pillow + FFmpeg."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

from ..models.schema import KenBurnsConfig, TransitionConfig


def _ease_in_out_cubic(t: float) -> float:
    """Smooth cubic S-curve easing for natural camera & parallax motion."""
    if t < 0.5:
        return 4.0 * t * t * t
    p = 2.0 * t - 2.0
    return 0.5 * p * p * p + 1.0


def _load_plaque_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, max(10, size))
            except Exception:
                continue
    return ImageFont.load_default()


def _build_double_matte_card(
    img: Image.Image,
    max_w: int,
    max_h: int,
    scale_factor: float,
    border_px: Optional[int] = None,
) -> Image.Image:
    """Create a foreground photo card with a double museum matte (warm ivory outer + antique gold inner hairline) and 3D drop shadow."""
    b_px = border_px if border_px is not None else max(10, int(round(16 * scale_factor)))
    gold_pad = max(3, int(round(4 * scale_factor)))
    shadow_margin = max(20, int(round(32 * scale_factor)))
    shadow_offset = max(5, int(round(8 * scale_factor)))
    shadow_blur = max(10, int(round(18 * scale_factor)))

    fg = img.copy()
    fg.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

    fw = fg.width + (b_px + gold_pad) * 2
    fh = fg.height + (b_px + gold_pad) * 2

    # Warm museum ivory outer matte
    framed = Image.new("RGBA", (fw, fh), (250, 248, 242, 255))
    f_draw = ImageDraw.Draw(framed)
    # Antique gold inner fillet hairline around the artwork
    f_draw.rectangle(
        [b_px - 2, b_px - 2, fw - b_px + 1, fh - b_px + 1],
        outline=(212, 175, 55, 235),
        width=max(2, int(round(2.5 * scale_factor))),
    )
    framed.paste(fg, (b_px + gold_pad, b_px + gold_pad))

    # Floating 3D drop shadow
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
        fill=(0, 0, 0, 175),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
    shadow.paste(framed, (shadow_margin, shadow_margin), framed)
    return shadow


def _build_museum_plaque_rgba(
    width: int,
    height: int,
    caption_title: str,
    caption_subtitle: str = "",
    camera_badge: str = "",
) -> tuple[Image.Image, int, int]:
    """Render a sleek museum exhibition plaque RGBA overlay and return (plaque_img, x, y)."""
    s = width / 1920.0
    f_badge = _load_plaque_font(int(round(17 * s)))
    f_title = _load_plaque_font(int(round(30 * s)))
    f_sub = _load_plaque_font(int(round(22 * s)))

    dummy = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    d_draw = ImageDraw.Draw(dummy)

    tw = d_draw.textbbox((0, 0), caption_title, font=f_title)[2]
    sw = d_draw.textbbox((0, 0), caption_subtitle, font=f_sub)[2] if caption_subtitle else 0
    bw = d_draw.textbbox((0, 0), camera_badge, font=f_badge)[2] if camera_badge else 0

    pad_x = int(round(34 * s))
    pw = min(int(width * 0.78), max(tw, sw, bw) + pad_x * 2 + int(round(24 * s)))
    ph = int(round((124 if (caption_subtitle and camera_badge) else (98 if caption_subtitle else 74)) * s))

    plaque = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    draw = ImageDraw.Draw(plaque)

    # Frosted dark museum glass body
    radius = int(round(16 * s))
    draw.rounded_rectangle(
        [0, 0, pw - 1, ph - 1],
        radius=radius,
        fill=(12, 15, 24, 224),
        outline=(218, 178, 72, 215),
        width=max(2, int(round(2.2 * s))),
    )

    # Gold vertical exhibition accent bar on left
    bar_x = int(round(18 * s))
    bar_w = max(4, int(round(5 * s)))
    draw.rounded_rectangle(
        [bar_x, int(round(14 * s)), bar_x + bar_w, ph - int(round(14 * s))],
        radius=max(2, bar_w // 2),
        fill=(255, 212, 64, 255),
    )

    text_x = bar_x + bar_w + int(round(16 * s))
    cur_y = int(round(12 * s))

    if camera_badge:
        draw.text((text_x, cur_y), camera_badge.upper(), font=f_badge, fill=(255, 212, 74, 245))
        cur_y += int(round(24 * s))

    draw.text((text_x, cur_y), caption_title, font=f_title, fill=(252, 252, 255, 255))
    cur_y += int(round(38 * s))

    if caption_subtitle:
        draw.text((text_x, cur_y), caption_subtitle, font=f_sub, fill=(218, 224, 236, 242))

    # Position at bottom-right museum corner (avoids top-left Act banners!)
    px = width - pw - int(round(58 * s))
    py = height - ph - int(round(46 * s))
    return plaque, px, py


def build_gallery_frame_image(
    src_path: str | Path,
    width: int = 1920,
    height: int = 1080,
    border_px: Optional[int] = None,
) -> Image.Image:
    """Create an uncropped double-matte gallery frame over a Gaussian-blurred ambient background."""
    img = Image.open(src_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    scale_factor = width / 1920.0
    blur_radius = max(18, int(round(28 * scale_factor)))

    bg = ImageOps.fit(img, (width, height), method=Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    dimmer = Image.new("RGB", (width, height), (8, 10, 16))
    bg = Image.blend(bg, dimmer, 0.26)

    aspect = img.width / max(1, img.height)
    if aspect < 1.0:
        max_w, max_h = int(width * 0.70), int(height * 0.84)
    else:
        max_w, max_h = int(width * 0.84), int(height * 0.82)

    card = _build_double_matte_card(img, max_w=max_w, max_h=max_h, scale_factor=scale_factor, border_px=border_px)
    px, py = (width - card.width) // 2, (height - card.height) // 2
    bg_rgba = bg.convert("RGBA")
    bg_rgba.paste(card, (px, py), card)
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
        return (1.0, max(1.10, kb.end_zoom), (0.5, 0.5), kb.end_center)
    if preset == "zoom_out":
        return (max(1.10, kb.start_zoom), 1.0, kb.start_center, (0.5, 0.5))
    if preset == "pan_left_to_right":
        return (1.08, 1.08, (0.46, 0.5), (0.54, 0.5))
    if preset == "pan_right_to_left":
        return (1.08, 1.08, (0.54, 0.5), (0.46, 0.5))
    if preset == "tilt_up":
        return (1.08, 1.08, (0.5, 0.54), (0.5, 0.46))
    return (kb.start_zoom, kb.end_zoom, kb.start_center, kb.end_center)


_NVENC_AVAILABLE: Optional[bool] = None


def has_hevc_nvenc() -> bool:
    """Check once whether hardware-accelerated hevc_nvenc is functional on this system."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "nullsrc=s=256x256:d=0.1",
            "-c:v",
            "hevc_nvenc",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        _NVENC_AVAILABLE = proc.returncode == 0
    return _NVENC_AVAILABLE


def render_photo_clip(
    photo_path: str | Path,
    output_mp4: str | Path,
    duration: float = 3.0,
    width: int = 3840,
    height: int = 2160,
    fps: int = 30,
    hdr: bool = True,
    ken_burns: Optional[KenBurnsConfig] = None,
    transition_in: Optional[TransitionConfig] = None,
    transition_out: Optional[TransitionConfig] = None,
) -> Path:
    """Render a photo into a 4K HDR MP4 video clip using 2.5D Parallax, Diptych Split-Screen, or Gallery Frame."""
    kb = ken_burns or KenBurnsConfig()
    out_path = Path(output_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_primary = Image.open(photo_path)
    raw_primary = ImageOps.exif_transpose(raw_primary).convert("RGB")
    scale_factor = width / 1920.0

    s_zoom, e_zoom, (sx, sy), (ex, ey) = _resolve_kb_endpoints(kb)
    total_frames = max(1, int(round(duration * fps)))
    resample_mode = Image.Resampling.BILINEAR if width >= 3840 else Image.Resampling.BICUBIC

    # Prepare optional museum exhibition plaque overlay
    plaque_rgba: Optional[Image.Image] = None
    plaque_x, plaque_y = 0, 0
    if kb.caption_title:
        plaque_rgba, plaque_x, plaque_y = _build_museum_plaque_rgba(
            width=width,
            height=height,
            caption_title=kb.caption_title,
            caption_subtitle=kb.caption_subtitle or "",
            camera_badge=kb.camera_badge or "",
        )

    # Prepare FFmpeg encoder & HDR color space pipeline
    vf_filters: list[str] = ["setsar=1:1"]
    if transition_in and transition_in.type == "fade_black" and transition_in.duration > 0:
        vf_filters.append(f"fade=t=in:st=0:d={min(duration * 0.45, transition_in.duration):.3f}")
    if transition_out and transition_out.type == "fade_black" and transition_out.duration > 0:
        fade_d = min(duration * 0.45, transition_out.duration)
        st = max(0.0, duration - fade_d)
        vf_filters.append(f"fade=t=out:st={st:.3f}:d={fade_d:.3f}")

    if hdr:
        vf_filters.extend(
            [
                "zscale=primariesin=709:transferin=iec61966-2-1:matrixin=709:"
                "primaries=2020:transfer=arib-std-b67:matrix=2020_ncl",
                "format=p010le",
                "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc",
            ]
        )
        if has_hevc_nvenc():
            v_enc = [
                "-c:v",
                "hevc_nvenc",
                "-profile:v",
                "main10",
                "-rc",
                "vbr",
                "-cq",
                "19",
                "-b:v",
                "28M",
                "-color_primaries",
                "bt2020",
                "-color_trc",
                "arib-std-b67",
                "-colorspace",
                "bt2020nc",
                "-tag:v",
                "hvc1",
            ]
        else:
            v_enc = [
                "-c:v",
                "libx265",
                "-pix_fmt",
                "yuv420p10le",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-color_primaries",
                "bt2020",
                "-color_trc",
                "arib-std-b67",
                "-colorspace",
                "bt2020nc",
                "-x265-params",
                "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
                "-tag:v",
                "hvc1",
            ]
    else:
        vf_filters.append("format=yuv420p")
        v_enc = [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
        ]

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
        *v_enc,
        "-video_track_timescale",
        "30000",
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        "-shortest",
        str(out_path),
    ]

    # Pre-build multi-plane assets depending on kb.mode
    is_diptych = (
        kb.mode == "diptych_frame"
        and kb.secondary_source is not None
        and Path(kb.secondary_source).exists()
    )
    is_parallax = kb.mode == "parallax_gallery"

    bg_oversample = 1.14
    bg_w = int(round(width * bg_oversample))
    bg_h = int(round(height * bg_oversample))
    blur_rad = max(20, int(round(30 * scale_factor)))
    bg_plane = ImageOps.fit(raw_primary, (bg_w, bg_h), method=Image.Resampling.BILINEAR)
    bg_plane = bg_plane.filter(ImageFilter.GaussianBlur(radius=blur_rad))
    bg_plane = Image.blend(bg_plane, Image.new("RGB", (bg_w, bg_h), (8, 10, 18)), 0.30)

    fg_card_1: Optional[Image.Image] = None
    fg_card_2: Optional[Image.Image] = None
    base_static_img: Optional[Image.Image] = None

    if is_diptych and kb.secondary_source:
        raw_sec = Image.open(kb.secondary_source)
        raw_sec = ImageOps.exif_transpose(raw_sec).convert("RGB")
        card_max_w = int(width * 0.41)
        card_max_h = int(height * 0.78)
        fg_card_1 = _build_double_matte_card(raw_primary, card_max_w, card_max_h, scale_factor, kb.border_px)
        fg_card_2 = _build_double_matte_card(raw_sec, card_max_w, card_max_h, scale_factor, kb.border_px)
    elif is_parallax:
        aspect = raw_primary.width / max(1, raw_primary.height)
        if aspect < 1.0:
            max_w, max_h = int(width * 0.70), int(height * 0.82)
        else:
            max_w, max_h = int(width * 0.82), int(height * 0.80)
        fg_card_1 = _build_double_matte_card(raw_primary, max_w, max_h, scale_factor, kb.border_px)
    else:
        oversample = 1.12 if width >= 3840 else 1.25
        work_w = int(round(width * oversample))
        work_h = int(round(height * oversample))
        if kb.mode == "gallery_frame":
            base_static_img = build_gallery_frame_image(
                photo_path, width=work_w, height=work_h, border_px=kb.border_px
            )
        else:
            base_static_img = ImageOps.fit(raw_primary, (work_w, work_h), method=Image.Resampling.LANCZOS)

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    try:
        for f_idx in range(total_frames):
            t_norm = f_idx / max(1, total_frames - 1)
            eased = _ease_in_out_cubic(t_norm)

            if is_diptych and fg_card_1 is not None and fg_card_2 is not None:
                # 1. Counter-drifting ambient background plane
                bg_zoom = 1.05 - 0.04 * eased
                bg_cx = 0.52 - 0.04 * eased
                bg_cy = 0.50
                bw_crop = bg_w / bg_zoom
                bh_crop = bg_h / bg_zoom
                bx0 = bg_cx * bg_w - bw_crop * 0.5
                by0 = bg_cy * bg_h - bh_crop * 0.5
                frame_rgb = bg_plane.transform(
                    (width, height),
                    Image.Transform.EXTENT,
                    (bx0, by0, bx0 + bw_crop, by0 + bh_crop),
                    resample=resample_mode,
                )
                frame_rgba = frame_rgb.convert("RGBA")

                # 2. Left & Right vertical portrait cards with gentle opposing vertical parallax drift
                drift_px = int(round((eased - 0.5) * 28.0 * scale_factor))
                left_cx = int(width * 0.28)
                right_cx = int(width * 0.72)
                mid_cy = int(height * 0.46)

                lx = left_cx - fg_card_1.width // 2
                ly = mid_cy - fg_card_1.height // 2 - drift_px
                rx = right_cx - fg_card_2.width // 2
                ry = mid_cy - fg_card_2.height // 2 + drift_px

                frame_rgba.paste(fg_card_1, (lx, ly), fg_card_1)
                frame_rgba.paste(fg_card_2, (rx, ry), fg_card_2)
                frame = frame_rgba.convert("RGB")

            elif is_parallax and fg_card_1 is not None:
                # 1. Plane 1: Counter-panning & counter-zooming ambient backdrop (2.5D depth separation)
                bg_zoom = 1.06 - 0.05 * eased
                fg_dx = (ex - sx) * eased
                fg_dy = (ey - sy) * eased
                bg_cx = 0.5 - fg_dx * 0.55
                bg_cy = 0.5 - fg_dy * 0.55
                bw_crop = bg_w / bg_zoom
                bh_crop = bg_h / bg_zoom
                bx0 = min(max(0.0, bg_cx * bg_w - bw_crop * 0.5), bg_w - bw_crop)
                by0 = min(max(0.0, bg_cy * bg_h - bh_crop * 0.5), bg_h - bh_crop)
                frame_rgb = bg_plane.transform(
                    (width, height),
                    Image.Transform.EXTENT,
                    (bx0, by0, bx0 + bw_crop, by0 + bh_crop),
                    resample=resample_mode,
                )
                frame_rgba = frame_rgb.convert("RGBA")

                # 2. Plane 2: Foreground museum double-matte card with independent scale & drift
                fg_scale = 0.96 + 0.06 * eased
                cw = max(1, int(round(fg_card_1.width * fg_scale)))
                ch = max(1, int(round(fg_card_1.height * fg_scale)))
                scaled_card = fg_card_1.resize((cw, ch), resample=resample_mode)

                pan_px_x = int(round(fg_dx * width * 0.32))
                pan_px_y = int(round(fg_dy * height * 0.32)) - int(round(height * 0.02))
                px = (width - cw) // 2 + pan_px_x
                py = (height - ch) // 2 + pan_px_y
                frame_rgba.paste(scaled_card, (px, py), scaled_card)
                frame = frame_rgba.convert("RGB")

            else:
                assert base_static_img is not None
                work_w, work_h = base_static_img.size
                cur_zoom = s_zoom + (e_zoom - s_zoom) * eased
                cx = sx + (ex - sx) * eased
                cy = sy + (ey - sy) * eased

                crop_w = work_w / cur_zoom
                crop_h = work_h / cur_zoom

                half_w = crop_w / 2.0
                half_h = crop_h / 2.0
                center_px = min(max(cx * work_w, half_w), work_w - half_w)
                center_py = min(max(cy * work_h, half_h), work_h - half_h)

                x0 = center_px - half_w
                y0 = center_py - half_h
                x1 = center_px + half_w
                y1 = center_py + half_h

                frame = base_static_img.transform(
                    (width, height),
                    Image.Transform.EXTENT,
                    (x0, y0, x1, y1),
                    resample=resample_mode,
                )

            # 3. Plane 3: Composite Museum Exhibition Plaque (with smooth fade-in / fade-out)
            if plaque_rgba is not None:
                t_sec = f_idx / max(1.0, float(fps))
                rem_sec = max(0.0, duration - t_sec)
                alpha_mult = min(1.0, t_sec / 0.35, rem_sec / 0.30)
                if alpha_mult > 0.05:
                    f_rgba = frame.convert("RGBA")
                    if alpha_mult < 0.98:
                        p_copy = plaque_rgba.copy()
                        r_ch, g_ch, b_ch, a_ch = p_copy.split()
                        a_ch = a_ch.point(lambda p: int(p * alpha_mult))
                        p_copy = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ch))
                        f_rgba.paste(p_copy, (plaque_x, plaque_y), p_copy)
                    else:
                        f_rgba.paste(plaque_rgba, (plaque_x, plaque_y), plaque_rgba)
                    frame = f_rgba.convert("RGB")

            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()

    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    ret = proc.wait()
    if ret != 0:
        err_msg = stderr_bytes.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg photo clip render failed (code {ret}): {err_msg}")

    return out_path
