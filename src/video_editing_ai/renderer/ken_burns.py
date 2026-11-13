"""Sub-pixel 16-Mode Cinema & Exhibition Photo Animator + 8-Finisher Optical Pipeline."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import subprocess
import textwrap
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

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


def _ease_out_cubic(t: float) -> float:
    """Smooth cubic ease-out for documentary lower-third card glide."""
    t_c = max(0.0, min(1.0, t))
    p = 1.0 - t_c
    return 1.0 - p * p * p


def _load_editorial_font(size: int, bold: bool = True, serif: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if serif:
        candidates = [
            # Windows
            "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
            # macOS (2019 MacBook Pro & Apple Silicon)
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/Library/Fonts/Georgia Bold.ttf" if bold else "/Library/Fonts/Georgia.ttf",
            "/System/Library/Fonts/NewYork.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            # Cross-platform fallbacks
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        ]
    else:
        candidates = [
            # Windows (Microsoft YaHei renders both crisp Segoe-style Latin & full CJK)
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            # macOS (2019 MacBook Pro & Apple Silicon — PingFang SC renders both Latin & CJK)
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Avenir Next.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, max(10, size))
            except Exception:
                continue
    return ImageFont.load_default()


def _build_matte_framed_rgba(
    img: Image.Image,
    max_w: int,
    max_h: int,
    scale_factor: float,
    border_px: Optional[int] = None,
) -> Image.Image:
    """Create a foreground photo card with a double museum passe-partout matte (warm cotton-rag ivory + archival stone bevel)."""
    b_px = border_px if border_px is not None else max(10, int(round(14 * scale_factor)))
    bevel_pad = max(2, int(round(3 * scale_factor)))

    fg = img.copy()
    fg.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

    fw = fg.width + (b_px + bevel_pad) * 2
    fh = fg.height + (b_px + bevel_pad) * 2

    framed = Image.new("RGBA", (fw, fh), (247, 245, 240, 255))
    f_draw = ImageDraw.Draw(framed)
    f_draw.rectangle(
        [b_px - 2, b_px - 2, fw - b_px + 1, fh - b_px + 1],
        outline=(212, 208, 198, 230),
        width=max(1, int(round(1.5 * scale_factor))),
    )
    framed.paste(fg, (b_px + bevel_pad, b_px + bevel_pad))
    return framed


def _build_double_matte_card(
    img: Image.Image,
    max_w: int,
    max_h: int,
    scale_factor: float,
    border_px: Optional[int] = None,
) -> Image.Image:
    """Create a foreground photo card with a double museum passe-partout matte and 3D drop shadow."""
    framed = _build_matte_framed_rgba(img, max_w=max_w, max_h=max_h, scale_factor=scale_factor, border_px=border_px)
    fw, fh = framed.size
    shadow_margin = max(18, int(round(28 * scale_factor)))
    shadow_offset = max(4, int(round(7 * scale_factor)))
    shadow_blur = max(10, int(round(16 * scale_factor)))

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
    return shadow


def _crop_plane_clamped(
    plane: Image.Image,
    out_size: tuple[int, int],
    zoom: float,
    cx: float,
    cy: float,
    resample_mode: Image.Resampling,
) -> Image.Image:
    """Safely extract and resample a zoomed region strictly clamped inside `plane` bounds."""
    pw, ph = plane.size
    crop_w = pw / max(1.0, zoom)
    crop_h = ph / max(1.0, zoom)
    bx0 = min(max(0.0, cx * pw - crop_w * 0.5), max(0.0, pw - crop_w))
    by0 = min(max(0.0, cy * ph - crop_h * 0.5), max(0.0, ph - crop_h))
    safe_resample = Image.Resampling.BICUBIC if resample_mode == Image.Resampling.LANCZOS else resample_mode
    return plane.transform(
        out_size,
        Image.Transform.EXTENT,
        (bx0, by0, bx0 + crop_w, by0 + crop_h),
        resample=safe_resample,
    )


def _solve_perspective_coeffs(
    dst_pts: list[tuple[float, float]],
    src_pts: list[tuple[float, float]],
) -> tuple[float, ...]:
    """Solve the 8-parameter projective homography mapping `dst_pts` back to `src_pts` for PIL PERSPECTIVE."""
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (u, v) in zip(dst_pts, src_pts):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        rhs.append(v)
    coeffs = np.linalg.solve(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64))
    return tuple(float(c) for c in coeffs)


_TECH_BADGE_TOKENS = (
    "SONY",
    "GOOGLE",
    "PIXEL",
    "APPLE",
    "IPHONE",
    "A7C",
    "33MP",
    "FULL FRAME",
    "DIGITAL CAMERA",
    "4K HDR",
    "HDR",
    "EXIF",
    "SOLO DAY TRIP",
)


def _clean_editorial_badge(raw_badge: str) -> str:
    """Filter out technical camera/sensor metadata from overline badges so plaques stay clean and artistic."""
    if not raw_badge:
        return ""
    cleaned = raw_badge.upper().replace("📍", "").strip()
    parts = [p.strip() for p in re.split(r"[·•|]", cleaned) if p.strip()]
    kept = [
        p
        for p in parts
        if not any(tok in p for tok in _TECH_BADGE_TOKENS) and len(p) >= 3
    ]
    return " · ".join(kept[:2])


def _build_frosted_card_assets(
    width: int,
    height: int,
    caption_title: str,
    caption_subtitle: str = "",
    camera_badge: str = "",
) -> tuple[Image.Image, Image.Image, Image.Image, int, int]:
    """
    Pre-render the static typography & smoked-glass highlight layers of an A24 / Criterion / Apple TV+
    style Museum & Documentary Lower-Third Plaque (free of garish yellow bars or camera jargon).
    Returns (text_fg_rgba, card_mask_l, dark_tint_rgb, card_x, card_y).
    """
    s = width / 1920.0
    f_badge = _load_editorial_font(int(round(13 * s)), bold=True, serif=False)
    f_title = _load_editorial_font(int(round(26 * s)), bold=True, serif=True)
    f_sub = _load_editorial_font(int(round(18 * s)), bold=False, serif=False)

    dummy = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    d_draw = ImageDraw.Draw(dummy)

    badge_str = _clean_editorial_badge(camera_badge)
    sub_lines: list[str] = []
    if caption_subtitle:
        wrapped = textwrap.wrap(caption_subtitle.strip(), width=78)
        sub_lines = wrapped[:2]

    tw = d_draw.textbbox((0, 0), caption_title, font=f_title)[2]
    bw = d_draw.textbbox((0, 0), badge_str, font=f_badge)[2] if badge_str else 0
    sw = max((d_draw.textbbox((0, 0), ln, font=f_sub)[2] for ln in sub_lines), default=0)

    pad_left = int(round(32 * s))
    pad_right = int(round(32 * s))
    pw = min(int(width * 0.80), max(tw, sw, bw) + pad_left + pad_right + int(round(18 * s)))
    pw = max(pw, int(round(380 * s)))

    # Compute exact vertical layout with balanced breathing room
    top_pad = int(round(14 * s))
    bottom_pad = int(round(15 * s))
    cur_y = top_pad
    if badge_str:
        cur_y += max(10, int(round(20 * s)))
    cur_y += max(14, int(round(34 * s)))  # title height
    if sub_lines:
        cur_y += max(2, int(round(4 * s))) + len(sub_lines) * max(12, int(round(25 * s)))
    ph = cur_y + bottom_pad

    # 1. Smooth rounded-rectangle alpha mask for smoked-glass region
    radius = int(round(14 * s))
    card_mask_l = Image.new("L", (pw, ph), 0)
    m_draw = ImageDraw.Draw(card_mask_l)
    m_draw.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=radius, fill=255)

    # 2. Neutral charcoal smoked-glass tint canvas
    dark_tint_rgb = Image.new("RGB", (pw, ph), (12, 14, 18))

    # 3. Foreground typography + subtle 1px neutral pearl specular rim + warm alabaster-silver hairline
    text_fg = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_fg)

    draw.rounded_rectangle(
        [0, 0, pw - 1, ph - 1],
        radius=radius,
        fill=(0, 0, 0, 0),
        outline=(255, 255, 255, 38),
        width=max(1, int(round(1.0 * s))),
    )

    # Delicate 2px warm pearl-silver vertical hairline on the left (never harsh yellow)
    bar_x = int(round(16 * s))
    bar_w = max(2, int(round(2.5 * s)))
    draw.rounded_rectangle(
        [bar_x, int(round(14 * s)), bar_x + bar_w, ph - int(round(14 * s))],
        radius=max(1, bar_w // 2),
        fill=(228, 224, 216, 185),
    )

    tx = bar_x + bar_w + int(round(14 * s))
    ty = top_pad

    if badge_str:
        draw.text((tx + 1, ty + 1), badge_str, font=f_badge, fill=(0, 0, 0, 160))
        draw.text((tx, ty), badge_str, font=f_badge, fill=(172, 168, 160, 225))
        ty += max(10, int(round(20 * s)))

    # Crisp editorial headline in Warm Alabaster with soft drop shadow
    draw.text((tx + 2, ty + 2), caption_title, font=f_title, fill=(0, 0, 0, 190))
    draw.text((tx, ty), caption_title, font=f_title, fill=(250, 248, 244, 255))
    ty += max(14, int(round(34 * s)))

    if sub_lines:
        ty += max(2, int(round(4 * s)))
        for ln in sub_lines:
            draw.text((tx + 1, ty + 1), ln, font=f_sub, fill=(0, 0, 0, 175))
            draw.text((tx, ty), ln, font=f_sub, fill=(202, 208, 216, 238))
            ty += max(12, int(round(25 * s)))

    # Center horizontally in the lower fifth for balanced documentary composition
    px = (width - pw) // 2
    py = height - ph - int(round(34 * s))
    return text_fg, card_mask_l, dark_tint_rgb, px, py


def render_frosted_video_overlay_png(
    output_png: Path,
    width: int,
    height: int,
    caption_title: str,
    caption_subtitle: str = "",
    camera_badge: str = "",
) -> Path:
    """Render a full-frame RGBA overlay PNG containing a soft bottom scrim + smoked-glass lower-third card for video segments."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # 1. Soft bottom vertical gradient scrim (20% height)
    scrim_h = max(16, int(round(height * 0.20)))
    grad_1d = Image.new("RGBA", (1, scrim_h), (0, 0, 0, 0))
    for y in range(scrim_h):
        t = y / max(1, scrim_h - 1)
        grad_1d.putpixel((0, y), (8, 10, 14, int(round(88 * (t * t)))))
    scrim_strip = grad_1d.resize((width, scrim_h), Image.Resampling.BILINEAR)
    canvas.paste(scrim_strip, (0, height - scrim_h), scrim_strip)

    # 2. Documentary card body + typography
    text_fg, card_mask_l, _dark_tint, px, py = _build_frosted_card_assets(
        width=width,
        height=height,
        caption_title=caption_title,
        caption_subtitle=caption_subtitle,
        camera_badge=camera_badge,
    )
    pw, ph = text_fg.size
    card_body = Image.new("RGBA", (pw, ph), (12, 14, 18, 148))
    card_body.putalpha(card_mask_l.point(lambda p: int(p * (148 / 255.0))))
    card_body = Image.alpha_composite(card_body, text_fg)
    canvas.paste(card_body, (px, py), card_body)
    canvas.save(output_png, "PNG")
    return output_png


def _composite_frosted_card_on_rgb(
    frame_rgb: Image.Image,
    text_fg_rgba: Image.Image,
    card_mask_l: Image.Image,
    dark_tint_rgb: Image.Image,
    px: int,
    py: int,
    alpha_mult: float = 1.0,
    slide_y: int = 0,
) -> None:
    """
    In-place composite of the smoked-glass documentary card onto `frame_rgb`:
    Crops only the small `(pw, ph)` region, blurs it at 1/4 scale (16x faster than full-res blur),
    tints it with translucent charcoal glass (0.58 blend), layers the typography, and pastes back onto `frame_rgb`.
    """
    pw, ph = text_fg_rgba.size
    w, h = frame_rgb.size
    dst_x = max(0, min(w - pw, px))
    dst_y = max(0, min(h - ph, py + slide_y))

    # 1. Fast backdrop blur on the underlying scene patch
    patch = frame_rgb.crop((dst_x, dst_y, dst_x + pw, dst_y + ph))
    small_w = max(16, pw // 4)
    small_h = max(16, ph // 4)
    blurred_patch = (
        patch.resize((small_w, small_h), Image.Resampling.BILINEAR)
        .filter(ImageFilter.GaussianBlur(radius=5))
        .resize((pw, ph), Image.Resampling.BILINEAR)
    )

    # 2. Blend with neutral smoked charcoal glass (58% opacity so scene colors glow softly through the glass)
    frosted_rgb = Image.blend(blurred_patch, dark_tint_rgb, 0.58)

    # 3. Composite crisp typography & specular rim onto the frosted glass patch
    frosted_rgb.paste(text_fg_rgba, (0, 0), text_fg_rgba)

    # 4. Paste rounded frosted card back onto frame_rgb with smooth fade mask
    if alpha_mult >= 0.98:
        frame_rgb.paste(frosted_rgb, (dst_x, dst_y), card_mask_l)
    else:
        scaled_mask = card_mask_l.point(lambda p: int(p * alpha_mult))
        frame_rgb.paste(frosted_rgb, (dst_x, dst_y), scaled_mask)


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
    bg = Image.blend(bg, dimmer, 0.28)

    aspect = img.width / max(1, img.height)
    if aspect < 1.0:
        max_w, max_h = int(width * 0.68), int(height * 0.78)
    else:
        max_w, max_h = int(width * 0.80), int(height * 0.76)

    card = _build_double_matte_card(img, max_w=max_w, max_h=max_h, scale_factor=scale_factor, border_px=border_px)
    px = (width - card.width) // 2
    py = int(height * 0.44) - card.height // 2
    bg.paste(card, (px, max(12, py)), card)
    return bg


def _resolve_kb_endpoints(
    kb: KenBurnsConfig,
) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    """Resolve preset names into concrete (start_zoom, end_zoom, start_center, end_center)."""
    preset = kb.preset
    if preset == "static":
        return (1.0, 1.0, (0.5, 0.5), (0.5, 0.5))
    if preset == "floating_gallery":
        return (1.0, 1.05, (0.5, 0.5), (0.5, 0.485))
    if preset == "zoom_in":
        return (1.0, max(1.08, kb.end_zoom), (0.5, 0.5), kb.end_center)
    if preset == "zoom_out":
        return (max(1.08, kb.start_zoom), 1.0, kb.start_center, (0.5, 0.5))
    if preset == "pan_left_to_right":
        return (1.07, 1.07, (0.46, 0.5), (0.54, 0.5))
    if preset == "pan_right_to_left":
        return (1.07, 1.07, (0.54, 0.5), (0.46, 0.5))
    if preset == "tilt_up":
        return (1.07, 1.07, (0.5, 0.54), (0.5, 0.46))
    return (kb.start_zoom, kb.end_zoom, kb.start_center, kb.end_center)


_ENCODER_CACHE: dict[bool, str] = {}
_FILTER_CACHE: dict[str, bool] = {}


def has_ffmpeg_filter(filter_name: str) -> bool:
    """Check once whether a specific FFmpeg filter (e.g. 'zscale') is compiled into the local ffmpeg binary."""
    if filter_name not in _FILTER_CACHE:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=False,
        )
        _FILTER_CACHE[filter_name] = any(
            f" {filter_name} " in line for line in (proc.stdout or "").splitlines()
        )
    return _FILTER_CACHE[filter_name]


def _probe_hw_encoder(encoder: str, pix_fmt: str = "yuv420p", extra_args: Optional[list[str]] = None) -> bool:
    """Test whether a specific hardware encoder is functional on the current machine/GPU."""
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
        encoder,
        "-pix_fmt",
        pix_fmt,
        *(extra_args or []),
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    return proc.returncode == 0


def detect_best_video_encoder(hdr: bool = True) -> str:
    """
    Auto-detect the fastest working hardware video encoder across platforms:
    - macOS (2019 MacBook Pro AMD Radeon Pro + Apple T2 / Apple Silicon): hevc_videotoolbox / h264_videotoolbox
    - Windows/Linux NVIDIA GPU: hevc_nvenc / h264_nvenc
    - Windows/BootCamp AMD Radeon GPU: hevc_amf / h264_amf
    - Intel QuickSync iGPU: hevc_qsv / h264_qsv
    - Software fallback: libx265 / libx264
    """
    if hdr in _ENCODER_CACHE:
        return _ENCODER_CACHE[hdr]

    if hdr:
        candidates = [
            ("hevc_videotoolbox", "p010le", ["-profile:v", "main10", "-allow_sw", "1"]),
            ("hevc_nvenc", "p010le", []),
            ("hevc_amf", "p010le", []),
            ("hevc_qsv", "p010le", []),
        ]
        fallback = "libx265"
    else:
        candidates = [
            ("h264_videotoolbox", "yuv420p", ["-allow_sw", "1"]),
            ("h264_nvenc", "yuv420p", []),
            ("h264_amf", "yuv420p", []),
            ("h264_qsv", "yuv420p", []),
        ]
        fallback = "libx264"

    for enc_name, p_fmt, ex in candidates:
        if _probe_hw_encoder(enc_name, pix_fmt=p_fmt, extra_args=ex):
            _ENCODER_CACHE[hdr] = enc_name
            return enc_name

    _ENCODER_CACHE[hdr] = fallback
    return fallback


def has_hevc_nvenc() -> bool:
    """Backward-compatible helper returning True when hevc_nvenc is the active 10-bit encoder."""
    return detect_best_video_encoder(hdr=True) == "hevc_nvenc"


def build_hdr_color_filters(
    src_trc: str = "iec61966-2-1",
    src_primaries: str = "bt709",
    src_space: str = "bt709",
    is_log: bool = False,
    diffuse_white_nits: float = 203.0,
    use_zscale: Optional[bool] = None,
) -> list[str]:
    """
    Build 10-bit BT.2020 HLG HDR color-space filters:
    - Native HLG (arib-std-b67): passthrough to p010le with BT.2020 HLG VUI tags.
    - SDR / sRGB / Log -> HLG with zscale: maps SDR peak white to HLG diffuse white (~203 nits / ~75% HLG signal).
    - Fallback without zscale: applies calibrated diffuse-white exposure/gamma curve prior to setparams so
      8-bit SDR highlights and Gallery Frame borders never clip harshly at 1000 nits.
    """
    filters: list[str] = []
    trc_norm = (src_trc or "709").strip().lower()

    if is_log:
        filters.append("eq=contrast=1.18:saturation=1.14:gamma=0.92")

    if trc_norm == "arib-std-b67" and not is_log:
        filters.extend(
            [
                "format=p010le",
                "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc",
            ]
        )
        return filters

    zscale_ok = has_ffmpeg_filter("zscale") if use_zscale is None else bool(use_zscale)
    if trc_norm in ("smpte2084", "pq"):
        z_trc_in = "smpte2084"
    elif trc_norm in ("iec61966-2-1", "srgb"):
        z_trc_in = "iec61966-2-1"
    else:
        z_trc_in = "709"

    z_prim_in = "2020" if "2020" in (src_primaries or "").lower() else "709"
    z_mat_in = "2020_ncl" if "2020" in (src_space or "").lower() else "709"
    npl_val = max(100, min(400, int(round(diffuse_white_nits))))

    if zscale_ok:
        if z_trc_in == "smpte2084":
            filters.append(
                "zscale=primariesin=2020:transferin=smpte2084:matrixin=2020_ncl:"
                "primaries=2020:transfer=arib-std-b67:matrix=2020_ncl"
            )
        else:
            filters.append(
                f"zscale=primariesin={z_prim_in}:transferin={z_trc_in}:matrixin={z_mat_in}:"
                f"primaries=2020:transfer=arib-std-b67:matrix=2020_ncl:npl={npl_val}"
            )
    else:
        # Calibrated SDR -> HLG diffuse-white (~75% signal / ~203 nits) curve prior to setparams fallback
        filters.extend(
            [
                "colorlevels=romax=0.75:gomax=0.75:bomax=0.75",
                "eq=gamma=1.14:contrast=0.96",
            ]
        )

    filters.extend(
        [
            "format=p010le",
            "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc",
        ]
    )
    return filters


def get_video_encoder_args(hdr: bool = True, force_software: bool = False) -> list[str]:
    """Return FFmpeg video encoder CLI arguments for the best available hardware or software encoder."""
    enc = ("libx265" if hdr else "libx264") if force_software else detect_best_video_encoder(hdr=hdr)
    if hdr:
        hdr_tags = [
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "arib-std-b67",
            "-colorspace",
            "bt2020nc",
            "-tag:v",
            "hvc1",
        ]
        if enc == "hevc_videotoolbox":
            return [
                "-c:v",
                "hevc_videotoolbox",
                "-profile:v",
                "main10",
                "-pix_fmt",
                "p010le",
                "-q:v",
                "68",
                "-b:v",
                "28M",
                "-allow_sw",
                "1",
                *hdr_tags,
            ]
        if enc == "hevc_nvenc":
            return [
                "-c:v",
                "hevc_nvenc",
                "-preset",
                "p4",
                "-tune",
                "hq",
                "-profile:v",
                "main10",
                "-pix_fmt",
                "p010le",
                "-rc",
                "vbr",
                "-cq",
                "19",
                "-b:v",
                "28M",
                *hdr_tags,
            ]
        if enc == "hevc_amf":
            return [
                "-c:v",
                "hevc_amf",
                "-quality",
                "balanced",
                "-pix_fmt",
                "p010le",
                "-rc",
                "vbr_peak",
                "-b:v",
                "28M",
                *hdr_tags,
            ]
        if enc == "hevc_qsv":
            return [
                "-c:v",
                "hevc_qsv",
                "-preset",
                "faster",
                "-pix_fmt",
                "p010le",
                "-global_quality",
                "20",
                *hdr_tags,
            ]
        return [
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p10le",
            "-preset",
            "superfast",
            "-crf",
            "18",
            *hdr_tags,
            "-x265-params",
            "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
        ]

    # 8-bit SDR encoders
    if enc == "h264_videotoolbox":
        return [
            "-c:v",
            "h264_videotoolbox",
            "-q:v",
            "68",
            "-b:v",
            "16M",
            "-allow_sw",
            "1",
            "-pix_fmt",
            "yuv420p",
        ]
    if enc == "h264_nvenc":
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-cq",
            "19",
            "-pix_fmt",
            "yuv420p",
        ]
    if enc == "h264_amf":
        return [
            "-c:v",
            "h264_amf",
            "-quality",
            "balanced",
            "-b:v",
            "16M",
            "-pix_fmt",
            "yuv420p",
        ]
    if enc == "h264_qsv":
        return [
            "-c:v",
            "h264_qsv",
            "-global_quality",
            "20",
            "-pix_fmt",
            "yuv420p",
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]


def _build_subject_depth_layers(
    raw_img: Image.Image,
    bg_w: int,
    bg_h: int,
    focal_point: tuple[float, float],
    scale_factor: float,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """
    Build 3 depth planes for 'cinematic_depth_3d':
      1. bg_sharp: fitted full-canvas RGB plane
      2. bg_bokeh: defocused creamy bokeh background plane
      3. fg_subject_rgba: extracted foreground subject with soft saliency + radial alpha mask and micro-contrast pop
    """
    bg_sharp = ImageOps.fit(raw_img, (bg_w, bg_h), method=Image.Resampling.BILINEAR)

    # Defocused creamy bokeh plane (computed fast at 1/4 scale)
    small_w, small_h = max(32, bg_w // 4), max(32, bg_h // 4)
    small_rgb = bg_sharp.resize((small_w, small_h), Image.Resampling.BILINEAR)
    bokeh_small = small_rgb.filter(ImageFilter.GaussianBlur(radius=max(8, int(round(12 * scale_factor)))))
    bg_bokeh = bokeh_small.resize((bg_w, bg_h), Image.Resampling.BILINEAR)
    bg_bokeh = Image.blend(bg_bokeh, Image.new("RGB", (bg_w, bg_h), (8, 11, 20)), 0.28)

    # Saliency + radial focal-point depth mask at 1/4 resolution
    gray_small = small_rgb.convert("L")
    edges = gray_small.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=6))
    edge_arr = np.asarray(edges, dtype=np.float32)
    p95 = float(np.percentile(edge_arr, 95)) if edge_arr.size else 1.0
    edge_norm = np.clip(edge_arr / max(12.0, p95), 0.0, 1.0)

    yy, xx = np.mgrid[0:small_h, 0:small_w]
    nx = xx / max(1.0, float(small_w - 1))
    ny = fy = yy / max(1.0, float(small_h - 1))
    fx_c = max(0.22, min(0.78, float(focal_point[0])))
    fy_c = max(0.22, min(0.78, float(focal_point[1])))

    dist_sq = ((nx - fx_c) ** 2) / 0.11 + ((ny - fy_c) ** 2) / 0.14
    radial_core = np.exp(-0.5 * dist_sq)
    radial_broad = np.exp(-0.22 * dist_sq)

    combined = np.clip(0.58 * radial_core + 0.62 * edge_norm * radial_broad, 0.0, 1.0)
    # Smooth S-curve mask so the foreground subject has a natural feathered boundary
    mask_float = np.clip((combined - 0.22) / 0.52, 0.0, 1.0)
    mask_float = mask_float * mask_float * (3.0 - 2.0 * mask_float)
    mask_u8 = (mask_float * 255.0).astype(np.uint8)
    mask_l = (
        Image.fromarray(mask_u8, mode="L")
        .filter(ImageFilter.GaussianBlur(radius=8))
        .resize((bg_w, bg_h), Image.Resampling.BILINEAR)
    )

    fg_enhanced = ImageEnhance.Contrast(ImageEnhance.Sharpness(bg_sharp).enhance(1.18)).enhance(1.06)
    fg_subject_rgba = fg_enhanced.copy().convert("RGBA")
    fg_subject_rgba.putalpha(mask_l)
    return bg_sharp, bg_bokeh, fg_subject_rgba


def _build_polaroid_card(
    img: Image.Image,
    max_w: int,
    max_h: int,
    scale_factor: float,
    angle_deg: float = -2.8,
    stamp_text: str = "",
) -> Image.Image:
    """Render a tactile vintage instant-film Polaroid print with warm ivory paper borders, stamp text, and 3D shadow."""
    side_pad = max(14, int(round(22 * scale_factor)))
    top_pad = max(14, int(round(22 * scale_factor)))
    bottom_chin = max(46, int(round(76 * scale_factor)))

    inner_max_w = max(64, max_w - side_pad * 2)
    inner_max_h = max(64, max_h - top_pad - bottom_chin)

    fg = img.copy()
    fg.thumbnail((inner_max_w, inner_max_h), Image.Resampling.LANCZOS)
    # Subtle warm nostalgic film tone on the print
    warm_tint = Image.new("RGB", fg.size, (252, 244, 228))
    fg = Image.blend(fg, warm_tint, 0.06)

    pw = fg.width + side_pad * 2
    ph = fg.height + top_pad + bottom_chin

    paper = Image.new("RGBA", (pw, ph), (249, 246, 238, 255))
    p_draw = ImageDraw.Draw(paper)
    # Outer subtle paper bevel & inner recessed photo well
    p_draw.rectangle([0, 0, pw - 1, ph - 1], outline=(218, 212, 198, 255), width=max(1, int(round(1.5 * scale_factor))))
    p_draw.rectangle(
        [side_pad - 2, top_pad - 2, side_pad + fg.width + 1, top_pad + fg.height + 1],
        outline=(195, 188, 174, 235),
        width=max(1, int(round(1.5 * scale_factor))),
    )
    paper.paste(fg, (side_pad, top_pad))

    # Editorial stamp in the Polaroid bottom chin
    if stamp_text:
        f_stamp = _load_editorial_font(int(round(17 * scale_factor)), bold=True, serif=True)
        short_stamp = stamp_text.split("—")[0].split("·")[0].strip()[:42]
        tw = p_draw.textbbox((0, 0), short_stamp, font=f_stamp)[2]
        tx = (pw - tw) // 2
        ty = top_pad + fg.height + (bottom_chin - int(round(22 * scale_factor))) // 2
        p_draw.text((tx, ty), short_stamp, font=f_stamp, fill=(48, 44, 38, 230))

    # Drop shadow behind the Polaroid before rotation
    sh_margin = max(24, int(round(36 * scale_factor)))
    sh_offset = max(6, int(round(10 * scale_factor)))
    canvas = Image.new("RGBA", (pw + sh_margin * 2, ph + sh_margin * 2), (0, 0, 0, 0))
    c_draw = ImageDraw.Draw(canvas)
    c_draw.rectangle(
        [sh_margin, sh_margin + sh_offset, sh_margin + pw, sh_margin + ph + sh_offset],
        fill=(0, 0, 0, 175),
    )
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=max(10, int(round(16 * scale_factor)))))
    canvas.paste(paper, (sh_margin, sh_margin), paper)

    if abs(angle_deg) > 0.05:
        canvas = canvas.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
    return canvas


def _build_triptych_panels(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    has_card: bool,
    scale_factor: float,
) -> list[Image.Image]:
    """Build 3 coordinated vertical gallery panels for 'triptych_cinematic'."""
    panel_w = int(width * 0.265)
    panel_h = int(height * (0.70 if has_card else 0.78))
    border_px = max(8, int(round(11 * scale_factor)))

    sec_img: Optional[Image.Image] = None
    tert_img: Optional[Image.Image] = None
    if kb.secondary_source and Path(kb.secondary_source).exists():
        sec_img = ImageOps.exif_transpose(Image.open(kb.secondary_source)).convert("RGB")
    if kb.tertiary_source and Path(kb.tertiary_source).exists():
        tert_img = ImageOps.exif_transpose(Image.open(kb.tertiary_source)).convert("RGB")

    if sec_img is not None and tert_img is not None:
        imgs = [
            ImageOps.fit(raw_primary, (panel_w, panel_h), method=Image.Resampling.LANCZOS),
            ImageOps.fit(sec_img, (panel_w, panel_h), method=Image.Resampling.LANCZOS),
            ImageOps.fit(tert_img, (panel_w, panel_h), method=Image.Resampling.LANCZOS),
        ]
    elif sec_img is not None:
        # 2 photos available: Left = Primary, Center = Secondary, Right = zoomed detail of Primary
        detail_crop = ImageOps.fit(
            raw_primary, (panel_w, panel_h), method=Image.Resampling.LANCZOS, centering=(0.62, 0.45)
        )
        imgs = [
            ImageOps.fit(raw_primary, (panel_w, panel_h), method=Image.Resampling.LANCZOS, centering=(0.38, 0.50)),
            ImageOps.fit(sec_img, (panel_w, panel_h), method=Image.Resampling.LANCZOS),
            detail_crop,
        ]
    else:
        # Single photo: slice a wide architectural triptych across the image
        total_w = panel_w * 3 + int(round(36 * scale_factor))
        pano = ImageOps.fit(raw_primary, (total_w, panel_h), method=Image.Resampling.LANCZOS)
        step = (total_w - panel_w) // 2
        imgs = [
            pano.crop((0, 0, panel_w, panel_h)),
            pano.crop((step, 0, step + panel_w, panel_h)),
            pano.crop((total_w - panel_w, 0, total_w, panel_h)),
        ]

    return [
        _build_double_matte_card(im, max_w=panel_w, max_h=panel_h, scale_factor=scale_factor, border_px=border_px)
        for im in imgs
    ]


def _build_curator_loupe_assets(
    raw_primary: Image.Image,
    width: int,
    height: int,
    has_card: bool,
    scale_factor: float,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, int]:
    """
    Pre-build assets for 'curator_loupe' (Art-Historian 1.65x Optical Magnifying Glass Spotlight):
    Returns (dimmed_canvas_rgb, sharp_canvas_rgb, loupe_mask_l, loupe_bezel_rgba, radius_px).
    """
    # Fit or frame the full artwork on the 16:9 canvas
    sharp_canvas = build_gallery_frame_image(raw_primary, width=width, height=height) if isinstance(raw_primary, (str, Path)) else None
    if sharp_canvas is None:
        # Build clean dimmed background + centered uncropped artwork
        bg_small = ImageOps.fit(raw_primary, (width // 4, height // 4), method=Image.Resampling.BILINEAR)
        bg_small = bg_small.filter(ImageFilter.GaussianBlur(radius=max(6, int(round(9 * scale_factor)))))
        sharp_canvas = bg_small.resize((width, height), Image.Resampling.BILINEAR)
        sharp_canvas = Image.blend(sharp_canvas, Image.new("RGB", (width, height), (8, 10, 18)), 0.35)

        max_w = int(width * 0.84)
        max_h = int(height * (0.76 if has_card else 0.84))
        card = _build_double_matte_card(raw_primary, max_w=max_w, max_h=max_h, scale_factor=scale_factor)
        cy = int(height * (0.43 if has_card else 0.48))
        sharp_canvas.paste(card, ((width - card.width) // 2, max(10, cy - card.height // 2)), card)

    dimmed_canvas = Image.blend(sharp_canvas, Image.new("RGB", (width, height), (6, 8, 14)), 0.36)

    radius_px = int(round(min(width, height) * 0.21))
    diam = radius_px * 2
    loupe_mask_l = Image.new("L", (diam, diam), 0)
    m_draw = ImageDraw.Draw(loupe_mask_l)
    m_draw.ellipse([2, 2, diam - 3, diam - 3], fill=255)

    # Brushed platinum-silver optical lens bezel + glass specular crescent + outer shadow
    pad = max(18, int(round(26 * scale_factor)))
    bz_size = diam + pad * 2
    bezel = Image.new("RGBA", (bz_size, bz_size), (0, 0, 0, 0))
    b_draw = ImageDraw.Draw(bezel)
    # Soft outer shadow ring
    b_draw.ellipse(
        [pad - 3, pad + 3, pad + diam + 3, pad + diam + 10],
        outline=(0, 0, 0, 155),
        width=max(5, int(round(8 * scale_factor))),
    )
    bezel = bezel.filter(ImageFilter.GaussianBlur(radius=max(5, int(round(8 * scale_factor)))))
    b_draw = ImageDraw.Draw(bezel)
    # Refined optical platinum-silver glass rim (no garish yellow)
    rim_w = max(2, int(round(3.0 * scale_factor)))
    b_draw.ellipse(
        [pad, pad, pad + diam - 1, pad + diam - 1],
        outline=(232, 235, 240, 215),
        width=rim_w,
    )
    # Inner crystal glass specular highlight arc
    b_draw.arc(
        [pad + rim_w + 3, pad + rim_w + 3, pad + diam - rim_w - 4, pad + diam - rim_w - 4],
        start=200,
        end=325,
        fill=(255, 255, 255, 130),
        width=max(1, int(round(2.0 * scale_factor))),
    )
    return dimmed_canvas, sharp_canvas, loupe_mask_l, bezel, radius_px


def _build_editorial_magazine_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
) -> tuple[Image.Image, Image.Image, int]:
    """
    Build asymmetric 62% / 38% luxury Kinfolk / Criterion editorial monograph spread:
    Returns (left_oversampled_rgb, right_column_rgb, left_w).
    """
    left_w = int(round(width * 0.62))
    left_w = left_w - (left_w % 2)
    right_w = width - left_w

    oversample = 1.14
    ow = int(round(left_w * oversample))
    oh = int(round(height * oversample))
    left_oversampled = ImageOps.fit(raw_primary, (ow, oh), method=Image.Resampling.LANCZOS)

    # Right 38% warm gallery charcoal editorial monograph column
    col = Image.new("RGB", (right_w, height), (14, 16, 20))
    draw = ImageDraw.Draw(col)

    # Subtle vertical stone-pewter divider rule on the left seam
    seam_w = max(1, int(round(1.5 * scale_factor)))
    draw.rectangle([0, 0, seam_w, height], fill=(56, 60, 68))

    pad_x = int(round(48 * scale_factor))
    cur_y = int(height * 0.25)

    # 1. Cleaned Location Overline Badge (only if non-technical location/collection text is present)
    badge_str = _clean_editorial_badge(kb.camera_badge or "")
    if badge_str:
        f_badge = _load_editorial_font(int(round(14 * scale_factor)), bold=True, serif=False)
        draw.text((pad_x, cur_y), badge_str, font=f_badge, fill=(168, 164, 156))
        cur_y += max(12, int(round(30 * scale_factor)))

    # Delicate warm alabaster-silver horizontal rule (no bright yellow)
    draw.rectangle(
        [pad_x, cur_y, pad_x + int(round(52 * scale_factor)), cur_y + max(1, int(round(2 * scale_factor)))],
        fill=(196, 192, 184),
    )
    cur_y += max(10, int(round(26 * scale_factor)))

    # 2. Multi-line Serif Monograph Title in Warm Alabaster
    title_str = (kb.caption_title or "Curated Masterpiece").strip()
    f_title = _load_editorial_font(int(round(32 * scale_factor)), bold=True, serif=True)
    for line in textwrap.wrap(title_str, width=22)[:4]:
        draw.text((pad_x, cur_y), line, font=f_title, fill=(248, 246, 240))
        cur_y += max(14, int(round(44 * scale_factor)))

    cur_y += max(6, int(round(16 * scale_factor)))

    # 3. Multi-line Curatorial Essay / Bilingual Subtitle in Soft Pearl Stone
    sub_str = (kb.caption_subtitle or "").strip()
    if sub_str:
        f_sub = _load_editorial_font(int(round(20 * scale_factor)), bold=False, serif=False)
        for line in textwrap.wrap(sub_str, width=32)[:6]:
            draw.text((pad_x, cur_y), line, font=f_sub, fill=(192, 198, 208))
            cur_y += max(12, int(round(30 * scale_factor)))

    # Bottom monograph folio stamp (only rendered if a clean location/folio badge is present)
    if badge_str:
        f_folio = _load_editorial_font(int(round(13 * scale_factor)), bold=False, serif=True)
        draw.text(
            (pad_x, int(height * 0.89)),
            badge_str,
            font=f_folio,
            fill=(128, 134, 144),
        )
    return left_oversampled, col, left_w


def _build_cinema_anamorphic_base(
    raw_primary: Image.Image,
    work_w: int,
    work_h: int,
    scale_factor: float,
) -> Image.Image:
    """Build an oversampled image with 35mm Anamorphic highlight halation (warm amber-red bloom on bright highlights)."""
    base = ImageOps.fit(raw_primary, (work_w, work_h), method=Image.Resampling.LANCZOS)
    small_w, small_h = max(32, work_w // 4), max(32, work_h // 4)
    small = base.resize((small_w, small_h), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    hi_mask = np.clip((luma - 185.0) / 70.0, 0.0, 1.0)

    hal_rgb = np.zeros_like(arr)
    hal_rgb[:, :, 0] = hi_mask * 255.0
    hal_rgb[:, :, 1] = hi_mask * 118.0
    hal_rgb[:, :, 2] = hi_mask * 42.0
    hal_img = (
        Image.fromarray(hal_rgb.astype(np.uint8), mode="RGB")
        .filter(ImageFilter.GaussianBlur(radius=max(6, int(round(10 * scale_factor)))))
        .resize((work_w, work_h), Image.Resampling.BILINEAR)
    )
    # Soft additive screen-like blend of warm halation bloom
    base_arr = np.asarray(base, dtype=np.uint16)
    hal_full = np.asarray(hal_img, dtype=np.uint16)
    out_arr = np.clip(base_arr + (hal_full * 28) // 100, 0, 255).astype(np.uint8)
    return Image.fromarray(out_arr, mode="RGB")


def _build_contact_sheet_zoom_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    has_card: bool,
    scale_factor: float,
) -> tuple[Image.Image, float, float]:
    """
    Build a high-resolution Magnum Photos-inspired 3x2 Darkroom Contact Sheet canvas
    so the virtual camera opens on all 6 proof frames and smoothly pushes into hero frame 02A.
    Returns (oversampled_sheet_rgb, hero_center_x, hero_center_y).
    """
    sheet_scale = 1.45 if width >= 3840 else 1.90
    sw = int(round(width * sheet_scale))
    sh = int(round(height * sheet_scale))
    s_fac = sw / 1920.0

    # Archival darkroom light-table background (rich obsidian slate)
    sheet = Image.new("RGB", (sw, sh), (14, 16, 22))
    draw = ImageDraw.Draw(sheet)

    cols, rows = 3, 2
    outer_x = int(round(92 * s_fac))
    outer_top = int(round(62 * s_fac))
    outer_bot = int(round((145 if has_card else 66) * s_fac))
    gap_x = int(round(44 * s_fac))
    gap_y = int(round(46 * s_fac))
    rebate_h = max(14, int(round(24 * s_fac)))

    cell_w = max(80, (sw - outer_x * 2 - gap_x * (cols - 1)) // cols)
    cell_h = max(60, (sh - outer_top - outer_bot - gap_y * (rows - 1) - rebate_h * rows) // rows)

    sec_img: Optional[Image.Image] = None
    tert_img: Optional[Image.Image] = None
    if kb.secondary_source and Path(kb.secondary_source).exists():
        sec_img = ImageOps.exif_transpose(Image.open(kb.secondary_source)).convert("RGB")
    if kb.tertiary_source and Path(kb.tertiary_source).exists():
        tert_img = ImageOps.exif_transpose(Image.open(kb.tertiary_source)).convert("RGB")

    # Populate the 6 cells; Hero frame sits at row=0, col=1 (idx=1 -> '02A SELECT') so zoom clearance fits above captions
    focal = kb.start_center
    p_left = ImageOps.fit(raw_primary, (cell_w, cell_h), method=Image.Resampling.LANCZOS, centering=(0.25, 0.50))
    p_hero = ImageOps.fit(raw_primary, (cell_w, cell_h), method=Image.Resampling.LANCZOS, centering=focal)
    p_right = ImageOps.fit(raw_primary, (cell_w, cell_h), method=Image.Resampling.LANCZOS, centering=(0.75, 0.45))

    cell_imgs = [
        ImageOps.fit(sec_img, (cell_w, cell_h), method=Image.Resampling.LANCZOS) if sec_img else p_left,
        p_hero,
        ImageOps.fit(tert_img, (cell_w, cell_h), method=Image.Resampling.LANCZOS) if tert_img else p_right,
        p_left,
        ImageOps.fit(sec_img, (cell_w, cell_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.3)) if sec_img else p_right,
        ImageOps.fit(tert_img, (cell_w, cell_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.7)) if tert_img else ImageEnhance.Brightness(p_hero).enhance(0.85),
    ]

    f_code = _load_editorial_font(max(10, int(round(13 * s_fac))), bold=True, serif=False)
    f_mark = _load_editorial_font(max(9, int(round(11 * s_fac))), bold=False, serif=False)
    hero_idx = 1
    hero_cx, hero_cy = 0.50, 0.34

    for idx in range(cols * rows):
        r = idx // cols
        c = idx % cols
        x0 = outer_x + c * (cell_w + gap_x)
        y0 = outer_top + r * (cell_h + rebate_h + gap_y)
        is_hero = idx == hero_idx

        plate = cell_imgs[idx]
        if not is_hero:
            # Slightly subdue neighbor proof frames so the selected curatorial frame radiates cleanly
            plate = ImageEnhance.Brightness(ImageEnhance.Color(plate).enhance(0.88)).enhance(0.74)

        # Thin darkroom negative film strip band above image with sprocket notches & frame number
        strip_pad = max(3, int(round(5 * s_fac)))
        draw.rectangle(
            [x0 - strip_pad, y0 - rebate_h, x0 + cell_w + strip_pad, y0 + cell_h + strip_pad],
            fill=(20, 22, 28),
            outline=(198, 194, 186) if is_hero else (46, 50, 60),
            width=max(2, int(round(3.0 * s_fac))) if is_hero else max(1, int(round(1.0 * s_fac))),
        )
        # Subtle sprocket holes along film strip rebate
        notch_w = max(3, int(round(5 * s_fac)))
        notch_h = max(2, int(round(4 * s_fac)))
        step_x = max(14, int(round(22 * s_fac)))
        for sx in range(x0 + 6, x0 + cell_w - 8, step_x):
            draw.rectangle([sx, y0 - rebate_h + 3, sx + notch_w, y0 - rebate_h + 3 + notch_h], fill=(38, 42, 52))

        frame_tag = f"0{idx + 1}A"
        if is_hero:
            draw.text((x0 + 4, y0 - int(round(15 * s_fac))), f"{frame_tag} · SELECTED PLATE", font=f_code, fill=(238, 234, 226))
        else:
            draw.text((x0 + 4, y0 - int(round(14 * s_fac))), frame_tag, font=f_mark, fill=(116, 122, 134))

        sheet.paste(plate, (x0, y0))

        if is_hero:
            # Fine warm ivory inner curatorial selection hairline around hero plate
            draw.rectangle(
                [x0 - 2, y0 - 2, x0 + cell_w + 1, y0 + cell_h + 1],
                outline=(236, 232, 224),
                width=max(2, int(round(2.5 * s_fac))),
            )
            hero_cx = (x0 + cell_w * 0.5) / float(sw)
            hero_cy = (y0 + cell_h * 0.52) / float(sh)

    return sheet, hero_cx, hero_cy


def _build_split_diopter_assets(
    raw_primary: Image.Image,
    focal_point: tuple[float, float],
    width: int,
    height: int,
    scale_factor: float,
) -> tuple[Image.Image, Image.Image, Image.Image, int]:
    """
    Build A24 / De Palma optical split-diopter assets:
    - Left 54%: Oversampled full-context wide view (slightly calmer tone)
    - Right 46%: 2.2x optical macro detail study centered tightly around `focal_point`
    - Seam: Refractive vertical frosted glass prism seam overlay
    Returns (left_over_rgb, right_over_rgb, seam_rgba, left_w).
    """
    left_w = int(round(width * 0.54))
    left_w = left_w - (left_w % 2)
    right_w = width - left_w

    oversample = 1.14
    lw, lh = int(round(left_w * oversample)), int(round(height * oversample))
    left_over = ImageOps.fit(raw_primary, (lw, lh), method=Image.Resampling.LANCZOS)
    left_over = ImageEnhance.Brightness(left_over).enhance(0.90)

    # Right plane: extract a true zoomed 2.2x macro detail region around focal_point
    rw_over, rh_over = int(round(right_w * 1.15)), int(round(height * 1.15))
    src_w, src_h = raw_primary.size
    target_ar = right_w / max(1.0, float(height))
    # Crop ~46% of source height/width centered on focal_point (clamped inside bounds)
    crop_h = max(64.0, src_h * 0.46)
    crop_w = crop_h * target_ar
    if crop_w > src_w * 0.72:
        crop_w = src_w * 0.72
        crop_h = crop_w / target_ar
    fx = min(max(0.18, float(focal_point[0])), 0.82)
    fy = min(max(0.18, float(focal_point[1])), 0.82)
    cx0 = min(max(0.0, fx * src_w - crop_w * 0.5), max(0.0, src_w - crop_w))
    cy0 = min(max(0.0, fy * src_h - crop_h * 0.5), max(0.0, src_h - crop_h))
    right_over = raw_primary.transform(
        (rw_over, rh_over),
        Image.Transform.EXTENT,
        (cx0, cy0, cx0 + crop_w, cy0 + crop_h),
        resample=Image.Resampling.BICUBIC,
    )
    right_over = ImageEnhance.Contrast(ImageEnhance.Sharpness(right_over).enhance(1.16)).enhance(1.06)

    # Refractive optical glass prism seam
    seam_w = max(14, int(round(26 * scale_factor)))
    seam = Image.new("RGBA", (seam_w, height), (0, 0, 0, 0))
    s_arr = np.zeros((height, seam_w, 4), dtype=np.uint8)
    xs = np.linspace(-1.0, 1.0, seam_w, dtype=np.float32)
    # Soft gaussian refraction band + crisp central crystal hairline
    glow = np.exp(-3.5 * (xs ** 2))
    rim = np.exp(-18.0 * (xs ** 2))
    s_arr[:, :, 0] = np.clip(glow * 180 + rim * 245, 0, 255).astype(np.uint8)
    s_arr[:, :, 1] = np.clip(glow * 195 + rim * 248, 0, 255).astype(np.uint8)
    s_arr[:, :, 2] = np.clip(glow * 225 + rim * 255, 0, 255).astype(np.uint8)
    s_arr[:, :, 3] = np.clip(glow * 62 + rim * 125, 0, 255).astype(np.uint8)
    seam = Image.fromarray(s_arr, mode="RGBA")
    return left_over, right_over, seam, left_w


def _build_glassmorphic_pedestal_assets(
    raw_primary: Image.Image,
    width: int,
    height: int,
    card_max_h_ratio: float,
    scale_factor: float,
) -> dict[str, Any]:
    """
    Build Apple VisionOS / Modern Museum acrylic glassmorphic pedestal assets:
    Returns inner crisp artwork + frosted glass frame mask/tint/rim + diagonal specular sweep strip.
    """
    aspect = raw_primary.width / max(1, raw_primary.height)
    max_w = int(width * (0.58 if aspect < 1.0 else 0.72))
    max_h = int(height * (card_max_h_ratio - 0.06))
    fg = raw_primary.copy()
    fg.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

    inner_pad = max(22, int(round(34 * scale_factor)))
    vitrine_w = fg.width + inner_pad * 2
    vitrine_h = fg.height + inner_pad * 2
    radius = max(12, int(round(20 * scale_factor)))

    mask_l = Image.new("L", (vitrine_w, vitrine_h), 0)
    m_draw = ImageDraw.Draw(mask_l)
    m_draw.rounded_rectangle([0, 0, vitrine_w - 1, vitrine_h - 1], radius=radius, fill=255)

    dark_tint = Image.new("RGB", (vitrine_w, vitrine_h), (20, 24, 34))

    # Build foreground overlay (1px specular pearl bevel + inner crisp matte hairline + artwork + subtle shadow)
    fg_rgba = Image.new("RGBA", (vitrine_w, vitrine_h), (0, 0, 0, 0))
    f_draw = ImageDraw.Draw(fg_rgba)
    # Soft dark recessed shadow behind inner photo inside the glass
    f_draw.rectangle(
        [inner_pad - 2, inner_pad - 1, inner_pad + fg.width + 2, inner_pad + fg.height + 4],
        fill=(0, 0, 0, 120),
    )
    fg_rgba = fg_rgba.filter(ImageFilter.GaussianBlur(radius=max(4, int(round(7 * scale_factor)))))
    f_draw = ImageDraw.Draw(fg_rgba)
    # Thin ivory-silver inner mount line
    f_draw.rectangle(
        [inner_pad - 2, inner_pad - 2, inner_pad + fg.width + 1, inner_pad + fg.height + 1],
        fill=(244, 242, 236, 255),
    )
    fg_rgba.paste(fg.convert("RGBA"), (inner_pad, inner_pad))
    # Outer glass specular pearl rim
    f_draw.rounded_rectangle(
        [1, 1, vitrine_w - 2, vitrine_h - 2],
        radius=radius,
        outline=(255, 255, 255, 115),
        width=max(1, int(round(1.8 * scale_factor))),
    )

    # Pre-render diagonal specular glass sheen band for per-frame reflection sweep
    sheen_w = max(80, int(round(vitrine_w * 0.28)))
    xx_s = np.linspace(-1.0, 1.0, sheen_w, dtype=np.float32)
    alpha_1d = (np.exp(-4.0 * (xx_s ** 2)) * 58.0).astype(np.uint8)
    sheen_rgba = np.zeros((vitrine_h, sheen_w, 4), dtype=np.uint8)
    sheen_rgba[:, :, 0] = 248
    sheen_rgba[:, :, 1] = 250
    sheen_rgba[:, :, 2] = 255
    sheen_rgba[:, :, 3] = alpha_1d[None, :]
    sheen_img = Image.fromarray(sheen_rgba, mode="RGBA").rotate(16, resample=Image.Resampling.BILINEAR, expand=True)

    return {
        "vitrine_w": vitrine_w,
        "vitrine_h": vitrine_h,
        "mask_l": mask_l,
        "dark_tint": dark_tint,
        "fg_rgba": fg_rgba,
        "sheen_img": sheen_img,
    }


def _build_medium_format_slide_card(
    img: Image.Image,
    max_w: int,
    max_h: int,
    scale_factor: float,
) -> Image.Image:
    """
    Build a Hasselblad 500C/M 120 positive transparency slide mount on an anodized ebony frame
    with rounded inner gate corners, signature V-notches, and warm highlight halation.
    """
    side_pad = max(24, int(round(40 * scale_factor)))
    top_pad = max(22, int(round(36 * scale_factor)))
    bot_pad = max(34, int(round(56 * scale_factor)))
    inner_w = max(64, max_w - side_pad * 2)
    inner_h = max(64, max_h - top_pad - bot_pad)

    fg = img.copy()
    fg.thumbnail((inner_w, inner_h), Image.Resampling.LANCZOS)
    # Subtle analog positive film richness + micro-contrast
    fg = ImageEnhance.Color(ImageEnhance.Contrast(fg).enhance(1.06)).enhance(1.05)

    sw = fg.width + side_pad * 2
    sh = fg.height + top_pad + bot_pad
    mount = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))

    # Outer slide mount rounded chassis in dark anodized charcoal
    chassis_mask = Image.new("L", (sw, sh), 0)
    ImageDraw.Draw(chassis_mask).rounded_rectangle([0, 0, sw - 1, sh - 1], radius=max(10, int(round(16 * scale_factor))), fill=255)
    chassis = Image.new("RGBA", (sw, sh), (24, 26, 31, 255))
    chassis.putalpha(chassis_mask)
    c_draw = ImageDraw.Draw(chassis)
    c_draw.rounded_rectangle(
        [0, 0, sw - 1, sh - 1],
        radius=max(10, int(round(16 * scale_factor))),
        outline=(84, 88, 98, 220),
        width=max(1, int(round(1.5 * scale_factor))),
    )

    # Rounded transparency aperture mask + signature two Hasselblad V-frame gate notches on left edge
    gate_mask = Image.new("L", fg.size, 0)
    gm_draw = ImageDraw.Draw(gate_mask)
    gm_draw.rounded_rectangle([0, 0, fg.width - 1, fg.height - 1], radius=max(4, int(round(8 * scale_factor))), fill=255)
    notch_sz = max(3, int(round(6 * scale_factor)))
    for ny_ratio in (0.14, 0.23):
        ny = int(fg.height * ny_ratio)
        gm_draw.polygon([(0, ny - notch_sz), (notch_sz + 1, ny), (0, ny + notch_sz)], fill=0)

    fg_rgba = fg.convert("RGBA")
    fg_rgba.putalpha(gate_mask)
    chassis.paste(fg_rgba, (side_pad, top_pad), fg_rgba)

    # Inner brushed silver bevel around slide gate
    c_draw.rounded_rectangle(
        [side_pad - 2, top_pad - 2, side_pad + fg.width + 1, top_pad + fg.height + 1],
        radius=max(5, int(round(9 * scale_factor))),
        outline=(118, 122, 132, 165),
        width=max(1, int(round(1.2 * scale_factor))),
    )

    # Clean darkroom film-rebate stamp in the bottom mount chin
    f_rebate = _load_editorial_font(max(9, int(round(12 * scale_factor))), bold=True, serif=False)
    rebate_txt = "120 MEDIUM FORMAT · POSITIVE TRANSPARENCY · ARCHIVAL PLATE"
    tw = c_draw.textbbox((0, 0), rebate_txt, font=f_rebate)[2]
    tx = max(side_pad, (sw - tw) // 2)
    ty = top_pad + fg.height + max(8, (bot_pad - int(round(16 * scale_factor))) // 2)
    c_draw.text((tx, ty), rebate_txt, font=f_rebate, fill=(148, 152, 162, 220))

    # Detached soft elevation drop shadow
    sh_margin = max(20, int(round(32 * scale_factor)))
    sh_offset = max(6, int(round(10 * scale_factor)))
    out = Image.new("RGBA", (sw + sh_margin * 2, sh + sh_margin * 2), (0, 0, 0, 0))
    o_draw = ImageDraw.Draw(out)
    o_draw.rounded_rectangle(
        [sh_margin, sh_margin + sh_offset, sh_margin + sw, sh_margin + sh + sh_offset],
        radius=max(10, int(round(16 * scale_factor))),
        fill=(0, 0, 0, 175),
    )
    out = out.filter(ImageFilter.GaussianBlur(radius=max(10, int(round(16 * scale_factor)))))
    out.paste(chassis, (sh_margin, sh_margin), chassis)
    return out


def _build_floating_tilt_plate_assets(
    raw_primary: Image.Image,
    width: int,
    height: int,
    card_max_h_ratio: float,
    scale_factor: float,
    border_px: Optional[int],
) -> tuple[Image.Image, Image.Image]:
    """
    Build detached foreground card & detached 3D drop shadow plane for `floating_tilt_plate`
    so the physical matte print can orbit smoothly in 3D projective perspective.
    Returns (framed_rgba, detached_shadow_rgba).
    """
    aspect = raw_primary.width / max(1, raw_primary.height)
    max_w = int(width * (0.58 if aspect < 1.0 else 0.72))
    max_h = int(height * (card_max_h_ratio - 0.04))
    framed = _build_matte_framed_rgba(
        raw_primary, max_w=max_w, max_h=max_h, scale_factor=scale_factor, border_px=border_px
    )
    fw, fh = framed.size
    sh_margin = max(24, int(round(38 * scale_factor)))
    shadow = Image.new("RGBA", (fw + sh_margin * 2, fh + sh_margin * 2), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow)
    s_draw.rectangle(
        [sh_margin + 4, sh_margin + 8, sh_margin + fw - 4, sh_margin + fh + 4],
        fill=(0, 0, 0, 155),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(14, int(round(22 * scale_factor)))))
    return framed, shadow


def _build_criterion_atelier_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
) -> dict[str, Any]:
    """
    Build Criterion / Kinfolk 3-Viewport Asymmetric Exhibition Bento Grid:
    - Left 56%: Primary hero plate
    - Right-Top 44%x49%: Secondary shot or upper architectural crown detail
    - Right-Bottom 44%x49%: Tertiary shot or zoomed focal detail around focal_point
    """
    seam_px = max(4, int(round(6 * scale_factor)))
    left_w = int(round(width * 0.56))
    left_w = left_w - (left_w % 2)
    right_w = max(32, width - left_w - seam_px)
    top_h = int(round((height - seam_px) * 0.49))
    bot_h = max(32, height - seam_px - top_h)

    osamp = 1.12
    left_over = ImageOps.fit(
        raw_primary,
        (int(round(left_w * osamp)), int(round(height * osamp))),
        method=Image.Resampling.LANCZOS,
    )

    sec_img: Optional[Image.Image] = None
    tert_img: Optional[Image.Image] = None
    if kb.secondary_source and Path(kb.secondary_source).exists():
        sec_img = ImageOps.exif_transpose(Image.open(kb.secondary_source)).convert("RGB")
    if kb.tertiary_source and Path(kb.tertiary_source).exists():
        tert_img = ImageOps.exif_transpose(Image.open(kb.tertiary_source)).convert("RGB")

    rt_src = sec_img if sec_img is not None else raw_primary
    rt_center = (0.50, 0.50) if sec_img is not None else (0.52, 0.28)
    right_top_over = ImageOps.fit(
        rt_src,
        (int(round(right_w * osamp)), int(round(top_h * osamp))),
        method=Image.Resampling.LANCZOS,
        centering=rt_center,
    )

    if tert_img is not None:
        rb_src = tert_img
        rb_over = ImageOps.fit(
            rb_src,
            (int(round(right_w * osamp)), int(round(bot_h * osamp))),
            method=Image.Resampling.LANCZOS,
        )
    else:
        # Tight macro crop around primary focal_point
        sw, sh = raw_primary.size
        fx, fy = kb.start_center
        crop_h = max(64.0, sh * 0.54)
        crop_w = crop_h * (right_w / max(1.0, float(bot_h)))
        if crop_w > sw * 0.85:
            crop_w = sw * 0.85
            crop_h = crop_w / (right_w / max(1.0, float(bot_h)))
        x0 = min(max(0.0, fx * sw - crop_w * 0.5), max(0.0, sw - crop_w))
        y0 = min(max(0.0, fy * sh - crop_h * 0.5), max(0.0, sh - crop_h))
        rb_over = raw_primary.transform(
            (int(round(right_w * osamp)), int(round(bot_h * osamp))),
            Image.Transform.EXTENT,
            (x0, y0, x0 + crop_w, y0 + crop_h),
            resample=Image.Resampling.BICUBIC,
        )

    return {
        "left_over": left_over,
        "right_top_over": right_top_over,
        "right_bot_over": rb_over,
        "left_w": left_w,
        "right_w": right_w,
        "top_h": top_h,
        "bot_h": bot_h,
        "seam_px": seam_px,
    }


def _prepare_overlay_fx_assets(
    overlay_fx: str,
    width: int,
    height: int,
    focal_point: tuple[float, float],
    scale_factor: float,
) -> dict[str, Any]:
    """Pre-compute reusable masks/textures for composable atmospheric photo finishers."""
    if not overlay_fx or overlay_fx == "none":
        return {"type": "none"}

    small_w, small_h = max(64, width // 4), max(36, height // 4)
    yy, xx = np.mgrid[0:small_h, 0:small_w]
    nx = xx / max(1.0, float(small_w - 1))
    ny = yy / max(1.0, float(small_h - 1))

    if overlay_fx == "golden_light_leak":
        # Warm amber sun-ray wash across top-left & right-bottom corners
        g1 = np.exp(-((nx - 0.08) ** 2 / 0.18 + (ny - 0.12) ** 2 / 0.22))
        g2 = np.exp(-((nx - 0.92) ** 2 / 0.25 + (ny - 0.25) ** 2 / 0.28)) * 0.65
        leak = np.clip(g1 + g2, 0.0, 1.0)
        rgb = np.zeros((small_h, small_w, 3), dtype=np.uint8)
        rgb[:, :, 0] = (leak * 255).astype(np.uint8)
        rgb[:, :, 1] = (leak * 168).astype(np.uint8)
        rgb[:, :, 2] = (leak * 52).astype(np.uint8)
        leak_img = (
            Image.fromarray(rgb, mode="RGB")
            .filter(ImageFilter.GaussianBlur(radius=8))
            .resize((width, height), Image.Resampling.BILINEAR)
        )
        return {"type": "golden_light_leak", "leak_img": leak_img}

    if overlay_fx == "museum_spotlight":
        fx_c = max(0.2, min(0.8, float(focal_point[0])))
        fy_c = max(0.2, min(0.8, float(focal_point[1])))
        dist_sq = ((nx - fx_c) ** 2) / 0.22 + ((ny - fy_c) ** 2) / 0.24
        spot = np.exp(-0.65 * dist_sq)
        dark_alpha = ((1.0 - spot) * 95.0).clip(0, 255).astype(np.uint8)
        mask_l = (
            Image.fromarray(dark_alpha, mode="L")
            .filter(ImageFilter.GaussianBlur(radius=6))
            .resize((width, height), Image.Resampling.BILINEAR)
        )
        dark_rgb = Image.new("RGB", (width, height), (5, 7, 12))
        return {"type": "museum_spotlight", "mask_l": mask_l, "dark_rgb": dark_rgb}

    if overlay_fx == "champagne_bokeh":
        # Deterministic set of 14 multi-scale golden bokeh orbs
        rng = np.random.default_rng(seed=42)
        orbs = []
        for _ in range(14):
            orbs.append(
                (
                    float(rng.uniform(0.08, 0.92)),
                    float(rng.uniform(0.12, 0.88)),
                    int(round(rng.uniform(22, 68) * scale_factor)),
                    int(rng.integers(38, 88)),
                    float(rng.uniform(0.0, math.pi * 2.0)),
                )
            )
        return {"type": "champagne_bokeh", "orbs": orbs}

    if overlay_fx == "nostalgic_film_grain":
        rng = np.random.default_rng(seed=1972)
        grain_frames: list[Image.Image] = []
        for _ in range(3):
            noise = rng.normal(loc=128.0, scale=28.0, size=(small_h * 2, small_w * 2)).clip(0, 255).astype(np.uint8)
            g_rgb = np.stack([noise, (noise * 0.96).astype(np.uint8), (noise * 0.88).astype(np.uint8)], axis=-1)
            grain_frames.append(
                Image.fromarray(g_rgb, mode="RGB").resize((width, height), Image.Resampling.BILINEAR)
            )
        return {"type": "nostalgic_film_grain", "grain_frames": grain_frames}

    if overlay_fx == "anamorphic_streak":
        # Panavision/Cooke sapphire-and-champagne horizontal flare profile
        flare_bands: list[tuple[float, float, tuple[int, int, int]]] = [
            (0.28, 0.85, (140, 195, 255)),
            (0.50, 1.00, (190, 225, 255)),
            (0.72, 0.65, (255, 215, 165)),
        ]
        streak_rgb = np.zeros((small_h, small_w, 3), dtype=np.float32)
        for by, amp, col in flare_bands:
            band = np.exp(-((ny - by) ** 2) / 0.0045) * amp
            horiz = np.exp(-((nx - 0.50) ** 2) / 0.34)
            w_map = band * horiz
            streak_rgb[:, :, 0] += w_map * col[0]
            streak_rgb[:, :, 1] += w_map * col[1]
            streak_rgb[:, :, 2] += w_map * col[2]
        streak_img = (
            Image.fromarray(np.clip(streak_rgb * 0.42, 0, 255).astype(np.uint8), mode="RGB")
            .filter(ImageFilter.GaussianBlur(radius=4))
            .resize((width, height), Image.Resampling.BILINEAR)
        )
        return {"type": "anamorphic_streak", "streak_img": streak_img}

    if overlay_fx == "prism_chromatic_edge":
        # Radial mask keeping central 62% focal area 100% sharp while smoothly revealing peripheral dispersion
        rad_sq = ((nx - 0.5) ** 2) / 0.16 + ((ny - 0.5) ** 2) / 0.18
        edge_frac = np.clip((np.sqrt(rad_sq) - 0.58) / 0.65, 0.0, 1.0)
        edge_frac = edge_frac * edge_frac * (3.0 - 2.0 * edge_frac)
        edge_mask_l = (
            Image.fromarray((edge_frac * 255.0).astype(np.uint8), mode="L")
            .filter(ImageFilter.GaussianBlur(radius=5))
            .resize((width, height), Image.Resampling.BILINEAR)
        )
        return {"type": "prism_chromatic_edge", "edge_mask_l": edge_mask_l}

    if overlay_fx == "velvet_vignette_rembrandt":
        fx_c = max(0.24, min(0.76, float(focal_point[0])))
        fy_c = max(0.24, min(0.76, float(focal_point[1])))
        # Directional upper-left Rembrandt key glow onto subject
        key_dist = ((nx - max(0.12, fx_c - 0.12)) ** 2) / 0.16 + ((ny - max(0.12, fy_c - 0.10)) ** 2) / 0.20
        key_alpha = (np.exp(-0.85 * key_dist) * 34.0).clip(0, 255).astype(np.uint8)
        key_mask = (
            Image.fromarray(key_alpha, mode="L")
            .filter(ImageFilter.GaussianBlur(radius=6))
            .resize((width, height), Image.Resampling.BILINEAR)
        )
        key_rgb = Image.new("RGB", (width, height), (255, 234, 200))

        # Asymmetric deep espresso gallery corner falloff (strictly capped below pure black)
        vig_dist = ((nx - fx_c) ** 2) / 0.24 + ((ny - fy_c) ** 2) / 0.26
        vig_alpha = ((1.0 - np.exp(-0.70 * vig_dist)) * 122.0).clip(0, 255).astype(np.uint8)
        vig_mask = (
            Image.fromarray(vig_alpha, mode="L")
            .filter(ImageFilter.GaussianBlur(radius=6))
            .resize((width, height), Image.Resampling.BILINEAR)
        )
        vig_rgb = Image.new("RGB", (width, height), (12, 10, 15))
        return {
            "type": "velvet_vignette_rembrandt",
            "key_mask": key_mask,
            "key_rgb": key_rgb,
            "vig_mask": vig_mask,
            "vig_rgb": vig_rgb,
        }

    if overlay_fx == "fine_art_rag_texture":
        rng = np.random.default_rng(seed=2026)
        coarse_h, coarse_w = max(32, small_h), max(32, small_w)
        fine = rng.normal(loc=130.0, scale=18.0, size=(small_h * 2, small_w * 2))
        fibers = np.asarray(
            Image.fromarray(
                rng.normal(loc=130.0, scale=14.0, size=(coarse_h, coarse_w)).clip(0, 255).astype(np.uint8),
                mode="L",
            )
            .filter(ImageFilter.GaussianBlur(radius=1))
            .resize((small_w * 2, small_h * 2), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        comb = np.clip(0.55 * fine + 0.45 * fibers, 0, 255).astype(np.uint8)
        rag_rgb = np.stack(
            [
                comb,
                np.clip(comb.astype(np.float32) * 0.985, 0, 255).astype(np.uint8),
                np.clip(comb.astype(np.float32) * 0.945, 0, 255).astype(np.uint8),
            ],
            axis=-1,
        )
        rag_img = Image.fromarray(rag_rgb, mode="RGB").resize((width, height), Image.Resampling.BILINEAR)
        return {"type": "fine_art_rag_texture", "rag_img": rag_img}

    return {"type": "none"}


def _apply_overlay_fx(
    frame_rgb: Image.Image,
    fx_assets: dict[str, Any],
    t_norm: float,
    f_idx: int,
    scale_factor: float,
) -> Image.Image:
    """Apply composable atmospheric finisher to frame_rgb."""
    fx_type = fx_assets.get("type", "none")
    if fx_type == "none":
        return frame_rgb

    if fx_type == "golden_light_leak":
        weight = 0.12 + 0.10 * math.sin(math.pi * t_norm)
        return Image.blend(frame_rgb, fx_assets["leak_img"], weight)

    if fx_type == "museum_spotlight":
        frame_rgb.paste(fx_assets["dark_rgb"], (0, 0), fx_assets["mask_l"])
        return frame_rgb

    if fx_type == "champagne_bokeh":
        w, h = frame_rgb.size
        sw, sh = max(64, w // 4), max(36, h // 4)
        overlay = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for ox, oy, rad, alpha, phase in fx_assets["orbs"]:
            cx = int(((ox + 0.03 * math.sin(phase + t_norm * 4.0)) % 1.0) * sw)
            cy = int(((oy - 0.10 * t_norm) % 1.0) * sh)
            r_small = max(3, rad // 4)
            pulse = int(alpha * (0.75 + 0.25 * math.sin(phase + t_norm * 6.0)))
            d.ellipse(
                [cx - r_small, cy - r_small, cx + r_small, cy + r_small],
                fill=(255, 222, 138, pulse),
                outline=(255, 240, 190, min(255, pulse + 25)),
            )
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3)).resize((w, h), Image.Resampling.BILINEAR)
        frame_rgb.paste(overlay.convert("RGB"), (0, 0), overlay.split()[3])
        return frame_rgb

    if fx_type == "nostalgic_film_grain":
        grains = fx_assets["grain_frames"]
        g_img = grains[f_idx % len(grains)]
        return Image.blend(frame_rgb, g_img, 0.065)

    if fx_type == "anamorphic_streak":
        # Dynamic specular highlight streak extracted at 1/8 scale + sapphire-champagne flare overlay
        w, h = frame_rgb.size
        sw, sh = max(48, w // 8), max(27, h // 8)
        small_arr = np.asarray(frame_rgb.resize((sw, sh), Image.Resampling.BILINEAR), dtype=np.float32)
        luma = 0.299 * small_arr[:, :, 0] + 0.587 * small_arr[:, :, 1] + 0.114 * small_arr[:, :, 2]
        hi = np.clip((luma - 205.0) / 50.0, 0.0, 1.0)
        # Fast 1D horizontal blur across rows to stretch bright specular spots into anamorphic streaks
        streak_x = hi.copy()
        for shift in (2, 5, 9, 14, 20):
            left_s = np.pad(hi[:, shift:], ((0, 0), (0, shift)), mode="constant")
            right_s = np.pad(hi[:, :-shift], ((0, 0), (shift, 0)), mode="constant")
            streak_x += 0.45 * (left_s + right_s)
        streak_x = np.clip(streak_x * 0.28, 0.0, 1.0)
        dyn_rgb = np.zeros((sh, sw, 3), dtype=np.uint8)
        dyn_rgb[:, :, 0] = (streak_x * 135.0).astype(np.uint8)
        dyn_rgb[:, :, 1] = (streak_x * 195.0).astype(np.uint8)
        dyn_rgb[:, :, 2] = (streak_x * 255.0).astype(np.uint8)
        dyn_full = Image.fromarray(dyn_rgb, mode="RGB").resize((w, h), Image.Resampling.BILINEAR)
        combined_flare = ImageChops.screen(dyn_full, fx_assets["streak_img"])
        screened = ImageChops.screen(frame_rgb, combined_flare)
        weight = 0.42 + 0.20 * math.sin(math.pi * t_norm)
        return Image.blend(frame_rgb, screened, weight)

    if fx_type == "prism_chromatic_edge":
        # Optical lateral chromatic dispersion restricted smoothly to peripheral borders via C-level PIL ops
        w, h = frame_rgb.size
        dx = max(2.0, min(w * 0.012, (4.0 + 1.5 * math.sin(math.pi * t_norm)) * scale_factor))
        dy = dx * (h / max(1.0, float(w)))
        r, g, b = frame_rgb.split()
        r_disp = r.transform((w, h), Image.Transform.EXTENT, (dx, dy, w - dx, h - dy), resample=Image.Resampling.BILINEAR)
        b_disp = b.transform((w, h), Image.Transform.EXTENT, (0.0, 0.0, w - 0.5 * dx, h - 0.5 * dy), resample=Image.Resampling.BILINEAR)
        ca_img = Image.merge("RGB", (r_disp, g, b_disp))
        return Image.composite(ca_img, frame_rgb, fx_assets["edge_mask_l"])

    if fx_type == "velvet_vignette_rembrandt":
        frame_rgb.paste(fx_assets["key_rgb"], (0, 0), fx_assets["key_mask"])
        frame_rgb.paste(fx_assets["vig_rgb"], (0, 0), fx_assets["vig_mask"])
        return frame_rgb

    if fx_type == "fine_art_rag_texture":
        return Image.blend(frame_rgb, fx_assets["rag_img"], 0.055)

    return frame_rgb



def _prepare_photo_scene_context(
    photo_path: str | Path,
    width: int,
    height: int,
    kb: KenBurnsConfig,
) -> dict[str, Any]:
    """Pre-build all multi-plane assets for any of the 10 photo presentation modes + 4 atmospheric finishers."""
    raw_primary = Image.open(photo_path)
    raw_primary = ImageOps.exif_transpose(raw_primary).convert("RGB")
    scale_factor = width / 1920.0
    s_zoom, e_zoom, (sx, sy), (ex, ey) = _resolve_kb_endpoints(kb)
    resample_mode = Image.Resampling.BILINEAR if width >= 3840 else Image.Resampling.BICUBIC

    mode = kb.mode
    # editorial_magazine renders its own right-hand monograph column so suppress the bottom lower-third card
    has_card = bool(kb.caption_title and kb.caption_title.strip()) and (mode != "editorial_magazine")
    card_fg: Optional[Image.Image] = None
    card_mask: Optional[Image.Image] = None
    card_tint: Optional[Image.Image] = None
    card_x, card_y = 0, 0
    if has_card:
        card_fg, card_mask, card_tint, card_x, card_y = _build_frosted_card_assets(
            width=width,
            height=height,
            caption_title=kb.caption_title or "",
            caption_subtitle=kb.caption_subtitle or "",
            camera_badge=kb.camera_badge or "",
        )
        if mode == "cinema_anamorphic":
            bar_h = max(0, int(round((height - width / 2.39) * 0.5)))
            card_y = min(card_y, height - bar_h - card_fg.height - int(round(18 * scale_factor)))

    bg_oversample = 1.12
    bg_w = int(round(width * bg_oversample))
    bg_h = int(round(height * bg_oversample))
    bg_small = ImageOps.fit(raw_primary, (bg_w // 4, bg_h // 4), method=Image.Resampling.BILINEAR)
    bg_small = bg_small.filter(ImageFilter.GaussianBlur(radius=max(6, int(round(8 * scale_factor)))))
    bg_plane = bg_small.resize((bg_w, bg_h), Image.Resampling.BILINEAR)
    bg_plane = Image.blend(bg_plane, Image.new("RGB", (bg_w, bg_h), (8, 10, 18)), 0.34)

    card_max_h_ratio = 0.74 if has_card else 0.80
    card_cy_ratio = 0.435 if has_card else 0.48

    ctx: dict[str, Any] = {
        "mode": mode,
        "kb": kb,
        "width": width,
        "height": height,
        "scale_factor": scale_factor,
        "s_zoom": s_zoom,
        "e_zoom": e_zoom,
        "sx": sx,
        "sy": sy,
        "ex": ex,
        "ey": ey,
        "resample_mode": resample_mode,
        "bg_plane": bg_plane,
        "bg_w": bg_w,
        "bg_h": bg_h,
        "card_cy_ratio": card_cy_ratio,
        "has_card": has_card,
        "card_fg": card_fg,
        "card_mask": card_mask,
        "card_tint": card_tint,
        "card_x": card_x,
        "card_y": card_y,
        "fx_assets": _prepare_overlay_fx_assets(
            kb.overlay_fx, width, height, kb.start_center, scale_factor
        ),
    }

    if mode == "diptych_frame":
        sec_path = kb.secondary_source
        if sec_path and Path(sec_path).exists():
            raw_sec = ImageOps.exif_transpose(Image.open(sec_path)).convert("RGB")
        else:
            raw_sec = ImageOps.fit(
                raw_primary,
                (raw_primary.width, raw_primary.height),
                method=Image.Resampling.LANCZOS,
                centering=(0.68, 0.45),
            )
        card_max_w = int(width * 0.41)
        card_max_h = int(height * card_max_h_ratio)
        ctx["fg_card_1"] = _build_double_matte_card(raw_primary, card_max_w, card_max_h, scale_factor, kb.border_px)
        ctx["fg_card_2"] = _build_double_matte_card(raw_sec, card_max_w, card_max_h, scale_factor, kb.border_px)

    elif mode == "triptych_cinematic":
        ctx["triptych_panels"] = _build_triptych_panels(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            has_card=has_card,
            scale_factor=scale_factor,
        )

    elif mode == "polaroid_scrapbook":
        stamp_1 = "" if has_card else (kb.caption_title or "")
        max_w = int(width * (0.42 if kb.secondary_source else 0.56))
        max_h = int(height * card_max_h_ratio)
        ctx["polaroid_1"] = _build_polaroid_card(
            raw_primary, max_w=max_w, max_h=max_h, scale_factor=scale_factor, angle_deg=-3.2, stamp_text=stamp_1
        )
        if kb.secondary_source and Path(kb.secondary_source).exists():
            raw_sec = ImageOps.exif_transpose(Image.open(kb.secondary_source)).convert("RGB")
            ctx["polaroid_2"] = _build_polaroid_card(
                raw_sec, max_w=max_w, max_h=max_h, scale_factor=scale_factor, angle_deg=3.6, stamp_text=""
            )
        else:
            ctx["polaroid_2"] = None

    elif mode == "cinematic_depth_3d":
        bg_sharp, bg_bokeh, fg_subject_rgba = _build_subject_depth_layers(
            raw_img=raw_primary,
            bg_w=bg_w,
            bg_h=bg_h,
            focal_point=kb.start_center,
            scale_factor=scale_factor,
        )
        ctx["depth_bg_sharp"] = bg_sharp
        ctx["depth_bg_bokeh"] = bg_bokeh
        ctx["depth_fg_rgba"] = fg_subject_rgba

    elif mode == "curator_loupe":
        dimmed_c, sharp_c, loupe_mask, loupe_bezel, rad_px = _build_curator_loupe_assets(
            raw_primary=raw_primary,
            width=width,
            height=height,
            has_card=has_card,
            scale_factor=scale_factor,
        )
        ctx["loupe_dimmed"] = dimmed_c
        ctx["loupe_sharp"] = sharp_c
        ctx["loupe_mask"] = loupe_mask
        ctx["loupe_bezel"] = loupe_bezel
        ctx["loupe_radius"] = rad_px

    elif mode == "editorial_magazine":
        left_over, right_col, left_w = _build_editorial_magazine_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
        )
        ctx["mag_left_over"] = left_over
        ctx["mag_right_col"] = right_col
        ctx["mag_left_w"] = left_w

    elif mode == "cinema_anamorphic":
        oversample = 1.12 if width >= 3840 else 1.20
        work_w = int(round(width * oversample))
        work_h = int(round(height * oversample))
        ctx["anamorphic_base"] = _build_cinema_anamorphic_base(
            raw_primary=raw_primary, work_w=work_w, work_h=work_h, scale_factor=scale_factor
        )
        ctx["anamorphic_bar_h"] = max(24, int(round((height - width / 2.39) * 0.5)))

    elif mode == "parallax_gallery":
        aspect = raw_primary.width / max(1, raw_primary.height)
        if aspect < 1.0:
            max_w, max_h = int(width * 0.66), int(height * card_max_h_ratio)
        else:
            max_w, max_h = int(width * 0.80), int(height * card_max_h_ratio)
        ctx["fg_card_1"] = _build_double_matte_card(raw_primary, max_w, max_h, scale_factor, kb.border_px)

    elif mode == "contact_sheet_zoom":
        sheet_rgb, hero_cx, hero_cy = _build_contact_sheet_zoom_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            has_card=has_card,
            scale_factor=scale_factor,
        )
        ctx["contact_sheet"] = sheet_rgb
        ctx["contact_hero_xy"] = (hero_cx, hero_cy)

    elif mode == "split_diopter_focus":
        sd_left, sd_right, sd_seam, sd_left_w = _build_split_diopter_assets(
            raw_primary=raw_primary,
            focal_point=kb.start_center,
            width=width,
            height=height,
            scale_factor=scale_factor,
        )
        ctx["sd_left_over"] = sd_left
        ctx["sd_right_over"] = sd_right
        ctx["sd_seam"] = sd_seam
        ctx["sd_left_w"] = sd_left_w

    elif mode == "glassmorphic_pedestal":
        ctx["glass_assets"] = _build_glassmorphic_pedestal_assets(
            raw_primary=raw_primary,
            width=width,
            height=height,
            card_max_h_ratio=card_max_h_ratio,
            scale_factor=scale_factor,
        )

    elif mode == "medium_format_slide":
        aspect = raw_primary.width / max(1, raw_primary.height)
        max_w = int(width * (0.64 if aspect < 1.0 else 0.78))
        max_h = int(height * (card_max_h_ratio + 0.04))
        ctx["mf_slide_card"] = _build_medium_format_slide_card(
            raw_primary, max_w=max_w, max_h=max_h, scale_factor=scale_factor
        )

    elif mode == "floating_tilt_plate":
        framed_rgba, detached_shadow = _build_floating_tilt_plate_assets(
            raw_primary=raw_primary,
            width=width,
            height=height,
            card_max_h_ratio=card_max_h_ratio,
            scale_factor=scale_factor,
            border_px=kb.border_px,
        )
        ctx["tilt_framed_rgba"] = framed_rgba
        ctx["tilt_shadow_rgba"] = detached_shadow

    elif mode == "criterion_atelier_split":
        ctx["atelier"] = _build_criterion_atelier_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
        )

    else:
        oversample = 1.10 if width >= 3840 else 1.20
        work_w = int(round(width * oversample))
        work_h = int(round(height * oversample))
        if mode == "gallery_frame":
            ctx["base_static_img"] = build_gallery_frame_image(
                photo_path, width=work_w, height=work_h, border_px=kb.border_px
            )
        else:
            ctx["base_static_img"] = ImageOps.fit(raw_primary, (work_w, work_h), method=Image.Resampling.LANCZOS)

    return ctx


def _render_photo_frame_from_context(
    ctx: dict[str, Any],
    t_norm: float,
    f_idx: int,
    duration: float,
    fps: int,
) -> Image.Image:
    """Synthesize a single RGB frame at normalized time t_norm in [0.0, 1.0] from pre-built scene context."""
    mode = ctx["mode"]
    width = ctx["width"]
    height = ctx["height"]
    scale_factor = ctx["scale_factor"]
    s_zoom, e_zoom = ctx["s_zoom"], ctx["e_zoom"]
    sx, sy, ex, ey = ctx["sx"], ctx["sy"], ctx["ex"], ctx["ey"]
    resample_mode = ctx["resample_mode"]
    bg_plane = ctx["bg_plane"]
    bg_w, bg_h = ctx["bg_w"], ctx["bg_h"]
    card_cy_ratio = ctx["card_cy_ratio"]
    eased = _ease_in_out_cubic(t_norm)

    if mode == "diptych_frame":
        fg_card_1 = ctx["fg_card_1"]
        fg_card_2 = ctx["fg_card_2"]
        bg_zoom = 1.10 - 0.04 * eased
        bg_cx = 0.52 - 0.04 * eased
        bg_cy = 0.50
        bw_crop = bg_w / bg_zoom
        bh_crop = bg_h / bg_zoom
        bx0 = min(max(0.0, bg_cx * bg_w - bw_crop * 0.5), bg_w - bw_crop)
        by0 = min(max(0.0, bg_cy * bg_h - bh_crop * 0.5), bg_h - bh_crop)
        frame = bg_plane.transform(
            (width, height),
            Image.Transform.EXTENT,
            (bx0, by0, bx0 + bw_crop, by0 + bh_crop),
            resample=resample_mode,
        )
        drift_px = int(round((eased - 0.5) * 24.0 * scale_factor))
        left_cx = int(width * 0.28)
        right_cx = int(width * 0.72)
        mid_cy = int(height * card_cy_ratio)
        lx = left_cx - fg_card_1.width // 2
        ly = mid_cy - fg_card_1.height // 2 - drift_px
        rx = right_cx - fg_card_2.width // 2
        ry = mid_cy - fg_card_2.height // 2 + drift_px
        frame.paste(fg_card_1, (lx, ly), fg_card_1)
        frame.paste(fg_card_2, (rx, ry), fg_card_2)

    elif mode == "triptych_cinematic":
        panels: list[Image.Image] = ctx["triptych_panels"]
        bg_zoom = 1.10 - 0.04 * eased
        bw_crop = bg_w / bg_zoom
        bh_crop = bg_h / bg_zoom
        bg_cx = 0.50 - 0.03 * (eased - 0.5)
        bx0 = min(max(0.0, bg_cx * bg_w - bw_crop * 0.5), bg_w - bw_crop)
        by0 = min(max(0.0, 0.50 * bg_h - bh_crop * 0.5), bg_h - bh_crop)
        frame = bg_plane.transform(
            (width, height),
            Image.Transform.EXTENT,
            (bx0, by0, bx0 + bw_crop, by0 + bh_crop),
            resample=resample_mode,
        )
        centers_x = [int(width * 0.20), int(width * 0.50), int(width * 0.80)]
        mid_cy = int(height * card_cy_ratio)
        drift_dirs = [-1.0, 0.85, -1.0]
        for p_i, p_card in enumerate(panels):
            # Staggered entrance cascade over the first 25% of the clip
            stagger_start = p_i * 0.06
            enter_p = _ease_out_cubic((t_norm - stagger_start) / 0.18) if t_norm > stagger_start else 0.0
            if enter_p <= 0.02:
                continue
            slide_in_y = int(round((1.0 - enter_p) * 38.0 * scale_factor))
            drift_y = int(round((eased - 0.5) * 20.0 * scale_factor * drift_dirs[p_i]))
            px = centers_x[p_i] - p_card.width // 2
            py = mid_cy - p_card.height // 2 + drift_y + slide_in_y
            frame.paste(p_card, (px, max(8, py)), p_card)

    elif mode == "polaroid_scrapbook":
        p1: Image.Image = ctx["polaroid_1"]
        p2: Optional[Image.Image] = ctx["polaroid_2"]
        bg_zoom = 1.10 - 0.04 * eased
        bw_crop = bg_w / bg_zoom
        bh_crop = bg_h / bg_zoom
        bg_cx = 0.50 + 0.03 * (eased - 0.5)
        bx0 = min(max(0.0, bg_cx * bg_w - bw_crop * 0.5), bg_w - bw_crop)
        by0 = min(max(0.0, 0.50 * bg_h - bh_crop * 0.5), bg_h - bh_crop)
        frame = bg_plane.transform(
            (width, height),
            Image.Transform.EXTENT,
            (bx0, by0, bx0 + bw_crop, by0 + bh_crop),
            resample=resample_mode,
        )
        mid_cy = int(height * card_cy_ratio)
        if p2 is not None:
            # Dual Polaroid Scrapbook Stack
            drift = int(round((eased - 0.5) * 18.0 * scale_factor))
            x1 = int(width * 0.33) - p1.width // 2
            y1 = mid_cy - p1.height // 2 - drift
            frame.paste(p1, (x1, max(8, y1)), p1)

            enter_2 = _ease_out_cubic((t_norm - 0.08) / 0.24) if t_norm > 0.08 else 0.0
            if enter_2 > 0.02:
                slide_x = int(round((1.0 - enter_2) * 52.0 * scale_factor))
                x2 = int(width * 0.67) - p2.width // 2 + slide_x
                y2 = mid_cy - p2.height // 2 + drift
                frame.paste(p2, (x2, max(8, y2)), p2)
        else:
            scale_p = 0.96 + 0.05 * eased
            cw = max(1, int(round(p1.width * scale_p)))
            ch = max(1, int(round(p1.height * scale_p)))
            scaled_p1 = p1.resize((cw, ch), resample=resample_mode)
            px = (width - cw) // 2 + int(round((eased - 0.5) * 22.0 * scale_factor))
            py = mid_cy - ch // 2
            frame.paste(scaled_p1, (px, max(8, py)), scaled_p1)

    elif mode == "cinematic_depth_3d":
        bg_sharp: Image.Image = ctx["depth_bg_sharp"]
        bg_bokeh: Image.Image = ctx["depth_bg_bokeh"]
        fg_rgba: Image.Image = ctx["depth_fg_rgba"]

        # Plane 1: Background counter-zooms (1.10 -> 1.06) and racks focus into creamy bokeh
        bg_zoom = 1.10 - 0.04 * eased
        fg_dx = (ex - sx) * eased
        fg_dy = (ey - sy) * eased
        bg_cx = 0.5 - fg_dx * 0.45
        bg_cy = 0.5 - fg_dy * 0.45
        bw_crop = bg_w / bg_zoom
        bh_crop = bg_h / bg_zoom
        bx0 = min(max(0.0, bg_cx * bg_w - bw_crop * 0.5), bg_w - bw_crop)
        by0 = min(max(0.0, bg_cy * bg_h - bh_crop * 0.5), bg_h - bh_crop)

        sharp_crop = bg_sharp.transform(
            (width, height), Image.Transform.EXTENT, (bx0, by0, bx0 + bw_crop, by0 + bh_crop), resample=resample_mode
        )
        bokeh_crop = bg_bokeh.transform(
            (width, height), Image.Transform.EXTENT, (bx0, by0, bx0 + bw_crop, by0 + bh_crop), resample=resample_mode
        )
        bokeh_mix = min(0.88, 0.30 + 0.58 * eased)
        frame = Image.blend(sharp_crop, bokeh_crop, bokeh_mix)

        # Plane 2: Foreground subject expands forward (1.00 -> 1.085) with true differential parallax
        fg_zoom = 1.00 + 0.085 * eased
        fg_cx = 0.5 + fg_dx * 0.35
        fg_cy = 0.5 + fg_dy * 0.35
        fw_crop = bg_w / fg_zoom
        fh_crop = bg_h / fg_zoom
        fx0 = min(max(0.0, fg_cx * bg_w - fw_crop * 0.5), bg_w - fw_crop)
        fy0 = min(max(0.0, fg_cy * bg_h - fh_crop * 0.5), bg_h - fh_crop)

        fg_warped = fg_rgba.transform(
            (width, height), Image.Transform.EXTENT, (fx0, fy0, fx0 + fw_crop, fy0 + fh_crop), resample=resample_mode
        )
        frame.paste(fg_warped.convert("RGB"), (0, 0), fg_warped.split()[3])

    elif mode == "curator_loupe":
        dimmed_c: Image.Image = ctx["loupe_dimmed"]
        sharp_c: Image.Image = ctx["loupe_sharp"]
        loupe_mask: Image.Image = ctx["loupe_mask"]
        loupe_bezel: Image.Image = ctx["loupe_bezel"]
        rad: int = ctx["loupe_radius"]
        diam = rad * 2

        frame = dimmed_c.copy()
        # Smoothly glide the optical loupe across the focal path
        lx_norm = (sx - 0.08) + ((ex + 0.08) - (sx - 0.08)) * eased
        ly_norm = (sy + 0.03 * math.sin(math.pi * eased)) * 0.88
        lcx = int(round(max(0.24, min(0.76, lx_norm)) * width))
        lcy = int(round(max(0.24, min(0.68, ly_norm)) * height))

        # Extract 1.65x magnified patch from sharp_c around (lcx, lcy)
        mag = 1.65
        src_half = rad / mag
        sx0 = max(0.0, min(float(width) - 2 * src_half, lcx - src_half))
        sy0 = max(0.0, min(float(height) - 2 * src_half, lcy - src_half))
        magnified_patch = sharp_c.transform(
            (diam, diam),
            Image.Transform.EXTENT,
            (sx0, sy0, sx0 + 2 * src_half, sy0 + 2 * src_half),
            resample=Image.Resampling.BICUBIC,
        )
        dst_x = lcx - rad
        dst_y = lcy - rad
        frame.paste(magnified_patch, (dst_x, dst_y), loupe_mask)
        bz_pad = (loupe_bezel.width - diam) // 2
        frame.paste(loupe_bezel, (dst_x - bz_pad, dst_y - bz_pad), loupe_bezel)

    elif mode == "editorial_magazine":
        left_over: Image.Image = ctx["mag_left_over"]
        right_col: Image.Image = ctx["mag_right_col"]
        left_w: int = ctx["mag_left_w"]
        ow, oh = left_over.size

        cur_zoom = 1.0 + 0.08 * eased
        cx = sx + (ex - sx) * eased
        cy = sy + (ey - sy) * eased
        cw = ow / cur_zoom
        ch = oh / cur_zoom
        x0 = min(max(0.0, cx * ow - cw * 0.5), ow - cw)
        y0 = min(max(0.0, cy * oh - ch * 0.5), oh - ch)

        left_crop = left_over.transform(
            (left_w, height),
            Image.Transform.EXTENT,
            (x0, y0, x0 + cw, y0 + ch),
            resample=resample_mode,
        )
        frame = Image.new("RGB", (width, height))
        frame.paste(left_crop, (0, 0))
        frame.paste(right_col, (left_w, 0))

    elif mode == "cinema_anamorphic":
        ana_base: Image.Image = ctx["anamorphic_base"]
        work_w, work_h = ana_base.size
        cur_zoom = s_zoom + (e_zoom - s_zoom) * eased
        cx = sx + (ex - sx) * eased
        cy = sy + (ey - sy) * eased
        crop_w = work_w / cur_zoom
        crop_h = work_h / cur_zoom
        x0 = min(max(0.0, cx * work_w - crop_w * 0.5), work_w - crop_w)
        y0 = min(max(0.0, cy * work_h - crop_h * 0.5), work_h - crop_h)

        frame = ana_base.transform(
            (width, height),
            Image.Transform.EXTENT,
            (x0, y0, x0 + crop_w, y0 + crop_h),
            resample=resample_mode,
        )

    elif mode == "parallax_gallery":
        fg_card_1 = ctx["fg_card_1"]
        fg_dx = (ex - sx) * eased
        fg_dy = (ey - sy) * eased
        frame = _crop_plane_clamped(
            bg_plane, (width, height), 1.10 - 0.04 * eased, 0.5 - fg_dx * 0.55, 0.5 - fg_dy * 0.55, resample_mode
        )
        fg_scale = 0.96 + 0.05 * eased
        cw = max(1, int(round(fg_card_1.width * fg_scale)))
        ch = max(1, int(round(fg_card_1.height * fg_scale)))
        scaled_card = fg_card_1.resize((cw, ch), resample=resample_mode)
        pan_px_x = int(round(fg_dx * width * 0.28))
        pan_px_y = int(round(fg_dy * height * 0.24))
        px = (width - cw) // 2 + pan_px_x
        py = int(height * card_cy_ratio) - ch // 2 + pan_px_y
        frame.paste(scaled_card, (px, max(10, py)), scaled_card)

    elif mode == "contact_sheet_zoom":
        sheet: Image.Image = ctx["contact_sheet"]
        hcx, hcy = ctx["contact_hero_xy"]
        if t_norm <= 0.14:
            cur_zoom = 1.00 + 0.04 * (t_norm / 0.14)
            cx, cy = 0.50, 0.50
        else:
            push_p = _ease_in_out_cubic(min(1.0, (t_norm - 0.14) / 0.86))
            cur_zoom = 1.04 + 1.48 * push_p
            cx = 0.50 + (hcx - 0.50) * push_p
            cy = 0.50 + (hcy - 0.50) * push_p
        frame = _crop_plane_clamped(sheet, (width, height), cur_zoom, cx, cy, resample_mode)

    elif mode == "split_diopter_focus":
        left_over: Image.Image = ctx["sd_left_over"]
        right_over: Image.Image = ctx["sd_right_over"]
        seam_rgba: Image.Image = ctx["sd_seam"]
        left_w: int = ctx["sd_left_w"]
        right_w = width - left_w

        left_crop = _crop_plane_clamped(
            left_over, (left_w, height), 1.09 - 0.08 * eased, 0.50 - 0.03 * (eased - 0.5), 0.50, resample_mode
        )
        right_crop = _crop_plane_clamped(
            right_over,
            (right_w, height),
            1.00 + 0.11 * eased,
            0.50 + (ex - sx) * eased * 0.35,
            0.50 + (ey - sy) * eased * 0.35,
            resample_mode,
        )
        frame = Image.new("RGB", (width, height))
        frame.paste(left_crop, (0, 0))
        frame.paste(right_crop, (left_w, 0))
        frame.paste(seam_rgba.convert("RGB"), (max(0, left_w - seam_rgba.width // 2), 0), seam_rgba.split()[3])

    elif mode == "glassmorphic_pedestal":
        ga = ctx["glass_assets"]
        vw, vh = ga["vitrine_w"], ga["vitrine_h"]
        fg_dx = (ex - sx) * eased
        frame = _crop_plane_clamped(
            bg_plane, (width, height), 1.10 - 0.04 * eased, 0.50 - fg_dx * 0.45, 0.50, resample_mode
        )
        px = max(4, min(width - vw - 4, (width - vw) // 2 + int(round(fg_dx * width * 0.16))))
        py = max(4, min(height - vh - 4, int(height * card_cy_ratio) - vh // 2))

        # True live backdrop blur inside the rounded frosted acrylic vitrine frame
        patch = frame.crop((px, py, px + vw, py + vh))
        sw_p, sh_p = max(16, vw // 4), max(16, vh // 4)
        blurred = (
            patch.resize((sw_p, sh_p), Image.Resampling.BILINEAR)
            .filter(ImageFilter.GaussianBlur(radius=6))
            .resize((vw, vh), Image.Resampling.BILINEAR)
        )
        vitrine = Image.blend(blurred, ga["dark_tint"], 0.42)
        fg_rgba: Image.Image = ga["fg_rgba"]
        vitrine.paste(fg_rgba.convert("RGB"), (0, 0), fg_rgba.split()[3])

        # Sweeping diagonal glass specular reflection sheen across the vitrine surface
        sheen: Image.Image = ga["sheen_img"]
        sheen_x = int(round(-sheen.width * 0.4 + (vw + sheen.width * 0.8) * eased))
        sheen_y = (vh - sheen.height) // 2
        vitrine.paste(sheen.convert("RGB"), (sheen_x, sheen_y), sheen.split()[3])

        frame.paste(vitrine, (px, py), ga["mask_l"])

    elif mode == "medium_format_slide":
        slide_card: Image.Image = ctx["mf_slide_card"]
        fg_dx = (ex - sx) * eased
        frame = _crop_plane_clamped(
            bg_plane, (width, height), 1.10 - 0.04 * eased, 0.50 + fg_dx * 0.35, 0.50, resample_mode
        )
        scale_s = 0.96 + 0.05 * eased
        cw = max(1, int(round(slide_card.width * scale_s)))
        ch = max(1, int(round(slide_card.height * scale_s)))
        scaled_s = slide_card.resize((cw, ch), resample=resample_mode)
        px = (width - cw) // 2 + int(round((eased - 0.5) * 18.0 * scale_factor))
        py = max(8, int(height * card_cy_ratio) - ch // 2)
        frame.paste(scaled_s, (px, py), scaled_s)

    elif mode == "floating_tilt_plate":
        framed_rgba: Image.Image = ctx["tilt_framed_rgba"]
        shadow_rgba: Image.Image = ctx["tilt_shadow_rgba"]
        frame = _crop_plane_clamped(
            bg_plane, (width, height), 1.10 - 0.04 * eased, 0.52 - 0.04 * eased, 0.50, resample_mode
        )
        fw, fh = framed_rgba.size
        sw_sh, sh_sh = shadow_rgba.size

        # Sculptural 3D yaw sweep (+11.5 deg toward +1.7 deg) + subtle elevation rise
        yaw_norm = 0.095 * (1.0 - 0.85 * eased)
        s_out = 0.95 + 0.045 * eased
        half_w = fw * s_out * (1.0 - 0.14 * abs(yaw_norm)) * 0.5
        half_h_l = (fh * s_out * 0.5) * (1.0 + yaw_norm)
        half_h_r = (fh * s_out * 0.5) * (1.0 - yaw_norm)
        cx_out = width * (0.49 + 0.02 * eased)
        cy_out = height * card_cy_ratio - (1.0 - eased) * 6.0 * scale_factor

        # 1. Detached 3D perspective shadow offset below and right of print plane
        sh_ratio_w = sw_sh / max(1.0, float(fw))
        sh_ratio_h = sh_sh / max(1.0, float(fh))
        sh_off_x = 12.0 * scale_factor + 14.0 * scale_factor * (1.0 - eased)
        sh_off_y = 18.0 * scale_factor
        sh_cx, sh_cy = cx_out + sh_off_x, cy_out + sh_off_y
        sh_hw = half_w * sh_ratio_w
        sh_hhl, sh_hhr = half_h_l * sh_ratio_h, half_h_r * sh_ratio_h
        sb0 = max(0, int(math.floor(sh_cx - sh_hw)) - 2)
        sb1 = min(width, int(math.ceil(sh_cx + sh_hw)) + 2)
        st0 = max(0, int(math.floor(sh_cy - max(sh_hhl, sh_hhr))) - 2)
        st1 = min(height, int(math.ceil(sh_cy + max(sh_hhl, sh_hhr))) + 2)
        if sb1 > sb0 + 4 and st1 > st0 + 4:
            sh_dst = [
                (sh_cx - sh_hw - sb0, sh_cy - sh_hhl - st0),
                (sh_cx + sh_hw - sb0, sh_cy - sh_hhr - st0),
                (sh_cx + sh_hw - sb0, sh_cy + sh_hhr - st0),
                (sh_cx - sh_hw - sb0, sh_cy + sh_hhl - st0),
            ]
            sh_src = [(0.0, 0.0), (float(sw_sh), 0.0), (float(sw_sh), float(sh_sh)), (0.0, float(sh_sh))]
            sh_coeffs = _solve_perspective_coeffs(sh_dst, sh_src)
            sh_warped = shadow_rgba.transform(
                (sb1 - sb0, st1 - st0), Image.Transform.PERSPECTIVE, sh_coeffs, resample=Image.Resampling.BILINEAR
            )
            frame.paste(sh_warped.convert("RGB"), (sb0, st0), sh_warped.split()[3])

        # 2. 3D perspective projective homography on crisp museum matte print
        bx0 = max(0, int(math.floor(cx_out - half_w)) - 2)
        bx1 = min(width, int(math.ceil(cx_out + half_w)) + 2)
        by0 = max(0, int(math.floor(cy_out - max(half_h_l, half_h_r))) - 2)
        by1 = min(height, int(math.ceil(cy_out + max(half_h_l, half_h_r))) + 2)
        if bx1 > bx0 + 4 and by1 > by0 + 4:
            fg_dst = [
                (cx_out - half_w - bx0, cy_out - half_h_l - by0),
                (cx_out + half_w - bx0, cy_out - half_h_r - by0),
                (cx_out + half_w - bx0, cy_out + half_h_r - by0),
                (cx_out - half_w - bx0, cy_out + half_h_l - by0),
            ]
            fg_src = [(0.0, 0.0), (float(fw), 0.0), (float(fw), float(fh)), (0.0, float(fh))]
            fg_coeffs = _solve_perspective_coeffs(fg_dst, fg_src)
            fg_warped = framed_rgba.transform(
                (bx1 - bx0, by1 - by0), Image.Transform.PERSPECTIVE, fg_coeffs, resample=resample_mode
            )
            frame.paste(fg_warped.convert("RGB"), (bx0, by0), fg_warped.split()[3])

    elif mode == "criterion_atelier_split":
        at = ctx["atelier"]
        lw, rw = at["left_w"], at["right_w"]
        th, bh = at["top_h"], at["bot_h"]
        seam_px = at["seam_px"]

        left_crop = _crop_plane_clamped(
            at["left_over"], (lw, height), 1.00 + 0.075 * eased, 0.49 + 0.02 * eased, 0.50, resample_mode
        )
        rt_crop = _crop_plane_clamped(
            at["right_top_over"], (rw, th), 1.075 - 0.075 * eased, 0.52 - 0.03 * eased, 0.50, resample_mode
        )
        rb_crop = _crop_plane_clamped(
            at["right_bot_over"], (rw, bh), 1.00 + 0.09 * eased, 0.50, 0.50, resample_mode
        )
        frame = Image.new("RGB", (width, height), (14, 16, 20))
        frame.paste(left_crop, (0, 0))
        frame.paste(rt_crop, (lw + seam_px, 0))
        frame.paste(rb_crop, (lw + seam_px, th + seam_px))
        # Subtle warm stone hairline accent inside the exhibition grid seams
        draw = ImageDraw.Draw(frame)
        draw.line([(lw + seam_px // 2, 0), (lw + seam_px // 2, height)], fill=(122, 118, 110), width=1)
        draw.line([(lw + seam_px, th + seam_px // 2), (width, th + seam_px // 2)], fill=(122, 118, 110), width=1)

    else:
        base_static_img: Image.Image = ctx["base_static_img"]
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

    # Apply composable atmospheric finisher (golden_light_leak, champagne_bokeh, museum_spotlight, nostalgic_film_grain)
    frame = _apply_overlay_fx(frame, ctx["fx_assets"], t_norm=t_norm, f_idx=f_idx, scale_factor=scale_factor)

    # If 2.39:1 cinema_anamorphic, mask top & bottom CinemaScope letterbox bars after atmospheric finishers so bars remain pure theatrical black
    if mode == "cinema_anamorphic":
        bar_h: int = ctx["anamorphic_bar_h"]
        draw = ImageDraw.Draw(frame)
        draw.rectangle([0, 0, width, bar_h], fill=(4, 5, 8))
        draw.rectangle([0, height - bar_h, width, height], fill=(4, 5, 8))

    # Plane 3: Frosted-Glass Documentary Lower-Third Card (with smooth upward glide + fade)
    if ctx["has_card"] and ctx["card_fg"] is not None and ctx["card_mask"] is not None and ctx["card_tint"] is not None:
        t_sec = f_idx / max(1.0, float(fps))
        rem_sec = max(0.0, duration - t_sec)
        entry_prog = _ease_out_cubic((t_sec - 0.15) / 0.50) if t_sec > 0.15 else 0.0
        exit_prog = min(1.0, rem_sec / 0.40)
        alpha_mult = min(entry_prog, exit_prog)
        if alpha_mult > 0.04:
            slide_y = int(round((1.0 - entry_prog) * 16.0 * scale_factor))
            _composite_frosted_card_on_rgb(
                frame_rgb=frame,
                text_fg_rgba=ctx["card_fg"],
                card_mask_l=ctx["card_mask"],
                dark_tint_rgb=ctx["card_tint"],
                px=ctx["card_x"],
                py=ctx["card_y"],
                alpha_mult=alpha_mult,
                slide_y=slide_y,
            )

    return frame


def render_photo_clip(
    photo_path: str | Path,
    output_mp4: str | Path,
    duration: float = 5.5,
    width: int = 3840,
    height: int = 2160,
    fps: int = 30,
    hdr: bool = True,
    ken_burns: Optional[KenBurnsConfig] = None,
    transition_in: Optional[TransitionConfig] = None,
    transition_out: Optional[TransitionConfig] = None,
) -> Path:
    """Render a photo into a 4K HDR MP4 video clip using any of the 10 Premium Photo Modes + 4 Atmospheric Finishers."""
    kb = ken_burns or KenBurnsConfig()
    out_path = Path(output_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = max(1, int(round(duration * fps)))
    ctx = _prepare_photo_scene_context(photo_path=photo_path, width=width, height=height, kb=kb)

    # Prepare FFmpeg encoder & HDR color space pipeline
    vf_filters: list[str] = ["setsar=1:1"]
    if transition_in and transition_in.type not in ("none", "cut") and transition_in.duration > 0:
        if transition_in.type == "dip_to_white":
            fd = min(duration * 0.35, transition_in.duration)
            vf_filters.append(f"fade=t=in:st=0:d={fd:.3f}:c=#F6F2E8")
        elif transition_in.type in ("dissolve", "crossfade"):
            fd = min(duration * 0.25, transition_in.duration, 0.26)
            vf_filters.append(f"fade=t=in:st=0:d={fd:.3f}:c=#0B0E14")
        else:
            fd = min(duration * 0.40, transition_in.duration)
            vf_filters.append(f"fade=t=in:st=0:d={fd:.3f}")
    if transition_out and transition_out.type not in ("none", "cut") and transition_out.duration > 0:
        if transition_out.type == "dip_to_white":
            fade_d = min(duration * 0.35, transition_out.duration)
            st = max(0.0, duration - fade_d)
            vf_filters.append(f"fade=t=out:st={st:.3f}:d={fade_d:.3f}:c=#F6F2E8")
        elif transition_out.type in ("dissolve", "crossfade"):
            fade_d = min(duration * 0.25, transition_out.duration, 0.26)
            st = max(0.0, duration - fade_d)
            vf_filters.append(f"fade=t=out:st={st:.3f}:d={fade_d:.3f}:c=#0B0E14")
        else:
            fade_d = min(duration * 0.40, transition_out.duration)
            st = max(0.0, duration - fade_d)
            vf_filters.append(f"fade=t=out:st={st:.3f}:d={fade_d:.3f}")

    if hdr:
        vf_filters.extend(build_hdr_color_filters("iec61966-2-1"))
    else:
        vf_filters.append("format=yuv420p")
    v_enc = get_video_encoder_args(hdr=hdr, force_software=False)

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

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    try:
        for f_idx in range(total_frames):
            t_norm = f_idx / max(1, total_frames - 1)
            frame = _render_photo_frame_from_context(
                ctx=ctx,
                t_norm=t_norm,
                f_idx=f_idx,
                duration=duration,
                fps=fps,
            )
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()

    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    ret = proc.wait()
    if ret != 0:
        err_msg = stderr_bytes.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg photo clip render failed (code {ret}): {err_msg}")

    return out_path


def render_photo_effects_showcase_sheet(
    photo_path: str | Path,
    output_jpg: str | Path,
    secondary_path: Optional[str | Path] = None,
    tertiary_path: Optional[str | Path] = None,
    caption_title: str = "Curated Architectural & Fine Art Showcase",
    caption_subtitle: str = "16-Mode Cinema & Exhibition Suite + 8 Optical Finishers · 4K HDR",
    camera_badge: str = "",
    tile_width: int = 960,
    tile_height: int = 540,
) -> Path:
    """
    Mechanically render a 4x4 (16-tile) high-resolution visual comparison showcase sheet
    covering all 16 Photo Presentation Modes and all 8 Atmospheric Finishers.
    """
    out_path = Path(output_jpg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sec_str = str(secondary_path) if (secondary_path and Path(secondary_path).exists()) else None
    tert_str = str(tertiary_path) if (tertiary_path and Path(tertiary_path).exists()) else None

    showcase_specs: list[tuple[str, str, str]] = [
        ("cinematic_depth_3d", "golden_light_leak", "1. 3D Depth Parallax + Golden Leak"),
        ("contact_sheet_zoom", "none", "2. Darkroom Contact Sheet Push-In"),
        ("split_diopter_focus", "prism_chromatic_edge", "3. Optical Split Diopter + Prism Edge"),
        ("glassmorphic_pedestal", "none", "4. Frosted Glassmorphic Vitrine"),
        ("medium_format_slide", "velvet_vignette_rembrandt", "5. 120 Medium-Format Lightbox Slide"),
        ("floating_tilt_plate", "fine_art_rag_texture", "6. 3D Floating Tilt Plate + Rag Paper"),
        ("criterion_atelier_split", "none", "7. Criterion 3-Viewport Atelier Grid"),
        ("triptych_cinematic", "none", "8. 3-Panel Staggered Triptych"),
        ("polaroid_scrapbook", "nostalgic_film_grain", "9. Vintage Polaroid + 35mm Grain"),
        ("curator_loupe", "museum_spotlight", "10. 1.65x Optical Curator Loupe"),
        ("editorial_magazine", "none", "11. 62/38 Editorial Magazine Spread"),
        ("cinema_anamorphic", "anamorphic_streak", "12. 2.39:1 Anamorphic + Flare Streak"),
        ("parallax_gallery", "museum_spotlight", "13. 2.5D Museum Parallax + Spotlight"),
        ("diptych_frame", "golden_light_leak", "14. Side-by-Side Diptych Split-Screen"),
        ("gallery_frame", "champagne_bokeh", "15. Classic Gallery Frame + Bokeh"),
        ("full_bleed", "velvet_vignette_rembrandt", "16. Full-Bleed Ken Burns + Chiaroscuro"),
    ]

    cols, rows = 4, 4
    pad = 18
    header_h = 68
    label_h = 36
    canvas_w = cols * tile_width + (cols + 1) * pad
    canvas_h = header_h + rows * (tile_height + label_h) + (rows + 1) * pad

    sheet = Image.new("RGB", (canvas_w, canvas_h), (12, 14, 18))
    draw = ImageDraw.Draw(sheet)
    f_hdr = _load_editorial_font(26, bold=True, serif=True)
    f_lbl = _load_editorial_font(15, bold=True, serif=False)

    draw.text(
        (pad, 20),
        "PREMIUM PHOTO EFFECTS SHOWCASE — 16 PRESENTATION MODES & 8 OPTICAL FINISHERS",
        font=f_hdr,
        fill=(242, 238, 230),
    )

    for idx, (mode_name, fx_name, label_text) in enumerate(showcase_specs):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (tile_width + pad)
        y0 = header_h + pad + r * (tile_height + label_h + pad)

        kb = KenBurnsConfig(
            mode=mode_name,  # type: ignore[arg-type]
            overlay_fx=fx_name,  # type: ignore[arg-type]
            preset="zoom_in",
            start_zoom=1.0,
            end_zoom=1.08,
            start_center=(0.50, 0.48),
            end_center=(0.52, 0.45),
            secondary_source=sec_str,
            tertiary_source=tert_str,
            caption_title=caption_title,
            caption_subtitle=caption_subtitle,
            camera_badge=camera_badge,
        )
        ctx = _prepare_photo_scene_context(photo_path=photo_path, width=tile_width, height=tile_height, kb=kb)
        tile_img = _render_photo_frame_from_context(ctx=ctx, t_norm=0.55, f_idx=16, duration=5.5, fps=30)

        draw.rectangle(
            [x0 - 1, y0 - 1, x0 + tile_width, y0 + tile_height + label_h],
            outline=(118, 122, 132),
            width=1,
        )
        sheet.paste(tile_img, (x0, y0))
        draw.rectangle([x0, y0 + tile_height, x0 + tile_width, y0 + tile_height + label_h], fill=(18, 21, 28))
        draw.text(
            (x0 + 10, y0 + tile_height + 8),
            f"{label_text}  [{mode_name} | {fx_name}]",
            font=f_lbl,
            fill=(232, 236, 242),
        )

    sheet.save(out_path, quality=92)
    return out_path
