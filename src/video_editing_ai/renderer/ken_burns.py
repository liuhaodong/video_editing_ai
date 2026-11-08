"""Sub-pixel 2.5D Parallax, Diptych Split-Screen, Museum Gallery Frame, and Frosted-Glass Documentary Card renderer."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import subprocess
import textwrap
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

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


def _build_double_matte_card(
    img: Image.Image,
    max_w: int,
    max_h: int,
    scale_factor: float,
    border_px: Optional[int] = None,
) -> Image.Image:
    """Create a foreground photo card with a double museum passe-partout matte (warm cotton-rag ivory + subtle archival stone bevel) and 3D drop shadow."""
    b_px = border_px if border_px is not None else max(10, int(round(14 * scale_factor)))
    bevel_pad = max(2, int(round(3 * scale_factor)))
    shadow_margin = max(18, int(round(28 * scale_factor)))
    shadow_offset = max(4, int(round(7 * scale_factor)))
    shadow_blur = max(10, int(round(16 * scale_factor)))

    fg = img.copy()
    fg.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

    fw = fg.width + (b_px + bevel_pad) * 2
    fh = fg.height + (b_px + bevel_pad) * 2

    # Warm museum cotton-rag ivory outer matte
    framed = Image.new("RGBA", (fw, fh), (247, 245, 240, 255))
    f_draw = ImageDraw.Draw(framed)
    # Subtle archival warm-stone inner bevel around the artwork (no garish yellow)
    f_draw.rectangle(
        [b_px - 2, b_px - 2, fw - b_px + 1, fh - b_px + 1],
        outline=(212, 208, 198, 230),
        width=max(1, int(round(1.5 * scale_factor))),
    )
    framed.paste(fg, (b_px + bevel_pad, b_px + bevel_pad))

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
        fill=(0, 0, 0, 165),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
    shadow.paste(framed, (shadow_margin, shadow_margin), framed)
    return shadow


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


def build_hdr_color_filters(src_trc: str = "iec61966-2-1") -> list[str]:
    """
    Build 10-bit HLG HDR color-space filters, automatically falling back when 'zscale'
    (libzimg) is omitted in macOS Homebrew / static FFmpeg builds.
    """
    filters: list[str] = []
    if src_trc not in ("arib-std-b67", "smpte2084") and has_ffmpeg_filter("zscale"):
        filters.append(
            f"zscale=primariesin=709:transferin={src_trc}:matrixin=709:"
            "primaries=2020:transfer=arib-std-b67:matrix=2020_ncl"
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
        bg_zoom = 1.10 - 0.04 * eased
        fg_dx = (ex - sx) * eased
        fg_dy = (ey - sy) * eased
        bg_cx = 0.5 - fg_dx * 0.55
        bg_cy = 0.5 - fg_dy * 0.55
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
        fg_scale = 0.96 + 0.05 * eased
        cw = max(1, int(round(fg_card_1.width * fg_scale)))
        ch = max(1, int(round(fg_card_1.height * fg_scale)))
        scaled_card = fg_card_1.resize((cw, ch), resample=resample_mode)
        pan_px_x = int(round(fg_dx * width * 0.28))
        pan_px_y = int(round(fg_dy * height * 0.24))
        px = (width - cw) // 2 + pan_px_x
        py = int(height * card_cy_ratio) - ch // 2 + pan_px_y
        frame.paste(scaled_card, (px, max(10, py)), scaled_card)

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
    caption_subtitle: str = "Cinema-Grade 2.5D Parallax, Loupe, Triptych & Monograph Suite · 4K HDR",
    camera_badge: str = "FULL FRAME · 4K 10-BIT HLG HDR",
    tile_width: int = 960,
    tile_height: int = 540,
) -> Path:
    """
    Mechanically render a 5x2 high-resolution visual comparison showcase sheet of all 10 Photo Modes
    + Atmospheric Finishers so the AI Agent or user can inspect and select effects.
    """
    out_path = Path(output_jpg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sec_str = str(secondary_path) if (secondary_path and Path(secondary_path).exists()) else None
    tert_str = str(tertiary_path) if (tertiary_path and Path(tertiary_path).exists()) else None

    showcase_specs: list[tuple[str, str, str]] = [
        ("cinematic_depth_3d", "golden_light_leak", "1. 3D Depth Parallax + Golden Light Leak"),
        ("triptych_cinematic", "none", "2. 3-Panel Staggered Storyboard Triptych"),
        ("polaroid_scrapbook", "nostalgic_film_grain", "3. Vintage Polaroid Scrapbook + Film Grain"),
        ("curator_loupe", "museum_spotlight", "4. Art-Historian 1.65x Optical Loupe + Spotlight"),
        ("editorial_magazine", "none", "5. Asymmetric 62/38 Editorial Magazine Spread"),
        ("cinema_anamorphic", "champagne_bokeh", "6. 2.39:1 Anamorphic Widescreen + Champagne Bokeh"),
        ("parallax_gallery", "museum_spotlight", "7. 2.5D Museum Parallax + Spotlight"),
        ("diptych_frame", "golden_light_leak", "8. Side-by-Side Diptych + Golden Light Leak"),
        ("gallery_frame", "champagne_bokeh", "9. Classic Gallery Frame + Champagne Bokeh"),
        ("full_bleed", "nostalgic_film_grain", "10. Full-Bleed Ken Burns + 35mm Film Grain"),
    ]

    cols, rows = 2, 5
    pad = 20
    header_h = 68
    label_h = 38
    canvas_w = cols * tile_width + (cols + 1) * pad
    canvas_h = header_h + rows * (tile_height + label_h) + (rows + 1) * pad

    sheet = Image.new("RGB", (canvas_w, canvas_h), (10, 14, 22))
    draw = ImageDraw.Draw(sheet)
    f_hdr = _load_editorial_font(28, bold=True, serif=True)
    f_lbl = _load_editorial_font(18, bold=True, serif=False)

    draw.text(
        (pad, 18),
        "PREMIUM PHOTO EFFECTS SHOWCASE — 10 PRESENTATION MODES & 4 ATMOSPHERIC FINISHERS",
        font=f_hdr,
        fill=(234, 194, 96),
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

        draw.rectangle([x0 - 2, y0 - 2, x0 + tile_width + 1, y0 + tile_height + label_h + 1], outline=(214, 176, 84), width=2)
        sheet.paste(tile_img, (x0, y0))
        draw.rectangle([x0, y0 + tile_height, x0 + tile_width, y0 + tile_height + label_h], fill=(16, 22, 34))
        draw.text((x0 + 14, y0 + tile_height + 7), f"{label_text}  [mode={mode_name}, overlay_fx={fx_name}]", font=f_lbl, fill=(242, 245, 250))

    sheet.save(out_path, quality=92)
    return out_path
