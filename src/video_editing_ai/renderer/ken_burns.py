"""Sub-pixel Cinema & Exhibition Photo Animator + Composable Optical Finisher Pipeline."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from typing import Any, Literal, Optional

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

from ..hardware import (
    detect_best_video_encoder as _hw_detect_best_video_encoder,
    get_hardware_profile,
    has_ffmpeg_bsf,
    has_ffmpeg_filter as _hw_has_ffmpeg_filter,
    probe_hw_encoder as _hw_probe_hw_encoder,
    supports_videotoolbox_qscale as _hw_supports_videotoolbox_qscale,
)
from ..models.schema import (
    DISABLED_PHOTO_MODE_FALLBACKS,
    EffectTrackItem,
    KenBurnsConfig,
    TransitionConfig,
)
from .color_grading import apply_color_grade_to_pil_image, apply_color_grade_to_rgb
from .depth_parallax import (
    _load_depth_anything_model,
    build_mpi_depth_layers,
    build_rack_focus_assets,
    estimate_depth_map,
)


def _ease_in_out_cubic(t: float) -> float:
    """Smooth cubic S-curve easing for natural camera & parallax motion."""
    t_c = max(0.0, min(1.0, t))
    if t_c < 0.5:
        return 4.0 * t_c * t_c * t_c
    p = 2.0 * t_c - 2.0
    return 0.5 * p * p * p + 1.0


def _ease_in_out_quintic(t: float) -> float:
    """C2-continuous quintic smootherstep (zero velocity & zero acceleration at endpoints and midpoint)."""
    t_c = max(0.0, min(1.0, t))
    return t_c * t_c * t_c * (t_c * (t_c * 6.0 - 15.0) + 10.0)


def _ease_out_cubic(t: float) -> float:
    """Smooth cubic ease-out for documentary lower-third card glide."""
    t_c = max(0.0, min(1.0, t))
    p = 1.0 - t_c
    return 1.0 - p * p * p


def _contains_cjk(text: str) -> bool:
    """Return True when text contains CJK/East Asian ideographs or punctuation."""
    return any(
        "\u2e80" <= ch <= "\u9fff"
        or "\u3000" <= ch <= "\u303f"
        or "\uf900" <= ch <= "\ufaff"
        or "\uff00" <= ch <= "\uffef"
        for ch in (text or "")
    )


def _load_editorial_font(
    size: int,
    bold: bool = True,
    serif: bool = False,
    text: str = "",
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    has_cjk = _contains_cjk(text)
    if serif:
        cjk_serif = [
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        ]
        latin_serif = [
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
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        ]
        candidates = (cjk_serif + latin_serif) if has_cjk else latin_serif
    else:
        cjk_sans = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        ]
        latin_sans = [
            # Windows (Microsoft YaHei renders both crisp Segoe-style Latin & full CJK)
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            # macOS (2019 MacBook Pro & Apple Silicon — PingFang SC / STHeiti render both Latin & CJK)
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Avenir Next.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        candidates = (cjk_sans + latin_sans) if has_cjk else latin_sans
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

    framed = Image.new("RGBA", (fw, fh), (252, 252, 250, 255))
    f_draw = ImageDraw.Draw(framed)
    f_draw.rectangle(
        [b_px - 2, b_px - 2, fw - b_px + 1, fh - b_px + 1],
        outline=(224, 222, 216, 220),
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


def _paste_rgba_subpixel_pil(
    canvas_rgb: Image.Image,
    src_rgba: Image.Image,
    px: float,
    py: float,
    dst_w: Optional[float] = None,
    dst_h: Optional[float] = None,
    alpha_mul: float = 1.0,
) -> None:
    """Alpha-composite `src_rgba` onto `canvas_rgb` at floating-point `(px, py)` with sub-pixel resampling."""
    if alpha_mul <= 1e-4:
        return
    cw, ch = canvas_rgb.size
    sw, sh = src_rgba.size
    tw = float(sw) if dst_w is None else max(1.0, float(dst_w))
    th = float(sh) if dst_h is None else max(1.0, float(dst_h))

    ox0 = max(0, int(math.floor(px)))
    oy0 = max(0, int(math.floor(py)))
    ox1 = min(cw, int(math.ceil(px + tw)))
    oy1 = min(ch, int(math.ceil(py + th)))
    if ox1 <= ox0 or oy1 <= oy0:
        return

    sx0 = (ox0 - px) * (sw / tw)
    sy0 = (oy0 - py) * (sh / th)
    sx1 = (ox1 - px) * (sw / tw)
    sy1 = (oy1 - py) * (sh / th)

    is_int_aligned = (
        dst_w is None
        and dst_h is None
        and abs(px - round(px)) < 1e-3
        and abs(py - round(py)) < 1e-3
    )
    if is_int_aligned and ox0 >= 0 and oy0 >= 0:
        ix0, iy0 = int(round(sx0)), int(round(sy0))
        sub = src_rgba.crop((ix0, iy0, ix0 + (ox1 - ox0), iy0 + (oy1 - oy0)))
    else:
        sub = src_rgba.transform(
            (ox1 - ox0, oy1 - oy0),
            Image.Transform.EXTENT,
            (sx0, sy0, sx1, sy1),
            resample=Image.Resampling.BILINEAR,
        )

    alpha = sub.getchannel("A")
    if alpha_mul < 0.999:
        a_arr = np.clip(np.asarray(alpha, dtype=np.float32) * float(alpha_mul), 0.0, 255.0).astype(np.uint8)
        alpha = Image.fromarray(a_arr, mode="L")
    canvas_rgb.paste(sub.convert("RGB"), (ox0, oy0), alpha)


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
    badge_str = _clean_editorial_badge(camera_badge)
    f_badge = _load_editorial_font(int(round(13 * s)), bold=True, serif=False, text=badge_str)
    f_title = _load_editorial_font(int(round(26 * s)), bold=True, serif=True, text=caption_title)
    f_sub = _load_editorial_font(int(round(18 * s)), bold=False, serif=False, text=caption_subtitle)

    dummy = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    d_draw = ImageDraw.Draw(dummy)
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
    tint_weight: float = 0.58,
) -> None:
    """
    In-place composite of the smoked-glass documentary card onto `frame_rgb`:
    Crops only the small `(pw, ph)` region, blurs it at 1/4 scale (16x faster than full-res blur),
    tints it with translucent charcoal glass, layers the typography, and pastes back onto `frame_rgb`.
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

    # 2. Blend with neutral smoked charcoal glass
    frosted_rgb = Image.blend(blurred_patch, dark_tint_rgb, max(0.0, min(1.0, float(tint_weight))))

    # 3. Composite crisp typography & specular rim onto the frosted glass patch
    frosted_rgb.paste(text_fg_rgba, (0, 0), text_fg_rgba)

    # 4. Paste rounded frosted card back onto frame_rgb with smooth fade mask
    if alpha_mult >= 0.98:
        frame_rgb.paste(frosted_rgb, (dst_x, dst_y), card_mask_l)
    else:
        scaled_mask = card_mask_l.point(lambda p: int(p * alpha_mult))
        frame_rgb.paste(frosted_rgb, (dst_x, dst_y), scaled_mask)


_COLOR_GRADE_3DLUT_LOCK = threading.Lock()
_COLOR_GRADE_3DLUT_CACHE: dict[tuple[Any, ...], ImageFilter.Color3DLUT] = {}
_IDENTITY_LUT_CUBE_33: Optional[np.ndarray] = None

_BG_PLANE_MODES = frozenset(
    {
        "diptych_frame",
        "triptych_cinematic",
        "polaroid_scrapbook",
        "parallax_gallery",
        "glassmorphic_pedestal",
        "medium_format_slide",
        "floating_tilt_plate",
    }
)


def _normalize_grade_cache_key(color_grade_fx: EffectTrackItem | dict[str, Any]) -> Optional[tuple[Any, ...]]:
    """Build a deterministic hashable cache key for a photo color-grade spec."""
    params = (
        dict(getattr(color_grade_fx, "parameters"))
        if hasattr(color_grade_fx, "parameters") and isinstance(getattr(color_grade_fx, "parameters"), dict)
        else (dict(color_grade_fx) if isinstance(color_grade_fx, dict) else None)
    )
    if not params or params.get("bypass"):
        return None
    items: list[tuple[str, Any]] = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, float):
            items.append((k, round(v, 5)))
        elif isinstance(v, (list, tuple)):
            items.append((k, tuple(round(x, 5) if isinstance(x, float) else x for x in v)))
        elif isinstance(v, (str, int, bool, type(None))):
            items.append((k, v))
        else:
            items.append((k, str(v)))
    return tuple(items)


def _apply_color_grade_fast_pil(
    img: Image.Image,
    color_grade_fx: EffectTrackItem | dict[str, Any],
) -> Image.Image:
    """
    Apply full DI color grading (`apply_color_grade_to_rgb`) to an RGB PIL photo via a cached
    33x33x33 `ImageFilter.Color3DLUT` (~100x faster on 15-33MP stills, MAE ~0.05/255).
    """
    global _IDENTITY_LUT_CUBE_33
    if img.mode != "RGB" or (img.width * img.height) < 120_000:
        return apply_color_grade_to_pil_image(img, color_grade_fx)
    cache_key = _normalize_grade_cache_key(color_grade_fx)
    if cache_key is None:
        return img
    try:
        with _COLOR_GRADE_3DLUT_LOCK:
            lut = _COLOR_GRADE_3DLUT_CACHE.get(cache_key)
        if lut is None:
            if _IDENTITY_LUT_CUBE_33 is None:
                ax = np.linspace(0.0, 1.0, 33, dtype=np.float32)
                b_g, g_g, r_g = np.meshgrid(ax, ax, ax, indexing="ij")
                _IDENTITY_LUT_CUBE_33 = np.stack([r_g, g_g, b_g], axis=-1)
            graded_cube = apply_color_grade_to_rgb(_IDENTITY_LUT_CUBE_33, color_grade_fx)
            lut = ImageFilter.Color3DLUT(33, graded_cube.reshape(-1, 3).ravel().tolist())
            with _COLOR_GRADE_3DLUT_LOCK:
                if len(_COLOR_GRADE_3DLUT_CACHE) >= 64:
                    _COLOR_GRADE_3DLUT_CACHE.clear()
                _COLOR_GRADE_3DLUT_CACHE[cache_key] = lut
        return img.filter(lut)
    except Exception:
        return apply_color_grade_to_pil_image(img, color_grade_fx)


def _load_graded_photo_rgb(
    photo_path: str | Path,
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
    max_dim: Optional[int] = None,
) -> Image.Image:
    """Open an EXIF-oriented RGB photograph, optionally pre-capping oversized 24-60MP stills, and apply DI color grade."""
    with Image.open(photo_path) as raw_im:
        if max_dim is not None and max_dim >= 512:
            try:
                raw_im.draft("RGB", (max_dim, max_dim))
            except Exception:
                pass
        img = ImageOps.exif_transpose(raw_im).convert("RGB")
    if max_dim is not None and max_dim >= 512 and max(img.width, img.height) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    if color_grade_fx is not None:
        return _apply_color_grade_fast_pil(img, color_grade_fx)
    return img


def build_gallery_frame_image(
    src_path: str | Path,
    width: int = 1920,
    height: int = 1080,
    border_px: Optional[int] = None,
    raw_img: Optional[Image.Image] = None,
) -> Image.Image:
    """Create an uncropped double-matte gallery frame over a Gaussian-blurred ambient background."""
    img = raw_img.convert("RGB") if raw_img is not None else _load_graded_photo_rgb(src_path)

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


def build_links_white_matte_image(
    src_path: str | Path,
    width: int = 1920,
    height: int = 1080,
    border_px: Optional[int] = None,
    raw_img: Optional[Image.Image] = None,
    exhibition_tone: Literal["pure_white", "soft_limestone"] = "pure_white",
) -> Image.Image:
    """Create a gallery exhibition plate (`pure_white` or calm `soft_limestone` museum wall) with crisp margins."""
    img = raw_img.convert("RGB") if raw_img is not None else _load_graded_photo_rgb(src_path)

    scale_factor = width / 1920.0
    margin_ratio = (
        max(0.04, min(0.14, float(border_px) / 100.0))
        if (border_px is not None and 4 <= border_px <= 14)
        else 0.0
    )
    aspect = img.width / max(1, img.height)
    if margin_ratio > 0.0:
        max_w = int(round(width * (1.0 - 2.0 * margin_ratio)))
        max_h = int(round(height * (1.0 - 2.0 * margin_ratio)))
    elif aspect < 1.0:
        max_w, max_h = int(round(width * 0.72)), int(round(height * 0.84))
    else:
        max_w, max_h = int(round(width * 0.84)), int(round(height * 0.82))

    fg = img.copy()
    fg.thumbnail((max(32, max_w), max(32, max_h)), Image.Resampling.LANCZOS)
    px = (width - fg.width) // 2
    py = (height - fg.height) // 2

    if exhibition_tone == "soft_limestone":
        canvas = _make_textured_surface_rgb(
            width,
            height,
            (206, 203, 196),
            seed=202,
            grad_top=3.5,
            grad_bot=-4.5,
            grain_std=1.4,
        )
        sh_blur = max(5, int(round(9 * scale_factor)))
        sh_pad = sh_blur * 2 + 4
        sh_off_y = max(3, int(round(5 * scale_factor)))
        sh_img = Image.new("RGBA", (fg.width + sh_pad * 2, fg.height + sh_pad * 2), (0, 0, 0, 0))
        ImageDraw.Draw(sh_img).rectangle(
            [sh_pad, sh_pad + sh_off_y, sh_pad + fg.width, sh_pad + fg.height + sh_off_y],
            fill=(0, 0, 0, 82),
        )
        sh_img = sh_img.filter(ImageFilter.GaussianBlur(radius=sh_blur))
        canvas.paste(sh_img, (px - sh_pad, py - sh_pad), sh_img)
        canvas.paste(fg, (px, py))
        draw = ImageDraw.Draw(canvas)
        hair_w = max(1, int(round(1.0 * scale_factor)))
        draw.rectangle(
            [max(0, px - hair_w), max(0, py - hair_w), min(width - 1, px + fg.width), min(height - 1, py + fg.height)],
            outline=(160, 157, 150),
            width=hair_w,
        )
        return canvas

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(fg, (px, py))

    draw = ImageDraw.Draw(canvas)
    hair_w = max(1, int(round(1.0 * scale_factor)))
    draw.rectangle(
        [max(0, px - hair_w), max(0, py - hair_w), min(width - 1, px + fg.width), min(height - 1, py + fg.height)],
        outline=(230, 232, 236),
        width=hair_w,
    )
    return canvas


_SALIENT_FACE_LOCK = threading.Lock()
_SALIENT_FACE_CASCADE: Optional[Any] = None


def _norm_saliency_01(arr: np.ndarray, p_lo: float = 5.0, p_hi: float = 99.0) -> np.ndarray:
    """Robustly normalize a float saliency channel into [0.0, 1.0]."""
    lo = float(np.percentile(arr, p_lo))
    hi = float(np.percentile(arr, p_hi))
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _detect_photo_salient_point(
    raw_img: Image.Image,
    hint_uv: Optional[tuple[float, float]] = None,
    use_depth: bool = True,
) -> tuple[float, float]:
    """
    Locate the photograph's primary visual interest center `(u, v)` in normalized photo coordinates `[0..1]`.

    Priority order:
    1. Explicit non-default curator/director focal hint (`hint_uv`).
    2. Verified frontal human face detection (strict Haar cascade — no wood/stone skin-blob false positives).
    3. Multi-cue 3D + CV subject localization combining:
       - Shi-Tomasi 2D structural detail richness gating (suppresses flat sky/water/walls and 1D stripes),
       - CIELAB border-background contrast (isolates foreground objects from surrounding sky/water/wall),
       - FFT spectral residual saliency (suppresses repetitive periodic grids like fences/dentils),
       - Perspective-plane-removed Depth-Anything-V2 3D foreground prominence (when neural depth is available),
       - Connected-component subject centroid + upper-core head/wings/crest anchoring for standing 3D sculptures.
    """
    if hint_uv is not None:
        hx, hy = float(hint_uv[0]), float(hint_uv[1])
        is_default_start = abs(hx - 0.50) <= 0.015 and abs(hy - 0.50) <= 0.015
        is_default_end = abs(hx - 0.50) <= 0.015 and abs(hy - 0.48) <= 0.015
        if not (is_default_start or is_default_end):
            return (max(0.16, min(0.84, hx)), max(0.16, min(0.82, hy)))

    thumb = raw_img.copy()
    thumb.thumbnail((480, 480), Image.Resampling.BILINEAR)
    tw, th = thumb.size
    if tw < 16 or th < 16:
        return (0.50, 0.45)

    arr_rgb = np.asarray(thumb.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1. Strict verified frontal human face detection (never falls back to tan wood/stone skin blobs)
    global _SALIENT_FACE_CASCADE
    try:
        with _SALIENT_FACE_LOCK:
            if _SALIENT_FACE_CASCADE is None:
                xml_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
                _SALIENT_FACE_CASCADE = cv2.CascadeClassifier(xml_path)
            if _SALIENT_FACE_CASCADE is not None and not _SALIENT_FACE_CASCADE.empty():
                min_f = max(28, int(round(min(tw, th) * 0.07)))
                max_f = max(min_f + 4, int(round(min(tw, th) * 0.42)))
                eq_gray = cv2.equalizeHist(gray)
                faces = _SALIENT_FACE_CASCADE.detectMultiScale(
                    eq_gray,
                    scaleFactor=1.12,
                    minNeighbors=7,
                    minSize=(min_f, min_f),
                    maxSize=(max_f, max_f),
                )
            else:
                faces = ()
        if len(faces) > 0:
            best_face = max(
                faces,
                key=lambda b: float(b[2] * b[3])
                * max(
                    0.35,
                    1.15 - 0.50 * math.hypot((b[0] + 0.5 * b[2]) / tw - 0.5, (b[1] + 0.42 * b[3]) / th - 0.46),
                ),
            )
            fx, fy, fw, fh = [int(val) for val in best_face]
            if float(np.std(gray[fy : fy + fh, fx : fx + fw])) >= 18.0:
                u = max(0.20, min(0.80, (fx + 0.50 * fw) / float(tw)))
                v = max(0.20, min(0.76, (fy + 0.44 * fh) / float(th)))
                return (float(u), float(v))
    except Exception:
        pass

    # 2. Local 2D sculptural/ornamental detail richness (Shi-Tomasi min-eigenvalue crushes flat sky/water & 1D stripes)
    smooth_g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.8)
    gx = cv2.Sobel(smooth_g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth_g, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)
    k_det = max(15, int(round(min(tw, th) * 0.06)) | 1)
    ixx = cv2.GaussianBlur(gx * gx, (k_det, k_det), 0)
    iyy = cv2.GaussianBlur(gy * gy, (k_det, k_det), 0)
    ixy = cv2.GaussianBlur(gx * gy, (k_det, k_det), 0)
    tr = ixx + iyy
    det = np.maximum(0.0, (ixx - iyy) ** 2 + 4.0 * ixy * ixy)
    lam_min = np.maximum(0.0, 0.5 * (tr - np.sqrt(det)))
    detail_2d = _norm_saliency_01(np.sqrt(lam_min + 1e-6), 10.0, 98.5)
    detail_grad = _norm_saliency_01(cv2.GaussianBlur(grad_mag, (k_det, k_det), 0), 10.0, 98.5)
    detail_rich = np.clip(0.65 * detail_2d + 0.35 * detail_grad, 0.0, 1.0)
    detail_gate = np.clip(0.12 + 0.88 * np.power(detail_rich, 0.70), 0.12, 1.0)

    # 3. Border-background contrast in CIELAB (separates foreground subjects from sky/water/wall surroundings)
    lab = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bw = max(4, int(round(min(tw, th) * 0.06)))
    bg_refs = [
        np.median(lab[:bw, :, :].reshape(-1, 3), axis=0),
        np.median(lab[-bw:, :, :].reshape(-1, 3), axis=0),
        np.median(lab[:, :bw, :].reshape(-1, 3), axis=0),
        np.median(lab[:, -bw:, :].reshape(-1, 3), axis=0),
    ]
    dists = [
        np.sqrt(
            0.55 * (lab[..., 0] - ref[0]) ** 2
            + 1.35 * (lab[..., 1] - ref[1]) ** 2
            + 1.35 * (lab[..., 2] - ref[2]) ** 2
        )
        for ref in bg_refs
    ]
    d_stack = np.sort(np.stack(dists, axis=-1), axis=-1)
    border_contrast = _norm_saliency_01(0.60 * d_stack[..., 0] + 0.40 * d_stack[..., 1])
    border_contrast = cv2.GaussianBlur(border_contrast, (21, 21), 0)

    # 4. Spectral Residual Saliency (suppresses repetitive periodic textures like fences, cornice dentils, ripples)
    s_w = 96
    s_h = max(48, int(round(96 * th / max(1, tw))))
    spec_accum = np.zeros((s_h, s_w), dtype=np.float32)
    for ch in range(3):
        ch_small = cv2.resize(lab[..., ch], (s_w, s_h), interpolation=cv2.INTER_AREA).astype(np.float32)
        fft = np.fft.fft2(ch_small)
        log_amp = np.log(np.abs(fft) + 1e-6)
        phase = np.angle(fft)
        spectral_res = log_amp - cv2.blur(log_amp, (5, 5))
        inv = np.fft.ifft2(np.exp(spectral_res + 1j * phase))
        mag2 = np.abs(inv) ** 2
        spec_accum += _norm_saliency_01(cv2.GaussianBlur(mag2.astype(np.float32), (9, 9), 2.5))
    spectral_sal = _norm_saliency_01(cv2.resize(spec_accum, (tw, th), interpolation=cv2.INTER_LINEAR))
    spectral_sal = cv2.GaussianBlur(spectral_sal, (25, 25), 0)

    # 5. Perspective-plane-removed 3D foreground prominence (only when real Depth-Anything-V2 is available)
    depth_prom = np.zeros((th, tw), dtype=np.float32)
    has_isolated_3d_subject = False
    if use_depth:
        try:
            if _load_depth_anything_model() is not None:
                d_map = estimate_depth_map(thumb, focal_point=(0.5, 0.48), target_size=(tw, th))
                if d_map is not None and d_map.shape[:2] == (th, tw):
                    d_f = d_map.astype(np.float32)
                    edge_w = max(6, int(round(tw * 0.08)))
                    left_d = np.median(d_f[:, :edge_w], axis=1, keepdims=True)
                    right_d = np.median(d_f[:, -edge_w:], axis=1, keepdims=True)
                    tx = np.linspace(0.0, 1.0, tw, dtype=np.float32)[None, :]
                    row_plane = left_d * (1.0 - tx) + right_d * tx
                    prom = np.maximum(0.0, d_f - row_plane - 0.05)
                    yy_d = np.linspace(0.0, 1.0, th, dtype=np.float32)[:, None]
                    prom = prom * np.where(yy_d > 0.78, np.clip((1.0 - yy_d) / 0.22, 0.15, 1.0), 1.0)
                    p98 = float(np.percentile(prom, 98.5))
                    if p98 >= 0.12:
                        strength = min(1.0, (p98 - 0.10) / 0.25)
                        depth_prom = _norm_saliency_01(cv2.GaussianBlur(prom, (21, 21), 0)) * strength
                        has_isolated_3d_subject = p98 >= 0.18
        except Exception:
            pass

    # 6. Photography center-composition & frame-safety prior
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    nx = xx / max(1.0, float(tw - 1))
    ny = yy / max(1.0, float(th - 1))
    center_prior = np.exp(-(((nx - 0.50) / 0.34) ** 2 + ((ny - 0.47) / 0.30) ** 2))
    edge_safe = (
        np.clip(np.minimum(nx, 1.0 - nx) / 0.16, 0.05, 1.0)
        * np.clip(np.minimum(ny, 1.0 - ny) / 0.16, 0.05, 1.0)
    )
    ground_damp = np.where(ny > 0.78, np.clip(1.0 - (ny - 0.78) * 2.5, 0.20, 1.0), 1.0)

    d_max = float(depth_prom.max())
    w_d = 0.34 * d_max
    rem = 1.0 - w_d
    subject_sal = w_d * depth_prom + rem * (0.42 * border_contrast + 0.34 * spectral_sal + 0.24 * detail_2d)

    score_map = subject_sal * detail_gate * (0.40 + 0.60 * center_prior) * edge_safe * ground_damp
    k_big = max(21, int(round(min(tw, th) * 0.12)) | 1)
    score_smooth = cv2.GaussianBlur(score_map, (k_big, k_big), 0)

    peak_val = float(score_smooth.max())
    if peak_val < 1e-5:
        return (0.50, 0.45)

    # 7. Connected-component dominant subject blob + upper-core head/wings anchoring
    peak_y, peak_x = np.unravel_index(int(np.argmax(score_smooth)), score_smooth.shape)
    mask = (score_smooth >= peak_val * 0.62).astype(np.uint8)
    _, labels = cv2.connectedComponents(mask)
    comp = labels == int(labels[peak_y, peak_x])
    ys, xs = np.where(comp)
    w_comp = score_smooth[ys, xs]
    cx = float(np.sum(xs * w_comp) / max(1e-6, float(np.sum(w_comp)))) / max(1.0, float(tw - 1))
    cy = float(np.sum(ys * w_comp) / max(1e-6, float(np.sum(w_comp)))) / max(1.0, float(th - 1))

    bx0, bx1 = float(xs.min()) / max(1.0, float(tw - 1)), float(xs.max()) / max(1.0, float(tw - 1))
    by0, by1 = float(ys.min()) / max(1.0, float(th - 1)), float(ys.max()) / max(1.0, float(th - 1))
    bh = by1 - by0

    local_d = float(depth_prom[int(round(cy * (th - 1))), int(round(cx * (tw - 1)))])
    if has_isolated_3d_subject and local_d >= 0.30 and bh >= 0.12:
        upper_zone_y0 = max(0, int(round((by0 - 0.18) * th)))
        upper_zone_y1 = int(round(cy * th))
        col_x0 = max(0, int(round(bx0 * tw)))
        col_x1 = min(tw, int(round(bx1 * tw)))
        if upper_zone_y1 > upper_zone_y0 + 4 and col_x1 > col_x0 + 4:
            upper_sig = (
                0.65 * depth_prom[upper_zone_y0:upper_zone_y1, col_x0:col_x1]
                + 0.35 * border_contrast[upper_zone_y0:upper_zone_y1, col_x0:col_x1]
            ) * detail_gate[upper_zone_y0:upper_zone_y1, col_x0:col_x1]
            row_sig = upper_sig.max(axis=1)
            active_rows = np.where(row_sig >= max(0.28, float(row_sig.max()) * 0.44))[0]
            if len(active_rows) > 0:
                subj_top = (upper_zone_y0 + int(active_rows[0])) / max(1.0, float(th - 1))
                subj_bot = by1
                ideal_y = subj_top + 0.36 * (subj_bot - subj_top)
                top_band_y0 = int(round(subj_top * th))
                top_band_y1 = min(th, int(round((subj_top + 0.55 * (subj_bot - subj_top)) * th)))
                if top_band_y1 > top_band_y0 + 3:
                    band_w = (
                        depth_prom[top_band_y0:top_band_y1, col_x0:col_x1]
                        * detail_gate[top_band_y0:top_band_y1, col_x0:col_x1]
                    )
                    if float(band_w.sum()) > 1e-4:
                        xs_band = np.arange(col_x0, col_x1, dtype=np.float32)[None, :]
                        ideal_x = float((xs_band * band_w).sum() / band_w.sum()) / max(1.0, float(tw - 1))
                        cx = 0.45 * cx + 0.55 * ideal_x
                cy = 0.30 * cy + 0.70 * ideal_y

    u = max(0.20, min(0.80, cx))
    v = max(0.20, min(0.76, cy))
    return (float(u), float(v))


def _resolve_kb_endpoints(
    kb: KenBurnsConfig,
) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    """Resolve preset names into concrete (start_zoom, end_zoom, start_center, end_center)."""
    preset = kb.preset
    if preset == "static":
        return (1.0, 1.0, (0.5, 0.5), (0.5, 0.5))
    if preset == "floating_gallery":
        if kb.end_center != (0.5, 0.48) or kb.start_center != (0.5, 0.5):
            return (1.0, max(1.05, kb.end_zoom), kb.start_center, kb.end_center)
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


def _supports_videotoolbox_qscale(encoder: str, pix_fmt: str) -> bool:
    """Check whether VideoToolbox accepts -q:v alongside -b:v (Apple Silicon supports it; Intel x86_64 rejects it)."""
    return _hw_supports_videotoolbox_qscale(encoder, pix_fmt)


def has_ffmpeg_filter(filter_name: str) -> bool:
    """Check once whether a specific FFmpeg filter (e.g. 'zscale') is compiled into the local ffmpeg binary."""
    return _hw_has_ffmpeg_filter(filter_name)


def _probe_hw_encoder(encoder: str, pix_fmt: str = "yuv420p", extra_args: Optional[list[str]] = None) -> bool:
    """Test whether a specific hardware encoder is functional on the current machine/GPU."""
    return _hw_probe_hw_encoder(encoder, pix_fmt=pix_fmt, extra_args=extra_args)


def detect_best_video_encoder(hdr: bool = True) -> str:
    """
    Auto-detect the fastest working hardware video encoder across platforms:
    - macOS (2019 MacBook Pro AMD Radeon Pro + Apple T2 / Apple Silicon): hevc_videotoolbox / h264_videotoolbox
    - Windows/Linux NVIDIA GPU: hevc_nvenc / h264_nvenc
    - Windows/BootCamp AMD Radeon GPU: hevc_amf / h264_amf
    - Intel QuickSync iGPU: hevc_qsv / h264_qsv
    - Software fallback: libx265 / libx264
    """
    return _hw_detect_best_video_encoder(hdr=hdr)


def has_hevc_nvenc() -> bool:
    """Backward-compatible helper returning True when hevc_nvenc is the active 10-bit encoder."""
    return detect_best_video_encoder(hdr=True) == "hevc_nvenc"


def build_hdr_color_filters(
    src_trc: str = "iec61966-2-1",
    src_primaries: str = "bt709",
    src_space: str = "bt709",
    is_log: bool = False,
    diffuse_white_nits: float = 380.0,
    use_zscale: Optional[bool] = None,
) -> list[str]:
    """
    Build 10-bit BT.2020 HLG HDR color-space filters:
    - Native HLG (arib-std-b67): passthrough to p010le with BT.2020 HLG VUI tags.
    - SDR / sRGB / Log -> HLG with zscale: maps SDR peak white to HLG diffuse white (~380 nits / ~85% HLG signal).
    - Fallback without zscale: scales SDR signal ceiling cleanly prior to setparams.
    """
    filters: list[str] = []
    trc_norm = (src_trc or "709").strip().lower()

    if is_log:
        filters.append("eq=contrast=1.18:saturation=1.14:gamma=0.92")

    if trc_norm == "arib-std-b67" and not is_log:
        filters.extend(
            [
                "format=p010le",
                "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc:range=tv",
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
    npl_val = max(100, min(1000, int(round(diffuse_white_nits))))

    if zscale_ok:
        if z_trc_in == "smpte2084":
            filters.append(
                "zscale=primariesin=2020:transferin=smpte2084:matrixin=2020_ncl:"
                "primaries=2020:transfer=arib-std-b67:matrix=2020_ncl:range=limited"
            )
        else:
            filters.append(
                f"zscale=primariesin={z_prim_in}:transferin={z_trc_in}:matrixin={z_mat_in}:"
                f"primaries=2020:transfer=arib-std-b67:matrix=2020_ncl:range=limited:npl={npl_val}"
            )
    else:
        romax = 0.75 if npl_val <= 220 else min(0.98, round(0.75 + (npl_val - 203.0) / 677.0 * 0.23, 2))
        filters.extend(
            [
                f"colorlevels=romax={romax:.2f}:gomax={romax:.2f}:bomax={romax:.2f}",
                "eq=gamma=1.08:contrast=0.98" if romax > 0.85 else "eq=gamma=1.14:contrast=0.96",
            ]
        )

    filters.extend(
        [
            "format=p010le",
            "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc:range=tv",
        ]
    )
    return filters


def build_rgba_hdr_overlay_filter(
    in_label: str,
    out_label: str,
    tag: str,
    fade_chain: str = "",
    pts_chain: str = "",
    diffuse_white_nits: float = 420.0,
) -> str:
    """
    Convert an sRGB RGBA overlay stream to 10-bit BT.2020 HLG (yuva444p10le) while preserving straight alpha.
    Prevents white titles, badges, and paired-photo frames from dimming when composited onto HLG base timelines.
    """
    npl_val = max(100, min(1000, int(round(diffuse_white_nits))))
    head_parts = ["format=rgba"]
    if fade_chain:
        head_parts.append(fade_chain)
    head = ",".join(head_parts)
    if not has_ffmpeg_filter("zscale"):
        tail = f",{pts_chain}" if pts_chain else ""
        return f"{in_label}{head}{tail}{out_label}"

    tail = f",{pts_chain}" if pts_chain else ""
    return (
        f"{in_label}{head},split=2[{tag}_rgb][{tag}_a];"
        f"[{tag}_rgb]zscale=primariesin=709:transferin=iec61966-2-1:matrixin=709:"
        f"primaries=2020:transfer=arib-std-b67:matrix=2020_ncl:range=limited:npl={npl_val},format=yuv444p10le[{tag}_hlg];"
        f"[{tag}_a]alphaextract,format=gray10le[{tag}_a10];"
        f"[{tag}_hlg][{tag}_a10]alphamerge{tail}{out_label}"
    )


def get_video_encoder_args(hdr: bool = True, force_software: bool = False) -> list[str]:
    """Return FFmpeg video encoder CLI arguments for the best available hardware or software encoder."""
    enc = ("libx265" if hdr else "libx264") if force_software else detect_best_video_encoder(hdr=hdr)
    if hdr:
        hdr_tags = [
            "-color_range",
            "tv",
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
            args = [
                "-c:v",
                "hevc_videotoolbox",
                "-profile:v",
                "main10",
                "-pix_fmt",
                "p010le",
            ]
            if _supports_videotoolbox_qscale("hevc_videotoolbox", "p010le"):
                args.extend(["-q:v", "68"])
            args.extend([
                "-b:v",
                "28M",
                "-allow_sw",
                "1",
                *hdr_tags,
            ])
            return args
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
                "18",
                "-spatial_aq",
                "1",
                "-aq-strength",
                "12",
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
        args = [
            "-c:v",
            "h264_videotoolbox",
        ]
        if _supports_videotoolbox_qscale("h264_videotoolbox", "yuv420p"):
            args.extend(["-q:v", "68"])
        args.extend([
            "-b:v",
            "16M",
            "-allow_sw",
            "1",
            "-pix_fmt",
            "yuv420p",
        ])
        return args
    if enc == "h264_nvenc":
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-cq",
            "18",
            "-spatial_aq",
            "1",
            "-aq-strength",
            "12",
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
    cache_dir: Optional[Path] = None,
    photo_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build neural-depth 4-layer Multi-Plane Image (MPI) depth strata for 'cinematic_depth_3d'
    and 'dolly_zoom_vertigo' with dilated disocclusion fill and natural far-plane optical softness.
    """
    bg_sharp = ImageOps.fit(raw_img, (bg_w, bg_h), method=Image.Resampling.BILINEAR)
    depth_map = estimate_depth_map(
        bg_sharp,
        focal_point=focal_point,
        cache_dir=cache_dir,
        photo_key=photo_key,
        target_size=(bg_w, bg_h),
    )
    return build_mpi_depth_layers(bg_sharp=bg_sharp, depth=depth_map, scale_factor=scale_factor)


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
        short_stamp = stamp_text.split("—")[0].split("·")[0].strip()[:42]
        f_stamp = _load_editorial_font(int(round(17 * scale_factor)), bold=True, serif=True, text=short_stamp)
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
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
) -> list[Image.Image]:
    """Build 3 coordinated vertical gallery panels for 'triptych_cinematic'."""
    panel_w = int(width * 0.265)
    panel_h = int(height * (0.70 if has_card else 0.78))
    border_px = max(8, int(round(11 * scale_factor)))

    sec_max_dim = max(512, (max(width, height) * 5) // 4)
    sec_img: Optional[Image.Image] = None
    tert_img: Optional[Image.Image] = None
    if kb.secondary_source and Path(kb.secondary_source).exists():
        sec_img = _load_graded_photo_rgb(kb.secondary_source, color_grade_fx, max_dim=sec_max_dim)
    if kb.tertiary_source and Path(kb.tertiary_source).exists():
        tert_img = _load_graded_photo_rgb(kb.tertiary_source, color_grade_fx, max_dim=sec_max_dim)

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
    focal_uv: tuple[float, float] = (0.50, 0.45),
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, int, tuple[float, float], tuple[float, float]]:
    """
    Pre-build assets for 'curator_loupe' (Art-Historian 1.65x Optical Magnifying Glass Spotlight):
    Maps `focal_uv` directly into the uncropped framed photograph coordinates so magnification stays locked
    on the actual subject/face rather than margins or empty floor.
    Returns (dimmed_canvas_rgb, sharp_canvas_rgb, loupe_mask_l, loupe_bezel_rgba, radius_px, start_xy, end_xy).
    """
    bg_small = ImageOps.fit(raw_primary, (max(32, width // 4), max(18, height // 4)), method=Image.Resampling.BILINEAR)
    bg_small = bg_small.filter(ImageFilter.GaussianBlur(radius=max(6, int(round(9 * scale_factor)))))
    sharp_canvas = bg_small.resize((width, height), Image.Resampling.BILINEAR)
    sharp_canvas = Image.blend(sharp_canvas, Image.new("RGB", (width, height), (8, 10, 18)), 0.35)

    max_w = int(width * 0.84)
    max_h = int(height * (0.76 if has_card else 0.84))
    b_px = max(10, int(round(14 * scale_factor)))
    bevel_pad = max(2, int(round(3 * scale_factor)))
    shadow_margin = max(18, int(round(28 * scale_factor)))
    pad_total = shadow_margin + b_px + bevel_pad

    card = _build_double_matte_card(raw_primary, max_w=max_w, max_h=max_h, scale_factor=scale_factor, border_px=b_px)
    cy = int(height * (0.43 if has_card else 0.48))
    card_x = (width - card.width) // 2
    card_y = max(10, cy - card.height // 2)
    sharp_canvas.paste(card, (card_x, card_y), card)

    img_x0 = card_x + pad_total
    img_y0 = card_y + pad_total
    img_w = max(32, card.width - pad_total * 2)
    img_h = max(32, card.height - pad_total * 2)

    dimmed_canvas = Image.blend(sharp_canvas, Image.new("RGB", (width, height), (6, 8, 14)), 0.36)

    # Adaptive lens radius: sleek ~20% on landscape prints, proportionally bounded on portrait prints
    radius_px = min(
        int(round(min(width, height) * 0.20)),
        max(36, int(round(min(img_w, img_h) * 0.36))),
    )
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

    mag = 1.65
    src_half = radius_px / mag
    fu, fv = float(focal_uv[0]), float(focal_uv[1])
    target_cx = img_x0 + fu * img_w
    target_cy = img_y0 + fv * img_h

    # Keep 100% of magnified crop AND >=88% of optical bezel gracefully inside the framed photograph
    safe_margin_x = max(src_half + 4.0, radius_px * 0.88)
    safe_margin_y = max(src_half + 4.0, radius_px * 0.88)

    if img_w > 2.0 * safe_margin_x:
        min_cx = img_x0 + safe_margin_x
        max_cx = img_x0 + img_w - safe_margin_x
    elif img_w > 2.0 * src_half:
        min_cx = img_x0 + src_half
        max_cx = img_x0 + img_w - src_half
    else:
        min_cx = max_cx = img_x0 + 0.5 * img_w
    min_cx = max(float(radius_px) + 6.0, min_cx)
    max_cx = min(float(width) - float(radius_px) - 6.0, max_cx)
    if min_cx > max_cx:
        min_cx = max_cx = img_x0 + 0.5 * img_w

    max_plaque_y = height * (0.78 if has_card else 0.90) - radius_px * 0.85
    if img_h > 2.0 * safe_margin_y:
        min_cy = img_y0 + safe_margin_y
        max_cy = min(img_y0 + img_h - safe_margin_y, max_plaque_y)
    elif img_h > 2.0 * src_half:
        min_cy = img_y0 + src_half
        max_cy = min(img_y0 + img_h - src_half, max_plaque_y)
    else:
        min_cy = max_cy = min(img_y0 + 0.5 * img_h, max_plaque_y)
    min_cy = max(float(radius_px) + 6.0, min_cy)
    if min_cy > max_cy:
        min_cy = max_cy = max(float(radius_px) + 6.0, min(float(height) - float(radius_px) - 6.0, img_y0 + 0.5 * img_h))

    drift_x = min(22.0 * scale_factor, img_w * 0.045)
    drift_y = min(10.0 * scale_factor, img_h * 0.025)
    start_xy = (
        float(min(max_cx, max(min_cx, target_cx - drift_x))),
        float(min(max_cy, max(min_cy, target_cy + drift_y))),
    )
    end_xy = (
        float(min(max_cx, max(min_cx, target_cx + drift_x))),
        float(min(max_cy, max(min_cy, target_cy - drift_y))),
    )
    return dimmed_canvas, sharp_canvas, loupe_mask_l, bezel, radius_px, start_xy, end_xy


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
        f_badge = _load_editorial_font(int(round(14 * scale_factor)), bold=True, serif=False, text=badge_str)
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
    f_title = _load_editorial_font(int(round(32 * scale_factor)), bold=True, serif=True, text=title_str)
    for line in textwrap.wrap(title_str, width=22)[:4]:
        draw.text((pad_x, cur_y), line, font=f_title, fill=(248, 246, 240))
        cur_y += max(14, int(round(44 * scale_factor)))

    cur_y += max(6, int(round(16 * scale_factor)))

    # 3. Multi-line Curatorial Essay / Bilingual Subtitle in Soft Pearl Stone
    sub_str = (kb.caption_subtitle or "").strip()
    if sub_str:
        f_sub = _load_editorial_font(int(round(20 * scale_factor)), bold=False, serif=False, text=sub_str)
        for line in textwrap.wrap(sub_str, width=32)[:6]:
            draw.text((pad_x, cur_y), line, font=f_sub, fill=(192, 198, 208))
            cur_y += max(12, int(round(30 * scale_factor)))

    # Bottom monograph folio stamp (only rendered if a clean location/folio badge is present)
    if badge_str:
        f_folio = _load_editorial_font(int(round(13 * scale_factor)), bold=False, serif=True, text=badge_str)
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
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
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

    sec_max_dim = max(512, (max(width, height) * 5) // 4)
    sec_img: Optional[Image.Image] = None
    tert_img: Optional[Image.Image] = None
    if kb.secondary_source and Path(kb.secondary_source).exists():
        sec_img = _load_graded_photo_rgb(kb.secondary_source, color_grade_fx, max_dim=sec_max_dim)
    if kb.tertiary_source and Path(kb.tertiary_source).exists():
        tert_img = _load_graded_photo_rgb(kb.tertiary_source, color_grade_fx, max_dim=sec_max_dim)

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
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
    focal_uv: Optional[tuple[float, float]] = None,
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

    sec_max_dim = max(512, (max(width, height) * 5) // 4)
    sec_img: Optional[Image.Image] = None
    tert_img: Optional[Image.Image] = None
    if kb.secondary_source and Path(kb.secondary_source).exists():
        sec_img = _load_graded_photo_rgb(kb.secondary_source, color_grade_fx, max_dim=sec_max_dim)
    if kb.tertiary_source and Path(kb.tertiary_source).exists():
        tert_img = _load_graded_photo_rgb(kb.tertiary_source, color_grade_fx, max_dim=sec_max_dim)

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
        # Tight macro crop around primary salient focal point
        sw, sh = raw_primary.size
        fx, fy = focal_uv if focal_uv is not None else kb.start_center
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


_SURFACE_PALETTE_SPECS: dict[str, dict[str, Any]] = {
    "warm_linen": {
        "name": "warm_linen",
        "bg": (200, 193, 182),
        "mat": (205, 198, 187),
        "ink_primary": (32, 30, 28),
        "ink_secondary": (78, 74, 66),
        "ink_muted": (110, 104, 96),
        "rule": (140, 133, 122),
        "bevel_hi": (220, 215, 205),
        "bevel_lo": (150, 143, 132),
        "is_light": True,
    },
    "gallery_white": {
        "name": "gallery_white",
        "bg": (210, 208, 202),
        "mat": (214, 212, 206),
        "ink_primary": (26, 28, 32),
        "ink_secondary": (74, 78, 86),
        "ink_muted": (108, 112, 120),
        "rule": (152, 149, 142),
        "bevel_hi": (228, 226, 220),
        "bevel_lo": (164, 161, 154),
        "is_light": True,
    },
    "obsidian_slate": {
        "name": "obsidian_slate",
        "bg": (14, 16, 21),
        "mat": (20, 23, 30),
        "ink_primary": (238, 234, 226),
        "ink_secondary": (182, 186, 196),
        "ink_muted": (124, 130, 142),
        "rule": (64, 68, 78),
        "bevel_hi": (56, 60, 70),
        "bevel_lo": (8, 10, 14),
        "is_light": False,
    },
}


def _resolve_surface_palette(raw_primary: Image.Image, kb: KenBurnsConfig) -> dict[str, Any]:
    """Select an HDR-safe museum paper or obsidian surface palette suited to the photograph."""
    req = str(getattr(kb, "surface_palette", "auto") or "auto")
    if req in _SURFACE_PALETTE_SPECS:
        return dict(_SURFACE_PALETTE_SPECS[req])

    sample = raw_primary.resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.asarray(sample, dtype=np.float32)
    mean_luma = float((0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).mean())
    warmth = float((arr[:, :, 0] - arr[:, :, 2]).mean())
    if mean_luma < 90.0:
        return dict(_SURFACE_PALETTE_SPECS["obsidian_slate"])
    if warmth >= 10.0:
        return dict(_SURFACE_PALETTE_SPECS["warm_linen"])
    return dict(_SURFACE_PALETTE_SPECS["gallery_white"])


def _wrap_editorial_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_px_w: int,
    max_lines: int = 4,
) -> list[str]:
    """Wrap Latin and CJK strings cleanly by measured pixel width."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    lines: list[str] = []
    for para in cleaned.splitlines():
        para = para.strip()
        if not para:
            continue
        cur = ""
        tokens = para.split(" ") if (" " in para and not _contains_cjk(para)) else list(para)
        joiner = " " if (" " in para and not _contains_cjk(para)) else ""
        for tok in tokens:
            cand = f"{cur}{joiner}{tok}" if cur else tok
            bw = draw.textbbox((0, 0), cand, font=font)[2]
            if bw <= max_px_w or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = tok
                if len(lines) >= max_lines:
                    return lines[:max_lines]
        if cur:
            lines.append(cur)
        if len(lines) >= max_lines:
            break
    return lines[:max_lines]


_SURFACE_BOARD_CACHE_LOCK = threading.Lock()
_SURFACE_BOARD_CACHE: dict[tuple[Any, ...], Image.Image] = {}


def _make_textured_surface_rgb(
    width: int,
    height: int,
    base_rgb: tuple[int, int, int],
    seed: int = 301,
    grad_top: float = 4.0,
    grad_bot: float = -5.0,
    grain_std: float = 1.8,
) -> Image.Image:
    """Create a deterministic museum surface board with gentle gallery top-light wash and radial corner falloff."""
    key = (int(width), int(height), tuple(int(c) for c in base_rgb), int(seed), round(float(grad_top), 3), round(float(grad_bot), 3), round(float(grain_std), 3))
    with _SURFACE_BOARD_CACHE_LOCK:
        cached = _SURFACE_BOARD_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    sw, sh = max(32, width // 3), max(32, height // 3)
    ny = np.linspace(0.0, 1.0, sh, dtype=np.float32).reshape(sh, 1)
    nx_2d = np.linspace(-1.0, 1.0, sw, dtype=np.float32).reshape(1, sw)
    ny_2d = np.linspace(-1.0, 1.0, sh, dtype=np.float32).reshape(sh, 1)
    vig = (-9.5 * (0.55 * (nx_2d * nx_2d) + 0.45 * (ny_2d * ny_2d))).reshape(sh, sw, 1)
    rng = np.random.default_rng(seed=seed)
    grain = rng.normal(loc=0.0, scale=grain_std, size=(sh, sw, 1)).astype(np.float32)
    wash = (grad_top * (1.0 - ny) + grad_bot * ny).reshape(sh, 1, 1)
    arr = np.empty((sh, sw, 3), dtype=np.float32)
    arr[:, :, 0] = base_rgb[0] + wash[:, :, 0] + vig[:, :, 0] + grain[:, :, 0]
    arr[:, :, 1] = base_rgb[1] + wash[:, :, 0] + vig[:, :, 0] + grain[:, :, 0]
    arr[:, :, 2] = base_rgb[2] + wash[:, :, 0] + vig[:, :, 0] + grain[:, :, 0]
    u8 = np.clip(arr, 4.0, 244.0).astype(np.uint8)
    out = Image.fromarray(u8, mode="RGB").resize((width, height), Image.Resampling.BILINEAR)
    with _SURFACE_BOARD_CACHE_LOCK:
        if len(_SURFACE_BOARD_CACHE) >= 16:
            _SURFACE_BOARD_CACHE.clear()
        _SURFACE_BOARD_CACHE[key] = out
    return out.copy()


def _build_passepartout_shadowbox_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
    pal: dict[str, Any],
) -> dict[str, Any]:
    """
    Build cotton-rag/obsidian museum passe-partout board with a 45-degree mat bevel,
    soft recessed inner shadow, and clean margin typography (zero blurry photo backdrop).
    """
    ow = int(round(width * 1.03))
    ow -= ow % 2
    oh = int(round(height * 1.03))
    oh -= oh % 2

    board = _make_textured_surface_rgb(ow, oh, pal["mat"], seed=301, grad_top=4.0, grad_bot=-5.0, grain_std=1.8)
    draw = ImageDraw.Draw(board)

    has_cap = bool(kb.caption_title and kb.caption_title.strip())
    aspect = raw_primary.width / max(1, raw_primary.height)
    max_ww = int(ow * 0.74)
    max_wh = int(oh * (0.67 if has_cap else 0.75))
    if max_ww / max(1.0, float(max_wh)) >= aspect:
        wh = max_wh
        ww = int(round(wh * aspect))
    else:
        ww = max_ww
        wh = int(round(ww / max(0.25, aspect)))
    ww = max(int(ow * 0.34), min(max_ww, ww))
    wh = max(int(oh * 0.44), min(max_wh, wh))

    wx = (ow - ww) // 2
    wy = max(int(oh * 0.08), int(oh * (0.445 if has_cap else 0.50)) - wh // 2)
    bev = max(4, int(round(6 * scale_factor)))

    bevel_left = tuple((a + b) // 2 for a, b in zip(pal["bevel_lo"], pal["mat"]))
    bevel_right = tuple((a + b) // 2 for a, b in zip(pal["bevel_hi"], pal["mat"]))
    # 4-edge 45-degree beveled mat aperture lit softly from upper-left
    draw.polygon([(wx - bev, wy - bev), (wx + ww + bev, wy - bev), (wx + ww, wy), (wx, wy)], fill=pal["bevel_lo"])
    draw.polygon([(wx - bev, wy - bev), (wx, wy), (wx, wy + wh), (wx - bev, wy + wh + bev)], fill=bevel_left)
    draw.polygon([(wx, wy + wh), (wx + ww, wy + wh), (wx + ww + bev, wy + wh + bev), (wx - bev, wy + wh + bev)], fill=pal["bevel_hi"])
    draw.polygon([(wx + ww, wy), (wx + ww + bev, wy - bev), (wx + ww + bev, wy + wh + bev), (wx + ww, wy + wh)], fill=bevel_right)
    draw.rectangle([wx - bev, wy - bev, wx + ww + bev, wy + wh + bev], outline=pal["rule"], width=max(1, int(round(scale_factor))))

    if has_cap:
        title_str = (kb.caption_title or "").strip()
        sub_str = (kb.caption_subtitle or "").strip()
        f_title = _load_editorial_font(max(12, int(round(19 * scale_factor))), bold=True, serif=True, text=title_str)
        f_sub = _load_editorial_font(max(10, int(round(13 * scale_factor))), bold=False, serif=False, text=sub_str)
        margin_top = wy + wh + bev + max(10, int(round(16 * scale_factor)))
        max_cap_y = int(oh * 0.94)
        tb = draw.textbbox((0, 0), title_str, font=f_title)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
        if margin_top + th <= max_cap_y:
            draw.text(((ow - tw) // 2, margin_top), title_str, font=f_title, fill=pal["ink_primary"])
        if sub_str:
            sub_y = margin_top + th + max(5, int(round(8 * scale_factor)))
            sb_box = draw.textbbox((0, 0), sub_str, font=f_sub)
            sw_t = sb_box[2] - sb_box[0]
            sh_t = sb_box[3] - sb_box[1]
            if sub_y + sh_t <= max_cap_y:
                # Truncate gracefully if excessively long on a small matte
                disp_sub = sub_str if sw_t <= int(ow * 0.78) else (sub_str[:74].rstrip() + "...")
                sb_box2 = draw.textbbox((0, 0), disp_sub, font=f_sub)
                draw.text(((ow - (sb_box2[2] - sb_box2[0])) // 2, sub_y), disp_sub, font=f_sub, fill=pal["ink_secondary"])

    photo_over = ImageOps.fit(
        raw_primary,
        (max(32, int(round(ww * 1.09))), max(32, int(round(wh * 1.09)))),
        method=Image.Resampling.LANCZOS,
    )
    recess_mask = Image.new("L", (ww, wh), 0)
    rm_draw = ImageDraw.Draw(recess_mask)
    thick = max(4, int(round(7 * scale_factor)))
    rm_draw.rectangle([0, 0, ww - 1, thick], fill=82)
    rm_draw.rectangle([0, 0, thick, wh - 1], fill=72)
    rm_draw.rectangle([0, wh - max(2, thick // 2), ww - 1, wh - 1], fill=28)
    rm_draw.rectangle([ww - max(2, thick // 2), 0, ww - 1, wh - 1], fill=28)
    recess_mask = recess_mask.filter(ImageFilter.GaussianBlur(radius=max(3, int(round(5 * scale_factor)))))
    recess_dark = Image.new("RGB", (ww, wh), (6, 8, 12))

    return {
        "board_rgb": board,
        "photo_over": photo_over,
        "recess_mask": recess_mask,
        "recess_dark": recess_dark,
        "wx": wx,
        "wy": wy,
        "ww": ww,
        "wh": wh,
    }


def _build_kinfolk_linen_folio_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
    pal: dict[str, Any],
) -> dict[str, Any]:
    """Build asymmetric 58/42 Kinfolk / Mack Books archival monograph spread with hairline rules & folio typography."""
    page = _make_textured_surface_rgb(width, height, pal["bg"], seed=404, grad_top=3.0, grad_bot=-4.0, grain_std=1.5)
    draw = ImageDraw.Draw(page)

    left_zone_w = int(round(width * 0.58))
    left_zone_w -= left_zone_w % 2
    max_pw = int(left_zone_w * 0.80)
    max_ph = int(height * 0.78)
    aspect = raw_primary.width / max(1, raw_primary.height)
    if max_pw / max(1.0, float(max_ph)) >= aspect:
        ph = max_ph
        pw = int(round(ph * aspect))
    else:
        pw = max_pw
        ph = int(round(pw / max(0.25, aspect)))
    pw = max(int(left_zone_w * 0.44), min(max_pw, pw))
    ph = max(int(height * 0.44), min(max_ph, ph))

    px = (left_zone_w - pw) // 2
    py = (height - ph) // 2

    pad_f = max(5, int(round(8 * scale_factor)))
    rule_w = max(1, int(round(1.2 * scale_factor)))
    draw.rectangle([px - pad_f, py - pad_f, px + pw + pad_f, py + ph + pad_f], outline=pal["rule"], width=rule_w)
    draw.line([(left_zone_w, int(height * 0.12)), (left_zone_w, int(height * 0.88))], fill=pal["rule"], width=rule_w)

    tx = left_zone_w + int(round(46 * scale_factor))
    max_tw = max(80, width - tx - int(round(42 * scale_factor)))
    cur_y = int(height * 0.23)

    badge_str = _clean_editorial_badge(kb.camera_badge or "") or "ARCHIVAL FOLIO"
    f_badge = _load_editorial_font(max(10, int(round(13 * scale_factor))), bold=True, serif=False, text=badge_str)
    draw.text((tx, cur_y), badge_str, font=f_badge, fill=pal["ink_muted"])
    cur_y += max(12, int(round(26 * scale_factor)))

    rule_len = min(max_tw, int(round(46 * scale_factor)))
    draw.rectangle([tx, cur_y, tx + rule_len, cur_y + max(1, int(round(1.5 * scale_factor)))], fill=pal["rule"])
    cur_y += max(10, int(round(24 * scale_factor)))

    title_str = (kb.caption_title or "Curated Exhibition Study").strip()
    f_title = _load_editorial_font(max(14, int(round(30 * scale_factor))), bold=True, serif=True, text=title_str)
    for ln in _wrap_editorial_text(draw, title_str, f_title, max_tw, max_lines=4):
        draw.text((tx, cur_y), ln, font=f_title, fill=pal["ink_primary"])
        cur_y += max(16, int(round(40 * scale_factor)))

    sub_str = (kb.caption_subtitle or "").strip()
    if sub_str:
        cur_y += max(6, int(round(12 * scale_factor)))
        f_sub = _load_editorial_font(max(11, int(round(18 * scale_factor))), bold=False, serif=False, text=sub_str)
        for ln in _wrap_editorial_text(draw, sub_str, f_sub, max_tw, max_lines=5):
            if cur_y > int(height * 0.80):
                break
            draw.text((tx, cur_y), ln, font=f_sub, fill=pal["ink_secondary"])
            cur_y += max(12, int(round(27 * scale_factor)))

    folio_mark = "EDITION PLATE"
    f_folio = _load_editorial_font(max(9, int(round(12 * scale_factor))), bold=False, serif=True, text=folio_mark)
    fy_rule = int(height * 0.84)
    draw.line([(tx, fy_rule), (tx + min(max_tw, int(round(140 * scale_factor))), fy_rule)], fill=pal["rule"], width=1)
    draw.text((tx, fy_rule + max(6, int(round(10 * scale_factor)))), folio_mark, font=f_folio, fill=pal["ink_muted"])

    photo_over = ImageOps.fit(
        raw_primary,
        (max(32, int(round(pw * 1.10))), max(32, int(round(ph * 1.10)))),
        method=Image.Resampling.LANCZOS,
    )
    return {
        "page_rgb": page,
        "photo_over": photo_over,
        "px": px,
        "py": py,
        "pw": pw,
        "ph": ph,
    }


def _build_panoramic_frieze_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    has_card: bool,
    scale_factor: float,
    pal: dict[str, Any],
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
    focal_uv: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """Build an exhibition gallery wall hanging 3 aspect-preserved matted prints across equal gutters."""
    sec_max_dim = max(512, (max(width, height) * 5) // 4)
    sec_img = (
        _load_graded_photo_rgb(kb.secondary_source, color_grade_fx, max_dim=sec_max_dim)
        if (kb.secondary_source and Path(kb.secondary_source).exists())
        else None
    )
    tert_img = (
        _load_graded_photo_rgb(kb.tertiary_source, color_grade_fx, max_dim=sec_max_dim)
        if (kb.tertiary_source and Path(kb.tertiary_source).exists())
        else None
    )

    sw, sh = raw_primary.size
    fx, fy = focal_uv if focal_uv is not None else kb.start_center
    if sec_img is not None and tert_img is not None:
        imgs = [raw_primary, sec_img, tert_img]
    elif sec_img is not None:
        cw_d, ch_d = max(48, int(sw * 0.62)), max(48, int(sh * 0.62))
        dx0 = int(round(min(max(0.0, fx * sw - cw_d * 0.5), max(0.0, float(sw - cw_d)))))
        dy0 = int(round(min(max(0.0, fy * sh - ch_d * 0.5), max(0.0, float(sh - ch_d)))))
        imgs = [raw_primary, sec_img, raw_primary.crop((dx0, dy0, dx0 + cw_d, dy0 + ch_d))]
    else:
        cw_s, ch_s = max(48, int(sw * 0.68)), max(48, int(sh * 0.68))
        centers = [(0.24, 0.50), (fx, fy), (0.76, 0.46)]
        imgs = []
        for cx_p, cy_p in centers:
            x0 = int(round(min(max(0.0, cx_p * sw - cw_s * 0.5), max(0.0, float(sw - cw_s)))))
            y0 = int(round(min(max(0.0, cy_p * sh - ch_s * 0.5), max(0.0, float(sh - ch_s)))))
            imgs.append(raw_primary.crop((x0, y0, x0 + cw_s, y0 + ch_s)))

    slot_w = int(width * 0.43)
    slot_h = int(height * (0.68 if has_card else 0.76))
    border_px = max(10, int(round(13 * scale_factor)))
    cards = [
        _build_double_matte_card(im, max_w=slot_w, max_h=slot_h, scale_factor=scale_factor, border_px=border_px)
        for im in imgs
    ]
    gutter = max(16, int(round(width * 0.045)))
    outer_pad = max(28, int(round(width * 0.08)))
    row_w = sum(c.width for c in cards) + gutter * max(0, len(cards) - 1)
    fw = max(int(round(width * 1.38)), row_w + outer_pad * 2)
    fw -= fw % 2
    fh = int(round(height * 1.04))
    fh -= fh % 2

    canvas = _make_textured_surface_rgb(fw, fh, pal["bg"], seed=505, grad_top=5.0, grad_bot=-6.0, grain_std=1.5)
    cy = int(fh * (0.435 if has_card else 0.49))
    x_cur = (fw - row_w) // 2
    for card in cards:
        py = max(8, min(fh - card.height - 8, cy - card.height // 2))
        canvas.paste(card, (x_cur, py), card)
        x_cur += card.width + gutter

    return {
        "canvas_rgb": canvas,
        "fw": fw,
        "fh": fh,
        "outer_pad": outer_pad,
    }


def _build_atelier_layered_duo_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    has_card: bool,
    scale_factor: float,
    pal: dict[str, Any],
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
    focal_uv: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """Build two strictly orthogonal (0-deg) layered archival prints with soft ambient-occlusion shadow."""
    bw = int(round(width * 1.05))
    bh = int(round(height * 1.05))
    bg_rgb = _make_textured_surface_rgb(bw, bh, pal["bg"], seed=606, grad_top=4.0, grad_bot=-5.0, grain_std=1.5)
    b_draw = ImageDraw.Draw(bg_rgb)
    inset_x = max(16, int(round(44 * scale_factor)))
    inset_y = max(14, int(round(36 * scale_factor)))
    b_draw.rectangle([inset_x, inset_y, bw - inset_x, bh - inset_y], outline=pal["rule"], width=1)

    anc_border = max(10, int(round(13 * scale_factor)))
    anchor_rgba = _build_double_matte_card(
        raw_primary,
        max_w=int(width * 0.54),
        max_h=int(height * (0.68 if has_card else 0.75)),
        scale_factor=scale_factor,
        border_px=anc_border,
    )

    if kb.secondary_source and Path(kb.secondary_source).exists():
        fg_src = _load_graded_photo_rgb(kb.secondary_source, color_grade_fx, max_dim=max(512, (max(width, height) * 5) // 4))
    else:
        sw, sh = raw_primary.size
        cw_d = max(48, int(round(sw * 0.56)))
        ch_d = max(48, int(round(sh * 0.56)))
        ex, ey = focal_uv if focal_uv is not None else kb.end_center
        dx0 = int(round(min(max(0.0, ex * sw - cw_d * 0.5), max(0.0, float(sw - cw_d)))))
        dy0 = int(round(min(max(0.0, ey * sh - ch_d * 0.5), max(0.0, float(sh - ch_d)))))
        fg_src = raw_primary.crop((dx0, dy0, dx0 + cw_d, dy0 + ch_d))

    fg_border = max(8, int(round(10 * scale_factor)))
    fg_rgba = _build_double_matte_card(
        fg_src,
        max_w=int(width * 0.35),
        max_h=int(height * (0.47 if has_card else 0.53)),
        scale_factor=scale_factor,
        border_px=fg_border,
    )
    return {
        "bg_rgb": bg_rgb,
        "anchor_rgba": anchor_rgba,
        "fg_rgba": fg_rgba,
    }


def _build_leica_frame_finder_assets(
    raw_primary: Image.Image,
    width: int,
    height: int,
    has_card: bool,
    scale_factor: float,
) -> dict[str, Any]:
    """Build crisp rangefinder brightline corner framelines with subtle outer surround calming."""
    sharp_over = ImageOps.fit(
        raw_primary,
        (max(32, int(round(width * 1.10))), max(32, int(round(height * 1.10)))),
        method=Image.Resampling.LANCZOS,
    )
    rx0 = int(round(width * 0.09))
    rx1 = width - rx0
    ry0 = int(round(height * 0.09))
    ry1 = int(round(height * (0.83 if has_card else 0.91)))

    surround_mask = Image.new("L", (width, height), 58)
    sm_draw = ImageDraw.Draw(surround_mask)
    sm_draw.rectangle([rx0, ry0, rx1, ry1], fill=0)
    surround_mask = surround_mask.filter(ImageFilter.GaussianBlur(radius=2))

    # Pre-seed transparent RGB to platinum-white so bilinear resampling never bleeds black fringes
    reticle = Image.new("RGBA", (width, height), (232, 236, 242, 0))
    r_draw = ImageDraw.Draw(reticle)
    lw = max(1, int(round(2.0 * scale_factor)))
    corner_len = max(16, int(round(36 * scale_factor)))
    tick_len = max(4, int(round(9 * scale_factor)))
    col = (232, 236, 242, 215)

    # Top-left brightline corner
    r_draw.line([(rx0, ry0), (rx0 + corner_len, ry0)], fill=col, width=lw)
    r_draw.line([(rx0, ry0), (rx0, ry0 + corner_len)], fill=col, width=lw)
    # Top-right brightline corner
    r_draw.line([(rx1 - corner_len, ry0), (rx1, ry0)], fill=col, width=lw)
    r_draw.line([(rx1, ry0), (rx1, ry0 + corner_len)], fill=col, width=lw)
    # Bottom-left brightline corner
    r_draw.line([(rx0, ry1), (rx0 + corner_len, ry1)], fill=col, width=lw)
    r_draw.line([(rx0, ry1 - corner_len), (rx0, ry1)], fill=col, width=lw)
    # Bottom-right brightline corner
    r_draw.line([(rx1 - corner_len, ry1), (rx1, ry1)], fill=col, width=lw)
    r_draw.line([(rx1, ry1 - corner_len), (rx1, ry1)], fill=col, width=lw)

    # Edge midpoint registration ticks
    mx = (rx0 + rx1) // 2
    my = (ry0 + ry1) // 2
    tick_col = (232, 236, 242, 165)
    r_draw.line([(mx, ry0), (mx, ry0 + tick_len)], fill=tick_col, width=lw)
    r_draw.line([(mx, ry1 - tick_len), (mx, ry1)], fill=tick_col, width=lw)
    r_draw.line([(rx0, my), (rx0 + tick_len, my)], fill=tick_col, width=lw)
    r_draw.line([(rx1 - tick_len, my), (rx1, my)], fill=tick_col, width=lw)

    # Subtle optical rangefinder center coincidence patch
    pw_r = max(12, int(round(22 * scale_factor)))
    ph_r = max(8, int(round(15 * scale_factor)))
    r_draw.rectangle([mx - pw_r, my - ph_r, mx + pw_r, my + ph_r], outline=(232, 236, 242, 110), width=max(1, int(round(1.2 * scale_factor))))

    return {
        "sharp_over": sharp_over,
        "surround_mask": surround_mask,
        "dim_rgb": Image.new("RGB", (width, height), (8, 10, 14)),
        "reticle_rgba": reticle,
    }


def _load_companion_photos(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
    focal_uv: Optional[tuple[float, float]] = None,
    count: int = 3,
) -> list[Image.Image]:
    """Return `count` (2 or 3) harmonized RGB photos using secondary/tertiary sources or compositional crops."""
    max_dim = max(512, (max(width, height) * 5) // 4)
    sec_img = (
        _load_graded_photo_rgb(kb.secondary_source, color_grade_fx, max_dim=max_dim)
        if (kb.secondary_source and Path(kb.secondary_source).exists())
        else None
    )
    tert_img = (
        _load_graded_photo_rgb(kb.tertiary_source, color_grade_fx, max_dim=max_dim)
        if (kb.tertiary_source and Path(kb.tertiary_source).exists())
        else None
    )
    sw, sh = raw_primary.size
    fx, fy = focal_uv if focal_uv is not None else kb.start_center

    def _subcrop(cx_p: float, cy_p: float, frac: float = 0.64) -> Image.Image:
        cw = max(32, int(round(sw * frac)))
        ch = max(32, int(round(sh * frac)))
        x0 = int(round(min(max(0.0, cx_p * sw - cw * 0.5), max(0.0, float(sw - cw)))))
        y0 = int(round(min(max(0.0, cy_p * sh - ch * 0.5), max(0.0, float(sh - ch)))))
        return raw_primary.crop((x0, y0, x0 + cw, y0 + ch))

    if count <= 2:
        p2 = sec_img if sec_img is not None else _subcrop(fx, fy, 0.58)
        return [raw_primary, p2]

    if sec_img is not None and tert_img is not None:
        return [raw_primary, sec_img, tert_img]
    if sec_img is not None:
        return [raw_primary, sec_img, _subcrop(fx, fy, 0.58)]
    return [_subcrop(0.26, 0.50, 0.68), _subcrop(fx, fy, 0.66), _subcrop(0.74, 0.46, 0.68)]


def _build_swiss_monograph_grid_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
    pal: dict[str, Any],
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
    focal_uv: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """#01 Taschen / Mack Books 64/36 Swiss architectural spread + live synchronized macro detail inset."""
    page = _make_textured_surface_rgb(width, height, pal["bg"], seed=1101, grad_top=3.0, grad_bot=-4.0, grain_std=1.5)
    draw = ImageDraw.Draw(page)
    split_x = int(round(width * 0.63))
    pad = max(16, int(round(36 * scale_factor)))
    rule_w = max(1, int(round(1.5 * scale_factor)))

    hx, hy = pad, pad
    hw = max(32, split_x - pad * 2)
    hh = max(32, height - pad * 2)
    draw.rectangle([hx - 2, hy - 2, hx + hw + 1, hy + hh + 1], outline=pal["rule"], width=1)
    draw.line([(split_x, pad), (split_x, height - pad)], fill=pal["rule"], width=rule_w)

    tx = split_x + pad
    tw = max(48, width - tx - pad)
    cur_y = pad + max(4, int(round(12 * scale_factor)))

    badge_str = _clean_editorial_badge(kb.camera_badge or "") or "MONOGRAPH  ·  PLATE 01"
    f_eyebrow = _load_editorial_font(max(9, int(round(12 * scale_factor))), bold=True, serif=False, text=badge_str)
    draw.text((tx, cur_y), badge_str, font=f_eyebrow, fill=pal["ink_muted"])
    cur_y += max(12, int(round(22 * scale_factor)))
    draw.line([(tx, cur_y), (tx + min(tw, int(round(72 * scale_factor))), cur_y)], fill=pal["rule"], width=rule_w)
    cur_y += max(10, int(round(18 * scale_factor)))

    title_str = (kb.caption_title or "Architectural Monograph").strip()
    f_title = _load_editorial_font(max(13, int(round(28 * scale_factor))), bold=True, serif=True, text=title_str)
    for ln in _wrap_editorial_text(draw, title_str, f_title, tw, max_lines=3):
        draw.text((tx, cur_y), ln, font=f_title, fill=pal["ink_primary"])
        cur_y += max(15, int(round(36 * scale_factor)))

    sub_str = (kb.caption_subtitle or "").strip()
    if sub_str:
        cur_y += max(4, int(round(8 * scale_factor)))
        f_sub = _load_editorial_font(max(10, int(round(15 * scale_factor))), bold=False, serif=False, text=sub_str)
        for ln in _wrap_editorial_text(draw, sub_str, f_sub, tw, max_lines=3):
            if cur_y > int(height * 0.44):
                break
            draw.text((tx, cur_y), ln, font=f_sub, fill=pal["ink_secondary"])
            cur_y += max(12, int(round(22 * scale_factor)))

    dx = tx
    dw = tw
    dh = max(32, int(height * 0.41))
    dy = max(int(height * 0.47), height - pad - dh - max(12, int(round(24 * scale_factor))))
    dh = max(32, min(dh, height - pad - max(12, int(round(24 * scale_factor))) - dy))
    draw.rectangle([dx - 2, dy - 2, dx + dw + 1, dy + dh + 1], outline=pal["rule"], width=1)
    detail_lbl = "DETAIL STUDY"
    f_det = _load_editorial_font(max(9, int(round(11 * scale_factor))), bold=False, serif=False, text=detail_lbl)
    draw.text((dx, dy + dh + max(4, int(round(8 * scale_factor)))), detail_lbl, font=f_det, fill=pal["ink_muted"])

    hero_over = ImageOps.fit(
        raw_primary, (max(32, int(round(hw * 1.12))), max(32, int(round(hh * 1.12)))), method=Image.Resampling.LANCZOS
    )
    if kb.secondary_source and Path(kb.secondary_source).exists():
        sec_im = _load_graded_photo_rgb(kb.secondary_source, color_grade_fx, max_dim=max(512, max(width, height)))
        det_over = ImageOps.fit(
            sec_im, (max(32, int(round(dw * 1.16))), max(32, int(round(dh * 1.16)))), method=Image.Resampling.LANCZOS
        )
    else:
        sw, sh = raw_primary.size
        fx, fy = focal_uv if focal_uv is not None else kb.start_center
        crop_w = max(32.0, sw * 0.56)
        crop_h = max(32.0, crop_w * (dh / max(1.0, float(dw))))
        if crop_h > sh * 0.84:
            crop_h = sh * 0.84
            crop_w = crop_h * (dw / max(1.0, float(dh)))
        x0 = min(max(0.0, fx * sw - crop_w * 0.5), max(0.0, sw - crop_w))
        y0 = min(max(0.0, fy * sh - crop_h * 0.5), max(0.0, sh - crop_h))
        det_over = raw_primary.transform(
            (max(32, int(round(dw * 1.16))), max(32, int(round(dh * 1.16)))),
            Image.Transform.EXTENT,
            (x0, y0, x0 + crop_w, y0 + crop_h),
            resample=Image.Resampling.BICUBIC,
        )

    return {
        "page_rgb": page,
        "hero_over": hero_over,
        "det_over": det_over,
        "hx": hx,
        "hy": hy,
        "hw": hw,
        "hh": hh,
        "dx": dx,
        "dy": dy,
        "dw": dw,
        "dh": dh,
    }


def _build_editorial_bookspread_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
    focal_uv: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """#02 Open hardcover coffee-table book spread with 3D concave center-spine gutter lighting."""
    p1, p2 = _load_companion_photos(raw_primary, kb, width, height, color_grade_fx, focal_uv, count=2)
    table = _make_textured_surface_rgb(width, height, (24, 26, 30), seed=1202, grad_top=3.0, grad_bot=-4.0, grain_std=1.6)

    bm_x, bm_y = int(round(width * 0.065)), int(round(height * 0.08))
    bw = max(64, width - bm_x * 2)
    bh = max(64, height - bm_y * 2)
    sh_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(sh_layer).rectangle(
        [bm_x - 6, bm_y + max(4, int(round(10 * scale_factor))), bm_x + bw + 6, bm_y + bh + max(8, int(round(18 * scale_factor)))],
        fill=(0, 0, 0, 175),
    )
    sh_layer = sh_layer.filter(ImageFilter.GaussianBlur(radius=max(6, int(round(20 * scale_factor)))))
    table.paste(sh_layer, (0, 0), sh_layer)

    # Cream cotton-rag open pages with physically inspired concave 3D center-spine shading
    pages = _make_textured_surface_rgb(bw, bh, (230, 226, 218), seed=1203, grad_top=3.0, grad_bot=-3.0, grain_std=1.4)
    parr = np.asarray(pages, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, bw, dtype=np.float32)[None, :]
    dist_spine = np.abs(xx)
    spine_shade = (
        1.0
        - 0.25 * np.exp(-((dist_spine / 0.045) ** 1.6))
        + 0.032 * np.exp(-(((dist_spine - 0.075) / 0.05) ** 2))
    )
    parr = np.clip(parr * spine_shade[:, :, None], 8.0, 240.0).astype(np.uint8)
    pages = Image.fromarray(parr, mode="RGB")
    pd = ImageDraw.Draw(pages)
    mid = bw // 2
    pd.line([(mid, 0), (mid, bh)], fill=(122, 116, 108), width=max(1, int(round(1.5 * scale_factor))))
    pd.rectangle([0, 0, bw - 1, bh - 1], outline=(186, 180, 170), width=max(1, int(round(1.5 * scale_factor))))

    m_out = max(14, int(round(56 * scale_factor)))
    m_in = max(12, int(round(48 * scale_factor)))
    top_m = max(14, int(round(58 * scale_factor)))
    bot_m = max(28, int(round(104 * scale_factor)))
    slot_w = max(32, mid - m_out - m_in)
    slot_h = max(32, bh - top_m - bot_m)

    def _fit_box(im: Image.Image, sw_b: int, sh_b: int) -> tuple[int, int]:
        a = im.width / max(1.0, float(im.height))
        if sw_b / max(1.0, float(sh_b)) >= a:
            return max(16, int(round(sh_b * a))), sh_b
        return sw_b, max(16, int(round(sw_b / max(0.2, a))))

    w1, h1 = _fit_box(p1, slot_w, slot_h)
    w2, h2 = _fit_box(p2, slot_w, slot_h)
    x1, y1 = bm_x + m_out + (slot_w - w1) // 2, bm_y + top_m + (slot_h - h1) // 2
    x2, y2 = bm_x + mid + m_in + (slot_w - w2) // 2, bm_y + top_m + (slot_h - h2) // 2

    pd.rectangle([x1 - bm_x - 2, y1 - bm_y - 2, x1 - bm_x + w1 + 1, y1 - bm_y + h1 + 1], outline=(178, 172, 162), width=1)
    pd.rectangle([x2 - bm_x - 2, y2 - bm_y - 2, x2 - bm_x + w2 + 1, y2 - bm_y + h2 + 1], outline=(178, 172, 162), width=1)

    f_fol = _load_editorial_font(max(9, int(round(12 * scale_factor))), bold=False, serif=False)
    pd.text((m_out, bh - max(12, int(round(38 * scale_factor)))), "01", font=f_fol, fill=(134, 128, 118))
    pd.text((bw - m_out - max(14, int(round(22 * scale_factor))), bh - max(12, int(round(38 * scale_factor)))), "02", font=f_fol, fill=(134, 128, 118))

    t_left = (kb.caption_title or "").strip()
    t_right = (kb.caption_subtitle or "").strip()
    if t_left:
        f_cap = _load_editorial_font(max(10, int(round(16 * scale_factor))), bold=True, serif=True, text=t_left)
        lines_l = _wrap_editorial_text(pd, t_left, f_cap, slot_w, max_lines=1)
        if lines_l:
            pd.text((x1 - bm_x, y1 - bm_y + h1 + max(6, int(round(14 * scale_factor)))), lines_l[0], font=f_cap, fill=(36, 34, 32))
    if t_right:
        f_sub = _load_editorial_font(max(9, int(round(13 * scale_factor))), bold=False, serif=False, text=t_right)
        lines_r = _wrap_editorial_text(pd, t_right, f_sub, slot_w, max_lines=2)
        cy_r = y2 - bm_y + h2 + max(6, int(round(14 * scale_factor)))
        for ln in lines_r:
            pd.text((x2 - bm_x, cy_r), ln, font=f_sub, fill=(84, 80, 74))
            cy_r += max(11, int(round(18 * scale_factor)))

    table.paste(pages, (bm_x, bm_y))
    p1_over = ImageOps.fit(p1, (max(32, int(round(w1 * 1.10))), max(32, int(round(h1 * 1.10)))), method=Image.Resampling.LANCZOS)
    p2_over = ImageOps.fit(p2, (max(32, int(round(w2 * 1.10))), max(32, int(round(h2 * 1.10)))), method=Image.Resampling.LANCZOS)

    return {
        "table_rgb": table,
        "p1_over": p1_over,
        "p2_over": p2_over,
        "x1": x1,
        "y1": y1,
        "w1": w1,
        "h1": h1,
        "x2": x2,
        "y2": y2,
        "w2": w2,
        "h2": h2,
    }


def _build_gallery_wall_triptych_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
    pal: dict[str, Any],
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
    focal_uv: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """#03 Tate Modern exhibition hanging-wall trio with 3 directional overhead track-light cones."""
    imgs = _load_companion_photos(raw_primary, kb, width, height, color_grade_fx, focal_uv, count=3)
    wall_w = int(round(width * 1.24))
    wall_w -= wall_w % 2
    wall_h = int(round(height * 1.06))
    wall_h -= wall_h % 2

    wall_base = (34, 36, 42) if not pal["is_light"] else pal["bg"]
    wall = _make_textured_surface_rgb(wall_w, wall_h, wall_base, seed=1303, grad_top=4.0, grad_bot=-5.0, grain_std=1.8)

    slot_w = int(round(width * 0.30))
    slot_h = int(round(height * 0.55))
    b_px = max(8, int(round(14 * scale_factor)))
    cards = [_build_double_matte_card(im, max_w=slot_w, max_h=slot_h, scale_factor=scale_factor, border_px=b_px) for im in imgs]
    gutter = max(14, int(round(34 * scale_factor)))
    row_w = sum(c.width for c in cards) + gutter * 2
    sx = max(16, (wall_w - row_w) // 2)
    cy = int(wall_h * 0.49)

    # Paint 3 soft overhead radial gallery track-light cones onto the wall before hanging prints
    warr = np.asarray(wall, dtype=np.float32)
    sw_q, sh_q = max(48, wall_w // 4), max(27, wall_h // 4)
    yy_q, xx_q = np.mgrid[0:sh_q, 0:sw_q].astype(np.float32)
    cone_q = np.zeros((sh_q, sw_q), dtype=np.float32)
    x_cur = sx
    for c in cards:
        cx_i = (x_cur + c.width * 0.5) * (sw_q / max(1.0, float(wall_w)))
        sig_x = max(4.0, c.width * 0.62 * (sw_q / max(1.0, float(wall_w))))
        sig_y = max(4.0, sh_q * 0.44)
        cone_q += np.exp(-(((xx_q - cx_i) / sig_x) ** 2) - (((yy_q - sh_q * 0.32) / sig_y) ** 2))
        x_cur += c.width + gutter
    cone_full = np.asarray(
        Image.fromarray(np.clip(cone_q * 180.0, 0, 255).astype(np.uint8), mode="L").resize(
            (wall_w, wall_h), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 180.0
    amp_light = 22.0 if pal["is_light"] else 42.0
    warr[:, :, 0] += cone_full * amp_light
    warr[:, :, 1] += cone_full * (amp_light * 0.90)
    warr[:, :, 2] += cone_full * (amp_light * 0.72)
    wall = Image.fromarray(np.clip(warr, 6.0, 244.0).astype(np.uint8), mode="RGB")

    wd = ImageDraw.Draw(wall)
    rail_y = int(wall_h * 0.08)
    rail_col = (140, 136, 130) if pal["is_light"] else (78, 82, 90)
    wire_col = (156, 152, 146) if pal["is_light"] else (92, 96, 104)
    wd.line([(0, rail_y), (wall_w, rail_y)], fill=rail_col, width=max(1, int(round(2.5 * scale_factor))))
    x_cur = sx
    max_card_bot = cy
    for c in cards:
        py = max(rail_y + 8, cy - c.height // 2)
        for wx_off in (int(c.width * 0.22), int(c.width * 0.78)):
            wd.line([(x_cur + wx_off, rail_y), (x_cur + wx_off, py + max(8, int(round(28 * scale_factor))))], fill=wire_col, width=1)
        wall.paste(c, (x_cur, py), c)
        max_card_bot = max(max_card_bot, py + c.height)
        x_cur += c.width + gutter

    t_str = (kb.caption_title or "").strip()
    s_str = (kb.caption_subtitle or "").strip()
    ink_hi = pal["ink_primary"] if pal["is_light"] else (234, 230, 222)
    ink_lo = pal["ink_secondary"] if pal["is_light"] else (168, 166, 160)
    cap_y = min(int(wall_h * 0.87), max_card_bot + max(8, int(round(16 * scale_factor))))
    if t_str:
        f_tit = _load_editorial_font(max(11, int(round(18 * scale_factor))), bold=True, serif=True, text=t_str)
        tb = wd.textbbox((0, 0), t_str, font=f_tit)
        wd.text(((wall_w - (tb[2] - tb[0])) // 2, cap_y), t_str, font=f_tit, fill=ink_hi)
        cap_y += (tb[3] - tb[1]) + max(4, int(round(8 * scale_factor)))
    if s_str and cap_y < int(wall_h * 0.94):
        f_sub = _load_editorial_font(max(9, int(round(13 * scale_factor))), bold=False, serif=False, text=s_str)
        disp_s = s_str if len(s_str) <= 84 else (s_str[:81].rstrip() + "...")
        sb = wd.textbbox((0, 0), disp_s, font=f_sub)
        wd.text(((wall_w - (sb[2] - sb[0])) // 2, cap_y), disp_s, font=f_sub, fill=ink_lo)

    return {"wall_rgb": wall, "wall_w": wall_w, "wall_h": wall_h}


def _build_archival_lightbox_strip_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
    focal_uv: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """#04 Backlit 120 medium-format positive film strip on a 5000K transilluminated darkroom light table."""
    imgs = _load_companion_photos(raw_primary, kb, width, height, color_grade_fx, focal_uv, count=3)
    sw_strip = int(round(width * 1.46))
    sw_strip -= sw_strip % 2
    sh_strip = int(round(height * 1.06))
    sh_strip -= sh_strip % 2

    # Build 5000K transilluminated diffuser glow bed on matte obsidian darkroom bench at 1/4 res -> smooth upscale
    qw, qh = max(48, sw_strip // 4), max(27, sh_strip // 4)
    yy_q, xx_q = np.mgrid[0:qh, 0:qw].astype(np.float32)
    bx0, bx1 = qw * 0.06, qw * 0.94
    by0, by1 = qh * 0.16, qh * 0.80
    dx = np.maximum(0.0, np.maximum(bx0 - xx_q, xx_q - bx1))
    dy = np.maximum(0.0, np.maximum(by0 - yy_q, yy_q - by1))
    sig = max(4.0, 20.0 * scale_factor)
    glow = np.exp(-((dx * dx + dy * dy) / (sig * sig)))
    bg_q = np.empty((qh, qw, 3), dtype=np.float32)
    bg_q[:, :, 0] = 13.0 + glow * 204.0
    bg_q[:, :, 1] = 14.0 + glow * 196.0
    bg_q[:, :, 2] = 18.0 + glow * 182.0
    table = Image.fromarray(np.clip(bg_q, 6.0, 236.0).astype(np.uint8), mode="RGB").resize(
        (sw_strip, sh_strip), Image.Resampling.BILINEAR
    )
    td = ImageDraw.Draw(table)

    strip_y0 = int(sh_strip * 0.19)
    strip_y1 = int(sh_strip * 0.77)
    strip_x0 = int(sw_strip * 0.08)
    strip_x1 = int(sw_strip * 0.92)
    td.rectangle([strip_x0, strip_y0, strip_x1, strip_y1], fill=(10, 11, 14))

    f_reb = _load_editorial_font(max(9, int(round(12 * scale_factor))), bold=True, serif=False)
    frame_gap = max(12, int(round(32 * scale_factor)))
    pad_top = max(16, int(round(40 * scale_factor)))
    pad_bot = max(16, int(round(40 * scale_factor)))
    inner_h = max(32, (strip_y1 - strip_y0) - pad_top - pad_bot)
    inner_w = max(32, ((strip_x1 - strip_x0) - frame_gap * 4) // 3)

    for i, im in enumerate(imgs):
        fx0 = strip_x0 + frame_gap + i * (inner_w + frame_gap)
        fy0 = strip_y0 + pad_top
        fit_im = ImageOps.fit(im, (inner_w, inner_h), method=Image.Resampling.LANCZOS)
        mask = Image.new("L", (inner_w, inner_h), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle([0, 0, inner_w - 1, inner_h - 1], radius=max(2, int(round(5 * scale_factor))), fill=255)
        n_sz = max(2, int(round(4 * scale_factor)))
        for ny_r in (0.16, 0.24):
            ny_p = int(inner_h * ny_r)
            md.polygon([(0, ny_p - n_sz), (n_sz + 2, ny_p), (0, ny_p + n_sz)], fill=0)
        table.paste(fit_im, (fx0, fy0), mask)
        td.text((fx0 + 4, strip_y0 + max(4, int(round(12 * scale_factor)))), f"120 POSITIVE   ·   0{i + 1}A", font=f_reb, fill=(210, 174, 98))
        td.text((fx0 + max(4, inner_w - int(round(92 * scale_factor))), strip_y1 - max(12, int(round(26 * scale_factor)))), f"6x7  #{i + 1:02d}", font=f_reb, fill=(142, 146, 156))

    cap_txt = (kb.caption_title or "").strip()
    if cap_txt:
        f_cap = _load_editorial_font(max(11, int(round(18 * scale_factor))), bold=True, serif=True, text=cap_txt)
        cb = td.textbbox((0, 0), cap_txt, font=f_cap)
        td.text(((sw_strip - (cb[2] - cb[0])) // 2, int(sh_strip * 0.85)), cap_txt, font=f_cap, fill=(224, 220, 212))

    return {"strip_rgb": table, "sw": sw_strip, "sh": sh_strip}


def _build_optical_tilt_shift_assets(
    raw_primary: Image.Image,
    width: int,
    height: int,
    scale_factor: float,
) -> dict[str, Any]:
    """#05 Zero-artifact 4x5 large-format Scheimpflug optical plane-of-focus sweep."""
    ov_w, ov_h = max(64, int(round(width * 1.12))), max(64, int(round(height * 1.12)))
    sharp = ImageOps.fit(raw_primary, (ov_w, ov_h), method=Image.Resampling.LANCZOS)
    bokeh = sharp.filter(ImageFilter.GaussianBlur(radius=max(4, int(round(10 * scale_factor)))))
    sharp_pop = ImageEnhance.Contrast(sharp).enhance(1.05)
    return {"sharp_over": sharp_pop, "bokeh_over": bokeh}


def _build_darkroom_developer_assets(
    raw_primary: Image.Image,
    width: int,
    height: int,
    focal_uv: tuple[float, float],
) -> dict[str, Any]:
    """#06 Two-stage wet darkroom silver-gelatin emergence -> radial warm living color bloom."""
    ov_w, ov_h = max(64, int(round(width * 1.10))), max(64, int(round(height * 1.10)))
    col_over = ImageOps.fit(raw_primary, (ov_w, ov_h), method=Image.Resampling.LANCZOS)
    bw_l = ImageEnhance.Contrast(col_over.convert("L")).enhance(1.12)
    ba = np.asarray(bw_l, dtype=np.float32)
    bw_rgb = np.stack(
        [np.clip(ba * 1.03, 0, 255), np.clip(ba * 1.00, 0, 255), np.clip(ba * 0.94, 0, 255)],
        axis=-1,
    ).astype(np.uint8)
    fx_c = max(0.22, min(0.78, float(focal_uv[0])))
    fy_c = max(0.22, min(0.78, float(focal_uv[1])))
    return {
        "color_over": col_over,
        "bw_over": Image.fromarray(bw_rgb, mode="RGB"),
        "paper_rgb": (224.0, 218.0, 206.0),
        "focal_xy": (fx_c, fy_c),
    }


def _build_cinema_gate_reveal_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
) -> dict[str, Any]:
    """#07 35mm Panavision velvet 2.39:1 matte-box curtain opening + native lower-letterbox typography."""
    ov_w, ov_h = max(64, int(round(width * 1.12))), max(64, int(round(height * 1.12)))
    photo_over = ImageOps.fit(raw_primary, (ov_w, ov_h), method=Image.Resampling.LANCZOS)
    bar_h_target = max(24, int(round((height - width / 2.39) * 0.5)))

    txt_bar = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    td = ImageDraw.Draw(txt_bar)
    t_s = (kb.caption_title or "").strip()
    s_s = (kb.caption_subtitle or "").strip()
    y_mid = height - bar_h_target // 2
    if t_s:
        disp_t = t_s.upper() if not _contains_cjk(t_s) else t_s
        f_tit = _load_editorial_font(max(11, int(round(24 * scale_factor))), bold=True, serif=True, text=disp_t)
        tb = td.textbbox((0, 0), disp_t, font=f_tit)
        ty = y_mid - (tb[3] - tb[1]) - max(4, int(round(8 * scale_factor))) if s_s else y_mid - (tb[3] - tb[1]) // 2
        td.text(((width - (tb[2] - tb[0])) // 2, ty), disp_t, font=f_tit, fill=(240, 236, 228, 245))
        if s_s:
            line_y = y_mid + max(2, int(round(2 * scale_factor)))
            half_l = max(16, int(round(34 * scale_factor)))
            td.line([(width // 2 - half_l, line_y), (width // 2 + half_l, line_y)], fill=(210, 180, 118, 210), width=max(1, int(round(1.4 * scale_factor))))
            disp_s = s_s if len(s_s) <= 84 else (s_s[:81].rstrip() + "...")
            f_sub = _load_editorial_font(max(9, int(round(13 * scale_factor))), bold=False, serif=False, text=disp_s)
            sb = td.textbbox((0, 0), disp_s, font=f_sub)
            td.text(((width - (sb[2] - sb[0])) // 2, line_y + max(5, int(round(8 * scale_factor)))), disp_s, font=f_sub, fill=(176, 174, 168, 225))
    return {
        "photo_over": photo_over,
        "txt_rgba": txt_bar,
        "bar_h_target": bar_h_target,
    }


def _build_prism_glass_refraction_assets(
    raw_primary: Image.Image,
    width: int,
    height: int,
    scale_factor: float,
) -> dict[str, Any]:
    """#08 Handheld K9 crystal prism subtle outer-edge optical refraction bevel (center 80% untouched)."""
    ov_w, ov_h = max(64, int(round(width * 1.12))), max(64, int(round(height * 1.12)))
    base_over = ImageOps.fit(raw_primary, (ov_w, ov_h), method=Image.Resampling.LANCZOS)
    soft_over = base_over.filter(ImageFilter.GaussianBlur(radius=max(3, int(round(6 * scale_factor)))))
    return {"base_over": base_over, "soft_over": soft_over}


def _build_venetian_sunbeam_assets(
    raw_primary: Image.Image,
    width: int,
    height: int,
) -> dict[str, Any]:
    """#09 Architectural golden-hour diagonal slatted sunbeams + microscopic floating dust motes."""
    ov_w, ov_h = max(64, int(round(width * 1.10))), max(64, int(round(height * 1.10)))
    base_over = ImageOps.fit(raw_primary, (ov_w, ov_h), method=Image.Resampling.LANCZOS)
    rng = np.random.default_rng(909)
    n_motes = 36
    motes = {
        "mx0": rng.uniform(0.08, 0.92, n_motes).astype(np.float32),
        "my0": rng.uniform(0.10, 0.90, n_motes).astype(np.float32),
        "m_rad": rng.uniform(0.0024, 0.0054, n_motes).astype(np.float32),
        "m_ph": rng.uniform(0.0, 6.2831853, n_motes).astype(np.float32),
    }
    return {"base_over": base_over, "motes": motes}


def _build_vintage_postcard_deck_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
    focal_uv: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """#10 Playful tactile European linen travel postcards sliding apart smoothly with margin postmark."""
    p1, p2 = _load_companion_photos(raw_primary, kb, width, height, color_grade_fx, focal_uv, count=2)
    desk = _make_textured_surface_rgb(width, height, (38, 34, 30), seed=1010, grad_top=4.0, grad_bot=-5.0, grain_std=2.0)

    def _make_card(im: Image.Image, caption: str, angle_deg: float, stamp_tag: str) -> Image.Image:
        pw = max(64, int(round(width * 0.44)))
        ph = max(64, int(round(height * 0.58)))
        b_side = max(6, int(round(16 * scale_factor)))
        b_bot = max(18, int(round(48 * scale_factor)))
        fg = ImageOps.fit(im, (max(16, pw - b_side * 2), max(16, ph - b_side - b_bot)), method=Image.Resampling.LANCZOS)
        card = _make_textured_surface_rgb(pw, ph, (236, 230, 216), seed=1011, grad_top=2.0, grad_bot=-2.0, grain_std=1.4).convert("RGBA")
        cd = ImageDraw.Draw(card)
        cd.rectangle([0, 0, pw - 1, ph - 1], outline=(190, 180, 162, 255), width=max(1, int(round(1.5 * scale_factor))))
        cd.rectangle([b_side - 1, b_side - 1, b_side + fg.width, b_side + fg.height], outline=(168, 158, 142, 220), width=1)
        card.paste(fg.convert("RGBA"), (b_side, b_side))
        if caption:
            f_ct = _load_editorial_font(max(9, int(round(15 * scale_factor))), bold=True, serif=True, text=caption)
            disp_c = caption if len(caption) <= 34 else (caption[:31].rstrip() + "...")
            cd.text((b_side, ph - b_bot + max(4, int(round(14 * scale_factor)))), disp_c, font=f_ct, fill=(52, 46, 40, 235))
        # Subtle miniature circular postmark strictly in lower-right margin (zero overlap on photo)
        rx = pw - b_side - max(10, int(round(20 * scale_factor)))
        ry = ph - b_bot // 2
        rr = max(7, int(round(15 * scale_factor)))
        cd.ellipse([rx - rr, ry - rr, rx + rr, ry + rr], outline=(156, 78, 66, 175), width=max(1, int(round(1.4 * scale_factor))))
        f_pm = _load_editorial_font(max(8, int(round(10 * scale_factor))), bold=True, serif=False, text=stamp_tag)
        sb = cd.textbbox((0, 0), stamp_tag, font=f_pm)
        cd.text((rx - (sb[2] - sb[0]) // 2, ry - (sb[3] - sb[1]) // 2 - 1), stamp_tag, font=f_pm, fill=(156, 78, 66, 185))
        sm = max(12, int(round(36 * scale_factor)))
        wrap = Image.new("RGBA", (pw + sm * 2, ph + sm * 2), (0, 0, 0, 0))
        ImageDraw.Draw(wrap).rectangle([sm + 3, sm + max(4, int(round(10 * scale_factor))), sm + pw - 3, sm + ph + max(4, int(round(10 * scale_factor)))], fill=(0, 0, 0, 150))
        wrap = wrap.filter(ImageFilter.GaussianBlur(radius=max(5, int(round(15 * scale_factor)))))
        wrap.paste(card, (sm, sm), card)
        return wrap.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True)

    c1_cap = (kb.caption_title or "").strip()
    c2_cap = (kb.caption_subtitle or "").strip()
    return {
        "desk_rgb": desk,
        "card1_rgba": _make_card(p1, c1_cap, -2.5, "01"),
        "card2_rgba": _make_card(p2, c2_cap, 2.7, "02"),
    }


def _build_viewfinder_shutter_snap_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
    pal: dict[str, Any],
) -> dict[str, Any]:
    """#11 Optical camera viewfinder framing lock -> 85ms dark mechanical shutter snap -> gallery matted print."""
    ov_w, ov_h = max(64, int(round(width * 1.14))), max(64, int(round(height * 1.14)))
    live_over = ImageOps.fit(raw_primary, (ov_w, ov_h), method=Image.Resampling.LANCZOS)

    vf = Image.new("RGBA", (width, height), (238, 240, 244, 0))
    vd = ImageDraw.Draw(vf)
    mx, my = int(round(width * 0.07)), int(round(height * 0.08))
    cl = max(12, int(round(30 * scale_factor)))
    col = (242, 244, 248, 220)
    lw = max(1, int(round(2.0 * scale_factor)))
    for x, y, dx, dy in [(mx, my, 1, 1), (width - mx, my, -1, 1), (mx, height - my, 1, -1), (width - mx, height - my, -1, -1)]:
        vd.line([(x, y), (x + dx * cl, y)], fill=col, width=lw)
        vd.line([(x, y), (x, y + dy * cl)], fill=col, width=lw)
    fcx, fcy = width // 2, int(height * 0.47)
    fr = max(10, int(round(24 * scale_factor)))
    vd.rectangle([fcx - fr, fcy - fr, fcx + fr, fcy + fr], outline=(110, 232, 165, 225), width=lw)
    hud_txt = "M   1/500   f/2.8   ISO 100   [ AF-S LOCKED ]"
    f_hud = _load_editorial_font(max(9, int(round(14 * scale_factor))), bold=True, serif=False, text=hud_txt)
    hb = vd.textbbox((0, 0), hud_txt, font=f_hud)
    vd.text(((width - (hb[2] - hb[0])) // 2, height - my + max(4, int(round(12 * scale_factor)))), hud_txt, font=f_hud, fill=(238, 240, 244, 230))

    mat_bg = pal["bg"] if pal["is_light"] else (224, 220, 212)
    mat = _make_textured_surface_rgb(width, height, mat_bg, seed=1111, grad_top=3.0, grad_bot=-4.0, grain_std=1.5)
    card = _build_double_matte_card(
        raw_primary,
        max_w=int(width * 0.72),
        max_h=int(height * 0.76),
        scale_factor=scale_factor,
        border_px=max(8, int(round(16 * scale_factor))),
    )
    mat.paste(card, ((width - card.width) // 2, max(8, int(height * 0.47) - card.height // 2)), card)
    cap_txt = (kb.caption_title or "").strip()
    if cap_txt:
        md = ImageDraw.Draw(mat)
        f_tit = _load_editorial_font(max(11, int(round(20 * scale_factor))), bold=True, serif=True, text=cap_txt)
        cb = md.textbbox((0, 0), cap_txt, font=f_tit)
        md.text(((width - (cb[2] - cb[0])) // 2, int(height * 0.89)), cap_txt, font=f_tit, fill=(38, 36, 34))

    return {"live_over": live_over, "vf_rgba": vf, "mat_rgb": mat}


def _build_accordion_leporello_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
    focal_uv: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """#12 Japanese/Swiss 3-panel concertina Leporello folding photo book unfolding smoothly in 3D."""
    imgs = _load_companion_photos(raw_primary, kb, width, height, color_grade_fx, focal_uv, count=3)
    bg = _make_textured_surface_rgb(width, height, (26, 28, 34), seed=1212, grad_top=3.0, grad_bot=-4.0, grain_std=1.6)
    pw = max(48, int(round(width * 0.27)))
    ph = max(48, int(round(height * 0.66)))
    b = max(6, int(round(14 * scale_factor)))
    panels: list[Image.Image] = []
    for im in imgs:
        fg = ImageOps.fit(im, (max(16, pw - b * 2), max(16, ph - b * 2)), method=Image.Resampling.LANCZOS)
        p = _make_textured_surface_rgb(pw, ph, (236, 232, 224), seed=1213, grad_top=2.0, grad_bot=-2.0, grain_std=1.2).convert("RGBA")
        ImageDraw.Draw(p).rectangle([0, 0, pw - 1, ph - 1], outline=(184, 178, 168, 255), width=max(1, int(round(1.5 * scale_factor))))
        p.paste(fg.convert("RGBA"), (b, b))
        panels.append(p)
    return {"bg_rgb": bg, "panels_rgba": panels, "pw": pw, "ph": ph}


def _build_film_strip_sprocket_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
    focal_uv: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """#13 Vertical 35mm cinema film strip with rounded dual-side sprockets gliding through a projector gate."""
    imgs = _load_companion_photos(raw_primary, kb, width, height, color_grade_fx, focal_uv, count=3)
    strip_w = max(96, int(round(width * 0.72)))
    sproc_w = max(24, int(round(78 * scale_factor)))
    inner_w = max(48, strip_w - sproc_w * 2)
    frame_h = max(36, int(round(inner_w * 0.66)))
    gap_h = max(14, int(round(32 * scale_factor)))
    strip_h = len(imgs) * (frame_h + gap_h) + gap_h

    strip = Image.new("RGB", (strip_w, strip_h), (12, 13, 17))
    sd = ImageDraw.Draw(strip)
    sp_w = max(8, int(round(24 * scale_factor)))
    sp_h = max(12, int(round(34 * scale_factor)))
    sp_pitch = max(22, int(round(68 * scale_factor)))
    for sy_p in range(max(6, int(round(16 * scale_factor))), max(1, strip_h - sp_h), sp_pitch):
        for sx_p in (max(4, int(round(20 * scale_factor))), strip_w - max(4, int(round(20 * scale_factor))) - sp_w):
            sd.rounded_rectangle(
                [sx_p, sy_p, sx_p + sp_w, sy_p + sp_h],
                radius=max(2, int(round(5 * scale_factor))),
                fill=(38, 40, 46),
                outline=(58, 60, 68),
                width=1,
            )

    f_code = _load_editorial_font(max(8, int(round(12 * scale_factor))), bold=True, serif=False)
    for i, im in enumerate(imgs):
        fy = gap_h + i * (frame_h + gap_h)
        fit_im = ImageOps.fit(im, (inner_w, frame_h), method=Image.Resampling.LANCZOS)
        strip.paste(fit_im, (sproc_w, fy))
        sd.rectangle([sproc_w - 1, fy - 1, sproc_w + inner_w, fy + frame_h], outline=(48, 50, 58), width=1)
        sd.text((sproc_w + 6, fy - max(10, int(round(22 * scale_factor)))), f"KODAK VISION3 500T   ·   FRAME {34 + i}A", font=f_code, fill=(214, 162, 74))

    full_strip = Image.new("RGB", (width, strip_h), (9, 10, 14))
    full_strip.paste(strip, ((width - strip_w) // 2, 0))
    return {"strip_rgb": full_strip, "strip_h": strip_h}


def _build_museum_acrylic_float_assets(
    raw_primary: Image.Image,
    width: int,
    height: int,
    scale_factor: float,
    pal: dict[str, Any],
) -> dict[str, Any]:
    """#14 Frameless Diasec gallery Perspex acrylic float mount with brass standoffs & swinging cast shadow."""
    wall_col = pal["bg"] if pal["is_light"] else (218, 214, 206)
    wall = _make_textured_surface_rgb(width, height, wall_col, seed=1414, grad_top=4.0, grad_bot=-5.0, grain_std=1.8)
    fg = raw_primary.copy()
    fg.thumbnail((max(48, int(round(width * 0.64))), max(48, int(round(height * 0.74)))), Image.Resampling.LANCZOS)
    pw, ph = fg.width, fg.height

    acrylic = fg.convert("RGBA")
    ad = ImageDraw.Draw(acrylic)
    ad.rectangle([0, 0, pw - 1, ph - 1], outline=(244, 244, 244, 175), width=max(1, int(round(2.2 * scale_factor))))
    st_off = max(6, int(round(16 * scale_factor)))
    st_r = max(2, int(round(4.5 * scale_factor)))
    for sx_p, sy_p in [(st_off, st_off), (pw - st_off, st_off), (st_off, ph - st_off), (pw - st_off, ph - st_off)]:
        ad.ellipse([sx_p - st_r, sy_p - st_r, sx_p + st_r, sy_p + st_r], fill=(198, 178, 134, 235), outline=(92, 82, 62, 220), width=1)

    sm = max(14, int(round(44 * scale_factor)))
    sh_im = Image.new("RGBA", (pw + sm * 2, ph + sm * 2), (0, 0, 0, 0))
    ImageDraw.Draw(sh_im).rectangle([sm, sm, sm + pw, sm + ph], fill=(12, 12, 16, 148))
    sh_im = sh_im.filter(ImageFilter.GaussianBlur(radius=max(6, int(round(19 * scale_factor)))))

    return {
        "wall_rgb": wall,
        "acrylic_rgba": acrylic,
        "shadow_rgba": sh_im,
        "px0": (width - pw) // 2,
        "py0": (height - ph) // 2,
        "pw": pw,
        "ph": ph,
        "sm": sm,
    }


def _build_kinetic_triptych_reveal_assets(
    raw_primary: Image.Image,
    kb: KenBurnsConfig,
    width: int,
    height: int,
    scale_factor: float,
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
    focal_uv: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """#15 Energetic 3-beat staggered vertical editorial column reveal with counter-directional motion."""
    imgs = _load_companion_photos(raw_primary, kb, width, height, color_grade_fx, focal_uv, count=3)
    gutter = max(4, int(round(9 * scale_factor)))
    w1 = max(24, int(round((width - gutter * 2) * 0.36)))
    w2 = max(24, int(round((width - gutter * 2) * 0.32)))
    w3 = max(24, width - gutter * 2 - w1 - w2)
    col_ws = [w1, w2, w3]
    xs = [0, w1 + gutter, w1 + gutter + w2 + gutter]
    cols_over = [
        ImageOps.fit(im, (max(32, int(round(cw * 1.14))), max(32, int(round(height * 1.14)))), method=Image.Resampling.LANCZOS)
        for im, cw in zip(imgs, col_ws)
    ]
    return {"cols_over": cols_over, "col_ws": col_ws, "xs": xs}


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

    if overlay_fx == "halation_bloom_35mm":
        return {"type": "halation_bloom_35mm"}

    if overlay_fx == "caustic_water_shimmer":
        # Oversampled dual-band sunlit water caustic shimmer for animated dual-phase lateral drift
        dw = int(round(width * 1.24))
        csw, csh = max(64, dw // 4), max(36, height // 4)
        yy_c, xx_c = np.mgrid[0:csh, 0:csw].astype(np.float32)
        nx_c = xx_c / max(1.0, float(csw - 1))
        ny_c = yy_c / max(1.0, float(csh - 1))
        ray1 = np.exp(-((nx_c * 0.72 + ny_c * 0.69 - 0.34) ** 2) / 0.018)
        ray2 = np.exp(-((nx_c * 0.68 + ny_c * 0.73 - 0.58) ** 2) / 0.024) * 0.78
        ripple = 0.72 + 0.28 * np.sin(nx_c * math.pi * 6.5 + ny_c * 3.4) * np.cos(ny_c * math.pi * 4.2 - nx_c * 2.2)
        falloff = np.clip(1.0 - ny_c * 0.88, 0.0, 1.0) * np.clip(0.28 + nx_c * 0.72, 0.0, 1.0)
        beams = np.clip((ray1 + ray2) * ripple * falloff, 0.0, 1.0)
        c_rgb = np.zeros((csh, csw, 3), dtype=np.uint8)
        c_rgb[:, :, 0] = (beams * 255.0).astype(np.uint8)
        c_rgb[:, :, 1] = (beams * 228.0).astype(np.uint8)
        c_rgb[:, :, 2] = (beams * 176.0).astype(np.uint8)
        shimmer_over = (
            Image.fromarray(c_rgb, mode="RGB")
            .filter(ImageFilter.GaussianBlur(radius=6))
            .resize((dw, height), Image.Resampling.BILINEAR)
        )
        return {"type": "caustic_water_shimmer", "shimmer_over": shimmer_over, "dw": dw}

    if overlay_fx == "architectural_window_dapple":
        dw = int(round(width * 1.20))
        dsw = max(64, dw // 6)
        dsh = max(36, height // 6)
        yy_d, xx_d = np.mgrid[0:dsh, 0:dsw]
        nx_d = xx_d / max(1.0, float(dsw - 1))
        ny_d = yy_d / max(1.0, float(dsh - 1))
        u = nx_d * 0.76 + ny_d * 0.65
        louvers = 0.5 + 0.5 * np.cos(u * math.pi * 5.2 - 0.8)
        foliage = 0.78 + 0.22 * np.sin(nx_d * math.pi * 4.5 + ny_d * 2.1) * np.cos(ny_d * math.pi * 3.8)
        sun_field = np.clip(louvers * foliage, 0.0, 1.0)
        shade_field = np.clip((1.0 - louvers) * (0.72 + 0.28 * ny_d), 0.0, 1.0)
        sun_mask_over = (
            Image.fromarray((sun_field * 22.0).astype(np.uint8), mode="L")
            .filter(ImageFilter.GaussianBlur(radius=6))
            .resize((dw, height), Image.Resampling.BILINEAR)
        )
        shade_mask_over = (
            Image.fromarray((shade_field * 24.0).astype(np.uint8), mode="L")
            .filter(ImageFilter.GaussianBlur(radius=6))
            .resize((dw, height), Image.Resampling.BILINEAR)
        )
        return {
            "type": "architectural_window_dapple",
            "dw": dw,
            "sun_mask_over": sun_mask_over,
            "shade_mask_over": shade_mask_over,
            "warm_rgb": Image.new("RGB", (width, height), (244, 232, 206)),
            "shade_rgb": Image.new("RGB", (width, height), (14, 16, 22)),
        }

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

    e_fx = _ease_in_out_quintic(t_norm)
    if fx_type == "golden_light_leak":
        # Black-preserving optical screen blend (shadows stay deep, warm amber glows naturally)
        screened = ImageChops.screen(frame_rgb, fx_assets["leak_img"])
        weight = 0.22 + 0.14 * math.sin(math.pi * e_fx)
        return Image.blend(frame_rgb, screened, weight)

    if fx_type == "museum_spotlight":
        frame_rgb.paste(fx_assets["dark_rgb"], (0, 0), fx_assets["mask_l"])
        return frame_rgb

    if fx_type == "champagne_bokeh":
        # Soft elliptical anamorphic optical bokeh disks (zero harsh outline strokes, black-preserving screen blend)
        w, h = frame_rgb.size
        sw, sh = max(64, w // 4), max(36, h // 4)
        yy_b, xx_b = np.mgrid[0:sh, 0:sw].astype(np.float32)
        acc = np.zeros((sh, sw), dtype=np.float32)
        for ox, oy, rad, alpha, phase in fx_assets["orbs"]:
            cx = ((ox + 0.028 * math.sin(phase + e_fx * 3.6)) % 1.0) * sw
            cy = ((oy - 0.085 * e_fx) % 1.0) * sh
            rx_s = max(3.0, float(rad) / 4.2)
            ry_s = rx_s * 1.24  # subtle Cooke anamorphic oval aperture ratio
            pulse = (alpha / 255.0) * (0.72 + 0.28 * math.sin(phase + e_fx * 5.2))
            r2 = ((xx_b - cx) / rx_s) ** 2 + ((yy_b - cy) / ry_s) ** 2
            disk = np.clip(1.0 - r2, 0.0, 1.0)
            acc += (disk * disk) * pulse * 0.85
        acc = np.clip(acc, 0.0, 1.0)
        bokeh_rgb = np.zeros((sh, sw, 3), dtype=np.uint8)
        bokeh_rgb[:, :, 0] = np.clip(acc * 255.0, 0, 255).astype(np.uint8)
        bokeh_rgb[:, :, 1] = np.clip(acc * 224.0, 0, 255).astype(np.uint8)
        bokeh_rgb[:, :, 2] = np.clip(acc * 150.0, 0, 255).astype(np.uint8)
        bokeh_full = (
            Image.fromarray(bokeh_rgb, mode="RGB")
            .filter(ImageFilter.GaussianBlur(radius=2))
            .resize((w, h), Image.Resampling.BILINEAR)
        )
        return ImageChops.screen(frame_rgb, bokeh_full)

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

    if fx_type == "halation_bloom_35mm":
        w, h = frame_rgb.size
        sw, sh = max(48, w // 6), max(27, h // 6)
        small_arr = np.asarray(frame_rgb.resize((sw, sh), Image.Resampling.BILINEAR), dtype=np.float32)
        luma = 0.299 * small_arr[:, :, 0] + 0.587 * small_arr[:, :, 1] + 0.114 * small_arr[:, :, 2]
        hi = np.clip((luma - 206.0) / 49.0, 0.0, 1.0)
        hi_u8 = (hi * 255.0).astype(np.uint8)
        halo_s = np.asarray(Image.fromarray(hi_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=4)), dtype=np.float32) / 255.0
        halo_rgb = np.zeros((sh, sw, 3), dtype=np.uint8)
        halo_rgb[:, :, 0] = np.clip(halo_s * 255.0, 0, 255).astype(np.uint8)
        halo_rgb[:, :, 1] = np.clip(halo_s * 124.0, 0, 255).astype(np.uint8)
        halo_rgb[:, :, 2] = np.clip(halo_s * 68.0, 0, 255).astype(np.uint8)
        halo_full = Image.fromarray(halo_rgb, mode="RGB").resize((w, h), Image.Resampling.BILINEAR)
        screened = ImageChops.screen(frame_rgb, halo_full)
        weight = 0.24 + 0.08 * math.sin(math.pi * t_norm)
        return Image.blend(frame_rgb, screened, weight)

    if fx_type == "caustic_water_shimmer":
        w, h = frame_rgb.size
        dw = int(fx_assets["dw"])
        sh_over = fx_assets["shimmer_over"]
        x0_a = (dw - w) * e_fx
        x0_b = (dw - w) * (1.0 - e_fx)
        crop_a = sh_over.transform((w, h), Image.Transform.EXTENT, (x0_a, 0.0, x0_a + float(w), float(h)), resample=Image.Resampling.BILINEAR)
        crop_b = sh_over.transform((w, h), Image.Transform.EXTENT, (x0_b, 0.0, x0_b + float(w), float(h)), resample=Image.Resampling.BILINEAR)
        anim_shimmer = Image.blend(crop_a, crop_b, 0.38)
        screened = ImageChops.screen(frame_rgb, anim_shimmer)
        weight = 0.11 + 0.055 * math.sin(math.pi * e_fx)
        return Image.blend(frame_rgb, screened, weight)

    if fx_type == "architectural_window_dapple":
        w, h = frame_rgb.size
        dw = int(fx_assets["dw"])
        x0 = (dw - w) * (1.0 - e_fx)
        sun_m = fx_assets["sun_mask_over"].transform(
            (w, h),
            Image.Transform.EXTENT,
            (x0, 0.0, x0 + float(w), float(h)),
            resample=Image.Resampling.BILINEAR,
        )
        shade_m = fx_assets["shade_mask_over"].transform(
            (w, h),
            Image.Transform.EXTENT,
            (x0, 0.0, x0 + float(w), float(h)),
            resample=Image.Resampling.BILINEAR,
        )
        frame_rgb.paste(fx_assets["warm_rgb"], (0, 0), sun_m)
        frame_rgb.paste(fx_assets["shade_rgb"], (0, 0), shade_m)
        return frame_rgb

    return frame_rgb



def _prepare_photo_scene_context(
    photo_path: str | Path,
    width: int,
    height: int,
    kb: KenBurnsConfig,
    color_grade_fx: Optional[EffectTrackItem | dict[str, Any]] = None,
) -> dict[str, Any]:
    """Pre-build all multi-plane assets for all active photo presentation modes + atmospheric finishers."""
    photo_max_dim = max(512, (max(width, height) * 5) // 4)
    raw_primary = _load_graded_photo_rgb(photo_path, color_grade_fx, max_dim=photo_max_dim)
    scale_factor = width / 1920.0
    s_zoom, e_zoom, (sx, sy), (ex, ey) = _resolve_kb_endpoints(kb)
    resample_mode = Image.Resampling.BILINEAR if width >= 3840 else Image.Resampling.BICUBIC

    # Route disabled neural-depth modes cleanly to crisp parallax/perspective alternatives
    mode = DISABLED_PHOTO_MODE_FALLBACKS.get(kb.mode, kb.mode)
    if mode == "links_white_matte":
        s_zoom = 1.0
        e_zoom = min(1.018, max(1.0, e_zoom))
        sx, sy, ex, ey = 0.5, 0.5, 0.5, 0.5
    # Modes with native integrated editorial/museum typography never overlay floating dark lower-third cards
    _integrated_caption_modes = (
        "editorial_magazine",
        "links_white_matte",
        "kinfolk_linen_folio",
        "passepartout_shadowbox",
        "swiss_monograph_grid",
        "editorial_bookspread",
        "gallery_wall_triptych",
        "archival_lightbox_strip",
        "cinema_gate_reveal",
        "vintage_postcard_deck",
        "viewfinder_shutter_snap",
    )
    has_card = bool(kb.caption_title and kb.caption_title.strip()) and (mode not in _integrated_caption_modes)
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
    bg_plane: Optional[Image.Image] = None
    if mode in _BG_PLANE_MODES:
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
            raw_sec = _load_graded_photo_rgb(sec_path, color_grade_fx, max_dim=photo_max_dim)
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
            color_grade_fx=color_grade_fx,
        )

    elif mode == "polaroid_scrapbook":
        stamp_1 = "" if has_card else (kb.caption_title or "")
        max_w = int(width * (0.42 if kb.secondary_source else 0.56))
        max_h = int(height * card_max_h_ratio)
        ctx["polaroid_1"] = _build_polaroid_card(
            raw_primary, max_w=max_w, max_h=max_h, scale_factor=scale_factor, angle_deg=-3.2, stamp_text=stamp_1
        )
        if kb.secondary_source and Path(kb.secondary_source).exists():
            raw_sec = _load_graded_photo_rgb(kb.secondary_source, color_grade_fx, max_dim=photo_max_dim)
            ctx["polaroid_2"] = _build_polaroid_card(
                raw_sec, max_w=max_w, max_h=max_h, scale_factor=scale_factor, angle_deg=3.6, stamp_text=""
            )
        else:
            ctx["polaroid_2"] = None

    elif mode in ("cinematic_depth_3d", "dolly_zoom_vertigo"):
        mpi = _build_subject_depth_layers(
            raw_img=raw_primary,
            bg_w=bg_w,
            bg_h=bg_h,
            focal_point=kb.start_center,
            scale_factor=scale_factor,
        )
        ctx["mpi_layers"] = mpi["layers"]
        ctx["parallax_amp"] = float(mpi["suitability"])

    elif mode == "aperture_rack_focus":
        rf = build_rack_focus_assets(
            raw_primary=raw_primary,
            width=width,
            height=height,
            focal_point=kb.start_center,
            scale_factor=scale_factor,
        )
        ctx["rf_sharp"] = rf["sharp"]
        ctx["rf_bokeh"] = rf["bokeh"]
        ctx["rf_depth_q"] = rf["depth_q"]
        ctx["rf_focus_start"] = rf["focus_start"]
        ctx["rf_focus_end"] = rf["focus_end"]

    elif mode == "archival_color_bloom":
        oversample = 1.07
        work_w = max(64, int(round(width * oversample)))
        work_h = max(64, int(round(height * oversample)))
        color_base = ImageOps.fit(raw_primary, (work_w, work_h), method=Image.Resampling.BILINEAR)
        # Selenium-toned high-contrast silver-gelatin B&W master
        bw_l = ImageEnhance.Contrast(color_base.convert("L")).enhance(1.08)
        bw_arr = np.asarray(bw_l, dtype=np.float32)
        bw_rgb = np.stack(
            [
                np.clip(bw_arr * 1.01, 0, 255).astype(np.uint8),
                np.clip(bw_arr * 1.00, 0, 255).astype(np.uint8),
                np.clip(bw_arr * 0.97, 0, 255).astype(np.uint8),
            ],
            axis=-1,
        )
        bw_img = Image.fromarray(bw_rgb, mode="RGB")
        # Depth-free optical focal + smoothed luminance tonal reveal map (zero silhouette halos)
        qw, qh = max(48, work_w // 4), max(27, work_h // 4)
        luma_q = (
            np.asarray(
                color_base.convert("L")
                .resize((qw, qh), Image.Resampling.BILINEAR)
                .filter(ImageFilter.GaussianBlur(radius=4)),
                dtype=np.float32,
            )
            / 255.0
        )
        yy_b, xx_b = np.mgrid[0:qh, 0:qw].astype(np.float32)
        nx_q = xx_b / max(1.0, float(qw - 1))
        ny_q = yy_b / max(1.0, float(qh - 1))
        fx_c = max(0.20, min(0.80, float(kb.start_center[0])))
        fy_c = max(0.20, min(0.80, float(kb.start_center[1])))
        dist_norm = np.sqrt(((nx_q - fx_c) / 0.62) ** 2 + ((ny_q - fy_c) / 0.62) ** 2)
        radial_prox = np.clip(1.0 - 0.72 * dist_norm, 0.0, 1.0)
        raw_field = 0.72 * radial_prox + 0.28 * luma_q
        f_min, f_max = float(raw_field.min()), float(raw_field.max())
        norm_field = (raw_field - f_min) / max(1e-5, f_max - f_min)
        depth_q = (0.16 + 0.84 * norm_field).astype(np.float32)
        ctx["bloom_color"] = color_base
        ctx["bloom_bw"] = bw_img
        ctx["bloom_depth_q"] = depth_q

    elif mode == "curator_loupe":
        loupe_hint = kb.loupe_focus_point or (
            kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center
        )
        salient_uv = _detect_photo_salient_point(raw_primary, loupe_hint, use_depth=True)
        dimmed_c, sharp_c, loupe_mask, loupe_bezel, rad_px, s_xy, e_xy = _build_curator_loupe_assets(
            raw_primary=raw_primary,
            width=width,
            height=height,
            has_card=has_card,
            scale_factor=scale_factor,
            focal_uv=salient_uv,
        )
        ctx["loupe_dimmed"] = dimmed_c
        ctx["loupe_sharp"] = sharp_c
        ctx["loupe_mask"] = loupe_mask
        ctx["loupe_bezel"] = loupe_bezel
        ctx["loupe_radius"] = rad_px
        ctx["loupe_start_xy"] = s_xy
        ctx["loupe_end_xy"] = e_xy

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
            color_grade_fx=color_grade_fx,
        )
        ctx["contact_sheet"] = sheet_rgb
        ctx["contact_hero_xy"] = (hero_cx, hero_cy)

    elif mode == "split_diopter_focus":
        sd_hint = kb.loupe_focus_point or (
            kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center
        )
        salient_uv = _detect_photo_salient_point(raw_primary, sd_hint, use_depth=True)
        sd_left, sd_right, sd_seam, sd_left_w = _build_split_diopter_assets(
            raw_primary=raw_primary,
            focal_point=salient_uv,
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
        salient_uv = _detect_photo_salient_point(
            raw_primary,
            kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center,
        )
        ctx["atelier"] = _build_criterion_atelier_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
            color_grade_fx=color_grade_fx,
            focal_uv=salient_uv,
        )

    elif mode == "passepartout_shadowbox":
        pal = _resolve_surface_palette(raw_primary, kb)
        ctx["surface_palette"] = pal["name"]
        ctx["shadowbox"] = _build_passepartout_shadowbox_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
            pal=pal,
        )

    elif mode == "kinfolk_linen_folio":
        pal = _resolve_surface_palette(raw_primary, kb)
        ctx["surface_palette"] = pal["name"]
        ctx["kinfolk_folio"] = _build_kinfolk_linen_folio_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
            pal=pal,
        )

    elif mode == "panoramic_frieze_scroll":
        pal = _resolve_surface_palette(raw_primary, kb)
        ctx["surface_palette"] = pal["name"]
        if pal["is_light"]:
            ctx["card_tint_weight"] = 0.78
        salient_uv = _detect_photo_salient_point(
            raw_primary,
            kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center,
        )
        ctx["frieze"] = _build_panoramic_frieze_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            has_card=has_card,
            scale_factor=scale_factor,
            pal=pal,
            color_grade_fx=color_grade_fx,
            focal_uv=salient_uv,
        )

    elif mode == "atelier_layered_duo":
        pal = _resolve_surface_palette(raw_primary, kb)
        ctx["surface_palette"] = pal["name"]
        if pal["is_light"]:
            ctx["card_tint_weight"] = 0.78
        salient_uv = _detect_photo_salient_point(
            raw_primary,
            kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center,
        )
        ctx["layered_duo"] = _build_atelier_layered_duo_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            has_card=has_card,
            scale_factor=scale_factor,
            pal=pal,
            color_grade_fx=color_grade_fx,
            focal_uv=salient_uv,
        )

    elif mode == "leica_frame_finder":
        ctx["leica_finder"] = _build_leica_frame_finder_assets(
            raw_primary=raw_primary,
            width=width,
            height=height,
            has_card=has_card,
            scale_factor=scale_factor,
        )

    elif mode == "swiss_monograph_grid":
        pal = _resolve_surface_palette(raw_primary, kb)
        ctx["surface_palette"] = pal["name"]
        salient_uv = _detect_photo_salient_point(
            raw_primary, kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center
        )
        ctx["swiss_grid"] = _build_swiss_monograph_grid_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
            pal=pal,
            color_grade_fx=color_grade_fx,
            focal_uv=salient_uv,
        )

    elif mode == "editorial_bookspread":
        salient_uv = _detect_photo_salient_point(
            raw_primary, kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center
        )
        ctx["bookspread"] = _build_editorial_bookspread_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
            color_grade_fx=color_grade_fx,
            focal_uv=salient_uv,
        )

    elif mode == "gallery_wall_triptych":
        pal = _resolve_surface_palette(raw_primary, kb)
        ctx["surface_palette"] = pal["name"]
        salient_uv = _detect_photo_salient_point(
            raw_primary, kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center
        )
        ctx["gallery_wall"] = _build_gallery_wall_triptych_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
            pal=pal,
            color_grade_fx=color_grade_fx,
            focal_uv=salient_uv,
        )

    elif mode == "archival_lightbox_strip":
        salient_uv = _detect_photo_salient_point(
            raw_primary, kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center
        )
        ctx["lightbox_strip"] = _build_archival_lightbox_strip_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
            color_grade_fx=color_grade_fx,
            focal_uv=salient_uv,
        )

    elif mode == "optical_tilt_shift":
        ctx["tilt_shift"] = _build_optical_tilt_shift_assets(
            raw_primary=raw_primary,
            width=width,
            height=height,
            scale_factor=scale_factor,
        )

    elif mode == "darkroom_developer_reveal":
        salient_uv = _detect_photo_salient_point(
            raw_primary, kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center
        )
        ctx["darkroom_dev"] = _build_darkroom_developer_assets(
            raw_primary=raw_primary,
            width=width,
            height=height,
            focal_uv=salient_uv,
        )

    elif mode == "cinema_gate_reveal":
        ctx["cinema_gate"] = _build_cinema_gate_reveal_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
        )

    elif mode == "prism_glass_refraction":
        ctx["prism_glass"] = _build_prism_glass_refraction_assets(
            raw_primary=raw_primary,
            width=width,
            height=height,
            scale_factor=scale_factor,
        )

    elif mode == "venetian_sunbeam_pan":
        ctx["venetian"] = _build_venetian_sunbeam_assets(
            raw_primary=raw_primary,
            width=width,
            height=height,
        )

    elif mode == "vintage_postcard_deck":
        salient_uv = _detect_photo_salient_point(
            raw_primary, kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center
        )
        ctx["postcard_deck"] = _build_vintage_postcard_deck_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
            color_grade_fx=color_grade_fx,
            focal_uv=salient_uv,
        )

    elif mode == "viewfinder_shutter_snap":
        pal = _resolve_surface_palette(raw_primary, kb)
        ctx["surface_palette"] = pal["name"]
        ctx["shutter_snap"] = _build_viewfinder_shutter_snap_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
            pal=pal,
        )

    elif mode == "accordion_leporello_fold":
        salient_uv = _detect_photo_salient_point(
            raw_primary, kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center
        )
        ctx["leporello"] = _build_accordion_leporello_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
            color_grade_fx=color_grade_fx,
            focal_uv=salient_uv,
        )

    elif mode == "film_strip_sprocket_glide":
        salient_uv = _detect_photo_salient_point(
            raw_primary, kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center
        )
        ctx["film_sprocket"] = _build_film_strip_sprocket_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
            color_grade_fx=color_grade_fx,
            focal_uv=salient_uv,
        )

    elif mode == "museum_acrylic_float":
        pal = _resolve_surface_palette(raw_primary, kb)
        ctx["surface_palette"] = pal["name"]
        ctx["acrylic_float"] = _build_museum_acrylic_float_assets(
            raw_primary=raw_primary,
            width=width,
            height=height,
            scale_factor=scale_factor,
            pal=pal,
        )

    elif mode == "kinetic_triptych_reveal":
        salient_uv = _detect_photo_salient_point(
            raw_primary, kb.end_center if kb.end_center != (0.5, 0.48) else kb.start_center
        )
        ctx["kinetic_triptych"] = _build_kinetic_triptych_reveal_assets(
            raw_primary=raw_primary,
            kb=kb,
            width=width,
            height=height,
            scale_factor=scale_factor,
            color_grade_fx=color_grade_fx,
            focal_uv=salient_uv,
        )

    else:
        oversample = 1.02 if mode == "links_white_matte" else (1.10 if width >= 3840 else 1.20)
        work_w = int(round(width * oversample))
        work_h = int(round(height * oversample))
        if mode == "links_white_matte":
            ctx["base_static_img"] = build_links_white_matte_image(
                photo_path,
                width=work_w,
                height=work_h,
                border_px=kb.border_px,
                raw_img=raw_primary,
                exhibition_tone="soft_limestone",
            )
        elif mode == "gallery_frame":
            ctx["base_static_img"] = build_gallery_frame_image(
                photo_path, width=work_w, height=work_h, border_px=kb.border_px, raw_img=raw_primary
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
    eased = _ease_in_out_quintic(t_norm)

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
        drift_px = (eased - 0.5) * 24.0 * scale_factor
        left_cx = width * 0.28
        right_cx = width * 0.72
        mid_cy = height * card_cy_ratio
        _paste_rgba_subpixel_pil(frame, fg_card_1, left_cx - fg_card_1.width * 0.5, mid_cy - fg_card_1.height * 0.5 - drift_px)
        _paste_rgba_subpixel_pil(frame, fg_card_2, right_cx - fg_card_2.width * 0.5, mid_cy - fg_card_2.height * 0.5 + drift_px)

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
        centers_x = [width * 0.20, width * 0.50, width * 0.80]
        mid_cy = height * card_cy_ratio
        drift_dirs = [-1.0, 0.85, -1.0]
        for p_i, p_card in enumerate(panels):
            # Velvet C2 quintic staggered entrance + soft alpha materialize (zero harsh pop-in)
            stagger_start = p_i * 0.06
            enter_p = _ease_in_out_quintic((t_norm - stagger_start) / 0.28) if t_norm > stagger_start else 0.0
            alpha_m = min(1.0, enter_p * 1.45)
            if alpha_m <= 0.01:
                continue
            slide_in_y = (1.0 - enter_p) * 32.0 * scale_factor
            drift_y = (eased - 0.5) * 20.0 * scale_factor * drift_dirs[p_i]
            px = centers_x[p_i] - p_card.width * 0.5
            py = max(8.0, mid_cy - p_card.height * 0.5 + drift_y + slide_in_y)
            _paste_rgba_subpixel_pil(frame, p_card, px, py, alpha_mul=alpha_m)

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
        mid_cy = height * card_cy_ratio
        if p2 is not None:
            # Dual Polaroid Scrapbook Stack with sub-pixel float drift & velvety C2 entrance
            drift = (eased - 0.5) * 18.0 * scale_factor
            x1 = width * 0.33 - p1.width * 0.5
            y1 = max(8.0, mid_cy - p1.height * 0.5 - drift)
            _paste_rgba_subpixel_pil(frame, p1, x1, y1)

            enter_2 = _ease_in_out_quintic((t_norm - 0.08) / 0.30) if t_norm > 0.08 else 0.0
            alpha_2 = min(1.0, enter_2 * 1.45)
            if alpha_2 > 0.01:
                slide_x = (1.0 - enter_2) * 44.0 * scale_factor
                x2 = width * 0.67 - p2.width * 0.5 + slide_x
                y2 = max(8.0, mid_cy - p2.height * 0.5 + drift)
                _paste_rgba_subpixel_pil(frame, p2, x2, y2, alpha_mul=alpha_2)
        else:
            scale_p = 0.96 + 0.05 * eased
            cw = max(1.0, p1.width * scale_p)
            ch = max(1.0, p1.height * scale_p)
            px = (width - cw) * 0.5 + (eased - 0.5) * 22.0 * scale_factor
            py = max(8.0, mid_cy - ch * 0.5)
            _paste_rgba_subpixel_pil(frame, p1, px, py, dst_w=cw, dst_h=ch)

    elif mode == "cinematic_depth_3d":
        mpi_layers: list[Image.Image] = ctx["mpi_layers"]
        amp = float(ctx.get("parallax_amp", 1.0))

        # Apple Spatial Photos restraint: unified gentle dolly + micro stereo orbital arc, zero counter-zoom
        base_zoom = 1.032 + 0.028 * eased
        arc_x = math.sin((eased - 0.5) * math.pi) * 0.014 * amp + (ex - sx) * 0.12 * amp * (eased - 0.5)
        arc_y = -0.004 * amp * math.cos((eased - 0.5) * math.pi) + (ey - sy) * 0.10 * amp * (eased - 0.5)

        frame = Image.new("RGB", (width, height))
        layer_specs = ((-0.55, 0.994), (-0.18, 1.000), (0.18, 1.008), (0.55, 1.016))
        for li, (d_mult, z_mul) in enumerate(layer_specs):
            layer_img = mpi_layers[li]
            lw, lh = layer_img.size
            cur_z = base_zoom * z_mul
            lcw = lw / cur_z
            lch = lh / cur_z
            lcx = 0.5 + d_mult * arc_x
            lcy = 0.5 + d_mult * arc_y
            lx0 = min(max(0.0, lcx * lw - lcw * 0.5), lw - lcw)
            ly0 = min(max(0.0, lcy * lh - lch * 0.5), lh - lch)
            cropped = layer_img.transform(
                (width, height),
                Image.Transform.EXTENT,
                (lx0, ly0, lx0 + lcw, ly0 + lch),
                resample=resample_mode,
            )
            if li == 0:
                frame = cropped.convert("RGB")
            else:
                frame.paste(cropped.convert("RGB"), (0, 0), cropped.split()[3])

    elif mode == "dolly_zoom_vertigo":
        mpi_layers = ctx["mpi_layers"]
        amp = float(ctx.get("parallax_amp", 1.0))
        v_mix = eased * (0.55 + 0.45 * amp)
        dx_v = (ex - sx) * 0.25 * eased
        dy_v = (ey - sy) * 0.20 * eased

        frame = Image.new("RGB", (width, height))
        vertigo_specs = (
            (1.045 - 0.030 * v_mix, -0.20),
            (1.035 - 0.020 * v_mix, -0.06),
            (1.026 - 0.011 * v_mix, 0.08),
            (1.018 + 0.004 * v_mix, 0.20),
        )
        for li, (cur_z, d_shift) in enumerate(vertigo_specs):
            layer_img = mpi_layers[li]
            lw, lh = layer_img.size
            lcw = lw / cur_z
            lch = lh / cur_z
            lcx = 0.5 + d_shift * dx_v
            lcy = 0.5 + d_shift * dy_v
            lx0 = min(max(0.0, lcx * lw - lcw * 0.5), lw - lcw)
            ly0 = min(max(0.0, lcy * lh - lch * 0.5), lh - lch)
            cropped = layer_img.transform(
                (width, height),
                Image.Transform.EXTENT,
                (lx0, ly0, lx0 + lcw, ly0 + lch),
                resample=resample_mode,
            )
            if li == 0:
                frame = cropped.convert("RGB")
            else:
                frame.paste(cropped.convert("RGB"), (0, 0), cropped.split()[3])

    elif mode == "aperture_rack_focus":
        rf_sharp: Image.Image = ctx["rf_sharp"]
        rf_bokeh: Image.Image = ctx["rf_bokeh"]
        rf_depth_q: np.ndarray = ctx["rf_depth_q"]
        f_start = float(ctx["rf_focus_start"])
        f_end = float(ctx["rf_focus_end"])

        rw, rh = rf_sharp.size
        cur_zoom = 1.01 + 0.04 * eased
        cx = 0.5 + (ex - sx) * 0.25 * eased
        cy = 0.5 + (ey - sy) * 0.25 * eased
        cw = rw / cur_zoom
        ch = rh / cur_zoom
        x0 = min(max(0.0, cx * rw - cw * 0.5), rw - cw)
        y0 = min(max(0.0, cy * rh - ch * 0.5), rh - ch)

        sharp_crop = rf_sharp.transform((width, height), Image.Transform.EXTENT, (x0, y0, x0 + cw, y0 + ch), resample=resample_mode)
        bokeh_crop = rf_bokeh.transform((width, height), Image.Transform.EXTENT, (x0, y0, x0 + cw, y0 + ch), resample=resample_mode)

        focus_p = _ease_in_out_quintic(min(1.0, t_norm / 0.85))
        focus_d = f_start + (f_end - f_start) * focus_p
        dist = np.abs(rf_depth_q - focus_d)
        lin = np.clip((dist - 0.06) / 0.24, 0.0, 1.0)
        blur_w_q = lin * lin * (3.0 - 2.0 * lin) * 0.85
        qh, qw = blur_w_q.shape
        qx0 = min(max(0.0, (x0 / rw) * qw), qw - 1.0)
        qy0 = min(max(0.0, (y0 / rh) * qh), qh - 1.0)
        qx1 = min(float(qw), qx0 + (cw / rw) * qw)
        qy1 = min(float(qh), qy0 + (ch / rh) * qh)
        blur_mask_img = Image.fromarray(np.clip(blur_w_q * 255.0, 0, 255).astype(np.uint8), mode="L").transform(
            (width, height),
            Image.Transform.EXTENT,
            (qx0, qy0, qx1, qy1),
            resample=Image.Resampling.BILINEAR,
        )
        frame = sharp_crop.copy()
        frame.paste(bokeh_crop, (0, 0), blur_mask_img)

    elif mode == "archival_color_bloom":
        bloom_color: Image.Image = ctx["bloom_color"]
        bloom_bw: Image.Image = ctx["bloom_bw"]
        bloom_depth_q: np.ndarray = ctx["bloom_depth_q"]

        bw_w, bw_h = bloom_color.size
        cur_zoom = 1.01 + 0.045 * eased
        cx = 0.5 + (ex - sx) * 0.30 * eased
        cy = 0.5 + (ey - sy) * 0.30 * eased
        cw = bw_w / cur_zoom
        ch = bw_h / cur_zoom
        x0 = min(max(0.0, cx * bw_w - cw * 0.5), bw_w - cw)
        y0 = min(max(0.0, cy * bw_h - ch * 0.5), bw_h - ch)

        bw_crop = bloom_bw.transform((width, height), Image.Transform.EXTENT, (x0, y0, x0 + cw, y0 + ch), resample=resample_mode)
        col_crop = bloom_color.transform((width, height), Image.Transform.EXTENT, (x0, y0, x0 + cw, y0 + ch), resample=resample_mode)

        bloom_p = _ease_in_out_quintic(min(1.0, t_norm / 0.88))
        thresh = 1.06 - 1.26 * bloom_p
        soft_k = np.clip((bloom_depth_q - thresh) / 0.32, 0.0, 1.0)
        color_w_q = soft_k * soft_k * (3.0 - 2.0 * soft_k)
        qh, qw = color_w_q.shape
        qx0 = min(max(0.0, (x0 / bw_w) * qw), qw - 1.0)
        qy0 = min(max(0.0, (y0 / bw_h) * qh), qh - 1.0)
        qx1 = min(float(qw), qx0 + (cw / bw_w) * qw)
        qy1 = min(float(qh), qy0 + (ch / bw_h) * qh)
        color_mask_img = Image.fromarray(np.clip(color_w_q * 255.0, 0, 255).astype(np.uint8), mode="L").transform(
            (width, height),
            Image.Transform.EXTENT,
            (qx0, qy0, qx1, qy1),
            resample=Image.Resampling.BILINEAR,
        )
        frame = bw_crop.copy()
        frame.paste(col_crop, (0, 0), color_mask_img)

    elif mode == "curator_loupe":
        dimmed_c: Image.Image = ctx["loupe_dimmed"]
        sharp_c: Image.Image = ctx["loupe_sharp"]
        loupe_mask: Image.Image = ctx["loupe_mask"]
        loupe_bezel: Image.Image = ctx["loupe_bezel"]
        rad: int = ctx["loupe_radius"]
        diam = rad * 2

        frame = dimmed_c.copy()
        # Smoothly glide the optical loupe tightly across the salient subject focal coordinates
        s_lx, s_ly = ctx["loupe_start_xy"]
        e_lx, e_ly = ctx["loupe_end_xy"]
        lcx = int(round(s_lx + (e_lx - s_lx) * eased))
        lcy = int(round(s_ly + (e_ly - s_ly) * eased - 4.0 * scale_factor * math.sin(math.pi * eased)))
        lcx = max(0, min(width - 1, lcx))
        lcy = max(0, min(height - 1, lcy))

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
            bg_plane, (width, height), 1.07 - 0.025 * eased, 0.5 - fg_dx * 0.22, 0.5 - fg_dy * 0.22, resample_mode
        )
        fg_scale = 0.982 + 0.030 * eased
        cw = max(1.0, fg_card_1.width * fg_scale)
        ch = max(1.0, fg_card_1.height * fg_scale)
        px = (width - cw) * 0.5 + fg_dx * width * 0.06
        py = max(10.0, height * card_cy_ratio - ch * 0.5 + fg_dy * height * 0.045)
        _paste_rgba_subpixel_pil(frame, fg_card_1, px, py, dst_w=cw, dst_h=ch)

    elif mode == "contact_sheet_zoom":
        sheet: Image.Image = ctx["contact_sheet"]
        hcx, hcy = ctx["contact_hero_xy"]
        # Continuous C2 quintic push-in with zero piecewise velocity kink
        push_p = _ease_in_out_quintic(max(0.0, (t_norm - 0.05) / 0.95))
        cur_zoom = 1.00 + 0.03 * eased + 1.48 * push_p
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
        cw = max(1.0, slide_card.width * scale_s)
        ch = max(1.0, slide_card.height * scale_s)
        px = (width - cw) * 0.5 + (eased - 0.5) * 18.0 * scale_factor
        py = max(8.0, height * card_cy_ratio - ch * 0.5)
        _paste_rgba_subpixel_pil(frame, slide_card, px, py, dst_w=cw, dst_h=ch)

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

    elif mode == "passepartout_shadowbox":
        sb = ctx["shadowbox"]
        wx, wy, ww, wh = int(sb["wx"]), int(sb["wy"]), int(sb["ww"]), int(sb["wh"])
        win_crop = _crop_plane_clamped(
            sb["photo_over"],
            (ww, wh),
            1.00 + 0.055 * eased,
            sx + (ex - sx) * eased * 0.65,
            sy + (ey - sy) * eased * 0.65,
            resample_mode,
        )
        win_crop.paste(sb["recess_dark"], (0, 0), sb["recess_mask"])
        board = sb["board_rgb"].copy()
        board.paste(win_crop, (wx, wy))
        frame = _crop_plane_clamped(
            board,
            (width, height),
            1.00 + 0.020 * eased,
            0.50 + (ex - sx) * 0.12 * (eased - 0.5),
            0.50 + (ey - sy) * 0.10 * (eased - 0.5),
            resample_mode,
        )

    elif mode == "kinfolk_linen_folio":
        kf = ctx["kinfolk_folio"]
        px, py, pw, ph = int(kf["px"]), int(kf["py"]), int(kf["pw"]), int(kf["ph"])
        win_crop = _crop_plane_clamped(
            kf["photo_over"],
            (pw, ph),
            1.00 + 0.060 * eased,
            sx + (ex - sx) * eased * 0.70,
            sy + (ey - sy) * eased * 0.70,
            resample_mode,
        )
        frame = kf["page_rgb"].copy()
        frame.paste(win_crop, (px, py))

    elif mode == "panoramic_frieze_scroll":
        fr = ctx["frieze"]
        kb_cfg = ctx.get("kb")
        preset_val = getattr(kb_cfg, "preset", "pan_left_to_right") if kb_cfg is not None else "pan_left_to_right"
        fw_f, fh_f = float(fr["fw"]), float(fr["fh"])
        outer_pad_f = float(fr.get("outer_pad", 0.0))
        pan_p = (1.0 - eased) if preset_val == "pan_right_to_left" else eased
        pan_z = 1.005 + 0.018 * math.sin(math.pi * eased)
        view_h = fh_f / pan_z
        view_w = view_h * (float(width) / max(1.0, float(height)))
        if view_w > fw_f:
            view_w = fw_f
            view_h = view_w * (float(height) / max(1.0, float(width)))
        max_x0 = max(0.0, fw_f - view_w)
        x_margin = min(max_x0 * 0.5, outer_pad_f * 0.25)
        bx0 = x_margin + (max_x0 - 2.0 * x_margin) * pan_p
        by0 = max(0.0, (fh_f - view_h) * 0.5)
        frame = fr["canvas_rgb"].transform(
            (width, height),
            Image.Transform.EXTENT,
            (bx0, by0, bx0 + view_w, by0 + view_h),
            resample=resample_mode,
        )

    elif mode == "atelier_layered_duo":
        ld = ctx["layered_duo"]
        frame = _crop_plane_clamped(ld["bg_rgb"], (width, height), 1.03 - 0.02 * eased, 0.50, 0.50, resample_mode)
        anc: Image.Image = ld["anchor_rgba"]
        fg_c: Image.Image = ld["fg_rgba"]
        has_card_f = bool(ctx.get("has_card"))

        s1 = 0.990 + 0.020 * eased
        w1 = max(1.0, anc.width * s1)
        h1 = max(1.0, anc.height * s1)
        x1 = width * 0.39 - w1 * 0.5 + (0.5 - eased) * 18.0 * scale_factor
        y1 = max(8.0, height * (0.42 if has_card_f else 0.46) - h1 * 0.5)
        _paste_rgba_subpixel_pil(frame, anc, x1, y1, dst_w=w1, dst_h=h1)

        s2 = 0.975 + 0.038 * eased
        w2 = max(1.0, fg_c.width * s2)
        h2 = max(1.0, fg_c.height * s2)
        x2 = width * 0.68 - w2 * 0.5 + (eased - 0.5) * 32.0 * scale_factor
        y2 = max(12.0, height * (0.53 if has_card_f else 0.58) - h2 * 0.5 + (0.5 - eased) * 12.0 * scale_factor)
        _paste_rgba_subpixel_pil(frame, fg_c, x2, y2, dst_w=w2, dst_h=h2)

    elif mode == "leica_frame_finder":
        lf = ctx["leica_finder"]
        frame = _crop_plane_clamped(
            lf["sharp_over"],
            (width, height),
            1.00 + 0.050 * eased,
            sx + (ex - sx) * eased * 0.55,
            sy + (ey - sy) * eased * 0.55,
            resample_mode,
        )
        frame.paste(lf["dim_rgb"], (0, 0), lf["surround_mask"])
        lock_p = _ease_in_out_quintic(min(1.0, t_norm / 0.38))
        reticle: Image.Image = lf["reticle_rgba"]
        if lock_p >= 0.999:
            frame.paste(reticle, (0, 0), reticle)
        else:
            ret_scale = 1.0 + 0.026 * (1.0 - lock_p)
            rw = max(1.0, width * ret_scale)
            rh = max(1.0, height * ret_scale)
            _paste_rgba_subpixel_pil(frame, reticle, (width - rw) * 0.5, (height - rh) * 0.5, dst_w=rw, dst_h=rh)

    elif mode == "swiss_monograph_grid":
        sg = ctx["swiss_grid"]
        frame = sg["page_rgb"].copy()
        h_crop = _crop_plane_clamped(
            sg["hero_over"], (sg["hw"], sg["hh"]), 1.00 + 0.065 * eased, 0.49 + 0.02 * eased, 0.50, resample_mode
        )
        d_crop = _crop_plane_clamped(
            sg["det_over"], (sg["dw"], sg["dh"]), 1.00 + 0.105 * eased, 0.50, 0.50 - 0.015 * eased, resample_mode
        )
        frame.paste(h_crop, (sg["hx"], sg["hy"]))
        frame.paste(d_crop, (sg["dx"], sg["dy"]))

    elif mode == "editorial_bookspread":
        bs = ctx["bookspread"]
        board = bs["table_rgb"].copy()
        c1 = _crop_plane_clamped(bs["p1_over"], (bs["w1"], bs["h1"]), 1.00 + 0.052 * eased, 0.50, 0.47, resample_mode)
        c2 = _crop_plane_clamped(bs["p2_over"], (bs["w2"], bs["h2"]), 1.052 - 0.044 * eased, 0.50, 0.50, resample_mode)
        board.paste(c1, (bs["x1"], bs["y1"]))
        board.paste(c2, (bs["x2"], bs["y2"]))
        frame = _crop_plane_clamped(
            board, (width, height), 1.00 + 0.032 * eased, 0.50 + 0.008 * (eased - 0.5), 0.50, resample_mode
        )
        # Subtle warm moving daylight wash across open book pages
        qw, qh = max(48, width // 4), max(27, height // 4)
        yy_g, xx_g = np.mgrid[0:qh, 0:qw].astype(np.float32)
        nx_g = xx_g / max(1.0, float(qw - 1))
        ny_g = yy_g / max(1.0, float(qh - 1))
        beam = np.exp(-(((nx_g - (0.28 + 0.35 * eased)) / 0.42) ** 2 + ((ny_g - 0.30) / 0.55) ** 2))
        beam_u8 = (np.clip(beam * 12.0, 0.0, 255.0)).astype(np.uint8)
        beam_full = np.asarray(Image.fromarray(beam_u8, mode="L").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
        f_arr = np.clip(np.asarray(frame, dtype=np.float32) + beam_full[:, :, None], 0.0, 255.0).astype(np.uint8)
        frame = Image.fromarray(f_arr, mode="RGB")

    elif mode == "gallery_wall_triptych":
        gw = ctx["gallery_wall"]
        ww_f, wh_f = float(gw["wall_w"]), float(gw["wall_h"])
        # Zero-clamp C2 quintic gallery walk keeping all 3 framed prints cleanly framed
        view_h = wh_f * (0.96 - 0.035 * math.sin(math.pi * eased))
        view_w = view_h * (float(width) / max(1.0, float(height)))
        if view_w > ww_f:
            view_w = ww_f
            view_h = view_w * (float(height) / max(1.0, float(width)))
        max_x0 = max(0.0, ww_f - view_w)
        bx0 = max_x0 * (0.18 + 0.64 * eased)
        by0 = max(0.0, (wh_f - view_h) * 0.50)
        frame = gw["wall_rgb"].transform(
            (width, height), Image.Transform.EXTENT, (bx0, by0, bx0 + view_w, by0 + view_h), resample=resample_mode
        )

    elif mode == "archival_lightbox_strip":
        lb = ctx["lightbox_strip"]
        sw_f, sh_f = float(lb["sw"]), float(lb["sh"])
        # Single unified C2 quintic smooth dolly (zero edge clamp, zero piecewise velocity jerk!)
        view_h = sh_f * (0.96 - 0.13 * eased)
        view_w = view_h * (float(width) / max(1.0, float(height)))
        if view_w > sw_f:
            view_w = sw_f
            view_h = view_w * (float(height) / max(1.0, float(width)))
        max_x0 = max(0.0, sw_f - view_w)
        bx0 = max_x0 * (0.20 + 0.34 * eased)
        by0 = max(0.0, (sh_f - view_h) * 0.49)
        frame = lb["strip_rgb"].transform(
            (width, height), Image.Transform.EXTENT, (bx0, by0, bx0 + view_w, by0 + view_h), resample=resample_mode
        )

    elif mode == "optical_tilt_shift":
        ts = ctx["tilt_shift"]
        cur_z = 1.01 + 0.075 * eased
        cx_t, cy_t = 0.49 + 0.025 * eased, 0.51 - 0.025 * eased
        sharp_c = _crop_plane_clamped(ts["sharp_over"], (width, height), cur_z, cx_t, cy_t, resample_mode)
        bokeh_c = _crop_plane_clamped(ts["bokeh_over"], (width, height), cur_z, cx_t, cy_t, resample_mode)
        qw, qh = max(48, width // 4), max(27, height // 4)
        yy_q, xx_q = np.mgrid[0:qh, 0:qw].astype(np.float32)
        nx_q = xx_q / max(1.0, float(qw - 1))
        ny_q = yy_q / max(1.0, float(qh - 1))
        tilt_coord = ny_q * 0.992 - (nx_q - 0.5) * 0.122
        focus_y = 0.70 - 0.38 * eased
        band_half = 0.165
        dist = np.abs(tilt_coord - focus_y)
        lin = np.clip((dist - band_half * 0.45) / (band_half * 1.45), 0.0, 1.0)
        blur_w = lin * lin * (3.0 - 2.0 * lin)
        mask_img = Image.fromarray(np.clip(blur_w * 255.0, 0, 255).astype(np.uint8), mode="L").resize(
            (width, height), Image.Resampling.BILINEAR
        )
        frame = sharp_c.copy()
        frame.paste(bokeh_c, (0, 0), mask_img)

    elif mode == "darkroom_developer_reveal":
        dd = ctx["darkroom_dev"]
        cur_z = 1.01 + 0.065 * eased
        col_c = _crop_plane_clamped(dd["color_over"], (width, height), cur_z, 0.50, 0.49, resample_mode)
        bw_c = _crop_plane_clamped(dd["bw_over"], (width, height), cur_z, 0.50, 0.49, resample_mode)
        col_arr = np.asarray(col_c, dtype=np.float32)
        bw_arr = np.asarray(bw_c, dtype=np.float32)
        pap = np.array(dd["paper_rgb"], dtype=np.float32).reshape(1, 1, 3)

        dev_p = _ease_in_out_quintic(min(1.0, t_norm / 0.34))
        luma = (0.299 * bw_arr[:, :, 0] + 0.587 * bw_arr[:, :, 1] + 0.114 * bw_arr[:, :, 2]) / 255.0
        ink_rev = np.clip(dev_p * 1.45 - luma * 0.45, 0.0, 1.0)[:, :, None]
        stage1 = pap * (1.0 - ink_rev) + bw_arr * ink_rev

        col_p = _ease_in_out_quintic(max(0.0, min(1.0, (t_norm - 0.22) / 0.58)))
        qw, qh = max(48, width // 4), max(27, height // 4)
        yy_q, xx_q = np.mgrid[0:qh, 0:qw].astype(np.float32)
        nx_q = xx_q / max(1.0, float(qw - 1))
        ny_q = yy_q / max(1.0, float(qh - 1))
        fx_c, fy_c = dd["focal_xy"]
        rad = np.sqrt(((nx_q - fx_c) * 1.70) ** 2 + ((ny_q - fy_c) * 2.00) ** 2)
        wave_r = col_p * 1.85 - 0.18
        cm_q = np.clip((wave_r - rad) / 0.42, 0.0, 1.0)
        cm_q = cm_q * cm_q * (3.0 - 2.0 * cm_q)
        cm_full = (
            np.asarray(
                Image.fromarray(np.clip(cm_q * 255.0, 0, 255).astype(np.uint8), mode="L").resize(
                    (width, height), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )[:, :, None]
            / 255.0
        )
        out_arr = np.clip(stage1 * (1.0 - cm_full) + col_arr * cm_full, 0.0, 255.0).astype(np.uint8)
        frame = Image.fromarray(out_arr, mode="RGB")

    elif mode == "cinema_gate_reveal":
        cg = ctx["cinema_gate"]
        frame = _crop_plane_clamped(cg["photo_over"], (width, height), 1.00 + 0.075 * eased, 0.49 + 0.02 * eased, 0.50, resample_mode)
        # Subtle 35mm highlight halation glow
        sw_h, sh_h = max(48, width // 6), max(27, height // 6)
        s_arr = np.asarray(frame.resize((sw_h, sh_h), Image.Resampling.BILINEAR), dtype=np.float32)
        luma = 0.299 * s_arr[:, :, 0] + 0.587 * s_arr[:, :, 1] + 0.114 * s_arr[:, :, 2]
        hi = np.clip((luma - 205.0) / 50.0, 0.0, 1.0)
        hi_blur = np.asarray(
            Image.fromarray(np.clip(hi * 255.0, 0, 255).astype(np.uint8), mode="L")
            .filter(ImageFilter.GaussianBlur(radius=4))
            .resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        f_arr = np.asarray(frame, dtype=np.float32)
        f_arr[:, :, 0] += hi_blur * 255.0 * 0.18
        f_arr[:, :, 1] += hi_blur * 138.0 * 0.18
        f_arr[:, :, 2] += hi_blur * 74.0 * 0.18
        frame = Image.fromarray(np.clip(f_arr, 0.0, 255.0).astype(np.uint8), mode="RGB")

        # Smooth C2 quintic velvet matte-box curtains opening cleanly to 2.39:1
        curt_p = _ease_in_out_quintic(min(1.0, t_norm / 0.34))
        cur_bar = int(round(cg["bar_h_target"] + (height * 0.16) * (1.0 - curt_p)))
        dr = ImageDraw.Draw(frame)
        dr.rectangle([0, 0, width, cur_bar], fill=(8, 9, 12))
        dr.rectangle([0, height - cur_bar, width, height], fill=(8, 9, 12))

        t_alpha = _ease_in_out_quintic(max(0.0, min(1.0, (t_norm - 0.16) / 0.26)))
        if t_alpha > 0.01:
            _paste_rgba_subpixel_pil(frame, cg["txt_rgba"], 0.0, 0.0, alpha_mul=t_alpha)

    elif mode == "prism_glass_refraction":
        pg = ctx["prism_glass"]
        base_c = _crop_plane_clamped(pg["base_over"], (width, height), 1.01 + 0.070 * eased, 0.49 + 0.02 * eased, 0.50, resample_mode)
        rr = _crop_plane_clamped(pg["soft_over"], (width, height), 1.04 + 0.070 * eased, 0.43 + 0.02 * eased, 0.50, resample_mode)
        rb = _crop_plane_clamped(pg["soft_over"], (width, height), 1.04 + 0.070 * eased, 0.45 + 0.02 * eased, 0.50, resample_mode)
        b_arr = np.asarray(base_c, dtype=np.float32)
        rr_arr = np.asarray(rr, dtype=np.float32)
        rb_arr = np.asarray(rb, dtype=np.float32)
        refr = np.stack([rr_arr[:, :, 0], 0.5 * (rr_arr[:, :, 1] + rb_arr[:, :, 1]), rb_arr[:, :, 2]], axis=-1)

        qw, qh = max(48, width // 4), max(27, height // 4)
        yy_q, xx_q = np.mgrid[0:qh, 0:qw].astype(np.float32)
        nx_q = xx_q / max(1.0, float(qw - 1))
        ny_q = yy_q / max(1.0, float(qh - 1))
        edge_line = (0.85 - 0.055 * math.sin(math.pi * eased)) - (ny_q - 0.5) * 0.24
        fm_q = np.clip((nx_q - edge_line) / 0.045, 0.0, 1.0)
        fm_q = fm_q * fm_q * (3.0 - 2.0 * fm_q)
        bev_q = np.exp(-(((nx_q - edge_line) / 0.012) ** 2))
        fm = np.asarray(Image.fromarray(np.clip(fm_q * 255.0, 0, 255).astype(np.uint8), mode="L").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)[:, :, None] / 255.0
        bev = np.asarray(Image.fromarray(np.clip(bev_q * 255.0, 0, 255).astype(np.uint8), mode="L").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)[:, :, None] / 255.0
        spec = np.array([255.0, 232.0, 198.0], dtype=np.float32).reshape(1, 1, 3)
        out_arr = b_arr * (1.0 - fm * 0.50) + refr * (fm * 0.50) + bev * spec * 0.24
        frame = Image.fromarray(np.clip(out_arr, 0.0, 255.0).astype(np.uint8), mode="RGB")

    elif mode == "venetian_sunbeam_pan":
        vn = ctx["venetian"]
        base_c = _crop_plane_clamped(vn["base_over"], (width, height), 1.00 + 0.070 * eased, 0.50, 0.52 - 0.03 * eased, resample_mode)
        f_arr = np.asarray(base_c, dtype=np.float32)
        qw, qh = max(48, width // 4), max(27, height // 4)
        yy_q, xx_q = np.mgrid[0:qh, 0:qw].astype(np.float32)
        nx_q = xx_q / max(1.0, float(qw - 1))
        ny_q = yy_q / max(1.0, float(qh - 1))
        diag = nx_q * 0.72 + ny_q * 0.69 - 0.14 * eased
        slats = np.power(np.clip(0.5 + 0.5 * np.sin(diag * 26.0), 0.0, 1.0), 2.4)
        win_env = np.exp(-(((nx_q - 0.38) / 0.62) ** 2 + ((ny_q - 0.36) / 0.60) ** 2))
        beam_q = slats * win_env * 0.22
        shade_q = 1.0 - (1.0 - slats) * win_env * 0.12

        # Vectorized microscopic sunlit dust motes glowing strictly inside the sunbeams
        m = vn["motes"]
        cx_m = (m["mx0"] + 0.022 * np.sin(m["m_ph"] + eased * 4.2))[:, None, None]
        cy_m = ((m["my0"] - 0.075 * eased) % 1.0)[:, None, None]
        r_m = m["m_rad"][:, None, None]
        twinkle = (0.65 + 0.35 * np.sin(m["m_ph"] + eased * 7.2))[:, None, None]
        aspect_r = float(width) / max(1.0, float(height))
        r2 = (((nx_q[None, :, :] - cx_m) * aspect_r) ** 2) + ((ny_q[None, :, :] - cy_m) ** 2)
        motes_acc = np.sum(np.exp(-r2 / (r_m * r_m)) * twinkle, axis=0)
        motes_q = np.clip(motes_acc, 0.0, 1.2) * slats * win_env * 0.40

        beam_f = np.asarray(Image.fromarray(np.clip(beam_q * 255.0, 0, 255).astype(np.uint8), mode="L").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)[:, :, None] / 255.0
        shade_f = np.asarray(Image.fromarray(np.clip(shade_q * 255.0, 0, 255).astype(np.uint8), mode="L").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)[:, :, None] / 255.0
        motes_f = np.asarray(Image.fromarray(np.clip(motes_q * 255.0, 0, 255).astype(np.uint8), mode="L").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)[:, :, None] / 255.0
        warm_sun = np.array([255.0, 224.0, 168.0], dtype=np.float32).reshape(1, 1, 3)
        out_arr = f_arr * shade_f
        light = warm_sun * beam_f
        out_arr = out_arr + light * (1.0 - out_arr / 255.0) + motes_f * warm_sun
        frame = Image.fromarray(np.clip(out_arr, 0.0, 255.0).astype(np.uint8), mode="RGB")

    elif mode == "vintage_postcard_deck":
        pd_ctx = ctx["postcard_deck"]
        frame = _crop_plane_clamped(pd_ctx["desk_rgb"], (width, height), 1.02 - 0.02 * eased, 0.50, 0.50, resample_mode)
        c1, c2 = pd_ctx["card1_rgba"], pd_ctx["card2_rgba"]
        # Smooth C2 quintic fan-apart using sub-pixel floating-point placement (zero harsh frame-0 snap!)
        slide = _ease_in_out_quintic(min(1.0, t_norm / 0.62))
        x1 = width * (0.34 - 0.05 * slide) - c1.width * 0.5
        y1 = height * (0.50 - 0.014 * eased) - c1.height * 0.5
        x2 = width * (0.64 + 0.06 * slide) - c2.width * 0.5
        y2 = height * (0.52 + 0.014 * eased) - c2.height * 0.5
        _paste_rgba_subpixel_pil(frame, c1, x1, y1)
        _paste_rgba_subpixel_pil(frame, c2, x2, y2)

    elif mode == "viewfinder_shutter_snap":
        ss = ctx["shutter_snap"]
        if t_norm < 0.34:
            p = _ease_in_out_quintic(t_norm / 0.34)
            frame = _crop_plane_clamped(ss["live_over"], (width, height), 1.07 - 0.04 * p, 0.52 - 0.02 * p, 0.48 + 0.02 * p, resample_mode)
            frame.paste(ss["vf_rgba"], (0, 0), ss["vf_rgba"])
        elif t_norm < 0.368:
            # Crisp ~85ms dark mechanical shutter curtain blink (rich charcoal, never blinding white)
            frame = Image.new("RGB", (width, height), (10, 11, 14))
        else:
            p2 = _ease_in_out_quintic((t_norm - 0.368) / 0.632)
            frame = _crop_plane_clamped(ss["mat_rgb"], (width, height), 1.038 - 0.034 * p2, 0.50, 0.50, resample_mode)

    elif mode == "accordion_leporello_fold":
        lp = ctx["leporello"]
        frame = lp["bg_rgb"].copy()
        pw, ph = float(lp["pw"]), float(lp["ph"])
        fold = (1.0 - eased) * 0.34 + 0.045
        eff_w = pw * math.cos(fold)
        persp_dy = ph * 0.055 * fold
        total_w = eff_w * 3.0
        x0 = (width - total_w) * 0.5
        cy_l = height * 0.50
        ys_top = [cy_l - ph * 0.5 + persp_dy, cy_l - ph * 0.5 - persp_dy, cy_l - ph * 0.5 + persp_dy, cy_l - ph * 0.5 - persp_dy]
        ys_bot = [cy_l + ph * 0.5 - persp_dy, cy_l + ph * 0.5 + persp_dy, cy_l + ph * 0.5 - persp_dy, cy_l + ph * 0.5 + persp_dy]
        shades = [1.0 - 0.14 * fold, 1.0 + 0.04 * fold, 1.0 - 0.12 * fold]
        src_pts = [(0.0, 0.0), (pw, 0.0), (pw, ph), (0.0, ph)]
        for i, pan in enumerate(lp["panels_rgba"]):
            lx = x0 + i * eff_w
            rx = lx + eff_w
            bx0 = max(0, int(math.floor(lx)) - 2)
            bx1 = min(width, int(math.ceil(rx)) + 2)
            by0 = max(0, int(math.floor(min(ys_top[i], ys_top[i + 1]))) - 2)
            by1 = min(height, int(math.ceil(max(ys_bot[i], ys_bot[i + 1]))) + 2)
            if bx1 <= bx0 + 2 or by1 <= by0 + 2:
                continue
            dst_pts = [
                (lx - bx0, ys_top[i] - by0),
                (rx - bx0, ys_top[i + 1] - by0),
                (rx - bx0, ys_bot[i + 1] - by0),
                (lx - bx0, ys_bot[i] - by0),
            ]
            coeffs = _solve_perspective_coeffs(dst_pts, src_pts)
            warped = pan.transform((bx1 - bx0, by1 - by0), Image.Transform.PERSPECTIVE, coeffs, resample=resample_mode)
            w_arr = np.asarray(warped, dtype=np.float32)
            w_arr[:, :, :3] = np.clip(w_arr[:, :, :3] * shades[i], 0.0, 255.0)
            w_shaded = Image.fromarray(w_arr.astype(np.uint8), mode="RGBA")
            frame.paste(w_shaded.convert("RGB"), (bx0, by0), w_shaded.split()[3])

    elif mode == "film_strip_sprocket_glide":
        fs = ctx["film_sprocket"]
        strip_h = float(fs["strip_h"])
        # Single continuous C2 quintic vertical film-gate glide (zero piecewise zoom kink, strictly >= 1.0x!)
        view_zoom = 1.00 + 0.055 * eased
        vh = float(height) / view_zoom
        vw = float(width) / view_zoom
        max_y0 = max(0.0, strip_h - vh)
        y0 = max_y0 * (0.20 + 0.32 * eased)
        x0 = (float(width) - vw) * 0.5
        frame = fs["strip_rgb"].transform(
            (width, height), Image.Transform.EXTENT, (x0, y0, x0 + vw, y0 + vh), resample=resample_mode
        )

    elif mode == "museum_acrylic_float":
        af = ctx["acrylic_float"]
        board = af["wall_rgb"].copy()
        px0, py0, pw, ph, sm = af["px0"], af["py0"], af["pw"], af["ph"], af["sm"]
        sh_dx = (26.0 - 52.0 * eased) * scale_factor
        sh_dy = (20.0 + 5.0 * math.sin(math.pi * eased)) * scale_factor
        _paste_rgba_subpixel_pil(board, af["shadow_rgba"], float(px0 - sm) + sh_dx, float(py0 - sm) + sh_dy)
        board.paste(af["acrylic_rgba"].convert("RGB"), (px0, py0), af["acrylic_rgba"].split()[3])
        # Diagonal glass specular sheen gliding smoothly across the acrylic face
        qw_a, qh_a = max(24, pw // 4), max(24, ph // 4)
        yy_a, xx_a = np.mgrid[0:qh_a, 0:qw_a].astype(np.float32)
        nx_a = xx_a / max(1.0, float(qw_a - 1))
        ny_a = yy_a / max(1.0, float(qh_a - 1))
        sweep_x = -0.25 + 1.50 * eased
        diag = (nx_a - sweep_x) + (ny_a - 0.5) * 0.38
        sheen_q = np.exp(-((diag / 0.085) ** 2)) * 36.0
        sheen_full = np.asarray(
            Image.fromarray(np.clip(sheen_q, 0.0, 255.0).astype(np.uint8), mode="L").resize((pw, ph), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )[:, :, None]
        roi = np.asarray(board.crop((px0, py0, px0 + pw, py0 + ph)), dtype=np.float32)
        board.paste(Image.fromarray(np.clip(roi + sheen_full, 0.0, 255.0).astype(np.uint8), mode="RGB"), (px0, py0))
        frame = _crop_plane_clamped(board, (width, height), 1.00 + 0.028 * eased, 0.50, 0.50, resample_mode)

    elif mode == "kinetic_triptych_reveal":
        kt = ctx["kinetic_triptych"]
        frame = Image.new("RGB", (width, height), (16, 18, 22))
        starts = [0.0, 0.10, 0.20]
        zooms = [1.00 + 0.085 * eased, 1.085 - 0.075 * eased, 1.01 + 0.075 * eased]
        for i in range(3):
            p = _ease_in_out_quintic(max(0.0, min(1.0, (t_norm - starts[i]) / 0.36)))
            if p <= 0.005:
                continue
            cw = kt["col_ws"][i]
            vis_h = max(2, int(round(height * p)))
            patch = _crop_plane_clamped(kt["cols_over"][i], (cw, height), zooms[i], 0.50, 0.50, resample_mode)
            if i == 1:
                frame.paste(patch.crop((0, height - vis_h, cw, height)), (kt["xs"][i], 0))
            else:
                frame.paste(patch.crop((0, 0, cw, vis_h)), (kt["xs"][i], height - vis_h))

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

    # Plane 3: Frosted-Glass Documentary Lower-Third Card (with smooth C2 quintic upward glide + fade)
    if ctx["has_card"] and ctx["card_fg"] is not None and ctx["card_mask"] is not None and ctx["card_tint"] is not None:
        t_sec = f_idx / max(1.0, float(fps))
        rem_sec = max(0.0, duration - t_sec)
        entry_prog = _ease_in_out_quintic((t_sec - 0.15) / 0.55) if t_sec > 0.15 else 0.0
        exit_prog = _ease_in_out_quintic(min(1.0, rem_sec / 0.45))
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
                tint_weight=float(ctx.get("card_tint_weight", 0.58)),
            )

    return frame


def build_uniform_mux_args(fps: int, hdr: bool, v_enc_args: list[str]) -> list[str]:
    """Return uniform container metadata, CFR rate, timescale, and HEVC SEI flags for -c copy concat."""
    args = [
        "-map_metadata",
        "-1",
        "-metadata:s:v:0",
        "rotate=0",
        "-r",
        str(fps),
        "-video_track_timescale",
        "30000",
    ]
    is_hevc = any(tok in str(arg).lower() for arg in v_enc_args for tok in ("hevc", "libx265"))
    if hdr and is_hevc and has_ffmpeg_bsf("filter_units"):
        args.extend(["-bsf:v", "filter_units=remove_types=39-40"])
    return args


def build_segment_overlay_filtergraph(
    base_label: str,
    hdr: bool,
    next_input_idx: int,
    segment_broll: Optional[list[tuple[Path, float, float]]] = None,
    segment_titles: Optional[list[tuple[Path, float, float, float, float] | dict[str, Any]]] = None,
    fps: int = 30,
) -> tuple[list[str], list[str], str]:
    """Build extra FFmpeg -i inputs & filter_complex stages for segment-scoped Track V2/V3 overlays."""
    extra_inputs: list[str] = []
    filter_parts: list[str] = []
    cur_v = base_label
    in_idx = next_input_idx
    # Blend overlays in full 4:4:4 chroma before final Lanczos 4:2:0 conversion so text edges stay razor-sharp
    ov_fmt = "yuv444p10" if hdr else "yuv444"

    for b_idx, (b_mp4, local_st, b_dur) in enumerate(segment_broll or []):
        extra_inputs.extend(["-i", str(b_mp4)])
        end_t = local_st + b_dur
        bv_label = f"[sbv{b_idx}]"
        next_v = f"[svb{b_idx + 1}]"
        filter_parts.append(f"[{in_idx}:v]setpts=PTS+{local_st:.3f}/TB{bv_label}")
        filter_parts.append(
            f"{cur_v}{bv_label}overlay=0:0:format={ov_fmt}:eof_action=pass:"
            f"enable='between(t,{local_st:.3f},{end_t:.3f})'{next_v}"
        )
        cur_v = next_v
        in_idx += 1

    for t_idx, t_ent in enumerate(segment_titles or []):
        if isinstance(t_ent, dict):
            local_st = float(t_ent["local_st"])
            t_dur = float(t_ent["local_dur"])
            end_t = local_st + t_dur
            fi = min(t_dur * 0.4, float(t_ent.get("fade_in", 0.35)))
            fo = min(t_dur * 0.4, float(t_ent.get("fade_out", 0.35)))
            fo_st = max(0.0, t_dur - fo)
            glass_rect = t_ent.get("glass_rect")
            glass_mask_path = t_ent.get("glass_mask_path")
            if glass_rect is not None and glass_mask_path is not None:
                gx, gy, gw, gh, _ = glass_rect
                sigma = max(4.0, float(t_ent.get("blur_sigma", 18.0)) * 0.65)
                extra_inputs.extend(["-loop", "1", "-t", f"{t_dur:.3f}", "-i", str(glass_mask_path)])
                filter_parts.append(f"{cur_v}split=2[v_smain{t_idx}][v_ssrc{t_idx}]")
                # Trim & align timestamps BEFORE alphamerge, blur at 1/4 scale in YUV (no bt601 RGB drop), and dim highlights cleanly
                filter_parts.append(
                    f"[v_ssrc{t_idx}]trim=start={local_st:.3f}:end={end_t:.3f},setpts=PTS-STARTPTS,"
                    f"crop={gw}:{gh}:{gx}:{gy},"
                    f"scale=w=max(16\\,trunc(iw/8)*2):h=max(16\\,trunc(ih/8)*2):flags=bilinear,"
                    f"gblur=sigma={sigma:.1f},scale={gw}:{gh}:flags=bicubic,"
                    f"lutyuv=y='max(16,(val-16)*0.74+16)',format=yuva444p[v_sblur{t_idx}]"
                )
                filter_parts.append(
                    f"[{in_idx}:v]format=gray,"
                    f"fade=t=in:st=0:d={fi:.3f},"
                    f"fade=t=out:st={fo_st:.3f}:d={fo:.3f}[v_sgmask{t_idx}]"
                )
                filter_parts.append(
                    f"[v_sblur{t_idx}][v_sgmask{t_idx}]alphamerge,setpts=PTS+{local_st:.3f}/TB[v_sglass{t_idx}]"
                )
                filter_parts.append(
                    f"[v_smain{t_idx}][v_sglass{t_idx}]overlay=x={gx}:y={gy}:format={ov_fmt}:"
                    f"eof_action=pass:enable='between(t,{local_st:.3f},{end_t:.3f})'[v_spg{t_idx}]"
                )
                cur_v = f"[v_spg{t_idx}]"
                in_idx += 1

            ov_path = t_ent["overlay_path"]
            rx, ry, _, _ = t_ent.get("roi", (0, 0, 0, 0))
            ov_label = f"[sov{t_idx}]"
            next_v = f"[svt{t_idx + 1}]"
            is_static_png = t_ent.get("kind") == "static_png"
            if is_static_png and hdr:
                # Decode single PNG frame, run 4K zscale ONCE, then loop+fade in HLG space (~2x faster)
                extra_inputs.extend(["-i", str(ov_path)])
                fade_chain = f"fade=t=in:st=0:d={fi:.3f}:alpha=1,fade=t=out:st={fo_st:.3f}:d={fo:.3f}:alpha=1"
                loop_pts_chain = (
                    f"loop=loop=-1:size=1:start=0,setpts=N/({max(1, int(fps))}*TB),"
                    f"trim=duration={t_dur:.3f},{fade_chain},setpts=PTS+{local_st:.3f}/TB"
                )
                filter_parts.append(
                    build_rgba_hdr_overlay_filter(
                        f"[{in_idx}:v]",
                        ov_label,
                        tag=f"sgov{t_idx}",
                        fade_chain="",
                        pts_chain=loop_pts_chain,
                    )
                )
            else:
                if is_static_png:
                    extra_inputs.extend(["-loop", "1", "-t", f"{t_dur:.3f}", "-i", str(ov_path)])
                    fade_chain = f"fade=t=in:st=0:d={fi:.3f}:alpha=1,fade=t=out:st={fo_st:.3f}:d={fo:.3f}:alpha=1"
                else:
                    extra_inputs.extend(["-i", str(ov_path)])
                    fade_chain = ""
                pts_chain = f"setpts=PTS+{local_st:.3f}/TB"
                if hdr:
                    filter_parts.append(
                        build_rgba_hdr_overlay_filter(
                            f"[{in_idx}:v]",
                            ov_label,
                            tag=f"sgov{t_idx}",
                            fade_chain=fade_chain,
                            pts_chain=pts_chain,
                        )
                    )
                else:
                    chain = f"format=rgba,{fade_chain},{pts_chain}" if fade_chain else f"format=rgba,{pts_chain}"
                    filter_parts.append(f"[{in_idx}:v]{chain}{ov_label}")
            filter_parts.append(
                f"{cur_v}{ov_label}overlay=x={rx}:y={ry}:format={ov_fmt}:eof_action=pass:"
                f"enable='between(t,{local_st:.3f},{end_t:.3f})'{next_v}"
            )
            cur_v = next_v
            in_idx += 1
        else:
            t_png, local_st, t_dur, fade_in, fade_out = t_ent
            end_t = local_st + t_dur
            fi = min(t_dur * 0.4, fade_in)
            fo = min(t_dur * 0.4, fade_out)
            fo_st = max(0.0, t_dur - fo)
            ov_label = f"[sov{t_idx}]"
            next_v = f"[svt{t_idx + 1}]"
            fade_chain = f"fade=t=in:st=0:d={fi:.3f}:alpha=1,fade=t=out:st={fo_st:.3f}:d={fo:.3f}:alpha=1"
            if hdr:
                extra_inputs.extend(["-i", str(t_png)])
                loop_pts_chain = (
                    f"loop=loop=-1:size=1:start=0,setpts=N/({max(1, int(fps))}*TB),"
                    f"trim=duration={t_dur:.3f},{fade_chain},setpts=PTS+{local_st:.3f}/TB"
                )
                filter_parts.append(
                    build_rgba_hdr_overlay_filter(
                        f"[{in_idx}:v]",
                        ov_label,
                        tag=f"sgov{t_idx}",
                        fade_chain="",
                        pts_chain=loop_pts_chain,
                    )
                )
            else:
                extra_inputs.extend(["-loop", "1", "-t", f"{t_dur:.3f}", "-i", str(t_png)])
                pts_chain = f"setpts=PTS+{local_st:.3f}/TB"
                filter_parts.append(f"[{in_idx}:v]format=rgba,{fade_chain},{pts_chain}{ov_label}")
            filter_parts.append(
                f"{cur_v}{ov_label}overlay=0:0:format={ov_fmt}:eof_action=pass:"
                f"enable='between(t,{local_st:.3f},{end_t:.3f})'{next_v}"
            )
            cur_v = next_v
            in_idx += 1

    if segment_broll or segment_titles:
        if hdr:
            filter_parts.append(
                f"{cur_v}scale=flags=lanczos+accurate_rnd,format=p010le,"
                f"setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc:range=tv[svhdr]"
            )
            cur_v = "[svhdr]"
        else:
            filter_parts.append(f"{cur_v}scale=flags=lanczos+accurate_rnd,format=yuv420p[svsdr]")
            cur_v = "[svsdr]"

    return extra_inputs, filter_parts, cur_v


_PHOTO_PREBUFFER_SEM = threading.BoundedSemaphore(2)
_CUDA_PHOTO_SEM = threading.BoundedSemaphore(2)
_CUDA_4K_LOCK = threading.Lock()


_HEAVY_VRAM_PHOTO_MODES = frozenset({"", "cinematic_depth_3d", "dolly_zoom_vertigo", "aperture_rack_focus"})


def _cuda_photo_budget(
    width: int,
    total_vram_gb: float,
    free_vram_gb: float,
    mode: str = "",
) -> tuple[int, tuple[int, ...], bool]:
    """Compute CUDA photo batch size, progressive retry schedule, and 4K/low-VRAM serialization."""
    free_gb = free_vram_gb if free_vram_gb > 0.0 else total_vram_gb
    if width >= 3840:
        if mode not in _HEAVY_VRAM_PHOTO_MODES:
            if free_gb >= 10.5:
                return 6, (6, 3, 1), False
            if free_gb >= 7.5:
                return 3, (3, 1), True
            return 1, (1,), True
        if free_gb >= 12.5:
            init_b = 3
        elif free_gb >= 7.5:
            init_b = 2
        else:
            init_b = 1
        retries = tuple(dict.fromkeys((init_b, max(1, init_b // 2), 1)))
        return init_b, retries, True
    if free_gb >= 8.0:
        init_b = 12
    elif free_gb >= 5.0:
        init_b = 6
    else:
        init_b = 3
    retries = tuple(sorted({init_b, max(1, init_b // 2), 2, 1}, reverse=True))
    return init_b, retries, free_gb < 6.0


def _resolve_ffmpeg_bin() -> str:
    """Resolve the direct ffmpeg executable, unwrapping Windows Scoop .shim files to avoid orphan child processes."""
    exe = shutil.which("ffmpeg") or "ffmpeg"
    if sys.platform == "win32" and exe.lower().endswith(".exe"):
        shim = Path(exe).with_suffix(".shim")
        if shim.is_file():
            try:
                for line in shim.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.lower().startswith("path"):
                        _, rhs = line.split("=", 1)
                        target = Path(rhs.strip().strip('"').strip("'"))
                        if target.is_file():
                            return str(target)
            except Exception:
                pass
    return exe


def _kill_proc_tree(proc: Optional[subprocess.Popen[Any]]) -> None:
    """Terminate a spawned subprocess and any child workers cleanly."""
    if proc is None:
        return
    try:
        if proc.poll() is None:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    check=False,
                    timeout=5,
                )
            else:
                proc.kill()
            proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
_MAX_PHOTO_PREBUFFER_BYTES = 4_800_000_000  # ~192 4K RGB24 frames (~6.4s @ 30fps) per worker


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
    segment_broll: Optional[list[tuple[Path, float, float]]] = None,
    segment_titles: Optional[list[tuple[Path, float, float, float, float] | dict[str, Any]]] = None,
    attached_effects: Optional[list[EffectTrackItem]] = None,
) -> Path:
    """Render a photo into a 4K HDR MP4 video clip using any of the 16 Premium Photo Modes + 8 Atmospheric Finishers."""
    kb = ken_burns or KenBurnsConfig()
    out_path = Path(output_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_name(f".{out_path.stem}.tmp_{os.getpid()}_{threading.get_ident()}.mp4")

    color_grade_fx = next((fx for fx in (attached_effects or []) if fx.effect_type == "color_grade"), None)
    total_frames = max(1, int(round(duration * fps)))
    ctx = _prepare_photo_scene_context(
        photo_path=photo_path,
        width=width,
        height=height,
        kb=kb,
        color_grade_fx=color_grade_fx,
    )

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

    has_seg_overlays = bool(segment_broll or segment_titles)

    def _build_photo_cmd(
        force_sw: bool,
        out_target: Optional[Path] = None,
        raw_pix_fmt: Literal["rgb24", "p010le", "yuv420p"] = "rgb24",
    ) -> list[str]:
        v_enc_local = get_video_encoder_args(hdr=hdr, force_software=force_sw)
        mux_local = build_uniform_mux_args(fps=fps, hdr=hdr, v_enc_args=v_enc_local)
        target_p = out_target or tmp_out
        raw_in_args = [
            "-f",
            "rawvideo",
            "-pix_fmt",
            raw_pix_fmt,
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
        ]
        if raw_pix_fmt == "p010le":
            raw_in_args.extend(
                [
                    "-color_primaries",
                    "bt2020",
                    "-color_trc",
                    "arib-std-b67",
                    "-colorspace",
                    "bt2020nc",
                    "-color_range",
                    "tv",
                ]
            )
            eff_vf: list[str] = [
                "setsar=1:1",
                "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc:range=tv",
            ]
        elif raw_pix_fmt == "yuv420p":
            raw_in_args.extend(
                [
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-colorspace",
                    "bt709",
                    "-color_range",
                    "tv",
                ]
            )
            eff_vf = ["setsar=1:1"]
        else:
            eff_vf = vf_filters

        c = [
            _resolve_ffmpeg_bin(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            *raw_in_args,
            "-i",
            "pipe:0",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
        ]
        if has_seg_overlays:
            extra_inputs, ov_parts, out_v_label = build_segment_overlay_filtergraph(
                base_label="[vbase]",
                hdr=hdr,
                next_input_idx=2,
                segment_broll=segment_broll,
                segment_titles=segment_titles,
                fps=fps,
            )
            fc_parts = [f"[0:v]{','.join(eff_vf)}[vbase]", *ov_parts]
            c.extend(extra_inputs)
            c.extend(
                [
                    "-t",
                    f"{duration:.3f}",
                    "-filter_complex",
                    ";".join(fc_parts),
                    "-map",
                    out_v_label,
                    "-map",
                    "1:a:0",
                ]
            )
        elif raw_pix_fmt in ("p010le", "yuv420p"):
            c.extend(
                [
                    "-t",
                    f"{duration:.3f}",
                    "-vf",
                    ",".join(eff_vf),
                ]
            )
        else:
            c.extend(
                [
                    "-t",
                    f"{duration:.3f}",
                    "-vf",
                    ",".join(eff_vf),
                ]
            )
        c.extend(
            [
                *v_enc_local,
                *mux_local,
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-shortest",
                str(target_p),
            ]
        )
        return c

    v_enc = get_video_encoder_args(hdr=hdr, force_software=False)
    cmd = _build_photo_cmd(force_sw=False)

    from ..hardware import get_cuda_free_vram_gb, nvenc_encode_guard, videotoolbox_hdr_encode_guard

    try:
        hw = get_hardware_profile()
        if hw.has_cuda and total_frames > 0:
            free_vram_gb = get_cuda_free_vram_gb()
            cuda_batch, batch_schedule, serialize_cuda = _cuda_photo_budget(
                width=width,
                total_vram_gb=hw.cuda_vram_gb,
                free_vram_gb=free_vram_gb,
                mode=str(ctx.get("mode") or ""),
            )
            gpu_pipe_fmt: Literal["rgb24", "p010le", "yuv420p"] = (
                ("p010le" if hdr else "yuv420p") if (width % 2 == 0 and height % 2 == 0) else "rgb24"
            )
            cuda_ok = False
            last_ok_batch = cuda_batch
            stderr_bytes = b""
            ret = 1
            proc: Optional[subprocess.Popen[bytes]] = None
            attempt_out: Optional[Path] = None
            with nvenc_encode_guard(v_enc):
                for attempt_idx, attempt_batch in enumerate(batch_schedule):
                    cuda_ok = False
                    cuda_sem_held = False
                    cuda_4k_held = False
                    proc = None
                    attempt_out = out_path.with_name(f".{out_path.stem}.tmp_{os.getpid()}_{threading.get_ident()}_a{attempt_idx}.mp4")
                    cmd_attempt = _build_photo_cmd(force_sw=False, out_target=attempt_out, raw_pix_fmt=gpu_pipe_fmt)
                    try:
                        if serialize_cuda:
                            _CUDA_4K_LOCK.acquire()
                            cuda_4k_held = True
                        _CUDA_PHOTO_SEM.acquire()
                        cuda_sem_held = True
                        proc = subprocess.Popen(cmd_attempt, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                        assert proc.stdin is not None
                        from .mps_photo_compositor import stream_photo_frames_to_pipe

                        stream_photo_frames_to_pipe(
                            ctx=ctx,
                            total_frames=total_frames,
                            duration=duration,
                            fps=fps,
                            pipe_stdin=proc.stdin,
                            batch_size=attempt_batch,
                            device_override="cuda",
                            output_pix_fmt=gpu_pipe_fmt,
                            transition_in=transition_in if gpu_pipe_fmt != "rgb24" else None,
                            transition_out=transition_out if gpu_pipe_fmt != "rgb24" else None,
                        )
                        cuda_ok = True
                        last_ok_batch = attempt_batch
                        break
                    except Exception as exc:
                        if proc is not None:
                            for pipe in (proc.stdin, proc.stderr):
                                if pipe is not None:
                                    try:
                                        pipe.close()
                                    except (BrokenPipeError, OSError, ValueError):
                                        pass
                            _kill_proc_tree(proc)
                        if attempt_out is not None:
                            for _attempt in range(3):
                                try:
                                    attempt_out.unlink(missing_ok=True)
                                    break
                                except OSError:
                                    time.sleep(0.05)
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        is_oom = "out of memory" in str(exc).lower()
                        if is_oom and attempt_idx < len(batch_schedule) - 1:
                            continue
                        print(f"[ken_burns] CUDA photo compositor fallback to CPU: {exc}", file=sys.stderr)
                        break
                    finally:
                        if proc is not None and proc.stdin is not None:
                            try:
                                proc.stdin.close()
                            except (BrokenPipeError, OSError, ValueError):
                                pass
                        if cuda_sem_held:
                            _CUDA_PHOTO_SEM.release()
                        if cuda_4k_held:
                            _CUDA_4K_LOCK.release()

                if cuda_ok and proc is not None and attempt_out is not None:
                    stderr_bytes = proc.stderr.read() if proc.stderr else b""
                    ret = proc.wait()
                    if ret == 0:
                        os.replace(attempt_out, out_path)
                        return out_path
                    attempt_out.unlink(missing_ok=True)
            if cuda_ok and ret != 0:
                sw_out = out_path.with_name(f".{out_path.stem}.tmp_{os.getpid()}_{threading.get_ident()}_sw.mp4")
                sw_cmd = _build_photo_cmd(force_sw=True, out_target=sw_out, raw_pix_fmt=gpu_pipe_fmt)
                sw_proc: Optional[subprocess.Popen[bytes]] = None
                sw_sem_held = False
                sw_4k_held = False
                try:
                    if serialize_cuda:
                        _CUDA_4K_LOCK.acquire()
                        sw_4k_held = True
                    _CUDA_PHOTO_SEM.acquire()
                    sw_sem_held = True
                    sw_proc = subprocess.Popen(sw_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                    assert sw_proc.stdin is not None
                    from .mps_photo_compositor import stream_photo_frames_to_pipe

                    stream_photo_frames_to_pipe(
                        ctx=ctx,
                        total_frames=total_frames,
                        duration=duration,
                        fps=fps,
                        pipe_stdin=sw_proc.stdin,
                        batch_size=last_ok_batch,
                        device_override="cuda",
                        output_pix_fmt=gpu_pipe_fmt,
                        transition_in=transition_in if gpu_pipe_fmt != "rgb24" else None,
                        transition_out=transition_out if gpu_pipe_fmt != "rgb24" else None,
                    )
                finally:
                    if sw_proc is not None and sw_proc.stdin is not None:
                        try:
                            sw_proc.stdin.close()
                        except (BrokenPipeError, OSError, ValueError):
                            pass
                    if sw_sem_held:
                        _CUDA_PHOTO_SEM.release()
                    if sw_4k_held:
                        _CUDA_4K_LOCK.release()
                assert sw_proc is not None
                sw_err = sw_proc.stderr.read() if sw_proc.stderr else b""
                sw_ret = sw_proc.wait()
                if sw_ret != 0:
                    sw_out.unlink(missing_ok=True)
                    err_msg = (sw_err or stderr_bytes).decode("utf-8", errors="replace")
                    raise RuntimeError(f"FFmpeg photo clip render failed (code {sw_ret}): {err_msg}")
                os.replace(sw_out, out_path)
                return out_path

        used_mps = False
        mps_batch_size = 4 if width >= 3840 else 8
        frame_bytes = max(1, width * height * 3)
        pre_frames = min(total_frames, max(mps_batch_size, _MAX_PHOTO_PREBUFFER_BYTES // frame_bytes))
        sem_held = False
        pre_batches: list[Optional[Any]] = []
        rem_comp: Optional[Any] = None

        try:
            # Stage A: Pre-synthesize MPS photo frames outside _VT_HDR_ENCODE_LOCK so AMD MPS
            # compute on Clip N+1 overlaps Apple T2 hevc_videotoolbox encoding on Clip N.
            if hw.has_mps and total_frames > 0:
                _PHOTO_PREBUFFER_SEM.acquire()
                sem_held = True
                comp: Optional[Any] = None
                try:
                    from .mps_photo_compositor import MPSPhotoCompositor

                    comp = MPSPhotoCompositor(ctx=ctx)
                    for start_idx in range(0, pre_frames, mps_batch_size):
                        batch_indices = list(range(start_idx, min(pre_frames, start_idx + mps_batch_size)))
                        pre_batches.append(
                            comp.render_batch_rgb(
                                frame_indices=batch_indices,
                                total_frames=total_frames,
                                duration=duration,
                                fps=fps,
                            )
                        )
                    if pre_frames >= total_frames:
                        comp.close()
                        comp = None
                    else:
                        rem_comp = comp
                    used_mps = True
                except Exception as exc:
                    pre_batches.clear()
                    if comp is not None:
                        try:
                            comp.close()
                        except Exception:
                            pass
                    if rem_comp is not None:
                        try:
                            rem_comp.close()
                        except Exception:
                            pass
                        rem_comp = None
                    if sem_held:
                        _PHOTO_PREBUFFER_SEM.release()
                        sem_held = False
                    used_mps = False
                    print(f"[ken_burns] MPS photo compositor fallback to CPU Pillow: {exc}", file=sys.stderr)

            # Stage B: Serialized HDR VideoToolbox encode consumes pre-buffered frames fast
            with videotoolbox_hdr_encode_guard(hdr=hdr, v_enc_args=v_enc):
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                assert proc.stdin is not None
                pipe_broken = False
                try:
                    if used_mps:
                        for b_idx in range(len(pre_batches)):
                            batch_t = pre_batches[b_idx]
                            pre_batches[b_idx] = None
                            if batch_t is None or pipe_broken:
                                continue
                            mv = memoryview(np.ascontiguousarray(batch_t.numpy()))
                            try:
                                proc.stdin.write(mv)
                            except (BrokenPipeError, OSError):
                                pipe_broken = True
                            finally:
                                mv.release()
                                del batch_t
                        pre_batches.clear()
                        if rem_comp is None and sem_held:
                            _PHOTO_PREBUFFER_SEM.release()
                            sem_held = False

                        if not pipe_broken and rem_comp is not None and pre_frames < total_frames:
                            for start_idx in range(pre_frames, total_frames, mps_batch_size):
                                batch_indices = list(range(start_idx, min(total_frames, start_idx + mps_batch_size)))
                                batch_t = rem_comp.render_batch_rgb(
                                    frame_indices=batch_indices,
                                    total_frames=total_frames,
                                    duration=duration,
                                    fps=fps,
                                )
                                mv = memoryview(np.ascontiguousarray(batch_t.numpy()))
                                try:
                                    proc.stdin.write(mv)
                                except (BrokenPipeError, OSError):
                                    break
                                finally:
                                    mv.release()
                                    del batch_t
                    else:
                        cpu_torch_ok = False
                        try:
                            from .mps_photo_compositor import stream_photo_frames_to_pipe

                            stream_photo_frames_to_pipe(
                                ctx=ctx,
                                total_frames=total_frames,
                                duration=duration,
                                fps=fps,
                                pipe_stdin=proc.stdin,
                                batch_size=4,
                                device_override="cpu",
                            )
                            cpu_torch_ok = True
                        except Exception:
                            cpu_torch_ok = False
                        if not cpu_torch_ok:
                            for f_idx in range(total_frames):
                                t_norm = f_idx / max(1, total_frames - 1)
                                frame = _render_photo_frame_from_context(
                                    ctx=ctx,
                                    t_norm=t_norm,
                                    f_idx=f_idx,
                                    duration=duration,
                                    fps=fps,
                                )
                                try:
                                    proc.stdin.write(frame.tobytes())
                                except (BrokenPipeError, OSError):
                                    break
                finally:
                    if rem_comp is not None:
                        try:
                            rem_comp.close()
                        except Exception:
                            pass
                        rem_comp = None
                    if sem_held:
                        _PHOTO_PREBUFFER_SEM.release()
                        sem_held = False
                    try:
                        proc.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass

                stderr_bytes = proc.stderr.read() if proc.stderr else b""
                ret = proc.wait()
        finally:
            pre_batches.clear()
            if rem_comp is not None:
                try:
                    rem_comp.close()
                except Exception:
                    pass
                rem_comp = None
            if sem_held:
                _PHOTO_PREBUFFER_SEM.release()
                sem_held = False

        if ret != 0:
            err_msg = stderr_bytes.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg photo clip render failed (code {ret}): {err_msg}")

        os.replace(tmp_out, out_path)
        return out_path
    finally:
        for _attempt in range(3):
            try:
                tmp_out.unlink(missing_ok=True)
                break
            except OSError:
                time.sleep(0.05)


def render_photo_effects_showcase_sheet(
    photo_path: str | Path,
    output_jpg: str | Path,
    secondary_path: Optional[str | Path] = None,
    tertiary_path: Optional[str | Path] = None,
    caption_title: str = "Curated Architectural & Fine Art Showcase",
    caption_subtitle: str = "22 Active Exhibition & Cinema Modes + 11 Optical Finishers · 4K HDR",
    camera_badge: str = "",
    tile_width: int = 960,
    tile_height: int = 540,
) -> Path:
    """
    Mechanically render a 4x6 (24-tile) high-resolution visual comparison showcase sheet
    covering all 22 active Photo Presentation Modes and all 11 Atmospheric Finishers.
    """
    out_path = Path(output_jpg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sec_str = str(secondary_path) if (secondary_path and Path(secondary_path).exists()) else None
    tert_str = str(tertiary_path) if (tertiary_path and Path(tertiary_path).exists()) else None

    showcase_specs: list[tuple[str, str, str, str]] = [
        ("swiss_monograph_grid", "none", "1. Swiss Monograph Grid + Detail", "warm_linen"),
        ("editorial_bookspread", "none", "2. Hardcover Bookspread (3D Spine)", "warm_linen"),
        ("gallery_wall_triptych", "none", "3. Gallery Wall Triptych + Track Lights", "obsidian_slate"),
        ("archival_lightbox_strip", "none", "4. Backlit 120 Film Lightbox Strip", "obsidian_slate"),
        ("optical_tilt_shift", "none", "5. 4x5 Optical Tilt-Shift Focus Sweep", "auto"),
        ("darkroom_developer_reveal", "none", "6. Darkroom Silver-Gelatin Emergence", "auto"),
        ("cinema_gate_reveal", "none", "7. 35mm Cinema Gate + Letterbox Titles", "auto"),
        ("prism_glass_refraction", "none", "8. Crystal Prism Edge Refraction", "auto"),
        ("venetian_sunbeam_pan", "none", "9. Venetian Sunbeams & Dust Motes", "auto"),
        ("vintage_postcard_deck", "none", "10. European Postcard Deck + Postmark", "warm_linen"),
        ("viewfinder_shutter_snap", "none", "11. Optical Viewfinder -> Shutter Snap", "warm_linen"),
        ("accordion_leporello_fold", "none", "12. 3D Accordion Leporello Unfold", "obsidian_slate"),
        ("film_strip_sprocket_glide", "none", "13. 35mm Film Strip Sprocket Glide", "obsidian_slate"),
        ("museum_acrylic_float", "none", "14. Acrylic Float Mount + Moving Shadow", "gallery_white"),
        ("kinetic_triptych_reveal", "none", "15. Kinetic 3-Beat Column Reveal", "auto"),
        ("passepartout_shadowbox", "architectural_window_dapple", "16. Museum Shadow-Box + Window Dapple", "warm_linen"),
        ("kinfolk_linen_folio", "fine_art_rag_texture", "17. Kinfolk Monograph Folio Spread", "warm_linen"),
        ("panoramic_frieze_scroll", "none", "18. Continuous Gallery Frieze Glide", "gallery_white"),
        ("atelier_layered_duo", "golden_light_leak", "19. Atelier Layered Duo + Screen Leak", "obsidian_slate"),
        ("leica_frame_finder", "nostalgic_film_grain", "20. Leica Rangefinder Brightline", "auto"),
        ("archival_color_bloom", "caustic_water_shimmer", "21. Color Bloom + Animated Caustics", "auto"),
        ("parallax_gallery", "golden_light_leak", "22. Calmed Museum Parallax", "auto"),
        ("floating_tilt_plate", "fine_art_rag_texture", "23. 3D Floating Tilt Plate", "auto"),
        ("glassmorphic_pedestal", "none", "24. Frosted Glassmorphic Vitrine", "auto"),
        ("contact_sheet_zoom", "none", "25. Darkroom Contact Sheet Push-In", "auto"),
        ("split_diopter_focus", "prism_chromatic_edge", "26. Optical Split Diopter + Prism Edge", "auto"),
        ("medium_format_slide", "velvet_vignette_rembrandt", "27. 120 Medium-Format Slide", "auto"),
        ("criterion_atelier_split", "none", "28. Criterion 3-Viewport Atelier Grid", "auto"),
        ("triptych_cinematic", "none", "29. 3-Panel Staggered Triptych", "auto"),
        ("polaroid_scrapbook", "nostalgic_film_grain", "30. Vintage Polaroid + 35mm Grain", "auto"),
        ("curator_loupe", "museum_spotlight", "31. 1.65x Optical Curator Loupe", "auto"),
        ("editorial_magazine", "none", "32. 62/38 Editorial Magazine Spread", "auto"),
        ("cinema_anamorphic", "anamorphic_streak", "33. 2.39:1 Anamorphic + Flare Streak", "auto"),
        ("diptych_frame", "golden_light_leak", "34. Side-by-Side Diptych Split-Screen", "auto"),
        ("gallery_frame", "champagne_bokeh", "35. Classic Gallery Frame + Soft Bokeh", "auto"),
        ("full_bleed", "halation_bloom_35mm", "36. Full-Bleed Ken Burns + Halation", "auto"),
        ("links_white_matte", "none", "37. Soft Limestone Exhibition Matte", "gallery_white"),
        ("passepartout_shadowbox", "none", "38. Shadow-Box (Gallery White)", "gallery_white"),
        ("kinfolk_linen_folio", "velvet_vignette_rembrandt", "39. Kinfolk Folio (Obsidian Slate)", "obsidian_slate"),
        ("swiss_monograph_grid", "none", "40. Swiss Monograph (Obsidian Slate)", "obsidian_slate"),
    ]

    cols, rows = 5, 8
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
        "PREMIUM PHOTO EFFECTS SHOWCASE — 37 EXHIBITION & CINEMA MODES + 11 OPTICAL FINISHERS",
        font=f_hdr,
        fill=(242, 238, 230),
    )

    for idx, (mode_name, fx_name, label_text, pal_name) in enumerate(showcase_specs):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (tile_width + pad)
        y0 = header_h + pad + r * (tile_height + label_h + pad)

        kb = KenBurnsConfig(
            mode=mode_name,  # type: ignore[arg-type]
            overlay_fx=fx_name,  # type: ignore[arg-type]
            surface_palette=pal_name,  # type: ignore[arg-type]
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
