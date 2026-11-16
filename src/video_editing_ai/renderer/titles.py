"""Programmatic 6-theme cinema typography, multi-tier font tracking, and RGBA title frame rendering via Pillow."""

from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path
import textwrap
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFont

from ..models.schema import TitleOverlayItem

# ~75% HLG signal (~203 nits diffuse white) prevents painful 1000-nit text bloom on HDR displays
HDR_DIFFUSE_WHITE_PEAK_RGB = 192
HDR_MAX_SHADOW_ALPHA = 152

LOCALIZED_CARD_STYLES: tuple[str, ...] = (
    "lower_third",
    "editorial_caption",
    "dialogue_badge",
    "exif_stamp",
    "route_map_card",
    "curved_gps_ribbon",
    "magazine_chapter_plate",
    "travel_map_pip_card",
)

# Art-directed visual design tokens for all 6 Cinema Title Themes
THEME_SPECS: dict[str, dict[str, Any]] = {
    "criterion_atelier": {
        "primary_tier": "serif",
        "primary_bold": True,
        "secondary_tier": "sans",
        "overline_tier": "sans",
        "default_tracking_px": 4.0,
        "overline_tracking_px": 9.0,
        "accent_hex": "#E8E4DC",
        "title_rgb": (250, 248, 244),
        "sub_rgb": (206, 210, 218),
        "card_fill": (12, 13, 17, 148),
        "glass_tint": (18, 20, 26, 62),
        "border_rgba": (232, 228, 220, 56),
        "stencil_matte": (46, 48, 54, 228),
        "glass_default": False,
        "fcpxml_font": "Georgia",
    },
    "a24_minimal": {
        "primary_tier": "sans",
        "primary_bold": True,
        "secondary_tier": "sans",
        "overline_tier": "mono",
        "default_tracking_px": 9.5,
        "overline_tracking_px": 11.0,
        "accent_hex": "#EFECE6",
        "title_rgb": (252, 250, 246),
        "sub_rgb": (196, 202, 210),
        "card_fill": (10, 11, 14, 132),
        "glass_tint": (16, 18, 22, 54),
        "border_rgba": (245, 243, 238, 42),
        "stencil_matte": (48, 49, 53, 232),
        "glass_default": False,
        "fcpxml_font": "Avenir Next",
    },
    "apple_tv_glass": {
        "primary_tier": "serif",
        "primary_bold": True,
        "secondary_tier": "sans",
        "overline_tier": "sans",
        "default_tracking_px": 2.0,
        "overline_tracking_px": 7.5,
        "accent_hex": "#DCE6F2",
        "title_rgb": (250, 250, 252),
        "sub_rgb": (212, 220, 230),
        "card_fill": (14, 16, 22, 142),
        "glass_tint": (22, 26, 36, 66),
        "border_rgba": (255, 255, 255, 58),
        "stencil_matte": (48, 52, 62, 224),
        "glass_default": True,
        "fcpxml_font": "PingFang SC",
    },
    "kinfolk_magazine": {
        "primary_tier": "serif",
        "primary_bold": False,
        "secondary_tier": "serif",
        "overline_tier": "sans",
        "default_tracking_px": 3.5,
        "overline_tracking_px": 8.5,
        "accent_hex": "#E2D2BA",
        "title_rgb": (248, 242, 232),
        "sub_rgb": (214, 204, 188),
        "card_fill": (20, 18, 15, 144),
        "glass_tint": (28, 25, 21, 64),
        "border_rgba": (226, 210, 186, 60),
        "stencil_matte": (156, 148, 138, 226),
        "glass_default": False,
        "fcpxml_font": "Georgia",
    },
    "archival_35mm": {
        "primary_tier": "mono",
        "primary_bold": True,
        "secondary_tier": "mono",
        "overline_tier": "mono",
        "default_tracking_px": 2.5,
        "overline_tracking_px": 6.5,
        "accent_hex": "#DFB472",
        "title_rgb": (246, 238, 222),
        "sub_rgb": (208, 196, 172),
        "card_fill": (16, 14, 12, 146),
        "glass_tint": (24, 21, 18, 66),
        "border_rgba": (223, 180, 114, 62),
        "stencil_matte": (50, 46, 42, 228),
        "glass_default": False,
        "fcpxml_font": "Consolas",
    },
    "kinetic_travelogue": {
        "primary_tier": "sans",
        "primary_bold": True,
        "secondary_tier": "sans",
        "overline_tier": "mono",
        "default_tracking_px": 3.5,
        "overline_tracking_px": 8.0,
        "accent_hex": "#BCE0D4",
        "title_rgb": (250, 250, 248),
        "sub_rgb": (206, 218, 214),
        "card_fill": (12, 16, 20, 144),
        "glass_tint": (18, 24, 30, 64),
        "border_rgba": (188, 224, 212, 58),
        "stencil_matte": (46, 54, 58, 226),
        "glass_default": True,
        "fcpxml_font": "Avenir Next",
    },
}


def get_theme_spec(theme: str) -> dict[str, Any]:
    """Return design tokens for a named Cinema Title Theme, falling back to apple_tv_glass."""
    return THEME_SPECS.get(theme, THEME_SPECS["apple_tv_glass"])


def _is_cjk_codepoint(cp: int) -> bool:
    """Return True if Unicode codepoint belongs to CJK / Hangul / Kana / fullwidth script ranges."""
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x3000 <= cp <= 0x303F)
        or (0x3040 <= cp <= 0x30FF)
        or (0xAC00 <= cp <= 0xD7AF)
        or (0xFF00 <= cp <= 0xFFEF)
    )


def clamp_rgba_for_hdr(
    rgba: tuple[int, int, int, int],
    hdr_safe: bool = False,
    peak_white: int = HDR_DIFFUSE_WHITE_PEAK_RGB,
    max_shadow_alpha: int = HDR_MAX_SHADOW_ALPHA,
) -> tuple[int, int, int, int]:
    """Clamp Pillow RGBA fill/shadow colors to HDR-safe diffuse white (~203 nits) when HDR is active."""
    r, g, b, a = rgba
    if not hdr_safe:
        return (int(r), int(g), int(b), int(a))
    if (r + g + b) <= 60:
        return (int(r), int(g), int(b), int(min(a, max_shadow_alpha)))
    max_ch = max(r, g, b)
    if max_ch > peak_white:
        scale = float(peak_white) / float(max(1, max_ch))
        return (
            int(round(r * scale)),
            int(round(g * scale)),
            int(round(b * scale)),
            int(a),
        )
    return (int(r), int(g), int(b), int(a))


def _contains_cjk(text: str) -> bool:
    """Return True when text contains CJK/East Asian ideographs or punctuation."""
    for ch in text or "":
        cp = ord(ch)
        if (
            0x2E80 <= cp <= 0x9FFF
            or 0x3000 <= cp <= 0x303F
            or 0xF900 <= cp <= 0xFAFF
            or 0xFF00 <= cp <= 0xFFEF
        ):
            return True
    return False


@lru_cache(maxsize=256)
def _load_font_tier(
    size: int,
    tier: str = "sans",
    bold: bool = True,
    italic: bool = False,
    text: str = "",
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Resolve and cache a FreeType font for 'serif', 'sans', 'mono', or 'cjk' typography tiers."""
    px = max(10, int(size))
    has_cjk = _contains_cjk(text)
    if tier == "cjk" or (has_cjk and tier in ("sans", "serif", "mono")):
        if tier == "serif":
            candidates = [
                "/System/Library/Fonts/Supplemental/Songti.ttc",
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Medium.ttc",
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simsun.ttc",
                "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ]
        else:
            candidates = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
                "/System/Library/Fonts/STHeiti Medium.ttc",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/msyhl.ttc",
                "C:/Windows/Fonts/simsun.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ]
    elif tier == "serif":
        if italic:
            candidates = [
                "C:/Windows/Fonts/georgiai.ttf",
                "C:/Windows/Fonts/palai.ttf",
                "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
            ]
        elif bold:
            candidates = [
                "C:/Windows/Fonts/georgiab.ttf",
                "C:/Windows/Fonts/palab.ttf",
                "C:/Windows/Fonts/BASKVILL.TTF",
                "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
                "/System/Library/Fonts/Supplemental/Baskerville.ttc",
                "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            ]
        else:
            candidates = [
                "C:/Windows/Fonts/georgia.ttf",
                "C:/Windows/Fonts/pala.ttf",
                "C:/Windows/Fonts/BASKVILL.TTF",
                "/System/Library/Fonts/Supplemental/Georgia.ttf",
                "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            ]
    elif tier == "mono":
        candidates = [
            "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/CascadiaMono.ttf",
            "C:/Windows/Fonts/consola.ttf",
            "/System/Library/Fonts/SFMono-Medium.otf",
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    else:
        if italic:
            candidates = [
                "C:/Windows/Fonts/segoeuii.ttf",
                "C:/Windows/Fonts/ariali.ttf",
                "/System/Library/Fonts/Avenir Next.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
            ]
        elif bold:
            candidates = [
                "/System/Library/Fonts/Avenir Next.ttc",
                "C:/Windows/Fonts/segoeuib.ttf",
                "C:/Windows/Fonts/bahnschrift.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "/System/Library/Fonts/HelveticaNeue.ttc",
                "/System/Library/Fonts/SFNS.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        else:
            candidates = [
                "/System/Library/Fonts/Avenir Next.ttc",
                "C:/Windows/Fonts/segoeuisl.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/bahnschrift.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "/System/Library/Fonts/HelveticaNeue.ttc",
                "/System/Library/Fonts/SFNS.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, px)
            except Exception:
                continue
    return _load_font(px, bold=bold, serif=(tier == "serif"), text=text) if tier != "cjk" else ImageFont.load_default()


def _load_font(
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
            "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
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
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Avenir Next.ttc",
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
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


def _format_tracked_eyebrow(text: str) -> str:
    txt = " ".join((text or "").strip().split())
    if not txt:
        return ""
    if _contains_cjk(txt):
        return txt.upper()
    up = txt.upper()
    if len(up) <= 22:
        words = up.split(" ")
        return "   ".join(" ".join(ch for ch in w) for w in words)
    return up


def _measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    if not text:
        return 0, 0
    box = draw.textbbox((0, 0), text, font=font)
    return max(0, box[2] - box[0]), max(0, box[3] - box[1])


def _fit_text_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    max_lines: int = 2,
) -> list[str]:
    """Wrap Latin or CJK text to fit within `max_width` pixels without overflowing glass card bounds."""
    clean = " ".join((text or "").strip().split())
    if not clean:
        return []
    if _measure_text(draw, clean, font)[0] <= max_width or max_lines <= 1:
        return [clean]

    tokens = clean.split(" ") if (" " in clean and not _contains_cjk(clean)) else list(clean)
    sep = " " if (" " in clean and not _contains_cjk(clean)) else ""
    lines: list[str] = []
    cur = ""
    for tok in tokens:
        cand = f"{cur}{sep}{tok}" if cur else tok
        if _measure_text(draw, cand, font)[0] <= max_width:
            cur = cand
        else:
            if cur:
                lines.append(cur)
                if len(lines) >= max_lines:
                    break
                cur = tok
            else:
                lines.append(tok)
                cur = ""
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines[:max_lines]


def _load_fitted_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    base_pt: float,
    min_pt: float,
    max_width: int,
    scale: float,
    bold: bool = True,
    serif: bool = True,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    pt = base_pt
    while pt > min_pt:
        font = _load_font(int(round(pt * scale)), bold=bold, serif=serif, text=text)
        if _measure_text(draw, text, font)[0] <= max_width:
            return font
        pt -= 2.0
    return _load_font(int(round(min_pt * scale)), bold=bold, serif=serif, text=text)


def _resolve_font_size(font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
    sz = getattr(font, "size", 24)
    return max(10, int(round(float(sz))))


def _split_script_runs(text: str) -> list[tuple[str, bool]]:
    """Split text into contiguous (substring, is_cjk) runs for clean mixed-script typography."""
    if not text:
        return []
    runs: list[tuple[str, bool]] = []
    cur_chars: list[str] = [text[0]]
    cur_cjk = _is_cjk_codepoint(ord(text[0]))
    for ch in text[1:]:
        ch_cjk = _is_cjk_codepoint(ord(ch))
        if ch_cjk == cur_cjk or ch == " ":
            cur_chars.append(ch)
        else:
            runs.append(("".join(cur_chars), cur_cjk))
            cur_chars = [ch]
            cur_cjk = ch_cjk
    runs.append(("".join(cur_chars), cur_cjk))
    return runs


def measure_tracked_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    letter_spacing_px: float = 0.0,
    cjk_font: Optional[ImageFont.FreeTypeFont | ImageFont.ImageFont] = None,
) -> tuple[int, int]:
    """Measure (width, height) of text with optional per-glyph tracking and automatic CJK fallback."""
    if not text:
        return (0, 0)
    fallback = cjk_font or _load_font_tier(_resolve_font_size(font), tier="cjk", bold=True)
    has_cjk = any(_is_cjk_codepoint(ord(ch)) for ch in text)
    if abs(letter_spacing_px) < 0.05 and not has_cjk:
        bb = draw.textbbox((0, 0), text, font=font)
        return (max(0, int(bb[2] - bb[0])), max(1, int(bb[3] - bb[1])))

    if abs(letter_spacing_px) < 0.05 and has_cjk:
        total_w = 0.0
        max_h = 0
        for run, is_cjk in _split_script_runs(text):
            f_use = fallback if is_cjk else font
            try:
                adv = float(draw.textlength(run, font=f_use))
            except Exception:
                bb = draw.textbbox((0, 0), run, font=f_use)
                adv = float(bb[2] - bb[0])
            bb = draw.textbbox((0, 0), run, font=f_use)
            total_w += adv
            max_h = max(max_h, int(bb[3] - bb[1]))
        return (max(1, int(round(total_w))), max(1, max_h))

    cursor_x = 0.0
    max_h = 0
    for idx, ch in enumerate(text):
        f_use = fallback if _is_cjk_codepoint(ord(ch)) else font
        try:
            adv = float(draw.textlength(ch, font=f_use))
        except Exception:
            bb = draw.textbbox((0, 0), ch, font=f_use)
            adv = float(bb[2] - bb[0])
        bb = draw.textbbox((0, 0), ch, font=f_use)
        max_h = max(max_h, int(bb[3] - bb[1]))
        cursor_x += adv
        if idx < len(text) - 1:
            cursor_x += float(letter_spacing_px)
    return (max(1, int(round(cursor_x))), max(1, max_h))


def _draw_tracked_on_draw(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    letter_spacing_px: float = 0.0,
    shadow: bool = True,
    scale: float = 1.0,
    hdr_safe: bool = False,
    cjk_font: Optional[ImageFont.FreeTypeFont | ImageFont.ImageFont] = None,
) -> int:
    """Render tracked mixed-script text onto a Pillow ImageDraw, returning rendered pixel width."""
    if not text:
        return 0
    x, y = float(xy[0]), float(xy[1])
    fallback = cjk_font or _load_font_tier(_resolve_font_size(font), tier="cjk", bold=True)
    fg_fill = clamp_rgba_for_hdr(fill, hdr_safe=hdr_safe)
    alpha_ratio = float(fg_fill[3]) / 255.0
    sh_outer = clamp_rgba_for_hdr((0, 0, 0, int(round(115 * alpha_ratio))), hdr_safe=hdr_safe)
    sh_inner = clamp_rgba_for_hdr((0, 0, 0, int(round(185 * alpha_ratio))), hdr_safe=hdr_safe)
    off1 = max(1, int(round(1.5 * scale)))
    off2 = max(2, int(round(3.0 * scale)))

    has_cjk = any(_is_cjk_codepoint(ord(ch)) for ch in text)
    if abs(letter_spacing_px) < 0.05 and not has_cjk:
        if shadow:
            draw.text((int(round(x)) + off2, int(round(y)) + off2), text, font=font, fill=sh_outer)
            draw.text((int(round(x)) + off1, int(round(y)) + off1), text, font=font, fill=sh_inner)
        draw.text((int(round(x)), int(round(y))), text, font=font, fill=fg_fill)
        bb = draw.textbbox((0, 0), text, font=font)
        return max(0, int(bb[2] - bb[0]))

    cursor = x
    if abs(letter_spacing_px) < 0.05:
        for run, is_cjk in _split_script_runs(text):
            f_use = fallback if is_cjk else font
            ix, iy = int(round(cursor)), int(round(y))
            if shadow:
                draw.text((ix + off2, iy + off2), run, font=f_use, fill=sh_outer)
                draw.text((ix + off1, iy + off1), run, font=f_use, fill=sh_inner)
            draw.text((ix, iy), run, font=f_use, fill=fg_fill)
            try:
                cursor += float(draw.textlength(run, font=f_use))
            except Exception:
                bb = draw.textbbox((0, 0), run, font=f_use)
                cursor += float(bb[2] - bb[0])
        return max(0, int(round(cursor - x)))

    for idx, ch in enumerate(text):
        f_use = fallback if _is_cjk_codepoint(ord(ch)) else font
        ix, iy = int(round(cursor)), int(round(y))
        if shadow:
            draw.text((ix + off2, iy + off2), ch, font=f_use, fill=sh_outer)
            draw.text((ix + off1, iy + off1), ch, font=f_use, fill=sh_inner)
        draw.text((ix, iy), ch, font=f_use, fill=fg_fill)
        try:
            adv = float(draw.textlength(ch, font=f_use))
        except Exception:
            bb = draw.textbbox((0, 0), ch, font=f_use)
            adv = float(bb[2] - bb[0])
        cursor += adv
        if idx < len(text) - 1:
            cursor += float(letter_spacing_px)
    return max(0, int(round(cursor - x)))


def draw_tracked_text(
    img: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    letter_spacing_px: float = 0.0,
    shadow: bool = True,
    scale: float = 1.0,
    hdr_safe: bool = False,
    cjk_font: Optional[ImageFont.FreeTypeFont | ImageFont.ImageFont] = None,
    clip_box: Optional[tuple[int, int, int, int]] = None,
) -> int:
    """Draw tracked mixed-script text with optional baseline clip masking."""
    if not text:
        return 0
    if clip_box is None:
        return _draw_tracked_on_draw(
            ImageDraw.Draw(img),
            xy,
            text,
            font=font,
            fill=fill,
            letter_spacing_px=letter_spacing_px,
            shadow=shadow,
            scale=scale,
            hdr_safe=hdr_safe,
            cjk_font=cjk_font,
        )
    cx1, cy1, cx2, cy2 = clip_box
    cx1 = max(0, min(img.width, int(cx1)))
    cy1 = max(0, min(img.height, int(cy1)))
    cx2 = max(cx1 + 1, min(img.width, int(cx2)))
    cy2 = max(cy1 + 1, min(img.height, int(cy2)))
    bw = cx2 - cx1
    bh = cy2 - cy1
    layer = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    w_px = _draw_tracked_on_draw(
        ImageDraw.Draw(layer),
        (xy[0] - cx1, xy[1] - cy1),
        text,
        font=font,
        fill=fill,
        letter_spacing_px=letter_spacing_px,
        shadow=shadow,
        scale=scale,
        hdr_safe=hdr_safe,
        cjk_font=cjk_font,
    )
    region = img.crop((cx1, cy1, cx2, cy2))
    region = Image.alpha_composite(region, layer)
    img.paste(region, (cx1, cy1))
    return w_px


def sample_shot_accent_hex(source_path: str | Path | None, fallback_hex: str = "#E8E4DC") -> str:
    """Extract a subtle, high-legibility accent hex tinted harmoniously toward an image's palette."""
    if not source_path:
        return fallback_hex
    p = Path(source_path)
    if not p.exists() or p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
        return fallback_hex
    try:
        with Image.open(p) as raw:
            thumb = raw.convert("RGB").resize((36, 36), Image.Resampling.BILINEAR)
            arr = np.asarray(thumb, dtype=np.float32)
            luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
            mid = arr[(luma >= 45.0) & (luma <= 215.0)]
            if mid.size == 0:
                mean_rgb = arr.reshape(-1, 3).mean(axis=0)
            else:
                mean_rgb = mid.mean(axis=0)
            # Blend 35% scene color with 65% crisp warm alabaster so text/rules always remain refined & readable
            blended = 0.35 * mean_rgb + 0.65 * np.array([232.0, 228.0, 220.0], dtype=np.float32)
            r, g, b = [max(140, min(248, int(round(c)))) for c in blended]
            return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return fallback_hex


def _parse_rgba(hex_color: str, alpha: int = 235, hdr_safe: bool = False) -> tuple[int, int, int, int]:
    norm = (hex_color or "").strip().upper()
    if not norm or norm in ("#FFD740", "#FFAB40", "#00E5FF", "#E040FB", "#F5C542"):
        raw = (232, 228, 220, alpha)
    else:
        try:
            rgb = ImageColor.getrgb(hex_color)
            raw = (rgb[0], rgb[1], rgb[2], alpha)
        except Exception:
            raw = (232, 228, 220, alpha)
    return clamp_rgba_for_hdr(raw, hdr_safe=hdr_safe)


def _scale_rgba_alpha(rgba: tuple[int, int, int, int], factor: float) -> tuple[int, int, int, int]:
    a = max(0, min(255, int(round(rgba[3] * max(0.0, min(1.0, factor))))))
    return (rgba[0], rgba[1], rgba[2], a)


def _draw_bottom_scrim(
    img: Image.Image,
    scrim_height_ratio: float = 0.24,
    max_alpha: int = 115,
    hdr_safe: bool = False,
) -> None:
    """Draw a smooth vertical bottom gradient scrim on an RGBA overlay so captions blend naturally."""
    w, h = img.size
    eff_alpha = min(max_alpha, HDR_MAX_SHADOW_ALPHA) if hdr_safe else max_alpha
    if eff_alpha <= 0:
        return
    scrim_h = max(16, int(round(h * scrim_height_ratio)))
    y_start = h - scrim_h
    grad_1d = Image.new("RGBA", (1, scrim_h), (0, 0, 0, 0))
    for y in range(scrim_h):
        t = y / max(1, scrim_h - 1)
        a = int(round(eff_alpha * (t * t)))
        grad_1d.putpixel((0, y), (8, 10, 14, a))
    scrim_strip = grad_1d.resize((w, scrim_h), Image.Resampling.BILINEAR)
    img.paste(scrim_strip, (0, y_start), scrim_strip)


def _draw_cinema_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    scale: float = 1.0,
    hdr_safe: bool = False,
) -> None:
    """Render typography with a soft two-pass cinema drop shadow for clean legibility."""
    _draw_tracked_on_draw(
        draw,
        xy,
        text,
        font=font,
        fill=fill,
        letter_spacing_px=0.0,
        shadow=True,
        scale=scale,
        hdr_safe=hdr_safe,
    )


def compute_overlay_bbox_norm(style: str) -> list[float]:
    """Return canonical normalized 2D screen bounding box [x1, y1, x2, y2] for Stage-55 overlap QC."""
    if style == "closing_card":
        return [0.16, 0.34, 0.84, 0.66]
    if style == "credits_colophon_card":
        return [0.16, 0.28, 0.84, 0.74]
    if style == "knockout_stencil_opening":
        return [0.08, 0.22, 0.92, 0.78]
    if style in ("opening_card", "act_banner"):
        return [0.14, 0.65, 0.86, 0.83]
    if style == "magazine_chapter_plate":
        return [0.07, 0.59, 0.78, 0.83]
    if style == "editorial_caption":
        return [0.05, 0.07, 0.62, 0.23]
    if style in ("route_map_card", "curved_gps_ribbon"):
        return [0.11, 0.80, 0.89, 0.96]
    if style == "travel_map_pip_card":
        return [0.62, 0.06, 0.95, 0.40]
    if style == "dialogue_badge":
        return [0.06, 0.10, 0.58, 0.25]
    if style == "exif_stamp":
        return [0.64, 0.06, 0.96, 0.16]
    return [0.12, 0.70, 0.88, 0.84]


def _even_floor(val: int, minimum: int = 2) -> int:
    return max(minimum, int(val) - (int(val) % 2))


def compute_glass_pill_rect(
    item: TitleOverlayItem,
    width: int = 1920,
    height: int = 1080,
) -> Optional[tuple[int, int, int, int, int]]:
    """
    Compute the exact even-aligned rounded-pill rectangle (x, y, w, h, radius) used by
    both Pillow card rendering and FFmpeg live frosted-glass backdrop blur.
    """
    if item.style not in LOCALIZED_CARD_STYLES:
        return None
    s = width / 1920.0
    sub_text = (item.subtitle or "").strip()
    overline_text = (getattr(item, "overline", "") or getattr(item, "eyebrow", None) or "").strip()
    dummy = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    spec = get_theme_spec(getattr(item, "theme", "apple_tv_glass"))

    if item.style in ("route_map_card", "curved_gps_ribbon"):
        waypoints = item.map_waypoints or [{"label": "Start"}, {"label": item.text[:16]}, {"label": "Finale"}]
        pw = min(int(width * 0.78), max(int(round(620 * s)), len(waypoints) * int(round(176 * s))))
        ph = int(round((142 if item.style == "curved_gps_ribbon" else 128) * s))
        pl = (width - pw) // 2
        pt = height - ph - int(round(34 * s))
        rad = int(round(16 * s))
    elif item.style == "travel_map_pip_card":
        pw = max(int(round(360 * s)), int(round(width * 0.33)))
        ph = max(int(round(220 * s)), int(round(height * 0.34)))
        pl = width - int(round(width * 0.05)) - pw
        pt = int(round(height * 0.06))
        rad = int(round(16 * s))
    elif item.style == "editorial_caption":
        eyebrow_str = _format_tracked_eyebrow(overline_text)
        f_eye = _load_font(int(round(12 * s)), bold=True, serif=False, text=eyebrow_str)
        max_card_w = int(width * 0.56)
        inner_max_w = max_card_w - int(round(68 * s))
        f_title = _load_fitted_font(draw, item.text, 25.0, 19.0, inner_max_w, scale=s, bold=True, serif=True)
        f_sub = _load_font(int(round(17 * s)), bold=False, serif=False, text=sub_text)
        title_lines = _fit_text_lines(draw, item.text, f_title, inner_max_w, max_lines=2)
        sub_lines = _fit_text_lines(draw, sub_text, f_sub, inner_max_w, max_lines=3) if sub_text else []
        ew = _measure_text(draw, eyebrow_str, f_eye)[0] if eyebrow_str else 0
        tw = max((_measure_text(draw, ln, f_title)[0] for ln in title_lines), default=0)
        sw = max((_measure_text(draw, ln, f_sub)[0] for ln in sub_lines), default=0)
        pw = min(max_card_w, max(int(round(360 * s)), max(ew, tw, sw) + int(round(72 * s))))
        eye_h = max(14, int(round(17 * s))) if eyebrow_str else 0
        title_line_h = max(20, int(round(31 * s)))
        sub_line_h = max(15, int(round(23 * s)))
        pad_y = int(round(15 * s))
        cur_h = (
            pad_y * 2
            + (eye_h + int(round(5 * s)) if eyebrow_str else 0)
            + len(title_lines) * title_line_h
            + (int(round(4 * s)) + len(sub_lines) * sub_line_h if sub_lines else 0)
        )
        ph = min(int(height * 0.24), cur_h)
        pl = int(round(width * 0.05))
        pt = int(round(height * 0.07))
        rad = int(round(13 * s))
    elif item.style == "dialogue_badge":
        f_title = _load_font_tier(int(round(23 * s)), tier=spec["primary_tier"], bold=spec["primary_bold"])
        f_sub = _load_font_tier(int(round(16 * s)), tier=spec["secondary_tier"], bold=False)
        tw, _ = measure_tracked_text(draw, item.text, f_title)
        sw, _ = measure_tracked_text(draw, sub_text[:54], f_sub) if sub_text else (0, 0)
        pw = min(int(width * 0.50), max(tw, sw) + int(round(62 * s)))
        ph = int(round((90 if sub_text else 60) * s))
        pl = int(round(width * 0.06))
        pt = int(round(height * 0.10))
        rad = int(round(14 * s))
    elif item.style == "exif_stamp":
        f_stamp = _load_font_tier(int(round(17 * s)), tier="mono", bold=True)
        f_sub = _load_font_tier(int(round(14 * s)), tier="mono", bold=False)
        tw, _ = measure_tracked_text(draw, item.text, f_stamp, letter_spacing_px=1.5 * s)
        sw, _ = measure_tracked_text(draw, sub_text, f_sub, letter_spacing_px=1.0 * s) if sub_text else (0, 0)
        pw = min(int(width * 0.34), max(tw, sw) + int(round(44 * s)))
        ph = int(round((62 if sub_text else 40) * s))
        pr = width - int(round(width * 0.04))
        pl = pr - pw
        pt = int(round(height * 0.06))
        rad = int(round(10 * s))
    elif item.style == "magazine_chapter_plate":
        f_title = _load_font_tier(int(round(34 * s)), tier=spec["primary_tier"], bold=spec["primary_bold"])
        f_num = _load_font_tier(int(round(90 * s)), tier="serif", bold=False, italic=True)
        num_str = (getattr(item, "chapter_numeral", "") or "I").strip()
        num_w, _ = measure_tracked_text(draw, num_str, f_num)
        tw, _ = measure_tracked_text(draw, item.text, f_title, letter_spacing_px=spec["default_tracking_px"] * s)
        left_pad = max(int(round(116 * s)), int(round(48 * s)) + num_w)
        pw = min(int(width * 0.72), max(int(round(480 * s)), left_pad + tw + int(round(42 * s))))
        ph = int(round((148 if sub_text else 118) * s))
        pl = int(round(width * 0.075))
        pt = int(round(height * 0.825)) - ph
        rad = int(round(16 * s))
    else:
        # lower_third
        f_title = _load_font_tier(int(round(27 * s)), tier=spec["primary_tier"], bold=spec["primary_bold"])
        f_sub = _load_font_tier(int(round(18 * s)), tier=spec["secondary_tier"], bold=False)
        sub_lines = textwrap.wrap(sub_text, width=74)[:2] if sub_text else []
        track = float(getattr(item, "letter_spacing_px", 0.0) or spec["default_tracking_px"] * 0.5) * s
        tw, _ = measure_tracked_text(draw, item.text, f_title, letter_spacing_px=track)
        sw = max((measure_tracked_text(draw, ln, f_sub)[0] for ln in sub_lines), default=0)
        pw = min(int(width * 0.80), max(int(round(420 * s)), max(tw, sw) + int(round(92 * s))))
        cur_h = int(round((34 if overline_text else 18) * s)) + max(16, int(round(36 * s)))
        if sub_lines:
            cur_h += int(round(6 * s)) + len(sub_lines) * max(14, int(round(25 * s)))
        ph = cur_h + int(round(16 * s))
        pl = (width - pw) // 2
        pt = int(round(height * 0.835)) - ph
        rad = int(round(15 * s))

    rx = max(0, min(width - 16, _even_floor(pl, 0)))
    ry = max(0, min(height - 16, _even_floor(pt, 0)))
    rw = min(width - rx, _even_floor(pw, 16))
    rh = min(height - ry, _even_floor(ph, 16))
    return (rx, ry, rw, rh, max(4, int(rad)))


def _ease_out_cubic(x: float) -> float:
    c = max(0.0, min(1.0, float(x)))
    return 1.0 - (1.0 - c) ** 3


def _smoothstep(x: float) -> float:
    c = max(0.0, min(1.0, float(x)))
    return c * c * (3.0 - 2.0 * c)


def _compute_kinetic_envelope(item: TitleOverlayItem, t_norm: float) -> tuple[float, float, float, float]:
    """Return (enter_p, exit_alpha, hold_breathe_p, master_alpha) for normalized playback time t_norm."""
    preset = getattr(item, "motion_preset", "masked_rise")
    if preset == "static":
        return (1.0, 1.0, 0.5, 1.0)
    dur = max(0.25, float(item.duration))
    t_sec = max(0.0, min(dur, float(t_norm) * dur))
    fi = min(dur * 0.42, max(0.04, float(item.fade_in)))
    fo = min(dur * 0.42, max(0.04, float(item.fade_out)))

    e_raw = min(1.0, t_sec / max(0.04, fi))
    enter_p = _ease_out_cubic(e_raw)
    exit_alpha = _smoothstep(min(1.0, max(0.0, (dur - t_sec) / max(0.04, fo))))
    hold_p = max(0.0, min(1.0, (t_sec - fi * 0.25) / max(0.2, dur - fi * 0.25)))
    master_alpha = min(1.0, e_raw * 1.35) * exit_alpha
    return (enter_p, exit_alpha, hold_p, master_alpha)


def _catmull_rom_polyline(pts: list[tuple[float, float]], samples_per_seg: int = 24) -> list[tuple[float, float]]:
    """Interpolate a smooth C1 spline passing cleanly through all GPS waypoint anchors."""
    if len(pts) < 2:
        return list(pts)
    ext = [pts[0]] + list(pts) + [pts[-1]]
    out: list[tuple[float, float]] = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for j in range(samples_per_seg):
            t = j / float(samples_per_seg)
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2.0 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2.0 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
            )
            out.append((x, y))
    out.append(pts[-1])
    return out


def _draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    pts: list[tuple[float, float]],
    fill: tuple[int, int, int, int],
    width_px: int,
    dash_px: float = 10.0,
    gap_px: float = 7.0,
) -> None:
    if len(pts) < 2:
        return
    accum = 0.0
    period = dash_px + gap_px
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if (accum % period) < dash_px:
            draw.line([(int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1)))], fill=fill, width=width_px)
        accum += seg_len


def _draw_card_container(
    img: Image.Image,
    rect: tuple[int, int, int, int, int],
    spec: dict[str, Any],
    hdr_safe: bool,
    master_alpha: float,
    transparent_glass_body: bool,
    backdrop_blur: bool,
) -> None:
    """Draw either a smoked glass fill or a subtle specular rim over live FFmpeg blurred backdrop."""
    rx, ry, rw, rh, rad = rect
    draw = ImageDraw.Draw(img)
    if transparent_glass_body or backdrop_blur:
        fill_raw = spec["glass_tint"] if transparent_glass_body else spec["card_fill"]
    else:
        fill_raw = spec["card_fill"]
    fill_col = clamp_rgba_for_hdr(_scale_rgba_alpha(fill_raw, master_alpha), hdr_safe=hdr_safe)
    border_col = clamp_rgba_for_hdr(_scale_rgba_alpha(spec["border_rgba"], master_alpha), hdr_safe=hdr_safe)
    draw.rounded_rectangle([rx, ry, rx + rw, ry + rh], radius=rad, fill=fill_col, outline=border_col, width=max(1, rw // 420))
    # Subtle inner specular top highlight rim for tactile acrylic depth
    inner_rim = clamp_rgba_for_hdr(_scale_rgba_alpha((255, 255, 255, 48), master_alpha), hdr_safe=hdr_safe)
    draw.line([(rx + rad, ry + 1), (rx + rw - rad, ry + 1)], fill=inner_rim, width=1)


def render_title_frame(
    item: TitleOverlayItem,
    width: int = 1920,
    height: int = 1080,
    t_norm: float = 0.55,
    hdr_safe: Optional[bool] = None,
    transparent_glass_body: bool = False,
) -> Image.Image:
    """
    Render a single keyframe of a TitleOverlayItem across all 6 Cinema Themes & 11 Archetypes
    as an RGBA Pillow image with HDR diffuse-white clamping when enabled.
    """
    use_hdr_safe = bool(item.hdr_safe) if hdr_safe is None else bool(hdr_safe)
    theme_name = getattr(item, "theme", "apple_tv_glass")
    spec = get_theme_spec(theme_name)
    s = width / 1920.0

    enter_p, exit_alpha, hold_p, master_alpha = _compute_kinetic_envelope(item, t_norm)
    sub_e = _ease_out_cubic(max(0.0, min(1.0, (enter_p - 0.15) / 0.85)))
    rule_e = _smoothstep(min(1.0, enter_p * 1.15))

    accent_hex = item.accent_color if item.accent_color and item.accent_color != "#E8E4DC" else spec["accent_hex"]
    accent_rgba = _scale_rgba_alpha(_parse_rgba(accent_hex, 215, hdr_safe=use_hdr_safe), master_alpha)
    title_rgb = spec["title_rgb"]
    sub_rgb = spec["sub_rgb"]
    title_fill = (*title_rgb, int(round(255 * master_alpha)))
    sub_fill = (*sub_rgb, int(round(234 * sub_e * exit_alpha)))

    sub_text = (item.subtitle or "").strip()
    overline_text = (getattr(item, "overline", "") or getattr(item, "eyebrow", None) or "").strip()
    track_override = float(getattr(item, "letter_spacing_px", 0.0) or 0.0)
    base_track = track_override if track_override != 0.0 else float(spec["default_tracking_px"])
    if getattr(item, "motion_preset", "masked_rise") == "tracking_expand":
        dyn_track = (base_track * (0.4 + 0.85 * enter_p) + 2.8 * hold_p) * s
    else:
        dyn_track = (base_track * 0.55 + 1.2 * hold_p) * s

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # ------------------------------------------------------------------
    # Archetype 1: Video-Through-Text Knockout Stencil Opening Title
    # ------------------------------------------------------------------
    if item.style == "knockout_stencil_opening":
        matte_col = clamp_rgba_for_hdr(
            (
                spec["stencil_matte"][0],
                spec["stencil_matte"][1],
                spec["stencil_matte"][2],
                int(round(spec["stencil_matte"][3] * exit_alpha)),
            ),
            hdr_safe=use_hdr_safe,
        )
        matte = Image.new("RGBA", (width, height), matte_col)
        stencil_rgba = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        s_draw = ImageDraw.Draw(stencil_rgba)

        # Clean editorial inner frame outline inside the matte
        pad_x = int(round(width * 0.07))
        pad_y = int(round(height * 0.14))
        stencil_size = int(round((84 if len(item.text) <= 22 else 66) * s * (1.0 + 0.04 * hold_p)))
        f_stencil = _load_font_tier(stencil_size, tier="sans" if theme_name != "criterion_atelier" else "serif", bold=True)
        stencil_track = max(5.0, base_track) * s * (0.9 + 0.25 * hold_p)
        tw, th = measure_tracked_text(s_draw, item.text.upper(), f_stencil, letter_spacing_px=stencil_track)

        # If long title, wrap neatly into 2 lines so every glyph sits inside [0.08..0.92, 0.22..0.78]
        lines = textwrap.wrap(item.text.upper(), width=18)[:2] if tw > int(width * 0.78) else [item.text.upper()]
        line_h = int(round(stencil_size * 1.18))
        total_h = len(lines) * line_h
        y_cursor = (height - total_h) // 2 - int(round(14 * s))
        for ln in lines:
            lw, _ = measure_tracked_text(s_draw, ln, f_stencil, letter_spacing_px=stencil_track)
            _draw_tracked_on_draw(
                s_draw,
                ((width - lw) // 2, y_cursor),
                ln,
                font=f_stencil,
                fill=(255, 255, 255, 255),
                letter_spacing_px=stencil_track,
                shadow=False,
                hdr_safe=False,
            )
            y_cursor += line_h

        # Punch glyph interior to alpha=0 so live video shines strictly through typography
        r_ch, g_ch, b_ch, a_ch = matte.split()
        inv_mask = ImageChops.invert(stencil_rgba.split()[3])
        punched_a = ImageChops.multiply(a_ch, inv_mask)
        matte = Image.merge("RGBA", (r_ch, g_ch, b_ch, punched_a))
        img = Image.alpha_composite(img, matte)

        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [pad_x, pad_y, width - pad_x, height - pad_y],
            outline=clamp_rgba_for_hdr(_scale_rgba_alpha((*title_rgb, 68), master_alpha), hdr_safe=use_hdr_safe),
            width=max(1, int(round(1.5 * s))),
        )
        over_txt = overline_text or "A CURATED DOCUMENTARY FILM"
        f_over = _load_font_tier(int(round(17 * s)), tier=spec["overline_tier"], bold=True)
        otw, _ = measure_tracked_text(draw, over_txt.upper(), f_over, letter_spacing_px=spec["overline_tracking_px"] * s)
        draw_tracked_text(
            img,
            ((width - otw) // 2, (height - total_h) // 2 - int(round(56 * s))),
            over_txt.upper(),
            font=f_over,
            fill=_scale_rgba_alpha((*title_rgb, 215), master_alpha),
            letter_spacing_px=spec["overline_tracking_px"] * s,
            shadow=False,
            hdr_safe=use_hdr_safe,
        )
        if sub_text:
            f_sub = _load_font_tier(int(round(20 * s)), tier=spec["secondary_tier"], bold=False)
            sw, _ = measure_tracked_text(draw, sub_text, f_sub, letter_spacing_px=3.0 * s)
            draw_tracked_text(
                img,
                ((width - sw) // 2, (height + total_h) // 2 + int(round(16 * s))),
                sub_text,
                font=f_sub,
                fill=sub_fill,
                letter_spacing_px=3.0 * s,
                shadow=False,
                hdr_safe=use_hdr_safe,
            )

    # ------------------------------------------------------------------
    # Archetype 2: Editorial Opening, Closing, Act Banner & Colophon Card
    # ------------------------------------------------------------------
    elif item.style in ("opening_card", "closing_card", "act_banner", "credits_colophon_card"):
        is_closing = item.style in ("closing_card", "credits_colophon_card")
        if is_closing:
            veil_a = min(108, HDR_MAX_SHADOW_ALPHA) if use_hdr_safe else 108
            veil_a = int(round(veil_a * master_alpha))
            img = Image.alpha_composite(img, Image.new("RGBA", (width, height), (10, 12, 16, veil_a)))
            cy = int(round(height * (0.42 if item.style == "credits_colophon_card" else 0.50)))
            title_pt = 48 if item.style == "closing_card" else 42
        else:
            _draw_bottom_scrim(img, scrim_height_ratio=0.34, max_alpha=int(round(135 * master_alpha)), hdr_safe=use_hdr_safe)
            cy = int(round(height * 0.745))
            title_pt = 46 if item.style == "opening_card" else 36

        draw = ImageDraw.Draw(img)
        f_over = _load_font_tier(int(round(15 * s)), tier=spec["overline_tier"], bold=True)
        f_title = _load_font_tier(int(round(title_pt * s)), tier=spec["primary_tier"], bold=spec["primary_bold"])
        f_sub = _load_font_tier(int(round(20 * s)), tier=spec["secondary_tier"], bold=False)

        cx = width // 2
        if overline_text:
            otw, _ = measure_tracked_text(draw, overline_text.upper(), f_over, letter_spacing_px=spec["overline_tracking_px"] * s)
            draw_tracked_text(
                img,
                (cx - otw // 2, cy - int(round(68 * s))),
                overline_text.upper(),
                font=f_over,
                fill=_scale_rgba_alpha(accent_rgba, sub_e),
                letter_spacing_px=spec["overline_tracking_px"] * s,
                scale=s,
                hdr_safe=use_hdr_safe,
            )

        half_rule = int(round(max(24.0, 74.0 * s) * rule_e))
        rule_y = cy - int(round((40 if sub_text else 24) * s))
        draw.line([(cx - half_rule, rule_y), (cx + half_rule, rule_y)], fill=accent_rgba, width=max(1, int(round(1.6 * s))))

        tw, th = measure_tracked_text(draw, item.text, f_title, letter_spacing_px=dyn_track)
        base_ty = cy - int(round((24 if sub_text else 10) * s))
        rise_off = int(round(24.0 * s * (1.0 - enter_p)))
        clip_h = max(th + int(round(18 * s)), int(round(title_pt * 1.45 * s)))
        draw_tracked_text(
            img,
            (cx - tw // 2, base_ty + rise_off),
            item.text,
            font=f_title,
            fill=title_fill,
            letter_spacing_px=dyn_track,
            scale=s,
            hdr_safe=use_hdr_safe,
            clip_box=(0, base_ty - int(round(6 * s)), width, base_ty + clip_h),
        )

        cur_y = base_ty + int(round((title_pt + 14) * s))
        if sub_text:
            sw, _ = measure_tracked_text(draw, sub_text, f_sub, letter_spacing_px=2.0 * s)
            sub_rise = int(round(12.0 * s * (1.0 - sub_e)))
            draw_tracked_text(
                img,
                (cx - sw // 2, cur_y + sub_rise),
                sub_text,
                font=f_sub,
                fill=sub_fill,
                letter_spacing_px=2.0 * s,
                scale=s,
                hdr_safe=use_hdr_safe,
            )
            cur_y += int(round(36 * s))

        # Structured colophon credit lines (chapters, camera stock, score attribution)
        if item.style == "credits_colophon_card":
            lines_c = list(getattr(item, "colophon_lines", []) or []) or [
                "DIRECTED & ASSEMBLED BY AUTONOMOUS POST STUDIO",
                "4K 10-BIT HLG REC.2020 · FCPXML 1.11",
            ]
            f_col = _load_font_tier(int(round(15 * s)), tier="mono", bold=False)
            draw.line(
                [(cx - int(round(160 * s * rule_e)), cur_y + int(round(4 * s))), (cx + int(round(160 * s * rule_e)), cur_y + int(round(4 * s)))],
                fill=_scale_rgba_alpha((*title_rgb, 75), master_alpha),
                width=1,
            )
            cur_y += int(round(18 * s))
            for idx_c, c_ln in enumerate(lines_c[:5]):
                line_e = _ease_out_cubic(max(0.0, min(1.0, (enter_p - 0.12 * idx_c) / 0.75)))
                cw_ln, _ = measure_tracked_text(draw, c_ln, f_col, letter_spacing_px=1.6 * s)
                draw_tracked_text(
                    img,
                    (cx - cw_ln // 2, cur_y + int(round(8 * s * (1.0 - line_e)))),
                    c_ln,
                    font=f_col,
                    fill=_scale_rgba_alpha((*sub_rgb, 215), line_e * exit_alpha),
                    letter_spacing_px=1.6 * s,
                    scale=s,
                    hdr_safe=use_hdr_safe,
                )
                cur_y += int(round(24 * s))

    # ------------------------------------------------------------------
    # Archetype 3: Asymmetric Magazine Chapter Plate with Roman Numeral Watermark
    # ------------------------------------------------------------------
    elif item.style == "magazine_chapter_plate":
        _draw_bottom_scrim(img, scrim_height_ratio=0.25, max_alpha=int(round(115 * master_alpha)), hdr_safe=use_hdr_safe)
        rect = compute_glass_pill_rect(item, width, height)
        if rect is not None:
            rx, ry, rw, rh, rad = rect
            _draw_card_container(img, rect, spec, use_hdr_safe, master_alpha, transparent_glass_body, item.backdrop_blur)
            draw = ImageDraw.Draw(img)

            num_str = (getattr(item, "chapter_numeral", "") or "I").strip()
            f_num = _load_font_tier(int(round(90 * s)), tier="serif", bold=False, italic=True)
            num_w, _ = measure_tracked_text(draw, num_str, f_num)
            num_fill = clamp_rgba_for_hdr(_scale_rgba_alpha((*accent_rgba[:3], 58), master_alpha), hdr_safe=use_hdr_safe)
            draw.text((rx + int(round(22 * s)), ry + int(round(10 * s))), num_str, font=f_num, fill=num_fill)

            tx = rx + max(int(round(116 * s)), int(round(48 * s)) + num_w)
            overline_use = (overline_text or "CHAPTER").upper()
            f_over = _load_font_tier(int(round(13 * s)), tier=spec["overline_tier"], bold=True)
            f_title = _load_font_tier(int(round(32 * s)), tier=spec["primary_tier"], bold=spec["primary_bold"])
            f_sub = _load_font_tier(int(round(17 * s)), tier=spec["secondary_tier"], bold=False)

            draw_tracked_text(
                img,
                (tx, ry + int(round(18 * s))),
                overline_use,
                font=f_over,
                fill=_scale_rgba_alpha(accent_rgba, sub_e),
                letter_spacing_px=spec["overline_tracking_px"] * s,
                scale=s,
                hdr_safe=use_hdr_safe,
            )
            rule_len = int(round(max(20.0, (rx + rw - tx - int(round(24 * s))) * rule_e)))
            rule_y = ry + int(round(38 * s))
            draw.line([(tx, rule_y), (tx + rule_len, rule_y)], fill=accent_rgba, width=max(1, int(round(1.5 * s))))

            ty = ry + int(round(48 * s))
            rise_off = int(round(20 * s * (1.0 - enter_p)))
            draw_tracked_text(
                img,
                (tx, ty + rise_off),
                item.text,
                font=f_title,
                fill=title_fill,
                letter_spacing_px=dyn_track * 0.65,
                scale=s,
                hdr_safe=use_hdr_safe,
                clip_box=(tx - 4, ty - 4, rx + rw - 16, ty + int(round(46 * s))),
            )
            if sub_text:
                draw_tracked_text(
                    img,
                    (tx, ty + int(round(42 * s + 10 * s * (1.0 - sub_e)))),
                    sub_text[:72],
                    font=f_sub,
                    fill=sub_fill,
                    letter_spacing_px=1.0 * s,
                    scale=s,
                    hdr_safe=use_hdr_safe,
                )

    # ------------------------------------------------------------------
    # Archetype 3b: Upper-Right Corner Animated Travel Map PiP Card
    # ------------------------------------------------------------------
    elif item.style == "travel_map_pip_card":
        rect = compute_glass_pill_rect(item, width, height)
        if rect is not None:
            pl, pt, pw, ph, rad = rect
            _draw_card_container(img, rect, spec, use_hdr_safe, master_alpha, transparent_glass_body, item.backdrop_blur)
            draw = ImageDraw.Draw(img)

            f_hdr_over = _load_font_tier(int(round(12 * s)), tier=spec["overline_tier"], bold=True)
            f_hdr_title = _load_font_tier(int(round(19 * s)), tier=spec["primary_tier"], bold=spec["primary_bold"])
            f_hdr_sub = _load_font_tier(int(round(13 * s)), tier=spec["overline_tier"], bold=False)

            pad_x = int(round(14 * s))
            hdr_h = int(round(48 * s))
            ov_lbl = (overline_text or "LIVE ROUTE TELEMETRY").upper()[:26]
            draw_tracked_text(
                img,
                (pl + pad_x, pt + int(round(9 * s))),
                ov_lbl,
                font=f_hdr_over,
                fill=accent_rgba,
                letter_spacing_px=1.6 * s,
                scale=s,
                hdr_safe=use_hdr_safe,
            )
            draw_tracked_text(
                img,
                (pl + pad_x, pt + int(round(23 * s))),
                item.text[:28],
                font=f_hdr_title,
                fill=title_fill,
                letter_spacing_px=0.5 * s,
                scale=s,
                hdr_safe=use_hdr_safe,
                clip_box=(pl + pad_x - 2, pt + int(round(20 * s)), pl + pw - pad_x, pt + hdr_h + 2),
            )
            if sub_text:
                sub_len = int(draw.textlength(sub_text[:24], font=f_hdr_sub))
                draw_tracked_text(
                    img,
                    (max(pl + pad_x + 120, pl + pw - pad_x - sub_len - int(round(4 * s))), pt + int(round(9 * s))),
                    sub_text[:24],
                    font=f_hdr_sub,
                    fill=sub_fill,
                    letter_spacing_px=0.5 * s,
                    scale=s,
                    hdr_safe=use_hdr_safe,
                )

            map_x = pl + pad_x
            map_y = pt + hdr_h
            map_w = max(120, pw - 2 * pad_x)
            map_h = max(80, ph - hdr_h - int(round(12 * s)))

            from .travel_map import render_travel_map_frame, resolve_travel_map_style

            raw_wps = item.map_waypoints or [
                {"label": "Origin", "lat": 37.7749, "lon": -122.4194, "completed": True, "active": False},
                {"label": (item.text or "Destination")[:18], "lat": 35.6762, "lon": 139.6503, "completed": True, "active": True},
            ]
            active_idx = next((idx_w for idx_w, w in enumerate(raw_wps) if isinstance(w, dict) and w.get("active")), len(raw_wps) - 1)
            leg_i = max(0, min(len(raw_wps) - 2, active_idx - 1))
            t_map_prog = min(0.98, max(0.08, 0.14 + 0.82 * hold_p))
            map_style = resolve_travel_map_style("auto", theme_name, f"{item.text} {sub_text}")

            map_rgba = render_travel_map_frame(
                waypoints=raw_wps,
                style=map_style,
                width=map_w,
                height=map_h,
                leg_index=leg_i,
                t_norm=t_map_prog,
                transport_mode="auto",
                hdr_safe=use_hdr_safe,
                compact_hud=True,
                allow_tile_fetch=False,
            )
            inner_rad = max(6, rad - int(round(5 * s)))
            mask_l = Image.new("L", (map_w, map_h), 0)
            m_draw = ImageDraw.Draw(mask_l)
            m_draw.rounded_rectangle([0, 0, map_w - 1, map_h - 1], radius=inner_rad, fill=int(round(248 * master_alpha)))
            map_rgba.putalpha(mask_l)
            img.alpha_composite(map_rgba, dest=(map_x, map_y))

            inner_border = clamp_rgba_for_hdr(_scale_rgba_alpha(spec["border_rgba"], master_alpha), hdr_safe=use_hdr_safe)
            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle([map_x, map_y, map_x + map_w - 1, map_y + map_h - 1], radius=inner_rad, outline=inner_border, width=1)

    # ------------------------------------------------------------------
    # Archetype 4: Curved Bezier GPS Journey Ribbon & Linear Route Map
    # ------------------------------------------------------------------
    elif item.style in ("curved_gps_ribbon", "route_map_card"):
        _draw_bottom_scrim(img, scrim_height_ratio=0.26, max_alpha=int(round(115 * master_alpha)), hdr_safe=use_hdr_safe)
        rect = compute_glass_pill_rect(item, width, height)
        if rect is not None:
            pl, pt, pw, ph, rad = rect
            _draw_card_container(img, rect, spec, use_hdr_safe, master_alpha, transparent_glass_body, item.backdrop_blur)
            draw = ImageDraw.Draw(img)

            f_title = _load_font_tier(int(round(25 * s)), tier=spec["primary_tier"], bold=spec["primary_bold"])
            f_sub = _load_font_tier(int(round(15 * s)), tier=spec["overline_tier"], bold=False)
            f_wp = _load_font_tier(int(round(13 * s)), tier=spec["overline_tier"], bold=True)

            waypoints = item.map_waypoints or [
                {"label": "Departure", "completed": True, "active": False},
                {"label": item.text[:20], "completed": True, "active": True},
                {"label": "Finale", "completed": False, "active": False},
            ]
            n_wp = max(1, len(waypoints))
            track_pad = int(round(76 * s))
            x_start = pl + track_pad
            x_end = pl + pw - track_pad

            if item.style == "curved_gps_ribbon" and n_wp >= 2:
                base_y = pt + int(round(42 * s))
                amp_y = int(round(13 * s))
                anchors: list[tuple[float, float]] = []
                for idx_w in range(n_wp):
                    frac = idx_w / float(max(1, n_wp - 1))
                    wx = x_start + (x_end - x_start) * frac
                    wy = base_y + ((-amp_y if idx_w % 2 == 0 else amp_y) * (0.75 if idx_w in (0, n_wp - 1) else 1.0))
                    anchors.append((wx, wy))
                curve_pts = _catmull_rom_polyline(anchors, samples_per_seg=22)

                # Dashed future base curve
                _draw_dashed_polyline(
                    draw,
                    curve_pts,
                    fill=clamp_rgba_for_hdr(_scale_rgba_alpha((150, 158, 172, 160), master_alpha), hdr_safe=use_hdr_safe),
                    width_px=max(2, int(round(2.0 * s))),
                    dash_px=8.0 * s,
                    gap_px=6.0 * s,
                )
                active_idx = next((i for i, w in enumerate(waypoints) if w.get("active")), n_wp - 1)
                target_frac = float(active_idx) / float(max(1, n_wp - 1))
                anim_frac = min(target_frac, max(0.05, target_frac * (0.18 + 0.82 * enter_p)))
                cutoff = max(2, int(round(len(curve_pts) * anim_frac)))
                if cutoff >= 2:
                    drawn_pts = [(int(round(px)), int(round(py))) for px, py in curve_pts[:cutoff]]
                    draw.line(drawn_pts, fill=accent_rgba, width=max(2, int(round(3.0 * s))))

                for idx_w, (wp, (wx_f, wy_f)) in enumerate(zip(waypoints, anchors)):
                    wx, wy = int(round(wx_f)), int(round(wy_f))
                    is_act = bool(wp.get("active"))
                    is_comp = bool(wp.get("completed")) or is_act
                    rad_o = int(round((8 if is_act else 5) * s))
                    if is_act:
                        pulse = 1.0 + 0.35 * abs(math.sin(hold_p * math.pi * 2.0))
                        hr = int(round((rad_o + 5 * s) * pulse))
                        draw.ellipse(
                            [wx - hr, wy - hr, wx + hr, wy + hr],
                            fill=clamp_rgba_for_hdr(_scale_rgba_alpha((*accent_rgba[:3], 76), master_alpha), hdr_safe=use_hdr_safe),
                        )
                    n_col = clamp_rgba_for_hdr(
                        _scale_rgba_alpha(
                            (*title_rgb, 255) if is_act else ((*accent_rgba[:3], 225) if is_comp else (120, 126, 138, 195)),
                            master_alpha,
                        ),
                        hdr_safe=use_hdr_safe,
                    )
                    draw.ellipse([wx - rad_o, wy - rad_o, wx + rad_o, wy + rad_o], fill=n_col)
                    lbl = str(wp.get("label") or f"Stop {idx_w + 1}")[:18]
                    lw, _ = measure_tracked_text(draw, lbl, f_wp)
                    draw_tracked_text(
                        img,
                        (wx - lw // 2, wy + int(round(10 * s))),
                        lbl,
                        font=f_wp,
                        fill=(*title_rgb, int(round(240 * master_alpha))) if is_act else (*sub_rgb, int(round(205 * master_alpha))),
                        shadow=False,
                        hdr_safe=use_hdr_safe,
                    )
                ty = pt + int(round(82 * s))
            else:
                node_y = pt + int(round(32 * s))
                if n_wp > 1:
                    draw.line(
                        [(x_start, node_y), (x_end, node_y)],
                        fill=clamp_rgba_for_hdr(_scale_rgba_alpha((120, 126, 138, 165), master_alpha), hdr_safe=use_hdr_safe),
                        width=max(2, int(round(2.0 * s))),
                    )
                for idx_w, wp in enumerate(waypoints):
                    wx = x_start + int(round((x_end - x_start) * (idx_w / max(1, n_wp - 1)))) if n_wp > 1 else (pl + pw // 2)
                    is_act = bool(wp.get("active"))
                    is_comp = bool(wp.get("completed")) or is_act
                    rad_o = int(round((8 if is_act else 5) * s))
                    if is_act:
                        draw.ellipse(
                            [wx - rad_o - 4, node_y - rad_o - 4, wx + rad_o + 4, node_y + rad_o + 4],
                            fill=clamp_rgba_for_hdr(_scale_rgba_alpha((*accent_rgba[:3], 68), master_alpha), hdr_safe=use_hdr_safe),
                        )
                    n_fill = clamp_rgba_for_hdr(
                        _scale_rgba_alpha((*title_rgb, 255) if is_act else ((186, 192, 202, 225) if is_comp else (96, 102, 112, 190)), master_alpha),
                        hdr_safe=use_hdr_safe,
                    )
                    draw.ellipse([wx - rad_o, node_y - rad_o, wx + rad_o, node_y + rad_o], fill=n_fill)
                    lbl = str(wp.get("label") or f"Stop {idx_w + 1}")[:18]
                    lw, _ = measure_tracked_text(draw, lbl, f_wp)
                    draw_tracked_text(
                        img,
                        (wx - lw // 2, node_y + int(round(11 * s))),
                        lbl,
                        font=f_wp,
                        fill=(*title_rgb, int(round(240 * master_alpha))) if is_act else (*sub_rgb, int(round(205 * master_alpha))),
                        shadow=False,
                        hdr_safe=use_hdr_safe,
                    )
                ty = pt + int(round(68 * s))

            tw, _ = measure_tracked_text(draw, item.text, f_title, letter_spacing_px=dyn_track * 0.4)
            draw_tracked_text(
                img,
                (pl + (pw - tw) // 2, ty + int(round(12 * s * (1.0 - enter_p)))),
                item.text,
                font=f_title,
                fill=title_fill,
                letter_spacing_px=dyn_track * 0.4,
                scale=s,
                hdr_safe=use_hdr_safe,
            )
            if sub_text:
                sw, _ = measure_tracked_text(draw, sub_text, f_sub, letter_spacing_px=1.2 * s)
                draw_tracked_text(
                    img,
                    (pl + (pw - sw) // 2, ty + int(round(30 * s))),
                    sub_text,
                    font=f_sub,
                    fill=sub_fill,
                    letter_spacing_px=1.2 * s,
                    scale=s,
                    hdr_safe=use_hdr_safe,
                )

    # ------------------------------------------------------------------
    # Archetype 5b: Upper-Left Museum Editorial Wall-Note Card
    # ------------------------------------------------------------------
    elif item.style == "editorial_caption":
        rect = compute_glass_pill_rect(item, width, height)
        if rect is not None:
            pl, pt, pw, ph, rad = rect
            _draw_card_container(img, rect, spec, use_hdr_safe, master_alpha, transparent_glass_body, item.backdrop_blur)
            draw = ImageDraw.Draw(img)
            eyebrow_str = _format_tracked_eyebrow(overline_text)
            f_eye = _load_font(int(round(12 * s)), bold=True, serif=False, text=eyebrow_str)
            inner_max_w = pw - int(round(48 * s))
            f_title = _load_fitted_font(draw, item.text, 25.0, 19.0, inner_max_w, scale=s, bold=True, serif=(spec["primary_tier"] == "serif"))
            f_sub = _load_font(int(round(17 * s)), bold=False, serif=False, text=sub_text)

            title_lines = _fit_text_lines(draw, item.text, f_title, inner_max_w, max_lines=2)
            sub_lines = _fit_text_lines(draw, sub_text, f_sub, inner_max_w, max_lines=3) if sub_text else []

            bar_x = pl + int(round(15 * s))
            bar_w = max(2, int(round(3.0 * s)))
            bar_h = int(round(max(8.0, (ph - int(round(28 * s))) * rule_e)))
            draw.rounded_rectangle(
                [bar_x, pt + int(round(14 * s)), bar_x + bar_w, pt + int(round(14 * s)) + bar_h],
                radius=max(1, bar_w // 2),
                fill=accent_rgba,
            )

            tx = bar_x + bar_w + int(round(15 * s))
            ty = pt + int(round(15 * s))
            eye_h = max(14, int(round(17 * s))) if eyebrow_str else 0
            title_line_h = max(20, int(round(31 * s)))
            sub_line_h = max(15, int(round(23 * s)))
            if eyebrow_str:
                draw_tracked_text(
                    img,
                    (tx, ty),
                    eyebrow_str,
                    font=f_eye,
                    fill=(*sub_rgb, int(round(226 * sub_e * exit_alpha))),
                    scale=s,
                    hdr_safe=use_hdr_safe,
                )
                ty += eye_h + int(round(5 * s))
            rise_off = int(round(12 * s * (1.0 - enter_p)))
            for ln in title_lines:
                draw_tracked_text(
                    img,
                    (tx, ty + rise_off),
                    ln,
                    font=f_title,
                    fill=title_fill,
                    letter_spacing_px=dyn_track * 0.25,
                    scale=s,
                    hdr_safe=use_hdr_safe,
                )
                ty += title_line_h
            if sub_lines:
                ty += int(round(3 * s))
                for ln in sub_lines:
                    draw_tracked_text(
                        img,
                        (tx, ty + int(round(6 * s * (1.0 - sub_e)))),
                        ln,
                        font=f_sub,
                        fill=sub_fill,
                        scale=s,
                        hdr_safe=use_hdr_safe,
                    )
                    ty += sub_line_h

    # ------------------------------------------------------------------
    # Archetype 5: Dialogue Quote Badge (Upper-Left)
    # ------------------------------------------------------------------
    elif item.style == "dialogue_badge":
        rect = compute_glass_pill_rect(item, width, height)
        if rect is not None:
            pl, pt, pw, ph, rad = rect
            _draw_card_container(img, rect, spec, use_hdr_safe, master_alpha, transparent_glass_body, item.backdrop_blur)
            draw = ImageDraw.Draw(img)
            f_title = _load_font_tier(int(round(23 * s)), tier=spec["primary_tier"], bold=spec["primary_bold"])
            f_sub = _load_font_tier(int(round(16 * s)), tier=spec["secondary_tier"], bold=False)

            bar_x = pl + int(round(14 * s))
            bar_top = pt + int(round(12 * s))
            bar_bot = bar_top + int(round(max(8.0, (ph - int(round(24 * s))) * rule_e)))
            draw.line([(bar_x, bar_top), (bar_x, bar_bot)], fill=accent_rgba, width=max(2, int(round(2.5 * s))))

            tx = bar_x + int(round(14 * s))
            ty = pt + int(round(14 * s))
            draw_tracked_text(
                img,
                (tx, ty + int(round(14 * s * (1.0 - enter_p)))),
                item.text,
                font=f_title,
                fill=title_fill,
                letter_spacing_px=dyn_track * 0.3,
                scale=s,
                hdr_safe=use_hdr_safe,
                clip_box=(tx - 2, ty - 2, pl + pw - 10, ty + int(round(34 * s))),
            )
            if sub_text:
                draw_tracked_text(
                    img,
                    (tx, ty + int(round(32 * s))),
                    sub_text[:54],
                    font=f_sub,
                    fill=sub_fill,
                    scale=s,
                    hdr_safe=use_hdr_safe,
                )

    # ------------------------------------------------------------------
    # Archetype 6: Archival EXIF & Telemetry Stamp (Upper-Right)
    # ------------------------------------------------------------------
    elif item.style == "exif_stamp":
        rect = compute_glass_pill_rect(item, width, height)
        if rect is not None:
            pl, pt, pw, ph, rad = rect
            _draw_card_container(img, rect, spec, use_hdr_safe, master_alpha, transparent_glass_body, item.backdrop_blur)
            draw = ImageDraw.Draw(img)
            f_stamp = _load_font_tier(int(round(17 * s)), tier="mono", bold=True)
            f_sub = _load_font_tier(int(round(14 * s)), tier="mono", bold=False)

            is_typewriter = getattr(item, "motion_preset", "") == "typewriter_telemetry"
            if is_typewriter and t_norm < 0.45:
                n_chars = max(1, int(round(len(item.text) * min(1.0, (t_norm / 0.30)))))
                disp_main = item.text[:n_chars] + ("_" if n_chars < len(item.text) else "")
            else:
                disp_main = item.text

            tx = pl + int(round(18 * s))
            ty = pt + int(round(9 * s))
            draw_tracked_text(
                img,
                (tx, ty),
                disp_main,
                font=f_stamp,
                fill=(*title_rgb, int(round(240 * master_alpha))),
                letter_spacing_px=1.5 * s,
                scale=s,
                hdr_safe=use_hdr_safe,
            )
            if sub_text:
                draw_tracked_text(
                    img,
                    (tx, ty + int(round(22 * s))),
                    sub_text,
                    font=f_sub,
                    fill=sub_fill,
                    letter_spacing_px=1.0 * s,
                    scale=s,
                    hdr_safe=use_hdr_safe,
                )

    # ------------------------------------------------------------------
    # Archetype 7: Frosted-Glass & Editorial Lower-Third (Positioned Above Subtitle Lane)
    # ------------------------------------------------------------------
    else:
        _draw_bottom_scrim(img, scrim_height_ratio=0.22, max_alpha=int(round(100 * master_alpha)), hdr_safe=use_hdr_safe)
        rect = compute_glass_pill_rect(item, width, height)
        if rect is not None:
            pl, pt, pw, ph, rad = rect
            _draw_card_container(img, rect, spec, use_hdr_safe, master_alpha, transparent_glass_body, item.backdrop_blur)
            draw = ImageDraw.Draw(img)

            f_over = _load_font_tier(int(round(12 * s)), tier=spec["overline_tier"], bold=True)
            f_title = _load_font_tier(int(round(27 * s)), tier=spec["primary_tier"], bold=spec["primary_bold"])
            f_sub = _load_font_tier(int(round(18 * s)), tier=spec["secondary_tier"], bold=False)
            sub_lines = textwrap.wrap(sub_text, width=74)[:2] if sub_text else []

            bar_x = pl + int(round(16 * s))
            bar_w = max(2, int(round(2.5 * s)))
            bar_h = int(round(max(6.0, (ph - int(round(28 * s))) * rule_e)))
            draw.rounded_rectangle(
                [bar_x, pt + int(round(14 * s)), bar_x + bar_w, pt + int(round(14 * s)) + bar_h],
                radius=max(1, bar_w // 2),
                fill=accent_rgba,
            )

            tx = bar_x + bar_w + int(round(16 * s))
            ty = pt + int(round(14 * s))
            if overline_text:
                draw_tracked_text(
                    img,
                    (tx, ty),
                    overline_text.upper(),
                    font=f_over,
                    fill=_scale_rgba_alpha(accent_rgba, sub_e),
                    letter_spacing_px=spec["overline_tracking_px"] * s,
                    scale=s,
                    hdr_safe=use_hdr_safe,
                )
                ty += int(round(18 * s))

            rise_off = int(round(18 * s * (1.0 - enter_p)))
            draw_tracked_text(
                img,
                (tx, ty + rise_off),
                item.text,
                font=f_title,
                fill=title_fill,
                letter_spacing_px=dyn_track * 0.45,
                scale=s,
                hdr_safe=use_hdr_safe,
                clip_box=(tx - 4, ty - 4, pl + pw - 14, ty + int(round(38 * s))),
            )
            ty += max(14, int(round(35 * s)))
            if sub_lines:
                ty += max(2, int(round(4 * s)))
                for ln in sub_lines:
                    draw_tracked_text(
                        img,
                        (tx, ty + int(round(8 * s * (1.0 - sub_e)))),
                        ln,
                        font=f_sub,
                        fill=sub_fill,
                        scale=s,
                        hdr_safe=use_hdr_safe,
                    )
                    ty += max(12, int(round(25 * s)))

    if use_hdr_safe:
        arr = np.asarray(img, dtype=np.uint8).copy()
        arr[:, :, :3] = np.minimum(arr[:, :, :3], HDR_DIFFUSE_WHITE_PEAK_RGB)
        dark_mask = (arr[:, :, 0] < 45) & (arr[:, :, 1] < 45) & (arr[:, :, 2] < 45)
        arr[:, :, 3] = np.where(dark_mask, np.minimum(arr[:, :, 3], HDR_MAX_SHADOW_ALPHA), arr[:, :, 3])
        img = Image.fromarray(arr, mode="RGBA")

    return img


def render_title_png(
    item: TitleOverlayItem,
    output_png: str | Path,
    width: int = 1920,
    height: int = 1080,
    hdr_safe: Optional[bool] = None,
) -> Path:
    """Render the settled hero keyframe of a TitleOverlayItem to a transparent RGBA PNG plate."""
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = render_title_frame(
        item=item,
        width=width,
        height=height,
        t_norm=0.55,
        hdr_safe=hdr_safe,
        transparent_glass_body=False,
    )
    img.save(out_path, "PNG")
    return out_path
