"""Programmatic RGBA graphic title cards, Chapter banners, EXIF stamps, and Lower-Third overlays via Pillow."""

from __future__ import annotations

import os
from pathlib import Path
import textwrap
from typing import Optional

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from ..models.schema import TitleOverlayItem

# ~75% HLG signal (~203 nits diffuse white) prevents painful 1000-nit text bloom on HDR displays
HDR_DIFFUSE_WHITE_PEAK_RGB = 192
HDR_MAX_SHADOW_ALPHA = 152


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
    scale = float(peak_white) / 255.0
    max_ch = max(r, g, b)
    if max_ch > peak_white:
        return (
            int(round(r * scale)),
            int(round(g * scale)),
            int(round(b * scale)),
            int(a),
        )
    return (int(r), int(g), int(b), int(a))


def _load_font(size: int, bold: bool = True, serif: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if serif:
        candidates = [
            "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Avenir Next.ttc",
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, max(10, size))
            except Exception:
                continue
    return ImageFont.load_default()


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


def _draw_bottom_scrim(
    img: Image.Image,
    scrim_height_ratio: float = 0.24,
    max_alpha: int = 115,
    hdr_safe: bool = False,
) -> None:
    """Draw a smooth vertical bottom gradient scrim on an RGBA overlay so lower-third captions blend naturally."""
    w, h = img.size
    eff_alpha = min(max_alpha, HDR_MAX_SHADOW_ALPHA) if hdr_safe else max_alpha
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
    """Render typography with a soft two-pass cinema drop shadow for clean legibility without heavy boxes."""
    x, y = xy
    off1 = max(1, int(round(1.5 * scale)))
    off2 = max(2, int(round(3.0 * scale)))
    sh_outer = clamp_rgba_for_hdr((0, 0, 0, 115), hdr_safe=hdr_safe)
    sh_inner = clamp_rgba_for_hdr((0, 0, 0, 185), hdr_safe=hdr_safe)
    fg_fill = clamp_rgba_for_hdr(fill, hdr_safe=hdr_safe)
    draw.text((x + off2, y + off2), text, font=font, fill=sh_outer)
    draw.text((x + off1, y + off1), text, font=font, fill=sh_inner)
    draw.text((x, y), text, font=font, fill=fg_fill)


def compute_overlay_bbox_norm(style: str) -> list[float]:
    """Return canonical normalized 2D screen bounding box [x1, y1, x2, y2] for overlap QC."""
    if style == "closing_card":
        return [0.18, 0.36, 0.82, 0.64]
    if style in ("opening_card", "act_banner"):
        return [0.16, 0.68, 0.84, 0.86]
    if style == "route_map_card":
        return [0.12, 0.80, 0.88, 0.96]
    if style == "dialogue_badge":
        return [0.06, 0.10, 0.56, 0.24]
    if style == "exif_stamp":
        return [0.68, 0.06, 0.96, 0.14]
    # lower_third positioned slightly higher when subtitles are active, default bottom third
    return [0.14, 0.72, 0.86, 0.84]


def render_title_png(
    item: TitleOverlayItem,
    output_png: str | Path,
    width: int = 1920,
    height: int = 1080,
    hdr_safe: Optional[bool] = None,
) -> Path:
    """Render a TitleOverlayItem to a transparent RGBA PNG plate using an A24 / Criterion editorial aesthetic."""
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_hdr_safe = bool(item.hdr_safe) if hdr_safe is None else bool(hdr_safe)

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    s = width / 1920.0

    accent_rule_rgba = _parse_rgba(item.accent_color, 165, hdr_safe=use_hdr_safe)
    sub_text = (item.subtitle or "").strip()

    if item.style in ("opening_card", "closing_card", "act_banner"):
        if item.style == "closing_card":
            veil_a = min(92, HDR_MAX_SHADOW_ALPHA) if use_hdr_safe else 92
            veil = Image.new("RGBA", (width, height), (8, 10, 14, veil_a))
            img = Image.alpha_composite(img, veil)
            cy = int(round(height * 0.50))
            title_pt = 48
        else:
            _draw_bottom_scrim(img, scrim_height_ratio=0.34, max_alpha=135, hdr_safe=use_hdr_safe)
            cy = int(round(height * 0.76))
            title_pt = 44 if item.style == "opening_card" else 36

        draw = ImageDraw.Draw(img)
        f_title = _load_font(int(round(title_pt * s)), bold=True, serif=True)
        f_sub = _load_font(int(round(21 * s)), bold=False, serif=False)

        bt = draw.textbbox((0, 0), item.text, font=f_title)
        tw = bt[2] - bt[0]
        sw = 0
        if sub_text:
            bs = draw.textbbox((0, 0), sub_text, font=f_sub)
            sw = bs[2] - bs[0]

        cx = width // 2
        rule_w = max(36, int(round(68 * s)))
        rule_y = cy - int(round((46 if sub_text else 30) * s))
        draw.line(
            [(cx - rule_w // 2, rule_y), (cx + rule_w // 2, rule_y)],
            fill=accent_rule_rgba,
            width=max(1, int(round(1.5 * s))),
        )

        ty = cy - int(round((28 if sub_text else 12) * s))
        _draw_cinema_text(
            draw,
            (cx - tw // 2, ty),
            item.text,
            font=f_title,
            fill=(250, 248, 244, 255),
            scale=s,
            hdr_safe=use_hdr_safe,
        )
        if sub_text:
            _draw_cinema_text(
                draw,
                (cx - sw // 2, ty + int(round((title_pt + 14) * s))),
                sub_text,
                font=f_sub,
                fill=(204, 210, 218, 236),
                scale=s,
                hdr_safe=use_hdr_safe,
            )

    elif item.style == "route_map_card":
        _draw_bottom_scrim(img, scrim_height_ratio=0.26, max_alpha=115, hdr_safe=use_hdr_safe)
        draw = ImageDraw.Draw(img)

        f_title = _load_font(int(round(25 * s)), bold=True, serif=True)
        f_sub = _load_font(int(round(16 * s)), bold=False, serif=False)
        f_wp = _load_font(int(round(14 * s)), bold=True, serif=False)

        waypoints = item.map_waypoints or [
            {"label": "Departure", "completed": True, "active": False},
            {"label": item.text[:20], "completed": True, "active": True},
            {"label": "Finale", "completed": False, "active": False},
        ]

        pw = min(int(width * 0.78), max(int(round(580 * s)), len(waypoints) * int(round(170 * s))))
        ph = int(round(126 * s))
        pl = (width - pw) // 2
        pt = height - ph - int(round(36 * s))
        pr, pb = pl + pw, pt + ph
        radius = int(round(14 * s))

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=radius,
            fill=clamp_rgba_for_hdr((12, 14, 18, 152), hdr_safe=use_hdr_safe),
            outline=clamp_rgba_for_hdr((255, 255, 255, 42), hdr_safe=use_hdr_safe),
            width=max(1, int(round(1.0 * s))),
        )

        n_wp = max(1, len(waypoints))
        track_pad = int(round(68 * s))
        node_y = pt + int(round(32 * s))
        x_start = pl + track_pad
        x_end = pr - track_pad
        if n_wp > 1:
            draw.line(
                [(x_start, node_y), (x_end, node_y)],
                fill=clamp_rgba_for_hdr((120, 126, 138, 165), hdr_safe=use_hdr_safe),
                width=max(2, int(round(2.0 * s))),
            )

        for idx_w, wp in enumerate(waypoints):
            wx = x_start + int(round((x_end - x_start) * (idx_w / max(1, n_wp - 1)))) if n_wp > 1 else (pl + pr) // 2
            is_act = bool(wp.get("active"))
            is_comp = bool(wp.get("completed")) or is_act
            rad_outer = int(round((8 if is_act else 5) * s))
            if is_act:
                draw.ellipse(
                    [wx - rad_outer - 4, node_y - rad_outer - 4, wx + rad_outer + 4, node_y + rad_outer + 4],
                    fill=clamp_rgba_for_hdr((232, 228, 220, 68), hdr_safe=use_hdr_safe),
                )
            node_fill = clamp_rgba_for_hdr(
                (250, 248, 244, 255) if is_act else ((186, 192, 202, 225) if is_comp else (96, 102, 112, 190)),
                hdr_safe=use_hdr_safe,
            )
            draw.ellipse([wx - rad_outer, node_y - rad_outer, wx + rad_outer, node_y + rad_outer], fill=node_fill)

            lbl = str(wp.get("label") or f"Stop {idx_w + 1}")[:18]
            lw = draw.textbbox((0, 0), lbl, font=f_wp)[2]
            lbl_col = clamp_rgba_for_hdr(
                (250, 248, 244, 245) if is_act else (168, 174, 184, 210),
                hdr_safe=use_hdr_safe,
            )
            draw.text((wx - lw // 2, node_y + int(round(11 * s))), lbl, font=f_wp, fill=lbl_col)

        tw = draw.textbbox((0, 0), item.text, font=f_title)[2]
        ty = pt + int(round(68 * s))
        _draw_cinema_text(
            draw,
            (pl + (pw - tw) // 2, ty),
            item.text,
            font=f_title,
            fill=(250, 248, 244, 255),
            scale=s,
            hdr_safe=use_hdr_safe,
        )
        if sub_text:
            sw = draw.textbbox((0, 0), sub_text, font=f_sub)[2]
            _draw_cinema_text(
                draw,
                (pl + (pw - sw) // 2, ty + int(round(30 * s))),
                sub_text,
                font=f_sub,
                fill=(196, 202, 212, 230),
                scale=s,
                hdr_safe=use_hdr_safe,
            )

    elif item.style == "dialogue_badge":
        # Upper-left editorial quote callout badge so it NEVER collides with bottom bilingual subtitles!
        draw = ImageDraw.Draw(img)
        f_title = _load_font(int(round(23 * s)), bold=True, serif=True)
        f_sub = _load_font(int(round(16 * s)), bold=False, serif=False)
        tw = draw.textbbox((0, 0), item.text, font=f_title)[2]
        sw = draw.textbbox((0, 0), sub_text, font=f_sub)[2] if sub_text else 0
        pw = min(int(width * 0.50), max(tw, sw) + int(round(56 * s)))
        ph = int(round((88 if sub_text else 58) * s))
        pl = int(round(width * 0.06))
        pt = int(round(height * 0.10))
        pr, pb = pl + pw, pt + ph
        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=int(round(12 * s)),
            fill=clamp_rgba_for_hdr((12, 14, 18, 138), hdr_safe=use_hdr_safe),
            outline=clamp_rgba_for_hdr((255, 255, 255, 34), hdr_safe=use_hdr_safe),
            width=max(1, int(round(1.0 * s))),
        )
        bar_x = pl + int(round(14 * s))
        draw.line(
            [(bar_x, pt + int(round(12 * s))), (bar_x, pb - int(round(12 * s)))],
            fill=accent_rule_rgba,
            width=max(2, int(round(2.5 * s))),
        )
        tx = bar_x + int(round(14 * s))
        ty = pt + int(round(14 * s))
        _draw_cinema_text(draw, (tx, ty), item.text, font=f_title, fill=(250, 248, 244, 255), scale=s, hdr_safe=use_hdr_safe)
        if sub_text:
            _draw_cinema_text(
                draw,
                (tx, ty + int(round(32 * s))),
                sub_text[:54],
                font=f_sub,
                fill=(196, 202, 212, 228),
                scale=s,
                hdr_safe=use_hdr_safe,
            )

    elif item.style == "exif_stamp":
        # Subtle upper-right archival location/date stamp (never overlaps bottom subtitles or lower-thirds)
        draw = ImageDraw.Draw(img)
        f_stamp = _load_font(int(round(17 * s)), bold=True, serif=False)
        f_sub = _load_font(int(round(14 * s)), bold=False, serif=False)
        tw = draw.textbbox((0, 0), item.text, font=f_stamp)[2]
        sw = draw.textbbox((0, 0), sub_text, font=f_sub)[2] if sub_text else 0
        pw = max(tw, sw) + int(round(36 * s))
        ph = int(round((58 if sub_text else 38) * s))
        pr = width - int(round(width * 0.04))
        pl = pr - pw
        pt = int(round(height * 0.06))
        pb = pt + ph
        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=int(round(8 * s)),
            fill=clamp_rgba_for_hdr((10, 12, 16, 122), hdr_safe=use_hdr_safe),
            outline=clamp_rgba_for_hdr((232, 228, 220, 32), hdr_safe=use_hdr_safe),
            width=1,
        )
        tx = pl + int(round(18 * s))
        ty = pt + int(round(9 * s))
        _draw_cinema_text(draw, (tx, ty), item.text, font=f_stamp, fill=(236, 232, 224, 240), scale=s, hdr_safe=use_hdr_safe)
        if sub_text:
            _draw_cinema_text(
                draw,
                (tx, ty + int(round(22 * s))),
                sub_text,
                font=f_sub,
                fill=(186, 192, 202, 220),
                scale=s,
                hdr_safe=use_hdr_safe,
            )

    else:
        # Refined smoked-glass documentary lower-third card positioned cleanly above the bottom subtitle lane
        _draw_bottom_scrim(img, scrim_height_ratio=0.22, max_alpha=100, hdr_safe=use_hdr_safe)
        draw = ImageDraw.Draw(img)

        f_title = _load_font(int(round(26 * s)), bold=True, serif=True)
        f_sub = _load_font(int(round(18 * s)), bold=False, serif=False)

        sub_lines = textwrap.wrap(sub_text, width=78)[:2] if sub_text else []

        tw = draw.textbbox((0, 0), item.text, font=f_title)[2]
        sw = max((draw.textbbox((0, 0), ln, font=f_sub)[2] for ln in sub_lines), default=0)

        pad_left = int(round(32 * s))
        pad_right = int(round(32 * s))
        pw = min(int(width * 0.80), max(tw, sw) + pad_left + pad_right + int(round(18 * s)))
        pw = max(pw, int(round(380 * s)))

        top_pad = int(round(15 * s))
        bottom_pad = int(round(15 * s))
        cur_h = top_pad + max(14, int(round(34 * s)))
        if sub_lines:
            cur_h += max(2, int(round(4 * s))) + len(sub_lines) * max(12, int(round(25 * s)))
        ph = cur_h + bottom_pad

        pl = (width - pw) // 2
        # Anchor above bottom 16% so lower-thirds never collide with bilingual subtitles at bottom 5-14%
        pt = int(round(height * 0.84)) - ph
        pr, pb = pl + pw, pt + ph
        radius = int(round(14 * s))

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=radius,
            fill=clamp_rgba_for_hdr((12, 14, 18, 142), hdr_safe=use_hdr_safe),
            outline=clamp_rgba_for_hdr((255, 255, 255, 36), hdr_safe=use_hdr_safe),
            width=max(1, int(round(1.0 * s))),
        )

        bar_x = pl + int(round(16 * s))
        bar_w = max(2, int(round(2.5 * s)))
        draw.rounded_rectangle(
            [bar_x, pt + int(round(14 * s)), bar_x + bar_w, pb - int(round(14 * s))],
            radius=max(1, bar_w // 2),
            fill=accent_rule_rgba,
        )

        tx = bar_x + bar_w + int(round(14 * s))
        ty = pt + top_pad

        _draw_cinema_text(draw, (tx, ty), item.text, font=f_title, fill=(250, 248, 244, 255), scale=s, hdr_safe=use_hdr_safe)
        ty += max(14, int(round(34 * s)))

        if sub_lines:
            ty += max(2, int(round(4 * s)))
            for ln in sub_lines:
                _draw_cinema_text(draw, (tx, ty), ln, font=f_sub, fill=(198, 204, 212, 236), scale=s, hdr_safe=use_hdr_safe)
                ty += max(12, int(round(25 * s)))

    if use_hdr_safe:
        arr = np.asarray(img, dtype=np.uint8).copy()
        arr[:, :, :3] = np.minimum(arr[:, :, :3], HDR_DIFFUSE_WHITE_PEAK_RGB)
        dark_mask = (arr[:, :, 0] < 45) & (arr[:, :, 1] < 45) & (arr[:, :, 2] < 45)
        arr[:, :, 3] = np.where(dark_mask, np.minimum(arr[:, :, 3], HDR_MAX_SHADOW_ALPHA), arr[:, :, 3])
        img = Image.fromarray(arr, mode="RGBA")

    img.save(out_path, "PNG")
    return out_path
