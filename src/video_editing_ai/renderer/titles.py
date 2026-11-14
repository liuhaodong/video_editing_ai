"""Programmatic RGBA graphic title cards, Chapter banners, EXIF stamps, and Lower-Third overlays via Pillow."""

from __future__ import annotations

import os
from pathlib import Path
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


def _contains_cjk(text: str) -> bool:
    """Return True when text contains CJK/East Asian ideographs or punctuation."""
    return any(
        "\u2e80" <= ch <= "\u9fff"
        or "\u3000" <= ch <= "\u303f"
        or "\uf900" <= ch <= "\ufaff"
        or "\uff00" <= ch <= "\uffef"
        for ch in (text or "")
    )


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


def _format_tracked_eyebrow(raw: str) -> str:
    """Uppercase editorial kicker with subtle Latin letter-spacing while keeping CJK compact."""
    txt = " ".join((raw or "").strip().split())
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


def compute_overlay_bbox_norm(style: str) -> list[float]:
    """Return canonical normalized 2D screen bounding box [x1, y1, x2, y2] for overlap QC."""
    if style == "closing_card":
        return [0.16, 0.34, 0.84, 0.66]
    if style in ("opening_card", "act_banner"):
        return [0.14, 0.64, 0.86, 0.83]
    if style == "editorial_caption":
        return [0.05, 0.07, 0.62, 0.23]
    if style == "route_map_card":
        return [0.12, 0.79, 0.88, 0.96]
    if style == "dialogue_badge":
        return [0.06, 0.09, 0.58, 0.24]
    if style == "exif_stamp":
        return [0.66, 0.06, 0.96, 0.15]
    # lower_third positioned cleanly above the [0.85..0.96] bottom subtitle lane
    return [0.14, 0.70, 0.86, 0.84]


def render_title_png(
    item: TitleOverlayItem,
    output_png: str | Path,
    width: int = 1920,
    height: int = 1080,
    hdr_safe: Optional[bool] = None,
) -> Path:
    """Render a TitleOverlayItem to a transparent RGBA PNG plate using an A24 / Criterion / SFMOMA editorial aesthetic."""
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_hdr_safe = bool(item.hdr_safe) if hdr_safe is None else bool(hdr_safe)

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    s = width / 1920.0

    accent_rule_rgba = _parse_rgba(item.accent_color, 185, hdr_safe=use_hdr_safe)
    sub_text = (item.subtitle or "").strip()
    raw_eyebrow = (getattr(item, "eyebrow", None) or "").strip()

    if item.style in ("opening_card", "closing_card", "act_banner"):
        draw = ImageDraw.Draw(img)
        if item.style == "closing_card":
            veil_a = min(96, HDR_MAX_SHADOW_ALPHA) if use_hdr_safe else 96
            veil = Image.new("RGBA", (width, height), (8, 10, 14, veil_a))
            img = Image.alpha_composite(img, veil)
            draw = ImageDraw.Draw(img)
            default_eyebrow = "MEMORIES PRESERVED"
            base_pt, min_pt, sub_pt = 44.0, 30.0, 20.0
        else:
            _draw_bottom_scrim(img, scrim_height_ratio=0.30, max_alpha=112, hdr_safe=use_hdr_safe)
            draw = ImageDraw.Draw(img)
            default_eyebrow = "CURATED DOCUMENTARY FILM" if item.style == "opening_card" else "CHAPTER"
            base_pt = 38.0 if item.style == "opening_card" else 33.0
            min_pt = 26.0
            sub_pt = 19.0

        eyebrow_str = _format_tracked_eyebrow(raw_eyebrow or default_eyebrow)
        f_eye = _load_font(int(round(12.5 * s)), bold=True, serif=False, text=eyebrow_str)
        max_card_w = int(width * 0.72)
        inner_max_w = max_card_w - int(round(92 * s))
        f_title = _load_fitted_font(draw, item.text, base_pt, min_pt, inner_max_w, scale=s, bold=True, serif=True)
        f_sub = _load_font(int(round(sub_pt * s)), bold=False, serif=False, text=sub_text)

        title_lines = _fit_text_lines(draw, item.text, f_title, inner_max_w, max_lines=2)
        sub_lines = _fit_text_lines(draw, sub_text, f_sub, inner_max_w, max_lines=2) if sub_text else []

        ew = _measure_text(draw, eyebrow_str, f_eye)[0] if eyebrow_str else 0
        tw = max((_measure_text(draw, ln, f_title)[0] for ln in title_lines), default=0)
        sw = max((_measure_text(draw, ln, f_sub)[0] for ln in sub_lines), default=0)

        pw = min(max_card_w, max(int(round(560 * s)), max(ew, tw, sw) + int(round(96 * s))))
        eye_h = max(14, int(round(18 * s))) if eyebrow_str else 0
        rule_gap = int(round(12 * s))
        title_line_h = max(24, int(round((base_pt + 8) * s)))
        sub_line_h = max(16, int(round((sub_pt + 8) * s)))
        pad_y = int(round(20 * s))
        cur_h = (
            pad_y * 2
            + (eye_h + int(round(8 * s)) if eyebrow_str else 0)
            + rule_gap
            + len(title_lines) * title_line_h
            + (int(round(6 * s)) + len(sub_lines) * sub_line_h if sub_lines else 0)
        )
        ph = cur_h
        pl = (width - pw) // 2
        if item.style == "closing_card":
            pt = (height - ph) // 2
        else:
            pt = int(round(height * 0.825)) - ph
        pr, pb = pl + pw, pt + ph
        radius = int(round(16 * s))

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=radius,
            fill=clamp_rgba_for_hdr((10, 12, 16, 148), hdr_safe=use_hdr_safe),
            outline=clamp_rgba_for_hdr((232, 228, 220, 56), hdr_safe=use_hdr_safe),
            width=max(1, int(round(1.2 * s))),
        )
        inset = max(4, int(round(5 * s)))
        draw.rounded_rectangle(
            [pl + inset, pt + inset, pr - inset, pb - inset],
            radius=max(4, radius - inset),
            outline=clamp_rgba_for_hdr((255, 255, 255, 22), hdr_safe=use_hdr_safe),
            width=1,
        )

        cx = width // 2
        y_cur = pt + pad_y
        if eyebrow_str:
            _draw_cinema_text(
                draw,
                (cx - ew // 2, y_cur),
                eyebrow_str,
                font=f_eye,
                fill=(214, 210, 200, 232),
                scale=s,
                hdr_safe=use_hdr_safe,
            )
            y_cur += eye_h + int(round(6 * s))

        rule_w = max(44, int(round(72 * s)))
        draw.line(
            [(cx - rule_w // 2, y_cur), (cx + rule_w // 2, y_cur)],
            fill=accent_rule_rgba,
            width=max(1, int(round(1.5 * s))),
        )
        y_cur += rule_gap

        for ln in title_lines:
            lw = _measure_text(draw, ln, f_title)[0]
            _draw_cinema_text(
                draw,
                (cx - lw // 2, y_cur),
                ln,
                font=f_title,
                fill=(250, 248, 244, 255),
                scale=s,
                hdr_safe=use_hdr_safe,
            )
            y_cur += title_line_h

        if sub_lines:
            y_cur += int(round(4 * s))
            for ln in sub_lines:
                lsw = _measure_text(draw, ln, f_sub)[0]
                _draw_cinema_text(
                    draw,
                    (cx - lsw // 2, y_cur),
                    ln,
                    font=f_sub,
                    fill=(204, 210, 218, 236),
                    scale=s,
                    hdr_safe=use_hdr_safe,
                )
                y_cur += sub_line_h

    elif item.style == "editorial_caption":
        # Upper-left Museum Curatorial Wall-Note Card: zero collision with bottom photo plaques & subtitles!
        draw = ImageDraw.Draw(img)
        eyebrow_str = _format_tracked_eyebrow(raw_eyebrow)
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
        pr, pb = pl + pw, pt + ph
        radius = int(round(13 * s))

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=radius,
            fill=clamp_rgba_for_hdr((10, 12, 16, 148), hdr_safe=use_hdr_safe),
            outline=clamp_rgba_for_hdr((236, 232, 224, 44), hdr_safe=use_hdr_safe),
            width=max(1, int(round(1.1 * s))),
        )
        bar_x = pl + int(round(15 * s))
        bar_w = max(2, int(round(3.0 * s)))
        draw.rounded_rectangle(
            [bar_x, pt + int(round(14 * s)), bar_x + bar_w, pb - int(round(14 * s))],
            radius=max(1, bar_w // 2),
            fill=accent_rule_rgba,
        )

        tx = bar_x + bar_w + int(round(15 * s))
        ty = pt + pad_y
        if eyebrow_str:
            _draw_cinema_text(draw, (tx, ty), eyebrow_str, font=f_eye, fill=(208, 210, 216, 226), scale=s, hdr_safe=use_hdr_safe)
            ty += eye_h + int(round(5 * s))
        for ln in title_lines:
            _draw_cinema_text(draw, (tx, ty), ln, font=f_title, fill=(250, 248, 244, 255), scale=s, hdr_safe=use_hdr_safe)
            ty += title_line_h
        if sub_lines:
            ty += int(round(3 * s))
            for ln in sub_lines:
                _draw_cinema_text(draw, (tx, ty), ln, font=f_sub, fill=(202, 208, 216, 236), scale=s, hdr_safe=use_hdr_safe)
                ty += sub_line_h

    elif item.style == "route_map_card":
        _draw_bottom_scrim(img, scrim_height_ratio=0.26, max_alpha=112, hdr_safe=use_hdr_safe)
        draw = ImageDraw.Draw(img)

        f_sub = _load_font(int(round(16 * s)), bold=False, serif=False, text=sub_text)
        f_wp = _load_font(int(round(14 * s)), bold=True, serif=False)

        waypoints = item.map_waypoints or [
            {"label": "Departure", "completed": True, "active": False},
            {"label": item.text[:20], "completed": True, "active": True},
            {"label": "Finale", "completed": False, "active": False},
        ]

        pw = min(int(width * 0.78), max(int(round(660 * s)), len(waypoints) * int(round(190 * s))))
        inner_max_w = pw - int(round(64 * s))
        f_title = _load_fitted_font(draw, item.text, 24.0, 19.0, inner_max_w, scale=s, bold=True, serif=True)
        ph = int(round(132 * s))
        pl = (width - pw) // 2
        pt = height - ph - int(round(34 * s))
        pr, pb = pl + pw, pt + ph
        radius = int(round(14 * s))

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=radius,
            fill=clamp_rgba_for_hdr((12, 14, 18, 150), hdr_safe=use_hdr_safe),
            outline=clamp_rgba_for_hdr((236, 232, 224, 46), hdr_safe=use_hdr_safe),
            width=max(1, int(round(1.1 * s))),
        )

        n_wp = max(1, len(waypoints))
        track_pad = int(round(88 * s))
        node_y = pt + int(round(30 * s))
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

            lbl = str(wp.get("label") or f"Stop {idx_w + 1}")[:20]
            lw = _measure_text(draw, lbl, f_wp)[0]
            lbl_col = clamp_rgba_for_hdr(
                (250, 248, 244, 245) if is_act else (168, 174, 184, 210),
                hdr_safe=use_hdr_safe,
            )
            draw.text((wx - lw // 2, node_y + int(round(11 * s))), lbl, font=f_wp, fill=lbl_col)

        tw = _measure_text(draw, item.text, f_title)[0]
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
            sub_clipped = _fit_text_lines(draw, sub_text, f_sub, inner_max_w, max_lines=1)[0]
            sw = _measure_text(draw, sub_clipped, f_sub)[0]
            _draw_cinema_text(
                draw,
                (pl + (pw - sw) // 2, ty + int(round(30 * s))),
                sub_clipped,
                font=f_sub,
                fill=(196, 202, 212, 230),
                scale=s,
                hdr_safe=use_hdr_safe,
            )

    elif item.style == "dialogue_badge":
        # Upper-left editorial quote callout badge so it NEVER collides with bottom bilingual subtitles!
        draw = ImageDraw.Draw(img)
        eyebrow_str = _format_tracked_eyebrow(raw_eyebrow)
        f_eye = _load_font(int(round(11.5 * s)), bold=True, serif=False, text=eyebrow_str)
        max_card_w = int(width * 0.52)
        inner_max_w = max_card_w - int(round(64 * s))
        f_title = _load_fitted_font(draw, item.text, 23.0, 18.0, inner_max_w, scale=s, bold=True, serif=True)
        f_sub = _load_font(int(round(16 * s)), bold=False, serif=False, text=sub_text)

        ew = _measure_text(draw, eyebrow_str, f_eye)[0] if eyebrow_str else 0
        tw = _measure_text(draw, item.text, f_title)[0]
        sub_lines = _fit_text_lines(draw, sub_text, f_sub, inner_max_w, max_lines=1) if sub_text else []
        sw = _measure_text(draw, sub_lines[0], f_sub)[0] if sub_lines else 0

        pw = min(max_card_w, max(int(round(300 * s)), max(ew, tw, sw) + int(round(64 * s))))
        eye_h = max(13, int(round(16 * s))) if eyebrow_str else 0
        pad_y = int(round(14 * s))
        ph = pad_y * 2 + (eye_h + int(round(4 * s)) if eyebrow_str else 0) + int(round(30 * s)) + (int(round(24 * s)) if sub_lines else 0)
        pl = int(round(width * 0.06))
        pt = int(round(height * 0.09))
        pr, pb = pl + pw, pt + ph
        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=int(round(12 * s)),
            fill=clamp_rgba_for_hdr((12, 14, 18, 144), hdr_safe=use_hdr_safe),
            outline=clamp_rgba_for_hdr((255, 255, 255, 40), hdr_safe=use_hdr_safe),
            width=max(1, int(round(1.1 * s))),
        )
        bar_x = pl + int(round(14 * s))
        bar_w = max(2, int(round(2.5 * s)))
        draw.rounded_rectangle(
            [bar_x, pt + int(round(12 * s)), bar_x + bar_w, pb - int(round(12 * s))],
            radius=max(1, bar_w // 2),
            fill=accent_rule_rgba,
        )
        tx = bar_x + bar_w + int(round(14 * s))
        ty = pt + pad_y
        if eyebrow_str:
            _draw_cinema_text(draw, (tx, ty), eyebrow_str, font=f_eye, fill=(208, 210, 216, 224), scale=s, hdr_safe=use_hdr_safe)
            ty += eye_h + int(round(4 * s))
        _draw_cinema_text(draw, (tx, ty), item.text, font=f_title, fill=(250, 248, 244, 255), scale=s, hdr_safe=use_hdr_safe)
        if sub_lines:
            _draw_cinema_text(
                draw,
                (tx, ty + int(round(31 * s))),
                sub_lines[0],
                font=f_sub,
                fill=(196, 202, 212, 228),
                scale=s,
                hdr_safe=use_hdr_safe,
            )

    elif item.style == "exif_stamp":
        # Subtle upper-right archival location/date stamp (never overlaps bottom subtitles or lower-thirds)
        draw = ImageDraw.Draw(img)
        eyebrow_str = _format_tracked_eyebrow(raw_eyebrow)
        f_eye = _load_font(int(round(11 * s)), bold=True, serif=False, text=eyebrow_str)
        f_stamp = _load_font(int(round(17 * s)), bold=True, serif=False, text=item.text)
        f_sub = _load_font(int(round(14 * s)), bold=False, serif=False, text=sub_text)
        ew = _measure_text(draw, eyebrow_str, f_eye)[0] if eyebrow_str else 0
        tw = _measure_text(draw, item.text, f_stamp)[0]
        sw = _measure_text(draw, sub_text, f_sub)[0] if sub_text else 0
        pw = max(ew, tw, sw) + int(round(38 * s))
        eye_h = max(12, int(round(15 * s))) if eyebrow_str else 0
        ph = int(round(18 * s)) + (eye_h + int(round(4 * s)) if eyebrow_str else 0) + int(round(22 * s)) + (int(round(20 * s)) if sub_text else 0)
        pr = width - int(round(width * 0.04))
        pl = pr - pw
        pt = int(round(height * 0.06))
        pb = pt + ph
        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=int(round(9 * s)),
            fill=clamp_rgba_for_hdr((10, 12, 16, 134), hdr_safe=use_hdr_safe),
            outline=clamp_rgba_for_hdr((232, 228, 220, 38), hdr_safe=use_hdr_safe),
            width=max(1, int(round(1.0 * s))),
        )
        tx = pl + int(round(19 * s))
        ty = pt + int(round(9 * s))
        if eyebrow_str:
            _draw_cinema_text(draw, (tx, ty), eyebrow_str, font=f_eye, fill=(208, 210, 216, 224), scale=s, hdr_safe=use_hdr_safe)
            ty += eye_h + int(round(4 * s))
        _draw_cinema_text(draw, (tx, ty), item.text, font=f_stamp, fill=(242, 238, 230, 246), scale=s, hdr_safe=use_hdr_safe)
        if sub_text:
            _draw_cinema_text(
                draw,
                (tx, ty + int(round(22 * s))),
                sub_text,
                font=f_sub,
                fill=(192, 198, 208, 226),
                scale=s,
                hdr_safe=use_hdr_safe,
            )

    else:
        # Refined smoked-glass documentary lower-third card positioned cleanly above the bottom subtitle lane
        _draw_bottom_scrim(img, scrim_height_ratio=0.20, max_alpha=92, hdr_safe=use_hdr_safe)
        draw = ImageDraw.Draw(img)

        eyebrow_str = _format_tracked_eyebrow(raw_eyebrow)
        f_eye = _load_font(int(round(12 * s)), bold=True, serif=False, text=eyebrow_str)
        max_card_w = int(width * 0.78)
        inner_max_w = max_card_w - int(round(88 * s))
        f_title = _load_fitted_font(draw, item.text, 26.0, 20.0, inner_max_w, scale=s, bold=True, serif=True)
        f_sub = _load_font(int(round(17.5 * s)), bold=False, serif=False, text=sub_text)

        title_lines = _fit_text_lines(draw, item.text, f_title, inner_max_w, max_lines=2)
        sub_lines = _fit_text_lines(draw, sub_text, f_sub, inner_max_w, max_lines=2) if sub_text else []

        ew = _measure_text(draw, eyebrow_str, f_eye)[0] if eyebrow_str else 0
        tw = max((_measure_text(draw, ln, f_title)[0] for ln in title_lines), default=0)
        sw = max((_measure_text(draw, ln, f_sub)[0] for ln in sub_lines), default=0)

        pad_left = int(round(32 * s))
        pad_right = int(round(32 * s))
        pw = min(max_card_w, max(ew, tw, sw) + pad_left + pad_right + int(round(20 * s)))
        pw = max(pw, int(round(400 * s)))

        top_pad = int(round(14 * s))
        bottom_pad = int(round(14 * s))
        eye_h = max(14, int(round(16 * s))) if eyebrow_str else 0
        title_line_h = max(16, int(round(32 * s)))
        sub_line_h = max(12, int(round(24 * s)))
        cur_h = (
            top_pad
            + (eye_h + int(round(4 * s)) if eyebrow_str else 0)
            + len(title_lines) * title_line_h
        )
        if sub_lines:
            cur_h += max(2, int(round(4 * s))) + len(sub_lines) * sub_line_h
        ph = cur_h + bottom_pad

        pl = (width - pw) // 2
        # Anchor strictly above bottom 16.5% so lower-thirds never collide with bilingual subtitles at bottom 5-14%
        pt = int(round(height * 0.835)) - ph
        pr, pb = pl + pw, pt + ph
        radius = int(round(14 * s))

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=radius,
            fill=clamp_rgba_for_hdr((12, 14, 18, 146), hdr_safe=use_hdr_safe),
            outline=clamp_rgba_for_hdr((255, 255, 255, 42), hdr_safe=use_hdr_safe),
            width=max(1, int(round(1.1 * s))),
        )

        bar_x = pl + int(round(16 * s))
        bar_w = max(2, int(round(3.0 * s)))
        draw.rounded_rectangle(
            [bar_x, pt + int(round(14 * s)), bar_x + bar_w, pb - int(round(14 * s))],
            radius=max(1, bar_w // 2),
            fill=accent_rule_rgba,
        )

        tx = bar_x + bar_w + int(round(14 * s))
        ty = pt + top_pad
        if eyebrow_str:
            _draw_cinema_text(draw, (tx, ty), eyebrow_str, font=f_eye, fill=(208, 210, 216, 226), scale=s, hdr_safe=use_hdr_safe)
            ty += eye_h + int(round(4 * s))

        for ln in title_lines:
            _draw_cinema_text(draw, (tx, ty), ln, font=f_title, fill=(250, 248, 244, 255), scale=s, hdr_safe=use_hdr_safe)
            ty += title_line_h

        if sub_lines:
            ty += max(2, int(round(4 * s)))
            for ln in sub_lines:
                _draw_cinema_text(draw, (tx, ty), ln, font=f_sub, fill=(200, 206, 214, 236), scale=s, hdr_safe=use_hdr_safe)
                ty += sub_line_h

    if use_hdr_safe:
        arr = np.asarray(img, dtype=np.uint8).copy()
        arr[:, :, :3] = np.minimum(arr[:, :, :3], HDR_DIFFUSE_WHITE_PEAK_RGB)
        dark_mask = (arr[:, :, 0] < 45) & (arr[:, :, 1] < 45) & (arr[:, :, 2] < 45)
        arr[:, :, 3] = np.where(dark_mask, np.minimum(arr[:, :, 3], HDR_MAX_SHADOW_ALPHA), arr[:, :, 3])
        img = Image.fromarray(arr, mode="RGBA")

    img.save(out_path, "PNG")
    return out_path
