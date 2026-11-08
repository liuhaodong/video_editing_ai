"""Programmatic RGBA graphic title cards, Chapter banners, and Documentary Lower-Third overlays via Pillow."""

from __future__ import annotations

import os
from pathlib import Path
import textwrap

from PIL import Image, ImageColor, ImageDraw, ImageFont

from ..models.schema import TitleOverlayItem


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
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Avenir Next.ttc",
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


def _parse_rgba(hex_color: str, alpha: int = 235) -> tuple[int, int, int, int]:
    norm = (hex_color or "").strip().upper()
    if not norm or norm in ("#FFD740", "#FFAB40", "#00E5FF", "#E040FB", "#F5C542"):
        return (232, 228, 220, alpha)
    try:
        rgb = ImageColor.getrgb(hex_color)
        return (rgb[0], rgb[1], rgb[2], alpha)
    except Exception:
        return (232, 228, 220, alpha)


def _draw_bottom_scrim(img: Image.Image, scrim_height_ratio: float = 0.24, max_alpha: int = 115) -> None:
    """Draw a smooth vertical bottom gradient scrim on an RGBA overlay so lower-third captions blend naturally."""
    w, h = img.size
    scrim_h = max(16, int(round(h * scrim_height_ratio)))
    y_start = h - scrim_h
    grad_1d = Image.new("RGBA", (1, scrim_h), (0, 0, 0, 0))
    for y in range(scrim_h):
        t = y / max(1, scrim_h - 1)
        a = int(round(max_alpha * (t * t)))
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
) -> None:
    """Render typography with a soft two-pass cinema drop shadow for clean legibility without heavy boxes."""
    x, y = xy
    off1 = max(1, int(round(1.5 * scale)))
    off2 = max(2, int(round(3.0 * scale)))
    draw.text((x + off2, y + off2), text, font=font, fill=(0, 0, 0, 115))
    draw.text((x + off1, y + off1), text, font=font, fill=(0, 0, 0, 185))
    draw.text((x, y), text, font=font, fill=fill)


def render_title_png(
    item: TitleOverlayItem,
    output_png: str | Path,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """Render a TitleOverlayItem to a transparent RGBA PNG plate using an A24 / Criterion editorial aesthetic."""
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    s = width / 1920.0  # Scale factor relative to 1080p

    accent_rule_rgba = _parse_rgba(item.accent_color, 165)
    sub_text = (item.subtitle or "").strip()

    if item.style in ("opening_card", "closing_card", "act_banner"):
        if item.style == "closing_card":
            # Subtle full-frame warm obsidian veil for the closing title
            veil = Image.new("RGBA", (width, height), (8, 10, 14, 92))
            img = Image.alpha_composite(img, veil)
            cy = int(round(height * 0.50))
            title_pt = 48
        else:
            # Soft bottom gradient veil so upper two-thirds of the footage stays 100% unobstructed
            _draw_bottom_scrim(img, scrim_height_ratio=0.34, max_alpha=135)
            cy = int(round(height * 0.78))
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
        )
        if sub_text:
            _draw_cinema_text(
                draw,
                (cx - sw // 2, ty + int(round((title_pt + 14) * s))),
                sub_text,
                font=f_sub,
                fill=(204, 210, 218, 236),
                scale=s,
            )

    else:
        # Refined smoked-glass documentary lower-third card (no yellow bars, no technical header)
        _draw_bottom_scrim(img, scrim_height_ratio=0.22, max_alpha=100)
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
        pt = height - ph - int(round(36 * s))
        pr, pb = pl + pw, pt + ph
        radius = int(round(14 * s))

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=radius,
            fill=(12, 14, 18, 142),
            outline=(255, 255, 255, 36),
            width=max(1, int(round(1.0 * s))),
        )

        # Delicate 2px warm pearl-silver hairline accent on the left
        bar_x = pl + int(round(16 * s))
        bar_w = max(2, int(round(2.5 * s)))
        draw.rounded_rectangle(
            [bar_x, pt + int(round(14 * s)), bar_x + bar_w, pb - int(round(14 * s))],
            radius=max(1, bar_w // 2),
            fill=accent_rule_rgba,
        )

        tx = bar_x + bar_w + int(round(14 * s))
        ty = pt + top_pad

        _draw_cinema_text(draw, (tx, ty), item.text, font=f_title, fill=(250, 248, 244, 255), scale=s)
        ty += max(14, int(round(34 * s)))

        if sub_lines:
            ty += max(2, int(round(4 * s)))
            for ln in sub_lines:
                _draw_cinema_text(draw, (tx, ty), ln, font=f_sub, fill=(198, 204, 212, 236), scale=s)
                ty += max(12, int(round(25 * s)))

    img.save(out_path, "PNG")
    return out_path
