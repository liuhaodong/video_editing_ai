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
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, max(10, size))
            except Exception:
                continue
    return ImageFont.load_default()


def _parse_rgba(hex_color: str, alpha: int = 240) -> tuple[int, int, int, int]:
    try:
        rgb = ImageColor.getrgb(hex_color)
        return (rgb[0], rgb[1], rgb[2], alpha)
    except Exception:
        return (234, 194, 96, alpha)


def _draw_bottom_scrim(img: Image.Image, scrim_height_ratio: float = 0.22, max_alpha: int = 105) -> None:
    """Draw a smooth vertical bottom gradient scrim on an RGBA overlay so lower-third captions blend naturally."""
    w, h = img.size
    scrim_h = max(16, int(round(h * scrim_height_ratio)))
    y_start = h - scrim_h
    # Build a 1xN vertical gradient strip and resize to WxN (extremely fast)
    grad_1d = Image.new("RGBA", (1, scrim_h), (0, 0, 0, 0))
    for y in range(scrim_h):
        t = y / max(1, scrim_h - 1)
        a = int(round(max_alpha * (t * t)))
        grad_1d.putpixel((0, y), (6, 8, 14, a))
    scrim_strip = grad_1d.resize((w, scrim_h), Image.Resampling.BILINEAR)
    img.paste(scrim_strip, (0, y_start), scrim_strip)


def render_title_png(
    item: TitleOverlayItem,
    output_png: str | Path,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """Render a TitleOverlayItem to a transparent RGBA PNG plate at (width, height)."""
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s = width / 1920.0  # Scale factor relative to 1080p

    accent_bar_rgba = _parse_rgba(item.accent_color, 248)
    accent_text = _parse_rgba(item.accent_color, 255)
    sub_text = (item.subtitle or "").strip()

    if item.style == "opening_card":
        f_title = _load_font(int(round(54 * s)), bold=True, serif=True)
        f_sub = _load_font(int(round(26 * s)), bold=False, serif=False)
        bt = draw.textbbox((0, 0), item.text, font=f_title)
        tw = bt[2] - bt[0]
        sw = 0
        if sub_text:
            bs = draw.textbbox((0, 0), sub_text, font=f_sub)
            sw = bs[2] - bs[0]

        pw = min(int(width * 0.88), max(tw, sw) + int(round(120 * s)))
        ph = int(round((156 if sub_text else 98) * s))
        cx, cy = width // 2, int(round(128 * s))
        pl, pt, pr, pb = cx - pw // 2, cy - ph // 2, cx + pw // 2, cy + ph // 2

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=int(round(22 * s)),
            fill=(10, 14, 22, 214),
            outline=(255, 245, 220, 76),
            width=max(1, int(round(1.5 * s))),
        )
        # Subtle top gold accent line inside the card
        line_w = min(pw - int(round(80 * s)), int(round(220 * s)))
        draw.rounded_rectangle(
            [cx - line_w // 2, pt + int(round(14 * s)), cx + line_w // 2, pt + int(round(18 * s))],
            radius=2,
            fill=accent_bar_rgba,
        )
        ty = cy - int(round(44 * s)) if sub_text else cy - int(round(22 * s))
        draw.text((cx - tw // 2 + 2, ty + 2), item.text, font=f_title, fill=(0, 0, 0, 210))
        draw.text((cx - tw // 2, ty), item.text, font=f_title, fill=(255, 255, 255, 255))
        if sub_text:
            draw.text((cx - sw // 2, cy + int(round(26 * s))), sub_text, font=f_sub, fill=accent_text)

    elif item.style == "act_banner":
        f_title = _load_font(int(round(36 * s)), bold=True, serif=True)
        f_sub = _load_font(int(round(21 * s)), bold=False, serif=False)
        bt = draw.textbbox((0, 0), item.text, font=f_title)
        tw = bt[2] - bt[0]
        sw = 0
        if sub_text:
            bs = draw.textbbox((0, 0), sub_text, font=f_sub)
            sw = bs[2] - bs[0]

        pad_left = int(round(40 * s))
        pad_right = int(round(36 * s))
        pw = min(int(width * 0.82), max(tw, sw) + pad_left + pad_right)
        ph = int(round((106 if sub_text else 70) * s))
        pl, pt = int(round(56 * s)), int(round(44 * s))
        pr, pb = pl + pw, pt + ph

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=int(round(18 * s)),
            fill=(10, 14, 22, 212),
            outline=(255, 245, 220, 70),
            width=max(1, int(round(1.5 * s))),
        )
        bar_x = pl + int(round(16 * s))
        bar_w = max(4, int(round(5 * s)))
        draw.rounded_rectangle(
            [bar_x, pt + int(round(14 * s)), bar_x + bar_w, pb - int(round(14 * s))],
            radius=max(2, bar_w // 2),
            fill=accent_bar_rgba,
        )
        tx = bar_x + bar_w + int(round(16 * s))
        draw.text((tx + 2, pt + int(round(14 * s)) + 2), item.text, font=f_title, fill=(0, 0, 0, 200))
        draw.text((tx, pt + int(round(14 * s))), item.text, font=f_title, fill=accent_text)
        if sub_text:
            draw.text(
                (tx, pt + int(round(62 * s))),
                sub_text,
                font=f_sub,
                fill=(232, 238, 246, 242),
            )

    elif item.style == "closing_card":
        f_title = _load_font(int(round(52 * s)), bold=True, serif=True)
        f_sub = _load_font(int(round(26 * s)), bold=False, serif=False)
        bt = draw.textbbox((0, 0), item.text, font=f_title)
        tw = bt[2] - bt[0]
        sw = 0
        if sub_text:
            bs = draw.textbbox((0, 0), sub_text, font=f_sub)
            sw = bs[2] - bs[0]

        pw = min(int(width * 0.88), max(tw, sw) + int(round(120 * s)))
        ph = int(round((166 if sub_text else 106) * s))
        cx, cy = width // 2, height // 2
        pl, pt, pr, pb = cx - pw // 2, cy - ph // 2, cx + pw // 2, cy + ph // 2

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=int(round(24 * s)),
            fill=(10, 14, 22, 224),
            outline=(255, 245, 220, 82),
            width=max(1, int(round(1.5 * s))),
        )
        ty = cy - int(round(42 * s)) if sub_text else cy - int(round(24 * s))
        draw.text((cx - tw // 2 + 2, ty + 2), item.text, font=f_title, fill=(0, 0, 0, 220))
        draw.text((cx - tw // 2, ty), item.text, font=f_title, fill=accent_text)
        if sub_text:
            draw.text((cx - sw // 2, cy + int(round(24 * s))), sub_text, font=f_sub, fill=(242, 246, 252, 248))

    else:
        # Documentary lower-third card (matches ken_burns.py frosted documentary card aesthetic!)
        _draw_bottom_scrim(img, scrim_height_ratio=0.22, max_alpha=95)
        draw = ImageDraw.Draw(img)

        f_badge = _load_font(int(round(16 * s)), bold=True, serif=False)
        f_title = _load_font(int(round(29 * s)), bold=True, serif=True)
        f_sub = _load_font(int(round(21 * s)), bold=False, serif=False)

        badge_str = "LIVE SCENE  ·  PARIS DOCUMENTARY"
        sub_lines = textwrap.wrap(sub_text, width=78)[:2] if sub_text else []

        tw = draw.textbbox((0, 0), item.text, font=f_title)[2]
        bw = draw.textbbox((0, 0), badge_str, font=f_badge)[2]
        sw = max((draw.textbbox((0, 0), ln, font=f_sub)[2] for ln in sub_lines), default=0)

        pad_left = int(round(38 * s))
        pad_right = int(round(36 * s))
        pw = min(int(width * 0.84), max(tw, sw, bw) + pad_left + pad_right + int(round(24 * s)))
        pw = max(pw, int(round(520 * s)))

        top_pad = int(round(16 * s))
        bottom_pad = int(round(18 * s))
        cur_h = top_pad + int(round(25 * s)) + int(round(38 * s))
        if sub_lines:
            cur_h += int(round(6 * s)) + len(sub_lines) * int(round(28 * s))
        ph = cur_h + bottom_pad

        pl = (width - pw) // 2
        pt = height - ph - int(round(34 * s))
        pr, pb = pl + pw, pt + ph
        radius = int(round(20 * s))

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=radius,
            fill=(10, 14, 22, 212),
            outline=(255, 245, 220, 72),
            width=max(1, int(round(1.5 * s))),
        )

        bar_x = pl + int(round(18 * s))
        bar_w = max(4, int(round(4.5 * s)))
        draw.rounded_rectangle(
            [bar_x, pt + int(round(16 * s)), bar_x + bar_w, pb - int(round(16 * s))],
            radius=max(2, bar_w // 2),
            fill=accent_bar_rgba,
        )

        tx = bar_x + bar_w + int(round(16 * s))
        ty = pt + top_pad

        draw.text((tx + 1, ty + 1), badge_str, font=f_badge, fill=(0, 0, 0, 180))
        draw.text((tx, ty), badge_str, font=f_badge, fill=accent_text)
        ty += int(round(25 * s))

        draw.text((tx + 2, ty + 2), item.text, font=f_title, fill=(0, 0, 0, 205))
        draw.text((tx, ty), item.text, font=f_title, fill=(255, 255, 255, 255))
        ty += int(round(38 * s))

        if sub_lines:
            ty += int(round(4 * s))
            for ln in sub_lines:
                draw.text((tx + 1, ty + 1), ln, font=f_sub, fill=(0, 0, 0, 185))
                draw.text((tx, ty), ln, font=f_sub, fill=(226, 232, 242, 248))
                ty += int(round(28 * s))

    img.save(out_path, "PNG")
    return out_path
