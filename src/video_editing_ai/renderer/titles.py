"""Programmatic RGBA graphic title cards, Act banners, and dialogue callouts via Pillow."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

from ..models.schema import TitleOverlayItem


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
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
        return (255, 215, 64, alpha)


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

    accent_rgba = _parse_rgba(item.accent_color, 240)
    accent_text = _parse_rgba(item.accent_color, 255)
    sub_text = (item.subtitle or "").strip()

    if item.style == "opening_card":
        f_title = _load_font(int(round(62 * s)))
        f_sub = _load_font(int(round(30 * s)))
        bt = draw.textbbox((0, 0), item.text, font=f_title)
        tw = bt[2] - bt[0]
        sw = 0
        if sub_text:
            bs = draw.textbbox((0, 0), sub_text, font=f_sub)
            sw = bs[2] - bs[0]

        pw = min(int(width * 0.90), max(tw, sw) + int(round(110 * s)))
        ph = int(round((152 if sub_text else 96) * s))
        cx, cy = width // 2, int(round(134 * s))
        pl, pt, pr, pb = cx - pw // 2, cy - ph // 2, cx + pw // 2, cy + ph // 2

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=int(round(24 * s)),
            fill=(12, 14, 24, 215),
            outline=accent_rgba,
            width=max(2, int(round(3 * s))),
        )
        ty = cy - int(round(52 * s)) if sub_text else cy - int(round(24 * s))
        draw.text((cx - tw // 2 + 2, ty + 2), item.text, font=f_title, fill=(0, 0, 0, 220))
        draw.text((cx - tw // 2, ty), item.text, font=f_title, fill=accent_text)
        if sub_text:
            draw.text((cx - sw // 2, cy + int(round(28 * s))), sub_text, font=f_sub, fill=(245, 245, 250, 255))

    elif item.style == "act_banner":
        f_title = _load_font(int(round(44 * s)))
        f_sub = _load_font(int(round(24 * s)))
        bt = draw.textbbox((0, 0), item.text, font=f_title)
        tw = bt[2] - bt[0]
        sw = 0
        if sub_text:
            bs = draw.textbbox((0, 0), sub_text, font=f_sub)
            sw = bs[2] - bs[0]

        pw = min(int(width * 0.85), max(tw, sw) + int(round(70 * s)))
        ph = int(round((108 if sub_text else 72) * s))
        pl, pt = int(round(68 * s)), int(round(68 * s))
        pr, pb = pl + pw, pt + ph

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=int(round(18 * s)),
            fill=(14, 18, 30, 215),
            outline=accent_rgba,
            width=max(2, int(round(3 * s))),
        )
        draw.text((pl + int(round(30 * s)), pt + int(round(14 * s))), item.text, font=f_title, fill=accent_text)
        if sub_text:
            draw.text(
                (pl + int(round(30 * s)), pt + int(round(66 * s))),
                sub_text,
                font=f_sub,
                fill=(235, 240, 248, 235),
            )

    elif item.style == "closing_card":
        f_title = _load_font(int(round(60 * s)))
        f_sub = _load_font(int(round(30 * s)))
        bt = draw.textbbox((0, 0), item.text, font=f_title)
        tw = bt[2] - bt[0]
        sw = 0
        if sub_text:
            bs = draw.textbbox((0, 0), sub_text, font=f_sub)
            sw = bs[2] - bs[0]

        pw = min(int(width * 0.90), max(tw, sw) + int(round(120 * s)))
        ph = int(round((170 if sub_text else 110) * s))
        cx, cy = width // 2, height // 2
        pl, pt, pr, pb = cx - pw // 2, cy - ph // 2, cx + pw // 2, cy + ph // 2

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=int(round(26 * s)),
            fill=(10, 12, 22, 228),
            outline=accent_rgba,
            width=max(2, int(round(4 * s))),
        )
        ty = cy - int(round(44 * s)) if sub_text else cy - int(round(26 * s))
        draw.text((cx - tw // 2 + 2, ty + 2), item.text, font=f_title, fill=(0, 0, 0, 220))
        draw.text((cx - tw // 2, ty), item.text, font=f_title, fill=accent_text)
        if sub_text:
            draw.text((cx - sw // 2, cy + int(round(26 * s))), sub_text, font=f_sub, fill=(255, 255, 255, 245))

    else:
        # lower_third or dialogue_badge at bottom-center
        f_title = _load_font(int(round(38 * s)))
        f_sub = _load_font(int(round(23 * s)))
        bt = draw.textbbox((0, 0), item.text, font=f_title)
        tw = bt[2] - bt[0]
        sw = 0
        if sub_text:
            bs = draw.textbbox((0, 0), sub_text, font=f_sub)
            sw = bs[2] - bs[0]

        pw = min(int(width * 0.90), max(tw, sw) + int(round(76 * s)))
        ph = int(round((98 if sub_text else 68) * s))
        cx, cy = width // 2, height - int(round(92 * s))
        pl, pt, pr, pb = cx - pw // 2, cy - ph // 2, cx + pw // 2, cy + ph // 2

        draw.rounded_rectangle(
            [pl, pt, pr, pb],
            radius=int(round(18 * s)),
            fill=(14, 15, 25, 212),
            outline=accent_rgba,
            width=max(2, int(round(2.5 * s))),
        )
        if sub_text:
            draw.text(
                (cx - tw // 2, cy - int(round(30 * s))),
                item.text,
                font=f_title,
                fill=(255, 255, 255, 255),
            )
            draw.text(
                (cx - sw // 2, cy + int(round(16 * s))),
                sub_text,
                font=f_sub,
                fill=(210, 215, 225, 245),
            )
        else:
            draw.text(
                (cx - tw // 2, cy - int(round(20 * s))),
                item.text,
                font=f_title,
                fill=(255, 255, 255, 255),
            )

    img.save(out_path, "PNG")
    return out_path
