"""Sub-pixel 2.5D Parallax, Diptych Split-Screen, Museum Gallery Frame, and Frosted-Glass Documentary Card renderer."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap
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


def _ease_out_cubic(t: float) -> float:
    """Smooth cubic ease-out for documentary lower-third card glide."""
    t_c = max(0.0, min(1.0, t))
    p = 1.0 - t_c
    return 1.0 - p * p * p


def _load_editorial_font(size: int, bold: bool = True, serif: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


def _build_double_matte_card(
    img: Image.Image,
    max_w: int,
    max_h: int,
    scale_factor: float,
    border_px: Optional[int] = None,
) -> Image.Image:
    """Create a foreground photo card with a double museum matte (warm ivory outer + antique gold inner hairline) and 3D drop shadow."""
    b_px = border_px if border_px is not None else max(10, int(round(14 * scale_factor)))
    gold_pad = max(3, int(round(4 * scale_factor)))
    shadow_margin = max(18, int(round(28 * scale_factor)))
    shadow_offset = max(4, int(round(7 * scale_factor)))
    shadow_blur = max(10, int(round(16 * scale_factor)))

    fg = img.copy()
    fg.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

    fw = fg.width + (b_px + gold_pad) * 2
    fh = fg.height + (b_px + gold_pad) * 2

    # Warm museum ivory outer matte
    framed = Image.new("RGBA", (fw, fh), (248, 246, 240, 255))
    f_draw = ImageDraw.Draw(framed)
    # Subtle antique gold inner fillet hairline around the artwork
    f_draw.rectangle(
        [b_px - 2, b_px - 2, fw - b_px + 1, fh - b_px + 1],
        outline=(208, 174, 78, 225),
        width=max(2, int(round(2.0 * scale_factor))),
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
        fill=(0, 0, 0, 170),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
    shadow.paste(framed, (shadow_margin, shadow_margin), framed)
    return shadow


def _build_frosted_card_assets(
    width: int,
    height: int,
    caption_title: str,
    caption_subtitle: str = "",
    camera_badge: str = "",
) -> tuple[Image.Image, Image.Image, Image.Image, int, int]:
    """
    Pre-render the static typography & glass highlight layers of an Apple TV+ / National Geographic
    style Frosted-Glass Documentary Lower-Third Card.
    Returns (text_fg_rgba, card_mask_l, dark_tint_rgb, card_x, card_y).
    """
    s = width / 1920.0
    f_badge = _load_editorial_font(int(round(16 * s)), bold=True, serif=False)
    f_title = _load_editorial_font(int(round(29 * s)), bold=True, serif=True)
    f_sub = _load_editorial_font(int(round(21 * s)), bold=False, serif=False)

    dummy = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    d_draw = ImageDraw.Draw(dummy)

    badge_str = camera_badge.upper().replace("📍", "").strip()
    sub_lines: list[str] = []
    if caption_subtitle:
        wrapped = textwrap.wrap(caption_subtitle.strip(), width=78)
        sub_lines = wrapped[:2]

    tw = d_draw.textbbox((0, 0), caption_title, font=f_title)[2]
    bw = d_draw.textbbox((0, 0), badge_str, font=f_badge)[2] if badge_str else 0
    sw = max((d_draw.textbbox((0, 0), ln, font=f_sub)[2] for ln in sub_lines), default=0)

    pad_left = int(round(38 * s))
    pad_right = int(round(36 * s))
    pw = min(int(width * 0.84), max(tw, sw, bw) + pad_left + pad_right + int(round(24 * s)))
    pw = max(pw, int(round(520 * s)))

    # Compute exact vertical layout with generous breathing room
    top_pad = int(round(16 * s))
    bottom_pad = int(round(18 * s))
    cur_y = top_pad
    if badge_str:
        cur_y += int(round(25 * s))
    cur_y += int(round(38 * s))  # title height
    if sub_lines:
        cur_y += int(round(6 * s)) + len(sub_lines) * int(round(28 * s))
    ph = cur_y + bottom_pad

    # 1. Smooth rounded-rectangle alpha mask for frosted glass region
    radius = int(round(20 * s))
    card_mask_l = Image.new("L", (pw, ph), 0)
    m_draw = ImageDraw.Draw(card_mask_l)
    m_draw.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=radius, fill=255)

    # 2. Dark obsidian glass tint canvas
    dark_tint_rgb = Image.new("RGB", (pw, ph), (10, 14, 22))

    # 3. Foreground typography + subtle 1px specular glass rim + champagne-gold left accent bar
    text_fg = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_fg)

    # Subtle 1.5px translucent pearl-gold specular border (never harsh neon yellow!)
    draw.rounded_rectangle(
        [0, 0, pw - 1, ph - 1],
        radius=radius,
        fill=(0, 0, 0, 0),
        outline=(255, 245, 220, 68),
        width=max(1, int(round(1.5 * s))),
    )

    # Champagne-gold vertical accent pill on the left
    bar_x = int(round(18 * s))
    bar_w = max(4, int(round(4.5 * s)))
    draw.rounded_rectangle(
        [bar_x, int(round(16 * s)), bar_x + bar_w, ph - int(round(16 * s))],
        radius=max(2, bar_w // 2),
        fill=(234, 194, 96, 245),
    )

    tx = bar_x + bar_w + int(round(16 * s))
    ty = top_pad

    if badge_str:
        draw.text((tx + 1, ty + 1), badge_str, font=f_badge, fill=(0, 0, 0, 180))
        draw.text((tx, ty), badge_str, font=f_badge, fill=(238, 198, 102, 245))
        ty += int(round(25 * s))

    # Crisp editorial headline with soft drop shadow
    draw.text((tx + 2, ty + 2), caption_title, font=f_title, fill=(0, 0, 0, 200))
    draw.text((tx, ty), caption_title, font=f_title, fill=(255, 255, 255, 255))
    ty += int(round(38 * s))

    if sub_lines:
        ty += int(round(4 * s))
        for ln in sub_lines:
            draw.text((tx + 1, ty + 1), ln, font=f_sub, fill=(0, 0, 0, 185))
            draw.text((tx, ty), ln, font=f_sub, fill=(226, 232, 242, 248))
            ty += int(round(28 * s))

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
    """Render a full-frame RGBA overlay PNG containing a soft bottom scrim + frosted documentary lower-third card for video segments."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # 1. Soft bottom vertical gradient scrim (22% height)
    scrim_h = max(16, int(round(height * 0.22)))
    grad_1d = Image.new("RGBA", (1, scrim_h), (0, 0, 0, 0))
    for y in range(scrim_h):
        t = y / max(1, scrim_h - 1)
        grad_1d.putpixel((0, y), (6, 8, 14, int(round(96 * (t * t)))))
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
    card_body = Image.new("RGBA", (pw, ph), (10, 14, 22, 212))
    card_body.putalpha(card_mask_l.point(lambda p: int(p * (212 / 255.0))))
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
    In-place composite of the frosted-glass documentary card onto `frame_rgb`:
    Crops only the small `(pw, ph)` region, blurs it at 1/4 scale (16x faster than full-res blur),
    tints it with dark obsidian glass, layers the typography, and pastes back onto `frame_rgb`.
    """
    pw, ph = text_fg_rgba.size
    w, h = frame_rgb.size
    dst_x = max(0, min(w - pw, px))
    dst_y = max(0, min(h - ph, py + slide_y))

    # 1. Fast backdrop blur on the underlying scene patch (melts away any sharp photo borders behind text!)
    patch = frame_rgb.crop((dst_x, dst_y, dst_x + pw, dst_y + ph))
    small_w = max(16, pw // 4)
    small_h = max(16, ph // 4)
    blurred_patch = (
        patch.resize((small_w, small_h), Image.Resampling.BILINEAR)
        .filter(ImageFilter.GaussianBlur(radius=5))
        .resize((pw, ph), Image.Resampling.BILINEAR)
    )

    # 2. Blend with warm obsidian glass tint (72% opacity so scene colors glow softly through the glass)
    frosted_rgb = Image.blend(blurred_patch, dark_tint_rgb, 0.72)

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


_NVENC_AVAILABLE: Optional[bool] = None


def has_hevc_nvenc() -> bool:
    """Check once whether hardware-accelerated 10-bit hevc_nvenc is functional on this system."""
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
            "-pix_fmt",
            "p010le",
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
    duration: float = 5.5,
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

    # Prepare Frosted-Glass Documentary Lower-Third Card assets once outside the frame loop
    has_card = bool(kb.caption_title and kb.caption_title.strip())
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

    # Prepare FFmpeg encoder & HDR color space pipeline
    vf_filters: list[str] = ["setsar=1:1"]
    if transition_in and transition_in.type == "fade_black" and transition_in.duration > 0:
        vf_filters.append(f"fade=t=in:st=0:d={min(duration * 0.40, transition_in.duration):.3f}")
    if transition_out and transition_out.type == "fade_black" and transition_out.duration > 0:
        fade_d = min(duration * 0.40, transition_out.duration)
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
                "-preset",
                "p4",
                "-tune",
                "hq",
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
                "superfast",
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

    bg_oversample = 1.12
    bg_w = int(round(width * bg_oversample))
    bg_h = int(round(height * bg_oversample))
    # Fast background blur by blurring at 1/4 resolution then upsampling
    bg_small = ImageOps.fit(raw_primary, (bg_w // 4, bg_h // 4), method=Image.Resampling.BILINEAR)
    bg_small = bg_small.filter(ImageFilter.GaussianBlur(radius=max(6, int(round(8 * scale_factor)))))
    bg_plane = bg_small.resize((bg_w, bg_h), Image.Resampling.BILINEAR)
    bg_plane = Image.blend(bg_plane, Image.new("RGB", (bg_w, bg_h), (8, 10, 18)), 0.32)

    fg_card_1: Optional[Image.Image] = None
    fg_card_2: Optional[Image.Image] = None
    base_static_img: Optional[Image.Image] = None

    # Reserve vertical space at the bottom if a lower-third documentary card is present
    card_max_h_ratio = 0.74 if has_card else 0.80
    card_cy_ratio = 0.435 if has_card else 0.48

    if is_diptych and kb.secondary_source:
        raw_sec = Image.open(kb.secondary_source)
        raw_sec = ImageOps.exif_transpose(raw_sec).convert("RGB")
        card_max_w = int(width * 0.41)
        card_max_h = int(height * card_max_h_ratio)
        fg_card_1 = _build_double_matte_card(raw_primary, card_max_w, card_max_h, scale_factor, kb.border_px)
        fg_card_2 = _build_double_matte_card(raw_sec, card_max_w, card_max_h, scale_factor, kb.border_px)
    elif is_parallax:
        aspect = raw_primary.width / max(1, raw_primary.height)
        if aspect < 1.0:
            max_w, max_h = int(width * 0.66), int(height * card_max_h_ratio)
        else:
            max_w, max_h = int(width * 0.80), int(height * card_max_h_ratio)
        fg_card_1 = _build_double_matte_card(raw_primary, max_w, max_h, scale_factor, kb.border_px)
    else:
        oversample = 1.10 if width >= 3840 else 1.20
        work_w = int(round(width * oversample))
        work_h = int(round(height * oversample))
        if kb.mode == "gallery_frame":
            base_static_img = build_gallery_frame_image(
                photo_path, width=work_w, height=work_h, border_px=kb.border_px
            )
        else:
            base_static_img = ImageOps.fit(raw_primary, (work_w, work_h), method=Image.Resampling.LANCZOS)

    # Pre-cache a small number of discrete foreground scale steps (e.g. 48 steps) or resize directly in RGB/RGBA
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    try:
        for f_idx in range(total_frames):
            t_norm = f_idx / max(1, total_frames - 1)
            eased = _ease_in_out_cubic(t_norm)

            if is_diptych and fg_card_1 is not None and fg_card_2 is not None:
                bg_zoom = 1.05 - 0.04 * eased
                bg_cx = 0.52 - 0.04 * eased
                bg_cy = 0.50
                bw_crop = bg_w / bg_zoom
                bh_crop = bg_h / bg_zoom
                bx0 = bg_cx * bg_w - bw_crop * 0.5
                by0 = bg_cy * bg_h - bh_crop * 0.5
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

                # Paste directly onto RGB frame using RGBA mask (zero full-frame RGBA conversion!)
                frame.paste(fg_card_1, (lx, ly), fg_card_1)
                frame.paste(fg_card_2, (rx, ry), fg_card_2)

            elif is_parallax and fg_card_1 is not None:
                bg_zoom = 1.06 - 0.05 * eased
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

            # 3. Plane 3: Frosted-Glass Documentary Lower-Third Card (with smooth upward glide + fade)
            if has_card and card_fg is not None and card_mask is not None and card_tint is not None:
                t_sec = f_idx / max(1.0, float(fps))
                rem_sec = max(0.0, duration - t_sec)
                entry_prog = _ease_out_cubic((t_sec - 0.15) / 0.50) if t_sec > 0.15 else 0.0
                exit_prog = min(1.0, rem_sec / 0.40)
                alpha_mult = min(entry_prog, exit_prog)
                if alpha_mult > 0.04:
                    slide_y = int(round((1.0 - entry_prog) * 16.0 * scale_factor))
                    _composite_frosted_card_on_rgb(
                        frame_rgb=frame,
                        text_fg_rgba=card_fg,
                        card_mask_l=card_mask,
                        dark_tint_rgb=card_tint,
                        px=card_x,
                        py=card_y,
                        alpha_mult=alpha_mult,
                        slide_y=slide_y,
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
