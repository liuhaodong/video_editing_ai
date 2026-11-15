"""SRT and styled ASS subtitle exporters with theme typography and word-highlight support."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

from ..models.schema import EditTimeline, SubtitleItem
from .titles import get_theme_spec


def resolve_subtitle_font_stack(preferred_font: str = "") -> tuple[str, str]:
    """Resolve primary CJK/bilingual font and Latin companion font across macOS, Windows, and Linux."""
    pref = (preferred_font or "").strip()
    if sys.platform == "darwin":
        primary = pref if (pref and pref != "Microsoft YaHei") else "PingFang SC"
        secondary = "Avenir Next"
    elif sys.platform.startswith("win"):
        primary = pref if (pref and pref not in ("PingFang SC", "Avenir Next")) else "Microsoft YaHei"
        secondary = "Segoe UI"
    else:
        primary = pref or "Noto Sans CJK SC"
        secondary = "DejaVu Sans"
    return primary, secondary


def _format_srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    hrs = ms // 3_600_000
    ms %= 3_600_000
    mins = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    ms %= 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def _format_ass_time(seconds: float) -> str:
    cs = int(round(max(0.0, seconds) * 100))
    hrs = cs // 360_000
    cs %= 360_000
    mins = cs // 6000
    cs %= 6000
    secs = cs // 100
    cs %= 100
    return f"{hrs:d}:{mins:02d}:{secs:02d}.{cs:02d}"


def _hex_to_ass_bgr(hex_color: str, hdr_safe: bool = False) -> str:
    """Convert '#RRGGBB' into ASS '&H00BBGGRR', clamping channels <= 192 (0xC0) in HDR mode."""
    h = (hex_color or "").strip().lstrip("#")
    if len(h) != 6:
        r, g, b = (232, 228, 220)
    else:
        try:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        except Exception:
            r, g, b = (232, 228, 220)
    if hdr_safe:
        cap = 192
        peak = max(r, g, b)
        if peak > cap:
            sc = float(cap) / float(max(1, peak))
            r, g, b = int(round(r * sc)), int(round(g * sc)), int(round(b * sc))
    return f"&H00{b:02X}{g:02X}{r:02X}"


def _build_karaoke_text(item: SubtitleItem, clean_primary: str) -> str:
    """Build smooth ASS \\kf word-highlight markup from WordTimestamp timings or proportional fallback."""
    words = list(getattr(item, "words", []) or [])
    if words:
        parts: list[str] = []
        cursor = float(item.start)
        for idx, wt in enumerate(words):
            w_str = str(wt.word or "").strip()
            if not w_str:
                continue
            if wt.start > cursor + 0.04:
                gap_cs = max(1, int(round((wt.start - cursor) * 100)))
                parts.append(f"{{\\kf{gap_cs}}} ")
            dur_cs = max(8, int(round(max(0.08, wt.end - wt.start) * 100)))
            prefix = "" if (idx == 0 and not parts) else " "
            parts.append(f"{prefix}{{\\kf{dur_cs}}}{w_str}")
            cursor = max(cursor, float(wt.end))
        if parts:
            return "".join(parts).strip()

    # Fallback proportional sweep across tokens
    tokens = clean_primary.split(" ") if " " in clean_primary else list(clean_primary)
    tokens = [t for t in tokens if t]
    if not tokens:
        return clean_primary
    total_cs = max(20, int(round(max(0.25, item.end - item.start) * 100)))
    weights = [max(1, len(t)) for t in tokens]
    sum_w = max(1, sum(weights))
    sep = " " if " " in clean_primary else ""
    out_parts: list[str] = []
    used_cs = 0
    for idx, (tok, w_val) in enumerate(zip(tokens, weights)):
        if idx == len(tokens) - 1:
            cs = max(8, total_cs - used_cs)
        else:
            cs = max(8, int(round(total_cs * (w_val / float(sum_w)))))
            used_cs += cs
        out_parts.append(f"{{\\kf{cs}}}{tok}")
    return sep.join(out_parts)


def export_srt_subtitles(timeline: EditTimeline, output_srt: str | Path) -> Path:
    """Write standard SubRip (.srt) subtitle file from an EditTimeline."""
    out_path = Path(output_srt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blocks: list[str] = []
    for idx, item in enumerate(timeline.subtitles.items, start=1):
        lines = [
            str(idx),
            f"{_format_srt_time(item.start)} --> {_format_srt_time(item.end)}",
            item.text,
        ]
        if item.secondary_text:
            lines.append(item.secondary_text)
        blocks.append("\n".join(lines))

    out_path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return out_path


def export_ass_subtitles(
    timeline: EditTimeline,
    output_ass: str | Path,
    hdr: Optional[bool] = None,
) -> Path:
    """Write styled Advanced SubStation Alpha (.ass) subtitles with theme styling, word highlights, and HDR clamp."""
    out_path = Path(output_ass)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_hdr = bool(timeline.settings.hdr) if hdr is None else bool(hdr)
    w = timeline.settings.width
    h = timeline.settings.height
    scale = w / 1920.0
    font_name, latin_font = resolve_subtitle_font_stack(timeline.subtitles.font_name)
    theme_name = getattr(timeline.subtitles, "theme", "apple_tv_glass")
    style_preset = getattr(timeline.subtitles, "style_preset", "cinema_editorial")
    spec = get_theme_spec(theme_name)

    font_size = max(24, int(round(timeline.subtitles.font_size * scale)))
    sec_font_size = max(18, int(round(font_size * 0.68)))
    margin_v = int(round(36 * scale))

    # Clamp peak white to ~75% HLG diffuse white (&H00C0C0C0) when HDR is active
    base_white_col = "&H00C0C0C0" if is_hdr else "&H00F8FAF8"
    accent_ass_col = _hex_to_ass_bgr(spec["accent_hex"], hdr_safe=is_hdr)
    sec_col = "&H00B0B0B0" if is_hdr else "&H00D4D8DE"
    back_col = "&H72101218" if style_preset == "minimal_glass" else ("&H64000000" if is_hdr else "&H80000000")

    if style_preset == "kinetic_word_highlight":
        # In ASS \kf karaoke fill: SecondaryColour is pre-highlight base; PrimaryColour sweeps active words
        primary_col = accent_ass_col if not is_hdr else "&H0094BAC0"
        secondary_col = base_white_col
    else:
        primary_col = base_white_col
        secondary_col = "&H000000FF"

    border_style = 3 if style_preset == "minimal_glass" else 1
    outline_w = 10.0 if style_preset == "minimal_glass" else 2.4
    shadow_w = 0.0 if style_preset == "minimal_glass" else 1.4
    bold_flag = -1 if spec.get("primary_bold", True) else 0

    header = f"""[Script Info]
Title: {timeline.project_name}
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: {w}
PlayResY: {h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_col},{secondary_col},&H00141418,{back_col},{bold_flag},0,0,0,100,100,0,0,{border_style},{outline_w:.1f},{shadow_w:.1f},2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    event_lines: list[str] = []
    for item in timeline.subtitles.items:
        start_s = _format_ass_time(item.start)
        end_s = _format_ass_time(item.end)
        clean_primary = item.text.replace("\n", " ").strip()
        if style_preset == "kinetic_word_highlight":
            primary_expr = _build_karaoke_text(item, clean_primary)
        else:
            primary_expr = clean_primary

        if item.secondary_text:
            clean_sec = item.secondary_text.replace("\n", " ").strip()
            text_payload = (
                f"{{\\fad(160,160)}}{primary_expr}\\N"
                f"{{\\fn{latin_font}\\fs{sec_font_size}\\c{sec_col}}}{clean_sec}"
            )
        else:
            text_payload = f"{{\\fad(160,160)}}{primary_expr}"
        event_lines.append(f"Dialogue: 0,{start_s},{end_s},Default,,0,0,0,,{text_payload}")

    out_path.write_text(header + "\n".join(event_lines) + "\n", encoding="utf-8")
    return out_path
