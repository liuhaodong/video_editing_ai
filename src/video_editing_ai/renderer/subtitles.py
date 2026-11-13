"""SRT and ASS subtitle exporters for dialogue and captions."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

from ..models.schema import EditTimeline


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
    """Write styled Advanced SubStation Alpha (.ass) subtitles supporting CJK, bilingual lines, and HDR-safe diffuse white."""
    out_path = Path(output_ass)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_hdr = bool(timeline.settings.hdr) if hdr is None else bool(hdr)
    w = timeline.settings.width
    h = timeline.settings.height
    scale = w / 1920.0
    font_name, latin_font = resolve_subtitle_font_stack(timeline.subtitles.font_name)
    font_size = max(24, int(round(timeline.subtitles.font_size * scale)))
    sec_font_size = max(18, int(round(font_size * 0.68)))
    margin_v = int(round(36 * scale))

    # Clamp peak white to ~75% HLG diffuse white (&H00C0C0C0) when HDR is active
    primary_col = "&H00C0C0C0" if is_hdr else "&H00FFFFFF"
    sec_col = "&HB0B0B0&" if is_hdr else "&HD0D0D0&"
    back_col = "&H64000000" if is_hdr else "&H80000000"

    header = f"""[Script Info]
Title: {timeline.project_name}
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: {w}
PlayResY: {h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_col},&H000000FF,&H00151515,{back_col},-1,0,0,0,100,100,0,0,1,2.5,1.5,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    event_lines: list[str] = []
    for item in timeline.subtitles.items:
        start_s = _format_ass_time(item.start)
        end_s = _format_ass_time(item.end)
        clean_primary = item.text.replace("\n", " ")
        if item.secondary_text:
            clean_sec = item.secondary_text.replace("\n", " ")
            text_payload = (
                f"{{\\fad(200,200)}}{clean_primary}\\N"
                f"{{\\fn{latin_font}\\fs{sec_font_size}\\c{sec_col}}}{clean_sec}"
            )
        else:
            text_payload = f"{{\\fad(200,200)}}{clean_primary}"
        event_lines.append(f"Dialogue: 0,{start_s},{end_s},Default,,0,0,0,,{text_payload}")

    out_path.write_text(header + "\n".join(event_lines) + "\n", encoding="utf-8")
    return out_path
