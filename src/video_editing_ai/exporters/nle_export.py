"""Export EditTimeline (timeline.yaml) to FCPXML (.fcpxml), CMX 3600 EDL (.edl), and OpenTimelineIO (.otio)."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
from urllib.parse import quote

from ..models.schema import EditTimeline


def _sec_to_fcpxml_rational(sec: float, fps: int = 30) -> str:
    frames = int(round(sec * fps))
    return f"{frames * 100}/{fps * 100}s"


def _sec_to_smpte(sec: float, fps: int = 30, start_hour: int = 0) -> str:
    total_frames = int(round(max(0.0, sec) * fps))
    ff = total_frames % fps
    total_secs = total_frames // fps
    ss = total_secs % 60
    total_mins = total_secs // 60
    mm = total_mins % 60
    hh = start_hour + (total_mins // 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _path_to_file_uri(path_str: str) -> str:
    p = Path(path_str).resolve()
    posix = p.as_posix()
    if not posix.startswith("/"):
        posix = "/" + posix
    return "file://localhost" + quote(posix, safe="/:")


def export_fcpxml(timeline: EditTimeline, output_fcpxml: str | Path) -> Path:
    """Export EditTimeline to FCPXML 1.9 format compatible with DaVinci Resolve and Final Cut Pro."""
    out_path = Path(output_fcpxml).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timeline.recalculate_sequential_starts()
    fps = timeline.settings.fps
    w = timeline.settings.width
    h = timeline.settings.height
    total_dur = timeline.total_duration

    # Collect unique media resources
    resource_map: dict[str, str] = {}
    resource_lines: list[str] = [
        f'        <format id="r1" name="FFVideoFormat{h}p{fps}" '
        f'frameDuration="100/{fps * 100}s" width="{w}" height="{h}"/>'
    ]

    next_res_idx = 2
    for clip in timeline.visual_track:
        if clip.source not in resource_map:
            rid = f"r{next_res_idx}"
            next_res_idx += 1
            resource_map[clip.source] = rid
            uri = _path_to_file_uri(clip.source)
            name = escape(Path(clip.source).name)
            dur_s = _sec_to_fcpxml_rational(
                max(clip.duration + clip.source_in, 10.0) if clip.type == "video" else 0.0,
                fps,
            )
            has_vid = "1"
            has_aud = "1" if clip.type == "video" else "0"
            resource_lines.append(
                f'        <asset id="{rid}" name="{name}" src="{uri}" start="0s" '
                f'duration="{dur_s}" hasVideo="{has_vid}" hasAudio="{has_aud}" format="r1"/>'
            )

    for aud in timeline.audio_tracks:
        if aud.source not in resource_map:
            rid = f"r{next_res_idx}"
            next_res_idx += 1
            resource_map[aud.source] = rid
            uri = _path_to_file_uri(aud.source)
            name = escape(Path(aud.source).name)
            dur_s = _sec_to_fcpxml_rational(aud.duration or total_dur, fps)
            resource_lines.append(
                f'        <asset id="{rid}" name="{name}" src="{uri}" start="0s" '
                f'duration="{dur_s}" hasVideo="0" hasAudio="1"/>'
            )

    spine_lines: list[str] = []
    for idx, clip in enumerate(timeline.visual_track):
        rid = resource_map[clip.source]
        name = escape(Path(clip.source).name)
        offset_s = _sec_to_fcpxml_rational(clip.timeline_start, fps)
        start_s = _sec_to_fcpxml_rational(clip.source_in if clip.type == "video" else 0.0, fps)
        dur_s = _sec_to_fcpxml_rational(clip.duration, fps)

        inner_elements: list[str] = []
        # Attach markers that fall inside this clip
        for m in timeline.markers:
            if clip.timeline_start <= m.time_sec < (clip.timeline_start + clip.duration):
                m_start = _sec_to_fcpxml_rational(
                    (clip.source_in if clip.type == "video" else 0.0) + (m.time_sec - clip.timeline_start),
                    fps,
                )
                inner_elements.append(
                    f'                            <marker start="{m_start}" duration="100/{fps * 100}s" '
                    f'value="{escape(m.name)}" note="{escape(m.note)}"/>'
                )

        # Attach BGM on the very first spine clip as lane -1
        if idx == 0 and timeline.audio_tracks:
            for a_idx, aud in enumerate(timeline.audio_tracks):
                arid = resource_map[aud.source]
                aname = escape(Path(aud.source).name)
                adur = _sec_to_fcpxml_rational(aud.duration or total_dur, fps)
                astart = _sec_to_fcpxml_rational(aud.source_in, fps)
                inner_elements.append(
                    f'                            <asset-clip name="{aname}" lane="-{a_idx + 1}" '
                    f'offset="0s" ref="{arid}" start="{astart}" duration="{adur}"/>'
                )

        if inner_elements:
            spine_lines.append(
                f'                        <asset-clip name="{name}" offset="{offset_s}" '
                f'ref="{rid}" start="{start_s}" duration="{dur_s}" format="r1">\n'
                + "\n".join(inner_elements)
                + "\n                        </asset-clip>"
            )
        else:
            spine_lines.append(
                f'                        <asset-clip name="{name}" offset="{offset_s}" '
                f'ref="{rid}" start="{start_s}" duration="{dur_s}" format="r1"/>'
            )

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.9">
    <resources>
{chr(10).join(resource_lines)}
    </resources>
    <library>
        <event name="{escape(timeline.project_name)}">
            <project name="{escape(timeline.project_name)}">
                <sequence format="r1" duration="{_sec_to_fcpxml_rational(total_dur, fps)}" tcStart="0s" tcFormat="NDF">
                    <spine>
{chr(10).join(spine_lines)}
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>
"""
    out_path.write_text(xml_content, encoding="utf-8")
    return out_path


def export_edl(timeline: EditTimeline, output_edl: str | Path) -> Path:
    """Export EditTimeline to CMX 3600 EDL (.edl) format."""
    out_path = Path(output_edl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timeline.recalculate_sequential_starts()
    fps = timeline.settings.fps
    lines: list[str] = [
        f"TITLE: {timeline.project_name}",
        "FCM: NON-DROP FRAME",
        "",
    ]

    for idx, clip in enumerate(timeline.visual_track, start=1):
        src_in = clip.source_in if clip.type == "video" else 0.0
        src_out = src_in + clip.duration
        rec_in = clip.timeline_start
        rec_out = rec_in + clip.duration

        tc_src_in = _sec_to_smpte(src_in, fps, start_hour=0)
        tc_src_out = _sec_to_smpte(src_out, fps, start_hour=0)
        tc_rec_in = _sec_to_smpte(rec_in, fps, start_hour=1)
        tc_rec_out = _sec_to_smpte(rec_out, fps, start_hour=1)

        reel = f"AX{idx:02d}"[:8].ljust(8)
        track_code = "AA/V" if clip.type == "video" else "V   "
        lines.append(f"{idx:03d}  {reel} {track_code}  C        {tc_src_in} {tc_src_out} {tc_rec_in} {tc_rec_out}")
        lines.append(f"* FROM CLIP NAME: {Path(clip.source).name}")
        if clip.narrative_note:
            lines.append(f"* COMMENT: {clip.narrative_note}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def export_otio(timeline: EditTimeline, output_otio: str | Path) -> Path:
    """Export EditTimeline to standard OpenTimelineIO (.otio) JSON format."""
    out_path = Path(output_otio).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timeline.recalculate_sequential_starts()
    fps = float(timeline.settings.fps)

    v1_children: list[dict] = []
    for clip in timeline.visual_track:
        src_in = clip.source_in if clip.type == "video" else 0.0
        v1_children.append(
            {
                "OTIO_SCHEMA": "Clip.2",
                "name": Path(clip.source).name,
                "metadata": {
                    "id": clip.id,
                    "type": clip.type,
                    "act": clip.act,
                    "narrative_note": clip.narrative_note,
                    "ken_burns": clip.ken_burns.model_dump() if clip.ken_burns else None,
                },
                "source_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "value": round(src_in * fps, 2),
                        "rate": fps,
                    },
                    "duration": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "value": round(clip.duration * fps, 2),
                        "rate": fps,
                    },
                },
                "media_references": {
                    "DEFAULT_MEDIA": {
                        "OTIO_SCHEMA": "ExternalReference.1",
                        "target_url": _path_to_file_uri(clip.source),
                    }
                },
            }
        )

    a1_children: list[dict] = []
    for aud in timeline.audio_tracks:
        a_dur = aud.duration or timeline.total_duration
        a1_children.append(
            {
                "OTIO_SCHEMA": "Clip.2",
                "name": Path(aud.source).name,
                "metadata": {"id": aud.id, "role": aud.role, "volume": aud.volume},
                "source_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "value": round(aud.source_in * fps, 2),
                        "rate": fps,
                    },
                    "duration": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "value": round(a_dur * fps, 2),
                        "rate": fps,
                    },
                },
                "media_references": {
                    "DEFAULT_MEDIA": {
                        "OTIO_SCHEMA": "ExternalReference.1",
                        "target_url": _path_to_file_uri(aud.source),
                    }
                },
            }
        )

    otio_doc = {
        "OTIO_SCHEMA": "Timeline.1",
        "name": timeline.project_name,
        "global_start_time": {"OTIO_SCHEMA": "RationalTime.1", "value": 0.0, "rate": fps},
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "name": "tracks",
            "children": [
                {
                    "OTIO_SCHEMA": "Track.1",
                    "name": "V1_Primary_Visual",
                    "kind": "Video",
                    "children": v1_children,
                },
                {
                    "OTIO_SCHEMA": "Track.1",
                    "name": "A1_Background_Audio",
                    "kind": "Audio",
                    "children": a1_children,
                },
            ],
        },
    }

    out_path.write_text(json.dumps(otio_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def export_all_nle_formats(timeline: EditTimeline, output_dir: str | Path) -> dict[str, Path]:
    """Export FCPXML, EDL, and OTIO files into output_dir."""
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stem = timeline.project_name or "timeline"
    return {
        "fcpxml": export_fcpxml(timeline, out_root / f"{stem}.fcpxml"),
        "edl": export_edl(timeline, out_root / f"{stem}.edl"),
        "otio": export_otio(timeline, out_root / f"{stem}.otio"),
    }
