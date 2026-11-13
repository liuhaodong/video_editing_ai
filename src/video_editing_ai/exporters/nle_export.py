"""Export EditTimeline (timeline.yaml) to FCPXML 1.11 (.fcpxml), CMX 3600 EDL (.edl), and OpenTimelineIO (.otio)."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re
from urllib.parse import quote

from ..models.schema import EditTimeline


def _frames_to_fcpxml_rational(frames: int, fps: int = 30) -> str:
    return f"{max(0, int(frames)) * 100}/{max(1, int(fps)) * 100}s"


def _sec_to_frames(sec: float, fps: int = 30) -> int:
    return max(0, int(round(max(0.0, float(sec)) * max(1, int(fps)))))


def _sec_to_fcpxml_rational(sec: float, fps: int = 30) -> str:
    return _frames_to_fcpxml_rational(_sec_to_frames(sec, fps), fps)


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
    """
    Export EditTimeline to frame-accurate FCPXML 1.11 compatible with DaVinci Resolve and Final Cut Pro.
    Maps Track V1 primary spine, V2 connected B-roll cutaways (lane=1), V3 titles (lane=2),
    A1-A3 audio lanes (lane=-1 dialogue/VO, lane=-2 SFX/ambience, lane=-3 music), and chapter markers.
    """
    out_path = Path(output_fcpxml).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timeline.recalculate_sequential_starts()
    fps = max(1, int(timeline.settings.fps or 30))
    w = int(timeline.settings.width or 3840)
    h = int(timeline.settings.height or 2160)
    color_space_attr = "9-18-9 (Rec. 2020 HLG)" if timeline.settings.hdr else "1-1-1 (Rec. 709)"

    # Pre-compute exact frame-aligned spine timings so sequence duration == sum(spine durations)
    spine_entries: list[tuple[int, int, int]] = []  # (offset_frames, start_frames, dur_frames)
    running_frames = 0
    for clip in timeline.visual_track:
        dur_f = max(1, _sec_to_frames(clip.duration, fps))
        start_f = _sec_to_frames(clip.source_in if clip.type == "video" else 0.0, fps)
        spine_entries.append((running_frames, start_f, dur_f))
        running_frames += dur_f
    total_seq_frames = running_frames

    # Collect unique media resources
    resource_map: dict[str, str] = {}
    resource_lines: list[str] = [
        f'        <format id="r1" name="FFVideoFormat{h}p{fps}" '
        f'frameDuration="100/{fps * 100}s" width="{w}" height="{h}" colorSpace="{color_space_attr}"/>',
        '        <effect id="r_title" name="Basic Title" '
        'uid=".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti"/>',
    ]

    next_res_idx = 2
    for clip in list(timeline.visual_track) + list(timeline.broll_track):
        if clip.source not in resource_map:
            rid = f"r{next_res_idx}"
            next_res_idx += 1
            resource_map[clip.source] = rid
            uri = _path_to_file_uri(clip.source)
            name = escape(Path(clip.source).name)
            asset_dur_sec = max(clip.duration + clip.source_in + 5.0, 30.0)
            dur_s = _sec_to_fcpxml_rational(asset_dur_sec, fps)
            has_vid = "1"
            has_aud = "1" if clip.type == "video" else "0"
            aud_attrs = ' audioSources="1" audioChannels="2" audioRate="48000"' if has_aud == "1" else ""
            resource_lines.append(
                f'        <asset id="{rid}" name="{name}" start="0s" duration="{dur_s}" '
                f'hasVideo="{has_vid}" hasAudio="{has_aud}" format="r1"{aud_attrs}>\n'
                f'            <media-rep kind="original-media" src="{uri}"/>\n'
                f"        </asset>"
            )

    for aud in timeline.audio_tracks:
        if aud.source not in resource_map:
            rid = f"r{next_res_idx}"
            next_res_idx += 1
            resource_map[aud.source] = rid
            uri = _path_to_file_uri(aud.source)
            name = escape(Path(aud.source).name)
            aud_dur_sec = max(float(aud.duration or timeline.total_duration) + float(aud.source_in) + 5.0, 30.0)
            dur_s = _sec_to_fcpxml_rational(aud_dur_sec, fps)
            resource_lines.append(
                f'        <asset id="{rid}" name="{name}" start="0s" duration="{dur_s}" '
                f'hasVideo="0" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">\n'
                f'            <media-rep kind="original-media" src="{uri}"/>\n'
                f"        </asset>"
            )

    def _find_parent_spine_idx(t_sec: float) -> int:
        if not spine_entries:
            return 0
        tf = _sec_to_frames(t_sec, fps)
        for s_i, (off_f, _, d_f) in enumerate(spine_entries):
            if off_f <= tf < (off_f + d_f):
                return s_i
        return len(spine_entries) - 1

    # Bucket connected items (markers, V2 B-roll, V3 titles, A1-A3 audio) onto their parent V1 spine clip
    children_by_spine: dict[int, list[str]] = {i: [] for i in range(len(timeline.visual_track))}

    # Chapter markers
    for m in timeline.markers:
        if not timeline.visual_track:
            break
        p_idx = _find_parent_spine_idx(m.time_sec)
        p_off_f, p_start_f, _ = spine_entries[p_idx]
        rel_f = max(0, _sec_to_frames(m.time_sec, fps) - p_off_f)
        m_start = _frames_to_fcpxml_rational(p_start_f + rel_f, fps)
        children_by_spine[p_idx].append(
            f'                            <marker start="{m_start}" duration="100/{fps * 100}s" '
            f'value="{escape(m.name)}" note="{escape(m.note)}"/>'
        )

    # Track V2 connected B-roll cutaways (lane="1")
    for b_clip in timeline.broll_track:
        if not timeline.visual_track:
            break
        p_idx = _find_parent_spine_idx(b_clip.timeline_start)
        p_off_f, p_start_f, _ = spine_entries[p_idx]
        rel_f = max(0, _sec_to_frames(b_clip.timeline_start, fps) - p_off_f)
        b_offset = _frames_to_fcpxml_rational(p_start_f + rel_f, fps)
        b_start = _sec_to_fcpxml_rational(b_clip.source_in if b_clip.type == "video" else 0.0, fps)
        b_dur = _frames_to_fcpxml_rational(max(1, _sec_to_frames(b_clip.duration, fps)), fps)
        brid = resource_map[b_clip.source]
        bname = escape(Path(b_clip.source).name)
        children_by_spine[p_idx].append(
            f'                            <asset-clip name="{bname}" lane="1" '
            f'offset="{b_offset}" ref="{brid}" start="{b_start}" duration="{b_dur}" format="r1"/>'
        )

    # Track V3 connected titles & graphics overlays (lane="2")
    for t_idx, t_item in enumerate(timeline.title_track, start=1):
        if not timeline.visual_track or t_item.style == "act_banner":
            continue
        p_idx = _find_parent_spine_idx(t_item.timeline_start)
        p_off_f, p_start_f, _ = spine_entries[p_idx]
        rel_f = max(0, _sec_to_frames(t_item.timeline_start, fps) - p_off_f)
        t_offset = _frames_to_fcpxml_rational(p_start_f + rel_f, fps)
        t_dur = _frames_to_fcpxml_rational(max(1, _sec_to_frames(t_item.duration, fps)), fps)
        ts_id = f"ts_{t_idx}"
        full_label = t_item.text if not t_item.subtitle else f"{t_item.text} — {t_item.subtitle}"
        esc_label = escape(full_label)
        esc_name = escape(f"{t_item.style}: {t_item.text[:32]}")
        children_by_spine[p_idx].append(
            f'                            <title name="{esc_name}" lane="2" offset="{t_offset}" '
            f'ref="r_title" start="0s" duration="{t_dur}">\n'
            f"                                <text>\n"
            f'                                    <text-style ref="{ts_id}">{esc_label}</text-style>\n'
            f"                                </text>\n"
            f'                                <text-style-def id="{ts_id}">\n'
            f'                                    <text-style font="PingFang SC" fontSize="42"/>\n'
            f"                                </text-style-def>\n"
            f"                            </title>"
        )

    # Track A1-A3 connected audio lanes (A1=-1 voiceover/dialogue, A2=-2 sfx/ambience, A3=-3 music)
    role_to_lane = {
        "voiceover": ("-1", "dialogue"),
        "sfx": ("-2", "effects"),
        "ambience": ("-2", "effects"),
        "music": ("-3", "music"),
    }
    for aud in timeline.audio_tracks:
        if not timeline.visual_track:
            break
        lane_str, audio_role = role_to_lane.get(aud.role, ("-2", "effects"))
        p_idx = _find_parent_spine_idx(aud.timeline_start)
        p_off_f, p_start_f, _ = spine_entries[p_idx]
        rel_f = max(0, _sec_to_frames(aud.timeline_start, fps) - p_off_f)
        a_offset = _frames_to_fcpxml_rational(p_start_f + rel_f, fps)
        a_start = _sec_to_fcpxml_rational(aud.source_in, fps)
        raw_adur = aud.duration if (aud.duration and aud.duration > 0) else timeline.total_duration
        a_dur_f = min(max(1, _sec_to_frames(raw_adur, fps)), max(1, total_seq_frames - _sec_to_frames(aud.timeline_start, fps)))
        a_dur = _frames_to_fcpxml_rational(a_dur_f, fps)
        arid = resource_map[aud.source]
        aname = escape(Path(aud.source).name)
        children_by_spine[p_idx].append(
            f'                            <asset-clip name="{aname}" lane="{lane_str}" '
            f'offset="{a_offset}" ref="{arid}" start="{a_start}" duration="{a_dur}" audioRole="{audio_role}"/>'
        )

    spine_lines: list[str] = []
    for idx, clip in enumerate(timeline.visual_track):
        rid = resource_map[clip.source]
        name = escape(Path(clip.source).name)
        off_f, start_f, dur_f = spine_entries[idx]
        offset_s = _frames_to_fcpxml_rational(off_f, fps)
        start_s = _frames_to_fcpxml_rational(start_f, fps)
        dur_s = _frames_to_fcpxml_rational(dur_f, fps)
        aud_role_attr = ' audioRole="dialogue"' if (clip.type == "video" and clip.roll_type == "a_roll") else ""
        inner = children_by_spine.get(idx, [])
        if inner:
            spine_lines.append(
                f'                        <asset-clip name="{name}" offset="{offset_s}" '
                f'ref="{rid}" start="{start_s}" duration="{dur_s}" format="r1"{aud_role_attr}>\n'
                + "\n".join(inner)
                + "\n                        </asset-clip>"
            )
        else:
            spine_lines.append(
                f'                        <asset-clip name="{name}" offset="{offset_s}" '
                f'ref="{rid}" start="{start_s}" duration="{dur_s}" format="r1"{aud_role_attr}/>'
            )

    seq_dur_s = _frames_to_fcpxml_rational(total_seq_frames, fps)
    proj_esc = escape(timeline.project_name or "Timeline")
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.11">
    <resources>
{chr(10).join(resource_lines)}
    </resources>
    <library>
        <event name="{proj_esc}">
            <project name="{proj_esc}">
                <sequence format="r1" duration="{seq_dur_s}" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
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

    v2_children: list[dict] = []
    for b_clip in timeline.broll_track:
        src_in = b_clip.source_in if b_clip.type == "video" else 0.0
        v2_children.append(
            {
                "OTIO_SCHEMA": "Clip.2",
                "name": Path(b_clip.source).name,
                "metadata": {
                    "id": b_clip.id,
                    "type": b_clip.type,
                    "track_lane": 2,
                    "timeline_start": b_clip.timeline_start,
                    "narrative_note": b_clip.narrative_note,
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
                        "value": round(b_clip.duration * fps, 2),
                        "rate": fps,
                    },
                },
                "media_references": {
                    "DEFAULT_MEDIA": {
                        "OTIO_SCHEMA": "ExternalReference.1",
                        "target_url": _path_to_file_uri(b_clip.source),
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
                    "name": "V2_BRoll_Cutaways",
                    "kind": "Video",
                    "children": v2_children,
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
    """Export canonical timeline.fcpxml (plus project-named .fcpxml/.edl/.otio) into output_dir."""
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    raw_stem = re.sub(r'[<>:"/\\|?*]+', "_", (timeline.project_name or "timeline")).strip() or "timeline"

    canonical_fcpxml = export_fcpxml(timeline, out_root / "timeline.fcpxml")
    if raw_stem.lower() != "timeline":
        named_fcpxml = out_root / f"{raw_stem}.fcpxml"
        named_fcpxml.write_text(canonical_fcpxml.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "fcpxml": canonical_fcpxml,
        "edl": export_edl(timeline, out_root / f"{raw_stem}.edl"),
        "otio": export_otio(timeline, out_root / f"{raw_stem}.otio"),
    }
