"""Multi-track FFmpeg video/audio compositor with title overlays and automatic dialogue ducking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable, Optional

from ..analyzer.media_probe import probe_video_file
from ..models.schema import EditTimeline, VisualTrackItem
from .ken_burns import has_hevc_nvenc, render_photo_clip
from .subtitles import export_ass_subtitles, export_srt_subtitles
from .titles import render_title_png


def _video_encoder_args(hdr: bool) -> list[str]:
    if hdr:
        if has_hevc_nvenc():
            return [
                "-c:v",
                "hevc_nvenc",
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
                "-bsf:v",
                "hevc_metadata=colour_primaries=9:transfer_characteristics=18:matrix_coefficients=9",
                "-tag:v",
                "hvc1",
            ]
        return [
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p10le",
            "-preset",
            "veryfast",
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
    return [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]


def _render_video_segment(
    item: VisualTrackItem,
    output_mp4: Path,
    width: int,
    height: int,
    fps: int,
    hdr: bool = True,
) -> Path:
    """Trim, scale/pad, fade, and normalize a single video clip to match master timeline settings."""
    info = probe_video_file(Path(item.source))
    has_audio = bool(info.get("has_audio", False))
    src_trc = str(info.get("color_transfer", "bt709"))
    dur = item.duration
    src_in = item.source_in

    vf_parts = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
        f"fps={fps}",
        "setsar=1:1",
    ]
    if item.speed != 1.0 and item.speed > 0:
        vf_parts.insert(0, f"setpts={1.0 / item.speed:.4f}*PTS")

    if item.transition_in.type == "fade_black" and item.transition_in.duration > 0:
        fd = min(dur * 0.45, item.transition_in.duration)
        vf_parts.append(f"fade=t=in:st=0:d={fd:.3f}")
    if item.transition_out.type == "fade_black" and item.transition_out.duration > 0:
        fd = min(dur * 0.45, item.transition_out.duration)
        st = max(0.0, dur - fd)
        vf_parts.append(f"fade=t=out:st={st:.3f}:d={fd:.3f}")

    if hdr:
        if src_trc not in ("arib-std-b67", "smpte2084"):
            vf_parts.append(
                "zscale=primariesin=709:transferin=709:matrixin=709:"
                "primaries=2020:transfer=arib-std-b67:matrix=2020_ncl"
            )
        vf_parts.extend(
            [
                "format=p010le",
                "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc",
            ]
        )
    else:
        vf_parts.append("format=yuv420p")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{src_in:.3f}",
        "-i",
        str(item.source),
    ]

    if not has_audio or item.volume <= 0.0:
        cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"])
        cmd.extend(
            [
                "-t",
                f"{dur:.3f}",
                "-vf",
                ",".join(vf_parts),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
        )
    else:
        af_parts = [
            "aresample=48000",
            "aformat=sample_fmts=fltp:channel_layouts=stereo",
            f"volume={item.volume:.3f}",
        ]
        if item.transition_in.duration > 0:
            af_parts.append(f"afade=t=in:st=0:d={min(0.4, item.transition_in.duration):.3f}")
        if item.transition_out.duration > 0:
            fd = min(0.4, item.transition_out.duration)
            af_parts.append(f"afade=t=out:st={max(0.0, dur - fd):.3f}:d={fd:.3f}")
        cmd.extend(
            [
                "-t",
                f"{dur:.3f}",
                "-vf",
                ",".join(vf_parts),
                "-af",
                ",".join(af_parts),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
            ]
        )

    cmd.extend(
        [
            *_video_encoder_args(hdr),
            "-video_track_timescale",
            "30000",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-shortest",
            str(output_mp4),
        ]
    )

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg video segment render failed for {item.source}: {proc.stderr}")
    return output_mp4


def _build_ducking_volume_expr(
    timeline: EditTimeline,
    base_vol: float,
    duck_vol: float,
) -> str:
    """Build an FFmpeg volume expression that ducks BGM during subtitle/dialogue windows and video clips."""
    windows: list[tuple[float, float]] = []
    for sub in timeline.subtitles.items:
        windows.append((max(0.0, sub.start - 0.25), sub.end + 0.35))

    # Also duck slightly if a video clip has active camera audio and no explicit subtitles
    if not windows:
        for clip in timeline.visual_track:
            if clip.type == "video" and clip.volume > 0.1:
                windows.append((clip.timeline_start, clip.timeline_start + clip.duration))

    if not windows:
        return f"{base_vol:.3f}"

    # Merge overlapping windows
    windows.sort(key=lambda w: w[0])
    merged: list[tuple[float, float]] = [windows[0]]
    for s, e in windows[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e + 0.3:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))

    cond_terms = [f"between(t,{s:.2f},{e:.2f})" for s, e in merged[:25]]
    cond_str = "+".join(cond_terms)
    return f"if(gt({cond_str},0),{duck_vol:.3f},{base_vol:.3f})"


def render_timeline(
    timeline: EditTimeline,
    output_mp4: str | Path,
    work_dir: Optional[str | Path] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> dict[str, Path]:
    """Render an EditTimeline into a final MP4 video along with SRT and ASS subtitle files."""
    if not timeline.visual_track:
        raise ValueError("Cannot render an empty visual_track.")

    out_path = Path(output_mp4).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w_dir = Path(work_dir).resolve() if work_dir else (out_path.parent / "render_cache")
    seg_dir = w_dir / "segments"
    titles_dir = w_dir / "titles"
    seg_dir.mkdir(parents=True, exist_ok=True)
    titles_dir.mkdir(parents=True, exist_ok=True)

    # Ensure sequential start times are consistent
    timeline.recalculate_sequential_starts()

    width = timeline.settings.width
    height = timeline.settings.height
    fps = timeline.settings.fps
    hdr = timeline.settings.hdr
    total_dur = timeline.total_duration

    # 1. Export .srt and .ass subtitle files alongside the output video
    srt_path = out_path.with_suffix(".srt")
    ass_path = out_path.with_suffix(".ass")
    export_srt_subtitles(timeline, srt_path)
    export_ass_subtitles(timeline, ass_path)

    # 2. Render each visual clip (video or Ken Burns / Gallery Frame photo) to normalized segment MP4
    segment_files: list[Path] = []
    num_clips = len(timeline.visual_track)
    for idx, item in enumerate(timeline.visual_track):
        if progress_callback:
            progress_callback(
                f"Rendering clip {idx + 1}/{num_clips} ({Path(item.source).name})",
                (idx / max(1, num_clips)) * 0.70,
            )
        sig_payload = json.dumps(
            {
                "item": item.model_dump(mode="json"),
                "w": width,
                "h": height,
                "fps": fps,
                "hdr": hdr,
            },
            sort_keys=True,
        )
        sig_hash = hashlib.sha1(sig_payload.encode("utf-8")).hexdigest()[:8]
        seg_mp4 = seg_dir / f"seg_{idx + 1:03d}_{item.id}_{sig_hash}.mp4"
        if not seg_mp4.exists() or seg_mp4.stat().st_size < 1024:
            if item.type == "photo":
                render_photo_clip(
                    photo_path=item.source,
                    output_mp4=seg_mp4,
                    duration=item.duration,
                    width=width,
                    height=height,
                    fps=fps,
                    hdr=hdr,
                    ken_burns=item.ken_burns,
                    transition_in=item.transition_in,
                    transition_out=item.transition_out,
                )
            else:
                _render_video_segment(
                    item=item,
                    output_mp4=seg_mp4,
                    width=width,
                    height=height,
                    fps=fps,
                    hdr=hdr,
                )
        segment_files.append(seg_mp4)

    # 3. Concatenate normalized segments into base visual + camera audio stream
    if progress_callback:
        progress_callback("Concatenating visual storyline segments...", 0.75)

    concat_list_path = w_dir / "segments_concat.txt"
    concat_lines = [f"file '{p.as_posix()}'" for p in segment_files]
    concat_list_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    base_concat_mp4 = w_dir / "base_concat.mp4"
    concat_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c",
        "copy",
        str(base_concat_mp4),
    ]
    proc = subprocess.run(concat_cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg concat failed: {proc.stderr}")

    # 4. Render RGBA PNG plates for title_track and composite with BGM audio ducking
    if progress_callback:
        progress_callback("Compositing titles, overlays, and mixing auto-ducked audio...", 0.85)

    active_titles = [t for t in timeline.title_track if t.timeline_start < total_dur]
    title_pngs: list[Path] = []
    for t_idx, t_item in enumerate(active_titles):
        png_path = titles_dir / f"title_{t_idx + 1:03d}_{t_item.id}.png"
        render_title_png(t_item, png_path, width=width, height=height)
        title_pngs.append(png_path)

    active_bgm = [a for a in timeline.audio_tracks if Path(a.source).exists()]

    # If no titles and no external BGM, just copy base_concat_mp4 to out_path
    if not active_titles and not active_bgm:
        out_path.write_bytes(base_concat_mp4.read_bytes())
        if progress_callback:
            progress_callback("Render complete!", 1.0)
        return {"video": out_path, "srt": srt_path, "ass": ass_path}

    final_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(base_concat_mp4),
    ]

    for t_item, png_path in zip(active_titles, title_pngs):
        final_cmd.extend(["-loop", "1", "-t", f"{t_item.duration:.3f}", "-i", str(png_path)])

    bgm_input_start_idx = 1 + len(active_titles)
    for a_item in active_bgm:
        final_cmd.extend(["-ss", f"{a_item.source_in:.3f}", "-i", str(a_item.source)])

    filter_parts: list[str] = []
    cur_v = "[0:v]"
    ov_fmt = "yuv420p10" if hdr else "auto"
    for idx, t_item in enumerate(active_titles):
        in_idx = 1 + idx
        st = t_item.timeline_start
        dur = t_item.duration
        end_t = st + dur
        fi = min(dur * 0.4, t_item.fade_in)
        fo = min(dur * 0.4, t_item.fade_out)
        fo_st = max(0.0, dur - fo)
        ov_label = f"[ov{idx}]"
        next_v = f"[v{idx + 1}]"
        filter_parts.append(
            f"[{in_idx}:v]format=rgba,"
            f"fade=t=in:st=0:d={fi:.3f}:alpha=1,"
            f"fade=t=out:st={fo_st:.3f}:d={fo:.3f}:alpha=1,"
            f"setpts=PTS+{st:.3f}/TB{ov_label}"
        )
        filter_parts.append(
            f"{cur_v}{ov_label}overlay=0:0:format={ov_fmt}:eof_action=pass:enable='between(t,{st:.3f},{end_t:.3f})'{next_v}"
        )
        cur_v = next_v

    if hdr and active_titles:
        filter_parts.append(
            f"{cur_v}format=p010le,setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc[vhdr]"
        )
        cur_v = "[vhdr]"

    # Audio mixing & auto-ducking
    if active_bgm:
        mix_inputs = ["[0:a]"]
        for a_idx, a_item in enumerate(active_bgm):
            in_idx = bgm_input_start_idx + a_idx
            a_dur = min(total_dur, a_item.duration or total_dur)
            vol_expr = (
                _build_ducking_volume_expr(timeline, a_item.volume, a_item.duck_volume)
                if a_item.duck_under_dialogue
                else f"{a_item.volume:.3f}"
            )
            af_chain = [
                f"atrim=0:{a_dur:.3f}",
                "asetpts=PTS-STARTPTS",
                "aresample=48000",
                "aformat=sample_fmts=fltp:channel_layouts=stereo",
                f"volume='{vol_expr}':eval=frame",
            ]
            if a_item.fade_in > 0:
                af_chain.append(f"afade=t=in:st=0:d={min(a_dur * 0.4, a_item.fade_in):.3f}")
            if a_item.fade_out > 0:
                fo = min(a_dur * 0.4, a_item.fade_out)
                af_chain.append(f"afade=t=out:st={max(0.0, a_dur - fo):.3f}:d={fo:.3f}")
            if a_item.timeline_start > 0:
                delay_ms = int(round(a_item.timeline_start * 1000))
                af_chain.append(f"adelay={delay_ms}|{delay_ms}")

            lbl = f"[bgm{a_idx}]"
            filter_parts.append(f"[{in_idx}:a]{','.join(af_chain)}{lbl}")
            mix_inputs.append(lbl)

        filter_parts.append(
            f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=2:normalize=0[aout]"
        )
        final_audio_map = "[aout]"
    else:
        final_audio_map = "0:a:0"

    final_cmd.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            cur_v,
            "-map",
            final_audio_map,
            "-t",
            f"{total_dur:.3f}",
            *_video_encoder_args(hdr),
            "-c:a",
            "aac",
            "-b:a",
            "320k" if hdr else "192k",
            str(out_path),
        ]
    )

    proc = subprocess.run(final_cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg final composite failed: {proc.stderr}")

    if progress_callback:
        progress_callback("Render complete!", 1.0)

    return {"video": out_path, "srt": srt_path, "ass": ass_path}
