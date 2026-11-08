"""Multi-track FFmpeg video/audio compositor with Track V2 B-Roll cutaways, pluggable VFX, parallel NVENC rendering, and smart audio ducking."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Optional

from ..analyzer.media_probe import probe_video_file
from ..models.schema import EditTimeline, EffectTrackItem, VisualTrackItem
from .ken_burns import (
    build_hdr_color_filters,
    get_video_encoder_args,
    has_hevc_nvenc,
    render_frosted_video_overlay_png,
    render_photo_clip,
)
from .subtitles import export_ass_subtitles, export_srt_subtitles
from .titles import render_title_png


# Extensible registry mapping Stage-40 EffectTrackItem.effect_type -> FFmpeg filter builder
EFFECT_FILTER_BUILDERS: dict[str, Callable[[EffectTrackItem], str]] = {
    "warm_glow": lambda fx: (
        f"eq=saturation={float(fx.parameters.get('saturation', 1.08)):.2f}:"
        f"brightness={float(fx.parameters.get('brightness', 0.02)):.3f}"
    ),
    "vignette_focus": lambda fx: f"vignette=angle={float(fx.parameters.get('angle_rad', 0.42)):.2f}",
    "contrast_pop": lambda fx: (
        f"eq=contrast={float(fx.parameters.get('contrast', 1.06)):.2f}:"
        f"saturation={float(fx.parameters.get('saturation', 1.05)):.2f}"
    ),
}


def register_effect_filter(effect_type: str, builder: Callable[[EffectTrackItem], str]) -> None:
    """Register a custom FFmpeg filter builder for a Stage-40 EffectTrackItem."""
    EFFECT_FILTER_BUILDERS[effect_type] = builder


def _video_encoder_args(hdr: bool, force_software: bool = False) -> list[str]:
    """Return cross-platform hardware-accelerated encoder arguments (VideoToolbox / NVENC / AMF / QSV / libx265)."""
    return get_video_encoder_args(hdr=hdr, force_software=force_software)


def _render_video_segment(
    item: VisualTrackItem,
    output_mp4: Path,
    width: int,
    height: int,
    fps: int,
    hdr: bool = True,
    attached_effects: Optional[list[EffectTrackItem]] = None,
) -> Path:
    """Trim, scale/composite (with ambient blurred backdrop for portrait 9:16 videos), overlay documentary card, and encode."""
    info = probe_video_file(Path(item.source))
    has_audio = bool(info.get("has_audio", False))
    src_trc = str(info.get("color_transfer", "bt709"))
    src_w = int(info.get("width", 1920) or 1920)
    src_h = int(info.get("height", 1080) or 1080)
    is_portrait = src_w < src_h
    dur = item.duration
    src_in = item.source_in

    kb = item.ken_burns
    has_doc_card = bool(kb and kb.caption_title and kb.caption_title.strip())
    card_png: Optional[Path] = None
    if has_doc_card:
        card_png = output_mp4.with_suffix(".card.png")
        render_frosted_video_overlay_png(
            output_png=card_png,
            width=width,
            height=height,
            caption_title=kb.caption_title or "",
            caption_subtitle=kb.caption_subtitle or "",
            camera_badge=kb.camera_badge or "",
        )

    post_filters: list[str] = [f"fps={fps}", "setsar=1:1"]
    for fx in attached_effects or []:
        builder = EFFECT_FILTER_BUILDERS.get(fx.effect_type)
        if builder is not None:
            post_filters.append(builder(fx))

    if item.transition_in.type not in ("none", "cut") and item.transition_in.duration > 0:
        if item.transition_in.type == "dip_to_white":
            fd = min(dur * 0.35, item.transition_in.duration)
            post_filters.append(f"fade=t=in:st=0:d={fd:.3f}:c=#F6F2E8")
        elif item.transition_in.type in ("dissolve", "crossfade"):
            fd = min(dur * 0.25, item.transition_in.duration, 0.26)
            post_filters.append(f"fade=t=in:st=0:d={fd:.3f}:c=#0B0E14")
        else:
            fd = min(dur * 0.40, item.transition_in.duration)
            post_filters.append(f"fade=t=in:st=0:d={fd:.3f}")
    if item.transition_out.type not in ("none", "cut") and item.transition_out.duration > 0:
        if item.transition_out.type == "dip_to_white":
            fd = min(dur * 0.35, item.transition_out.duration)
            st = max(0.0, dur - fd)
            post_filters.append(f"fade=t=out:st={st:.3f}:d={fd:.3f}:c=#F6F2E8")
        elif item.transition_out.type in ("dissolve", "crossfade"):
            fd = min(dur * 0.25, item.transition_out.duration, 0.26)
            st = max(0.0, dur - fd)
            post_filters.append(f"fade=t=out:st={st:.3f}:d={fd:.3f}:c=#0B0E14")
        else:
            fd = min(dur * 0.40, item.transition_out.duration)
            st = max(0.0, dur - fd)
            post_filters.append(f"fade=t=out:st={st:.3f}:d={fd:.3f}")

    if hdr:
        post_filters.extend(build_hdr_color_filters(src_trc if src_trc != "bt709" else "709"))
    else:
        post_filters.append("format=yuv420p")

    speed_prefix = f"setpts={1.0 / item.speed:.4f}*PTS," if (item.speed != 1.0 and item.speed > 0) else ""

    # Build filtergraph: handles portrait 9:16 ambient blurred backdrop + optional frosted documentary card overlay
    filter_complex_parts: list[str] = []
    if is_portrait:
        fg_h = int(round(height * (0.76 if has_doc_card else 0.84)))
        fg_h = fg_h - (fg_h % 2)
        pad_px = max(6, int(round(12 * (width / 1920.0))))
        pad_px = pad_px - (pad_px % 2)
        y_pos = int(round(height * (0.04 if has_doc_card else 0.08)))
        filter_complex_parts.append(
            f"[0:v]{speed_prefix}split=2[vbg][vfg];"
            f"[vbg]scale={width // 4}:{height // 4}:force_original_aspect_ratio=increase,"
            f"crop={width // 4}:{height // 4},boxblur=16:8,eq=brightness=-0.18:saturation=0.85,"
            f"scale={width}:{height}[bg_blur];"
            f"[vfg]scale=-2:{fg_h},pad=iw+{pad_px * 2}:ih+{pad_px * 2}:{pad_px}:{pad_px}:color=#F8F6F0[fg_framed];"
            f"[bg_blur][fg_framed]overlay=(W-w)/2:{y_pos}[vbase]"
        )
    else:
        filter_complex_parts.append(
            f"[0:v]{speed_prefix}scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black[vbase]"
        )

    cur_v = "[vbase]"
    card_input_idx = 1 if (has_audio and item.volume > 0.0) else 2
    if card_png is not None and card_png.exists():
        fo_st = max(0.2, dur - 0.45)
        filter_complex_parts.append(
            f"[{card_input_idx}:v]format=rgba,"
            f"fade=t=in:st=0.15:d=0.45:alpha=1,"
            f"fade=t=out:st={fo_st:.3f}:d=0.40:alpha=1[vcard];"
            f"{cur_v}[vcard]overlay=0:0:eof_action=pass[vwithcard]"
        )
        cur_v = "[vwithcard]"

    filter_complex_parts.append(f"{cur_v}{','.join(post_filters)}[vout]")

    def _build_cmd(force_sw: bool) -> list[str]:
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
        if card_png is not None and card_png.exists():
            cmd.extend(["-loop", "1", "-t", f"{dur:.3f}", "-i", str(card_png)])

        if not has_audio or item.volume <= 0.0:
            cmd.extend(
                [
                    "-t",
                    f"{dur:.3f}",
                    "-filter_complex",
                    ";".join(filter_complex_parts),
                    "-map",
                    "[vout]",
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
            cf_in = max(item.audio_crossfade_sec, item.transition_in.duration)
            cf_out = max(item.audio_crossfade_sec, item.transition_out.duration)
            if cf_in > 0:
                af_parts.append(f"afade=t=in:st=0:d={min(0.45, cf_in):.3f}")
            if cf_out > 0:
                fd = min(0.45, cf_out)
                af_parts.append(f"afade=t=out:st={max(0.0, dur - fd):.3f}:d={fd:.3f}")
            cmd.extend(
                [
                    "-t",
                    f"{dur:.3f}",
                    "-filter_complex",
                    ";".join(filter_complex_parts),
                    "-af",
                    ",".join(af_parts),
                    "-map",
                    "[vout]",
                    "-map",
                    "0:a:0",
                ]
            )

        cmd.extend(
            [
                *_video_encoder_args(hdr, force_software=force_sw),
                "-metadata:s:v:0",
                "rotate=0",
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
        return cmd

    proc = subprocess.run(_build_cmd(force_sw=False), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        # Automatic retry with software libx265 if NVENC session limit was reached
        proc = subprocess.run(_build_cmd(force_sw=True), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg video segment render failed for {item.source}: {proc.stderr}")
    return output_mp4


def _build_ducking_volume_expr(
    timeline: EditTimeline,
    base_vol: float,
    duck_vol: float,
) -> str:
    """
    Build an FFmpeg volume expression that ducks BGM ONLY during spoken dialogue or loud A-roll camera audio.
    Never ducks BGM for silent documentary/educational text captions (`speaker` in {'', 'narrator', 'caption', 'plaque'}).
    """
    windows: list[tuple[float, float]] = []
    for sub in timeline.subtitles.items:
        spk = (sub.speaker or "").strip().lower()
        if spk and spk not in ("narrator", "caption", "plaque", "info", "subtitle", "doc"):
            windows.append((max(0.0, sub.start - 0.35), sub.end + 0.45))

    for clip in timeline.visual_track:
        if clip.type == "video" and clip.volume >= 0.45:
            windows.append((clip.timeline_start, clip.timeline_start + clip.duration))

    if not windows:
        return f"{base_vol:.3f}"

    windows.sort(key=lambda w: w[0])
    merged: list[tuple[float, float]] = [windows[0]]
    for s, e in windows[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e + 0.4:
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
    """Render an EditTimeline (Track V1 + Track V2 B-Roll Cutaways + Track V3 Titles + 3-Layer Audio) into MP4."""
    if not timeline.visual_track:
        raise ValueError("Cannot render an empty visual_track.")

    out_path = Path(output_mp4).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w_dir = Path(work_dir).resolve() if work_dir else (out_path.parent / "render_cache")
    seg_dir = w_dir / "segments"
    titles_dir = w_dir / "titles"
    seg_dir.mkdir(parents=True, exist_ok=True)
    titles_dir.mkdir(parents=True, exist_ok=True)

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

    effects_by_clip: dict[str, list[EffectTrackItem]] = {}
    for fx in timeline.effects_track:
        if fx.target_clip_id:
            effects_by_clip.setdefault(fx.target_clip_id, []).append(fx)

    # 2. Render Track V1 visual clips in parallel across workers (RTX 4080 SUPER NVENC + 32-Thread CPU)
    num_clips = len(timeline.visual_track)
    segment_files: list[Path] = [Path()] * num_clips
    tasks_to_run: list[tuple[int, VisualTrackItem, Path, list[EffectTrackItem]]] = []

    for idx, item in enumerate(timeline.visual_track):
        clip_fxs = effects_by_clip.get(item.id, [])
        sig_payload = json.dumps(
            {
                "v": 2,
                "item": item.model_dump(mode="json"),
                "fx": [f.model_dump(mode="json") for f in clip_fxs],
                "w": width,
                "h": height,
                "fps": fps,
                "hdr": hdr,
            },
            sort_keys=True,
        )
        sig_hash = hashlib.sha1(sig_payload.encode("utf-8")).hexdigest()[:8]
        seg_mp4 = seg_dir / f"seg_{idx + 1:03d}_{item.id}_{sig_hash}.mp4"
        segment_files[idx] = seg_mp4
        if not seg_mp4.exists() or seg_mp4.stat().st_size < 1024:
            tasks_to_run.append((idx, item, seg_mp4, clip_fxs))

    def _render_one_v1(task_tuple: tuple[int, VisualTrackItem, Path, list[EffectTrackItem]]) -> tuple[int, Path]:
        t_idx, t_item, t_mp4, t_fxs = task_tuple
        if t_item.type == "photo":
            render_photo_clip(
                photo_path=t_item.source,
                output_mp4=t_mp4,
                duration=t_item.duration,
                width=width,
                height=height,
                fps=fps,
                hdr=hdr,
                ken_burns=t_item.ken_burns,
                transition_in=t_item.transition_in,
                transition_out=t_item.transition_out,
            )
        else:
            _render_video_segment(
                item=t_item,
                output_mp4=t_mp4,
                width=width,
                height=height,
                fps=fps,
                hdr=hdr,
                attached_effects=t_fxs,
            )
        return t_idx, t_mp4

    if tasks_to_run:
        # 5 concurrent workers stays within consumer NVIDIA GeForce NVENC session limits; 4 on macOS keeps 2019 Intel MacBook Pro thermally balanced
        worker_cap = 4 if sys.platform == "darwin" else 5
        max_workers = min(worker_cap, max(1, (os.cpu_count() or 8) // 4), len(tasks_to_run))
        completed_count = num_clips - len(tasks_to_run)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_render_one_v1, t): t for t in tasks_to_run}
            for fut in as_completed(futures):
                done_idx, _ = fut.result()
                completed_count += 1
                if progress_callback:
                    progress_callback(
                        f"Rendered V1 clip {completed_count}/{num_clips} ({Path(timeline.visual_track[done_idx].source).name})",
                        (completed_count / max(1, num_clips)) * 0.68,
                    )

    # 2b. Render Track V2 B-Roll Cutaway segments
    active_broll = [b for b in timeline.broll_track if b.timeline_start < total_dur]
    broll_files: list[Path] = []
    for b_idx, b_item in enumerate(active_broll):
        sig_payload = json.dumps(
            {
                "v": 2,
                "b_item": b_item.model_dump(mode="json"),
                "w": width,
                "h": height,
                "fps": fps,
                "hdr": hdr,
            },
            sort_keys=True,
        )
        sig_hash = hashlib.sha1(sig_payload.encode("utf-8")).hexdigest()[:8]
        b_mp4 = seg_dir / f"broll_{b_idx + 1:03d}_{b_item.id}_{sig_hash}.mp4"
        if not b_mp4.exists() or b_mp4.stat().st_size < 1024:
            if b_item.type == "photo":
                render_photo_clip(
                    photo_path=b_item.source,
                    output_mp4=b_mp4,
                    duration=b_item.duration,
                    width=width,
                    height=height,
                    fps=fps,
                    hdr=hdr,
                    ken_burns=b_item.ken_burns,
                    transition_in=b_item.transition_in,
                    transition_out=b_item.transition_out,
                )
            else:
                _render_video_segment(
                    item=b_item,
                    output_mp4=b_mp4,
                    width=width,
                    height=height,
                    fps=fps,
                    hdr=hdr,
                )
        broll_files.append(b_mp4)

    # 3. Concatenate normalized V1 segments into base visual + camera audio stream
    if progress_callback:
        progress_callback("Concatenating Track V1 storyline segments...", 0.75)

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
        "-display_rotation",
        "0",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c",
        "copy",
        "-metadata:s:v:0",
        "rotate=0",
        str(base_concat_mp4),
    ]
    proc = subprocess.run(concat_cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg concat failed: {proc.stderr}")

    # 4. Render RGBA PNG plates for title_track and composite V2 B-Roll + V3 Titles + Full-Length BGM
    if progress_callback:
        progress_callback("Compositing Track V2 B-Roll cutaways, Track V3 titles, and mastered BGM...", 0.85)

    active_titles = [t for t in timeline.title_track if t.timeline_start < total_dur]
    title_pngs: list[Path] = []
    for t_idx, t_item in enumerate(active_titles):
        png_path = titles_dir / f"title_{t_idx + 1:03d}_{t_item.id}.png"
        render_title_png(t_item, png_path, width=width, height=height)
        title_pngs.append(png_path)

    active_bgm = [a for a in timeline.audio_tracks if Path(a.source).exists()]

    if not active_broll and not active_titles and not active_bgm:
        out_path.write_bytes(base_concat_mp4.read_bytes())
        if progress_callback:
            progress_callback("Render complete!", 1.0)
        return {"video": out_path, "srt": srt_path, "ass": ass_path}

    def _build_final_cmd(force_sw: bool) -> list[str]:
        final_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(base_concat_mp4),
        ]

        for b_mp4 in broll_files:
            final_cmd.extend(["-i", str(b_mp4)])

        title_input_start_idx = 1 + len(active_broll)
        for t_item, png_path in zip(active_titles, title_pngs):
            final_cmd.extend(["-loop", "1", "-t", f"{t_item.duration:.3f}", "-i", str(png_path)])

        bgm_input_start_idx = title_input_start_idx + len(active_titles)
        for a_item in active_bgm:
            # Always enable -stream_loop -1 so BGM never cuts out before the end of the video!
            final_cmd.extend(["-stream_loop", "-1", "-ss", f"{a_item.source_in:.3f}", "-i", str(a_item.source)])

        filter_parts: list[str] = []
        cur_v = "[0:v]"
        ov_fmt = "yuv420p10" if hdr else "auto"

        # Composite Track V2 B-Roll Cutaways over Track V1
        for b_idx, b_item in enumerate(active_broll):
            in_idx = 1 + b_idx
            st = b_item.timeline_start
            dur = b_item.duration
            end_t = st + dur
            bv_label = f"[bv{b_idx}]"
            next_v = f"[vb{b_idx + 1}]"
            filter_parts.append(f"[{in_idx}:v]setpts=PTS+{st:.3f}/TB{bv_label}")
            filter_parts.append(
                f"{cur_v}{bv_label}overlay=0:0:format={ov_fmt}:eof_action=pass:enable='between(t,{st:.3f},{end_t:.3f})'{next_v}"
            )
            cur_v = next_v

        # Composite Track V3 RGBA Title Overlays
        for idx, t_item in enumerate(active_titles):
            in_idx = title_input_start_idx + idx
            st = t_item.timeline_start
            dur = t_item.duration
            end_t = st + dur
            fi = min(dur * 0.4, t_item.fade_in)
            fo = min(dur * 0.4, t_item.fade_out)
            fo_st = max(0.0, dur - fo)
            ov_label = f"[ov{idx}]"
            next_v = f"[vt{idx + 1}]"
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

        if hdr and (active_broll or active_titles):
            filter_parts.append(
                f"{cur_v}format=p010le,setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc[vhdr]"
            )
            cur_v = "[vhdr]"

        # Audio mixing & smart auto-ducking
        if active_bgm:
            mix_inputs = ["[0:a]"]
            for a_idx, a_item in enumerate(active_bgm):
                in_idx = bgm_input_start_idx + a_idx
                # Ensure primary BGM covers 100% of total_dur when starting at 0.0
                a_dur = total_dur if a_item.timeline_start <= 0.1 else min(total_dur - a_item.timeline_start, a_item.duration or total_dur)
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

        if active_broll or active_titles:
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
                    *_video_encoder_args(hdr, force_software=force_sw),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "320k" if hdr else "192k",
                    str(out_path),
                ]
            )
        else:
            # Fast stream-copy video when only BGM audio is being mixed!
            final_cmd.extend(
                [
                    "-filter_complex",
                    ";".join(filter_parts),
                    "-map",
                    "0:v:0",
                    "-map",
                    final_audio_map,
                    "-t",
                    f"{total_dur:.3f}",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "320k" if hdr else "192k",
                    str(out_path),
                ]
            )
        return final_cmd

    proc = subprocess.run(_build_final_cmd(force_sw=False), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        proc = subprocess.run(_build_final_cmd(force_sw=True), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg final composite failed: {proc.stderr}")

    if progress_callback:
        progress_callback("Render complete!", 1.0)

    return {"video": out_path, "srt": srt_path, "ass": ass_path}


def build_segment_subtimeline(
    timeline: EditTimeline,
    start_sec: float,
    end_sec: float,
    fast_preview: bool = True,
) -> EditTimeline:
    """Mechanically slice an EditTimeline to [start_sec, end_sec] for fast surgical preview rendering."""
    st_bound = max(0.0, float(start_sec))
    en_bound = min(float(timeline.total_duration), float(end_sec))
    if en_bound <= st_bound:
        en_bound = min(float(timeline.total_duration), st_bound + 5.0)

    sub_tl = timeline.model_copy(deep=True)
    if fast_preview:
        sub_tl.settings.width = 1280
        sub_tl.settings.height = 720
        sub_tl.settings.fps = 24
        sub_tl.settings.hdr = False

    new_v1 = []
    for clip in sub_tl.visual_track:
        c_st = clip.timeline_start
        c_en = c_st + clip.duration
        if c_en <= st_bound + 0.02 or c_st >= en_bound - 0.02:
            continue
        ov_st = max(st_bound, c_st)
        ov_en = min(en_bound, c_en)
        lead_trim = round(ov_st - c_st, 3)
        new_dur = round(max(0.5, ov_en - ov_st), 2)
        if clip.type == "video":
            clip.source_in = round(clip.source_in + lead_trim, 2)
        clip.duration = new_dur
        clip.source_out = round(clip.source_in + new_dur, 2)
        new_v1.append(clip)

    sub_tl.visual_track = new_v1
    sub_tl.recalculate_sequential_starts()
    seg_dur = round(sub_tl.total_duration, 2)

    new_broll = []
    for b in sub_tl.broll_track:
        b_st = b.timeline_start
        b_en = b_st + b.duration
        if b_en <= st_bound + 0.05 or b_st >= en_bound - 0.05:
            continue
        ov_st = max(st_bound, b_st)
        ov_en = min(en_bound, b_en)
        b.timeline_start = round(max(0.0, ov_st - st_bound), 3)
        b.duration = round(max(0.5, min(seg_dur - b.timeline_start, ov_en - ov_st)), 2)
        b.source_out = round(b.source_in + b.duration, 2)
        new_broll.append(b)
    sub_tl.broll_track = new_broll

    new_titles = []
    for t in sub_tl.title_track:
        t_st = t.timeline_start
        t_en = t_st + t.duration
        if t_en <= st_bound + 0.05 or t_st >= en_bound - 0.05:
            continue
        ov_st = max(st_bound, t_st)
        ov_en = min(en_bound, t_en)
        t.timeline_start = round(max(0.0, ov_st - st_bound), 3)
        t.duration = round(max(0.5, min(seg_dur - t.timeline_start, ov_en - ov_st)), 2)
        new_titles.append(t)
    sub_tl.title_track = new_titles

    kept_ids = {c.id for c in sub_tl.visual_track}
    new_fx = []
    for fx in sub_tl.effects_track:
        if fx.target_clip_id in kept_ids:
            target_c = next((c for c in sub_tl.visual_track if c.id == fx.target_clip_id), None)
            if target_c is not None:
                fx.timeline_start = target_c.timeline_start
                fx.duration = target_c.duration
                new_fx.append(fx)
    sub_tl.effects_track = new_fx

    new_subs = []
    for s in sub_tl.subtitles.items:
        if s.end <= st_bound + 0.05 or s.start >= en_bound - 0.05:
            continue
        s.start = round(max(0.0, s.start - st_bound), 2)
        s.end = round(min(seg_dur, s.end - st_bound), 2)
        if s.end - s.start >= 0.2:
            new_subs.append(s)
    sub_tl.subtitles.items = new_subs

    for a in sub_tl.audio_tracks:
        a.source_in = round(a.source_in + st_bound, 2)
        a.timeline_start = 0.0
        a.duration = seg_dur
        a.fade_in = min(0.4, a.fade_in)
        a.fade_out = min(0.5, a.fade_out)

    return sub_tl


def render_timeline_segment(
    timeline: EditTimeline,
    output_mp4: Optional[str | Path] = None,
    start_sec: float = 0.0,
    end_sec: float = 8.0,
    fast_preview: bool = True,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    output_mp4_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Render a [start_sec, end_sec] time slice of an EditTimeline for rapid surgical previewing."""
    target_mp4 = Path(output_mp4_path or output_mp4 or "segment_preview.mp4")
    sub_tl = build_segment_subtimeline(
        timeline=timeline,
        start_sec=start_sec,
        end_sec=end_sec,
        fast_preview=fast_preview,
    )
    out = render_timeline(sub_tl, target_mp4, progress_callback=progress_callback)
    return {
        "video": str(out["video"]),
        "srt": str(out["srt"]),
        "ass": str(out["ass"]),
        "start_sec": round(float(start_sec), 2),
        "end_sec": round(float(end_sec), 2),
        "duration_sec": round(sub_tl.total_duration, 2),
        "fast_preview": fast_preview,
    }
