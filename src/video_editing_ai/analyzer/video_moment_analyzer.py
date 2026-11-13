"""Temporal Video Moment Analyzer: Dense Optical Flow Stability, Prosody Peaks & Multi-Moment Splitting."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Literal, Optional

import cv2
import numpy as np

from ..models.schema import DialogueSegment, VideoAsset
from .media_probe import compute_file_fingerprint


_OPERATOR_FILLER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(is\s+it|are\s+you|are\s+we)\s+(recording|filming|rolling|on)\b", re.IGNORECASE),
    re.compile(r"^\s*(wait(\s+wait|\s+a\s+sec|\s+a\s+second|\s+up)?|hold\s+on|hang\s+on)\s*[.!?]*\s*$", re.IGNORECASE),
    re.compile(r"\b(wait\s+is\s+it\s+recording|let\s+me\s+(record|film|zoom|focus|check)|press\s+record|hit\s+record|stop\s+recording)\b", re.IGNORECASE),
    re.compile(r"^\s*(one\s+two\s+three|1\s*2\s*3|testing(\s+one\s+two)?|okay\s+go|ready\s+set\s+go)\s*[.!?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(can\s+you\s+see\s+me|is\s+this\s+thing\s+on|oops|sorry\s+wrong\s+button)\s*[.!?]*\s*$", re.IGNORECASE),
)


def is_operator_filler_utterance(text: str) -> bool:
    """Return True if a spoken segment is logistical camera-operator chatter rather than story content."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return True
    words = re.findall(r"[a-zA-Z0-9']+|[\u4e00-\u9fff]", cleaned)
    if len(words) <= 12:
        for pat in _OPERATOR_FILLER_PATTERNS:
            if pat.search(cleaned):
                return True
    return False


def _decode_lowres_gray_frames(
    video_path: Path,
    sample_fps: float = 4.0,
    width: int = 160,
    height: int = 90,
    max_duration: float = 90.0,
) -> np.ndarray:
    """Decode low-resolution grayscale frames via FFmpeg pipe for fast optical-flow & blur telemetry."""
    if not video_path.exists():
        return np.zeros((0, height, width), dtype=np.uint8)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-t",
        f"{max_duration:.2f}",
        "-i",
        str(video_path),
        "-vf",
        f"fps={sample_fps:.2f},scale={width}:{height}:flags=fast_bilinear",
        "-pix_fmt",
        "gray",
        "-vcodec",
        "rawvideo",
        "-an",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    frame_bytes = width * height
    if proc.returncode != 0 or len(proc.stdout) < frame_bytes:
        return np.zeros((0, height, width), dtype=np.uint8)
    n_frames = len(proc.stdout) // frame_bytes
    raw = np.frombuffer(proc.stdout[: n_frames * frame_bytes], dtype=np.uint8)
    return raw.reshape((n_frames, height, width))


def _decode_prosody_excitement_curve(
    video_path: Path,
    n_steps: int,
    step_sec: float = 0.25,
    max_duration: float = 90.0,
) -> np.ndarray:
    """Compute per-step acoustic laughter/cheer/vocal-excitement prosody curve in [0.0, 1.0]."""
    out = np.full(max(1, n_steps), 0.25, dtype=np.float32)
    if not video_path.exists() or n_steps <= 0:
        return out

    sr = 8000
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-t",
        f"{max_duration:.2f}",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    hop = int(round(sr * step_sec))
    if proc.returncode != 0 or len(proc.stdout) < hop * 2:
        return out

    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    rms_vals: list[float] = []
    bright_vals: list[float] = []
    mod_vals: list[float] = []

    for i in range(n_steps):
        s0 = i * hop
        s1 = min(len(pcm), s0 + hop)
        if s1 - s0 < 200:
            rms_vals.append(0.0)
            bright_vals.append(0.0)
            mod_vals.append(0.0)
            continue
        seg = pcm[s0:s1]
        rms = float(math.sqrt(float(np.mean(seg * seg)) + 1e-12))
        rms_vals.append(rms)

        # Sub-envelope modulation (laughter/cheering exhibits rhythmic 4-8 Hz bursts)
        sub_hop = max(40, len(seg) // 8)
        sub_n = len(seg) // sub_hop
        if sub_n >= 4:
            sub_f = seg[: sub_n * sub_hop].reshape(sub_n, sub_hop)
            sub_rms = np.sqrt(np.mean(sub_f * sub_f, axis=1) + 1e-12)
            mod_depth = float(np.std(sub_rms) / (float(np.mean(sub_rms)) + 1e-5))
        else:
            mod_depth = 0.0
        mod_vals.append(min(1.0, mod_depth))

        spec = np.abs(np.fft.rfft(seg)) + 1e-9
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
        tot_spec = float(np.sum(spec)) + 1e-9
        vocal_excitement_band = float(np.sum(spec[(freqs >= 650.0) & (freqs <= 3000.0)])) / tot_spec
        bright_vals.append(vocal_excitement_band)

    rms_arr = np.array(rms_vals, dtype=np.float32)
    p85 = float(np.percentile(rms_arr, 85)) + 1e-5
    norm_rms = np.clip(rms_arr / max(0.02, p85), 0.0, 1.0)
    mod_arr = np.array(mod_vals, dtype=np.float32)
    bright_arr = np.array(bright_vals, dtype=np.float32)

    # Penalize digital clipping (harsh wind/mic bump where peak >= 0.99 and low modulation)
    excitement = 0.48 * norm_rms + 0.30 * np.clip(mod_arr * 1.35, 0.0, 1.0) + 0.22 * np.clip(bright_arr * 1.6, 0.0, 1.0)
    return np.clip(excitement, 0.05, 1.0).astype(np.float32)


def analyze_video_temporal_telemetry(
    video_path: Path,
    duration: float,
    cache_dir: Optional[Path] = None,
    sample_fps: float = 4.0,
) -> dict[str, Any]:
    """
    Compute dense optical-flow camera stability, sharpness, camera motion classification,
    usable uncorrupted margins [clean_usable_in, clean_usable_out], and acoustic prosody curve.
    Cached on disk keyed by file fingerprint.
    """
    step_sec = 1.0 / sample_fps
    n_expected = max(1, int(math.ceil(max(0.5, duration) * sample_fps)))

    cache_file: Optional[Path] = None
    if cache_dir is not None and video_path.exists():
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            _, _, fhash = compute_file_fingerprint(video_path)
            cache_file = cache_dir / f"vid_moments_{fhash}.json"
            if cache_file.exists():
                return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cache_file = None

    frames = _decode_lowres_gray_frames(video_path, sample_fps=sample_fps, width=160, height=90)
    n_steps = len(frames) if len(frames) > 0 else n_expected

    stability_curve = np.full(n_steps, 0.88, dtype=np.float32)
    sharpness_curve = np.full(n_steps, 0.75, dtype=np.float32)
    vx_curve = np.zeros(n_steps, dtype=np.float32)
    vy_curve = np.zeros(n_steps, dtype=np.float32)
    div_curve = np.zeros(n_steps, dtype=np.float32)

    if len(frames) >= 2:
        raw_sharp = np.zeros(n_steps, dtype=np.float32)
        for i in range(n_steps):
            lap = cv2.Laplacian(frames[i], cv2.CV_32F)
            raw_sharp[i] = float(np.percentile(np.abs(lap), 85))
        med_sharp = max(8.0, float(np.median(raw_sharp)))
        sharpness_curve = np.clip(raw_sharp / (med_sharp * 1.25), 0.10, 1.0).astype(np.float32)

        mag_means = np.zeros(n_steps, dtype=np.float32)
        coherence = np.ones(n_steps, dtype=np.float32)
        frame_diffs = np.zeros(n_steps, dtype=np.float32)

        # Grid coordinates centered at (0, 0) for radial zoom/walk-forward divergence
        yy, xx = np.mgrid[0:90, 0:160].astype(np.float32)
        rx = (xx - 80.0) / 80.0
        ry = (yy - 45.0) / 45.0
        r_norm = np.hypot(rx, ry) + 1e-4
        rx /= r_norm
        ry /= r_norm

        for i in range(1, n_steps):
            flow = cv2.calcOpticalFlowFarneback(
                frames[i - 1],
                frames[i],
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            u = flow[:, :, 0]
            v = flow[:, :, 1]
            u_mean = float(np.mean(u))
            v_mean = float(np.mean(v))
            vx_curve[i] = u_mean
            vy_curve[i] = v_mean

            pix_mag = np.hypot(u, v)
            mean_mag = float(np.mean(pix_mag))
            vec_mag = math.hypot(u_mean, v_mean)
            mag_means[i] = mean_mag
            coherence[i] = float(vec_mag / (mean_mag + 0.35))
            div_curve[i] = float(np.mean((u - u_mean) * rx + (v - v_mean) * ry))
            frame_diffs[i] = float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))))

        vx_curve[0] = vx_curve[1]
        vy_curve[0] = vy_curve[1]
        mag_means[0] = mag_means[1]
        frame_diffs[0] = frame_diffs[1]

        # Frame-to-frame velocity jerk ||v_t - v_{t-1}|| detects erratic handshake & camera-drop
        jerk = np.zeros(n_steps, dtype=np.float32)
        for i in range(1, n_steps):
            jerk[i] = math.hypot(float(vx_curve[i] - vx_curve[i - 1]), float(vy_curve[i] - vy_curve[i - 1]))
        jerk[0] = jerk[1]

        for i in range(n_steps):
            m_val = float(mag_means[i])
            j_val = float(jerk[i])
            s_val = float(sharpness_curve[i])
            f_diff = float(frame_diffs[i])
            coh = float(coherence[i])

            # Smooth tripod/gimbal pans have low jerk (< 2.8), high coherence (> 0.60), and crisp edges.
            # Violent whip-pans, camera drops, or blurred motion shake exhibit high jerk, large frame_diff,
            # incoherent flow, or sharp drop in Laplacian edge contrast.
            whip_pen = max(0.0, (m_val - 9.0) / 9.5) + max(0.0, (f_diff - 24.0) / 28.0)
            jerk_pen = max(0.0, (j_val - 2.1) / 4.8)
            incoh_pen = max(0.0, (0.55 - coh) * 0.85) if (m_val > 2.2 or f_diff > 18.0) else 0.0
            blur_pen = max(0.0, (0.52 - s_val) * 1.45)
            stab = 1.0 - min(0.95, 0.50 * jerk_pen + 0.45 * whip_pen + incoh_pen + blur_pen)
            stability_curve[i] = round(float(np.clip(stab, 0.05, 1.0)), 3)

    prosody_curve = _decode_prosody_excitement_curve(video_path, n_steps=n_steps, step_sec=step_sec)

    # 3-sample median smoothing prevents a single reversal frame inside a violent drop from halting trim
    smooth_stab = np.array(
        [
            float(np.median(stability_curve[max(0, k - 1) : min(n_steps, k + 2)]))
            for k in range(n_steps)
        ],
        dtype=np.float32,
    )

    # Trim corrupted head & tail camera-drop / whip-pan margins (smoothed stability < 0.48)
    max_head_trim_steps = int(min(2.5, duration * 0.28) / step_sec)
    max_tail_trim_steps = int(min(4.0, duration * 0.38) / step_sec)

    head_idx = 0
    while head_idx < min(n_steps - 2, max_head_trim_steps) and smooth_stab[head_idx] < 0.48:
        head_idx += 1

    tail_idx = n_steps
    while (n_steps - tail_idx) < max_tail_trim_steps and tail_idx > head_idx + 2 and smooth_stab[tail_idx - 1] < 0.48:
        tail_idx -= 1

    clean_usable_in = round(min(max(0.0, duration - 0.6), head_idx * step_sec), 2)
    clean_usable_out = round(max(clean_usable_in + 0.6, min(duration, tail_idx * step_sec)), 2)
    if tail_idx == n_steps:
        clean_usable_out = round(max(0.6, duration), 2)

    # Classify dominant camera movement on the clean span
    sl = slice(head_idx, max(head_idx + 1, tail_idx))
    mean_u = float(np.mean(vx_curve[sl])) if n_steps > 0 else 0.0
    mean_v = float(np.mean(vy_curve[sl])) if n_steps > 0 else 0.0
    mean_div = float(np.mean(div_curve[sl])) if n_steps > 0 else 0.0

    # Optical flow (u, v) is scene pixel displacement: camera panning right shifts scene pixels LEFT (u < 0).
    cam_motion: Literal["static", "pan_left", "pan_right", "tilt_up", "walk_forward"] = "static"
    if abs(mean_u) < 0.65 and abs(mean_v) < 0.65 and abs(mean_div) < 0.30:
        cam_motion = "static"
    elif mean_div > 0.42 and abs(mean_u) < 1.6:
        cam_motion = "walk_forward"
    elif abs(mean_v) > abs(mean_u) * 1.25 and mean_v > 0.75:
        cam_motion = "tilt_up"
    elif mean_u < -0.70:
        cam_motion = "pan_right"
    elif mean_u > 0.70:
        cam_motion = "pan_left"
    elif abs(mean_u) + abs(mean_v) >= 0.75:
        cam_motion = "walk_forward"

    salience_curve = (
        0.36 * stability_curve
        + 0.32 * sharpness_curve
        + 0.32 * prosody_curve
    ).astype(np.float32)

    doc = {
        "step_sec": round(step_sec, 4),
        "n_steps": int(n_steps),
        "clean_usable_in": clean_usable_in,
        "clean_usable_out": clean_usable_out,
        "camera_movement": cam_motion,
        "mean_stability": round(float(np.mean(stability_curve[sl])), 3),
        "stability_curve": [round(float(x), 3) for x in stability_curve],
        "sharpness_curve": [round(float(x), 3) for x in sharpness_curve],
        "prosody_curve": [round(float(x), 3) for x in prosody_curve],
        "salience_curve": [round(float(x), 3) for x in salience_curve],
    }
    if cache_file is not None:
        try:
            cache_file.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        except Exception:
            pass
    return doc


def _window_stats(
    telemetry: dict[str, Any],
    in_tc: float,
    out_tc: float,
) -> tuple[float, float, float, float]:
    """Return (mean_stability, max_prosody, mean_salience, peak_tc) inside [in_tc, out_tc]."""
    step = float(telemetry.get("step_sec") or 0.25)
    stab = telemetry.get("stability_curve") or [0.85]
    pros = telemetry.get("prosody_curve") or [0.3]
    sal = telemetry.get("salience_curve") or [0.75]
    n = len(sal)
    i0 = max(0, min(n - 1, int(math.floor(in_tc / step))))
    i1 = max(i0 + 1, min(n, int(math.ceil(out_tc / step))))

    sub_stab = stab[i0:i1] or [0.85]
    sub_pros = pros[i0:i1] or [0.3]
    sub_sal = sal[i0:i1] or [0.75]
    best_rel = int(np.argmax(sub_sal))
    peak_tc = round(min(out_tc, max(in_tc, (i0 + best_rel + 0.5) * step)), 2)
    return (
        round(float(np.mean(sub_stab)), 3),
        round(float(np.max(sub_pros)), 3),
        round(float(np.mean(sub_sal)), 3),
        peak_tc,
    )


def _best_broll_window_in_span(
    telemetry: dict[str, Any],
    span_start: float,
    span_end: float,
    golden_cap: float,
) -> Optional[tuple[float, float, float, float, float, float]]:
    """
    Find the highest-salience stable B-roll window [golden_in, golden_out, usable_in, usable_out, stab, peak_tc]
    inside [span_start, span_end].
    """
    avail = span_end - span_start
    if avail < 0.6:
        return None

    step = float(telemetry.get("step_sec") or 0.25)
    stab = np.asarray(telemetry.get("stability_curve") or [0.85], dtype=np.float32)
    sal = np.asarray(telemetry.get("salience_curve") or [0.75], dtype=np.float32)
    n = len(sal)

    # Trim unstable edges inside this span if span is long enough
    i_lo = max(0, min(n - 1, int(math.floor(span_start / step))))
    i_hi = max(i_lo + 1, min(n, int(math.ceil(span_end / step))))
    while i_hi - i_lo > 8 and stab[i_lo] < 0.45:
        i_lo += 1
    while i_hi - i_lo > 8 and stab[i_hi - 1] < 0.45:
        i_hi -= 1

    u_in = round(max(span_start, i_lo * step), 2)
    u_out = round(min(span_end, i_hi * step), 2)
    if u_out - u_in < min(avail, 1.8):
        u_in = round(span_start, 2)
        u_out = round(span_end, 2)

    u_span = max(0.5, u_out - u_in)
    if u_span >= 2.7:
        target_dur = min(golden_cap, max(2.0, u_span - 0.75))
    else:
        target_dur = u_span

    win_steps = max(1, int(round(target_dur / step)))
    best_score = -1e9
    best_in = u_in
    s_start = max(0, min(n - 1, int(math.floor(u_in / step))))
    s_end = max(s_start + 1, min(n, int(math.ceil(u_out / step))))

    if s_end - s_start > win_steps:
        # Prefer leaving ~0.35s head & tail margin when quality is similar
        for idx in range(s_start, s_end - win_steps + 1):
            cand_in = idx * step
            cand_out = cand_in + target_dur
            margin_bonus = 0.04 if (cand_in - u_in >= 0.30 and u_out - cand_out >= 0.30) else 0.0
            w_stab = float(np.mean(stab[idx : idx + win_steps]))
            if w_stab < 0.40:
                continue
            w_sal = float(np.mean(sal[idx : idx + win_steps])) + margin_bonus
            if w_sal > best_score:
                best_score = w_sal
                best_in = round(max(u_in, min(u_out - target_dur, cand_in)), 2)
        if best_score < -1e8:
            spare = max(0.0, u_span - target_dur)
            best_in = round(u_in + spare * 0.45, 2)
    else:
        spare = max(0.0, u_span - target_dur)
        best_in = round(u_in + spare * 0.45, 2)

    best_out = round(min(u_out, best_in + target_dur), 2)
    if best_out <= best_in:
        best_in = u_in
        best_out = u_out

    m_stab, _, _, peak_tc = _window_stats(telemetry, best_in, best_out)
    return best_in, best_out, u_in, u_out, m_stab, peak_tc


def extract_candidate_video_moments(
    vid: VideoAsset,
    telemetry: dict[str, Any],
    max_broll_sec: float = 4.5,
) -> list[dict[str, Any]]:
    """
    Extract 1 or more golden-window moments from a VideoAsset:
    - Filters out camera-operator filler utterances ('wait is it recording', etc.).
    - Groups coherent spoken dialogue turns (merging pauses < 1.2s, splitting long pauses >= 1.2s).
    - For longer videos (>= 8.5s), also rescues distinct stable B-roll moments outside speech/filler spans.
    """
    clean_u_in = float(telemetry.get("clean_usable_in", 0.0))
    clean_u_out = float(telemetry.get("clean_usable_out", max(0.6, vid.duration)))
    golden_cap = min(4.5, max(2.0, float(max_broll_sec)))

    clean_dialogues: list[DialogueSegment] = []
    filler_spans: list[tuple[float, float]] = []
    for d in sorted(vid.dialogues, key=lambda x: x.start_time):
        if is_operator_filler_utterance(d.text):
            filler_spans.append((max(0.0, d.start_time - 0.15), min(vid.duration, d.end_time + 0.25)))
        else:
            clean_dialogues.append(d)

    # Group clean dialogues into coherent quote blocks
    quote_groups: list[list[DialogueSegment]] = []
    for seg in clean_dialogues:
        if not quote_groups:
            quote_groups.append([seg])
            continue
        prev_grp = quote_groups[-1]
        gap = seg.start_time - prev_grp[-1].end_time
        merged_span = seg.end_time - prev_grp[0].start_time
        # Keep short videos (< 8.5s) as a single A-roll unit; split longer videos on natural pauses >= 1.25s
        if vid.duration < 8.5 or (gap < 1.25 and merged_span <= 10.5):
            prev_grp.append(seg)
        else:
            quote_groups.append([seg])

    moments: list[dict[str, Any]] = []
    occupied_spans: list[tuple[float, float]] = list(filler_spans)

    # 1. Emit A-Roll spoken moments
    for g_idx, grp in enumerate(quote_groups):
        first_start = grp[0].start_time
        last_end = grp[-1].end_time
        u_lo = clean_u_in if g_idx == 0 else max(clean_u_in, round(first_start - 0.8, 2))
        u_hi = clean_u_out if g_idx == len(quote_groups) - 1 else min(clean_u_out, round(last_end + 0.8, 2))
        u_lo = min(u_lo, max(0.0, first_start))
        u_hi = max(u_hi, min(vid.duration, last_end))

        g_in = round(max(u_lo, first_start - 0.25), 2)
        g_out = round(min(u_hi, max(last_end + 0.35, min(u_hi, g_in + 2.0))), 2)
        if g_out <= g_in:
            g_in = round(max(0.0, first_start), 2)
            g_out = round(max(g_in + 0.5, min(vid.duration, last_end + 0.25)), 2)
            u_lo = min(u_lo, g_in)
            u_hi = max(u_hi, g_out)

        m_stab, m_pros, m_sal, peak_tc = _window_stats(telemetry, g_in, g_out)
        occupied_spans.append((max(0.0, g_in - 0.35), min(vid.duration, g_out + 0.35)))
        moments.append(
            {
                "roll_type": "a_roll",
                "in_tc": g_in,
                "out_tc": g_out,
                "golden_in": g_in,
                "golden_out": g_out,
                "usable_in": round(u_lo, 2),
                "usable_out": round(max(g_out, u_hi), 2),
                "dialogues": grp,
                "stability_score": m_stab,
                "prosody_excitement_score": m_pros,
                "salience_score": m_sal,
                "action_peak_tc": peak_tc,
            }
        )

    # 2. Extract B-Roll Golden Windows
    # - Always extract when there are no clean A-Roll moments
    # - Also extract for longer videos (>= 8.5s) from unclaimed stable intervals >= 2.6s!
    if not moments or vid.duration >= 8.5:
        occupied_spans.sort(key=lambda iv: iv[0])
        free_spans: list[tuple[float, float]] = []
        cursor = clean_u_in
        for occ_s, occ_e in occupied_spans:
            if occ_s - cursor >= (2.6 if moments else 0.8):
                free_spans.append((cursor, min(clean_u_out, occ_s)))
            cursor = max(cursor, occ_e)
        if clean_u_out - cursor >= (2.6 if moments else 0.8):
            free_spans.append((cursor, clean_u_out))

        if not moments and not free_spans:
            # Fallback if whole short video was flagged or tight: use best clean window
            free_spans = [(clean_u_in, max(clean_u_in + 0.6, clean_u_out))]

        broll_candidates: list[dict[str, Any]] = []
        for sp_s, sp_e in free_spans:
            # Allow splitting a very long pure B-roll span (>= 11.0s) into up to 2 distinct highlights
            sub_intervals = [(sp_s, sp_e)]
            if not moments and (sp_e - sp_s) >= 11.0:
                mid = round(0.5 * (sp_s + sp_e), 2)
                sub_intervals = [(sp_s, mid), (mid, sp_e)]

            for sub_s, sub_e in sub_intervals:
                picked = _best_broll_window_in_span(telemetry, sub_s, sub_e, golden_cap=golden_cap)
                if picked is None:
                    continue
                b_in, b_out, bu_in, bu_out, b_stab, b_peak = picked
                if moments and b_stab < 0.50:
                    continue
                _, b_pros, b_sal, _ = _window_stats(telemetry, b_in, b_out)
                broll_candidates.append(
                    {
                        "roll_type": "b_roll",
                        "in_tc": b_in,
                        "out_tc": b_out,
                        "golden_in": b_in,
                        "golden_out": b_out,
                        "usable_in": bu_in,
                        "usable_out": bu_out,
                        "dialogues": [],
                        "stability_score": b_stab,
                        "prosody_excitement_score": b_pros,
                        "salience_score": b_sal,
                        "action_peak_tc": b_peak,
                    }
                )

        broll_candidates.sort(key=lambda m: m["salience_score"], reverse=True)
        max_brolls = 2 if moments else 2
        chosen_brolls = sorted(broll_candidates[:max_brolls], key=lambda m: m["in_tc"])
        moments.extend(chosen_brolls)

    moments.sort(key=lambda m: m["in_tc"])
    return moments
