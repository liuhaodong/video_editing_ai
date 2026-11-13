"""Stage 55: Proxy Watchdown & Technical QC Agent (visual_qc)."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageOps

from ..analyzer.media_probe import probe_video_file
from ..models.schema import EditTimeline, VisualQCIssue, VisualQCReport
from ..renderer.titles import HDR_DIFFUSE_WHITE_PEAK_RGB, compute_overlay_bbox_norm, render_title_png
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


def compute_bbox_intersection_area(box_a: list[float], box_b: list[float]) -> float:
    """Return normalized 2D intersection area between [x1, y1, x2, y2] boxes."""
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ix1 = max(float(box_a[0]), float(box_b[0]))
    iy1 = max(float(box_a[1]), float(box_b[1]))
    ix2 = min(float(box_a[2]), float(box_b[2]))
    iy2 = min(float(box_a[3]), float(box_b[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return round((ix2 - ix1) * (iy2 - iy1), 5)


def _sample_proxy_luma_grid(img: Image.Image) -> tuple[float, int]:
    """Compute (mean_luma_0_255, clipped_blocks_count_out_of_64) on a 320x180 proxy image."""
    proxy = img.convert("RGB").resize((320, 180), Image.Resampling.BILINEAR)
    arr = np.asarray(proxy, dtype=np.float32)
    luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    mean_luma = float(np.mean(luma))
    # Reshape 180x320 into 8x8 blocks (each ~22x40)
    bh, bw = 180 // 8, 320 // 8
    clipped_blocks = 0
    for r in range(8):
        for c in range(8):
            blk = luma[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
            if float(np.mean(blk)) >= 253.5:
                clipped_blocks += 1
    return mean_luma, clipped_blocks


def _extract_video_proxy_frame(video_path: Path, timestamp_sec: float, out_jpg: Path) -> Optional[Image.Image]:
    """Extract a fast 320x180 single-frame JPEG proxy via FFmpeg."""
    if not video_path.exists():
        return None
    if not out_jpg.exists():
        out_jpg.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{max(0.0, timestamp_sec):.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                "scale=320:180",
                "-q:v",
                "5",
                str(out_jpg),
            ],
            capture_output=True,
            timeout=6,
            check=False,
        )
    if out_jpg.exists() and out_jpg.stat().st_size > 100:
        try:
            with Image.open(out_jpg) as im:
                return im.convert("RGB")
        except Exception:
            return None
    return None


def inspect_timeline_proxy_qc(
    timeline: EditTimeline,
    cache_dir: Optional[Path] = None,
) -> VisualQCReport:
    """Run deterministic low-res proxy watchdown QC across cut boundaries, V3 overlays, luma blocks, and audio cuts."""
    issues: list[VisualQCIssue] = []
    frames_sampled = 0
    black_flashes = 0
    frozen_frames = 0
    overlay_overlaps = 0
    clipped_blocks_flagged = 0
    audio_spikes = 0

    fps = max(1, int(timeline.settings.fps or 30))
    frame_dt = 1.0 / float(fps)
    is_hdr = bool(timeline.settings.hdr)
    proxy_dir = cache_dir / "qc_proxies" if cache_dir is not None else None

    # 1. Sample boundary & mid frames across Track V1 clips for black flashes, frozen tail overruns, and luma clipping
    probed_dur_cache: dict[str, float] = {}
    for idx, clip in enumerate(timeline.visual_track):
        src_p = Path(clip.source)
        if clip.type == "video" and src_p.exists():
            if clip.source not in probed_dur_cache:
                info = probe_video_file(src_p)
                probed_dur_cache[clip.source] = float(info.get("duration", 0.0) or 0.0)
            media_dur = probed_dur_cache[clip.source]

            # Timebase overrun check: source_out past actual media duration causes 1-2 frozen tail frames
            if media_dur > 0.2 and (clip.source_in + clip.duration) > (media_dur + 1.5 * frame_dt):
                frozen_frames += 1
                issues.append(
                    VisualQCIssue(
                        issue_type="frozen_frame",
                        severity="warning",
                        timeline_sec=round(clip.timeline_start + clip.duration, 3),
                        clip_id=clip.id,
                        chapter_id=clip.act,
                        details=(
                            f"Clip {clip.id} source_out ({clip.source_in + clip.duration:.3f}s) exceeds media duration "
                            f"({media_dur:.3f}s), risking frozen tail frames."
                        ),
                    )
                )

            if proxy_dir is not None and idx < 16:
                mid_t = clip.source_in + clip.duration * 0.5
                head_t = clip.source_in + frame_dt
                mid_img = _extract_video_proxy_frame(src_p, mid_t, proxy_dir / f"{clip.id}_mid.jpg")
                head_img = _extract_video_proxy_frame(src_p, head_t, proxy_dir / f"{clip.id}_head.jpg")
                if mid_img is not None:
                    frames_sampled += 1
                    mid_luma, mid_clip_cnt = _sample_proxy_luma_grid(mid_img)
                    if mid_clip_cnt >= 10:
                        clipped_blocks_flagged += 1
                        issues.append(
                            VisualQCIssue(
                                issue_type="clipped_luminance",
                                severity="warning",
                                timeline_sec=round(clip.timeline_start + clip.duration * 0.5, 3),
                                clip_id=clip.id,
                                chapter_id=clip.act,
                                details=f"Detected {mid_clip_cnt}/64 clipped luminance blocks in proxy sample of {clip.id}.",
                            )
                        )
                    if head_img is not None:
                        frames_sampled += 1
                        head_luma, _ = _sample_proxy_luma_grid(head_img)
                        if (
                            clip.transition_in.type in ("none", "cut")
                            and head_luma < 2.5
                            and mid_luma > 28.0
                        ):
                            black_flashes += 1
                            issues.append(
                                VisualQCIssue(
                                    issue_type="black_flash",
                                    severity="error",
                                    timeline_sec=round(clip.timeline_start, 3),
                                    clip_id=clip.id,
                                    chapter_id=clip.act,
                                    details=(
                                        f"1-frame black flash detected at cut entry of {clip.id} "
                                        f"(head_luma={head_luma:.1f} vs mid_luma={mid_luma:.1f})."
                                    ),
                                )
                            )
        elif clip.type == "photo" and src_p.exists() and idx < 16:
            try:
                with Image.open(src_p) as raw_im:
                    im = ImageOps.exif_transpose(raw_im).convert("RGB")
                    frames_sampled += 1
                    _, p_clip_cnt = _sample_proxy_luma_grid(im)
                    if p_clip_cnt >= 16 and not is_hdr:
                        clipped_blocks_flagged += 1
                        issues.append(
                            VisualQCIssue(
                                issue_type="clipped_luminance",
                                severity="warning",
                                timeline_sec=round(clip.timeline_start + clip.duration * 0.5, 3),
                                clip_id=clip.id,
                                chapter_id=clip.act,
                                details=f"Photo {clip.id} has {p_clip_cnt}/64 blown highlight blocks.",
                            )
                        )
            except Exception:
                pass

    # 2. Check 2D bounding-box overlap between bilingual subtitles and V3 overlays / photo plaques
    subtitle_bbox = [0.10, 0.85, 0.90, 0.96]
    for sub in timeline.subtitles.items:
        for ov in timeline.title_track:
            if ov.style == "act_banner":
                continue
            ov_start = ov.timeline_start
            ov_end = ov.timeline_start + ov.duration
            if sub.start < ov_end and sub.end > ov_start:
                ov_box = compute_overlay_bbox_norm(ov.style)
                inter = compute_bbox_intersection_area(subtitle_bbox, ov_box)
                if inter > 0.015:
                    overlay_overlaps += 1
                    issues.append(
                        VisualQCIssue(
                            issue_type="overlay_overlap",
                            severity="error",
                            timeline_sec=round(max(sub.start, ov_start), 3),
                            clip_id=ov.id,
                            details=(
                                f"Subtitle '{sub.id}' ({sub.start:.2f}-{sub.end:.2f}s) overlaps V3 '{ov.style}' "
                                f"overlay '{ov.id}' (intersection={inter:.3f})."
                            ),
                        )
                    )

    # Also verify HDR peak white clamping on V3 title overlays when HDR is active
    if is_hdr and proxy_dir is not None:
        for ov in timeline.title_track[:4]:
            tmp_png = proxy_dir / f"_qc_ov_{ov.id}.png"
            render_title_png(ov, tmp_png, width=320, height=180, hdr_safe=ov.hdr_safe)
            frames_sampled += 1
            with Image.open(tmp_png) as ov_im:
                arr = np.asarray(ov_im.convert("RGBA"), dtype=np.uint8)
                vis_mask = arr[:, :, 3] > 16
                if np.any(vis_mask):
                    peak_rgb = int(np.max(arr[:, :, :3][vis_mask]))
                    if peak_rgb > HDR_DIFFUSE_WHITE_PEAK_RGB + 8:
                        clipped_blocks_flagged += 1
                        issues.append(
                            VisualQCIssue(
                                issue_type="clipped_luminance",
                                severity="warning",
                                timeline_sec=ov.timeline_start,
                                clip_id=ov.id,
                                details=(
                                    f"V3 overlay '{ov.id}' peak RGB={peak_rgb} exceeds HDR diffuse-white ceiling "
                                    f"({HDR_DIFFUSE_WHITE_PEAK_RGB})."
                                ),
                            )
                        )

    # 3. Check abrupt audio amplitude spikes at cut points
    v1 = timeline.visual_track
    for idx in range(1, len(v1)):
        prev_c = v1[idx - 1]
        curr_c = v1[idx]
        if prev_c.type != "video" or curr_c.type != "video":
            continue
        v_prev = max(0.01, float(prev_c.volume))
        v_curr = max(0.01, float(curr_c.volume))
        db_jump = abs(20.0 * math.log10(v_curr / v_prev))
        cf_sec = max(float(prev_c.audio_crossfade_sec), float(curr_c.audio_crossfade_sec))
        if db_jump > 16.0 and cf_sec < 0.15:
            audio_spikes += 1
            issues.append(
                VisualQCIssue(
                    issue_type="audio_spike",
                    severity="error",
                    timeline_sec=round(curr_c.timeline_start, 3),
                    clip_id=curr_c.id,
                    chapter_id=curr_c.act,
                    details=(
                        f"Uncrossfaded {db_jump:.1f} dB amplitude jump between {prev_c.id} and {curr_c.id} "
                        f"(crossfade={cf_sec:.2f}s < 0.15s)."
                    ),
                )
            )

    error_count = sum(1 for i in issues if i.severity == "error")
    return VisualQCReport(
        project_title=timeline.project_name,
        passed_qc=(error_count == 0),
        proxy_frames_sampled=frames_sampled,
        black_flashes_detected=black_flashes,
        frozen_frames_detected=frozen_frames,
        overlay_overlaps_detected=overlay_overlaps,
        clipped_luminance_blocks_detected=clipped_blocks_flagged,
        audio_cut_spikes_detected=audio_spikes,
        issues=issues,
    )


@register_agent
class ProxyWatchdownQCAgent(BaseEditingAgent):
    """Stage 55 Agent: Samples low-res boundary proxy frames around cuts & V3 overlays to catch technical defects."""

    role_id = "visual_qc"
    display_name = "5b. Proxy Watchdown & Technical QC Inspector"
    stage_order = 55
    substage_priority = 0
    description = (
        "Runs between Stage 50 (showrunner) and Stage 60 (production_director), sampling low-resolution "
        "boundary frames around cuts and Track V3 overlays to detect black flashes, frozen tail frames, "
        "subtitle/lower-third bounding-box overlaps, clipped luminance blocks, and uncrossfaded audio cut spikes."
    )
    enabled_by_default = True
    requires_artifacts = ["pacing_audit.json", "timeline.yaml"]
    produces_artifacts = ["visual_qc_report.json"]
    skill_reference_file = "17_visual_qc_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline
        if timeline is None and (blackboard.project_dir / "timeline.yaml").exists():
            timeline = EditTimeline.from_yaml(blackboard.project_dir / "timeline.yaml")
            blackboard.timeline = timeline
        if timeline is None:
            raise RuntimeError("ProxyWatchdownQCAgent requires blackboard.timeline to be populated.")

        report = inspect_timeline_proxy_qc(
            timeline=timeline,
            cache_dir=blackboard.project_dir / "cache",
        )
        out_path = blackboard.project_dir / "visual_qc_report.json"
        report.to_json_file(out_path)
        blackboard.visual_qc_report = report

        # Emit targeted RevisionDirectives if any error-severity QC defect was detected
        if not report.passed_qc and blackboard.revision_round < blackboard.max_revisions:
            if report.overlay_overlaps_detected > 0:
                blackboard.add_revision_directive(
                    target_role_id="title_designer",
                    chapter_id=None,
                    reason=f"Resolve {report.overlay_overlaps_detected} subtitle/lower-third bounding-box overlaps",
                    constraints={"resolve_subtitle_overlaps": True},
                    issued_by=self.role_id,
                )
            if report.frozen_frames_detected > 0:
                blackboard.add_revision_directive(
                    target_role_id="lead_editor",
                    chapter_id=None,
                    reason=f"Clamp {report.frozen_frames_detected} clip out-points inside usable_out handles",
                    constraints={"clamp_usable_handles": True},
                    issued_by=self.role_id,
                )

        blackboard.emit_activity(
            self.role_id,
            f"Proxy Watchdown QC {'PASSED' if report.passed_qc else 'FLAGGED ISSUES'} ({report.proxy_frames_sampled} proxy frames sampled)",
            detail=(
                f"black_flashes={report.black_flashes_detected}, frozen={report.frozen_frames_detected}, "
                f"overlaps={report.overlay_overlaps_detected}, luma_clips={report.clipped_luminance_blocks_detected}, "
                f"audio_spikes={report.audio_cut_spikes_detected}"
            ),
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="production_director",
            message_type="audit_note",
            subject=f"Visual & Audio Proxy QC {'PASSED' if report.passed_qc else 'NEEDS REVIEW'}",
            payload=report.model_dump(mode="json"),
        )
        return {
            "passed_qc": report.passed_qc,
            "proxy_frames_sampled": report.proxy_frames_sampled,
            "overlay_overlap_count": report.overlay_overlaps_detected,
            "issues_count": len(report.issues),
            "visual_qc_report_path": str(out_path),
        }
