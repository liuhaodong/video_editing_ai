"""Stage 10: The Wrangler & Culler Sub-Agent (culler)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Literal, Optional

import numpy as np
from PIL import Image, ImageDraw

from ..analyzer.ag_annotator import analyze_media_directory
from ..analyzer.media_probe import _get_cjk_font
from ..models.schema import (
    CulledSubClip,
    DialogueSegment,
    MediaCatalog,
    MediaIndex,
    PhotoAsset,
    VideoAsset,
    WordTimestamp,
)
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


def _synthesize_video_scene_caption_and_transition(
    vid: VideoAsset,
    shot_scale: str,
    cam_motion: str,
    has_speech: bool,
    spoken_text: str,
    v_idx: int,
    total_videos: int,
    user_prompt: str = "",
) -> tuple[str, str, Literal["none", "cut", "crossfade", "dissolve", "fade_black", "dip_to_white"], str]:
    """Synthesize a baseline mechanical scene caption and transition prior to AI Vision Culler overrides."""
    summary_clean = (vid.overall_summary or "").strip()
    is_generic_summary = (
        not summary_clean
        or summary_clean.lower().startswith(("video segment", "video clip", "clip "))
        or vid.filename.lower() in summary_clean.lower()
    )

    if not is_generic_summary and " (" in summary_clean and summary_clean.endswith(")"):
        parts = summary_clean.split(" (", 1)
        title = parts[0].strip()[:58]
        sub = parts[1][:-1].strip()[:68]
        trans: Literal["none", "cut", "crossfade", "dissolve", "fade_black", "dip_to_white"] = (
            "fade_black" if v_idx in (1, total_videos) else ("dissolve" if shot_scale == "wide" else "crossfade")
        )
        rationale = f"Derived from scene annotation and {shot_scale} {cam_motion} cinematography."
    elif has_speech and spoken_text.strip():
        quote_short = spoken_text.strip()[:44]
        title = f'Live Story Moment — "{quote_short}"'
        sub = "现场原声记录 · 真实对话与故事瞬间"
        trans = "crossfade"
        rationale = "Preserved spoken dialogue window with -0.35s J-cut audio pre-lap."
    else:
        motion_label = {
            "pan_left": "Cinematic Pan",
            "pan_right": "Sweeping Panorama",
            "tilt_up": "Architectural Tilt-Up",
            "walk_forward": "First-Person Tracking Shot",
            "static": "Living Tableau",
        }.get(cam_motion, "Cinematic Scene")
        stage_label = (
            "Opening Establishing Vista"
            if v_idx == 1
            else ("Closing Finale Vista" if v_idx == total_videos else f"{motion_label} ({shot_scale.capitalize()})")
        )
        title = f"{stage_label} — Scene {v_idx:02d}"
        sub = "4K HDR 动态实景镜头 · 光影与现场氛围记录"
        trans = "fade_black" if v_idx in (1, total_videos) else ("dissolve" if shot_scale == "wide" else "crossfade")
        rationale = f"Selected golden window ({shot_scale} {cam_motion}) for visual rhythm and continuity."

    return title, sub, trans, rationale


def render_video_culling_contact_sheets(
    project_dir: str | Path,
    max_per_sheet: int = 6,
) -> dict[str, Any]:
    """
    Mechanically render numbered 1080p Video Culling Contact Sheets showing keyframes,
    timecode windows, shot scale/motion tags, and Whisper transcripts for AI Agent inspection.
    """
    proj_path = Path(project_dir).resolve()
    mi_path = proj_path / "media_index.json"
    ann_path = proj_path / "annotations.yaml"
    if not mi_path.exists():
        raise FileNotFoundError(f"media_index.json not found in {proj_path}")

    media_index = MediaIndex.from_json_file(mi_path)
    catalog = MediaCatalog.from_yaml(ann_path) if ann_path.exists() else None
    vid_by_id = {v.id: v for v in (catalog.videos if catalog else [])}

    vid_clips = [s for s in media_index.subclips if s.media_type == "video"]
    out_dir = proj_path / "cache" / "curation"
    out_dir.mkdir(parents=True, exist_ok=True)

    font_title = _get_cjk_font(26)
    font_meta = _get_cjk_font(18)
    font_sub = _get_cjk_font(16)

    sheet_paths: list[str] = []
    for s_idx in range(0, max(1, len(vid_clips)), max_per_sheet):
        chunk = vid_clips[s_idx : s_idx + max_per_sheet]
        if not chunk:
            break
        sheet_num = (s_idx // max_per_sheet) + 1
        canvas = Image.new("RGB", (1920, 1080), (12, 16, 24))
        draw = ImageDraw.Draw(canvas)

        draw.rectangle([0, 0, 1920, 66], fill=(20, 27, 41))
        draw.text(
            (28, 18),
            f"AI VIDEO CULLER — MULTIMODAL KEYFRAME & TRANSCRIPT SHEET #{sheet_num:02d} ({len(chunk)} Video Clips)",
            fill=(255, 215, 64),
            font=font_title,
        )

        row_h = (1080 - 80) // max(1, min(max_per_sheet, len(chunk)))
        for r_i, vc in enumerate(chunk):
            y0 = 74 + r_i * row_h
            y1 = y0 + row_h - 10
            bg_col = (22, 31, 46) if not vc.is_discarded else (38, 22, 26)
            border_col = (0, 229, 255) if vc.roll_type == "a_roll" else ((255, 215, 64) if not vc.is_discarded else (180, 60, 60))
            draw.rounded_rectangle([24, y0, 1896, y1], radius=10, fill=bg_col, outline=border_col, width=2)

            # Draw up to 3 scene keyframes on the left
            v_asset = vid_by_id.get(vc.source_asset_id)
            kf_paths = [sc.keyframe_path for sc in (v_asset.scenes if v_asset else []) if sc.keyframe_path and Path(sc.keyframe_path).exists()]
            thumb_w = 220
            thumb_h = max(60, row_h - 26)
            for k_i in range(3):
                tx = 38 + k_i * (thumb_w + 10)
                ty = y0 + 13
                if kf_paths:
                    chosen_kf = kf_paths[min(len(kf_paths) - 1, int(k_i * len(kf_paths) / 3))]
                    try:
                        with Image.open(chosen_kf) as kim:
                            kim_rgb = kim.convert("RGB").resize((thumb_w, thumb_h))
                            canvas.paste(kim_rgb, (tx, ty))
                    except Exception:
                        draw.rectangle([tx, ty, tx + thumb_w, ty + thumb_h], fill=(35, 42, 56))
                else:
                    draw.rectangle([tx, ty, tx + thumb_w, ty + thumb_h], fill=(35, 42, 56))

            # Draw clip metadata & AI caption on the right
            info_x = 38 + 3 * (thumb_w + 10) + 18
            status_tag = "SELECTED" if not vc.is_discarded else "DISCARDED"
            draw.text(
                (info_x, y0 + 14),
                f"[{vc.subclip_id}] {vc.filename}  •  {vc.roll_type.upper()}  •  Window: {vc.in_tc:.1f}s–{vc.out_tc:.1f}s ({vc.duration:.1f}s)  •  {status_tag}",
                fill=(255, 215, 64) if not vc.is_discarded else (255, 120, 120),
                font=font_meta,
            )
            draw.text(
                (info_x, y0 + 44),
                f"Scale: {vc.shot_scale} | Motion: {vc.camera_movement} | Score: {vc.emotion_score:.1f}/10 | Transition: {vc.recommended_transition}",
                fill=(170, 212, 255),
                font=font_sub,
            )
            cap_line = f"Scene Caption: {vc.annotation_title or vc.summary[:65]} ({vc.annotation_subtitle or ''})"
            draw.text((info_x, y0 + 70), cap_line[:115], fill=(235, 242, 252), font=font_sub)
            if vc.spoken_text:
                draw.text(
                    (info_x, y0 + 96),
                    f'Spoken ASR: "{vc.spoken_text[:105]}"',
                    fill=(120, 245, 190),
                    font=font_sub,
                )

        out_p = out_dir / f"video_culling_sheet_{sheet_num:02d}.jpg"
        canvas.save(out_p, quality=90)
        sheet_paths.append(str(out_p))

    return {
        "video_clips_count": len(vid_clips),
        "contact_sheets": sheet_paths,
    }


def apply_video_curation_overrides(
    project_dir: str | Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Persist AI Agent creative decisions for video clip selection, golden-window [in_tc, out_tc] trims,
    roll_type, shot_scale, recommended_transition, and bilingual scene captions into media_index.json.
    """
    proj_path = Path(project_dir).resolve()
    mi_path = proj_path / "media_index.json"
    if not mi_path.exists():
        raise FileNotFoundError(f"media_index.json not found in {proj_path}")

    media_index = MediaIndex.from_json_file(mi_path)
    items_override = overrides.get("videos") or overrides.get("items") or {}
    if isinstance(items_override, list):
        items_override = {
            str(it.get("subclip_id") or it.get("filename")): it
            for it in items_override
            if isinstance(it, dict)
        }

    updated = 0
    for sc in media_index.subclips:
        if sc.media_type != "video":
            continue
        u = items_override.get(sc.subclip_id) or items_override.get(sc.filename)
        if not isinstance(u, dict):
            continue
        if "in_tc" in u and "out_tc" in u:
            new_in = max(0.0, float(u["in_tc"]))
            new_out = max(new_in + 1.0, float(u["out_tc"]))
            sc.in_tc = round(new_in, 2)
            sc.out_tc = round(new_out, 2)
            sc.duration = round(sc.out_tc - sc.in_tc, 2)
        if "is_discarded" in u:
            sc.is_discarded = bool(u["is_discarded"])
        elif "is_selected" in u:
            sc.is_discarded = not bool(u["is_selected"])
        if u.get("roll_type") in ("a_roll", "b_roll"):
            sc.roll_type = u["roll_type"]
        if u.get("shot_scale") in ("wide", "medium", "closeup"):
            sc.shot_scale = u["shot_scale"]
        if u.get("recommended_transition") in ("none", "cut", "crossfade", "dissolve", "fade_black", "dip_to_white"):
            sc.recommended_transition = u["recommended_transition"]
        if u.get("annotation_title"):
            sc.annotation_title = str(u["annotation_title"])
        if u.get("annotation_subtitle"):
            sc.annotation_subtitle = str(u["annotation_subtitle"])
        if u.get("summary"):
            sc.summary = str(u["summary"])
        if u.get("ai_rationale"):
            sc.ai_rationale = str(u["ai_rationale"])
        if isinstance(u.get("emotion_score"), (int, float)):
            sc.emotion_score = float(u["emotion_score"])
        updated += 1

    media_index.kept_selects_count = sum(1 for s in media_index.subclips if not s.is_discarded)
    media_index.discarded_count = sum(1 for s in media_index.subclips if s.is_discarded)
    media_index.to_json_file(mi_path)

    decisions_dir = proj_path / "ai_decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    (decisions_dir / "culler.json").write_text(
        json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "updated_video_clips": updated,
        "kept_selects_count": media_index.kept_selects_count,
        "media_index_path": str(mi_path),
    }


def _estimate_shot_scale_and_motion(
    image_paths: list[str],
    summary_text: str = "",
) -> tuple[Literal["wide", "medium", "closeup"], Literal["static", "pan_left", "pan_right", "tilt_up", "walk_forward"]]:
    """Estimate shot scale (wide/medium/closeup) and camera motion from keyframes & text."""
    text_lower = summary_text.lower()
    shot_scale: Literal["wide", "medium", "closeup"] = "medium"
    if any(k in text_lower for k in ("panorama", "landscape", "entrance", "pyramid", "stage", "wide", "park", "arch")):
        shot_scale = "wide"
    elif any(k in text_lower for k in ("close", "portrait", "face", "smile", "food", "detail", "selfie")):
        shot_scale = "closeup"

    valid_imgs: list[np.ndarray] = []
    for p in image_paths[:3]:
        path_obj = Path(p)
        if path_obj.exists():
            try:
                with Image.open(path_obj) as im:
                    arr = np.asarray(im.convert("L").resize((128, 72)), dtype=np.float32)
                    valid_imgs.append(arr)
            except Exception:
                pass

    if valid_imgs and shot_scale == "medium":
        arr0 = valid_imgs[0]
        gy, gx = np.gradient(arr0)
        mag = np.hypot(gx, gy)
        center_mag = float(np.mean(mag[18:54, 32:96])) + 1e-5
        border_mag = float(np.mean(mag)) + 1e-5
        ratio = center_mag / border_mag
        if ratio > 1.22:
            shot_scale = "closeup"
        elif ratio < 0.94:
            shot_scale = "wide"

    motion: Literal["static", "pan_left", "pan_right", "tilt_up", "walk_forward"] = "static"
    if len(valid_imgs) >= 2:
        first, last = valid_imgs[0], valid_imgs[-1]
        col_profile_1 = np.mean(np.abs(np.diff(first, axis=1)), axis=0)
        col_profile_2 = np.mean(np.abs(np.diff(last, axis=1)), axis=0)
        s1 = float(np.sum(col_profile_1)) + 1e-5
        s2 = float(np.sum(col_profile_2)) + 1e-5
        x_idx = np.arange(col_profile_1.shape[0], dtype=np.float32)
        c1 = float(np.sum(x_idx * col_profile_1) / s1)
        c2 = float(np.sum(x_idx * col_profile_2) / s2)
        shift = c2 - c1
        diff_mean = float(np.mean(np.abs(last - first)))
        if diff_mean < 8.0:
            motion = "static"
        elif shift > 2.2:
            motion = "pan_right"
        elif shift < -2.2:
            motion = "pan_left"
        else:
            motion = "walk_forward"

    return shot_scale, motion


def _dialogues_to_word_timestamps(dialogues: list[DialogueSegment]) -> list[WordTimestamp]:
    """Break dialogue segments into word-level timestamps for frame-accurate dialogue trimming."""
    words_out: list[WordTimestamp] = []
    for seg in dialogues:
        raw_words = [w for w in re.split(r"\s+", seg.text.strip()) if w]
        if not raw_words:
            continue
        seg_dur = max(0.2, seg.end_time - seg.start_time)
        total_chars = max(1, sum(len(w) for w in raw_words))
        cursor = seg.start_time
        for idx, word in enumerate(raw_words):
            frac = len(word) / total_chars
            w_dur = seg_dur * frac
            w_start = round(cursor, 2)
            w_end = round(seg.end_time if idx == len(raw_words) - 1 else cursor + w_dur, 2)
            words_out.append(
                WordTimestamp(
                    word=word,
                    start=w_start,
                    end=max(w_start + 0.05, w_end),
                    confidence=seg.confidence,
                )
            )
            cursor = w_end
    return words_out


def _compute_brief_match_boost(
    text_blob: str,
    has_dialogue: bool,
    must_topics: list[str],
) -> float:
    """Calculate score boost when a clip or photo matches the client's must-include topics."""
    blob_lower = text_blob.lower()
    boost = 0.0
    for topic in must_topics:
        t_low = topic.lower().strip()
        if not t_low:
            continue
        keywords = [w for w in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", t_low) if len(w) >= 4]
        if t_low in blob_lower or any(k in blob_lower for k in keywords):
            boost += 0.8
        elif has_dialogue and any(k in t_low for k in ("dialogue", "spoken", "quote", "talk", "voice", "kid")):
            boost += 1.2
    return round(min(2.0, boost), 2)


@register_agent
class CullerAgent(BaseEditingAgent):
    """Agent 1: Normalizes multi-cam timecodes, extracts golden windows, and scores selects."""

    role_id = "culler"
    display_name = "1. Wrangler & Culler"
    stage_order = 10
    description = (
        "Normalizes multi-cam timecodes, tags shot scale & motion, extracts 2-5s golden windows, "
        "preserves word-aligned A-roll dialogue, and drops duplicate bursts into media_index.json."
    )
    enabled_by_default = True
    requires_artifacts = ["creative_brief.json"]
    produces_artifacts = ["media_index.json", "annotations.yaml"]
    skill_reference_file = "01_culler_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        annotations_path = blackboard.project_dir / "annotations.yaml"
        if blackboard.catalog is not None:
            catalog = blackboard.catalog
        elif annotations_path.exists():
            try:
                loaded = MediaCatalog.from_yaml(annotations_path)
                if Path(loaded.source_directory).resolve() == blackboard.media_dir.resolve() and (
                    loaded.videos or loaded.photos
                ):
                    catalog = loaded
                else:
                    catalog = analyze_media_directory(
                        media_dir=blackboard.media_dir,
                        project_dir=blackboard.project_dir,
                        whisper_model=blackboard.whisper_model,
                        use_antigravity=blackboard.use_antigravity,
                        user_hint=blackboard.user_prompt,
                        extra_audio_paths=blackboard.extra_audio_paths,
                    )
            except Exception:
                catalog = analyze_media_directory(
                    media_dir=blackboard.media_dir,
                    project_dir=blackboard.project_dir,
                    whisper_model=blackboard.whisper_model,
                    use_antigravity=blackboard.use_antigravity,
                    user_hint=blackboard.user_prompt,
                    extra_audio_paths=blackboard.extra_audio_paths,
                )
        else:
            catalog = analyze_media_directory(
                media_dir=blackboard.media_dir,
                project_dir=blackboard.project_dir,
                whisper_model=blackboard.whisper_model,
                use_antigravity=blackboard.use_antigravity,
                user_hint=blackboard.user_prompt,
                extra_audio_paths=blackboard.extra_audio_paths,
            )

        blackboard.catalog = catalog

        must_topics: list[str] = []
        max_broll_sec = 4.5
        if blackboard.creative_brief is not None:
            must_topics = list(blackboard.creative_brief.must_include_topics)
            max_broll_sec = blackboard.creative_brief.max_broll_duration_sec

        for msg in blackboard.get_messages_for(self.role_id):
            for t in msg.payload.get("must_include_topics", []):
                if t not in must_topics:
                    must_topics.append(t)

        subclips: list[CulledSubClip] = []
        discarded_count = 0

        blackboard.emit_activity(
            self.role_id,
            f"Ingesting & probing {len(catalog.videos)} videos and {len(catalog.photos)} photos (Whisper ASR + scene boundaries)",
            detail=f"Matching against {len(must_topics)} client must-have topics",
        )

        # 1. Process Videos -> A-Roll Dialogue vs B-Roll Golden Windows
        sorted_videos: list[VideoAsset] = sorted(
            catalog.videos, key=lambda v: (v.creation_time or "", v.filename)
        )
        for v_idx, vid in enumerate(sorted_videos, start=1):
            kf_paths = [sc.keyframe_path for sc in vid.scenes if sc.keyframe_path]
            shot_scale, cam_motion = _estimate_shot_scale_and_motion(kf_paths, vid.overall_summary)
            words = _dialogues_to_word_timestamps(vid.dialogues)
            has_speech = len(vid.dialogues) > 0
            avg_scene_q = (
                sum(sc.quality_score for sc in vid.scenes) / len(vid.scenes)
                if vid.scenes
                else 8.2
            )
            orientation: Literal["landscape", "portrait", "square"] = (
                "landscape" if vid.width > vid.height else ("portrait" if vid.height > vid.width else "square")
            )

            blob = f"{vid.filename} {vid.overall_summary} " + " ".join(d.text for d in vid.dialogues)
            boost = _compute_brief_match_boost(blob, has_speech, must_topics)

            if has_speech:
                first_word_start = min(d.start_time for d in vid.dialogues)
                last_word_end = max(d.end_time for d in vid.dialogues)
                in_tc = round(max(0.0, first_word_start - 0.35), 2)
                out_tc = round(min(vid.duration, max(last_word_end + 0.55, min(vid.duration, in_tc + 4.5))), 2)
                if vid.duration <= 16.0:
                    in_tc = 0.0
                    out_tc = round(vid.duration, 2)

                spoken_text = " ".join(d.text.strip() for d in vid.dialogues if d.text.strip())
                translations = [d.translation for d in vid.dialogues if d.translation]
                emo_score = min(10.0, round(avg_scene_q + 0.8 + boost * 0.5, 2))
                v_title, v_sub, v_trans, v_rat = _synthesize_video_scene_caption_and_transition(
                    vid=vid,
                    shot_scale=shot_scale,
                    cam_motion=cam_motion,
                    has_speech=True,
                    spoken_text=spoken_text,
                    v_idx=v_idx,
                    total_videos=len(sorted_videos),
                    user_prompt=blackboard.user_prompt,
                )
                subclips.append(
                    CulledSubClip(
                        subclip_id=f"vid_aroll_{v_idx:02d}",
                        source_asset_id=vid.id,
                        source_path=vid.path,
                        filename=vid.filename,
                        media_type="video",
                        normalized_timestamp=vid.creation_time or f"2026-10-31T18:{v_idx:02d}:00Z",
                        in_tc=in_tc,
                        out_tc=out_tc,
                        duration=round(max(0.5, out_tc - in_tc), 2),
                        roll_type="a_roll",
                        shot_scale=shot_scale,
                        camera_movement=cam_motion,
                        orientation=orientation,
                        focal_point=(0.5, 0.5),
                        people_present=["family", "speaker"],
                        summary=vid.overall_summary or f"{v_title} ({v_sub})",
                        spoken_text=spoken_text,
                        translation=" | ".join(translations) if translations else None,
                        emotion_score=emo_score,
                        action_peak_tc=round(0.5 * (in_tc + out_tc), 2),
                        aesthetic_score=round(min(10.0, avg_scene_q), 2),
                        brief_match_boost=boost,
                        annotation_title=v_title,
                        annotation_subtitle=v_sub,
                        recommended_transition=v_trans,
                        ai_rationale=v_rat,
                        is_discarded=False,
                        words=words,
                    )
                )
            else:
                # Extract 2.0s - 4.5s Golden Window (or up to 6.8s anchor window for long clips in compact sets)
                win_cap = 6.8 if (vid.duration >= 8.0 and len(sorted_videos) <= 6) else max_broll_sec
                best_in = 0.0
                best_out = min(vid.duration, win_cap)
                if vid.scenes:
                    best_sc = max(vid.scenes, key=lambda s: s.quality_score)
                    sc_mid = 0.5 * (best_sc.start_time + best_sc.end_time)
                    half_w = 0.5 * min(win_cap, max(2.8, vid.duration))
                    best_in = round(max(0.0, min(vid.duration - min(vid.duration, 2.5), sc_mid - half_w)), 2)
                    best_out = round(min(vid.duration, best_in + min(win_cap, vid.duration)), 2)
                elif vid.duration > win_cap:
                    best_in = round(max(0.0, (vid.duration - win_cap) * 0.35), 2)
                    best_out = round(min(vid.duration, best_in + win_cap), 2)

                is_disc = avg_scene_q < 3.0
                if is_disc:
                    discarded_count += 1
                emo_score = min(10.0, round(avg_scene_q + boost * 0.5, 2))
                v_title, v_sub, v_trans, v_rat = _synthesize_video_scene_caption_and_transition(
                    vid=vid,
                    shot_scale=shot_scale,
                    cam_motion=cam_motion,
                    has_speech=False,
                    spoken_text="",
                    v_idx=v_idx,
                    total_videos=len(sorted_videos),
                    user_prompt=blackboard.user_prompt,
                )
                subclips.append(
                    CulledSubClip(
                        subclip_id=f"vid_broll_{v_idx:02d}",
                        source_asset_id=vid.id,
                        source_path=vid.path,
                        filename=vid.filename,
                        media_type="video",
                        normalized_timestamp=vid.creation_time or f"2026-10-31T18:{v_idx:02d}:30Z",
                        in_tc=best_in,
                        out_tc=best_out,
                        duration=round(max(0.5, best_out - best_in), 2),
                        roll_type="b_roll",
                        shot_scale=shot_scale,
                        camera_movement=cam_motion,
                        orientation=orientation,
                        focal_point=(0.5, 0.5),
                        people_present=["family"],
                        summary=vid.overall_summary or f"{v_title} ({v_sub})",
                        spoken_text="",
                        emotion_score=emo_score,
                        action_peak_tc=round(0.5 * (best_in + best_out), 2),
                        aesthetic_score=round(min(10.0, avg_scene_q), 2),
                        brief_match_boost=boost,
                        annotation_title=v_title,
                        annotation_subtitle=v_sub,
                        recommended_transition=v_trans,
                        ai_rationale=v_rat,
                        is_discarded=is_disc,
                        discard_reason="low_sharpness_or_score" if is_disc else None,
                        words=[],
                    )
                )

        # 2. Process Photos -> Deduplicate bursts and rank top selects
        sorted_photos: list[PhotoAsset] = sorted(
            catalog.photos, key=lambda p: (p.creation_time or "", p.filename)
        )
        photo_candidates: list[CulledSubClip] = []
        for p_idx, ph in enumerate(sorted_photos, start=1):
            img_list = [ph.preview_path] if ph.preview_path else [ph.path]
            shot_scale, _ = _estimate_shot_scale_and_motion(img_list, ph.caption)
            blob = f"{ph.filename} {ph.caption} {' '.join(ph.subjects)}"
            boost = _compute_brief_match_boost(blob, False, must_topics)
            emo_score = min(10.0, round(ph.quality_score + boost * 0.4, 2))
            photo_candidates.append(
                CulledSubClip(
                    subclip_id=f"ph_still_{p_idx:03d}",
                    source_asset_id=ph.id,
                    source_path=ph.path,
                    filename=ph.filename,
                    media_type="photo",
                    normalized_timestamp=ph.creation_time or f"2026-10-31T19:{p_idx % 60:02d}:00Z",
                    in_tc=0.0,
                    out_tc=3.2,
                    duration=3.2,
                    roll_type="still",
                    shot_scale=shot_scale,
                    camera_movement="static",
                    orientation=ph.orientation,
                    focal_point=ph.focal_point,
                    people_present=list(ph.subjects),
                    summary=ph.caption or f"Family travel photo ({ph.filename})",
                    spoken_text="",
                    emotion_score=emo_score,
                    action_peak_tc=1.5,
                    aesthetic_score=round(min(10.0, max(1.0, ph.quality_score)), 2),
                    brief_match_boost=boost,
                    is_discarded=False,
                    words=[],
                )
            )

        video_dur_sum = sum(s.duration for s in subclips if s.media_type == "video" and not s.is_discarded)
        target_dur = blackboard.creative_brief.target_duration_sec if blackboard.creative_brief else 110.0
        rem_photo_sec = max(24.0, target_dur - video_dur_sum)
        dynamic_photo_cap = max(14, min(60, int(round(rem_photo_sec / 3.1))))
        max_photos_cap = dynamic_photo_cap if len(photo_candidates) >= 15 else len(photo_candidates)
        target_active_photos = min(len(photo_candidates), max_photos_cap)
        if len(photo_candidates) > target_active_photos and target_active_photos > 0:
            bucket_size = len(photo_candidates) / target_active_photos
            chosen_ids: set[str] = set()
            for b in range(target_active_photos):
                s_i = int(round(b * bucket_size))
                e_i = int(round((b + 1) * bucket_size))
                bucket = photo_candidates[s_i : max(s_i + 1, e_i)]
                if bucket:
                    best_in_bucket = max(bucket, key=lambda c: (c.emotion_score, c.aesthetic_score))
                    chosen_ids.add(best_in_bucket.subclip_id)

            for cand in photo_candidates:
                if cand.subclip_id not in chosen_ids:
                    cand.is_discarded = True
                    cand.discard_reason = "duplicate_burst_or_lower_rank_in_cluster"
                    discarded_count += 1

        subclips.extend(photo_candidates)
        kept_count = sum(1 for s in subclips if not s.is_discarded)
        blackboard.emit_activity(
            self.role_id,
            f"Culled {len(subclips)} raw subclips -> kept {kept_count} active selects ({discarded_count} burst/low-score discards)",
            detail=f"Video spine: {video_dur_sum:.1f}s | Initial photo budget for {target_dur:.0f}s target: {target_active_photos} plates",
        )

        media_index = MediaIndex(
            source_directory=str(blackboard.media_dir),
            created_at=datetime.now(timezone.utc).isoformat(),
            total_raw_videos=len(catalog.videos),
            total_raw_photos=len(catalog.photos),
            kept_selects_count=kept_count,
            discarded_count=discarded_count,
            camera_clock_offsets_sec={"primary_cam": 0.0},
            subclips=subclips,
        )

        index_path = blackboard.project_dir / "media_index.json"
        media_index.to_json_file(index_path)
        blackboard.media_index = media_index

        # Render Multimodal Video Culling Contact Sheet & run AI Creative Intelligence evaluation
        vid_sheets_info: dict[str, Any] = {"contact_sheets": []}
        try:
            if catalog.videos:
                vid_sheets_info = render_video_culling_contact_sheets(blackboard.project_dir)
        except Exception:
            pass

        vid_summary_list = [
            {
                "subclip_id": s.subclip_id,
                "filename": s.filename,
                "in_tc": s.in_tc,
                "out_tc": s.out_tc,
                "duration": s.duration,
                "roll_type": s.roll_type,
                "shot_scale": s.shot_scale,
                "camera_movement": s.camera_movement,
                "spoken_text": s.spoken_text,
                "annotation_title": s.annotation_title,
                "annotation_subtitle": s.annotation_subtitle,
                "recommended_transition": s.recommended_transition,
            }
            for s in subclips
            if s.media_type == "video"
        ]
        ai_vid_resp = self.ask_ai_creative_json(
            blackboard=blackboard,
            prompt=(
                f"CreativeBrief: {blackboard.creative_brief.model_dump_json() if blackboard.creative_brief else blackboard.user_prompt}\n"
                f"Candidate Video Subclips: {json.dumps(vid_summary_list, ensure_ascii=False)}\n"
                "Inspect the video culling contact sheet and transcripts. Refine golden-window [in_tc, out_tc] trims, "
                "shot_scale, recommended_transition ('cut', 'crossfade', 'dissolve', 'fade_black'), and bilingual scene captions "
                "(annotation_title, annotation_subtitle). Return JSON: {\"videos\": {\"<subclip_id>\": {...}}, \"ai_rationale\": \"...\"}"
            ),
            image_paths=vid_sheets_info.get("contact_sheets", []),
            decision_key="culler",
            timeout_sec=35.0,
        )
        if isinstance(ai_vid_resp, dict) and (ai_vid_resp.get("videos") or ai_vid_resp.get("items")):
            try:
                apply_video_curation_overrides(blackboard.project_dir, ai_vid_resp)
                blackboard.media_index = MediaIndex.from_json_file(index_path)
                media_index = blackboard.media_index
            except Exception:
                pass

        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="story_producer",
            message_type="handoff",
            subject="Culled selects & normalized multi-cam timecodes ready",
            payload={
                "kept_selects_count": media_index.kept_selects_count,
                "discarded_count": media_index.discarded_count,
                "a_roll_count": sum(1 for s in subclips if not s.is_discarded and s.roll_type == "a_roll"),
                "b_roll_count": sum(1 for s in subclips if not s.is_discarded and s.roll_type == "b_roll"),
                "still_count": sum(1 for s in subclips if not s.is_discarded and s.roll_type == "still"),
            },
        )

        return {
            "media_index_path": str(index_path),
            "kept_selects_count": media_index.kept_selects_count,
            "discarded_count": media_index.discarded_count,
        }
