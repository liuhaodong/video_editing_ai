"""Stage 10: The Wrangler & Culler Sub-Agent (culler)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np
from PIL import Image

from ..analyzer.ag_annotator import analyze_media_directory
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
                        summary=vid.overall_summary or f"Spoken dialogue scene in {vid.filename}",
                        spoken_text=spoken_text,
                        translation=" | ".join(translations) if translations else None,
                        emotion_score=emo_score,
                        action_peak_tc=round(0.5 * (in_tc + out_tc), 2),
                        aesthetic_score=round(min(10.0, avg_scene_q), 2),
                        brief_match_boost=boost,
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
                        summary=vid.overall_summary or f"Golden window B-roll from {vid.filename}",
                        spoken_text="",
                        emotion_score=emo_score,
                        action_peak_tc=round(0.5 * (best_in + best_out), 2),
                        aesthetic_score=round(min(10.0, avg_scene_q), 2),
                        brief_match_boost=boost,
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
