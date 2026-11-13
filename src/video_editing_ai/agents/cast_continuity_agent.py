"""Stage 15: Cast Continuity, Utterance Language & Face-Safe Framing Agent (cast_tracker)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from PIL import Image, ImageOps

from ..models.schema import (
    CastManifest,
    CastSubjectEntry,
    CulledSubClip,
    SubclipSubjectDetection,
    UtteranceLanguageTag,
)
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


def classify_utterance_language(text: str) -> Literal["zh", "en", "mixed", "other"]:
    """Classify spoken utterance text into Mandarin ('zh'), English ('en'), 'mixed', or 'other'."""
    if not text or not text.strip():
        return "other"
    cjk_count = 0
    latin_count = 0
    other_alpha = 0
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            cjk_count += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            latin_count += 1
        elif ch.isalpha():
            other_alpha += 1

    if cjk_count > 0 and latin_count >= 3:
        return "mixed"
    if cjk_count > 0:
        return "zh"
    if latin_count > 0:
        return "en"
    if other_alpha > 0:
        return "other"
    return "other"


def detect_subject_bbox_normalized(
    image_path: Optional[str | Path],
    fallback_focal: tuple[float, float] = (0.5, 0.38),
) -> tuple[tuple[float, float, float, float], str]:
    """
    Detect normalized [x1, y1, x2, y2] subject/face bounding box in [0.0, 1.0].
    Uses OpenCV Haar frontal-face cascade when installed, with deterministic YCbCr skin + saliency fallback.
    """
    if image_path is None or not Path(image_path).exists():
        fx, fy = fallback_focal
        x1 = round(max(0.05, min(0.65, fx - 0.18)), 4)
        y1 = round(max(0.05, min(0.55, fy - 0.18)), 4)
        return (x1, y1, round(min(0.95, x1 + 0.36), 4), round(min(0.95, y1 + 0.42), 4)), "focal_fallback"

    try:
        with Image.open(image_path) as raw_im:
            rgb = ImageOps.exif_transpose(raw_im).convert("RGB")
            rgb.thumbnail((320, 320), Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.uint8)
    except Exception:
        fx, fy = fallback_focal
        return (
            round(max(0.05, fx - 0.18), 4),
            round(max(0.05, fy - 0.18), 4),
            round(min(0.95, fx + 0.18), 4),
            round(min(0.95, fy + 0.24), 4),
        ), "focal_fallback"

    h, w, _ = arr.shape
    if h < 16 or w < 16:
        return (0.30, 0.14, 0.70, 0.68), "focal_fallback"

    # 1. Optional OpenCV Haar face detector if cv2 is present
    try:
        import cv2  # type: ignore[import-not-found]

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        if Path(cascade_path).exists():
            detector = cv2.CascadeClassifier(cascade_path)
            faces = detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(20, 20))
            if len(faces) > 0:
                # Choose largest / highest-prominence face box
                bx, by, bw, bh = max(faces, key=lambda b: int(b[2]) * int(b[3]))
                x1 = round(max(0.0, float(bx) / w), 4)
                y1 = round(max(0.0, float(by) / h), 4)
                x2 = round(min(1.0, float(bx + bw) / w), 4)
                y2 = round(min(1.0, float(by + bh) / h), 4)
                if x2 > x1 + 0.04 and y2 > y1 + 0.04:
                    return (x1, y1, x2, y2), "opencv_haar_face"
    except Exception:
        pass

    # 2. Deterministic YCbCr skin-tone + upper-half gradient saliency bounding box
    arr_f = arr.astype(np.float32)
    r, g, b = arr_f[:, :, 0], arr_f[:, :, 1], arr_f[:, :, 2]
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    skin = (cr >= 135.0) & (cr <= 173.0) & (cb >= 80.0) & (cb <= 125.0)
    # Restrict face search to upper 78% of frame so warm floors/wood don't pull the box down
    upper_mask = np.zeros_like(skin)
    upper_mask[: int(h * 0.78), int(w * 0.08) : int(w * 0.92)] = True
    face_cand = skin & upper_mask

    if int(np.count_nonzero(face_cand)) >= max(12, int(h * w * 0.012)):
        ys, xs = np.nonzero(face_cand)
        x1 = round(max(0.02, float(np.percentile(xs, 10)) / w), 4)
        y1 = round(max(0.02, float(np.percentile(ys, 8)) / h), 4)
        x2 = round(min(0.98, float(np.percentile(xs, 90)) / w), 4)
        y2 = round(min(0.98, float(np.percentile(ys, 92)) / h), 4)
        if x2 - x1 >= 0.08 and y2 - y1 >= 0.08:
            return (x1, y1, x2, y2), "ycbcr_skin_cluster"

    # 3. Saliency energy centroid biased to upper-center portrait region
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    gy, gx = np.gradient(luma)
    yy, _ = np.mgrid[0:h, 0:w]
    head_prior = np.exp(-(((yy / max(1.0, float(h))) - 0.34) ** 2) / 0.18)
    sal = np.hypot(gx, gy) * head_prior
    total_sal = float(np.sum(sal)) + 1e-6
    ys_grid, xs_grid = np.mgrid[0:h, 0:w]
    cx = float(np.sum(xs_grid * sal) / total_sal) / w
    cy = float(np.sum(ys_grid * sal) / total_sal) / h
    half_bw = 0.18 if h > w else 0.16
    half_bh = 0.18 if h > w else 0.22
    x1 = round(max(0.04, min(0.80, cx - half_bw)), 4)
    y1 = round(max(0.04, min(0.72, cy - half_bh)), 4)
    x2 = round(min(0.96, max(x1 + 0.12, cx + half_bw)), 4)
    y2 = round(min(0.96, max(y1 + 0.12, cy + half_bh)), 4)
    return (x1, y1, x2, y2), "saliency_head_prior"


def compute_face_safe_ken_burns_centers(
    bbox: Optional[tuple[float, float, float, float] | list[float]] = None,
    orientation: str = "portrait",
    start_zoom: float = 1.0,
    end_zoom: float = 1.08,
    pan_sign: float = 1.0,
    fallback_focal: Optional[tuple[float, float]] = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute (start_center, end_center) biased toward subject/face bbox [x1, y1, x2, y2]
    so vertical photos and clips never crop off heads or faces during Ken Burns / widescreen framing.
    """
    if bbox is None or len(bbox) != 4:
        fx, fy = fallback_focal if fallback_focal is not None else (0.50, 0.38 if orientation == "portrait" else 0.48)
        fx = max(0.12, min(0.88, float(fx)))
        fy = max(0.14, min(0.82, float(fy)))
        dx = 0.018 * (1.0 if pan_sign >= 0 else -1.0)
        return (round(fx - dx, 4), round(fy - 0.01, 4)), (round(fx + dx, 4), round(fy + 0.01, 4))

    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    x1 = max(0.0, min(0.95, float(x1)))
    y1 = max(0.0, min(0.95, float(y1)))
    x2 = max(x1 + 0.04, min(1.0, float(x2)))
    y2 = max(y1 + 0.04, min(1.0, float(y2)))

    face_cx = 0.5 * (x1 + x2)
    # Anchor vertical center toward upper-eye line (38% down the bbox) to preserve crown/headroom above y1
    eye_cy = y1 + 0.38 * (y2 - y1)

    max_z = max(1.0, float(start_zoom), float(end_zoom))
    # Visible vertical half-window in normalized source coordinates (tighter for portrait cropped into 16:9)
    if orientation == "portrait":
        vis_half_h = min(0.5, 0.28 / max_z)
        vis_half_w = min(0.5, 0.46 / max_z)
    else:
        vis_half_h = min(0.5, 0.46 / max_z)
        vis_half_w = min(0.5, 0.46 / max_z)

    headroom = 0.025
    # Ensure top of head (y1 - headroom) stays >= cy - vis_half_h -> cy <= y1 - headroom + vis_half_h
    max_cy_for_headroom = max(0.08, (y1 - headroom) + vis_half_h)
    min_cy_for_chin = min(0.92, y2 - vis_half_h)
    if min_cy_for_chin <= max_cy_for_headroom:
        safe_cy = max(min_cy_for_chin, min(max_cy_for_headroom, eye_cy))
    else:
        # Face is taller than the tightest crop window: prioritize keeping crown/eyes inside frame
        safe_cy = max_cy_for_headroom

    safe_cy = max(0.08, min(0.90, safe_cy))
    safe_cx = max(max(0.08, x1 + 0.2 * (x2 - x1) - vis_half_w * 0.3), min(0.92, face_cx))

    drift_x = 0.015 * (1.0 if pan_sign >= 0 else -1.0)
    drift_y = -0.012
    start_c = (round(max(0.06, min(0.94, safe_cx - drift_x)), 4), round(max(0.06, min(0.92, safe_cy - drift_y)), 4))
    end_c = (
        round(max(0.06, min(0.94, min(safe_cx + drift_x, x2))), 4),
        round(max(0.06, min(max_cy_for_headroom, safe_cy + drift_y)), 4),
    )
    return start_c, end_c


@register_agent
class CastContinuityAgent(BaseEditingAgent):
    """Stage 15 Agent: Tracks recurring speakers/faces, tags spoken language per utterance, and records face bboxes."""

    role_id = "cast_tracker"
    display_name = "1d. Cast Continuity & Face-Safe Framing Tracker"
    stage_order = 15
    substage_priority = 1
    description = (
        "Groups recurring family/cast speakers and faces across clips, tags spoken language per utterance "
        "(Mandarin vs. English from faster-whisper segments), and records normalized subject/face bounding "
        "boxes (x1, y1, x2, y2) in cast_manifest.json for face-safe Ken Burns framing."
    )
    enabled_by_default = True
    requires_artifacts = ["media_index.json"]
    produces_artifacts = ["cast_manifest.json"]
    skill_reference_file = "15_cast_tracker_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        mi = blackboard.media_index
        catalog = blackboard.catalog
        brief = blackboard.creative_brief
        if mi is None:
            raise RuntimeError("CastContinuityAgent requires blackboard.media_index.")

        vid_by_id = {v.id: v for v in (catalog.videos if catalog else [])}
        ph_by_id = {p.id: p for p in (catalog.photos if catalog else [])}

        utterances: list[UtteranceLanguageTag] = []
        detections: dict[str, SubclipSubjectDetection] = {}
        speaker_buckets: dict[str, list[str]] = {}
        speaker_langs: dict[str, set[str]] = {}
        speaker_utt_counts: dict[str, int] = {}
        speaker_bboxes: dict[str, list[tuple[float, float, float, float]]] = {}
        detected_langs: set[str] = set()

        for sc in mi.subclips:
            if sc.is_discarded:
                continue

            sample_img: Optional[str] = None
            if sc.media_type == "photo":
                ph_asset = ph_by_id.get(sc.source_asset_id)
                sample_img = (ph_asset.preview_path if ph_asset and ph_asset.preview_path else None) or sc.source_path
            else:
                v_asset = vid_by_id.get(sc.source_asset_id)
                if v_asset and v_asset.scenes:
                    sample_img = next((s.keyframe_path for s in v_asset.scenes if s.keyframe_path), None)

            bbox, det_kind = detect_subject_bbox_normalized(sample_img, fallback_focal=sc.focal_point)
            face_center = (round(0.5 * (bbox[0] + bbox[2]), 4), round(bbox[1] + 0.38 * (bbox[3] - bbox[1]), 4))
            # Update subclip focal_point toward detected face center so downstream consumers benefit automatically
            sc.focal_point = face_center

            assigned_cast: list[str] = []
            if sc.media_type == "video":
                v_asset = vid_by_id.get(sc.source_asset_id)
                segs = list(v_asset.dialogues) if (v_asset and v_asset.dialogues) else []
                if segs:
                    for seg in segs:
                        lang = classify_utterance_language(seg.text)
                        detected_langs.add(lang)
                        raw_spk = (seg.speaker or "Speaker").strip()
                        cast_id = "cast_01" if raw_spk.lower() in ("speaker", "speaker 1", "adult", "parent") else f"cast_{abs(hash(raw_spk)) % 4 + 1:02d}"
                        if sc.integrated_lufs < -22.0 and cast_id == "cast_01" and len(segs) > 1:
                            cast_id = "cast_02"
                        utterances.append(
                            UtteranceLanguageTag(
                                subclip_id=sc.subclip_id,
                                start_sec=round(float(seg.start_time), 2),
                                end_sec=round(float(seg.end_time), 2),
                                language=lang,
                                speaker_id=cast_id,
                                text=seg.text.strip(),
                            )
                        )
                        if cast_id not in assigned_cast:
                            assigned_cast.append(cast_id)
                        speaker_langs.setdefault(cast_id, set()).add(lang)
                        speaker_utt_counts[cast_id] = speaker_utt_counts.get(cast_id, 0) + 1
                elif sc.spoken_text.strip():
                    lang = classify_utterance_language(sc.spoken_text)
                    detected_langs.add(lang)
                    cast_id = "cast_01"
                    utterances.append(
                        UtteranceLanguageTag(
                            subclip_id=sc.subclip_id,
                            start_sec=round(float(sc.in_tc), 2),
                            end_sec=round(float(sc.out_tc), 2),
                            language=lang,
                            speaker_id=cast_id,
                            text=sc.spoken_text.strip(),
                        )
                    )
                    assigned_cast.append(cast_id)
                    speaker_langs.setdefault(cast_id, set()).add(lang)
                    speaker_utt_counts[cast_id] = speaker_utt_counts.get(cast_id, 0) + 1

            if not assigned_cast and det_kind in ("opencv_haar_face", "ycbcr_skin_cluster"):
                assigned_cast.append("cast_01")

            for cid in assigned_cast:
                speaker_buckets.setdefault(cid, [])
                if sc.subclip_id not in speaker_buckets[cid]:
                    speaker_buckets[cid].append(sc.subclip_id)
                speaker_bboxes.setdefault(cid, []).append(bbox)

            detections[sc.subclip_id] = SubclipSubjectDetection(
                subclip_id=sc.subclip_id,
                media_type=sc.media_type,
                bbox=bbox,
                face_center=face_center,
                cast_ids=assigned_cast,
                detector=det_kind,
            )

        if not speaker_buckets and detections:
            first_sid = next(iter(detections.keys()))
            speaker_buckets["cast_01"] = [first_sid]
            speaker_bboxes["cast_01"] = [detections[first_sid].bbox]

        cast_members: list[CastSubjectEntry] = []
        for idx, cid in enumerate(sorted(speaker_buckets.keys()), start=1):
            bboxes = speaker_bboxes.get(cid) or [(0.30, 0.14, 0.70, 0.68)]
            rep_bbox = (
                round(sum(b[0] for b in bboxes) / len(bboxes), 4),
                round(sum(b[1] for b in bboxes) / len(bboxes), 4),
                round(sum(b[2] for b in bboxes) / len(bboxes), 4),
                round(sum(b[3] for b in bboxes) / len(bboxes), 4),
            )
            cast_members.append(
                CastSubjectEntry(
                    cast_id=cid,
                    label=f"Recurring Subject {idx}",
                    appearance_subclip_ids=speaker_buckets[cid],
                    primary_languages=sorted(speaker_langs.get(cid, set())),
                    utterance_count=speaker_utt_counts.get(cid, 0),
                    representative_bbox=rep_bbox,
                )
            )

        proj_title = brief.project_title if brief else blackboard.media_dir.name
        manifest = CastManifest(
            project_title=proj_title,
            recurring_cast_count=len(cast_members),
            languages_detected=sorted(detected_langs),
            cast_members=cast_members,
            utterances=utterances,
            subclip_detections=detections,
        )
        out_path = blackboard.project_dir / "cast_manifest.json"
        manifest.to_json_file(out_path)
        blackboard.cast_manifest = manifest
        blackboard.set_custom_artifact("cast_manifest.json", manifest.model_dump(mode="json"))
        # Persist updated face-aware focal points to media_index.json
        mi.to_json_file(blackboard.project_dir / "media_index.json")

        blackboard.emit_activity(
            self.role_id,
            f"Tracked {len(cast_members)} recurring cast members, {len(utterances)} language-tagged utterances, and {len(detections)} face/subject boxes",
            detail=f"Languages detected: {sorted(detected_langs) or ['instrumental_visual']}",
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="montage_editor",
            message_type="handoff",
            subject=f"Cast continuity & {len(detections)} subject bounding boxes ready for face-safe framing",
            payload={
                "recurring_cast_count": len(cast_members),
                "utterances_count": len(utterances),
                "languages_detected": sorted(detected_langs),
            },
        )
        return {
            "cast_manifest_path": str(out_path),
            "recurring_cast_count": len(cast_members),
            "utterances_count": len(utterances),
            "detections_count": len(detections),
            "subject_boxes": len(detections),
        }
