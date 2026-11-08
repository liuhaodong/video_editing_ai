"""Stage 15: Aesthetic Photo Curator & Subject Annotator Sub-Agent (photo_curator)."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
from typing import Any, Literal, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..analyzer.media_probe import _extract_exif_camera_info, _get_cjk_font
from ..models.schema import (
    VALID_PHOTO_MODES,
    VALID_PHOTO_OVERLAY_FX,
    CulledSubClip,
    CuratedPhotoItem,
    EditTimeline,
    MediaIndex,
    PhotoAsset,
    PhotoCurationReport,
)
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


def _compute_dhash_hex(img_arr_gray: np.ndarray) -> str:
    """Compute 64-bit horizontal gradient difference hash (dHash) from a grayscale array."""
    try:
        pil_g = Image.fromarray(img_arr_gray.astype(np.uint8)).resize((9, 8), Image.Resampling.BILINEAR)
        small = np.asarray(pil_g, dtype=np.float32)
        diff = small[:, 1:] > small[:, :-1]
        bits = diff.flatten()
        val = 0
        for b in bits:
            val = (val << 1) | int(bool(b))
        return f"{val:016x}"
    except Exception:
        return "0000000000000000"


def _hamming_distance_hex(h1: str, h2: str) -> int:
    try:
        return (int(h1, 16) ^ int(h2, 16)).bit_count()
    except Exception:
        return 64


def _evaluate_photo_taste_metrics(img_path: str | Path, filename: str) -> dict[str, Any]:
    """Compute 5-factor aesthetic & editorial taste metrics on a photo preview."""
    try:
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            rgb.thumbnail((256, 256), Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32)
    except Exception:
        return {
            "dhash": "0000000000000000",
            "saliency_score": 7.5,
            "composition_score": 7.5,
            "portrait_warmth_score": 7.0,
            "art_architecture_score": 7.5,
            "overall_taste_score": 7.5,
            "inferred_category": "landmark_architecture",
        }

    h, w, _ = arr.shape
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    dhash = _compute_dhash_hex(luma)

    # Gradient energy map
    gy, gx = np.gradient(luma)
    energy = np.hypot(gx, gy)

    # 1. Subject Saliency: compare central/rule-of-thirds box [0.18..0.82] vs bottom ground strip [0.82..1.0]
    y0, y1 = int(h * 0.16), int(h * 0.80)
    x0, x1 = int(w * 0.16), int(w * 0.84)
    center_energy = float(np.mean(energy[y0:y1, x0:x1])) + 1e-5
    outer_energy = float(np.mean(energy)) + 1e-5
    bottom_strip_energy = float(np.mean(energy[int(h * 0.80) :, :])) + 1e-5

    focus_ratio = center_energy / outer_energy
    ground_penalty = max(0.0, (bottom_strip_energy / center_energy) - 1.08) * 1.8
    saliency_score = round(min(9.9, max(4.5, 7.4 + (focus_ratio - 0.95) * 3.8 - ground_penalty)), 2)

    # 2. Composition & Dynamic Range (well-balanced contrast without blown sky or flat gray wall)
    luma_std = float(np.std(luma))
    luma_mean = float(np.mean(luma))
    exposure_balance = 1.0 - min(1.0, abs(luma_mean - 122.0) / 110.0)
    composition_score = round(
        min(9.9, max(5.0, 6.8 + min(1.8, luma_std / 34.0) + exposure_balance * 1.1 - ground_penalty * 0.5)),
        2,
    )

    # 3. Portrait & Skin-Tone Warmth Detection (YCbCr skin cluster + warm palette)
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    center_cb = cb[y0:y1, x0:x1]
    center_cr = cr[y0:y1, x0:x1]
    skin_mask = (center_cr >= 136.0) & (center_cr <= 172.0) & (center_cb >= 82.0) & (center_cb <= 124.0)
    skin_frac = float(np.mean(skin_mask))
    warm_diff = float(np.mean(r - b))
    portrait_warmth_score = round(
        min(9.9, max(5.5, 6.8 + min(2.2, skin_frac * 9.5) + min(0.9, max(0.0, warm_diff) / 28.0))),
        2,
    )

    # 4. Art, Gilded Frame, Marble Sculpture & Landmark Architecture Score
    sat = np.max(arr, axis=2) - np.min(arr, axis=2)
    color_richness = float(np.mean(sat[y0:y1, x0:x1]))
    # Gilded frame signature: warm golden border around the inner canvas
    border_warmth = float(np.mean((r - b)[int(h * 0.08) : int(h * 0.22), :]))
    art_arch_score = round(
        min(
            9.9,
            max(
                5.5,
                7.0
                + min(1.4, color_richness / 42.0)
                + min(0.9, max(0.0, border_warmth) / 30.0)
                + min(0.6, (focus_ratio - 0.9) * 2.0),
            ),
        ),
        2,
    )

    # Infer visual archetype when not in explicit masterpiece table
    sky_blue = float(np.mean((b - r)[: int(h * 0.35), :]))
    if skin_frac > 0.16 and h > w:
        inferred_cat = "family_portrait"
    elif border_warmth > 18.0 and color_richness > 26.0:
        inferred_cat = "masterpiece_painting"
    elif sky_blue > 14.0 or w > h * 1.2:
        inferred_cat = "landmark_architecture"
    else:
        inferred_cat = "sculpture"

    overall = round(
        min(
            9.95,
            0.34 * saliency_score
            + 0.28 * composition_score
            + 0.22 * art_arch_score
            + 0.16 * portrait_warmth_score,
        ),
        2,
    )

    return {
        "dhash": dhash,
        "saliency_score": saliency_score,
        "composition_score": composition_score,
        "portrait_warmth_score": portrait_warmth_score,
        "art_architecture_score": art_arch_score,
        "overall_taste_score": overall,
        "inferred_category": inferred_cat,
    }


def _build_camera_badge(
    filename: str,
    camera_make_model: Optional[str] = None,
    project_badge_tag: str = "",
) -> str:
    """Return an empty badge by default so captions remain clean and free of technical camera/EXIF metadata."""
    return ""


def _generate_semantic_annotation(
    filename: str,
    inferred_category: str,
    orientation: str,
    story_pct: int,
    project_label: str = "Featured Story",
) -> tuple[str, str, str]:
    """Return baseline mechanical (annotation_title, annotation_subtitle, curation_rationale) prior to AI Vision curation."""
    if inferred_category == "masterpiece_painting":
        return (
            "Fine Art & Cultural Detail",
            f"{project_label} · Curated Visual Study",
            "Selected for high color richness, central framing, and detailed visual texture.",
        )
    if inferred_category == "sculpture":
        return (
            "Sculptural & Architectural Study",
            f"{project_label} · Form & Light",
            "Selected for strong central subject saliency and balanced lighting.",
        )
    if inferred_category == "family_portrait":
        return (
            "Candid Keepsake Portrait",
            f"{project_label} · Personal Moment",
            "Selected for warm skin-tone balance and personal storytelling value.",
        )
    if story_pct < 25:
        return (
            "Opening Horizon & Establishing Vista",
            f"{project_label} · Opening Chapter",
            "Selected as a wide establishing composition in the opening chapter.",
        )
    if story_pct > 75:
        return (
            "Twilight Atmosphere & Closing Finale",
            f"{project_label} · Closing Chapter",
            "Selected for warm tonal balance in the closing chapter.",
        )
    return (
        "Architectural & Scenic Perspective",
        f"{project_label} · Selected Frame",
        "Selected for balanced structural symmetry and subject focus.",
    )


@register_agent
class PhotoCuratorAgent(BaseEditingAgent):
    """Stage 15: Aesthetic Photo Curator & Subject Annotator (photo_curator)."""

    role_id = "photo_curator"
    display_name = "1c. Aesthetic Photo Curator & Annotator"
    stage_order = 15
    description = (
        "Evaluates all photos using 5-factor aesthetic taste scoring (subject saliency, rule-of-thirds "
        "composition, portrait warmth, artwork/landmark recognition, and 64-bit dHash burst deduplication), "
        "selects the most compelling shots, and delegates visual subject annotation & semantic photo effect pairing to the AI Agent."
    )
    enabled_by_default = True
    requires_artifacts = ["creative_brief.json", "media_index.json"]
    produces_artifacts = ["photo_curation.json"]
    skill_reference_file = "01_culler_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        media_index = blackboard.media_index
        catalog = blackboard.catalog
        brief = blackboard.creative_brief
        if media_index is None:
            raise RuntimeError("PhotoCuratorAgent requires blackboard.media_index from CullerAgent.")

        project_label = (
            (brief.project_title if brief and brief.project_title else blackboard.media_dir.name)
            .replace("_", " ")
            .strip()
        )
        project_badge_tag = project_label.upper()[:22] if project_label else "4K HDR ARCHIVE"

        # Map filename -> preview_path and EXIF camera_make_model for fast image analysis
        preview_by_name: dict[str, str] = {}
        camera_by_name: dict[str, Optional[str]] = {}
        if catalog is not None:
            for p in catalog.photos:
                preview_by_name[p.filename] = p.preview_path or p.path
                camera_by_name[p.filename] = getattr(p, "camera_make_model", None)

        photo_subclips = [s for s in media_index.subclips if s.media_type == "photo"]
        total_photos = len(photo_subclips)

        # Scale number of curated photos dynamically from brief.target_duration_sec
        video_dur_sum = sum(s.duration for s in media_index.subclips if s.media_type == "video" and not s.is_discarded)
        target_dur = brief.target_duration_sec if brief else 110.0
        rem_photo_sec = max(24.0, target_dur - video_dur_sum)
        dynamic_select_count = max(14, min(60, int(round(rem_photo_sec / 3.1))))
        target_select_count = min(total_photos, dynamic_select_count if total_photos >= 15 else total_photos)
        min_taste_floor = 7.8
        for msg in blackboard.get_messages_for(self.role_id):
            if msg.message_type == "director_directive":
                if "min_taste_score" in msg.payload:
                    min_taste_floor = float(msg.payload["min_taste_score"])

        blackboard.emit_activity(
            self.role_id,
            f"Evaluating {total_photos} photos for aesthetic taste (saliency, rule-of-thirds, portrait warmth & 64-bit dHash clusters)",
            detail=f"Target duration {target_dur:.0f}s (video spine {video_dur_sum:.1f}s) -> curating top {target_select_count} gallery plates (floor >= {min_taste_floor})",
        )

        curated_items: list[CuratedPhotoItem] = []
        subclip_by_id: dict[str, CulledSubClip] = {s.subclip_id: s for s in photo_subclips}

        # Step 1: Score every photo mechanically and assign perceptual dHash burst clusters
        cluster_id = 0
        prev_hash: Optional[str] = None

        for idx, sc in enumerate(photo_subclips):
            img_path = preview_by_name.get(sc.filename, sc.source_path)
            metrics = _evaluate_photo_taste_metrics(img_path, sc.filename)
            cur_hash = str(metrics["dhash"])

            if prev_hash is None or _hamming_distance_hex(prev_hash, cur_hash) > 10:
                cluster_id += 1
            prev_hash = cur_hash

            story_pct = int(round((idx / max(1, total_photos - 1)) * 100))
            inferred_cat = str(metrics["inferred_category"])
            title, subtitle, rationale = _generate_semantic_annotation(
                sc.filename,
                inferred_cat,
                sc.orientation,
                story_pct,
                project_label,
            )
            cam_model = camera_by_name.get(sc.filename)
            if not cam_model and Path(sc.source_path).exists():
                try:
                    with Image.open(sc.source_path) as raw_im:
                        cam_model, _ = _extract_exif_camera_info(raw_im, Path(sc.source_path))
                except Exception:
                    cam_model = None
            badge = _build_camera_badge(sc.filename, cam_model, project_badge_tag)

            curated_items.append(
                CuratedPhotoItem(
                    subclip_id=sc.subclip_id,
                    filename=sc.filename,
                    source_path=sc.source_path,
                    orientation=sc.orientation,
                    perceptual_hash=cur_hash,
                    cluster_id=cluster_id,
                    saliency_score=float(metrics["saliency_score"]),
                    composition_score=float(metrics["composition_score"]),
                    portrait_warmth_score=float(metrics["portrait_warmth_score"]),
                    art_architecture_score=float(metrics["art_architecture_score"]),
                    overall_taste_score=float(metrics["overall_taste_score"]),
                    is_selected=False,
                    subject_category=inferred_cat,
                    annotation_title=title,
                    annotation_subtitle=subtitle,
                    camera_badge=badge,
                    recommended_photo_mode="parallax_gallery",
                    curation_rationale=rationale,
                )
            )

        # Step 2: Deduplicate burst clusters (keep only the highest-taste frame per perceptual cluster)
        best_in_cluster: dict[int, CuratedPhotoItem] = {}
        for item in curated_items:
            cid = item.cluster_id
            existing = best_in_cluster.get(cid)
            if existing is None or item.overall_taste_score > existing.overall_taste_score:
                if existing is not None:
                    existing.rejection_reason = f"burst_duplicate_of_{item.filename}"
                best_in_cluster[cid] = item
            else:
                item.rejection_reason = f"burst_duplicate_of_{existing.filename}"

        deduped_pool = list(best_in_cluster.values())

        # Step 3: Select top photos across chronological story buckets by highest overall_taste_score
        chosen_ids: set[str] = set()
        remaining_needed = target_select_count
        if remaining_needed > 0 and deduped_pool:
            num_buckets = min(remaining_needed, 8)
            bucket_len = max(1, len(deduped_pool) / num_buckets)
            for b_i in range(num_buckets):
                s_idx = int(round(b_i * bucket_len))
                e_idx = int(round((b_i + 1) * bucket_len))
                candidates_in_b = [
                    it
                    for it in deduped_pool[s_idx : max(s_idx + 1, e_idx)]
                    if it.subclip_id not in chosen_ids and it.overall_taste_score >= min_taste_floor
                ]
                if not candidates_in_b:
                    candidates_in_b = [
                        it
                        for it in deduped_pool[s_idx : max(s_idx + 1, e_idx)]
                        if it.subclip_id not in chosen_ids
                    ]
                candidates_in_b.sort(key=lambda x: x.overall_taste_score, reverse=True)
                for pick in candidates_in_b[: max(1, int(np.ceil(remaining_needed / num_buckets)))]:
                    if len(chosen_ids) < target_select_count:
                        chosen_ids.add(pick.subclip_id)

        # If bucket rounding left any unfilled slots, top up with the highest-scoring remaining deduped photos
        if len(chosen_ids) < target_select_count:
            leftover = sorted(
                [it for it in deduped_pool if it.subclip_id not in chosen_ids],
                key=lambda x: x.overall_taste_score,
                reverse=True,
            )
            for pick in leftover[: target_select_count - len(chosen_ids)]:
                chosen_ids.add(pick.subclip_id)

        # Step 4: Pair complementary vertical ('portrait') selects for Diptych / Polaroid Split-Screen
        # and assign diverse cinema-grade photo modes & atmospheric finishers across selected plates
        selected_items = [it for it in curated_items if it.subclip_id in chosen_ids]
        portrait_selects = [it for it in selected_items if it.orientation == "portrait"]
        absorbed_partner_ids: set[str] = set()
        if len(portrait_selects) >= 2:
            p1, p2 = portrait_selects[0], portrait_selects[1]
            p1.recommended_photo_mode = "diptych_frame"
            p1.overlay_fx = "golden_light_leak"
            p1.paired_with_subclip_id = p2.subclip_id
            t1_short = p1.annotation_title.split("—")[0].strip()
            t2_short = p2.annotation_title.split("—")[0].strip()
            if t1_short != t2_short:
                p1.annotation_title = f"Curated Diptych: {t1_short} & {t2_short}"
                s1_short = p1.annotation_subtitle.split("·")[0].strip()
                s2_short = p2.annotation_subtitle.split("·")[0].strip()
                p1.annotation_subtitle = f"{s1_short} · {s2_short}"
            absorbed_partner_ids.add(p2.subclip_id)

        # Assign semantic AI-paired photo modes & atmospheric finishers to remaining primary selects,
        # honoring both the photo's visual content (portrait vs. fine art vs. culinary vs. skyline)
        # and the client's multi-select photo_effects_suite & photo_overlay_finishers from CreativeBrief
        active_selects = [it for it in selected_items if it.subclip_id not in absorbed_partner_ids]
        default_fx_map: dict[str, str] = {
            "cinematic_depth_3d": "golden_light_leak",
            "parallax_gallery": "museum_spotlight",
            "editorial_magazine": "none",
            "curator_loupe": "museum_spotlight",
            "polaroid_scrapbook": "nostalgic_film_grain",
            "triptych_cinematic": "none",
            "cinema_anamorphic": "champagne_bokeh",
            "gallery_frame": "museum_spotlight",
            "full_bleed": "golden_light_leak",
        }
        client_modes = [
            m for m in (brief.enabled_photo_modes if brief and brief.enabled_photo_modes else [])
            if m != "diptych_frame"
        ]
        if not client_modes:
            client_modes = [
                "cinematic_depth_3d",
                "parallax_gallery",
                "editorial_magazine",
                "curator_loupe",
                "polaroid_scrapbook",
                "triptych_cinematic",
                "cinema_anamorphic",
            ]
        client_overlays = set(brief.enabled_overlay_fx if brief and brief.enabled_overlay_fx else default_fx_map.values())
        allow_overlays = not ("none" in client_overlays and len(client_overlays) == 1)
        non_none_overlays = [fx for fx in client_overlays if fx != "none"]

        def _pick_overlay_for_mode(m_code: str, idx_hint: int) -> str:
            if not allow_overlays:
                return "none"
            pref_fx = default_fx_map.get(m_code, "none")
            if pref_fx in client_overlays or pref_fx == "none":
                return pref_fx
            if non_none_overlays:
                return non_none_overlays[idx_hint % len(non_none_overlays)]
            return "none"

        prev_assigned_mode = ""
        for idx_sel, sel_it in enumerate(active_selects):
            if sel_it.recommended_photo_mode == "diptych_frame":
                if not allow_overlays:
                    sel_it.overlay_fx = "none"
                prev_assigned_mode = "diptych_frame"
                continue

            blob_low = f"{sel_it.annotation_title} {sel_it.annotation_subtitle} {sel_it.subject_category}".lower()
            if any(k in blob_low for k in ("mosaic", "sculpture", "relief", "painting", "stained glass", "dome", "carving", "fresco", "detail", "masterpiece")):
                semantic_prefs = ["curator_loupe", "parallax_gallery", "editorial_magazine", "cinematic_depth_3d"]
            elif any(k in blob_low for k in ("culinary", "bakery", "cheese", "café", "cafe", "menu", "dining", "food", "pastry", "candid", "smile", "selfie")):
                semantic_prefs = ["polaroid_scrapbook", "cinematic_depth_3d", "parallax_gallery", "editorial_magazine"]
            elif sel_it.subject_category == "family_portrait" or sel_it.portrait_warmth_score >= 8.9:
                semantic_prefs = ["cinematic_depth_3d", "polaroid_scrapbook", "parallax_gallery", "editorial_magazine"]
            elif any(k in blob_low for k in ("panorama", "skyline", "river", "waterfront", "twilight", "horizon", "illuminated", "night")):
                semantic_prefs = ["cinema_anamorphic", "editorial_magazine", "parallax_gallery", "cinematic_depth_3d"]
            else:
                semantic_prefs = ["editorial_magazine", "parallax_gallery", "triptych_cinematic", "cinematic_depth_3d", "curator_loupe"]

            # Filter by client-enabled modes and avoid consecutive identical modes
            valid_prefs = [m for m in semantic_prefs if m in client_modes]
            if not valid_prefs:
                valid_prefs = list(client_modes)

            chosen_mode = valid_prefs[0]
            if chosen_mode == prev_assigned_mode and len(valid_prefs) > 1:
                chosen_mode = valid_prefs[1]
            elif chosen_mode == prev_assigned_mode and len(client_modes) > 1:
                alt_modes = [m for m in client_modes if m != prev_assigned_mode]
                chosen_mode = alt_modes[idx_sel % len(alt_modes)]

            chosen_fx = _pick_overlay_for_mode(chosen_mode, idx_sel)
            sel_it.recommended_photo_mode = chosen_mode  # type: ignore[assignment]
            sel_it.overlay_fx = chosen_fx  # type: ignore[assignment]
            prev_assigned_mode = chosen_mode

        # Step 5: Sync curated selections, scores, modes, semantic transitions, and bilingual annotations back to media_index.subclips
        for item in curated_items:
            sc = subclip_by_id[item.subclip_id]
            sc.aesthetic_score = item.overall_taste_score
            sc.emotion_score = round(min(10.0, max(item.overall_taste_score, item.portrait_warmth_score)), 2)
            sc.annotation_title = item.annotation_title
            sc.annotation_subtitle = item.annotation_subtitle
            sc.camera_badge = item.camera_badge
            sc.subject_category = item.subject_category
            sc.recommended_photo_mode = item.recommended_photo_mode
            sc.overlay_fx = item.overlay_fx
            sc.recommended_transition = (
                "dissolve"
                if item.recommended_photo_mode in ("curator_loupe", "cinema_anamorphic", "triptych_cinematic", "diptych_frame")
                else "crossfade"
            )
            sc.ai_rationale = (
                f"Semantic Mode='{item.recommended_photo_mode}' + Finisher='{item.overlay_fx}' — {item.curation_rationale}"
            )
            sc.summary = f"{item.annotation_title} ({item.annotation_subtitle}) — {item.curation_rationale}"

            if item.subclip_id in chosen_ids and item.subclip_id not in absorbed_partner_ids:
                item.is_selected = True
                item.rejection_reason = None
                sc.is_discarded = False
                sc.discard_reason = None
                if item.paired_with_subclip_id and item.paired_with_subclip_id in subclip_by_id:
                    partner_sc = subclip_by_id[item.paired_with_subclip_id]
                    sc.secondary_source_path = partner_sc.source_path
                if item.tertiary_subclip_id and item.tertiary_subclip_id in subclip_by_id:
                    tert_sc = subclip_by_id[item.tertiary_subclip_id]
                    sc.tertiary_source_path = tert_sc.source_path
            elif item.subclip_id in absorbed_partner_ids:
                item.is_selected = True
                item.rejection_reason = None
                sc.is_discarded = True
                sc.discard_reason = "paired_as_secondary_panel_in_diptych_frame"
            else:
                item.is_selected = False
                if not item.rejection_reason:
                    item.rejection_reason = "lower_aesthetic_taste_or_saliency_in_chapter"
                sc.is_discarded = True
                sc.discard_reason = item.rejection_reason

        media_index.kept_selects_count = sum(1 for s in media_index.subclips if not s.is_discarded)
        media_index.discarded_count = sum(1 for s in media_index.subclips if s.is_discarded)
        media_index.to_json_file(blackboard.project_dir / "media_index.json")

        avg_taste = (
            round(sum(it.overall_taste_score for it in selected_items) / len(selected_items), 2)
            if selected_items
            else 9.0
        )

        report = PhotoCurationReport(
            project_title=brief.project_title if brief else "Family_Travel_Story",
            total_evaluated=len(curated_items),
            burst_clusters_found=cluster_id,
            selected_count=len(selected_items),
            rejected_count=len(curated_items) - len(selected_items),
            average_taste_score_selected=avg_taste,
            curatorial_statement=(
                f"Curated {len(selected_items)} high-taste photographs (avg taste score {avg_taste}/10) across "
                f"{cluster_id} perceptual burst clusters, prioritizing iconic masterpieces, architectural symmetry, "
                "and candid family portraits while rejecting low-saliency floor/bench shots."
            ),
            items=curated_items,
        )

        out_path = blackboard.project_dir / "photo_curation.json"
        report.to_json_file(out_path)
        blackboard.photo_curation = report

        # Mechanically generate visual contact sheets so Antigravity / AI Agent can inspect & override
        cs_info = render_curation_contact_sheets(blackboard.project_dir)

        # Invoke Multimodal AI Creative Intelligence (or load cached ai_decisions/photo_curator.json)
        ai_photo_resp = self.ask_ai_creative_json(
            blackboard=blackboard,
            prompt=(
                f"CreativeBrief: {brief.model_dump_json() if brief else blackboard.user_prompt}\n"
                f"Enabled Photo Modes: {client_modes}\n"
                f"Curated Photo Selects ({len(selected_items)}): "
                f"{json.dumps([{'subclip_id': it.subclip_id, 'filename': it.filename, 'title': it.annotation_title, 'subtitle': it.annotation_subtitle, 'mode': it.recommended_photo_mode, 'fx': it.overlay_fx} for it in selected_items], ensure_ascii=False)}\n"
                "Inspect the numbered photo contact sheets. Refine any photo titles/subtitles, recommended_photo_mode, "
                "overlay_fx, or diptych pairings if helpful. Return JSON compatible with apply_vision_curation_overrides."
            ),
            image_paths=cs_info.get("photo_sheets", []),
            decision_key="photo_curator",
            timeout_sec=40.0,
        )
        if isinstance(ai_photo_resp, dict) and (
            ai_photo_resp.get("items")
            or ai_photo_resp.get("overrides")
            or ai_photo_resp.get("select_ids")
            or ai_photo_resp.get("annotations")
        ):
            try:
                apply_vision_curation_overrides(
                    blackboard.project_dir,
                    ai_photo_resp,
                    curatorial_statement=ai_photo_resp.get("curatorial_statement"),
                )
                blackboard.photo_curation = PhotoCurationReport.from_json_file(out_path)
                blackboard.media_index = MediaIndex.from_json_file(blackboard.project_dir / "media_index.json")
                report = blackboard.photo_curation
                selected_items = [it for it in report.items if it.is_selected]
                avg_taste = report.average_taste_score_selected
                cs_info = render_curation_contact_sheets(blackboard.project_dir)
            except Exception:
                pass

        sample_titles = ", ".join(it.annotation_title.split("—")[0].strip() for it in selected_items[:4])
        blackboard.emit_activity(
            self.role_id,
            f"Curated {len(selected_items)} high-taste photos (avg {avg_taste}/10 across {cluster_id} burst clusters) + rendered {len(cs_info.get('photo_sheets', []))} contact sheets",
            detail=f"Featured plates: {sample_titles}...",
        )

        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="story_producer",
            message_type="handoff",
            subject=f"Curated {len(selected_items)} high-taste photos (avg {avg_taste}/10) with bilingual annotations",
            payload={
                "selected_photos": len(selected_items),
                "average_taste_score": avg_taste,
                "annotated_count": len(selected_items),
                "contact_sheets": cs_info.get("photo_sheets", []),
            },
        )

        return {
            "photo_curation_path": str(out_path),
            "selected_photos": len(selected_items),
            "average_taste_score": avg_taste,
            "contact_sheets": cs_info.get("photo_sheets", []),
        }


def render_curation_contact_sheets(
    project_dir: str | Path,
    cols: int = 5,
    rows: int = 4,
    max_items: int = 60,
) -> dict[str, Any]:
    """Mechanically render numbered 5x4 contact sheets for photos and video subclips so the AI Agent can visually inspect them via read_file."""
    proj = Path(project_dir).resolve()
    pc_path = proj / "photo_curation.json"
    mi_path = proj / "media_index.json"
    cs_dir = proj / "contact_sheets"
    cs_dir.mkdir(parents=True, exist_ok=True)

    font_hdr = _get_cjk_font(14)
    font_sub = _get_cjk_font(12)

    photo_sheets: list[str] = []
    manifest_tiles: list[dict[str, Any]] = []

    if pc_path.exists():
        report = PhotoCurationReport.from_json_file(pc_path)
        selected = [it for it in report.items if it.is_selected]
        candidates = sorted(
            [
                it
                for it in report.items
                if not it.is_selected and it.rejection_reason != "near_duplicate_burst_frame"
            ],
            key=lambda x: x.overall_taste_score,
            reverse=True,
        )
        ordered_items = (selected + candidates)[:max_items]
        per_sheet = max(1, cols * rows)
        cell_w, cell_h = 360, 260
        pad = 12

        for sheet_idx in range(0, len(ordered_items), per_sheet):
            batch = ordered_items[sheet_idx : sheet_idx + per_sheet]
            sheet_num = (sheet_idx // per_sheet) + 1
            canvas_w = cols * cell_w + (cols + 1) * pad
            canvas_h = rows * cell_h + (rows + 1) * pad + 44
            canvas = Image.new("RGB", (canvas_w, canvas_h), (13, 17, 26))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (pad, 10),
                f"AI PHOTO CURATION CONTACT SHEET #{sheet_num:02d} — {report.project_title} (Green = Selected, Amber = Candidate)",
                fill=(255, 215, 64),
                font=font_hdr,
            )

            for local_i, item in enumerate(batch):
                global_num = sheet_idx + local_i + 1
                tile_code = f"#{global_num:02d}"
                r = local_i // cols
                c = local_i % cols
                x0 = pad + c * (cell_w + pad)
                y0 = 44 + pad + r * (cell_h + pad)

                border_col = (16, 185, 129) if item.is_selected else (245, 158, 11)
                draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=(22, 29, 43), outline=border_col, width=2)
                draw.rectangle([x0, y0, x0 + cell_w, y0 + 26], fill=(15, 21, 33))

                status_tag = f"★ SEL {item.overall_taste_score:.1f}" if item.is_selected else f"CAND {item.overall_taste_score:.1f}"
                header_text = f"[{tile_code}] {item.filename[:20]} • {status_tag}"
                draw.text((x0 + 6, y0 + 4), header_text, fill=border_col, font=font_hdr)

                img_box_w = cell_w - 12
                img_box_h = cell_h - 62
                try:
                    with Image.open(item.source_path) as im:
                        rgb = im.convert("RGB")
                        rgb.thumbnail((img_box_w, img_box_h), Image.Resampling.BILINEAR)
                        ix = x0 + 6 + (img_box_w - rgb.width) // 2
                        iy = y0 + 30 + (img_box_h - rgb.height) // 2
                        canvas.paste(rgb, (ix, iy))
                except Exception:
                    draw.text((x0 + 16, y0 + 90), "[Preview Unavailable]", fill=(148, 163, 184), font=font_hdr)

                caption_str = (item.annotation_title or item.subject_category)[:38]
                draw.rectangle([x0, y0 + cell_h - 28, x0 + cell_w, y0 + cell_h], fill=(15, 21, 33))
                draw.text((x0 + 6, y0 + cell_h - 24), caption_str, fill=(226, 232, 240), font=font_sub)

                manifest_tiles.append(
                    {
                        "tile_id": tile_code,
                        "tile_number": global_num,
                        "subclip_id": item.subclip_id,
                        "filename": item.filename,
                        "source_path": item.source_path,
                        "orientation": item.orientation,
                        "is_selected": item.is_selected,
                        "overall_taste_score": item.overall_taste_score,
                        "annotation_title": item.annotation_title,
                        "annotation_subtitle": item.annotation_subtitle,
                        "camera_badge": item.camera_badge,
                        "recommended_photo_mode": item.recommended_photo_mode,
                        "paired_with_subclip_id": item.paired_with_subclip_id,
                    }
                )

            sheet_path = cs_dir / f"contact_sheet_{sheet_num:02d}.jpg"
            canvas.save(sheet_path, quality=88)
            photo_sheets.append(str(sheet_path))

    # Also render a Video Subclips Keyframe Contact Sheet if video subclips exist
    video_sheet_path: Optional[str] = None
    video_tiles: list[dict[str, Any]] = []
    if mi_path.exists():
        mi = MediaIndex.from_json_file(mi_path)
        vid_subclips = [s for s in mi.subclips if s.media_type == "video" and not s.is_discarded][:20]
        if vid_subclips:
            v_cols = min(4, max(1, len(vid_subclips)))
            v_rows = (len(vid_subclips) + v_cols - 1) // v_cols
            cell_w, cell_h, pad = 420, 270, 12
            canvas_w = v_cols * cell_w + (v_cols + 1) * pad
            canvas_h = v_rows * cell_h + (v_rows + 1) * pad + 44
            v_canvas = Image.new("RGB", (canvas_w, canvas_h), (13, 17, 26))
            v_draw = ImageDraw.Draw(v_canvas)
            v_draw.text(
                (pad, 10),
                "VIDEO SUBCLIPS KEYFRAME CONTACT SHEET — A-Roll & B-Roll Selects",
                fill=(0, 229, 255),
                font=font_hdr,
            )
            for idx, vsc in enumerate(vid_subclips):
                v_code = f"#V{idx + 1:02d}"
                r = idx // v_cols
                c = idx % v_cols
                x0 = pad + c * (cell_w + pad)
                y0 = 44 + pad + r * (cell_h + pad)
                v_draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=(22, 29, 43), outline=(0, 229, 255), width=2)
                v_draw.rectangle([x0, y0, x0 + cell_w, y0 + 26], fill=(15, 21, 33))
                v_draw.text(
                    (x0 + 6, y0 + 4),
                    f"[{v_code}] {vsc.subclip_id} ({vsc.roll_type.upper()} • {vsc.duration:.1f}s) {vsc.filename[:16]}",
                    fill=(0, 229, 255),
                    font=font_hdr,
                )
                mid_tc = round((vsc.in_tc + vsc.out_tc) * 0.5, 2)
                tmp_jpg = cs_dir / f"_tmp_{vsc.subclip_id}.jpg"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        str(mid_tc),
                        "-i",
                        vsc.source_path,
                        "-frames:v",
                        "1",
                        "-q:v",
                        "3",
                        str(tmp_jpg),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if tmp_jpg.exists():
                    try:
                        with Image.open(tmp_jpg) as vim:
                            rgb = vim.convert("RGB")
                            rgb.thumbnail((cell_w - 12, cell_h - 62), Image.Resampling.BILINEAR)
                            v_canvas.paste(rgb, (x0 + 6, y0 + 30))
                    except Exception:
                        pass
                    try:
                        tmp_jpg.unlink()
                    except Exception:
                        pass
                quote_or_sum = (vsc.spoken_text or vsc.summary or "")[:44]
                v_draw.rectangle([x0, y0 + cell_h - 28, x0 + cell_w, y0 + cell_h], fill=(15, 21, 33))
                v_draw.text((x0 + 6, y0 + cell_h - 24), quote_or_sum, fill=(226, 232, 240), font=font_sub)
                video_tiles.append(
                    {
                        "tile_id": v_code,
                        "subclip_id": vsc.subclip_id,
                        "filename": vsc.filename,
                        "roll_type": vsc.roll_type,
                        "in_tc": vsc.in_tc,
                        "out_tc": vsc.out_tc,
                        "duration": vsc.duration,
                        "spoken_text": vsc.spoken_text,
                        "summary": vsc.summary,
                        "emotion_score": vsc.emotion_score,
                    }
                )
            v_sheet = cs_dir / "video_subclips_sheet.jpg"
            v_canvas.save(v_sheet, quality=88)
            video_sheet_path = str(v_sheet)

    manifest_path = cs_dir / "contact_sheets_manifest.json"
    photo_contact_sheets = [
        {
            "sheet_index": idx + 1,
            "sheet_path": sp,
            "items": [t for t in manifest_tiles if t.get("sheet_path") == sp] or manifest_tiles,
        }
        for idx, sp in enumerate(photo_sheets)
    ]
    manifest_doc = {
        "project_dir": str(proj),
        "photo_sheets": photo_sheets,
        "photo_contact_sheets": photo_contact_sheets,
        "video_sheet": video_sheet_path,
        "video_subclips_sheet": video_sheet_path,
        "photo_tiles": manifest_tiles,
        "video_tiles": video_tiles,
    }
    manifest_path.write_text(json.dumps(manifest_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_doc


def apply_vision_curation_overrides(
    project_dir: str | Path,
    overrides: list[dict[str, Any]] | dict[str, Any],
    curatorial_statement: Optional[str] = None,
) -> dict[str, Any]:
    """Mechanically persist the AI Agent's visual curation & annotation decisions into photo_curation.json, media_index.json, and timeline.yaml."""
    proj = Path(project_dir).resolve()
    pc_path = proj / "photo_curation.json"
    mi_path = proj / "media_index.json"
    tl_path = proj / "timeline.yaml"
    manifest_path = proj / "contact_sheets" / "contact_sheets_manifest.json"

    if not pc_path.exists():
        raise FileNotFoundError(f"photo_curation.json not found in {proj}")

    if isinstance(overrides, dict):
        if not curatorial_statement and overrides.get("curatorial_statement"):
            curatorial_statement = str(overrides["curatorial_statement"])
        override_list = list(
            overrides.get("overrides")
            or overrides.get("items")
            or overrides.get("annotations")
            or []
        )
    else:
        override_list = list(overrides)

    report = PhotoCurationReport.from_json_file(pc_path)
    tile_to_subclip: dict[str, str] = {}
    if manifest_path.exists():
        try:
            m_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
            for t in m_doc.get("photo_tiles", []):
                tile_to_subclip[str(t.get("tile_id", "")).upper()] = t["subclip_id"]
                tile_to_subclip[f"#{int(t.get('tile_number', 0)):02d}"] = t["subclip_id"]
                tile_to_subclip[str(t.get("tile_number", ""))] = t["subclip_id"]
        except Exception:
            pass

    item_by_subclip = {it.subclip_id: it for it in report.items}
    item_by_filename = {it.filename.lower(): it for it in report.items}

    explicit_selected_count = sum(
        1 for ov in override_list if isinstance(ov, dict) and (ov.get("is_selected") or ov.get("selected"))
    )
    exclusive_slate = (
        (isinstance(overrides, dict) and bool(overrides.get("exclusive_selection")))
        or explicit_selected_count >= 10
    )
    if exclusive_slate:
        for it in report.items:
            if it.is_selected:
                it.is_selected = False
                it.rejection_reason = "not_in_ai_curated_slate"

    applied_count = 0
    for ov in override_list:
        key_raw = str(
            ov.get("subclip_id")
            or ov.get("tile_id")
            or ov.get("id_or_filename")
            or ov.get("filename")
            or ""
        ).strip()
        if not key_raw:
            continue

        resolved_sid = tile_to_subclip.get(key_raw.upper(), key_raw)
        target_item = item_by_subclip.get(resolved_sid) or item_by_filename.get(key_raw.lower())
        if target_item is None:
            continue

        sel_val = ov.get("is_selected") if "is_selected" in ov else ov.get("selected")
        if sel_val is not None:
            target_item.is_selected = bool(sel_val)
            if target_item.is_selected:
                target_item.rejection_reason = None
            elif ov.get("rejection_reason") or ov.get("vision_note"):
                target_item.rejection_reason = str(ov.get("rejection_reason") or ov.get("vision_note"))

        new_title = ov.get("annotation_title") or ov.get("caption_title")
        if new_title:
            target_item.annotation_title = str(new_title)
        new_sub = ov.get("annotation_subtitle") or ov.get("caption_subtitle")
        if new_sub:
            target_item.annotation_subtitle = str(new_sub)
        if "camera_badge" in ov and ov["camera_badge"] is not None:
            target_item.camera_badge = str(ov["camera_badge"])
        if ov.get("subject_category"):
            target_item.subject_category = str(ov["subject_category"])
        if ov.get("overall_taste_score") is not None:
            target_item.overall_taste_score = float(ov["overall_taste_score"])
        new_mode = ov.get("recommended_photo_mode") or ov.get("photo_mode") or ov.get("mode")
        if new_mode in VALID_PHOTO_MODES:
            target_item.recommended_photo_mode = new_mode
        new_fx = ov.get("overlay_fx")
        if new_fx in VALID_PHOTO_OVERLAY_FX:
            target_item.overlay_fx = new_fx
        if "paired_with_subclip_id" in ov:
            target_item.paired_with_subclip_id = ov["paired_with_subclip_id"]
        if "tertiary_subclip_id" in ov:
            target_item.tertiary_subclip_id = ov["tertiary_subclip_id"]
        if ov.get("curation_rationale") or ov.get("vision_note") or ov.get("curatorial_note"):
            target_item.curation_rationale = str(
                ov.get("curation_rationale") or ov.get("vision_note") or ov.get("curatorial_note")
            )

        applied_count += 1

    selected_items = [it for it in report.items if it.is_selected]
    report.selected_count = len(selected_items)
    report.rejected_count = len(report.items) - len(selected_items)
    if selected_items:
        report.average_taste_score_selected = round(
            sum(it.overall_taste_score for it in selected_items) / len(selected_items), 2
        )
    if curatorial_statement:
        report.curatorial_statement = curatorial_statement
    report.to_json_file(pc_path)

    # Sync changes into media_index.json
    if mi_path.exists():
        mi = MediaIndex.from_json_file(mi_path)
        sc_by_id = {s.subclip_id: s for s in mi.subclips}
        absorbed_partners: set[str] = set()
        for it in selected_items:
            if it.paired_with_subclip_id and it.recommended_photo_mode in ("diptych_frame", "polaroid_scrapbook", "triptych_cinematic"):
                absorbed_partners.add(it.paired_with_subclip_id)
            if it.tertiary_subclip_id and it.recommended_photo_mode == "triptych_cinematic":
                absorbed_partners.add(it.tertiary_subclip_id)
        for it in report.items:
            sc = sc_by_id.get(it.subclip_id)
            if sc is None:
                continue
            sc.annotation_title = it.annotation_title
            sc.annotation_subtitle = it.annotation_subtitle
            sc.camera_badge = it.camera_badge
            sc.subject_category = it.subject_category
            sc.recommended_photo_mode = it.recommended_photo_mode
            sc.overlay_fx = it.overlay_fx
            sc.aesthetic_score = it.overall_taste_score
            sc.summary = f"{it.annotation_title} ({it.annotation_subtitle}) — {it.curation_rationale}"
            if it.is_selected and it.subclip_id not in absorbed_partners:
                sc.is_discarded = False
                sc.discard_reason = None
                if it.paired_with_subclip_id and it.paired_with_subclip_id in sc_by_id:
                    sc.secondary_source_path = sc_by_id[it.paired_with_subclip_id].source_path
                if it.tertiary_subclip_id and it.tertiary_subclip_id in sc_by_id:
                    sc.tertiary_source_path = sc_by_id[it.tertiary_subclip_id].source_path
            elif it.subclip_id in absorbed_partners:
                sc.is_discarded = True
                sc.discard_reason = "paired_as_secondary_or_tertiary_panel"
            else:
                sc.is_discarded = True
                sc.discard_reason = it.rejection_reason or "rejected_by_ai_vision_curator"
        mi.kept_selects_count = sum(1 for s in mi.subclips if not s.is_discarded)
        mi.discarded_count = sum(1 for s in mi.subclips if s.is_discarded)
        mi.to_json_file(mi_path)

    # Sync updated plaques & photo effects into timeline.yaml if already generated
    updated_timeline_cuts = 0
    if tl_path.exists():
        tl = EditTimeline.from_yaml(tl_path)
        by_src = {str(Path(it.source_path).resolve()).lower(): it for it in report.items}
        by_fname = {it.filename.lower(): it for it in report.items}
        for track in (tl.visual_track, tl.broll_track):
            for clip in track:
                if clip.type == "photo" and clip.ken_burns is not None:
                    src_key = str(Path(clip.source).resolve()).lower() if Path(clip.source).exists() else ""
                    fname_key = Path(clip.source).name.lower()
                    matched = by_src.get(src_key) or by_fname.get(fname_key)
                    if matched is not None:
                        clip.ken_burns.caption_title = matched.annotation_title
                        clip.ken_burns.caption_subtitle = matched.annotation_subtitle
                        clip.ken_burns.camera_badge = matched.camera_badge
                        clip.ken_burns.mode = matched.recommended_photo_mode
                        clip.ken_burns.overlay_fx = matched.overlay_fx
                        clip.narrative_note = f"[STILL · {matched.annotation_title}] {matched.annotation_subtitle}"
                        updated_timeline_cuts += 1
        tl.to_yaml(tl_path)

    return {
        "applied_overrides": applied_count,
        "selected_count": report.selected_count,
        "selected_photos": report.selected_count,
        "rejected_photos": report.rejected_count,
        "updated_timeline_cuts": updated_timeline_cuts,
        "average_taste_score_selected": report.average_taste_score_selected,
        "photo_curation_path": str(pc_path),
    }
