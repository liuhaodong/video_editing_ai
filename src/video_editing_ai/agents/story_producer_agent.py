"""Stage 20: Story Producer Sub-Agent (story_producer)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..models.schema import ChapterPlan, CulledSubClip, PaperEdit
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


@register_agent
class StoryProducerAgent(BaseEditingAgent):
    """Agent 2 (Stage 20): Clusters selects into vignette chapters and builds paper_edit.json."""

    role_id = "story_producer"
    display_name = "2. Story Producer (Day-in-the-Life Architect)"
    stage_order = 20
    description = (
        "Clusters culled selects by time and location into 4 vignette chapters, switches between "
        "dialogue_with_broll and music_montage modes, and maps Track V2 B-roll visual anchors."
    )
    enabled_by_default = True
    requires_artifacts = ["creative_brief.json", "media_index.json"]
    produces_artifacts = ["paper_edit.json"]
    skill_reference_file = "02_story_producer_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        brief = blackboard.creative_brief
        media_index = blackboard.media_index
        if media_index is None:
            raise RuntimeError("StoryProducerAgent requires blackboard.media_index to be populated by CullerAgent.")

        active_selects = [s for s in media_index.subclips if not s.is_discarded]
        videos = [s for s in active_selects if s.media_type == "video"]
        photos = [s for s in active_selects if s.media_type == "photo"]

        forced_select_ids: set[str] = set()
        min_aroll_cutaway_dur = 8.5
        for msg in blackboard.get_messages_for(self.role_id):
            if msg.message_type in ("revision_request", "director_directive"):
                for sid in msg.payload.get("include_select_ids", []):
                    forced_select_ids.add(str(sid))
                if "min_aroll_cutaway_duration" in msg.payload:
                    min_aroll_cutaway_dur = float(msg.payload["min_aroll_cutaway_duration"])

        for s in media_index.subclips:
            if s.subclip_id in forced_select_ids and s not in active_selects:
                s.is_discarded = False
                if s.media_type == "video":
                    videos.append(s)
                else:
                    photos.append(s)

        prompt_text = (brief.raw_user_prompt if brief else blackboard.user_prompt) or ""
        all_text_lower = (
            prompt_text.lower()
            + " "
            + " ".join(s.spoken_text.lower() + " " + s.summary.lower() for s in active_selects)
        )
        is_halloween_park = any(
            k in all_text_lower for k in ("halloween", "gilroy", "pumpkin", "trick", "万圣", "南瓜")
        )
        is_louvre_paris = any(
            k in all_text_lower
            for k in ("louvre", "paris", "pyramid", "museum", "卢浮宫", "巴黎")
        ) or "louvre" in str(blackboard.media_dir).lower()

        if is_louvre_paris:
            chapter_headers = [
                (
                    "chap_01",
                    "Act I - Arrival at the Louvre Pyramid",
                    "卢浮宫金字塔 · 晨光初见",
                    "Cour Napoleon & Glass Pyramid",
                    "morning",
                    "Iconic arrival at I. M. Pei's glass pyramid and sunlit palace courtyard.",
                ),
                (
                    "chap_02",
                    "Act II - Grand Galleries & Masterpieces",
                    "艺术长廊 · 经典名作",
                    "Denon & Sully Galleries",
                    "afternoon",
                    "Exploring gilded ceilings, Renaissance masterpieces, and family awe.",
                ),
                (
                    "chap_03",
                    "Act III - Sculptures & Palace Courtyards",
                    "雕塑庭院 · 光影漫步",
                    "Richelieu Wing & Marly Courtyard",
                    "afternoon",
                    "Marble sculptures under vaulted glass roofs and candid family portraits.",
                ),
                (
                    "chap_04",
                    "Act IV - Parisian Memories & Finale",
                    "巴黎印记 · 难忘一日",
                    "Louvre Palace Finale",
                    "twilight",
                    "Golden-hour reflections, keepsake portraits, and warm closing moments in Paris.",
                ),
            ]
        elif is_halloween_park:
            chapter_headers = [
                (
                    "chap_01",
                    "Act I - Entering the Enchanted Night",
                    "步入奇幻万圣夜 · 欢声启程",
                    "Park Entrance",
                    "twilight",
                    "Opening hook at the glowing park entrance with family excitement.",
                ),
                (
                    "chap_02",
                    "Act II - Whirling Rides & Joyful Thrills",
                    "欢笑与旋转的奇遇 · 游乐时光",
                    "Adventure Rides",
                    "twilight",
                    "Dynamic action rides, spinning tubs, and candid kid reactions.",
                ),
                (
                    "chap_03",
                    "Act III - The Great Pumpkin Pyramid",
                    "南瓜金字塔的温暖记忆 · 经典打卡",
                    "Pumpkin Pyramid Plaza",
                    "night",
                    "Iconic Halloween landmarks, glowing displays, and keepsake family portraits.",
                ),
                (
                    "chap_04",
                    "Act IV - Under the Neon Stage Lights",
                    "霓虹舞台下的终章 · 完美谢幕",
                    "Main Neon Stage",
                    "night",
                    "High-energy stage finale, music celebration, and warm closing reflection.",
                ),
            ]
        else:
            chapter_headers = [
                (
                    "chap_01",
                    "Act I - The Journey Begins",
                    "启程 · 第一印象",
                    "Arrival",
                    "morning",
                    "Establishing the destination and opening family excitement.",
                ),
                (
                    "chap_02",
                    "Act II - Exploring Together",
                    "探索 · 欢乐奇遇",
                    "Adventure Zone",
                    "afternoon",
                    "Candid adventures, motion, and playful family moments.",
                ),
                (
                    "chap_03",
                    "Act III - Golden Memories",
                    "定格 · 温暖时光",
                    "Iconic Landmarks",
                    "twilight",
                    "Heartwarming portraits and iconic highlights from the trip.",
                ),
                (
                    "chap_04",
                    "Act IV - Grand Finale",
                    "终章 · 难忘回忆",
                    "Evening Finale",
                    "night",
                    "Climax celebration and closing family reflection.",
                ),
            ]

        num_chapters = min(4, max(1, len(videos) + (1 if photos else 0)))
        chapter_headers = chapter_headers[:num_chapters]

        v_buckets: list[list[CulledSubClip]] = [[] for _ in range(num_chapters)]
        p_buckets: list[list[CulledSubClip]] = [[] for _ in range(num_chapters)]

        for idx, v in enumerate(videos):
            b_idx = min(num_chapters - 1, int(idx * num_chapters / max(1, len(videos))))
            v_buckets[b_idx].append(v)

        for idx, p in enumerate(photos):
            b_idx = min(num_chapters - 1, int(idx * num_chapters / max(1, len(photos))))
            p_buckets[b_idx].append(p)

        chapters: list[ChapterPlan] = []
        used_as_v2_cutaway: set[str] = set()

        for c_idx in range(num_chapters):
            cid, title_en, sub_zh, loc_tag, tod, summary = chapter_headers[c_idx]
            c_vids = v_buckets[c_idx]
            c_photos = p_buckets[c_idx]

            # Build semantically matched Track V2 B-Roll cutaway anchor map for A-roll clips
            broll_anchor_map: dict[str, list[str]] = {}
            for v in c_vids:
                eligible_for_cutaway = (
                    (v.roll_type == "a_roll" and v.duration >= min_aroll_cutaway_dur and len(c_photos) >= 1)
                    or (v.duration >= 6.0 and len(c_photos) >= 3 and not broll_anchor_map)
                )
                if eligible_for_cutaway:
                    v_words = set(
                        w.lower()
                        for w in (v.spoken_text + " " + (v.annotation_title or "") + " " + v.summary).split()
                        if len(w) >= 4
                    )
                    avail_cutaways = [
                        p
                        for p in c_photos
                        if p.subclip_id not in used_as_v2_cutaway and not p.secondary_source_path
                    ]
                    cutaway_candidate = None
                    if avail_cutaways:
                        cutaway_candidate = max(
                            avail_cutaways,
                            key=lambda p: (
                                sum(
                                    1
                                    for w in v_words
                                    if w in ((p.annotation_title or "") + " " + p.summary).lower()
                                ),
                                p.aesthetic_score,
                            ),
                        )
                    if cutaway_candidate is not None:
                        broll_anchor_map[v.subclip_id] = [cutaway_candidate.subclip_id]
                        used_as_v2_cutaway.add(cutaway_candidate.subclip_id)

            primary_photos = [p for p in c_photos if p.subclip_id not in used_as_v2_cutaway]
            # Sort primary photos so wide establishing landmarks lead, followed by medium/closeup portraits & details
            scale_rank = {"wide": 0, "medium": 1, "closeup": 2}
            primary_photos.sort(
                key=lambda p: (
                    scale_rank.get(p.shot_scale, 1),
                    p.normalized_timestamp,
                )
            )
            ordered_selects: list[CulledSubClip] = []
            p_cursor = 0
            for v in c_vids:
                ordered_selects.append(v)
                take_n = 2 if len(primary_photos) - p_cursor >= 2 else max(0, len(primary_photos) - p_cursor)
                for _ in range(take_n):
                    ordered_selects.append(primary_photos[p_cursor])
                    p_cursor += 1
            while p_cursor < len(primary_photos):
                ordered_selects.append(primary_photos[p_cursor])
                p_cursor += 1

            has_aroll = any(s.roll_type == "a_roll" for s in c_vids)
            editing_mode = "dialogue_with_broll" if has_aroll else "music_montage"

            chapters.append(
                ChapterPlan(
                    chapter_id=cid,
                    title=title_en,
                    subtitle=sub_zh,
                    location_tag=loc_tag,
                    time_of_day=tod,  # type: ignore[arg-type]
                    editing_mode=editing_mode,  # type: ignore[arg-type]
                    photo_integration_mode="ken_burns_pause" if has_aroll else "beat_burst",
                    selected_subclip_ids=[s.subclip_id for s in ordered_selects],
                    broll_anchor_map=broll_anchor_map,
                    narrative_goal=summary,
                )
            )

        # Mechanically export candidate_story_digest.json so the AI Agent can review transcripts & peaks
        digest_doc = {
            "project_title": brief.project_title if brief else "Family_Travel_Story",
            "user_prompt": prompt_text,
            "video_candidates": [
                {
                    "subclip_id": v.subclip_id,
                    "filename": v.filename,
                    "roll_type": v.roll_type,
                    "duration": v.duration,
                    "action_peak_tc": v.action_peak_tc,
                    "emotion_score": v.emotion_score,
                    "aesthetic_score": v.aesthetic_score,
                    "spoken_text": v.spoken_text,
                    "summary": v.summary,
                }
                for v in videos
            ],
            "photo_candidates": [
                {
                    "subclip_id": p.subclip_id,
                    "filename": p.filename,
                    "annotation_title": p.annotation_title,
                    "annotation_subtitle": p.annotation_subtitle,
                    "aesthetic_score": p.aesthetic_score,
                }
                for p in photos
            ],
        }
        (blackboard.project_dir / "candidate_story_digest.json").write_text(
            json.dumps(digest_doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Select 3.0s-3.8s Cold-Open Teaser Hook (if enabled in CreativeBrief)
        want_cold_open = brief.cold_open_hook if brief is not None else True
        if str(blackboard.interview_answers.get("cold_open_hook", "")).lower() in ("false", "0", "no"):
            want_cold_open = False

        cold_open_sid: Optional[str] = None
        cold_open_in: Optional[float] = None
        cold_open_out: Optional[float] = None
        cold_open_caption = ""
        cold_open_rationale = ""

        if want_cold_open and videos:
            hook_pool = videos[1:] if len(videos) > 1 else videos
            best_hook = max(
                hook_pool,
                key=lambda s: s.emotion_score + s.aesthetic_score + (1.2 if s.spoken_text else 0.0),
            )
            cold_open_sid = best_hook.subclip_id
            hook_dur = min(3.4, max(2.2, round(best_hook.duration * 0.45, 2)))
            peak_center = best_hook.action_peak_tc if best_hook.action_peak_tc > 0 else (best_hook.in_tc + best_hook.out_tc) * 0.5
            cold_open_in = round(max(best_hook.in_tc, min(best_hook.out_tc - hook_dur, peak_center - hook_dur * 0.5)), 2)
            cold_open_out = round(min(best_hook.out_tc, cold_open_in + hook_dur), 2)
            if best_hook.spoken_text:
                first_clause = best_hook.spoken_text.split(".")[0].strip()
                cold_open_caption = f"“{first_clause[:56]}”"
            else:
                cold_open_caption = best_hook.annotation_title or best_hook.summary[:52] or "COMING UP IN THIS STORY..."
            cold_open_rationale = (
                f"Selected {best_hook.subclip_id} ({best_hook.filename}, emotion {best_hook.emotion_score}/10) "
                "as a 3.4s Cold-Open Teaser Hook before the Main Title Card drop to maximize viewer retention."
            )

        ag_plan = self.ask_ai_creative_json(
            blackboard=blackboard,
            prompt=(
                "You are the Story Producer (Stage 20). Review this candidate_story_digest.json and return JSON with optional "
                "keys: cold_open_subclip_id, cold_open_hook_caption, cold_open_rationale, story_arc_summary, chapter_titles "
                "(list of {title, subtitle, narrative_goal}), or chapters (list of chapter overrides).\n\n"
                + json.dumps(digest_doc, ensure_ascii=False)[:4200]
            ),
            decision_key="story_producer",
            timeout_sec=35.0,
        )
        story_arc_summary = (
            "Cold-Open Teaser Hook followed by a four-chapter Day-in-the-Life narrative arc alternating between "
            "spoken A-Roll story anchors (with Track V2 B-Roll cutaways) and AI-choreographed 10-mode photo galleries."
        )
        if isinstance(ag_plan, dict):
            if ag_plan.get("cold_open_subclip_id") in {v.subclip_id for v in videos}:
                cold_open_sid = str(ag_plan["cold_open_subclip_id"])
            if ag_plan.get("cold_open_hook_caption"):
                cold_open_caption = str(ag_plan["cold_open_hook_caption"])
            if ag_plan.get("cold_open_rationale"):
                cold_open_rationale = str(ag_plan["cold_open_rationale"])
            if ag_plan.get("story_arc_summary"):
                story_arc_summary = str(ag_plan["story_arc_summary"])
            ag_titles = ag_plan.get("chapter_titles")
            if isinstance(ag_titles, list):
                for idx_t, t_obj in enumerate(ag_titles[: len(chapters)]):
                    if isinstance(t_obj, dict):
                        if t_obj.get("title"):
                            chapters[idx_t].title = str(t_obj["title"])
                        if t_obj.get("subtitle"):
                            chapters[idx_t].subtitle = str(t_obj["subtitle"])
                        if t_obj.get("narrative_goal"):
                            chapters[idx_t].narrative_goal = str(t_obj["narrative_goal"])

        paper_edit = PaperEdit(
            project_title=brief.project_title if brief else "Family_Travel_Story",
            story_arc_summary=story_arc_summary,
            cold_open_subclip_id=cold_open_sid,
            cold_open_in_tc=cold_open_in,
            cold_open_out_tc=cold_open_out,
            cold_open_hook_caption=cold_open_caption,
            cold_open_rationale=cold_open_rationale,
            chapters=chapters,
        )

        out_path = blackboard.project_dir / "paper_edit.json"
        paper_edit.to_json_file(out_path)
        if isinstance(ag_plan, dict) and isinstance(ag_plan.get("chapters"), list):
            try:
                apply_story_plan_overrides(blackboard.project_dir, ag_plan)
                paper_edit = PaperEdit.from_json_file(out_path)
            except Exception:
                pass
        blackboard.paper_edit = paper_edit
        hook_note = f" + Cold-Open Hook ({cold_open_sid})" if cold_open_sid else ""
        blackboard.emit_activity(
            self.role_id,
            f"Architected {len(paper_edit.chapters)}-Act Paper Edit{hook_note} with {sum(len(c.selected_subclip_ids) for c in paper_edit.chapters)} sequenced story beats",
            detail="Chapters: " + " | ".join(f"{c.title} ({len(c.selected_subclip_ids)} clips)" for c in paper_edit.chapters),
        )

        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="dialogue_editor",
            message_type="handoff",
            subject="Paper edit ready with Cold-Open Hook, dialogue_with_broll chapters & V2 cutaway anchors",
            payload={
                "cold_open_subclip_id": cold_open_sid,
                "chapters": len(paper_edit.chapters),
                "v2_cutaway_anchors": sum(len(c.broll_anchor_map) for c in paper_edit.chapters),
            },
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="montage_editor",
            message_type="handoff",
            subject="Paper edit ready for beat-synced B-roll and photo montage assembly",
            payload={"chapters": len(paper_edit.chapters), "cold_open_subclip_id": cold_open_sid},
        )

        return {
            "paper_edit_path": str(out_path),
            "cold_open_subclip_id": cold_open_sid,
            "chapter_count": len(paper_edit.chapters),
            "v2_cutaway_count": sum(len(c.broll_anchor_map) for c in paper_edit.chapters),
        }


def apply_story_plan_overrides(
    project_dir: str | Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Mechanically apply AI-directed Cold-Open Hook and chapter edits to paper_edit.json."""
    proj = Path(project_dir).resolve()
    pe_path = proj / "paper_edit.json"
    if not pe_path.exists():
        raise FileNotFoundError(f"paper_edit.json not found in {proj}")

    pe = PaperEdit.from_json_file(pe_path)
    if "cold_open_subclip_id" in overrides:
        pe.cold_open_subclip_id = overrides["cold_open_subclip_id"]
    if "cold_open_in_tc" in overrides:
        pe.cold_open_in_tc = float(overrides["cold_open_in_tc"]) if overrides["cold_open_in_tc"] is not None else None
    if "cold_open_out_tc" in overrides:
        pe.cold_open_out_tc = float(overrides["cold_open_out_tc"]) if overrides["cold_open_out_tc"] is not None else None
    if "cold_open_hook_caption" in overrides:
        pe.cold_open_hook_caption = str(overrides["cold_open_hook_caption"] or "")
    if "cold_open_rationale" in overrides:
        pe.cold_open_rationale = str(overrides["cold_open_rationale"] or "")
    if overrides.get("story_arc_summary"):
        pe.story_arc_summary = str(overrides["story_arc_summary"])

    chap_overrides = overrides.get("chapters")
    if isinstance(chap_overrides, list):
        chap_by_id = {c.chapter_id: c for c in pe.chapters}
        for idx, cov in enumerate(chap_overrides):
            if not isinstance(cov, dict):
                continue
            cid = cov.get("chapter_id") or (pe.chapters[idx].chapter_id if idx < len(pe.chapters) else None)
            target_c = chap_by_id.get(cid) if cid else None
            if target_c is None:
                continue
            if cov.get("title"):
                target_c.title = str(cov["title"])
            if "subtitle" in cov:
                target_c.subtitle = str(cov["subtitle"] or "")
            if cov.get("narrative_goal"):
                target_c.narrative_goal = str(cov["narrative_goal"])
            if isinstance(cov.get("selected_subclip_ids"), list):
                target_c.selected_subclip_ids = [str(x) for x in cov["selected_subclip_ids"]]
            if isinstance(cov.get("broll_anchor_map"), dict):
                target_c.broll_anchor_map = {
                    str(k): [str(v) for v in val] for k, val in cov["broll_anchor_map"].items() if isinstance(val, list)
                }

    pe.to_json_file(pe_path)
    return {
        "paper_edit_path": str(pe_path),
        "cold_open_subclip_id": pe.cold_open_subclip_id,
        "cold_open_hook_caption": pe.cold_open_hook_caption,
        "chapters_count": len(pe.chapters),
    }
