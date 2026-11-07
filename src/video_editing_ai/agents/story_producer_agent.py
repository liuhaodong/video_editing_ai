"""Stage 20: Story Producer Sub-Agent (story_producer)."""

from __future__ import annotations

from typing import Any

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
        for msg in blackboard.get_messages_for(self.role_id):
            if msg.message_type == "revision_request":
                for sid in msg.payload.get("include_select_ids", []):
                    forced_select_ids.add(str(sid))

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

        if is_halloween_park:
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

            # Build Track V2 B-Roll cutaway anchor map for long A-roll clips (>= 8.5s)
            broll_anchor_map: dict[str, list[str]] = {}
            for v in c_vids:
                if v.roll_type == "a_roll" and v.duration >= 8.5 and len(c_photos) >= 2:
                    cutaway_candidate = next(
                        (p for p in c_photos if p.subclip_id not in used_as_v2_cutaway),
                        None,
                    )
                    if cutaway_candidate is not None:
                        broll_anchor_map[v.subclip_id] = [cutaway_candidate.subclip_id]
                        used_as_v2_cutaway.add(cutaway_candidate.subclip_id)

            primary_photos = [p for p in c_photos if p.subclip_id not in used_as_v2_cutaway]
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

        paper_edit = PaperEdit(
            project_title=brief.project_title if brief else "Family_Travel_Story",
            story_arc_summary=(
                "Four-chapter Day-in-the-Life narrative arc alternating between spoken A-Roll story anchors "
                "(with Track V2 B-Roll cutaways) and beat-synced Gallery Frame photo montages."
            ),
            chapters=chapters,
        )

        out_path = blackboard.project_dir / "paper_edit.json"
        paper_edit.to_json_file(out_path)
        blackboard.paper_edit = paper_edit

        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="dialogue_editor",
            message_type="handoff",
            subject="Paper edit ready with dialogue_with_broll chapters & V2 cutaway anchors",
            payload={
                "chapters": len(paper_edit.chapters),
                "v2_cutaway_anchors": sum(len(c.broll_anchor_map) for c in paper_edit.chapters),
            },
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="montage_editor",
            message_type="handoff",
            subject="Paper edit ready for beat-synced B-roll and photo montage assembly",
            payload={"chapters": len(paper_edit.chapters)},
        )

        return {
            "paper_edit_path": str(out_path),
            "chapter_count": len(paper_edit.chapters),
            "v2_cutaway_count": sum(len(c.broll_anchor_map) for c in paper_edit.chapters),
        }
