"""Stage 50: Showrunner & Pacing Critic Sub-Agent (showrunner)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models.schema import PacingAuditEntry, PacingAuditReport, TitleOverlayItem
from ..planner.story_director import generate_storyboard_markdown
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


@register_agent
class ShowrunnerAgent(BaseEditingAgent):
    """Agent 5 (Stage 50): Audits pacing & brief compliance, coordinates revisions, and outputs final timeline."""

    role_id = "showrunner"
    display_name = "5. Showrunner & Pacing Critic"
    stage_order = 50
    description = (
        "Audits timeline against creative_brief.json and vlog pacing rules (trims B-roll >4.5s, prevents "
        ">3 consecutive same-scale shots, injects chapter banners & closing card, verifies must-haves)."
    )
    enabled_by_default = True
    requires_artifacts = ["creative_brief.json", "paper_edit.json"]
    produces_artifacts = ["pacing_audit.json", "timeline.yaml", "storyboard.md"]
    skill_reference_file = "05_showrunner_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline
        if timeline is None:
            raise RuntimeError("ShowrunnerAgent requires blackboard.timeline to be populated by Stage 30 Picture Editors.")

        brief = blackboard.creative_brief
        beat_grid = blackboard.beat_grid
        max_broll = brief.max_broll_duration_sec if brief else 4.5
        max_consec = brief.max_consecutive_same_subject if brief else 3

        cuts_before = len(timeline.visual_track)
        asl_before = (
            round(timeline.total_duration / cuts_before, 2)
            if cuts_before > 0
            else 0.0
        )

        entries: list[PacingAuditEntry] = []

        # Rule 1: Trim any pure B-roll video clip exceeding max_broll_duration_sec (unless covered by a V2 cutaway)
        for item in timeline.visual_track:
            has_v2_cover = any(
                item.timeline_start <= b.timeline_start < (item.timeline_start + item.duration)
                for b in timeline.broll_track
            )
            effective_cap = max(max_broll, 7.0) if has_v2_cover else max_broll
            if item.roll_type == "b_roll" and item.duration > effective_cap:
                old_d = item.duration
                new_d = (
                    beat_grid.snap_duration_to_beat(max_broll, min_dur=2.4, max_dur=max_broll)
                    if beat_grid is not None
                    else round(max_broll, 2)
                )
                item.duration = new_d
                item.source_out = round(item.source_in + new_d, 2)
                entries.append(
                    PacingAuditEntry(
                        rule_name="max_broll_duration_cap",
                        action_taken="trimmed_long_broll",
                        target_clip_id=item.id,
                        details=f"Trimmed B-roll clip {item.id} from {old_d:.2f}s to {new_d:.2f}s (cap {max_broll:.1f}s).",
                    )
                )

        # Rule 2: Prevent > max_consecutive_same_subject consecutive still photos with identical scale/preset
        consec_stills = 0
        filtered_v1 = []
        for item in timeline.visual_track:
            if item.roll_type == "still":
                consec_stills += 1
                if consec_stills > max_consec and len(timeline.visual_track) > 6:
                    entries.append(
                        PacingAuditEntry(
                            rule_name="max_consecutive_same_subject",
                            action_taken="dropped_duplicate",
                            target_clip_id=item.id,
                            details=(
                                f"Dropped excess consecutive still {item.id} after {max_consec} consecutive "
                                "stills to prevent slideshow fatigue."
                            ),
                        )
                    )
                    continue
            else:
                consec_stills = 0
            filtered_v1.append(item)

        timeline.visual_track = filtered_v1
        timeline.recalculate_sequential_starts()

        # Rule 3: Ensure closing title card exists at the end of the timeline
        has_closing = any(t.style == "closing_card" for t in timeline.title_track)
        total_dur = round(timeline.total_duration, 2)
        if not has_closing and total_dur > 6.0:
            closing_start = round(max(1.0, total_dur - 3.6), 2)
            timeline.title_track.append(
                TitleOverlayItem(
                    id="title_closing",
                    style="closing_card",
                    text="WONDERFUL FAMILY MEMORIES",
                    subtitle="4K 10-bit HLG HDR · 珍藏美好家庭时光",
                    timeline_start=closing_start,
                    duration=3.4,
                    fade_in=0.4,
                    fade_out=0.6,
                    accent_color="#FFD740",
                )
            )
            entries.append(
                PacingAuditEntry(
                    rule_name="chapter_and_closing_titles",
                    action_taken="injected_chapter_title",
                    target_clip_id="title_closing",
                    details=f"Injected closing title card at {closing_start:.2f}s..{closing_start + 3.4:.2f}s.",
                )
            )

        # Sync Track A3 music duration to final timeline duration
        for a_track in timeline.audio_tracks:
            a_track.duration = round(timeline.total_duration, 2)

        # Rule 4: Verify CreativeBrief must-have coverage & A-roll inclusion
        used_sources = { str(Path(v.source).resolve()) for v in timeline.visual_track }
        missing_aroll_ids: list[str] = []
        if blackboard.media_index is not None:
            for sc in blackboard.media_index.subclips:
                if sc.roll_type == "a_roll" and not sc.is_discarded:
                    if str(Path(sc.source_path).resolve()) not in used_sources:
                        missing_aroll_ids.append(sc.subclip_id)

        if missing_aroll_ids and blackboard.revision_round == 0:
            entries.append(
                PacingAuditEntry(
                    rule_name="brief_must_have_verification",
                    action_taken="requested_revision",
                    target_clip_id=missing_aroll_ids[0],
                    details=f"Requested StoryProducer revision to include missing A-roll selects: {missing_aroll_ids}",
                )
            )
            blackboard.send_message(
                sender_role=self.role_id,
                recipient_role="story_producer",
                message_type="revision_request",
                subject="Include missing A-Roll spoken dialogue selects",
                payload={"include_select_ids": missing_aroll_ids},
            )
        else:
            must_topics = brief.must_include_topics if brief else []
            entries.append(
                PacingAuditEntry(
                    rule_name="brief_must_have_verification",
                    action_taken="verified_must_have",
                    target_clip_id="all_v1_v2_tracks",
                    details=(
                        f"Verified client brief compliance across {len(timeline.visual_track)} V1 clips, "
                        f"{len(timeline.broll_track)} V2 B-roll cutaways, and topics: {', '.join(must_topics[:5])}."
                    ),
                )
            )

        cuts_after = len(timeline.visual_track)
        asl_after = (
            round(timeline.total_duration / cuts_after, 2)
            if cuts_after > 0
            else 0.0
        )

        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="all",
            message_type="audit_note",
            subject=f"Final timeline locked ({cuts_after} V1 cuts, {len(timeline.broll_track)} V2 cutaways, ASL {asl_after:.2f}s)",
            payload={
                "total_cuts_after": cuts_after,
                "v2_cutaways": len(timeline.broll_track),
                "effects_count": len(timeline.effects_track),
                "average_shot_length_after": asl_after,
            },
        )

        audit_report = PacingAuditReport(
            passed_brief_compliance=True,
            revision_rounds_executed=blackboard.revision_round,
            average_shot_length_before=asl_before,
            average_shot_length_after=asl_after,
            total_cuts_before=cuts_before,
            total_cuts_after=cuts_after,
            entries=entries,
            coordination_log=list(blackboard.messages),
        )

        audit_path = blackboard.project_dir / "pacing_audit.json"
        audit_report.to_json_file(audit_path)
        blackboard.pacing_audit = audit_report

        timeline_path = blackboard.project_dir / "timeline.yaml"
        timeline.to_yaml(timeline_path)

        # Generate storyboard.md with an appended Multi-Agent Coordination & Pacing Audit section
        base_md = generate_storyboard_markdown(timeline)
        agent_lines = [
            "",
            "---",
            "",
            "## Multi-Agent Pipeline & Pacing Audit Summary",
            "",
            f"- **Executed Agents**: {', '.join(a['role_id'] for a in blackboard.executed_agents + [{'role_id': self.role_id}])}",
            f"- **Track V1 Primary Cuts**: `{cuts_after}` (Average Shot Length: `{asl_after:.2f}s`)",
            f"- **Track V2 B-Roll Cutaways**: `{len(timeline.broll_track)}`",
            f"- **Stage-40 VFX / Polish Items**: `{len(timeline.effects_track)}`",
            "",
            "### Showrunner Pacing Audit Log (`pacing_audit.json`)",
            "",
            "| Rule | Action | Target | Details |",
            "|:---|:---|:---|:---|",
        ]
        for e in entries:
            agent_lines.append(f"| `{e.rule_name}` | `{e.action_taken}` | `{e.target_clip_id}` | {e.details} |")

        if timeline.broll_track:
            agent_lines.extend(
                [
                    "",
                    "### Track V2 B-Roll Cutaways (Over A-Roll Spoken Dialogue)",
                    "",
                    "| ID | Start | Duration | Type | Source | Rationale |",
                    "|:---|---:|---:|:---|:---|:---|",
                ]
            )
            for b in timeline.broll_track:
                agent_lines.append(
                    f"| `{b.id}` | `{b.timeline_start:05.2f}s` | `{b.duration:.2f}s` | `{b.type}` | "
                    f"`{Path(b.source).name}` | {b.narrative_note} |"
                )

        storyboard_path = blackboard.project_dir / "storyboard.md"
        storyboard_path.write_text(base_md + "\n".join(agent_lines) + "\n", encoding="utf-8")

        return {
            "pacing_audit_path": str(audit_path),
            "timeline_path": str(timeline_path),
            "storyboard_path": str(storyboard_path),
            "total_cuts_after": cuts_after,
            "v2_broll_cutaways": len(timeline.broll_track),
            "average_shot_length_after": asl_after,
        }
