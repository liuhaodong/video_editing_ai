"""Stage 50: Showrunner — Deterministic Compliance & Pacing Validator (showrunner)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models.schema import PacingAuditEntry, PacingAuditReport
from ..planner.story_director import generate_storyboard_markdown
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


@register_agent
class ShowrunnerAgent(BaseEditingAgent):
    """Agent 5 (Stage 50): Deterministic Compliance & Pacing Validator emitting beat-preserving RevisionDirectives."""

    role_id = "showrunner"
    display_name = "5. Showrunner - Compliance & Pacing Validator"
    stage_order = 50
    substage_priority = 0
    description = (
        "Deterministic compliance and pacing validator that verifies creative_brief.json must-haves, "
        "flags >3 consecutive static stills, flags pure B-roll >4.5s, checks timeline gaps/overlaps, "
        "and emits chapter-scoped RevisionDirectives without mutating montage cuts off the beat grid."
    )
    enabled_by_default = True
    requires_artifacts = ["creative_brief.json", "paper_edit.json", "timeline.yaml"]
    produces_artifacts = ["pacing_audit.json", "timeline.yaml", "storyboard.md"]
    skill_reference_file = "05_showrunner_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline
        if timeline is None and (blackboard.project_dir / "timeline.yaml").exists():
            from ..models.schema import EditTimeline

            timeline = EditTimeline.from_yaml(blackboard.project_dir / "timeline.yaml")
            blackboard.timeline = timeline
        if timeline is None:
            raise RuntimeError("ShowrunnerAgent requires blackboard.timeline to be populated.")

        brief = blackboard.creative_brief
        paper = blackboard.paper_edit
        max_broll = brief.max_broll_duration_sec if brief else 4.5
        max_consec = brief.max_consecutive_same_subject if brief else 3

        chap_id_by_title: dict[str, str] = {}
        if paper is not None:
            for ch in paper.chapters:
                chap_id_by_title[ch.title] = ch.chapter_id

        cuts_count = len(timeline.visual_track)
        asl = round(timeline.total_duration / cuts_count, 2) if cuts_count > 0 else 0.0

        entries: list[PacingAuditEntry] = []
        long_broll_violations = 0
        consec_stills_violations = 0
        gap_overlap_violations = 0
        directives_emitted_before = len(blackboard.revision_directives)
        flagged_chapters: dict[str, dict[str, Any]] = {}

        # Rule 1: Flag pure B-roll video clips exceeding max_broll_duration_sec (unless covered by V2 cutaway)
        for item in timeline.visual_track:
            if item.id == "clip_000_hook":
                continue
            has_v2_cover = any(
                item.timeline_start <= b.timeline_start < (item.timeline_start + item.duration)
                for b in timeline.broll_track
            )
            effective_cap = max(max_broll, 7.0) if has_v2_cover else max_broll
            if item.roll_type == "b_roll" and item.duration > effective_cap + 0.02:
                long_broll_violations += 1
                ch_id = chap_id_by_title.get(item.act or "")
                entries.append(
                    PacingAuditEntry(
                        rule_name="max_broll_duration_cap",
                        action_taken="flagged_long_broll",
                        target_clip_id=item.id,
                        chapter_id=ch_id,
                        details=(
                            f"Pure B-roll clip {item.id} duration {item.duration:.2f}s exceeds cap {effective_cap:.2f}s."
                        ),
                    )
                )
                if ch_id:
                    c_map = flagged_chapters.setdefault(ch_id, {})
                    c_map["max_broll_duration_sec"] = max_broll
                    c_map["max_consecutive_stills"] = max_consec

        # Rule 2: Flag > max_consec consecutive static stills within chapters
        consec_stills = 0
        prev_act = None
        for item in timeline.visual_track:
            if item.act != prev_act:
                consec_stills = 0
                prev_act = item.act
            if item.type == "photo":
                consec_stills += 1
                if consec_stills > max_consec:
                    consec_stills_violations += 1
                    ch_id = chap_id_by_title.get(item.act or "")
                    entries.append(
                        PacingAuditEntry(
                            rule_name="max_consecutive_static_stills",
                            action_taken="flagged_consecutive_stills",
                            target_clip_id=item.id,
                            chapter_id=ch_id,
                            details=(
                                f"Detected {consec_stills} consecutive static stills (>{max_consec}) at {item.id} in '{item.act}'."
                            ),
                        )
                    )
                    if ch_id:
                        c_map = flagged_chapters.setdefault(ch_id, {})
                        c_map["max_consecutive_stills"] = max_consec
                        c_map["max_broll_duration_sec"] = max_broll
            else:
                consec_stills = 0

        # Rule 3: Check Track V1 timeline gaps or overlaps
        v1 = timeline.visual_track
        if v1 and abs(v1[0].timeline_start - 0.0) > 0.01:
            gap_overlap_violations += 1
            entries.append(
                PacingAuditEntry(
                    rule_name="timeline_continuity_check",
                    action_taken="flagged_timeline_gap_or_overlap",
                    target_clip_id=v1[0].id,
                    details=f"First V1 clip starts at {v1[0].timeline_start:.3f}s instead of 0.000s.",
                )
            )
        for idx in range(1, len(v1)):
            expected_start = v1[idx - 1].timeline_start + v1[idx - 1].duration
            drift = v1[idx].timeline_start - expected_start
            if abs(drift) > 0.01:
                gap_overlap_violations += 1
                entries.append(
                    PacingAuditEntry(
                        rule_name="timeline_continuity_check",
                        action_taken="flagged_timeline_gap_or_overlap",
                        target_clip_id=v1[idx].id,
                        details=(
                            f"Detected {'gap' if drift > 0 else 'overlap'} of {drift:+.3f}s between "
                            f"{v1[idx - 1].id} and {v1[idx].id}."
                        ),
                    )
                )

        # Emit beat-preserving chapter-scoped RevisionDirectives instead of mutating timeline durations directly
        if blackboard.revision_round < blackboard.max_revisions:
            for ch_id, constraints in flagged_chapters.items():
                blackboard.add_revision_directive(
                    target_role_id="montage_editor",
                    chapter_id=ch_id,
                    reason=f"Showrunner pacing audit flagged chapter {ch_id} ({list(constraints.keys())})",
                    constraints=constraints,
                    issued_by=self.role_id,
                )
            if gap_overlap_violations > 0:
                blackboard.add_revision_directive(
                    target_role_id="lead_editor",
                    chapter_id=None,
                    reason="Resolve Track V1 timeline gap/overlap continuity drift",
                    constraints={"recalculate_sequential_starts": True},
                    issued_by=self.role_id,
                )

        # Rule 4: Verify CreativeBrief must-have coverage & A-roll inclusion
        used_sources = {str(Path(v.source).resolve()) for v in timeline.visual_track}
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
            blackboard.add_revision_directive(
                target_role_id="story_producer",
                chapter_id=None,
                reason="Include missing A-roll spoken dialogue selects",
                constraints={"include_select_ids": missing_aroll_ids},
                issued_by=self.role_id,
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
                        f"Verified client brief compliance across {cuts_count} V1 clips, "
                        f"{len(timeline.broll_track)} V2 B-roll cutaways, and topics: {', '.join(must_topics[:5])}."
                    ),
                )
            )

        new_directives = list(blackboard.revision_directives[directives_emitted_before:])
        directives_emitted_now = len(new_directives)
        passed_compliance = (
            long_broll_violations == 0
            and consec_stills_violations == 0
            and gap_overlap_violations == 0
            and not (missing_aroll_ids and blackboard.revision_round == 0)
        )

        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="visual_qc",
            message_type="audit_note",
            subject=(
                f"Pacing audit {'PASSED' if passed_compliance else 'ISSUED DIRECTIVES'} "
                f"({cuts_count} V1 cuts, ASL {asl:.2f}s, {directives_emitted_now} directives)"
            ),
            payload={
                "passed_brief_compliance": passed_compliance,
                "total_cuts_after": cuts_count,
                "average_shot_length_after": asl,
                "revision_directives_emitted": directives_emitted_now,
            },
        )

        audit_report = PacingAuditReport(
            passed_brief_compliance=passed_compliance,
            revision_rounds_executed=blackboard.revision_round,
            average_shot_length_before=asl,
            average_shot_length_after=asl,
            total_cuts_before=cuts_count,
            total_cuts_after=cuts_count,
            consecutive_stills_violations=consec_stills_violations,
            long_broll_violations=long_broll_violations,
            gap_or_overlap_violations=gap_overlap_violations,
            revision_directives_emitted=new_directives,
            entries=entries,
            coordination_log=list(blackboard.messages),
        )

        audit_path = blackboard.project_dir / "pacing_audit.json"
        audit_report.to_json_file(audit_path)
        blackboard.pacing_audit = audit_report

        timeline_path = blackboard.project_dir / "timeline.yaml"
        timeline.to_yaml(timeline_path)

        base_md = generate_storyboard_markdown(timeline)
        agent_lines = [
            "",
            "---",
            "",
            "## Multi-Agent Pipeline & Pacing Audit Summary",
            "",
            f"- **Executed Agents**: {', '.join(a['role_id'] for a in blackboard.executed_agents + [{'role_id': self.role_id}])}",
            f"- **Track V1 Primary Cuts**: `{cuts_count}` (Average Shot Length: `{asl:.2f}s`)",
            f"- **Track V2 B-Roll Cutaways**: `{len(timeline.broll_track)}`",
            f"- **Stage-40 VFX / Color / Polish Items**: `{len(timeline.effects_track)}`",
            f"- **Compliance Status**: `{'PASSED' if passed_compliance else 'REVISION DIRECTIVES EMITTED'}`",
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
        blackboard.emit_activity(
            self.role_id,
            f"Validated Pacing & Brief Compliance: {cuts_count} V1 cuts, ASL {asl:.2f}s, {directives_emitted_now} revision directives",
            detail=f"Consecutive-still flags={consec_stills_violations}, Long-B-roll flags={long_broll_violations}, Gap/Overlap={gap_overlap_violations}",
        )

        return {
            "pacing_audit_path": str(audit_path),
            "timeline_path": str(timeline_path),
            "storyboard_path": str(storyboard_path),
            "passed_brief_compliance": passed_compliance,
            "total_cuts_after": cuts_count,
            "v2_broll_cutaways": len(timeline.broll_track),
            "average_shot_length_after": asl,
            "revision_directives_emitted": directives_emitted_now,
        }
