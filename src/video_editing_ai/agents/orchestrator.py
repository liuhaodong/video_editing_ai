"""Multi-Agent Pipeline Orchestrator with Beat-Preserving Chapter Revision Loops (max_revisions=2)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..exporters.nle_export import export_all_nle_formats
from ..models.schema import DirectorStageReview
from ..renderer.ffmpeg_composer import render_timeline
from ..renderer.subtitles import export_ass_subtitles, export_srt_subtitles
from . import (
    audio_vibe_agent as _audio_vibe,  # noqa: F401
    cast_continuity_agent as _cast_tracker,  # noqa: F401
    culler_agent as _culler,  # noqa: F401
    director_agent as _director,  # noqa: F401
    dit_ingest_agent as _dit_ingest,  # noqa: F401
    intake_agent as _intake,  # noqa: F401
    lead_editor_agent as _lead_editor,  # noqa: F401
    music_supervisor_agent as _music_sup,  # noqa: F401
    nle_export_agent as _nle_exporter,  # noqa: F401
    photo_curator_agent as _photo_curator,  # noqa: F401
    picture_editors as _editors,  # noqa: F401
    showrunner_agent as _showrunner,  # noqa: F401
    story_producer_agent as _story_prod,  # noqa: F401
    visual_qc_agent as _visual_qc,  # noqa: F401
)
from .base import AgentRegistry, BaseEditingAgent, PipelineBlackboard
from .director_agent import ProductionDirectorAgent
from .plugins import (
    camera_operator_agent as _camera_op,  # noqa: F401
    colorist_agent as _colorist,  # noqa: F401
    motion_graphics_agent as _motion_gfx,  # noqa: F401
    narrator_agent as _narrator,  # noqa: F401
    sound_designer_agent as _sound_des,  # noqa: F401
    vfx_agent as _vfx,  # noqa: F401
)


class MultiAgentOrchestrator:
    """Executes registered video editing sub-agents in stage_order under Production Director supervision."""

    def __init__(
        self,
        enabled_optional_roles: Optional[list[str]] = None,
        disabled_roles: Optional[list[str]] = None,
    ) -> None:
        self.enabled_optional_roles = list(enabled_optional_roles or [])
        self.disabled_roles = list(disabled_roles or [])

    def get_execution_plan(self) -> list[BaseEditingAgent]:
        return AgentRegistry.resolve_execution_plan(
            enabled_optional_roles=self.enabled_optional_roles,
            disabled_roles=self.disabled_roles,
        )

    def run(
        self,
        blackboard: PipelineBlackboard,
        render_video: bool = False,
        export_fcpxml: bool = True,
        output_video_name: str = "final_story.mp4",
        progress_callback: Optional[Callable[[str, str, int], None]] = None,
        event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> PipelineBlackboard:
        plan = self.get_execution_plan()
        if not export_fcpxml:
            plan = [a for a in plan if a.role_id != "nle_exporter"]

        pre_export_plan = [a for a in plan if a.stage_order <= 60]
        post_export_plan = [a for a in plan if a.stage_order > 60]
        total_steps = max(1, len(plan))

        director: Optional[ProductionDirectorAgent] = next(
            (a for a in pre_export_plan if isinstance(a, ProductionDirectorAgent)),
            None,
        )
        stage_reviews: list[DirectorStageReview] = []
        setattr(blackboard, "_director_stage_reviews", stage_reviews)

        if event_callback is not None:
            def _on_bb_activity(entry: dict[str, Any]) -> None:
                event_callback(
                    {
                        "event": entry.get("event_type", "agent_action"),
                        **entry,
                    }
                )

            blackboard.activity_callback = _on_bb_activity

        agent_pct_span = 75 if render_video else 95

        def _execute_single_agent(agent: BaseEditingAgent, step_idx: int) -> None:
            nonlocal total_steps
            pct = int(round((step_idx / max(1, total_steps)) * agent_pct_span))
            t0 = time.monotonic()
            if progress_callback:
                progress_callback(agent.role_id, f"Running {agent.display_name}...", pct)
            if event_callback:
                event_callback(
                    {
                        "event": "stage_started",
                        "role_id": agent.role_id,
                        "display_name": agent.display_name,
                        "stage_order": agent.stage_order,
                        "progress_pct": pct,
                        "message": f"Running {agent.display_name}...",
                    }
                )

            summary = agent.run(blackboard)
            dur_sec = round(time.monotonic() - t0, 2)
            blackboard.executed_agents.append(
                {
                    "role_id": agent.role_id,
                    "display_name": agent.display_name,
                    "stage_order": agent.stage_order,
                    "round": blackboard.revision_round,
                    "summary": summary,
                }
            )

            review_obj: Optional[DirectorStageReview] = None
            if director is not None and agent.role_id != director.role_id and agent.stage_order < 60:
                review, should_rerun = director.supervise_stage(agent, blackboard)
                review_obj = review
                stage_reviews.append(review)
                if should_rerun:
                    if progress_callback:
                        progress_callback(
                            director.role_id,
                            f"[Director Directive -> {agent.role_id}] {review.directive_sent}",
                            min(agent_pct_span, pct + 2),
                        )
                    if event_callback:
                        event_callback(
                            {
                                "event": "director_directive",
                                "role_id": agent.role_id,
                                "display_name": agent.display_name,
                                "stage_order": agent.stage_order,
                                "progress_pct": min(agent_pct_span, pct + 2),
                                "directive_sent": review.directive_sent,
                                "critique": review.critique,
                                "quality_score": review.quality_score,
                                "message": f"Director directed improvement on {agent.display_name}: {review.directive_sent}",
                            }
                        )
                    improved_summary = agent.run(blackboard)
                    summary = improved_summary
                    dur_sec = round(time.monotonic() - t0, 2)
                    blackboard.executed_agents.append(
                        {
                            "role_id": agent.role_id,
                            "display_name": f"{agent.display_name} (Director-Guided Improvement)",
                            "stage_order": agent.stage_order,
                            "round": blackboard.revision_round,
                            "summary": improved_summary,
                        }
                    )

            end_pct = int(round(((step_idx + 1) / max(1, total_steps)) * agent_pct_span))
            if event_callback:
                event_callback(
                    {
                        "event": "stage_completed",
                        "role_id": agent.role_id,
                        "display_name": agent.display_name,
                        "stage_order": agent.stage_order,
                        "progress_pct": end_pct,
                        "duration_sec": dur_sec,
                        "summary": summary,
                        "review_status": review_obj.status if review_obj else "approved",
                        "quality_score": (
                            review_obj.quality_score
                            if review_obj
                            else (
                                blackboard.director_review.overall_quality_score
                                if blackboard.director_review
                                else 9.5
                            )
                        ),
                        "critique": (
                            review_obj.critique
                            if review_obj
                            else (
                                f"Master Cut Approved — Grade {blackboard.director_review.quality_grade} ({blackboard.director_review.overall_quality_score}/10)"
                                if blackboard.director_review
                                else f"Completed {agent.display_name}"
                            )
                        ),
                        "directive_sent": review_obj.directive_sent if review_obj else None,
                        "message": f"Completed {agent.display_name} ({dur_sec:.1f}s)",
                    }
                )

            blackboard.save_snapshot()

        # 1. Execute Stages 00 -> 60
        idx = 0
        while idx < len(pre_export_plan):
            agent = pre_export_plan[idx]
            _execute_single_agent(agent, idx)
            if agent.role_id == "client_intake" and blackboard.creative_brief is not None:
                for opt_role in blackboard.creative_brief.enabled_optional_agents:
                    new_cls = AgentRegistry.get(opt_role)
                    if new_cls is not None:
                        canon_id = new_cls.role_id
                        if canon_id not in self.enabled_optional_roles and canon_id not in self.disabled_roles:
                            self.enabled_optional_roles.append(canon_id)
                            if not any(a.role_id == canon_id for a in pre_export_plan):
                                pre_export_plan.append(new_cls())
                                pre_export_plan.sort(
                                    key=lambda a: (a.stage_order, a.substage_priority, a.role_id)
                                )
                                total_steps = max(1, len(pre_export_plan) + len(post_export_plan))
            idx += 1

        # 2. Structured Beat-Preserving Chapter Revision Loop (strictly capped at max_revisions = 2)
        while blackboard.pending_revision_directives() and blackboard.revision_round < blackboard.max_revisions:
            pending = blackboard.pending_revision_directives()
            blackboard.revision_round += 1

            target_stages: list[int] = []
            needs_story_rebuild = False
            for d in pending:
                t_cls = AgentRegistry.get(d.target_role_id)
                if t_cls is not None:
                    target_stages.append(t_cls.stage_order)
                    if t_cls.role_id == "story_producer":
                        needs_story_rebuild = True
                else:
                    target_stages.append(30)

            min_stage = min(target_stages) if target_stages else 30
            if needs_story_rebuild:
                blackboard.timeline = None
                blackboard.chapter_timeline_cache.clear()

            revision_agents = [
                a for a in pre_export_plan if min_stage <= a.stage_order <= 60
            ]
            for r_agent in revision_agents:
                if progress_callback:
                    progress_callback(
                        r_agent.role_id,
                        f"[Revision Round {blackboard.revision_round}/{blackboard.max_revisions}] Re-running {r_agent.display_name}...",
                        min(92, agent_pct_span),
                    )
                r_summary = r_agent.run(blackboard)
                blackboard.executed_agents.append(
                    {
                        "role_id": r_agent.role_id,
                        "display_name": r_agent.display_name,
                        "stage_order": r_agent.stage_order,
                        "round": blackboard.revision_round,
                        "summary": r_summary,
                    }
                )
            blackboard.send_message(
                sender_role="orchestrator",
                recipient_role="all",
                message_type="revision_completed",
                subject=f"Completed Revision Round {blackboard.revision_round}/{blackboard.max_revisions}",
                payload={"revision_round": blackboard.revision_round, "min_stage": min_stage},
            )
            blackboard.save_snapshot()

        # 3. Execute Post-Signoff Stage 65 NLE Exporter (timeline.fcpxml + EDL/OTIO/SRT/ASS)
        for p_idx, p_agent in enumerate(post_export_plan):
            _execute_single_agent(p_agent, len(pre_export_plan) + p_idx)

        # Ensure subtitle sidecars exist even if nle_exporter was disabled
        if blackboard.timeline is not None:
            if export_fcpxml and not (blackboard.project_dir / "timeline.fcpxml").exists():
                export_all_nle_formats(blackboard.timeline, blackboard.project_dir)
            export_srt_subtitles(blackboard.timeline, blackboard.project_dir / "subtitles.srt")
            export_ass_subtitles(
                blackboard.timeline,
                blackboard.project_dir / "subtitles.ass",
                hdr=blackboard.timeline.settings.hdr,
            )

            if render_video:
                render_t0 = time.monotonic()
                if progress_callback:
                    progress_callback("renderer", "Rendering 4K 10-bit HLG HDR MP4...", 78)
                if event_callback:
                    event_callback(
                        {
                            "event": "stage_started",
                            "role_id": "renderer",
                            "display_name": "7. 4K 10-bit HLG HDR Video & NLE Renderer",
                            "stage_order": 70,
                            "progress_pct": 78,
                            "message": "Starting 4K 10-bit HLG HDR video synthesis & audio mixing...",
                        }
                    )

                def _on_render_prog(msg: str, frac: float) -> None:
                    r_pct = min(99, int(round(78 + frac * 21)))
                    if progress_callback:
                        progress_callback("renderer", msg, r_pct)
                    if event_callback:
                        event_callback(
                            {
                                "event": "render_progress",
                                "role_id": "renderer",
                                "display_name": "7. 4K 10-bit HLG HDR Video & NLE Renderer",
                                "stage_order": 70,
                                "progress_pct": r_pct,
                                "message": msg,
                            }
                        )

                out_mp4 = blackboard.project_dir / output_video_name
                render_timeline(
                    timeline=blackboard.timeline,
                    output_mp4=out_mp4,
                    work_dir=blackboard.project_dir / "render_cache",
                    progress_callback=_on_render_prog,
                )
                render_dur = round(time.monotonic() - render_t0, 2)
                if event_callback:
                    event_callback(
                        {
                            "event": "stage_completed",
                            "role_id": "renderer",
                            "display_name": "7. 4K 10-bit HLG HDR Video & NLE Renderer",
                            "stage_order": 70,
                            "progress_pct": 100,
                            "duration_sec": render_dur,
                            "review_status": "approved",
                            "quality_score": 9.8,
                            "critique": f"Rendered {output_video_name} in 4K UHD 10-bit HLG HDR ({render_dur:.1f}s) + FCPXML/EDL/OTIO.",
                            "message": f"Rendered 4K HDR master video ({output_video_name}) in {render_dur:.1f}s",
                        }
                    )

        if progress_callback:
            progress_callback("complete", "Multi-Agent Editing Pipeline Complete!", 100)
        if event_callback:
            event_callback(
                {
                    "event": "pipeline_complete",
                    "role_id": "complete",
                    "progress_pct": 100,
                    "message": "Multi-Agent Editing Pipeline & 4K HDR Deliverables Complete!",
                }
            )

        return blackboard


def run_multi_agent_pipeline(
    media_dir: str | Path,
    project_dir: str | Path,
    user_prompt: str = "",
    extra_audio_paths: Optional[list[str | Path]] = None,
    interview_answers: Optional[dict[str, str]] = None,
    enabled_optional_roles: Optional[list[str]] = None,
    disabled_roles: Optional[list[str]] = None,
    width: int = 3840,
    height: int = 2160,
    fps: int = 30,
    hdr: bool = True,
    use_antigravity: bool = True,
    whisper_model: str = "small",
    render_video: bool = False,
    export_fcpxml: bool = True,
    output_video_name: str = "final_story.mp4",
    progress_callback: Optional[Callable[[str, str, int], None]] = None,
    event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> PipelineBlackboard:
    """High-level entry point to run the coordinated multi-agent editing pipeline."""
    blackboard = PipelineBlackboard(
        media_dir=media_dir,
        project_dir=project_dir,
        user_prompt=user_prompt,
        extra_audio_paths=extra_audio_paths,
        interview_answers=interview_answers,
        width=width,
        height=height,
        fps=fps,
        hdr=hdr,
        use_antigravity=use_antigravity,
        whisper_model=whisper_model,
    )
    orchestrator = MultiAgentOrchestrator(
        enabled_optional_roles=enabled_optional_roles,
        disabled_roles=disabled_roles,
    )
    return orchestrator.run(
        blackboard=blackboard,
        render_video=render_video,
        export_fcpxml=export_fcpxml,
        output_video_name=output_video_name,
        progress_callback=progress_callback,
        event_callback=event_callback,
    )
