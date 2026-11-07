"""Multi-Agent Pipeline Orchestrator with Executive Movie Production Director supervision and revision loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from ..exporters.nle_export import export_all_nle_formats
from ..models.schema import DirectorStageReview
from ..renderer.ffmpeg_composer import render_timeline
from ..renderer.subtitles import export_ass_subtitles, export_srt_subtitles
from . import (
    audio_vibe_agent as _audio_vibe,  # noqa: F401
    culler_agent as _culler,  # noqa: F401
    director_agent as _director,  # noqa: F401
    intake_agent as _intake,  # noqa: F401
    picture_editors as _editors,  # noqa: F401
    showrunner_agent as _showrunner,  # noqa: F401
)
from .base import AgentRegistry, BaseEditingAgent, PipelineBlackboard
from .director_agent import ProductionDirectorAgent
from .plugins import vfx_agent as _vfx  # noqa: F401


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
        output_video_name: str = "final_story.mp4",
        progress_callback: Optional[Callable[[str, str, int], None]] = None,
    ) -> PipelineBlackboard:
        plan = self.get_execution_plan()
        total_steps = max(1, len(plan))

        director: Optional[ProductionDirectorAgent] = next(
            (a for a in plan if isinstance(a, ProductionDirectorAgent)),
            None,
        )
        stage_reviews: list[DirectorStageReview] = []
        setattr(blackboard, "_director_stage_reviews", stage_reviews)

        for idx, agent in enumerate(plan):
            pct = int(round((idx / total_steps) * 85))
            if progress_callback:
                progress_callback(agent.role_id, f"Running {agent.display_name}...", pct)

            summary = agent.run(blackboard)
            blackboard.executed_agents.append(
                {
                    "role_id": agent.role_id,
                    "display_name": agent.display_name,
                    "stage_order": agent.stage_order,
                    "round": blackboard.revision_round,
                    "summary": summary,
                }
            )

            # Executive Production Director supervises each stage (< 60) and directs immediate improvements if needed
            if director is not None and agent.role_id != director.role_id:
                review, should_rerun = director.supervise_stage(agent, blackboard)
                stage_reviews.append(review)
                if should_rerun:
                    if progress_callback:
                        progress_callback(
                            director.role_id,
                            f"[Director Directive -> {agent.role_id}] {review.directive_sent}",
                            pct + 2,
                        )
                    improved_summary = agent.run(blackboard)
                    blackboard.executed_agents.append(
                        {
                            "role_id": agent.role_id,
                            "display_name": f"{agent.display_name} (Director-Guided Improvement)",
                            "stage_order": agent.stage_order,
                            "round": blackboard.revision_round,
                            "summary": improved_summary,
                        }
                    )

            blackboard.save_snapshot()

            # If Stage 0 ClientIntakeAgent enabled optional agents in creative_brief, merge them into the plan
            if agent.role_id == "client_intake" and blackboard.creative_brief is not None:
                for opt_role in blackboard.creative_brief.enabled_optional_agents:
                    if opt_role not in self.enabled_optional_roles and opt_role not in self.disabled_roles:
                        self.enabled_optional_roles.append(opt_role)
                        new_cls = AgentRegistry.get(opt_role)
                        if new_cls is not None and not any(a.role_id == opt_role for a in plan):
                            plan.append(new_cls())
                            plan.sort(key=lambda a: (a.stage_order, a.role_id))
                            total_steps = max(1, len(plan))

        # Check if Showrunner requested a full cross-stage revision loop (at most 1 revision round)
        has_revision_request = any(
            m.message_type == "revision_request" and m.sender_role == "showrunner"
            for m in blackboard.messages
        )
        if has_revision_request and blackboard.revision_round == 0:
            blackboard.revision_round = 1
            blackboard.timeline = None
            revision_agents = [a for a in self.get_execution_plan() if a.stage_order >= 20]
            for r_agent in revision_agents:
                if progress_callback:
                    progress_callback(
                        r_agent.role_id,
                        f"[Revision Round 1] Re-running {r_agent.display_name}...",
                        88,
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
                sender_role="showrunner",
                recipient_role="all",
                message_type="revision_completed",
                subject="Completed Showrunner revision loop and verified must-have selects",
                payload={"revision_round": blackboard.revision_round},
            )
            blackboard.save_snapshot()

        # Export NLE timelines (.fcpxml, .edl, .otio) and subtitle sidecars (.srt, .ass)
        if blackboard.timeline is not None:
            export_all_nle_formats(blackboard.timeline, blackboard.project_dir)
            export_srt_subtitles(blackboard.timeline, blackboard.project_dir / "subtitles.srt")
            export_ass_subtitles(blackboard.timeline, blackboard.project_dir / "subtitles.ass")

            if render_video:
                if progress_callback:
                    progress_callback("renderer", "Rendering 4K 10-bit HLG HDR MP4...", 92)
                out_mp4 = blackboard.project_dir / output_video_name
                render_timeline(
                    timeline=blackboard.timeline,
                    output_mp4=out_mp4,
                    work_dir=blackboard.project_dir / "render_cache",
                )

        if progress_callback:
            progress_callback("complete", "Multi-Agent Editing Pipeline Complete!", 100)

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
    output_video_name: str = "final_story.mp4",
    progress_callback: Optional[Callable[[str, str, int], None]] = None,
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
        output_video_name=output_video_name,
        progress_callback=progress_callback,
    )
