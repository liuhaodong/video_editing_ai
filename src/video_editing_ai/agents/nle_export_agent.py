"""Stage 65: NLE Interchange Exporter Agent (nle_exporter)."""

from __future__ import annotations

from typing import Any

from ..exporters.nle_export import export_all_nle_formats
from ..models.schema import EditTimeline
from ..renderer.subtitles import export_ass_subtitles, export_srt_subtitles
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


@register_agent
class NLEExportAgent(BaseEditingAgent):
    """Stage 65 Agent: Emits frame-accurate timeline.fcpxml (FCPXML 1.11) + EDL/OTIO/SRT/ASS sidecars."""

    role_id = "nle_exporter"
    display_name = "6b. FCPXML 1.11 & NLE Interchange Exporter"
    stage_order = 65
    substage_priority = 0
    description = (
        "Emits a frame-accurate timeline.fcpxml (FCPXML 1.11 for DaVinci Resolve & Final Cut Pro) "
        "mapping Track V1 spine, V2 connected B-roll cutaways, V3 title lanes, A1-A3 audio lanes, "
        "and chapter markers alongside EDL, OTIO, SRT, and ASS sidecars."
    )
    enabled_by_default = True
    requires_artifacts = ["timeline.yaml"]
    produces_artifacts = ["timeline.fcpxml"]
    skill_reference_file = "18_nle_exporter_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        timeline = blackboard.timeline
        if timeline is None and (blackboard.project_dir / "timeline.yaml").exists():
            timeline = EditTimeline.from_yaml(blackboard.project_dir / "timeline.yaml")
            blackboard.timeline = timeline
        if timeline is None:
            raise RuntimeError("NLEExportAgent requires blackboard.timeline or timeline.yaml.")

        nle_paths = export_all_nle_formats(timeline, blackboard.project_dir)
        srt_p = export_srt_subtitles(timeline, blackboard.project_dir / "subtitles.srt")
        ass_p = export_ass_subtitles(timeline, blackboard.project_dir / "subtitles.ass", hdr=timeline.settings.hdr)

        fcpxml_path = nle_paths["fcpxml"]
        blackboard.emit_activity(
            self.role_id,
            f"Exported frame-accurate timeline.fcpxml (FCPXML 1.11) + EDL/OTIO/SRT/ASS ({len(timeline.visual_track)} spine clips)",
            detail=str(fcpxml_path),
        )
        return {
            "fcpxml_path": str(fcpxml_path),
            "edl_path": str(nle_paths["edl"]),
            "otio_path": str(nle_paths["otio"]),
            "srt_path": str(srt_p),
            "ass_path": str(ass_p),
        }
