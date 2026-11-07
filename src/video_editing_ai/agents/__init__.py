"""Modular Multi-Agent Video Editing Architecture (Stage 0 -> Stage 60 + Pluggable Slots)."""

from .audio_vibe_agent import AudioVibeAgent
from .base import AgentRegistry, BaseEditingAgent, PipelineBlackboard, register_agent
from .culler_agent import CullerAgent
from .director_agent import ProductionDirectorAgent
from .intake_agent import (
    ClientIntakeAgent,
    compute_director_duration_recommendation,
    generate_intake_questions,
    inspect_media_library,
    refine_intake_interactively,
)
from .music_supervisor_agent import MusicSupervisorAgent
from .orchestrator import MultiAgentOrchestrator, run_multi_agent_pipeline
from .photo_curator_agent import (
    PhotoCuratorAgent,
    apply_vision_curation_overrides,
    render_curation_contact_sheets,
)
from .picture_editors import DialogueEditorAgent, MontageEditorAgent
from .plugins.vfx_agent import VFXAgent
from .showrunner_agent import ShowrunnerAgent
from .story_producer_agent import StoryProducerAgent, apply_story_plan_overrides

__all__ = [
    "AgentRegistry",
    "AudioVibeAgent",
    "BaseEditingAgent",
    "ClientIntakeAgent",
    "CullerAgent",
    "DialogueEditorAgent",
    "MontageEditorAgent",
    "MultiAgentOrchestrator",
    "MusicSupervisorAgent",
    "PhotoCuratorAgent",
    "PipelineBlackboard",
    "ProductionDirectorAgent",
    "ShowrunnerAgent",
    "StoryProducerAgent",
    "VFXAgent",
    "apply_story_plan_overrides",
    "apply_vision_curation_overrides",
    "compute_director_duration_recommendation",
    "generate_intake_questions",
    "inspect_media_library",
    "refine_intake_interactively",
    "register_agent",
    "render_curation_contact_sheets",
    "run_multi_agent_pipeline",
]
