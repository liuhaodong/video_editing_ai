"""Modular Multi-Agent Video Editing Architecture (Stage 0 -> Stage 60 + Pluggable Slots)."""

from .audio_vibe_agent import AudioVibeAgent
from .base import AgentRegistry, BaseEditingAgent, PipelineBlackboard, register_agent
from .culler_agent import CullerAgent
from .director_agent import ProductionDirectorAgent
from .intake_agent import ClientIntakeAgent, generate_intake_questions
from .music_supervisor_agent import MusicSupervisorAgent
from .orchestrator import MultiAgentOrchestrator, run_multi_agent_pipeline
from .photo_curator_agent import PhotoCuratorAgent
from .picture_editors import DialogueEditorAgent, MontageEditorAgent
from .plugins.vfx_agent import VFXAgent
from .showrunner_agent import ShowrunnerAgent
from .story_producer_agent import StoryProducerAgent

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
    "generate_intake_questions",
    "register_agent",
    "run_multi_agent_pipeline",
]
