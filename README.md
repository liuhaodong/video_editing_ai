# AI Photo & Video Editing Studio (`video_editing_ai`)

An end-to-end **Modular Multi-Agent Family Travel & Vlog Editing Studio**, narrative story planner, text-based timeline editor, and programmatic **4K UHD 10-bit HLG HDR** open-source video renderer built with the **Google Antigravity SDK**, **Antigravity Skills**, **faster-whisper**, **FFmpeg**, and **Pillow**.

---

## Modular Multi-Agent Architecture (Stages 0 → 50 + Pluggable Slots)

Unscripted family travel videos and vlogs require solving a **massive signal-to-noise problem** across multi-camera clips and photo bursts. `video_editing_ai` implements a coordinated multi-agent pipeline communicating over a shared `PipelineBlackboard` (`blackboard.json`) and strict JSON data contracts:

| Stage | Role ID | Sub-Agent Class | JSON Contract / Output | Responsibilities |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 00** | `client_intake` | `ClientIntakeAgent` | `creative_brief.json` | Interviews the client on narrative arc, tone, must-have moments, dialogue handling, and B-roll pacing guardrails. |
| **Stage 10** | `culler` | `CullerAgent` | `media_index.json`, `annotations.yaml` | Multi-cam timecode normalization, shot scale (`wide`/`medium`/`closeup`) & camera motion tagging, word-level ASR timestamps, 2–4.5s B-roll "golden window" extraction, and photo burst deduplication. |
| **Stage 10** | `audio_vibe` | `AudioVibeAgent` | `beat_grid.json` | Music waveform onset & autocorrelation analysis for BPM, beats, downbeats, energy sections (`intro`, `verse`, `chorus_drop`, `outro`), `-20 dB` Nat-Sound bed, and `-14 dB` auto-ducking envelopes. |
| **Stage 20** | `story_producer` | `StoryProducerAgent` | `paper_edit.json` | Clusters selects into 4 vignette chapters, switches between `dialogue_with_broll` and `music_montage` modes, and maps **Track V2 B-Roll Cutaway** visual anchors over long A-roll clips. |
| **Stage 30** | `dialogue_editor` | `DialogueEditorAgent` | `EditTimeline` (`V1`, `V2`, `subtitles`) | Trims A-roll clips on word timestamps, applies `-0.35s` J-cut audio lead-ins, generates bilingual subtitles, and places Track V2 B-roll cutaways over long dialogue takes. |
| **Stage 30** | `montage_editor` | `MontageEditorAgent` | `EditTimeline` (`V1`, `A3`) | Snaps B-roll and photo durations to the musical beat grid, sequences `Wide -> Medium -> Close-up` progression, applies Gallery Frame Ken Burns, and configures `-20 dB` Nat-Sound (`0.33s` crossfades) + `-14 dB` auto-ducking. |
| **Stage 40** *(Plugin)* | `vfx_designer` | `VFXAgent` | `vfx_manifest.json`, `effects_track` | Optional `@register_agent` plugin demonstrating modular extensibility; attaches non-destructive `warm_glow`, `vignette_focus`, and `contrast_pop` effects to `timeline.effects_track`. |
| **Stage 50** | `showrunner` | `ShowrunnerAgent` | `pacing_audit.json`, `timeline.yaml`, `storyboard.md` | Audits the assembled timeline against `creative_brief.json` (caps pure B-roll at `4.5s`, drops `>3` consecutive static stills, injects chapter lower-thirds & closing card, verifies must-haves, and triggers a revision loop if needed). |

---

## Adding Custom Sub-Agent Roles (`@register_agent`)

The pipeline uses a plug-and-play `AgentRegistry`. You can introduce new sub-agent roles (e.g., a future 3D VFX tracker, HDR colorist, or sound-effects designer) without modifying the core orchestrator:

```python
from video_editing_ai.agents import BaseEditingAgent, PipelineBlackboard, register_agent
from video_editing_ai.models.schema import EffectTrackItem

@register_agent
class CustomHDRColoristAgent(BaseEditingAgent):
    role_id = "hdr_colorist"
    display_name = "4b. Custom HDR Colorist"
    stage_order = 40  # Runs in Stage 40 between Picture Editors (30) and Showrunner (50)
    enabled_by_default = False
    requires_artifacts = ["paper_edit.json"]
    produces_artifacts = ["color_lut_manifest.json"]

    def run(self, blackboard: PipelineBlackboard) -> dict:
        # Read/modify blackboard.timeline.effects_track and write custom JSON contracts
        path = blackboard.set_custom_artifact("color_lut_manifest.json", {"lut": "Rec2020_Golden"})
        blackboard.send_message(self.role_id, "showrunner", "Applied HDR color pass", {"path": str(path)})
        return {"color_manifest": str(path)}
```

---

## Antigravity Skill Package

This workspace includes a native Antigravity Skill at `.agent/skills/ai-video-editor/SKILL.md` along with role-specific reference prompts in `.agent/skills/ai-video-editor/references/`:
- `00_client_intake_agent.md`
- `01_culler_agent.md`
- `02_story_producer_agent.md`
- `03_picture_editors.md`
- `04_audio_vibe_agent.md`
- `05_showrunner_agent.md`
- `06_vfx_agent.md`

---

## Quickstart

### 1. Multi-Agent CLI Usage

```powershell
# List all registered core and pluggable sub-agent roles
python -m video_editing_ai.cli agents-list

# Run the coordinated Multi-Agent Pipeline (Stages 0 -> 50) + optional VFX Designer plugin
python -m video_editing_ai.cli agents-run "C:\Users\hihao\Documents\projects\ai_editor\raw_media" `
  -o ".\gilroy_halloween_project" `
  --bgm "C:\Users\hihao\Documents\projects\ai_editor\bgm_sneaky_snitch.mp3" `
  -p "Create an upbeat 4-act family Halloween highlight video at Gilroy Gardens. Keep the kids' excited Mandarin dialogue, cover long takes with B-roll cutaways, feature the pumpkin pyramid with gallery frames, and include the stage show." `
  --enable-agent vfx_designer `
  --render
```

### 2. Local Web Studio

```powershell
python -m video_editing_ai.cli serve --port 8000
```
Then open `http://127.0.0.1:8000` in your browser to:
- Answer the **Stage 0 Client Intake Interview** and toggle pluggable sub-agent roles (`vfx_designer`, etc.).
- Inspect `creative_brief.json`, `media_index.json`, `beat_grid.json`, `paper_edit.json`, `pacing_audit.json`, `vfx_manifest.json`, and the `blackboard.json` inter-agent message bus.
- Edit `timeline.yaml` and preview **Track V1 Storyline**, **Track V2 B-Roll Cutaways**, **Track V3 Titles**, **Stage-40 VFX**, and the **4K 10-bit HLG HDR** rendered video.
