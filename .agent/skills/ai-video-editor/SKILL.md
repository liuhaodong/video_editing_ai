---
name: ai-video-editor
description: Modular multi-agent AI photo and video editing skill for family travel, vlogs, and unscripted highlights. Orchestrates specialized sub-agents (Client Intake, Culler & Sync, Audio/Beat Vibe Designer, Story Producer, Dual Dialogue & Montage Editors, Pluggable VFX/Color Agents, Showrunner Pacing Critic, and Executive Movie Production Director) through deterministic JSON contracts (creative_brief.json, media_index.json, beat_grid.json, paper_edit.json, pacing_audit.json, director_review.json, timeline.yaml) and renders 4K 10-bit HLG HDR deliverables + FCPXML/EDL/OTIO.
---

# Modular Multi-Agent Video & Photo Editor Skill

Use this skill whenever the user wants to analyze a folder of videos and photos, interview/clarify story preferences, run the multi-agent vlog & family travel editing pipeline, add a custom editing sub-agent role (such as VFX or Color), or render a 4K 10-bit HLG HDR video.

## 1. Sub-Agent Role Reference Files (Progressive Disclosure)

Each agent role has its own reference specification in `references/` (used by both Antigravity's `define_subagent` / `invoke_subagent` tools and the Python `BaseEditingAgent` classes):

- **Stage 0 — Client Intake & Brief Producer (`client_intake`)**: [references/00_client_intake_agent.md](references/00_client_intake_agent.md) → outputs `creative_brief.json`
- **Stage 10 — Wrangler & Culler (`culler`)**: [references/01_culler_agent.md](references/01_culler_agent.md) → outputs `media_index.json`
- **Stage 10 — Sound & Vibe Designer (`audio_vibe`)**: [references/04_audio_vibe_agent.md](references/04_audio_vibe_agent.md) → outputs `beat_grid.json`
- **Stage 20 — Story Producer (`story_producer`)**: [references/02_story_producer_agent.md](references/02_story_producer_agent.md) → outputs `paper_edit.json`
- **Stage 30 — Dual Picture Editors (`dialogue_editor` + `montage_editor`)**: [references/03_picture_editors.md](references/03_picture_editors.md) → outputs draft multi-track `timeline.yaml`
- **Stage 40 — Pluggable Post / VFX / Color Slot (`vfx_designer`)**: [references/06_vfx_agent.md](references/06_vfx_agent.md) → enriches `timeline.yaml` (`effects_track`) & outputs `vfx_manifest.json`
- **Stage 50 — Showrunner & Pacing Critic (`showrunner`)**: [references/05_showrunner_agent.md](references/05_showrunner_agent.md) → audits against `creative_brief.json`, coordinates revisions, and outputs `pacing_audit.json` + draft `timeline.yaml` + `storyboard.md`
- **Stage 60 & Cross-Stage Supervisor — Executive Movie Production Director (`production_director`)**: [references/07_production_director_agent.md](references/07_production_director_agent.md) → oversees all sub-agents at every stage gate, directs sub-agents to fix/improve coverage, motion variety, and audio/title continuity, and outputs `director_review.json` + final signed-off `timeline.yaml` & `storyboard.md`

---

## 2. Running the Multi-Agent Pipeline

### Option A: Native Antigravity Subagents + CLI Orchestrator
1. **Interview the Client (Stage 0)**:
   Use `ask_question` (guided by [references/00_client_intake_agent.md](references/00_client_intake_agent.md)) to clarify narrative style, must-have moments, target duration, and photo framing, or run:
   ```powershell
   python -m video_editing_ai.cli agents-run "<media_dir>" -o "<project_dir>" --prompt "<user_story_idea>" --bgm "<optional_bgm.mp3>"
   ```
2. **Spawn Specialized Subagents (`define_subagent` / `invoke_subagent`)**:
   You can spawn any of the roles in isolated context windows using `define_subagent` with the system prompt from its corresponding `references/*.md` file so each role inspects and enriches its JSON contract on disk (`creative_brief.json` → `media_index.json` & `beat_grid.json` → `paper_edit.json` → `timeline.yaml` → `pacing_audit.json`).
3. **Render 4K 10-bit HLG HDR Video & NLE Exports**:
   ```powershell
   python -m video_editing_ai.cli render "<project_dir>/timeline.yaml" -o "<project_dir>/final_story.mp4"
   ```

### Option B: List or Toggle Pluggable Sub-Agents
```powershell
# List all registered core and optional plugin agents
python -m video_editing_ai.cli agents-list

# Run the pipeline with the optional Stage-40 VFX Designer plugin enabled
python -m video_editing_ai.cli agents-run "<media_dir>" -o "<project_dir>" --prompt "<idea>" --enable-agent vfx_designer
```

### Option C: Local Web Studio
```powershell
python -m video_editing_ai.cli serve --port 8000
```
