---
name: ai-video-editor
description: >-
  End-to-end 11-agent AI photo and video editing skill for family travel, museum trips, vlogs, and unscripted highlights. Use whenever the user asks to edit a video from a folder of clips/photos, run interactive client intake and get the Director's length recommendation (30s-600s), scout/download royalty-free BGM, curate and annotate photos with aesthetic taste and museum plaques, apply 2.5D parallax and diptych photo effects, supervise editing quality with the Executive Movie Production Director, manage editing projects (create/load/reset), or render 4K 10-bit HLG HDR MP4 videos and NLE timelines (.fcpxml, .edl, .otio).
---

# AI Photo & Video Story Studio Skill (`ai-video-editor`)

## Overview

This skill orchestrates an **11-Agent AI Film & Vlog Editing Room** that transforms raw folders of multi-camera videos (`.MP4`, `.MOV`) and photos (`.JPG`, `.HEIC`, `.PNG`) into a **4K 10-bit HLG HDR Master Video (`final_story.mp4`)** plus professional NLE timelines (`timeline.fcpxml`, `timeline.edl`, `timeline.otio`).

All programmatic steps write structured JSON/YAML contracts to disk in `<project_dir>` via the skill's standalone helper CLI:
- **Helper CLI**: [`scripts/ai_video_studio_skill.py`](scripts/ai_video_studio_skill.py)
- **Python Interpreter**: `C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe`

---

## 1. Progressive Disclosure: 11 Sub-Agent Role Playbooks (`references/`)

Read only the specific reference file(s) needed for your current step or when spawning native Antigravity subagents via `define_subagent` / `invoke_subagent`:

1. **Stage 00 — Client Intake & Brief Producer (`client_intake`)**: [references/00_client_intake_agent.md](references/00_client_intake_agent.md) → `creative_brief.json`
2. **Stage 05 — Music Supervisor & BGM Scout (`music_supervisor`)**: [references/08_music_supervisor_agent.md](references/08_music_supervisor_agent.md) → `bgm_selection.json`
3. **Stage 10 — Wrangler & Culler (`culler`)**: [references/01_culler_agent.md](references/01_culler_agent.md) → `media_index.json`
4. **Stage 10 — Sound & Vibe Beat Specialist (`audio_vibe`)**: [references/04_audio_vibe_agent.md](references/04_audio_vibe_agent.md) → `beat_grid.json`
5. **Stage 15 — Aesthetic Photo Curator & Museum Annotator (`photo_curator`)**: [references/09_photo_curator_agent.md](references/09_photo_curator_agent.md) → `photo_curation.json`
6. **Stage 20 — Story Producer (`story_producer`)**: [references/02_story_producer_agent.md](references/02_story_producer_agent.md) → `paper_edit.json`
7. **Stage 30 — Dual Picture Editors (`dialogue_editor` + `montage_editor`)**: [references/03_picture_editors.md](references/03_picture_editors.md) → `timeline.yaml`
8. **Stage 40 — 2.5D Parallax, Diptych & HDR Polish Designer (`vfx_designer`)**: [references/06_vfx_agent.md](references/06_vfx_agent.md) → `vfx_manifest.json`
9. **Stage 50 — Showrunner & Pacing Critic (`showrunner`)**: [references/05_showrunner_agent.md](references/05_showrunner_agent.md) → `pacing_audit.json`
10. **Stage 60 & Cross-Stage Supervisor — Executive Movie Production Director (`production_director`)**: [references/07_production_director_agent.md](references/07_production_director_agent.md) → `director_review.json`
11. **JSON Contracts & 4K HLG HDR Rendering Spec**: [references/10_contracts_and_rendering.md](references/10_contracts_and_rendering.md)

---

## 2. Six-Phase End-to-End Editing Workflow

When the user asks you to edit a video from a media folder, follow these 6 phases:

### Phase 1: Project Setup & Media Library Inspection (`inspect-library` & `project`)
Scan the raw media folder to detect video/photo counts, camera models (`Sony A7C II`, `Google Pixel`, `Apple iPhone`), and the **Executive Director's Recommended Video Length (`30s–600s`)**:
```powershell
C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py inspect-library --media-dir "<media_dir>" --project-dir "<project_dir>" --prompt "<user_prompt>"
```
- Read `<project_dir>/library_inspection.json` to inspect `director_duration_recommendation` (`director_recommended_duration_sec`, `recommended_photo_plates`, `director_duration_rationale`).
- To create, load, list, or reset a project workspace:
  ```powershell
  C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py project --action create --name "Louvre Family Trip" --media-dir "<media_dir>" --project-dir "<project_dir>"
  ```

### Phase 2: Interactive Client Intake & Director Length Co-Refinement (`intake-refine`)
If the user's preferences are open-ended, use `ask_question` to confirm their preferred duration (`Director's Recommended Length` vs. custom `30s–600s`), narrative arc, and photo style, then run `intake-refine` to generate `creative_brief.json`:
```powershell
C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py intake-refine --media-dir "<media_dir>" --project-dir "<project_dir>" --prompt "<user_prompt>" --duration 155 --photo-mode parallax_gallery --bgm-mood auto_match
```

### Phase 3: Run the 11-Agent Pipeline (`run-pipeline` or Modular `run-stage`)
- **Option A — Full 11-Agent Pipeline + 4K 10-bit HLG HDR Render**:
  ```powershell
  C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py run-pipeline --media-dir "<media_dir>" --project-dir "<project_dir>" --prompt "<user_prompt>" --duration 155 --photo-mode parallax_gallery --render
  ```
- **Option B — Stage-by-Stage Execution with Custom Antigravity Visual Inspection (`run-stage`)**:
  Run individual stages when you want to inspect intermediate contracts or visually examine curated photos (`read_file` on `.jpg` frames) to write custom artwork/landmark annotations before assembling the timeline:
  ```powershell
  # 1. Run Intake, BGM Scout, Culler, Audio Vibe, and Photo Curator
  C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py run-stage --stage client_intake,music_supervisor,culler,audio_vibe,photo_curator --project-dir "<project_dir>" --media-dir "<media_dir>"

  # 2. Inspect/enrich <project_dir>/photo_curation.json & media_index.json if needed, then run Story, Editors, VFX, Showrunner & Director:
  C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py run-stage --stage story_producer,dialogue_editor,montage_editor,vfx_designer,showrunner,production_director --project-dir "<project_dir>" --media-dir "<media_dir>"
  ```

### Phase 4: Executive Movie Production Director Audit (`director-audit`)
Read `<project_dir>/director_review.json` (or run `director-audit`) to verify:
- `approved_for_master_render == true` and `overall_quality_score >= 9.0` (Grade `A` or `A+`).
- All 10 stage reviews (`stage_reviews`) recorded the Director's **Observation**, **Thinking**, and **Verdict/Directives**.
- `bgm_selection.json` selected a high-scoring soundtrack (`mood_match_score >= 8.5`) and rejected any mismatched audio.
- `photo_curation.json` selected high-taste photos (`average_taste_score_selected >= 8.5`) with 100% `annotation_title` museum plaque coverage.

### Phase 5: 4K 10-bit HLG HDR Rendering & Visual Keyframe Verification (`render-video`)
If `--no-render` was used during planning or if `timeline.yaml` was edited, render the final 4K HDR MP4 and extract verification frames across the 4 acts:
```powershell
C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py render-video --project-dir "<project_dir>" --extract-frames "6,9,16,30,36,52"
```
Use `read_file` on the extracted JPEGs in `<project_dir>/verification_frames/` to visually confirm uncropped 2.5D parallax frames, diptych split-screens, and lower-left museum exhibition plaques.

### Phase 6: Interactive 5-Tab Web Studio (`serve`)
To let the user interactively manage projects, co-refine intake, watch real-time Agent Doing + Director Thinking streams, or download the newly rendered `.mp4` in their browser:
```powershell
C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe -m video_editing_ai.cli serve --project-dir "<project_dir>" --host 0.0.0.0 --port 8000
```

---

## 3. Common Mistakes to Avoid

1. **Do not hardcode a 2-minute (`120s`) duration cap**: Always check `director_duration_recommendation` in `library_inspection.json` (`30s–600s`) and let photo plate count scale dynamically with duration.
2. **Never crop vertical or 4:3 photos to 16:9 (`full_bleed`) unless explicitly requested**: Default to `photo_mode: "parallax_gallery"` (with automatic `diptych_frame` pairing for vertical portraits) so heads and artwork are never cropped off.
3. **Never use unverified audio tracks blindly**: Always run `music_supervisor` (Stage 05) so inappropriate user audio tracks are audited and replaced with a fitting CC0 / Public Domain score.
