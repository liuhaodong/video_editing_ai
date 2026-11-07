---
name: ai-video-editor
description: >-
  End-to-end 11-agent AI photo and video editing skill for family travel, museum trips, vlogs, and unscripted highlights. Use whenever the user asks to edit a video from a folder of clips/photos, run interactive client intake and get the Director's length recommendation (30s-600s), scout/download royalty-free BGM, curate and annotate photos with aesthetic taste and museum plaques, apply 2.5D parallax and diptych photo effects, supervise editing quality with the Executive Movie Production Director, manage editing projects (create/load/reset), or render 4K 10-bit HLG HDR MP4 videos and NLE timelines (.fcpxml, .edl, .otio).
---

# AI Photo & Video Story Studio Skill (`ai-video-editor`)

## Overview

This skill orchestrates an **11-Agent AI Film & Vlog Editing Room** that transforms raw folders of multi-camera videos (`.MP4`, `.MOV`) and photos (`.JPG`, `.HEIC`, `.PNG`) into a **4K 10-bit HLG HDR Master Video (`final_story.mp4`)** plus professional NLE timelines (`timeline.fcpxml`, `timeline.edl`, `timeline.otio`).

> **Core Architectural Rule**:
> - **Anything mechanical** (contact-sheet grid rendering, FFmpeg segment slicing, ripple timecode shifting, LUFS audio ducking, JSON/YAML contract serialization) is executed **programmatically** via [`scripts/ai_video_studio_skill.py`](scripts/ai_video_studio_skill.py).
> - **Anything requiring creativity, visual intelligence, or storytelling taste** (inspecting contact sheets to identify artworks/landmarks/expressions, choosing the Cold-Open Teaser Hook, pairing diptychs, writing poetic exhibition plaques, and interpreting user revision notes) is **always delegated to the AI Agent** (`read_file` vision inspection or `invoke_subagent`).

All programmatic steps write structured JSON/YAML contracts to disk in `<project_dir>` via the skill's standalone helper CLI:
- **Helper CLI**: [`scripts/ai_video_studio_skill.py`](scripts/ai_video_studio_skill.py)
- **Python Interpreter**:
  - **Windows**: `C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe` (or `python`)
  - **macOS (2019 MacBook Pro Intel + AMD Radeon Pro / Apple Silicon)**: `python3` (uses `hevc_videotoolbox` hardware encoding, macOS fonts, and native Finder folder picker automatically)

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

### Phase 3: Run the 11-Agent Pipeline & Delegate Creative Curation to the AI Agent
- **Option A — Full 11-Agent Pipeline + Contact Sheet Generation (`run-pipeline`)**:
  ```powershell
  C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py run-pipeline --media-dir "<media_dir>" --project-dir "<project_dir>" --prompt "<user_prompt>" --duration 155 --photo-mode parallax_gallery --render
  ```
- **Option B — Mechanical Contact Sheets → AI Agent Visual Curation → Mechanical Persistence (`render-contact-sheets` & `apply-vision-curation`)**:
  1. **Mechanical Step**: Generate numbered `5x4` photo contact sheets (`contact_sheet_01.jpg`, ...) + video subclips keyframe sheet (`video_subclips_sheet.jpg`) in `<project_dir>/contact_sheets/`:
     ```powershell
     C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py render-contact-sheets --project-dir "<project_dir>"
     ```
  2. **AI Agent Creative Step**: Call `read_file` on `<project_dir>/contact_sheets/contact_sheet_01.jpg` (and `video_subclips_sheet.jpg`) to visually inspect every candidate photo & clip. Identify specific artworks, landmarks, facial expressions, and vertical diptych pairs, and reject blurry/accidental shots.
  3. **Mechanical Persistence Step**: Write your creative decisions into `photo_curation.json`, `media_index.json`, and `timeline.yaml`:
     ```powershell
     C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py apply-vision-curation --project-dir "<project_dir>" --overrides-json "[{\"filename\":\"IMG_7553.JPG\",\"selected\":true,\"hero_pick\":true,\"caption_title\":\"Mona Lisa (La Joconde)\",\"caption_subtitle\":\"Leonardo da Vinci · Denon Wing, Room 711\"}]"
     ```
- **Option C — AI-Directed Cold-Open Teaser Hook (`0.0s–3.8s`) & Story Architecture (`apply-story-plan`)**:
  1. Read `<project_dir>/candidate_story_digest.json` and `<project_dir>/contact_sheets/video_subclips_sheet.jpg` to choose the most emotionally magnetic 3.0s–3.8s **Cold-Open Teaser Hook (`clip_000_hook`)** before the Main Title Card (`title_opening`).
  2. Apply your creative story plan overrides and optionally reassemble `timeline.yaml`:
     ```powershell
     C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py apply-story-plan --project-dir "<project_dir>" --overrides-json "{\"cold_open_subclip_id\":\"sub_002\",\"cold_open_hook_caption\":\"Inside the Louvre · A Family Day in Paris\"}" --reassemble-timeline
     ```
- **Option D — Dedicated Royalty-Free BGM Scouting & Full-Length Mastering Agent (`scout-bgm` & `apply-bgm-plan`)**:
  1. **Mechanical Step**: Search and download royalty-free candidate tracks from online catalogs (Pixabay Music / Incompetech 1,443-track CC-BY API / Public Domain) and measure each candidate's exact `duration_sec`, `mean_volume_db`, and `tempo_bpm`:
     ```powershell
     C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py scout-bgm --project-dir "<project_dir>" --keywords "paris,accordion,waltz,cafe,piano" --mood-profile parisian_classical_museum
     ```
  2. **AI Agent Creative Step**: Read `<project_dir>/bgm_candidates.json` (or delegate to a dedicated BGM subagent via `invoke_subagent`) to pick the best single track or **Multi-Chapter BGM Suite** matching the emotional progression of the chapters.
  3. **Mechanical Mastering Step**: Chain/crossfade the selected tracks and master to **100% of `timeline.total_duration`** with broadcast **EBU R128 (`-14.0 LUFS`)** loudness normalization so the music never ends early and sits at a warm, vibrant volume:
     ```powershell
     C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py apply-bgm-plan --project-dir "<project_dir>" --tracks "parisian_accordion.mp3,valse_gymnopedie.mp3,evening_debussy_chamber.mp3,jazz_brunch_bistro.mp3" --target-lufs -14.0 --bgm-volume 0.90
     ```
- **Documentary Reading-Paced Scene Durations & Unified Frosted-Glass Captions**:
  - For annotated educational/travel scenes, give viewers comfortable reading time (`160–180 WPM` + visual appreciation): **`5.5s–6.8s`** per photo plate and **`6.5s–9.0s`** per video clip.
  - Attach `caption_title`, `caption_subtitle`, and `camera_badge` directly to `item.ken_burns` on both `photo` and `video` clips so the renderer composites a single unified **Apple TV+ / National Geographic Frosted-Glass Documentary Lower-Third Card** (avoiding redundant overlapping Track V3 subtitle boxes).

### Phase 4: Surgical Timeline Revisions (`edit-cut`) & Fast 2-Second Segment Previews (`render-segment`)
When the user asks for targeted edits (e.g., *"trim the second clip by 1.5s"*, *"drop clip_005"*, *"change the plaque on clip_003"*, *"lower BGM volume under speech"*), **never re-run the entire 11-agent pipeline**:
1. **AI Agent Creative Step**: Read `<project_dir>/timeline.yaml` and translate the user's natural-language feedback into surgical operation(s) (`update-plaque`, `trim`, `drop`, `swap`, `move`, `set-cold-open`, `audio-mix`).
2. **Mechanical Ripple Edit (`edit-cut`)**: Execute the surgical edit with automatic downstream ripple-shifting across V1, V2, V3 titles, VFX, markers, and subtitles:
   ```powershell
   C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py edit-cut --project-dir "<project_dir>" --action trim --clip-id clip_001 --delta-sec -1.0
   ```
3. **Mechanical Fast Segment Render (`render-segment`)**: Render only the affected `[start_sec, end_sec]` window in ~2 seconds (`720p` fast preview or `--full-quality` 4K HLG HDR) and inspect frames via `read_file`:
   ```powershell
   C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py render-segment --project-dir "<project_dir>" --start 0 --end 10 --fast-preview --extract-frames "1,5,8"
   ```

### Phase 5: Executive Movie Production Director Audit (`director-audit`) & Full 4K HDR Render (`render-video`)
Read `<project_dir>/director_review.json` (or run `director-audit`) to verify `approved_for_master_render == true` and `overall_quality_score >= 9.0` (Grade `A` or `A+`), then render the full 4K 10-bit HLG HDR master when ready:
```powershell
C:\Users\hihao\AppData\Local\Programs\Python\Python312\python.exe C:\Users\hihao\Documents\projects\video_editing_ai\.agent\skills\ai-video-editor\scripts\ai_video_studio_skill.py render-video --project-dir "<project_dir>" --extract-frames "2,6,16,30,52"
```

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
