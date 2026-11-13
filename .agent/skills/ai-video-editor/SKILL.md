---
name: ai-video-editor
description: >-
  End-to-end 21-agent AI photo and video editing skill for family travel, museum trips, vlogs, and unscripted highlights. Use whenever the user asks to edit a video from a folder of clips/photos, run interactive client intake and get the Director's length recommendation (30s-600s), scout/download royalty-free BGM, curate and annotate photos with aesthetic taste and museum plaques, apply 2.5D parallax and diptych photo effects, script documentary voiceover narration, pair vertical 9:16 videos into Live Video+Photo Diptychs, render GPS route progression maps, apply 10-bit DI camera color matching, synthesize spatial soundscapes, supervise editing quality with the Executive Movie Production Director, manage editing projects (create/load/reset), or render 4K 10-bit HLG HDR MP4 videos and NLE timelines (.fcpxml, .edl, .otio).
---

# AI Photo & Video Story Studio Skill (`ai-video-editor`)

## Overview

This skill orchestrates a **21-Agent Big-Budget AI Film & Vlog Post-Production Studio** that transforms raw folders of multi-camera videos (`.MP4`, `.MOV`) and photos (`.JPG`, `.HEIC`, `.PNG`) into a **4K 10-bit HLG HDR Master Video (`final_story.mp4`)** plus professional NLE timelines (`timeline.fcpxml`, `timeline.edl`, `timeline.otio`).

> **Core Architectural Rule**:
> - **Anything mechanical** (contact-sheet grid rendering, FFmpeg segment slicing, ripple timecode shifting, LUFS audio ducking, RGB histogram & Kelvin white-balance measurement, EXIF GPS extraction, JSON/YAML contract serialization) is executed **programmatically** via [`scripts/ai_video_studio_skill.py`](scripts/ai_video_studio_skill.py).
> - **Anything requiring creativity, visual intelligence, or storytelling taste** (inspecting contact sheets to identify artworks/landmarks/expressions, choosing the Cold-Open Teaser Hook, pairing diptychs, scripting voiceover narration, selecting chapter film-stock color looks, scoring spatial soundscapes, and interpreting user revision notes) is **always delegated to the AI Agent** (`read_file` vision inspection or `invoke_subagent`).

All programmatic steps write structured JSON/YAML contracts to disk in `<project_dir>` via the skill's standalone helper CLI:
- **Helper CLI**: [`scripts/ai_video_studio_skill.py`](scripts/ai_video_studio_skill.py)
- **Python Interpreter**:
  - **Windows**: `python`
  - **macOS (2019 MacBook Pro Intel + AMD Radeon Pro / Apple Silicon)**: `python3` (uses `hevc_videotoolbox` hardware encoding, macOS fonts, and native Finder folder picker automatically)

---

## 1. Progressive Disclosure: 21 Sub-Agent Role Playbooks (`references/`)

Read only the specific reference file(s) needed for your current step or when spawning native Antigravity subagents via `define_subagent` / `invoke_subagent` (auto-synchronized from `AgentRegistry` via `python -m video_editing_ai.cli docs-sync`):

<!-- AGENT_REGISTRY_TABLE_START -->
| Stage | Role ID | Sub-Agent Class | Default | Contracts (`Requires -> Produces`) | Reference Playbook & Summary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 00** | `client_intake` | `ClientIntakeAgent` | `True` | *(root)* -> `creative_brief.json` | [references/00_client_intake_agent.md](references/00_client_intake_agent.md) — Interviews the client on story arc, tone, must-haves, and pacing to produce creative_brief.json. |
| **Stage 05** | `dit_ingest` | `DITIngestAgent` | `True` | `creative_brief.json` -> `dit_manifest.json` | [references/01a_dit_ingest_agent.md](references/01a_dit_ingest_agent.md) — Deterministic Digital Imaging Technician (DIT) ingest specialist that performs recursive media discovery, caches technical FFprobe/EXIF metadata by file hash + mtime, normalizes multi-camera DateTimeOriginal clock drift, and probes HDR/SDR color spaces and audio LUFS/SNR. |
| **Stage 05** | `music_supervisor` | `MusicSupervisorAgent` | `True` | `creative_brief.json` -> `bgm_selection.json` | [references/08_music_supervisor_agent.md](references/08_music_supervisor_agent.md) — Analyzes destination, story tone, and creative brief to find, download, and EBU-R128 master mood-matched royalty-free background music across the full video duration, producing bgm_selection.json. |
| **Stage 10** | `culler` | `CullerAgent` | `True` | `creative_brief.json`, `dit_manifest.json` -> `media_index.json`, `annotations.yaml` | [references/01_culler_agent.md](references/01_culler_agent.md) — Consumes cached DIT ingest metadata, runs Whisper ASR, optical-flow camera stability & prosody peak analysis, filters operator filler chatter, splits multi-moment videos into golden windows with elastic usable_in/usable_out handles, and scores selects in media_index.json. |
| **Stage 10** | `audio_vibe` | `AudioVibeAgent` | `True` | `creative_brief.json`, `bgm_selection.json` -> `beat_grid.json` | [references/04_audio_vibe_agent.md](references/04_audio_vibe_agent.md) — Analyzes selected background music strictly for BPM, beat/downbeat grid, and musical energy sections (intro, verse, chorus_drop, outro) into beat_grid.json (delegating audio mixing to Stage 40 sound_mixer). |
| **Stage 15** | `photo_curator` | `PhotoCuratorAgent` | `True` | `creative_brief.json`, `media_index.json` -> `photo_curation.json` | [references/09_photo_curator_agent.md](references/09_photo_curator_agent.md) — Evaluates photos with human portrait perception (open-eye blink rejection, smile boost, subject-vs-bokeh sharpness), non-adjacent zoom-tolerant graph burst clustering, Submodular MMR diversity optimization, and CIE-Lab chromatic diptych pairing. |
| **Stage 15** | `cast_tracker` | `CastContinuityAgent` | `True` | `media_index.json` -> `cast_manifest.json` | [references/15_cast_tracker_agent.md](references/15_cast_tracker_agent.md) — Groups recurring family/cast speakers and faces across clips, tags spoken language per utterance (Mandarin vs. English from faster-whisper segments), and records normalized subject/face bounding boxes (x1, y1, x2, y2) in cast_manifest.json for face-safe Ken Burns framing. |
| **Stage 20** | `story_producer` | `StoryProducerAgent` | `True` | `creative_brief.json`, `media_index.json` -> `paper_edit.json` | [references/02_story_producer_agent.md](references/02_story_producer_agent.md) — Clusters culled selects into 2-8 chapters via spatio-temporal, GPS, and neural semantic boundaries, supports chronological_cinema, hero_journey, thematic_essay, and high_energy_reel story modes, pairs Track V2 cutaways via semantic similarity, sequences chapters with Five-Shot film grammar, and curates a non-duplicating 00:00-00:06 cold_open_hook in paper_edit.json. |
| **Stage 25** | `narrator_voiceover` | `NarratorVoiceoverAgent` | `True` | `creative_brief.json`, `paper_edit.json` -> `narration_script.json` | [references/10_narrator_voiceover_agent.md](references/10_narrator_voiceover_agent.md) — Big-budget documentary scriptwriter and voiceover producer that writes brief, evocative narration cues at chapter openings and finales (never colliding with live camera speech) and synthesizes broadcast voiceover audio tracks. |
| **Stage 30** | `dialogue_editor` | `DialogueEditorAgent` | `True` | `paper_edit.json`, `media_index.json` -> `dialogue_selects.json` | [references/03_picture_editors.md](references/03_picture_editors.md) — Trims A-roll clips on word timestamps within usable_in/usable_out handles, sets -0.35s J-cut audio lead-ins, populates bilingual subtitles, and places Track V2 B-roll cutaways over spoken takes. |
| **Stage 30** | `montage_editor` | `MontageEditorAgent` | `True` | `paper_edit.json`, `beat_grid.json`, `media_index.json` -> `timeline.yaml` | [references/03_picture_editors.md](references/03_picture_editors.md) — Snaps B-roll and photo durations to beat_grid.json beats/downbeats, biases Ken Burns framing toward CastContinuityAgent subject bboxes, supports hybrid candid-reaction chapters, and executes chapter-scoped beat-preserving revisions. |
| **Stage 32** | `camera_operator` | `CameraOperatorAgent` | `True` | `media_index.json`, `paper_edit.json`, `timeline.yaml` -> `cinematography_report.json` | [references/11_camera_operator_agent.md](references/11_camera_operator_agent.md) — Big-budget camera operator and reframing specialist that pairs vertical 9:16 video clips with complementary vertical photos into Live Video+Photo Diptychs and applies optical stabilization to handheld walking B-roll. |
| **Stage 35** | `lead_editor` | `LeadAssemblyEditorAgent` | `True` | `paper_edit.json`, `media_index.json`, `beat_grid.json`, `timeline.yaml` -> `assembly_report.json`, `timeline.yaml` | [references/16_lead_editor_agent.md](references/16_lead_editor_agent.md) — Stitches chapter timelines into one continuous EditTimeline, inserts 0.35-0.75s J/L-cut audio bridges across chapter transitions, keeps Track A3 BGM continuous across chapters, and clamps V1/V2 trims within usable_in/usable_out handles. |
| **Stage 40** | `title_designer` *(alias: `motion_graphics`)* | `MotionGraphicsTitleAgent` | `True` | `creative_brief.json`, `paper_edit.json`, `timeline.yaml` -> `graphics_manifest.json`, `motion_graphics_plan.json` | [references/12_motion_graphics_agent.md](references/12_motion_graphics_agent.md) — Consolidates all Track V3 graphics (opening card, chapter lower-thirds, GPS route progression cards, dialogue quote badges, optional EXIF stamps, and closing card), configures bilingual subtitles (PingFang SC / Avenir Next / platform fallbacks), and clamps RGBA peak white to HDR-safe diffuse white. |
| **Stage 40** | `vfx_designer` | `VFXAgent` | `True` | `paper_edit.json`, `beat_grid.json`, `timeline.yaml` -> `vfx_manifest.json` | [references/06_vfx_agent.md](references/06_vfx_agent.md) — AI-guided Stage-40 specialist that adds warm golden glow, subtle portrait vignettes, and beat-drop contrast pops to timeline.effects_track and writes vfx_manifest.json. |
| **Stage 40** | `hdr_colorist` *(alias: `colorist`)* | `HDRColoristAgent` | `True` | `media_index.json`, `paper_edit.json`, `timeline.yaml` -> `color_transform_manifest.json`, `color_grade_plan.json` | [references/13_colorist_agent.md](references/13_colorist_agent.md) — Inspects per-clip color tags (10-bit BT.2020 HLG vs. Camera Log vs. 8-bit BT.709/sRGB stills) from media_index.json, maps SDR clips & stills to ~203-nit HLG diffuse white (~75% HLG signal), and writes color_transform_manifest.json. |
| **Stage 40** | `sound_mixer` *(alias: `sound_designer`)* | `SoundDesignerMixerAgent` | `True` | `media_index.json`, `paper_edit.json`, `beat_grid.json`, `timeline.yaml` -> `audio_mix_manifest.json`, `sound_design_plan.json` | [references/14_sound_designer_agent.md](references/14_sound_designer_agent.md) — Post-assembly re-recording mixer that normalizes dialogue clips toward -14 LUFS, generates frame-accurate BGM ducking (-14 dB under speech, -20 dB Nat-Sound under pure B-roll, and <=-28 dB deep muting during has_diegetic_music clips), applies 0.33s audio crossfades, and synthesizes spatial foley/ambience stems. |
| **Stage 50** | `showrunner` | `ShowrunnerAgent` | `True` | `creative_brief.json`, `paper_edit.json`, `timeline.yaml` -> `pacing_audit.json`, `timeline.yaml`, `storyboard.md` | [references/05_showrunner_agent.md](references/05_showrunner_agent.md) — Deterministic compliance and pacing validator that verifies creative_brief.json must-haves, flags >3 consecutive static stills, flags pure B-roll >4.5s, checks timeline gaps/overlaps, and emits chapter-scoped RevisionDirectives without mutating montage cuts off the beat grid. |
| **Stage 55** | `visual_qc` | `ProxyWatchdownQCAgent` | `True` | `pacing_audit.json`, `timeline.yaml` -> `visual_qc_report.json` | [references/17_visual_qc_agent.md](references/17_visual_qc_agent.md) — Runs between Stage 50 (showrunner) and Stage 60 (production_director), sampling low-resolution boundary frames around cuts and Track V3 overlays to detect black flashes, frozen tail frames, subtitle/lower-third bounding-box overlaps, clipped luminance blocks, and uncrossfaded audio cut spikes. |
| **Stage 60** | `production_director` | `ProductionDirectorAgent` | `True` | `creative_brief.json`, `paper_edit.json`, `pacing_audit.json`, `visual_qc_report.json`, `timeline.yaml` -> `director_review.json`, `timeline.yaml`, `storyboard.md` | [references/07_production_director_agent.md](references/07_production_director_agent.md) — Sole creative sign-off gatekeeper evaluating pacing_audit.json and visual_qc_report.json against the 6-dimension cinematic rubric, issuing beat-preserving chapter-scoped RevisionDirectives when needed, and writing director_review.json. |
| **Stage 65** | `nle_exporter` | `NLEExportAgent` | `True` | `timeline.yaml` -> `timeline.fcpxml` | [references/18_nle_exporter_agent.md](references/18_nle_exporter_agent.md) — Emits a frame-accurate timeline.fcpxml (FCPXML 1.11 for DaVinci Resolve & Final Cut Pro) mapping Track V1 spine, V2 connected B-roll cutaways, V3 title lanes, A1-A3 audio lanes, and chapter markers alongside EDL, OTIO, SRT, and ASS sidecars. |
<!-- AGENT_REGISTRY_TABLE_END -->

Additional rendering & contract schema specification: [references/10_contracts_and_rendering.md](references/10_contracts_and_rendering.md)

---

## 2. Six-Phase End-to-End Editing Workflow

When the user asks you to edit a video from a media folder, follow these 6 phases:

### Phase 1: Project Setup & Media Library Inspection (`inspect-library` & `project`)
Scan the raw media folder to detect video/photo counts, EXIF camera models, and the **Executive Director's Recommended Video Length (`30s–600s`)**:
```powershell
python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py inspect-library --media-dir "<media_dir>" --project-dir "<project_dir>" --prompt "<user_prompt>"
```
- Read `<project_dir>/library_inspection.json` to inspect `director_duration_recommendation` (`director_recommended_duration_sec`, `recommended_photo_plates`, `director_duration_rationale`).
- To create, load, list, or reset a project workspace:
  ```powershell
  python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py project --action create --name "Featured Story" --media-dir "<media_dir>" --project-dir "<project_dir>"
  ```

### Phase 2: Interactive Multi-Choice Client Intake & Director Recommendations (`ask_question` & `intake-refine`)
Read `ask_question_tool_payload` from `<project_dir>/library_inspection.json` and call Antigravity's native `ask_question` tool to present interactive single-choice and **multi-select (`is_multi_select: true`)** questions with `(Recommended)` prefixed on the Director's context-aware recommendations (covering Target Duration `30s–600s`, Multi-Select 10-Mode Photo Effects Suite, Soundtrack Direction, and Multi-Select Must-Include Chapters). Then run `intake-refine` to lock `creative_brief.json`:
```powershell
python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py intake-refine --media-dir "<media_dir>" --project-dir "<project_dir>" --prompt "<user_prompt>" --duration 155 --photo-mode cinematic_depth_3d --photo-modes "cinematic_depth_3d,split_diopter_focus,contact_sheet_zoom,glassmorphic_pedestal,medium_format_slide,floating_tilt_plate,criterion_atelier_split,parallax_gallery,curator_loupe,editorial_magazine,polaroid_scrapbook,diptych_frame" --overlay-fxs "golden_light_leak,museum_spotlight,champagne_bokeh,nostalgic_film_grain,anamorphic_streak,prism_chromatic_edge,velvet_vignette_rembrandt,fine_art_rag_texture" --bgm-mood auto_match
```

### Phase 3: Run the 21-Agent Pipeline & Delegate Creative Curation to the AI Agent
- **Option A — Full 21-Agent Pipeline + Contact Sheet Generation (`run-pipeline`)**:
  ```powershell
  python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py run-pipeline --media-dir "<media_dir>" --project-dir "<project_dir>" --prompt "<user_prompt>" --duration 155 --photo-mode parallax_gallery --render
  ```
- **Option B — Mechanical Contact Sheets → AI Agent Visual Curation → Mechanical Persistence (`render-contact-sheets` & `apply-vision-curation`)**:
  1. **Mechanical Step**: Generate numbered `5x4` photo contact sheets (`contact_sheet_01.jpg`, ...) + video subclips keyframe sheet (`video_subclips_sheet.jpg`) in `<project_dir>/contact_sheets/`:
     ```powershell
     python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py render-contact-sheets --project-dir "<project_dir>"
     ```
  2. **AI Agent Creative Step**: Call `read_file` on `<project_dir>/contact_sheets/contact_sheet_01.jpg` (and `video_subclips_sheet.jpg`) to visually inspect every candidate photo & clip. Identify specific artworks, landmarks, facial expressions, and vertical diptych pairs, and reject blurry/accidental shots.
  3. **Mechanical Persistence Step (`apply-vision-curation` & `apply-video-curation`)**: Write your creative photo & video clip decisions (golden-window `[in_tc, out_tc]`, semantic transitions, video & photo captions, and 16-mode photo effects) into `photo_curation.json`, `media_index.json`, and `timeline.yaml`:
     ```powershell
     python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py apply-vision-curation --project-dir "<project_dir>" --overrides-json "[{\"tile_id\":\"#01\",\"selected\":true,\"hero_pick\":true,\"recommended_photo_mode\":\"curator_loupe\",\"overlay_fx\":\"museum_spotlight\",\"caption_title\":\"Curated Architectural Detail\",\"caption_subtitle\":\"Gallery Exhibition Study\"}]"
     python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py apply-video-curation --project-dir "<project_dir>" --overrides-json "{\"videos\":{\"vid_broll_01\":{\"in_tc\":1.0,\"out_tc\":7.5,\"recommended_transition\":\"dissolve\",\"annotation_title\":\"Opening Establishing Vista\",\"annotation_subtitle\":\"Panoramic Tracking Shot\"}}}"
     ```
- **Option C — AI-Directed Cold-Open Teaser Hook (`0.0s–3.8s`) & Story Architecture (`apply-story-plan`)**:
  1. Read `<project_dir>/candidate_story_digest.json` and `<project_dir>/contact_sheets/video_subclips_sheet.jpg` to choose the most emotionally magnetic 3.0s–3.8s **Cold-Open Teaser Hook (`clip_000_hook`)** before the Main Title Card (`title_opening`).
  2. Apply your creative story plan overrides and optionally reassemble `timeline.yaml`:
     ```powershell
     python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py apply-story-plan --project-dir "<project_dir>" --overrides-json "{\"cold_open_subclip_id\":\"vid_broll_01\",\"cold_open_hook_caption\":\"Opening Teaser Hook\"}" --reassemble-timeline
     ```
- **Option D — Dedicated Royalty-Free BGM Scouting & Full-Length Mastering Agent (`scout-bgm` & `apply-bgm-plan`)**:
  1. **Mechanical Step**: Search and download royalty-free candidate tracks from online catalogs (Pixabay Music / Incompetech 1,443-track CC-BY API / Public Domain) and measure each candidate's exact `duration_sec`, `mean_volume_db`, and `tempo_bpm`:
     ```powershell
     python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py scout-bgm --project-dir "<project_dir>" --keywords "cinematic,chamber,waltz,piano,documentary" --mood-profile classical_chamber_documentary
     ```
  2. **AI Agent Creative Step**: Read `<project_dir>/bgm_candidates.json` (or delegate to a dedicated BGM subagent via `invoke_subagent`) to pick the best single track or **Multi-Chapter BGM Suite** matching the emotional progression of the chapters.
  3. **Mechanical Mastering Step**: Chain/crossfade the selected tracks and master to **100% of `timeline.total_duration`** with broadcast **EBU R128 (`-14.0 LUFS`)** loudness normalization so the music never ends early and sits at a warm, vibrant volume:
     ```powershell
     python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py apply-bgm-plan --project-dir "<project_dir>" --tracks "valse_gymnopedie.mp3,evening_debussy_chamber.mp3" --target-lufs -14.0 --bgm-volume 0.90
     ```
- **Documentary Reading-Paced Scene Durations, 16 Premium Photo Modes & Unified Frosted-Glass Captions**:
  - For annotated educational/travel scenes, give viewers comfortable reading time (`160–180 WPM` + visual appreciation): **`5.5s–6.8s`** per photo plate and **`6.5s–9.0s`** per video clip.
  - Attach `caption_title`, `caption_subtitle`, and `camera_badge` directly to `item.ken_burns` on both `photo` and `video` clips so the renderer composites a single unified **Smoked-Glass Museum Plaque** (avoiding redundant overlapping Track V3 subtitle boxes).
  - Curate a diverse choreography of the **16 Premium Photo Presentation Modes (`mode`)** and **8 Composable Atmospheric Finishers (`overlay_fx`)** across chapters (preview all 16 modes side-by-side in a 4x4 showcase sheet via `preview-photo-effects --project-dir "<project_dir>"`):
    - **Modes (`mode`)**: `cinematic_depth_3d` (3-plane saliency depth + bokeh rack-focus), `contact_sheet_zoom` (`3x2` darkroom contact sheet -> hero push-in), `split_diopter_focus` (`54/46` wide + `2.2x` optical macro split), `glassmorphic_pedestal` (frosted acrylic vitrine + live backdrop refraction), `medium_format_slide` (`6x6` 120 film transparency mount + notches), `floating_tilt_plate` (3D perspective projective orbit), `criterion_atelier_split` (3-viewport asymmetric bento grid), `triptych_cinematic` (3-panel staggered storyboard), `polaroid_scrapbook` (tilted vintage instant-film print stack), `curator_loupe` (`1.65x` optical magnifying spotlight), `editorial_magazine` (asymmetric `62/38` luxury monograph spread), `cinema_anamorphic` (`2.39:1` widescreen + 35mm highlight halation), `parallax_gallery`, `diptych_frame`, `gallery_frame`, `full_bleed`.
    - **Finishers (`overlay_fx`)**: `golden_light_leak`, `champagne_bokeh`, `museum_spotlight`, `nostalgic_film_grain`, `anamorphic_streak`, `prism_chromatic_edge`, `velvet_vignette_rembrandt`, `fine_art_rag_texture`, `none`.

### Phase 4: Surgical Timeline Revisions (`edit-cut`) & Fast 2-Second Segment Previews (`render-segment`)
When the user asks for targeted edits (e.g., *"trim the second clip by 1.5s"*, *"drop clip_005"*, *"change the plaque on clip_003"*, *"lower BGM volume under speech"*), **never re-run the entire 21-agent pipeline**:
1. **AI Agent Creative Step**: Read `<project_dir>/timeline.yaml` and translate the user's natural-language feedback into surgical operation(s) (`update-plaque`, `trim`, `drop`, `swap`, `move`, `set-cold-open`, `audio-mix`).
2. **Mechanical Ripple Edit (`edit-cut`)**: Execute the surgical edit with automatic downstream ripple-shifting across V1, V2, V3 titles, VFX, markers, and subtitles:
   ```powershell
   python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py edit-cut --project-dir "<project_dir>" --action trim --clip-id clip_001 --delta-sec -1.0
   ```
3. **Mechanical Fast Segment Render (`render-segment`)**: Render only the affected `[start_sec, end_sec]` window in ~2 seconds (`720p` fast preview or `--full-quality` 4K HLG HDR) and inspect frames via `read_file`:
   ```powershell
   python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py render-segment --project-dir "<project_dir>" --start 0 --end 10 --fast-preview --extract-frames "1,5,8"
   ```

### Phase 5: Executive Movie Production Director Visual Screening (`render-screening-sheet` & `director-audit`) & Full 4K HDR Render (`render-video`)
1. Render the **4x3 Multimodal Director Screening Sheet** (`<project_dir>/cache/curation/director_screening_sheet.jpg`) and inspect it via `read_file` to verify that every chapter's photo modes, burned-in museum plaques, Track V3 lower-third video captions, and transitions look cohesive:
   ```powershell
   python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py render-screening-sheet --project-dir "<project_dir>"
   ```
2. Apply any final transition, duration, or Track V3 lower-third refinements via `apply-timeline-intelligence`, verify `director_review.json` (`overall_quality_score >= 9.0`), and render the full 4K 10-bit HLG HDR master:
```powershell
python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py render-video --project-dir "<project_dir>" --extract-frames "2,6,16,30,52"
```

### Phase 6: Interactive 5-Tab Web Studio (`serve`)
To let the user interactively manage projects, co-refine intake, watch real-time Agent Doing + Director Thinking streams, or download the newly rendered `.mp4` in their browser:
```powershell
python -m video_editing_ai.cli serve --project-dir "<project_dir>" --host 0.0.0.0 --port 8000
```

---

## 3. Common Mistakes to Avoid

1. **Do not hardcode a 2-minute (`120s`) duration cap**: Always check `director_duration_recommendation` in `library_inspection.json` (`30s–600s`) and let photo plate count scale dynamically with duration.
2. **Never crop vertical or 4:3 photos to 16:9 (`full_bleed`) unless explicitly requested**: Default to `photo_mode: "parallax_gallery"` (with automatic `diptych_frame` pairing for vertical portraits) so heads and artwork are never cropped off.
3. **Never use unverified audio tracks blindly**: Always run `music_supervisor` (Stage 05) so inappropriate user audio tracks are audited and replaced with a fitting CC0 / Public Domain score.
