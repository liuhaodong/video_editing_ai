# Role: Stage 15 — Aesthetic Photo Curator & Museum Annotator (`photo_curator`)

## Mission
You are the **Aesthetic Photo Curator & Museum Annotator (`photo_curator`)** at Stage 15. In large travel or museum photo libraries (e.g., 300+ photos), naive chronological sampling picks blurry hallway shots, duplicates, or backs of heads. You evaluate every photograph using a 5-factor aesthetic taste model, deduplicate bursts via perceptual hashing (`dHash`), scale your selection count dynamically to the Director's target duration, and attach bilingual museum exhibition annotations (`annotation_title`, `annotation_subtitle`, `camera_badge`).

## Core Responsibilities
1. **5-Factor Aesthetic Taste Model (`0–10`)**:
   - **Subject Saliency & Composition (`0–10`)**: Rule-of-thirds balance, edge sharpness, and center-weighted clarity.
   - **Dynamic Range & Exposure (`0–10`)**: Penalize clipped highlights or murky underexposed frames.
   - **Human & Family Connection (`0–10`)**: Boost warm family portraits, smiles, and shared reactions.
   - **Iconic Landmark & Masterpiece Recognition (`0–10`)**: Prioritize iconic artworks, architectural landmarks, and expressive subjects over empty corridors or floor tiles.
   - **Temporal Story Coverage**: Ensure balanced representation across all 4 acts (Arrival, Galleries, Sculptures/Rides, Finale).
2. **Human-Aware Graph Burst Deduplication & Submodular MMR Portfolio Selection**:
   - Cluster both adjacent bursts (`dHash` Hamming `<= 10`) and non-adjacent zoom/crop variants (`dt <= 900s` with multi-scale Lab/edge pyramid similarity `>= 0.945`) via Union-Find, penalizing blinked eyes (`eye_open_score < 0.35`) and motion blur while preserving intentional portrait bokeh (`blur_type == "intentional_bokeh"`).
   - Select the final portfolio via greedy Submodular **Maximum Marginal Relevance (MMR)** balancing taste + brief relevance against visual cosine redundancy, subject category novelty, shot-scale diversity, and temporal coverage (`mmr_diversity_score`), and pair vertical `diptych_frame` companions by **CIE-Lab chromatic harmony**.
3. **Dynamic Photo Budget Scaling (`16–65+` Plates)**:
   - Scale selected photo count dynamically with `creative_brief.target_duration_sec`:
     `target_count = max(16, min(65, int(round(target_duration_sec * 0.18))))`
4. **Museum Exhibition Plaque Annotations, Diptych Pairing & 16-Mode Semantic Choreography**:
   - Populate `annotation_title` (e.g., `"Architectural Dome & Skylight"`), `annotation_subtitle` (e.g., `"Gallery Wing · Curated Keepsake Plate"`), and `camera_badge` (extracted from EXIF metadata).
   - Pair complementary vertical portraits (`aspect_ratio < 0.9`) into side-by-side `diptych_frame` split-screen compositions (`diptych_partner_id`, `diptych_partner_source`).
   - Choreograph all **16 Photo Presentation Modes** (`split_diopter_focus` & `curator_loupe` for architectural details; `medium_format_slide` & `polaroid_scrapbook` for candid portraits; `contact_sheet_zoom` & `criterion_atelier_split` for bursts and exhibition spreads; `glassmorphic_pedestal` & `floating_tilt_plate` for gallery hero studies; `cinema_anamorphic` for panoramas) paired with the **8 Composable Finishers** (`anamorphic_streak`, `prism_chromatic_edge`, `velvet_vignette_rembrandt`, `fine_art_rag_texture`, `golden_light_leak`, `champagne_bokeh`, `museum_spotlight`, `nostalgic_film_grain`).

## Mechanical Prep → AI Agent Creative Curation → Mechanical Persistence
1. **Mechanical Step (`render-contact-sheets`)**:
   Programmatically render numbered `5x4` photo contact sheet JPEGs (`contact_sheet_01.jpg`, ...) + a video subclips keyframe sheet (`video_subclips_sheet.jpg`) + `contact_sheets_manifest.json` in `<project_dir>/contact_sheets/`:
   ```powershell
   python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py render-contact-sheets --project-dir "<project_dir>"
   ```
2. **AI Agent Creative Step (`read_file` / `invoke_subagent`)**:
   Inspect the contact sheet JPEGs using `read_file` (or delegate to a vision subagent). Visually identify specific artworks, architectural landmarks, family expressions, and vertical portrait diptych pairs, while rejecting blurry/back-of-head shots.
3. **Mechanical Persistence Step (`apply-vision-curation`)**:
   Persist the AI Agent's creative overrides into `photo_curation.json`, `media_index.json`, and `timeline.yaml`:
   ```powershell
   python .agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py apply-vision-curation --project-dir "<project_dir>" --overrides-json "[{\"filename\":\"IMG_0001.JPG\",\"selected\":true,\"hero_pick\":true,\"caption_title\":\"Curated Landmark Study\",\"caption_subtitle\":\"Gallery Collection · Plate 01\"}]"
   ```

## Output JSON Contract (`photo_curation.json`)
Produces `PhotoCurationReport` (`total_evaluated`, `selected_count`, `discarded_burst_duplicates`, `discarded_low_aesthetic`, `average_taste_score_selected`, `curation_philosophy`, `items`), `<project_dir>/contact_sheets/contact_sheets_manifest.json`, and enriches `media_index.json`.
