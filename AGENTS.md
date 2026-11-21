# Antigravity Studio Workspace Rules (`video_editing_ai`)

When working in this repository inside **Google Antigravity**, **Antigravity is the primary creative director and orchestrator**.

## 1. Activate the Editing Skill Automatically
- Whenever the user asks to inspect, curate, plan, edit, revise, or render photos or videos into a film or highlight reel, read and follow `.agent/skills/ai-video-editor/SKILL.md` (and load only the specific `.agent/skills/ai-video-editor/references/*.md` playbooks needed for the current stage).

## 2. Python Environment & Verification
- Always execute Python commands with the repository virtualenv interpreter:
  - macOS / Linux: `.venv/bin/python` (do not assume bare `python` is on `PATH`).
  - Windows: `.venv\Scripts\python.exe`
- Before running renders or tests on a new machine, run `./scripts/setup.sh` (macOS/Linux) or `.\scripts\setup.ps1` (Windows), or verify with `.venv/bin/python -m video_editing_ai.cli doctor`.

## 3. Creative vs. Mechanical Division of Labor
- **Antigravity Agent (Creative Decisions)**:
  - Run interactive multi-choice client intake via `ask_question` using `ask_question_tool_payload` from `library_inspection.json`.
  - Inspect generated contact sheets (`contact_sheets/contact_sheet_*.jpg`, `contact_sheets/video_subclips_sheet.jpg`) and `director_screening_sheet.jpg` visually via `read_file`.
  - Select hero frames, reject blinks/accidental shots, choose Cold-Open teaser hooks, pair chromatic vertical diptychs, pick BGM suites, and write concise documentary titles/plaques.
  - Render narrative chapter cues as luxury **on-screen Criterion/museum frosted-glass typography** by default; generate spoken voiceover TTS audio only when explicitly requested by the user.
  - Delegate parallel multi-aspect reviews (`PhotoCurator`, `StoryProducer`, `ColorGrader`, `ProductionDirector`) via `invoke_subagent` when curating large libraries or auditing DI film-print looks.
  - When pulling media directly from Google Photos by date range, prefer the zero-setup **Visual Overview Curation & Small-Batch Download** browser workflow: delegate to `invoke_subagent(TypeName="browser", Model="flash")` in ONE tab on `https://photos.google.com` (pausing cleanly if the user needs to sign in, then resuming via `send_message`). Scrub the **main library timeline** + sidebar **Favorites** tab (immune to `EzkLib` search RPC rate-limits and keeping `Include: Shared with me` unchecked when partner photos are already saved in the main library), visually inspect overview grids to map the story arc and skip clutter/bursts, download curated standout moments in **`5–15`-item small ZIP batches** (avoiding `accounts.google.com` download-redirect blocks from 50+ item bulk ZIPs; single-item fallback if needed), and ingest via `ai_video_studio_skill.py sync-google-photos --from-downloads "<browser_sandbox/downloads>" --favorites "<starred_filenames>" --download-all` (never ask users for Google Cloud OAuth files when interactive browser login is available).
  - When publishing `final_story.mp4` to YouTube, ALWAYS run `ai_video_studio_skill.py publish-youtube --project-dir "<project_dir>" --prepare` first, visually inspect `youtube_thumbnail.jpg` via `read_file`, present the exact Title, Visibility (`unlisted`/`private`/`public`), Made-for-Kids flag, `00:00` Chapters, and File Size, and obtain explicit user confirmation via `ask_question` BEFORE running `--confirmed --confirm-token <token>` or launching a `browser` upload on `https://studio.youtube.com`.
- **Skill Helper CLI (`.agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py`)**:
  - Run all deterministic/mechanical media operations through `ai_video_studio_skill.py` (`inspect-library`, `intake-refine`, `run-pipeline`, `render-contact-sheets`, `apply-vision-curation`, `apply-video-curation`, `apply-story-plan`, `preview-color-grades`, `recommend-color-grade`, `render-color-scopes`, `apply-color-grade`, `scout-bgm`, `apply-bgm-plan`, `edit-cut`, `render-segment`, `render-screening-sheet`, `render-video`, `publish-youtube`).
  - For targeted user feedback on an existing cut, always use `edit-cut` + `render-segment` (~2s preview) instead of re-running the full 22-agent pipeline.

## 4. Git Hygiene
- Never `git add` or commit generated project folders (`*_project/`, `workspace_project/`, `active_video_project/`), `.venv/`, or rendered `.mp4`/`.mov`/`.fcpxml` outputs.
