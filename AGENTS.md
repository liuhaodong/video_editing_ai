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
  - Delegate parallel multi-aspect reviews (`PhotoCurator`, `StoryProducer`, `ProductionDirector`) via `invoke_subagent` when curating large libraries.
- **Skill Helper CLI (`.agent/skills/ai-video-editor/scripts/ai_video_studio_skill.py`)**:
  - Run all deterministic/mechanical media operations through `ai_video_studio_skill.py` (`inspect-library`, `intake-refine`, `run-pipeline`, `render-contact-sheets`, `apply-vision-curation`, `apply-video-curation`, `apply-story-plan`, `scout-bgm`, `apply-bgm-plan`, `edit-cut`, `render-segment`, `render-screening-sheet`, `render-video`).
  - For targeted user feedback on an existing cut, always use `edit-cut` + `render-segment` (~2s preview) instead of re-running the full 21-agent pipeline.

## 4. Git Hygiene
- Never `git add` or commit generated project folders (`*_project/`, `workspace_project/`, `active_video_project/`), `.venv/`, or rendered `.mp4`/`.mov`/`.fcpxml` outputs.
