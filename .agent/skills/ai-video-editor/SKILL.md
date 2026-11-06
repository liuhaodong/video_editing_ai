---
name: ai-video-editor
description: End-to-end AI photo and video analysis, story timeline planning, and open-source programmatic video rendering (Ken Burns, Gallery Frames, RGBA Titles, Subtitles, NLE FCPXML/EDL/OTIO exports). Use when the user asks to analyze a folder of videos/photos, generate an editing timeline, or render a story-driven video.
---

# AI Photo & Video Editor Skill

Use this skill whenever the user wants to analyze raw videos/photos, plan a story edit, tweak a text-based timeline (`timeline.yaml`), or render a finished video using open-source tools (`ffmpeg` + `Pillow` + `faster-whisper` + `google-antigravity`).

## 3-Step Workflow

### Step 1: Media Analysis & Timestamped Annotation
Run the CLI or Python API to scan `.MOV`/`.MP4` videos and `.HEIC`/`.JPG` photos, extract keyframes and contact sheets, transcribe spoken dialogue with timestamps via `faster-whisper`, and output `annotations.yaml`:

```powershell
python -m video_editing_ai.cli analyze "<media_dir>" -o "<project_dir>" --bgm "<optional_bgm.mp3>"
```

### Step 2: AI Story Plan & Editable Text Timeline
Generate a 4-act narrative `storyboard.md` and `timeline.yaml` (including video trims that preserve complete spoken sentences, Gallery Frame & Ken Burns photo animations, RGBA title cards, dialogue callouts, and auto-ducked audio):

```powershell
python -m video_editing_ai.cli plan "<project_dir>" --prompt "<user_story_idea>"
```

You or the user can directly edit `<project_dir>/timeline.yaml` in any text editor or in the Local Web Studio.

### Step 3: Programmatic Video Render & NLE Exports
Render `<project_dir>/timeline.yaml` to `<project_dir>/final_story.mp4` along with `.srt`, `.ass`, `.fcpxml` (for DaVinci Resolve / Final Cut Pro), `.edl`, and `.otio`:

```powershell
python -m video_editing_ai.cli render "<project_dir>/timeline.yaml" -o "<project_dir>/final_story.mp4"
```

### Local Web Studio
To launch the interactive browser UI for reviewing annotations, editing `timeline.yaml`, and previewing renders:

```powershell
python -m video_editing_ai.cli serve --port 8000
```
