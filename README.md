# AI Photo & Video Editing Studio (`video_editing_ai`)

An end-to-end AI-powered photo and video analysis, narrative story planner, text-based timeline editor, and programmatic open-source video renderer built with **Google Antigravity SDK**, **faster-whisper**, **FFmpeg**, and **Pillow**.

## Key Capabilities

1. **Step 1 — Multimodal Media Analysis (`annotations.yaml`)**
   - Scans raw videos (`.MOV`, `.MP4`, `.MKV`, including iPhone 10-bit HLG HDR) and photos (`.HEIC`, `.JPG`, `.PNG`).
   - Detects scene cuts and extracts representative JPEG keyframes and labeled contact sheets.
   - Runs `faster-whisper` speech-to-text to annotate spoken dialogue and song lyrics with exact timestamps.
   - Uses local visual saliency/lighting metrics + the **Google Antigravity SDK** to annotate what is happening, emotional tone, and focal points.

2. **Step 2 — AI Story Director & Editable Text Timeline (`timeline.yaml` & `storyboard.md`)**
   - Structures clips into a cohesive **4-Act Narrative Arc** based on your rough story prompt.
   - **Never cuts off spoken sentences**: automatically aligns video in/out points around complete spoken sentences and emotional reactions.
   - Selects the highest-quality non-duplicate photos and assigns **Gallery Frame** (uncropped photo with crisp white border + 3D drop shadow over a Gaussian-blurred ambient background) and smooth **Ken Burns** motion.
   - Produces a human-editable `timeline.yaml` and a detailed `storyboard.md`.

3. **Step 3 — Open-Source Programmatic Video Renderer & NLE Exporters**
   - **Sub-Pixel Ken Burns & Gallery Frame Engine**: Uses floating-point bicubic extent resampling piped to FFmpeg for zero-jitter photo animation.
   - **Programmatic RGBA Typography**: Renders CJK + English Opening Title Cards, Act Banners, Lower Thirds, Dialogue Callouts, and Closing Cards via Pillow.
   - **Multi-Track Audio Mixing & Auto-Ducking**: Automatically ducks background music volume during spoken dialogue segments.
   - **NLE & Subtitle Exports**: Outputs `.mp4`, `.fcpxml` (DaVinci Resolve / Final Cut Pro), `.edl` (CMX 3600), `.otio` (OpenTimelineIO), `.srt`, and `.ass`.

## Quickstart

### CLI Usage

```powershell
# Step 1: Analyze videos & photos
python -m video_editing_ai.cli analyze "C:\Users\hihao\Documents\projects\ai_editor\raw_media" `
  -o ".\gilroy_halloween_project" `
  --bgm "C:\Users\hihao\Documents\projects\ai_editor\bgm_sneaky_snitch.mp3"

# Step 2: Generate story plan & text timeline
python -m video_editing_ai.cli plan ".\gilroy_halloween_project" `
  --prompt "Create an upbeat 4-act family Halloween highlight video at Gilroy Gardens focusing on the kids' excited dialogue, rides, pumpkin pyramid, and Boo Alley night lights."

# Step 3: Render MP4 video and NLE exports
python -m video_editing_ai.cli render ".\gilroy_halloween_project\timeline.yaml" `
  -o ".\gilroy_halloween_project\final_story.mp4"
```

### Local Web Studio

```powershell
python -m video_editing_ai.cli serve --port 8000
```
Then open `http://127.0.0.1:8000` in your browser.
