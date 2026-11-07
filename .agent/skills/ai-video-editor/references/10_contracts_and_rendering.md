# Deterministic JSON Contracts & 4K 10-bit HLG HDR Rendering Reference

## 1. Stage-Order JSON Contracts (`<project_dir>/`)

Every stage in the 11-agent pipeline reads upstream contracts from `<project_dir>` and writes its own deterministic JSON/YAML artifact:

| Stage | Agent Role (`role_id`) | Output Contract File | Key Schema Fields |
|:---|:---|:---|:---|
| **00** | `client_intake` | `creative_brief.json` | `project_title`, `narrative_style`, `target_duration_sec` (`30–600s`), `photo_mode` (`parallax_gallery`), `must_include_topics` |
| **05** | `music_supervisor` | `bgm_selection.json` | `selected_audio_path`, `track_title`, `composer_or_artist`, `mood_match_score`, `rejected_user_bgm_reason` |
| **10** | `culler` | `media_index.json` | `selects` (`clip_id`, `role`, `in_point`, `out_point`, `shot_scale`, `annotation_title`, `annotation_subtitle`) |
| **10** | `audio_vibe` | `beat_grid.json` | `bpm`, `beats`, `downbeats`, `energy_curve`, `ducking_windows` |
| **15** | `photo_curator` | `photo_curation.json` | `selected_count`, `average_taste_score_selected`, `items` (`overall_taste_score`, `recommended_photo_mode`, `diptych_partner_id`) |
| **20** | `story_producer` | `paper_edit.json` | `chapters` (`chapter_id`, `title`, `editing_mode`, `select_ids`, `broll_anchor_map`) |
| **30** | `dialogue_editor` + `montage_editor` | `timeline.yaml` | `settings`, `visual_track` (V1), `broll_track` (V2), `title_track` (V3), `audio_tracks` (A1–A3), `subtitles` |
| **40** | `vfx_designer` | `vfx_manifest.json` | `applied_effects_count`, `effects` (`warm_glow`, `vignette_focus`, `contrast_pop` on `timeline.effects_track`) |
| **50** | `showrunner` | `pacing_audit.json` | `passed`, `overall_score`, `chapter_audits`, `fixes_applied` |
| **60** | `production_director` | `director_review.json` | `quality_grade` (`A+`), `overall_quality_score`, `rubric_scores`, `stage_reviews`, `interventions_count` |
| **Bus** | `PipelineBlackboard` | `blackboard.json` | `executed_agents`, `messages` (`handoff`, `director_directive`, `revision_request`) |

---

## 2. 4K 10-bit HLG HDR FFmpeg Rendering Specification (`render-video`)

- **Resolution & Frame Rate**: `3840x2160` (4K UHD) @ `30 fps`
- **Pixel Format & Transfer**: `yuv420p10le`, `color_primaries=bt2020`, `color_trc=arib-std-b67` (HLG HDR), `colorspace=bt2020nc`
- **Still Photo Rendering Modes**:
  - `parallax_gallery`: 2.5D counter-panning blurred backdrop + uncropped foreground photo with double champagne-gold museum matte + lower-left museum exhibition plaque (`annotation_title` + `annotation_subtitle` + `camera_badge`).
  - `diptych_frame`: Dual vertical portraits (`aspect_ratio < 0.9`) paired side by side with individual museum matte borders over a 4K blurred backdrop.
  - `gallery_frame`: Classic uncropped floating frame over blurred backdrop.
- **3-Layer Audio Mastering**:
  - **Track A1 (Dialogue Spine)**: Full spoken sentences with `-0.35s` J-cut audio leads.
  - **Track A2 (Continuous Nat-Sound Bed)**: Ambient sound from B-roll at `-20 dB` with `0.33s` crossfades (`acrossfade`).
  - **Track A3 (Auto-Ducked BGM)**: Curated soundtrack smoothly ducked by `-14 dB` (`volume * 0.20`) with `0.4s` attack/release ramps during speech.
- **NLE Interchange Deliverables**:
  - `timeline.fcpxml` (Final Cut Pro / DaVinci Resolve)
  - `timeline.edl` (CMX 3600 EDL)
  - `timeline.otio` (OpenTimelineIO)
  - `final_story.srt` & `final_story.ass`
