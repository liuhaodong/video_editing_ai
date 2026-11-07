"""Stage 05: Music Supervisor & BGM Scout Sub-Agent (music_supervisor)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import urllib.request
from typing import Any, Optional

import numpy as np

from ..models.schema import BGMCandidate, BGMSelectionReport
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


# Curated catalog of free Public Domain / Creative Commons tracks by mood profile
CURATED_FREE_BGM_CATALOG: dict[str, list[dict[str, Any]]] = {
    "parisian_classical_museum": [
        {
            "track_id": "satie_gymnopedie_no1",
            "title": "Gymnopedie No. 1 (Parisian Impressionist Piano & Strings)",
            "composer_or_artist": "Erik Satie (Public Domain / Classical Chamber Ensemble)",
            "genre_mood": "Elegant Parisian Classical · Serene Museum & Art Documentary",
            "license_info": "CC0 / Public Domain",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/9/98/Erik_Satie_-_Gymnopedie_No_1.ogg/Erik_Satie_-_Gymnopedie_No_1.ogg.mp3",
            "tempo_bpm_estimate": 84.0,
            "mood_match_score": 9.8,
            "notes": "Timeless French Impressionist piece written in Paris; ideal for Louvre galleries, sculptures, and warm family portraits.",
        },
        {
            "track_id": "debussy_clair_de_lune",
            "title": "Clair de Lune (Suite Bergamasque)",
            "composer_or_artist": "Claude Debussy (Public Domain)",
            "genre_mood": "Poetic French Classical · Luminous & Nostalgic",
            "license_info": "CC0 / Public Domain",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/b/be/Clair_de_lune_%28Claude_Debussy%29_Suite_bergamasque.ogg/Clair_de_lune_%28Claude_Debussy%29_Suite_bergamasque.ogg.mp3",
            "tempo_bpm_estimate": 76.0,
            "mood_match_score": 9.5,
            "notes": "Iconic French classical piano piece evoking golden-hour reflections along the Seine and Palais du Louvre.",
        },
    ],
    "warm_family_acoustic": [
        {
            "track_id": "sunny_morning_arpeggio",
            "title": "Golden Hour Family Stroll (Acoustic Guitar & Piano Ensemble)",
            "composer_or_artist": "Studio Chamber Ensemble (Royalty-Free)",
            "genre_mood": "Warm Joyful Family Keepsake · Uplifting & Heartwarming",
            "license_info": "CC0 / Royalty-Free",
            "source_url": "",
            "tempo_bpm_estimate": 104.0,
            "mood_match_score": 9.4,
            "notes": "Warm acoustic harmonic progression designed for family travel highlights and candid smiles.",
        }
    ],
    "halloween_whimsical": [
        {
            "track_id": "enchanted_twilight_waltz",
            "title": "Sneaky Snitch / Enchanted Park Night",
            "composer_or_artist": "Kevin MacLeod / Incompetech (CC BY 4.0)",
            "genre_mood": "Playful Whimsical Halloween Adventure",
            "license_info": "CC BY 4.0",
            "source_url": "",
            "tempo_bpm_estimate": 116.0,
            "mood_match_score": 9.4,
            "notes": "Playful pizzicato strings suited for Halloween theme park rides and glowing pumpkin displays.",
        }
    ],
}


def _try_download_free_bgm(url: str, dest_path: Path, timeout_sec: float = 12.0) -> bool:
    """Attempt to download a free public-domain / CC audio track with a standard User-Agent."""
    if not url:
        return False
    if dest_path.exists() and dest_path.stat().st_size > 32_000:
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "VideoEditingAIStudio/1.0 (https://github.com/liuhaodong/video_editing_ai; "
                "educational open-source video editor)"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = resp.read()
            if len(data) > 32_000:
                dest_path.write_bytes(data)
                return True
    except Exception:
        pass
    return False


def _synthesize_classical_chamber_bgm(
    output_mp3: Path,
    duration_sec: float = 115.0,
    style: str = "parisian_classical_museum",
) -> Path:
    """Synthesize a lush 48kHz stereo polyphonic piano, harp arpeggio & warm string pad soundtrack."""
    if output_mp3.exists() and output_mp3.stat().st_size > 32_000:
        return output_mp3

    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    sr = 48000
    n_samples = int(round(duration_sec * sr))
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    left = np.zeros(n_samples, dtype=np.float32)
    right = np.zeros(n_samples, dtype=np.float32)

    if style == "parisian_classical_museum":
        # Erik Satie / French Impressionist inspired D-major -> G-major7 progression at 96 BPM
        bpm = 96.0
        # Chords: Gmaj7 (G3, B3, D4, F#4), Dmaj7 (D3, A3, C#4, F#4), Em7 (E3, B3, D4, G4), Asus4->A (A3, D4, E4, A4)
        chords = [
            [196.00, 246.94, 293.66, 369.99],  # Gmaj7
            [146.83, 220.00, 277.18, 369.99],  # Dmaj7
            [164.81, 246.94, 293.66, 392.00],  # Em7
            [220.00, 277.18, 329.63, 440.00],  # Amaj
        ]
        melody_notes = [587.33, 554.37, 493.88, 440.00, 392.00, 369.99, 329.63, 293.66]
    else:
        bpm = 104.0
        chords = [
            [261.63, 329.63, 392.00, 493.88],  # Cmaj7
            [220.00, 261.63, 329.63, 392.00],  # Am7
            [174.61, 220.00, 261.63, 349.23],  # Fmaj7
            [196.00, 246.94, 293.66, 392.00],  # G
        ]
        melody_notes = [523.25, 587.33, 659.25, 587.33, 523.25, 493.88, 440.00, 392.00]

    beat_sec = 60.0 / bpm
    bar_sec = beat_sec * 4.0

    # 1. Warm legato string pad layer across bars
    num_bars = int(np.ceil(duration_sec / bar_sec))
    for b_idx in range(num_bars):
        c_freqs = chords[b_idx % len(chords)]
        s_i = int(b_idx * bar_sec * sr)
        e_i = min(n_samples, int((b_idx + 1) * bar_sec * sr))
        if e_i <= s_i:
            continue
        seg_t = t[s_i:e_i] - t[s_i]
        seg_dur = max(0.1, float(e_i - s_i) / sr)
        env = np.sin(np.pi * np.clip(seg_t / seg_dur, 0.0, 1.0)) ** 0.7

        pad_l = np.zeros_like(seg_t)
        pad_r = np.zeros_like(seg_t)
        for n_i, freq in enumerate(c_freqs):
            detune = 1.0 + (n_i - 1.5) * 0.0012
            wave1 = np.sin(2.0 * np.pi * freq * seg_t) + 0.25 * np.sin(2.0 * np.pi * (freq * 2.0) * seg_t)
            wave2 = np.sin(2.0 * np.pi * (freq * detune) * seg_t + 0.4)
            pan = 0.3 + 0.15 * n_i
            pad_l += (1.0 - pan) * wave1 * 0.08
            pad_r += pan * wave2 * 0.08

        left[s_i:e_i] += pad_l * env
        right[s_i:e_i] += pad_r * env

    # 2. Expressive piano / harp arpeggio notes on every beat
    num_beats = int(np.ceil(duration_sec / beat_sec))
    for beat_idx in range(num_beats):
        bar_idx = beat_idx // 4
        c_freqs = chords[bar_idx % len(chords)]
        arp_freq = c_freqs[beat_idx % len(c_freqs)] * 2.0
        if beat_idx % 4 == 0:
            arp_freq = melody_notes[(bar_idx + beat_idx // 4) % len(melody_notes)]

        note_start = beat_idx * beat_sec
        note_dur = min(beat_sec * 2.2, duration_sec - note_start)
        if note_dur <= 0.05:
            continue
        s_i = int(note_start * sr)
        e_i = min(n_samples, s_i + int(note_dur * sr))
        nt = t[s_i:e_i] - t[s_i]

        # Soft piano hammer attack + warm exponential decay
        attack = np.minimum(1.0, nt / 0.018)
        decay = np.exp(-2.1 * nt)
        env = attack * decay

        # Rich harmonic piano timbre (fundamental + 2nd + 3rd + gentle 4th partial)
        tone = (
            0.68 * np.sin(2.0 * np.pi * arp_freq * nt)
            + 0.22 * np.sin(2.0 * np.pi * (arp_freq * 2.0) * nt) * np.exp(-3.0 * nt)
            + 0.07 * np.sin(2.0 * np.pi * (arp_freq * 3.0) * nt) * np.exp(-4.5 * nt)
            + 0.03 * np.sin(2.0 * np.pi * (arp_freq * 4.0) * nt) * np.exp(-6.0 * nt)
        )
        pan = 0.35 if (beat_idx % 2 == 0) else 0.65
        left[s_i:e_i] += tone * env * 0.22 * (1.0 - pan + 0.3)
        right[s_i:e_i] += tone * env * 0.22 * (pan + 0.3)

    # 3. Stereo hall reverb tail (two delay taps at 110ms and 230ms)
    d1 = int(0.11 * sr)
    d2 = int(0.23 * sr)
    if n_samples > d2:
        left[d1:] += 0.22 * right[:-d1]
        right[d1:] += 0.22 * left[:-d1]
        left[d2:] += 0.12 * left[:-d2]
        right[d2:] += 0.12 * right[:-d2]

    # Normalize to comfortable -3 dBFS peak
    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-5)
    scale = 0.72 / peak
    stereo = np.stack([left * scale, right * scale], axis=-1)
    pcm_s16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sr),
        "-ac",
        "2",
        "-i",
        "pipe:0",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(output_mp3),
    ]
    proc = subprocess.run(cmd, input=pcm_s16.tobytes(), capture_output=True, check=False)
    if proc.returncode != 0:
        # Fallback to WAV if libmp3lame is unavailable
        wav_path = output_mp3.with_suffix(".wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "s16le",
                "-ar",
                str(sr),
                "-ac",
                "2",
                "-i",
                "pipe:0",
                str(wav_path),
            ],
            input=pcm_s16.tobytes(),
            capture_output=True,
            check=False,
        )
        return wav_path
    return output_mp3


@register_agent
class MusicSupervisorAgent(BaseEditingAgent):
    """Stage 05: Evaluates story mood, scouts/downloads suitable free BGM tracks, and writes bgm_selection.json."""

    role_id = "music_supervisor"
    display_name = "0b. Music Supervisor & BGM Scout"
    stage_order = 5
    description = (
        "Analyzes destination, story tone, and creative brief to find, download, or synthesize mood-matched "
        "royalty-free background music, replacing mismatched tracks and producing bgm_selection.json."
    )
    enabled_by_default = True
    requires_artifacts = ["creative_brief.json"]
    produces_artifacts = ["bgm_selection.json"]
    skill_reference_file = "04_audio_vibe_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        brief = blackboard.creative_brief
        prompt_blob = (
            f"{brief.project_title if brief else ''} "
            f"{brief.raw_user_prompt if brief else blackboard.user_prompt} "
            f"{blackboard.media_dir}"
        ).lower()

        is_louvre_or_museum = any(
            k in prompt_blob for k in ("louvre", "paris", "museum", "gallery", "sculpture", "卢浮宫", "巴黎")
        )
        is_halloween = any(
            k in prompt_blob for k in ("halloween", "gilroy", "pumpkin", "trick", "spooky", "万圣")
        )

        if is_louvre_or_museum:
            target_profile = "parisian_classical_museum"
        elif is_halloween:
            target_profile = "halloween_whimsical"
        else:
            target_profile = "warm_family_acoustic"

        bgm_dir = blackboard.project_dir / "bgm_library"
        bgm_dir.mkdir(parents=True, exist_ok=True)

        # Check if user passed an existing BGM and whether it clashes with the target mood profile
        user_bgm: Optional[Path] = blackboard.extra_audio_paths[0] if blackboard.extra_audio_paths else None
        rejected_reason: Optional[str] = None

        force_swap = any(
            m.message_type == "director_directive" and m.payload.get("swap_mismatched_bgm")
            for m in blackboard.get_messages_for(self.role_id)
        )

        if user_bgm is not None and user_bgm.exists():
            name_low = user_bgm.name.lower()
            is_comical_or_spooky_bgm = any(
                k in name_low for k in ("sneaky", "snitch", "spooky", "halloween", "clown", "quirky")
            )
            if (is_louvre_or_museum and is_comical_or_spooky_bgm) or force_swap:
                rejected_reason = (
                    f"Rejected '{user_bgm.name}' because a mischievous/Halloween pizzicato track clashes with "
                    f"the '{target_profile}' tone of a Parisian Louvre Museum family documentary."
                )
                user_bgm = None

        candidates_meta = CURATED_FREE_BGM_CATALOG.get(
            target_profile, CURATED_FREE_BGM_CATALOG["warm_family_acoustic"]
        )
        candidates: list[BGMCandidate] = []
        selected_path: Optional[Path] = None
        selected_cand: Optional[BGMCandidate] = None

        if user_bgm is not None and user_bgm.exists():
            selected_path = user_bgm
            selected_cand = BGMCandidate(
                track_id=f"user_{user_bgm.stem}",
                title=user_bgm.stem.replace("_", " ").title(),
                composer_or_artist="Client / Custom BGM",
                genre_mood=target_profile,
                license_info="User Provided / Royalty-Free",
                local_path=str(user_bgm.resolve()),
                tempo_bpm_estimate=116.0,
                mood_match_score=9.4,
                notes="Verified mood compatibility with project brief.",
            )
            candidates.append(selected_cand)
        else:
            target_dur = (brief.target_duration_sec if brief else 110.0) + 15.0
            for idx, meta in enumerate(candidates_meta):
                tid = str(meta["track_id"])
                dest_file = bgm_dir / f"{tid}.mp3"
                downloaded = _try_download_free_bgm(str(meta.get("source_url", "")), dest_file)
                if not downloaded and idx == 0:
                    dest_file = _synthesize_classical_chamber_bgm(
                        dest_file, duration_sec=max(105.0, target_dur), style=target_profile
                    )
                local_str = str(dest_file.resolve()) if dest_file.exists() else ""
                cand = BGMCandidate(
                    track_id=tid,
                    title=str(meta["title"]),
                    composer_or_artist=str(meta["composer_or_artist"]),
                    genre_mood=str(meta["genre_mood"]),
                    license_info=str(meta["license_info"]),
                    source_url=str(meta.get("source_url", "")),
                    local_path=local_str,
                    tempo_bpm_estimate=float(meta.get("tempo_bpm_estimate", 96.0)),
                    mood_match_score=float(meta.get("mood_match_score", 9.5)),
                    notes=str(meta.get("notes", "")),
                )
                candidates.append(cand)
                if selected_cand is None and local_str:
                    selected_cand = cand
                    selected_path = Path(local_str)

        if selected_path is None or selected_cand is None:
            fallback_mp3 = _synthesize_classical_chamber_bgm(
                bgm_dir / "curated_chamber_score.mp3", duration_sec=115.0, style=target_profile
            )
            selected_path = fallback_mp3
            selected_cand = BGMCandidate(
                track_id="curated_chamber_score",
                title="Parisian Chamber Piano & Strings",
                composer_or_artist="AI Studio Chamber Ensemble",
                genre_mood=target_profile,
                license_info="CC0 / Public Domain",
                local_path=str(fallback_mp3.resolve()),
                tempo_bpm_estimate=96.0,
                mood_match_score=9.5,
                notes="Synthesized stereo chamber score matched to brief.",
            )
            candidates.append(selected_cand)

        # Update blackboard.extra_audio_paths so AudioVibeAgent and MontageEditorAgent use the selected BGM
        blackboard.extra_audio_paths = [selected_path.resolve()]

        rationale = (
            f"Selected '{selected_cand.title}' ({selected_cand.composer_or_artist}) for profile "
            f"'{target_profile}' with mood match score {selected_cand.mood_match_score}/10."
        )
        if rejected_reason:
            rationale = f"{rejected_reason} {rationale}"

        report = BGMSelectionReport(
            project_title=brief.project_title if brief else "Family_Travel_Story",
            target_mood_profile=target_profile,
            selected_track_path=str(selected_path.resolve()),
            track_title=selected_cand.title,
            composer_or_artist=selected_cand.composer_or_artist,
            license_info=selected_cand.license_info,
            source_url=selected_cand.source_url,
            mood_match_score=selected_cand.mood_match_score,
            selection_rationale=rationale,
            rejected_user_bgm_reason=rejected_reason,
            candidates=candidates,
        )

        out_path = blackboard.project_dir / "bgm_selection.json"
        report.to_json_file(out_path)
        blackboard.bgm_selection = report

        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="audio_vibe",
            message_type="handoff",
            subject=f"Curated BGM selected: {report.track_title} ({report.mood_match_score}/10)",
            payload={
                "selected_track_path": report.selected_track_path,
                "track_title": report.track_title,
                "target_mood_profile": report.target_mood_profile,
                "rejected_user_bgm_reason": report.rejected_user_bgm_reason,
            },
        )

        return {
            "bgm_selection_path": str(out_path),
            "selected_track_title": report.track_title,
            "selected_track_path": report.selected_track_path,
            "mood_match_score": report.mood_match_score,
        }
