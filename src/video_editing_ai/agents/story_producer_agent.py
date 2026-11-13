"""Stage 20: Story Producer Sub-Agent (story_producer) — Spatio-Temporal/Semantic Clustering & Five-Shot Film Grammar."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Optional

import numpy as np

from ..analyzer.semantic_intelligence import SemanticEmbeddingEngine
from ..models.schema import ChapterPlan, ColdOpenHookSlot, CulledSubClip, PaperEdit
from .base import BaseEditingAgent, PipelineBlackboard, register_agent


_WORD_TO_NUM = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}

_MUNDANE_HOOK_RE = re.compile(
    r"\b(where\s+is\s+the\s+bathroom|restroom|what\s+time\s+is\s+it|did\s+you\s+lock|battery\s+is\s+low|parking\s+lot)\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "over", "under", "very", "just",
    "really", "here", "there", "look", "live", "scene", "clip", "video", "photo", "still", "moment",
    "story", "family", "travel", "selected", "frame", "curated", "visual", "study", "featured",
    "natural", "atmosphere", "candid", "spoken", "dialogue", "audio", "medium", "wide", "closeup",
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two decimal-degree GPS points."""
    r_earth = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * r_earth * math.asin(min(1.0, math.sqrt(a)))


def _infer_clip_time_of_day(sc: CulledSubClip) -> Literal["morning", "afternoon", "twilight", "night"]:
    from .dit_ingest_agent import _parse_iso_dt

    blob = f"{sc.summary} {sc.annotation_title or ''} {sc.annotation_subtitle or ''}".lower()
    if any(k in blob for k in ("night", "firework", "illuminated", "midnight")):
        return "night"
    if any(k in blob for k in ("twilight", "sunset", "golden", "dusk", "evening")):
        return "twilight"

    dt, _ = _parse_iso_dt(sc.normalized_timestamp)
    if dt is not None and dt.year > 1971:
        hr = dt.hour
        if 5 <= hr < 12:
            return "morning"
        if 12 <= hr < 17:
            return "afternoon"
        if 17 <= hr < 20:
            return "twilight"
        return "night"
    return "afternoon"


def _select_semantic_text(sc: CulledSubClip) -> str:
    parts = [
        sc.spoken_text or "",
        sc.translation or "",
        sc.annotation_title or "",
        sc.annotation_subtitle or "",
        sc.subject_category or "",
        sc.summary or "",
    ]
    return " ".join(p.strip() for p in parts if p and p.strip())


def _determine_dynamic_chapter_count(
    active_selects: list[CulledSubClip],
    prompt_text: str,
    target_duration_sec: float,
) -> int:
    """Determine dynamic chapter count (2-8) from prompt overrides, GPS/time clusters, and select count."""
    if len(active_selects) <= 1:
        return 1

    m_num = re.search(r"\b([2-8])\s*[- ]?(?:act|chapter)s?\b", prompt_text, re.IGNORECASE)
    if m_num:
        return min(len(active_selects), max(2, min(8, int(m_num.group(1)))))

    m_word = re.search(
        r"\b(two|three|four|five|six|seven|eight)\s*[- ]?(?:act|chapter)s?\b",
        prompt_text,
        re.IGNORECASE,
    )
    if m_word:
        val = _WORD_TO_NUM[m_word.group(1).lower()]
        return min(len(active_selects), max(2, min(8, val)))

    from .dit_ingest_agent import _parse_iso_dt

    ordered = sorted(active_selects, key=lambda x: (x.normalized_timestamp, x.filename))
    parsed_ts: list[Optional[float]] = []
    for s in ordered:
        dt, _ = _parse_iso_dt(s.normalized_timestamp)
        parsed_ts.append(dt.timestamp() if (dt is not None and dt.year > 1971) else None)

    natural_clusters = 1
    for i in range(1, len(ordered)):
        t_prev, t_curr = parsed_ts[i - 1], parsed_ts[i]
        time_jump = (t_prev is not None and t_curr is not None and (t_curr - t_prev) >= 2100.0)
        gps_jump = False
        if (
            ordered[i - 1].gps_lat is not None
            and ordered[i - 1].gps_lon is not None
            and ordered[i].gps_lat is not None
            and ordered[i].gps_lon is not None
        ):
            gps_jump = _haversine_km(
                float(ordered[i - 1].gps_lat),
                float(ordered[i - 1].gps_lon),
                float(ordered[i].gps_lat),
                float(ordered[i].gps_lon),
            ) >= 0.30
        if time_jump or gps_jump:
            natural_clusters += 1

    valid_ts = [t for t in parsed_ts if t is not None]
    span_hours = (max(valid_ts) - min(valid_ts)) / 3600.0 if len(valid_ts) >= 2 else 0.0

    # If clear multi-location / multi-session breaks exist (2..8), honor natural scene count when well-populated
    if 2 <= natural_clusters <= 8 and len(active_selects) >= natural_clusters * 2:
        return natural_clusters

    by_count = max(2, min(8, int(round(len(active_selects) / 4.5)) if len(active_selects) >= 7 else 2))
    by_dur = max(2, min(8, int(round(target_duration_sec / 48.0))))
    by_span = min(8, max(2, natural_clusters + (1 if span_hours >= 4.0 else 0)))

    dynamic_n = max(by_count, min(8, max(2, (by_dur + by_span) // 2)))
    if len(active_selects) <= 4:
        dynamic_n = min(len(active_selects), max(2, min(3, dynamic_n)))
    return max(2, min(8, min(len(active_selects), dynamic_n)))


def _infer_story_arc_mode(
    prompt_text: str,
    brief_mode: Optional[str],
) -> Literal["chronological_cinema", "hero_journey", "thematic_essay", "high_energy_reel"]:
    p_low = (prompt_text or "").lower()
    if "thematic" in p_low or "visual essay" in p_low or "by theme" in p_low:
        return "thematic_essay"
    if "hero" in p_low or "odyssey" in p_low:
        return "hero_journey"
    if "high energy" in p_low or "hype reel" in p_low or "fast reel" in p_low:
        return "high_energy_reel"
    if brief_mode in ("chronological_cinema", "hero_journey", "thematic_essay", "high_energy_reel"):
        return brief_mode  # type: ignore[return-value]
    return "chronological_cinema"


def _partition_selects_spatiotemporal_dp(
    chron_selects: list[CulledSubClip],
    num_chapters: int,
    sem_vecs: np.ndarray,
) -> list[list[CulledSubClip]]:
    """
    Partition chronologically sorted selects into `num_chapters` contiguous chapters using
    Dynamic Programming over time gaps, Haversine GPS distance jumps (>250m), semantic shifts,
    and lighting transitions so real-world scenes are never torn across arbitrary index boundaries.
    """
    n = len(chron_selects)
    if num_chapters <= 1 or n <= 1:
        return [list(chron_selects)]
    if n <= num_chapters:
        out = [[s] for s in chron_selects]
        while len(out) < num_chapters:
            out.append([])
        return [b for b in out if b]

    from .dit_ingest_agent import _parse_iso_dt

    ts_list: list[Optional[float]] = []
    tods: list[str] = []
    for s in chron_selects:
        dt, _ = _parse_iso_dt(s.normalized_timestamp)
        ts_list.append(dt.timestamp() if (dt is not None and dt.year > 1971) else None)
        tods.append(_infer_clip_time_of_day(s))

    # Break affinity B[i] for cutting BEFORE index i (1 <= i < n)
    break_scores = np.full(n, 0.10, dtype=np.float64)
    for i in range(1, n):
        score = 0.10
        t0, t1 = ts_list[i - 1], ts_list[i]
        if t0 is not None and t1 is not None:
            dt_sec = max(0.0, t1 - t0)
            score += min(4.8, math.log1p(dt_sec / 300.0) * 1.45)

        s_prev, s_curr = chron_selects[i - 1], chron_selects[i]
        if (
            s_prev.gps_lat is not None
            and s_prev.gps_lon is not None
            and s_curr.gps_lat is not None
            and s_curr.gps_lon is not None
        ):
            dist_km = _haversine_km(
                float(s_prev.gps_lat),
                float(s_prev.gps_lon),
                float(s_curr.gps_lat),
                float(s_curr.gps_lon),
            )
            if dist_km >= 0.25:
                score += min(4.5, 1.65 + math.log1p(dist_km * 4.0))

        if sem_vecs.shape[0] == n:
            cos_sim = float(np.dot(sem_vecs[i - 1], sem_vecs[i]))
            score += 0.85 * max(0.0, 1.0 - cos_sim)

        if tods[i - 1] != tods[i]:
            score += 0.65

        break_scores[i] = score

    min_chap_len = 2 if n >= num_chapters * 2 else 1
    ideal_len = n / float(num_chapters)

    def _balance_pen(seg_len: int) -> float:
        rel = (seg_len - ideal_len) / max(1.0, ideal_len)
        return 0.12 * rel * rel

    neg_inf = -1e12
    dp = np.full((num_chapters + 1, n + 1), neg_inf, dtype=np.float64)
    back = np.zeros((num_chapters + 1, n + 1), dtype=np.int32)

    for end_i in range(min_chap_len, n - (num_chapters - 1) * min_chap_len + 1):
        dp[1, end_i] = -_balance_pen(end_i)
        back[1, end_i] = 0

    for k in range(2, num_chapters + 1):
        min_end = k * min_chap_len
        max_end = n - (num_chapters - k) * min_chap_len
        for end_i in range(min_end, max_end + 1):
            best_v = neg_inf
            best_p = (k - 1) * min_chap_len
            for prev_i in range((k - 1) * min_chap_len, end_i - min_chap_len + 1):
                cand = dp[k - 1, prev_i] + break_scores[prev_i] - _balance_pen(end_i - prev_i)
                if cand > best_v:
                    best_v = cand
                    best_p = prev_i
            dp[k, end_i] = best_v
            back[k, end_i] = best_p

    # Reconstruct optimal boundaries
    bounds: list[tuple[int, int]] = []
    curr_end = n
    for k in range(num_chapters, 0, -1):
        prev_i = int(back[k, curr_end])
        bounds.append((prev_i, curr_end))
        curr_end = prev_i
    bounds.reverse()

    return [chron_selects[s_i:e_i] for (s_i, e_i) in bounds if e_i > s_i]


def _partition_selects_thematic(
    selects: list[CulledSubClip],
    num_chapters: int,
    sem_vecs: np.ndarray,
) -> list[list[CulledSubClip]]:
    """Group selects across the library into coherent thematic/motif chapters via farthest-first cosine medoids."""
    n = len(selects)
    if num_chapters <= 1 or n <= num_chapters:
        return [[s] for s in selects]

    # Deterministic farthest-point seeding in semantic embedding space
    medoids: list[int] = [0]
    for _ in range(1, num_chapters):
        best_idx = 0
        best_min_dist = -1.0
        for idx in range(n):
            if idx in medoids:
                continue
            min_dist = min(1.0 - float(np.dot(sem_vecs[idx], sem_vecs[m])) for m in medoids)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        medoids.append(best_idx)

    buckets: list[list[CulledSubClip]] = [[] for _ in range(num_chapters)]
    for idx, sc in enumerate(selects):
        sims = [float(np.dot(sem_vecs[idx], sem_vecs[m])) for m in medoids]
        best_c = int(np.argmax(sims))
        buckets[best_c].append(sc)

    non_empty = [b for b in buckets if b]
    # Order thematic movements from wide/exterior openings toward intimate/evening finales
    def _theme_order_key(cluster: list[CulledSubClip]) -> float:
        wide_frac = sum(1 for c in cluster if c.shot_scale == "wide") / max(1, len(cluster))
        twilight_frac = sum(1 for c in cluster if _infer_clip_time_of_day(c) in ("twilight", "night")) / max(1, len(cluster))
        return twilight_frac * 2.0 - wide_frac

    non_empty.sort(key=_theme_order_key)
    return non_empty


def _assign_story_stage_and_energy(
    c_idx: int,
    num_chapters: int,
    story_arc_mode: str,
    beat_grid: Any = None,
) -> tuple[Literal["hook_arrival", "rising_discovery", "midpoint_wonder", "intimate_connection", "climax_highlight", "warm_resolution"], float]:
    if num_chapters == 1:
        return "climax_highlight", 0.78
    frac = c_idx / max(1, num_chapters - 1)
    if c_idx == 0:
        stage: Literal["hook_arrival", "rising_discovery", "midpoint_wonder", "intimate_connection", "climax_highlight", "warm_resolution"] = "hook_arrival"
        base_e = 0.62 if story_arc_mode != "high_energy_reel" else 0.82
    elif c_idx == num_chapters - 1:
        stage = "warm_resolution"
        base_e = 0.55 if story_arc_mode != "high_energy_reel" else 0.78
    elif abs(frac - 0.75) <= 0.18:
        stage = "climax_highlight"
        base_e = 0.88
    elif abs(frac - 0.50) <= 0.16:
        stage = "midpoint_wonder"
        base_e = 0.76
    elif frac > 0.50:
        stage = "intimate_connection"
        base_e = 0.64
    else:
        stage = "rising_discovery"
        base_e = 0.70

    if beat_grid is not None and getattr(beat_grid, "energy_sections", None):
        secs = beat_grid.energy_sections
        sec_idx = min(len(secs) - 1, int(frac * len(secs)))
        sec_lbl = secs[sec_idx].label
        if sec_lbl == "chorus_drop":
            base_e = min(0.96, base_e + 0.12)
        elif sec_lbl in ("intro", "outro"):
            base_e = max(0.45, base_e - 0.06)

    return stage, round(base_e, 2)


def _synthesize_grounded_chapter_header(
    c_idx: int,
    num_chapters: int,
    chapter_selects: list[CulledSubClip],
    story_stage: str,
    fallback_header: tuple[str, str, str, str, str, str],
) -> tuple[str, str, str, str, str, str]:
    """Derive grounded chapter title, subtitle, location tag, time-of-day, and narrative goal from chapter content."""
    cid, fb_title, fb_sub, fb_loc, fb_tod, fb_goal = fallback_header
    if not chapter_selects:
        return fallback_header

    # Dominant time-of-day from clips
    tod_counts: dict[str, int] = {}
    for s in chapter_selects:
        t = _infer_clip_time_of_day(s)
        tod_counts[t] = tod_counts.get(t, 0) + 1
    dom_tod = max(tod_counts.items(), key=lambda kv: kv[1])[0] if tod_counts else fb_tod

    # Extract salient non-generic capitalized / meaningful words from annotations & speech
    word_freq: dict[str, int] = {}
    for s in chapter_selects:
        src_txt = f"{s.spoken_text or ''} {s.annotation_title or ''}"
        for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", src_txt):
            wl = w.lower()
            if wl in _STOPWORDS or wl.startswith(("scene", "clip", "vid_", "img_", "dsc")):
                continue
            word_freq[w.capitalize()] = word_freq.get(w.capitalize(), 0) + (2 if s.spoken_text else 1)

    top_terms = [w for w, _ in sorted(word_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:2]]
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"][min(7, c_idx)]
    prefix = f"Act {roman}" if num_chapters <= 4 else f"Chapter {c_idx + 1}"

    stage_labels = {
        "hook_arrival": "Opening & First Impressions",
        "rising_discovery": "Exploration & Motion",
        "midpoint_wonder": "Signature Highlights",
        "intimate_connection": "Candid Encounters",
        "climax_highlight": "Peak Perspectives",
        "warm_resolution": "Closing Reflections",
    }
    stage_phrase = stage_labels.get(story_stage, fb_sub)

    if top_terms and len(" & ".join(top_terms)) <= 28:
        focus_label = " & ".join(top_terms)
        grounded_title = f"{prefix} - {focus_label}"
        loc_tag = top_terms[0]
    else:
        grounded_title = fb_title
        loc_tag = fb_loc

    return (cid, grounded_title, stage_phrase, loc_tag, dom_tod, fb_goal)


def _sequence_chapter_with_film_grammar(
    primary_selects: list[CulledSubClip],
    story_arc_mode: str,
) -> tuple[list[CulledSubClip], list[str]]:
    """
    Order a chapter's primary selects using Five-Shot Cinematic Grammar:
    1. Establishing Wide Shot (anchors geography & scale)
    2. Spoken A-Roll Spine (human story anchor, preserving dialogue chronology)
    3. Scale-Alternating Action & Closeup Detail Weave (prevents jump-cuts & same-scale monotony)
    4. Resolving Chapter Button (warm smile/portrait or lingering horizon cadence)
    """
    if len(primary_selects) <= 2:
        roles = [
            "establishing_wide" if (i == 0 and s.shot_scale == "wide") else ("aroll_anchor" if s.roll_type == "a_roll" else "montage_beat")
            for i, s in enumerate(primary_selects)
        ]
        return list(primary_selects), roles

    chron = sorted(primary_selects, key=lambda s: (s.normalized_timestamp, s.filename))
    chron_rank = {s.subclip_id: idx for idx, s in enumerate(chron)}
    remaining = list(chron)

    # 1. Pick Establishing Wide Opener FIRST (so a chapter's wide establishing shot always anchors slot 0)
    establishing_clip: Optional[CulledSubClip] = None
    wide_cands = [s for s in remaining if s.shot_scale == "wide" and s.roll_type != "a_roll"]
    if not wide_cands:
        wide_cands = [s for s in remaining if s.shot_scale == "wide"]
    if wide_cands:
        establishing_clip = max(
            wide_cands,
            key=lambda s: (
                0.55 * (s.aesthetic_score / 10.0)
                + 0.25 * s.stability_score
                - 0.20 * (chron_rank[s.subclip_id] / max(1, len(chron) - 1))
            ),
        )
        remaining.remove(establishing_clip)

    # 2. Reserve Chapter Resolving Button from remaining selects (when >= 3 clips remain)
    button_clip: Optional[CulledSubClip] = None
    if len(remaining) >= 3:
        button_pool = [s for s in remaining if s.roll_type != "a_roll"]
        if button_pool:
            button_clip = max(
                button_pool,
                key=lambda s: (
                    0.40 * (s.emotion_score / 10.0)
                    + 0.30 * s.expression_score
                    + 0.20 * (chron_rank[s.subclip_id] / max(1, len(chron) - 1))
                    + (0.12 if _infer_clip_time_of_day(s) in ("twilight", "night") else 0.0)
                ),
            )
            remaining.remove(button_clip)

    ordered: list[CulledSubClip] = []
    grammar_roles: list[str] = []
    if establishing_clip is not None:
        ordered.append(establishing_clip)
        grammar_roles.append("establishing_wide")

    # 3. Separate A-Roll spoken spine (chronological) and B-Roll / Still visual weave
    arolls = sorted([s for s in remaining if s.roll_type == "a_roll"], key=lambda s: chron_rank[s.subclip_id])
    visuals = [s for s in remaining if s.roll_type != "a_roll"]

    if story_arc_mode == "high_energy_reel":
        visuals.sort(key=lambda s: (s.prosody_excitement_score + s.emotion_score / 10.0), reverse=True)

    # Interleave A-Roll anchors with scale-alternating B-Roll / Still weave
    prev_scale = ordered[-1].shot_scale if ordered else ""
    prev_cat = ordered[-1].subject_category if ordered else ""

    def _pop_next_visual(target_chron_pos: float) -> CulledSubClip:
        nonlocal prev_scale, prev_cat
        best_v = max(
            visuals,
            key=lambda v: (
                (0.42 if v.shot_scale != prev_scale else 0.0)
                + (0.18 if (v.subject_category or "") != (prev_cat or "") else 0.0)
                + 0.24 * (v.aesthetic_score / 10.0)
                - 0.16 * abs((chron_rank[v.subclip_id] / max(1, len(chron) - 1)) - target_chron_pos)
            ),
        )
        visuals.remove(best_v)
        prev_scale = best_v.shot_scale
        prev_cat = best_v.subject_category or ""
        return best_v

    if arolls:
        # Place primary A-Roll anchor early (right after Establishing Wide) and weave remaining visuals around any extra A-Roll
        per_aroll_vis = max(0, len(visuals) // max(1, len(arolls)))
        for a_i, ar in enumerate(arolls):
            ordered.append(ar)
            grammar_roles.append("aroll_anchor")
            prev_scale = ar.shot_scale
            t_pos = chron_rank[ar.subclip_id] / max(1, len(chron) - 1)
            take_n = len(visuals) if a_i == len(arolls) - 1 else per_aroll_vis
            for _ in range(take_n):
                if not visuals:
                    break
                nxt = _pop_next_visual(t_pos)
                ordered.append(nxt)
                grammar_roles.append("closeup_detail" if nxt.shot_scale == "closeup" else "medium_weave")
    else:
        step_idx = 0
        total_v = max(1, len(visuals))
        while visuals:
            t_pos = step_idx / total_v
            nxt = _pop_next_visual(t_pos)
            ordered.append(nxt)
            grammar_roles.append("closeup_detail" if nxt.shot_scale == "closeup" else "montage_weave")
            step_idx += 1

    if button_clip is not None:
        ordered.append(button_clip)
        grammar_roles.append("resolving_button")

    return ordered, grammar_roles


@register_agent
class StoryProducerAgent(BaseEditingAgent):
    """Agent 2 (Stage 20): Dramaturgical Chapter Architect & Five-Shot Storyline Producer."""

    role_id = "story_producer"
    display_name = "2. Story Producer (Dynamic Chapter Architect)"
    stage_order = 20
    description = (
        "Clusters culled selects into 2-8 chapters via spatio-temporal, GPS, and neural semantic boundaries, "
        "supports chronological_cinema, hero_journey, thematic_essay, and high_energy_reel story modes, "
        "pairs Track V2 cutaways via semantic similarity, sequences chapters with Five-Shot film grammar, "
        "and curates a non-duplicating 00:00-00:06 cold_open_hook in paper_edit.json."
    )
    enabled_by_default = True
    requires_artifacts = ["creative_brief.json", "media_index.json"]
    produces_artifacts = ["paper_edit.json"]
    skill_reference_file = "02_story_producer_agent.md"

    def run(self, blackboard: PipelineBlackboard) -> dict[str, Any]:
        brief = blackboard.creative_brief
        media_index = blackboard.media_index
        if media_index is None:
            raise RuntimeError("StoryProducerAgent requires blackboard.media_index to be populated by CullerAgent.")

        active_selects = [s for s in media_index.subclips if not s.is_discarded]
        videos = [s for s in active_selects if s.media_type == "video"]
        photos = [s for s in active_selects if s.media_type == "photo"]

        forced_select_ids: set[str] = set()
        min_aroll_cutaway_dur = 8.5
        for msg in blackboard.get_messages_for(self.role_id):
            if msg.message_type in ("revision_request", "director_directive"):
                for sid in msg.payload.get("include_select_ids", []):
                    forced_select_ids.add(str(sid))
                if "min_aroll_cutaway_duration" in msg.payload:
                    min_aroll_cutaway_dur = float(msg.payload["min_aroll_cutaway_duration"])

        for rev in blackboard.pending_revision_directives(self.role_id):
            for sid in rev.constraints.get("include_select_ids", []):
                forced_select_ids.add(str(sid))
        blackboard.mark_directives_resolved(self.role_id)

        for s in media_index.subclips:
            if s.subclip_id in forced_select_ids and s not in active_selects:
                s.is_discarded = False
                active_selects.append(s)
                if s.media_type == "video":
                    videos.append(s)
                else:
                    photos.append(s)

        prompt_text = (brief.raw_user_prompt if brief else blackboard.user_prompt) or ""
        target_dur = brief.target_duration_sec if brief else 110.0
        story_arc_mode = _infer_story_arc_mode(prompt_text, getattr(brief, "story_arc_mode", None))

        # 1. Select Curiosity-Driven Cold-Open Teaser Hook FIRST so we can prevent verbatim replay in chapters
        want_cold_open = brief.cold_open_hook if brief is not None else True
        if str(blackboard.interview_answers.get("cold_open_hook", "")).lower() in ("false", "0", "no"):
            want_cold_open = False

        cold_open_sid: Optional[str] = None
        cold_open_in: Optional[float] = None
        cold_open_out: Optional[float] = None
        cold_open_caption = ""
        cold_open_rationale = ""

        if want_cold_open and videos:
            vid_idx_map = {v.subclip_id: idx for idx, v in enumerate(videos)}

            def _hook_score(v: CulledSubClip) -> float:
                mundane_pen = 2.5 if (v.spoken_text and _MUNDANE_HOOK_RE.search(v.spoken_text)) else 0.0
                quote_bonus = 0.45 if (v.spoken_text and len(v.spoken_text.strip()) >= 8 and not mundane_pen) else 0.0
                motion_bonus = 0.35 if v.camera_movement != "static" else 0.0
                later_teaser_bonus = 0.15 if vid_idx_map.get(v.subclip_id, 0) >= 1 else 0.0
                return (
                    0.36 * v.emotion_score
                    + 0.24 * v.aesthetic_score
                    + 2.2 * v.prosody_excitement_score
                    + 1.6 * v.stability_score
                    + quote_bonus
                    + motion_bonus
                    + later_teaser_bonus
                    - mundane_pen
                )

            best_hook = max(videos, key=_hook_score)
            cold_open_sid = best_hook.subclip_id
            hook_dur = min(3.4, max(2.2, round(best_hook.duration * 0.45, 2)))
            peak_center = (
                best_hook.action_peak_tc
                if best_hook.action_peak_tc > 0
                else (best_hook.in_tc + best_hook.out_tc) * 0.5
            )
            cold_open_in = round(
                max(best_hook.in_tc, min(best_hook.out_tc - hook_dur, peak_center - hook_dur * 0.5)), 2
            )
            cold_open_out = round(min(best_hook.out_tc, cold_open_in + hook_dur), 2)
            if best_hook.spoken_text:
                first_clause = best_hook.spoken_text.split(".")[0].strip()
                cold_open_caption = f'"{first_clause[:56]}"'
            else:
                cold_open_caption = best_hook.annotation_title or best_hook.summary[:52] or "COMING UP IN THIS STORY..."
            cold_open_rationale = (
                f"Selected {best_hook.subclip_id} ({best_hook.filename}, emotion {best_hook.emotion_score}/10, "
                f"prosody {best_hook.prosody_excitement_score:.2f}, stability {best_hook.stability_score:.2f}) "
                "as a curiosity-driven Cold-Open Teaser Hook before the Main Title Card."
            )

        # De-duplicate short pure B-roll Cold-Open hooks when plenty of video selects exist,
        # while always preserving spoken A-Roll clips in their narrative chapter!
        chapter_pool = list(active_selects)
        if (
            cold_open_sid is not None
            and cold_open_sid not in forced_select_ids
            and len(videos) >= 4
        ):
            hook_obj = next((v for v in videos if v.subclip_id == cold_open_sid), None)
            if hook_obj is not None and hook_obj.roll_type == "b_roll" and hook_obj.duration <= 4.6:
                chapter_pool = [s for s in chapter_pool if s.subclip_id != cold_open_sid]

        chapter_templates = [
            ("chap_01", "Act I - Opening & Arrival", "Opening & First Impressions", "Opening Chapter", "morning", "Establishing the setting, atmosphere, and opening anticipation."),
            ("chap_02", "Act II - Exploration & Discovery", "Architecture & Motion", "Exploration", "morning", "Dynamic scenes, architectural scale, and candid exploration."),
            ("chap_03", "Act III - Signature Highlights & Details", "Curated Focal Moments", "Signature Highlights", "afternoon", "Curated focal points, artistic details, and emotional keepsakes."),
            ("chap_04", "Act IV - Midday Vignettes & Encounters", "Candid Perspectives", "Central Gallery", "afternoon", "Spontaneous reactions, textures, and shared discoveries."),
            ("chap_05", "Act V - Afternoon Rhythm & Movement", "Kinetic Passage", "Promenade", "afternoon", "Rhythmic movement across scenic corridors and open vistas."),
            ("chap_06", "Act VI - Golden Hour Perspectives", "Warm Light & Portraits", "Terrace", "twilight", "Softening golden-hour light and expressive portraits."),
            ("chap_07", "Act VII - Twilight Atmosphere", "Evening Illumination", "Evening Walk", "twilight", "Luminous twilight ambiance and quiet reflection."),
            ("chap_08", "Act VIII - Closing Finale & Reflections", "Lasting Keepsakes", "Closing Finale", "night", "Warm closing reflections and final cadence."),
        ]

        num_chapters = _determine_dynamic_chapter_count(chapter_pool, prompt_text, target_dur)

        # 2. Compute Semantic Embeddings for all selects once (used for both chapter clustering & V2 cutaway matching)
        emb_engine = SemanticEmbeddingEngine.instance()
        chron_selects = sorted(chapter_pool, key=lambda s: (s.normalized_timestamp, s.filename))
        sem_texts = [_select_semantic_text(s) for s in chron_selects]
        sem_vecs = emb_engine.embed_texts(sem_texts)
        vec_by_id: dict[str, np.ndarray] = {
            chron_selects[i].subclip_id: sem_vecs[i] for i in range(len(chron_selects))
        }

        # 3. Partition selects into chapters via Spatio-Temporal/GPS/Semantic DP or Thematic Clustering
        if story_arc_mode == "thematic_essay":
            raw_clusters = _partition_selects_thematic(chron_selects, num_chapters, sem_vecs)
        else:
            raw_clusters = _partition_selects_spatiotemporal_dp(chron_selects, num_chapters, sem_vecs)

        num_chapters = len(raw_clusters)
        if num_chapters == len(chapter_templates):
            chapter_headers = list(chapter_templates)
        elif num_chapters <= 1:
            chapter_headers = [chapter_templates[0]]
        else:
            idxs = [round(i * (len(chapter_templates) - 1) / (num_chapters - 1)) for i in range(num_chapters)]
            chapter_headers = []
            for k, t_i in enumerate(idxs):
                _, t_raw, sub_s, loc_s, tod_s, goal_s = chapter_templates[t_i]
                t_clean = t_raw.split(" - ", 1)[-1] if num_chapters > 4 else t_raw
                if not t_clean.startswith("Act "):
                    t_clean = f"Chapter {k + 1} - {t_clean}"
                chapter_headers.append((f"chap_{k + 1:02d}", t_clean, sub_s, loc_s, tod_s, goal_s))

        chapters: list[ChapterPlan] = []
        used_as_v2_cutaway: set[str] = set()

        for c_idx, cluster_items in enumerate(raw_clusters):
            story_stage, target_energy = _assign_story_stage_and_energy(
                c_idx, num_chapters, story_arc_mode, beat_grid=blackboard.beat_grid
            )
            cid, title_en, sub_en, loc_tag, tod, summary = _synthesize_grounded_chapter_header(
                c_idx,
                num_chapters,
                cluster_items,
                story_stage,
                chapter_headers[c_idx],
            )
            c_vids = [s for s in cluster_items if s.media_type == "video"]
            c_photos = [s for s in cluster_items if s.media_type == "photo"]

            # 4. Build Neural Semantic Track V2 B-Roll Cutaway Anchor Map over Spoken A-Roll
            # Protect the chapter's sole wide establishing photo (if no wide video exists) so Five-Shot Grammar keeps slot 0!
            has_wide_vid = any(v.shot_scale == "wide" and v.roll_type != "a_roll" for v in c_vids)
            wide_photos = [p for p in c_photos if p.shot_scale == "wide"]
            protected_wide_id: Optional[str] = (
                wide_photos[0].subclip_id if (not has_wide_vid and len(wide_photos) == 1 and len(c_photos) >= 2) else None
            )

            broll_anchor_map: dict[str, list[str]] = {}
            prioritized_vids = sorted(
                c_vids,
                key=lambda x: (1 if x.roll_type == "a_roll" else 0, x.duration, x.emotion_score),
                reverse=True,
            )
            for v in prioritized_vids:
                avail_cutaways = [
                    p
                    for p in c_photos
                    if p.subclip_id not in used_as_v2_cutaway
                    and p.subclip_id != protected_wide_id
                    and not p.secondary_source_path
                    and len(c_photos) - len(used_as_v2_cutaway) >= 2
                ]
                # If chapter has at least 1 photo and A-roll is long OR strongly semantically matched, allow cutaway
                if not avail_cutaways and len(c_photos) == 1 and v.roll_type == "a_roll":
                    avail_cutaways = [
                        p for p in c_photos if p.subclip_id not in used_as_v2_cutaway and not p.secondary_source_path
                    ]

                if not avail_cutaways:
                    continue

                v_vec = vec_by_id.get(v.subclip_id)
                v_speech_vec = (
                    emb_engine.embed_texts([v.spoken_text])[0]
                    if (v.spoken_text and v.spoken_text.strip())
                    else v_vec
                )

                best_cand: Optional[CulledSubClip] = None
                best_sem_sim = -1.0
                best_raw_sim = 0.0
                for cand_p in avail_cutaways:
                    p_vec = vec_by_id.get(cand_p.subclip_id)
                    if v_speech_vec is not None and p_vec is not None:
                        sim = float(np.dot(v_speech_vec, p_vec))
                    else:
                        sim = 0.0
                    # Lexical tie-breaker
                    v_words = {
                        w.lower()
                        for w in (v.spoken_text + " " + (v.annotation_title or "") + " " + v.summary).split()
                        if len(w) >= 4
                    }
                    p_blob = ((cand_p.annotation_title or "") + " " + cand_p.summary).lower()
                    lex_hits = sum(1 for w in v_words if w in p_blob)
                    comb_sim = sim + 0.08 * min(3, lex_hits) + 0.015 * cand_p.aesthetic_score
                    if comb_sim > best_sem_sim:
                        best_sem_sim = comb_sim
                        best_raw_sim = sim
                        best_cand = cand_p

                eligible_by_len = (
                    (v.roll_type == "a_roll" and v.duration >= min_aroll_cutaway_dur)
                    or (v.duration >= 6.0 and len(c_photos) >= 3 and not broll_anchor_map)
                )
                eligible_by_semantics = (
                    v.roll_type == "a_roll"
                    and v.duration >= 3.5
                    and bool(v.spoken_text.strip())
                    and best_raw_sim >= 0.22
                    and best_sem_sim >= 0.34
                )
                if best_cand is not None and (eligible_by_len or eligible_by_semantics) and best_sem_sim >= 0.28:
                    broll_anchor_map[v.subclip_id] = [best_cand.subclip_id]
                    used_as_v2_cutaway.add(best_cand.subclip_id)

            primary_photos = [p for p in c_photos if p.subclip_id not in used_as_v2_cutaway]
            ordered_selects, grammar_seq = _sequence_chapter_with_film_grammar(
                [*c_vids, *primary_photos],
                story_arc_mode=story_arc_mode,
            )

            aroll_vids = [s for s in c_vids if s.roll_type == "a_roll"]
            short_candid_arolls = [s for s in aroll_vids if 1.5 <= s.duration <= 4.0]
            long_arolls = [s for s in aroll_vids if s.duration > 4.0]
            montage_elements = [s for s in ordered_selects if s.roll_type in ("b_roll", "still")]

            candid_ids: list[str] = []
            if short_candid_arolls and not long_arolls and len(montage_elements) >= 1:
                editing_mode = "hybrid"
                candid_ids = [s.subclip_id for s in short_candid_arolls]
            elif aroll_vids:
                editing_mode = "dialogue_with_broll"
            else:
                editing_mode = "music_montage"

            chapters.append(
                ChapterPlan(
                    chapter_id=cid,
                    title=title_en,
                    subtitle=sub_en,
                    location_tag=loc_tag,
                    time_of_day=tod,  # type: ignore[arg-type]
                    editing_mode=editing_mode,  # type: ignore[arg-type]
                    photo_integration_mode="ken_burns_pause" if editing_mode == "dialogue_with_broll" else "beat_burst",
                    selected_subclip_ids=[s.subclip_id for s in ordered_selects],
                    candid_reaction_ids=candid_ids,
                    broll_anchor_map=broll_anchor_map,
                    story_stage=story_stage,
                    target_energy=target_energy,
                    shot_grammar_sequence=grammar_seq,
                    narrative_goal=summary,
                )
            )

        # Mechanically export candidate_story_digest.json so Multimodal VLM / AI Agent can review & refine
        digest_doc = {
            "project_title": brief.project_title if brief else "Family_Travel_Story",
            "user_prompt": prompt_text,
            "story_arc_mode": story_arc_mode,
            "video_candidates": [
                {
                    "subclip_id": v.subclip_id,
                    "filename": v.filename,
                    "roll_type": v.roll_type,
                    "shot_scale": v.shot_scale,
                    "duration": v.duration,
                    "action_peak_tc": v.action_peak_tc,
                    "stability_score": v.stability_score,
                    "prosody_excitement_score": v.prosody_excitement_score,
                    "emotion_score": v.emotion_score,
                    "aesthetic_score": v.aesthetic_score,
                    "spoken_text": v.spoken_text,
                    "summary": v.summary,
                }
                for v in videos
            ],
            "photo_candidates": [
                {
                    "subclip_id": p.subclip_id,
                    "filename": p.filename,
                    "shot_scale": p.shot_scale,
                    "subject_category": p.subject_category,
                    "annotation_title": p.annotation_title,
                    "annotation_subtitle": p.annotation_subtitle,
                    "aesthetic_score": p.aesthetic_score,
                }
                for p in photos
            ],
        }
        (blackboard.project_dir / "candidate_story_digest.json").write_text(
            json.dumps(digest_doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        ag_plan = self.ask_ai_creative_json(
            blackboard=blackboard,
            prompt=(
                "You are the Story Producer (Stage 20). Review this candidate_story_digest.json and return JSON with optional "
                "keys: cold_open_subclip_id, cold_open_hook_caption, cold_open_rationale, story_arc_summary, chapter_titles "
                "(list of {title, subtitle, narrative_goal}), or chapters (list of chapter overrides).\n\n"
                + json.dumps(digest_doc, ensure_ascii=False)[:4600]
            ),
            decision_key="story_producer",
            timeout_sec=35.0,
        )
        story_arc_summary = (
            f"Cold-Open Teaser Hook followed by a {len(chapters)}-chapter {story_arc_mode.replace('_', ' ')} narrative arc "
            "sequenced with Five-Shot cinematic grammar, semantic Track V2 cutaways, and beat-synced photo choreography."
        )
        if isinstance(ag_plan, dict):
            if ag_plan.get("cold_open_subclip_id") in {v.subclip_id for v in videos}:
                cold_open_sid = str(ag_plan["cold_open_subclip_id"])
            if ag_plan.get("cold_open_hook_caption"):
                cold_open_caption = str(ag_plan["cold_open_hook_caption"])
            if ag_plan.get("cold_open_rationale"):
                cold_open_rationale = str(ag_plan["cold_open_rationale"])
            if ag_plan.get("story_arc_summary"):
                story_arc_summary = str(ag_plan["story_arc_summary"])
            ag_titles = ag_plan.get("chapter_titles")
            if isinstance(ag_titles, list):
                for idx_t, t_obj in enumerate(ag_titles[: len(chapters)]):
                    if isinstance(t_obj, dict):
                        if t_obj.get("title"):
                            chapters[idx_t].title = str(t_obj["title"])
                        if t_obj.get("subtitle"):
                            chapters[idx_t].subtitle = str(t_obj["subtitle"])
                        if t_obj.get("narrative_goal"):
                            chapters[idx_t].narrative_goal = str(t_obj["narrative_goal"])

        cold_open_slot: Optional[ColdOpenHookSlot] = None
        if cold_open_sid is not None:
            cold_open_slot = ColdOpenHookSlot(
                enabled=True,
                subclip_id=cold_open_sid,
                in_tc=float(cold_open_in or 0.0),
                out_tc=float(cold_open_out or min(6.0, (cold_open_in or 0.0) + 3.8)),
                target_window_sec=(0.0, 6.0),
                caption=cold_open_caption,
                rationale=cold_open_rationale,
            )

        paper_edit = PaperEdit(
            project_title=brief.project_title if brief else "Family_Travel_Story",
            story_arc_summary=story_arc_summary,
            story_arc_mode=story_arc_mode,
            cold_open_hook=cold_open_slot,
            cold_open_subclip_id=cold_open_sid,
            cold_open_in_tc=cold_open_in,
            cold_open_out_tc=cold_open_out,
            cold_open_hook_caption=cold_open_caption,
            cold_open_rationale=cold_open_rationale,
            chapters=chapters,
        )

        out_path = blackboard.project_dir / "paper_edit.json"
        paper_edit.to_json_file(out_path)
        if isinstance(ag_plan, dict) and isinstance(ag_plan.get("chapters"), list):
            try:
                apply_story_plan_overrides(blackboard.project_dir, ag_plan)
                paper_edit = PaperEdit.from_json_file(out_path)
            except Exception:
                pass
        blackboard.paper_edit = paper_edit
        hook_note = f" + Cold-Open Hook ({cold_open_sid})" if cold_open_sid else ""
        blackboard.emit_activity(
            self.role_id,
            f"Architected {len(paper_edit.chapters)}-Chapter Paper Edit ({story_arc_mode}){hook_note} with {sum(len(c.selected_subclip_ids) for c in paper_edit.chapters)} Five-Shot sequenced story beats",
            detail="Chapters: " + " | ".join(f"{c.title} ({len(c.selected_subclip_ids)} clips)" for c in paper_edit.chapters),
        )

        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="dialogue_editor",
            message_type="handoff",
            subject="Paper edit ready with Cold-Open Hook, Five-Shot grammar & semantic V2 cutaway anchors",
            payload={
                "cold_open_subclip_id": cold_open_sid,
                "story_arc_mode": story_arc_mode,
                "chapters": len(paper_edit.chapters),
                "v2_cutaway_anchors": sum(len(c.broll_anchor_map) for c in paper_edit.chapters),
            },
        )
        blackboard.send_message(
            sender_role=self.role_id,
            recipient_role="montage_editor",
            message_type="handoff",
            subject="Paper edit ready for beat-synced B-roll and photo montage assembly",
            payload={
                "chapters": len(paper_edit.chapters),
                "cold_open_subclip_id": cold_open_sid,
                "story_arc_mode": story_arc_mode,
            },
        )

        return {
            "paper_edit_path": str(out_path),
            "story_arc_mode": story_arc_mode,
            "cold_open_subclip_id": cold_open_sid,
            "chapter_count": len(paper_edit.chapters),
            "v2_cutaway_count": sum(len(c.broll_anchor_map) for c in paper_edit.chapters),
        }


def apply_story_plan_overrides(
    project_dir: str | Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Mechanically apply AI-directed Cold-Open Hook and chapter edits to paper_edit.json."""
    proj = Path(project_dir).resolve()
    pe_path = proj / "paper_edit.json"
    if not pe_path.exists():
        raise FileNotFoundError(f"paper_edit.json not found in {proj}")

    pe = PaperEdit.from_json_file(pe_path)
    if "cold_open_subclip_id" in overrides:
        pe.cold_open_subclip_id = overrides["cold_open_subclip_id"]
    if "cold_open_in_tc" in overrides:
        pe.cold_open_in_tc = float(overrides["cold_open_in_tc"]) if overrides["cold_open_in_tc"] is not None else None
    if "cold_open_out_tc" in overrides:
        pe.cold_open_out_tc = float(overrides["cold_open_out_tc"]) if overrides["cold_open_out_tc"] is not None else None
    if "cold_open_hook_caption" in overrides:
        pe.cold_open_hook_caption = str(overrides["cold_open_hook_caption"] or "")
    if "cold_open_rationale" in overrides:
        pe.cold_open_rationale = str(overrides["cold_open_rationale"] or "")
    if overrides.get("story_arc_summary"):
        pe.story_arc_summary = str(overrides["story_arc_summary"])
    if overrides.get("story_arc_mode") in ("chronological_cinema", "hero_journey", "thematic_essay", "high_energy_reel"):
        pe.story_arc_mode = overrides["story_arc_mode"]

    chap_overrides = overrides.get("chapters")
    if isinstance(chap_overrides, list):
        chap_by_id = {c.chapter_id: c for c in pe.chapters}
        for idx, cov in enumerate(chap_overrides):
            if not isinstance(cov, dict):
                continue
            cid = cov.get("chapter_id") or (pe.chapters[idx].chapter_id if idx < len(pe.chapters) else None)
            target_c = chap_by_id.get(cid) if cid else None
            if target_c is None:
                continue
            if cov.get("title"):
                target_c.title = str(cov["title"])
            if "subtitle" in cov:
                target_c.subtitle = str(cov["subtitle"] or "")
            if cov.get("narrative_goal"):
                target_c.narrative_goal = str(cov["narrative_goal"])
            if isinstance(cov.get("selected_subclip_ids"), list):
                target_c.selected_subclip_ids = [str(x) for x in cov["selected_subclip_ids"]]
            if isinstance(cov.get("broll_anchor_map"), dict):
                target_c.broll_anchor_map = {
                    str(k): [str(v) for v in val] for k, val in cov["broll_anchor_map"].items() if isinstance(val, list)
                }

    pe.to_json_file(pe_path)
    return {
        "paper_edit_path": str(pe_path),
        "cold_open_subclip_id": pe.cold_open_subclip_id,
        "cold_open_hook_caption": pe.cold_open_hook_caption,
        "chapters_count": len(pe.chapters),
    }
