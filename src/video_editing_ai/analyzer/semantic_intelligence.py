"""Unified Local Neural Semantic Embedding & Perceptual Computer Vision Intelligence Engine."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
from typing import Any, Literal, Optional

import cv2
import numpy as np
from PIL import Image


# Cross-domain semantic concept clusters for instant zero-network expansion & fallback vectorization
_CONCEPT_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("pastry", "pastries", "bakery", "croissant", "bread", "cake", "dessert", "cafe", "coffee", "espresso", "latte", "tea", "dining", "food", "culinary", "meal", "gelato", "icecream", "snack", "breakfast", "lunch", "dinner", "restaurant", "bistro"),
    ("family", "kid", "kids", "child", "children", "daughter", "son", "parent", "mom", "dad", "portrait", "smile", "smiling", "laugh", "laughing", "happy", "joy", "joyful", "candid", "hug", "together", "selfie", "person", "people"),
    ("architecture", "building", "palace", "castle", "cathedral", "church", "tower", "arch", "facade", "column", "dome", "courtyard", "plaza", "square", "monument", "landmark", "skyline", "bridge", "street", "avenue", "cityscape"),
    ("art", "artwork", "painting", "canvas", "sculpture", "statue", "marble", "gallery", "exhibit", "museum", "fresco", "mural", "mosaic", "relief", "masterpiece", "artifact", "carving", "frame", "gilded"),
    ("nature", "park", "garden", "tree", "trees", "flower", "flowers", "forest", "trail", "mountain", "hill", "lake", "river", "water", "waterfront", "ocean", "sea", "beach", "sky", "clouds", "lawn", "outdoor"),
    ("sunset", "twilight", "dusk", "evening", "golden", "glow", "warm", "night", "illuminated", "lights", "lantern", "fireworks", "reflection", "silhouette", "horizon", "dawn", "sunrise"),
    ("stage", "show", "performance", "concert", "music", "musical", "singing", "song", "dance", "dancing", "parade", "festival", "applause", "cheer", "crowd", "theater"),
    ("arrival", "entrance", "gate", "welcome", "morning", "start", "opening", "journey", "travel", "walk", "walking", "stroll", "exploring", "discovery"),
)

_WORD_TO_CONCEPT_IDS: dict[str, list[int]] = {}
for _c_idx, _words in enumerate(_CONCEPT_SYNONYM_GROUPS):
    for _w in _words:
        _WORD_TO_CONCEPT_IDS.setdefault(_w, []).append(_c_idx)


def compute_dhash_hex_from_gray(gray_arr: np.ndarray) -> str:
    """Compute 64-bit horizontal gradient difference hash (dHash) from a 2D grayscale array."""
    try:
        resized = cv2.resize(gray_arr.astype(np.uint8), (9, 8), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        val = 0
        for b in diff.flatten():
            val = (val << 1) | int(bool(b))
        return f"{val:016x}"
    except Exception:
        return "0000000000000000"


def hamming_distance_hex(h1: str, h2: str) -> int:
    """Return bit-level Hamming distance between two 64-bit hex dHash strings."""
    try:
        return (int(h1, 16) ^ int(h2, 16)).bit_count()
    except Exception:
        return 64


def _tokenize_clean(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", (text or "").lower())
    stemmed: list[str] = []
    for tok in tokens:
        stemmed.append(tok)
        if len(tok) > 4 and tok.endswith("ies"):
            stemmed.append(tok[:-3] + "y")
        elif len(tok) > 4 and tok.endswith("ing"):
            stemmed.append(tok[:-3])
        elif len(tok) > 3 and tok.endswith("ed"):
            stemmed.append(tok[:-2])
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            stemmed.append(tok[:-1])
    return stemmed


def _deterministic_concept_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Fast, offline-deterministic 384-D semantic concept + hashed n-gram vector."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize_clean(text)
    if not tokens:
        vec[0] = 1.0
        return vec

    # 1. Concept-cluster activation slots [0 .. 63]
    for tok in tokens:
        for cid in _WORD_TO_CONCEPT_IDS.get(tok, ()):
            slot = cid % 64
            vec[slot] += 2.4

    # 2. Hashed token + character 3-gram projection [64 .. dim-1]
    proj_span = dim - 64
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        h_val = int.from_bytes(digest[:4], "little")
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[64 + (h_val % proj_span)] += sign * 1.0
        if len(tok) >= 4:
            for i in range(len(tok) - 2):
                tri = tok[i : i + 3]
                td = hashlib.blake2b(tri.encode("utf-8"), digest_size=6).digest()
                th = int.from_bytes(td[:4], "little")
                tsign = 1.0 if (td[4] & 1) == 0 else -1.0
                vec[64 + (th % proj_span)] += tsign * 0.35

    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    else:
        vec[0] = 1.0
    return vec


class SemanticEmbeddingEngine:
    """
    Hybrid Neural + Concept Semantic Embedding Engine.
    Uses locally cached SentenceTransformer('all-MiniLM-L6-v2') when available in cache,
    fused with domain concept-cluster features, and falls back cleanly with zero network blocking.
    """

    _instance: Optional["SemanticEmbeddingEngine"] = None

    def __init__(self) -> None:
        self._st_model: Any = None
        self._st_attempted: bool = False
        self._text_cache: dict[str, np.ndarray] = {}
        self._vis_cache: dict[tuple[str, int], dict[str, Any]] = {}

    @classmethod
    def instance(cls) -> "SemanticEmbeddingEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_st_model(self) -> Any:
        if self._st_attempted:
            return self._st_model
        self._st_attempted = True
        mode = os.environ.get("AI_EDITOR_EMBEDDING_MODE", "auto").strip().lower()
        if mode == "fast":
            return None
        allow_download = os.environ.get("AI_EDITOR_ALLOW_HF_DOWNLOAD", "0").strip() == "1"
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                local_files_only=not allow_download,
            )
        except Exception:
            self._st_model = None
        return self._st_model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings into L2-normalized 384-D semantic vectors."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        missing_indices: list[int] = []
        missing_texts: list[str] = []
        out = np.zeros((len(texts), 384), dtype=np.float32)

        for idx, raw in enumerate(texts):
            key = (raw or "").strip()
            cached = self._text_cache.get(key)
            if cached is not None:
                out[idx] = cached
            else:
                missing_indices.append(idx)
                missing_texts.append(key)

        if missing_texts:
            model = self._get_st_model()
            neural_vecs: Optional[np.ndarray] = None
            if model is not None:
                try:
                    encoded = model.encode(
                        missing_texts,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    neural_vecs = np.asarray(encoded, dtype=np.float32)
                except Exception:
                    neural_vecs = None

            for m_i, key in enumerate(missing_texts):
                det_v = _deterministic_concept_embedding(key, dim=384)
                if neural_vecs is not None and neural_vecs.shape[1] == 384:
                    # Blend 82% neural transformer representation + 18% concept-synonym anchor
                    combined = 0.82 * neural_vecs[m_i] + 0.18 * det_v
                    norm = float(np.linalg.norm(combined))
                    vec = combined / norm if norm > 1e-8 else det_v
                else:
                    vec = det_v
                if len(self._text_cache) > 4096:
                    self._text_cache.clear()
                self._text_cache[key] = vec
                out[missing_indices[m_i]] = vec

        return out

    def semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic cosine similarity between two strings in [-1.0, 1.0]."""
        if not (text_a or "").strip() or not (text_b or "").strip():
            return 0.0
        vecs = self.embed_texts([text_a, text_b])
        return float(np.clip(np.dot(vecs[0], vecs[1]), -1.0, 1.0))

    def best_topic_similarity(self, text_blob: str, topics: list[str]) -> float:
        """Return maximum semantic cosine similarity between text_blob and any topic phrase."""
        clean_topics = [t.strip() for t in topics if t and t.strip()]
        if not clean_topics or not (text_blob or "").strip():
            return 0.0
        vecs = self.embed_texts([text_blob, *clean_topics])
        sims = vecs[1:] @ vecs[0]
        return float(np.max(sims)) if sims.size > 0 else 0.0


def _compute_region_signature_128d(bgr_img: np.ndarray) -> np.ndarray:
    """
    Compute a 128-D L2-normalized visual descriptor combining:
    - 48-D CIE-Lab 3D joint color histogram (4 L x 4 a x 3 b)
    - 72-D 3x3 Spatial Pyramid 8-bin Sobel Edge-Orientation Histogram
    - 8-D global luminance/chromaticity/contrast moments
    """
    resized = cv2.resize(bgr_img, (192, 192), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 1. 48-D Lab color distribution
    hist_lab = cv2.calcHist([lab], [0, 1, 2], None, [4, 4, 3], [0, 256, 0, 256, 0, 256]).flatten().astype(np.float32)
    hist_lab /= float(np.sum(hist_lab)) + 1e-6

    # 2. 72-D 3x3 Spatial Edge-Orientation Pyramid (3 x 3 cells x 8 orientation bins)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + math.pi) / (2.0 * math.pi)  # [0, 1]
    bins = np.clip((ang * 8.0).astype(np.int32), 0, 7)

    edge_feats = np.zeros(72, dtype=np.float32)
    cell_h, cell_w = 64, 64
    idx = 0
    for cy in range(3):
        for cx in range(3):
            sub_mag = mag[cy * cell_h : (cy + 1) * cell_h, cx * cell_w : (cx + 1) * cell_w]
            sub_bin = bins[cy * cell_h : (cy + 1) * cell_h, cx * cell_w : (cx + 1) * cell_w]
            for b_i in range(8):
                edge_feats[idx + b_i] = float(np.sum(sub_mag[sub_bin == b_i]))
            cell_norm = float(np.sum(edge_feats[idx : idx + 8])) + 1e-5
            edge_feats[idx : idx + 8] /= cell_norm
            idx += 8

    # 3. 8-D global Lab & sharpness moments
    l_ch = lab[:, :, 0].astype(np.float32) / 255.0
    a_ch = (lab[:, :, 1].astype(np.float32) - 128.0) / 128.0
    b_ch = (lab[:, :, 2].astype(np.float32) - 128.0) / 128.0
    moments = np.array(
        [
            float(np.mean(l_ch)),
            float(np.std(l_ch)),
            float(np.mean(a_ch)),
            float(np.std(a_ch)),
            float(np.mean(b_ch)),
            float(np.std(b_ch)),
            float(np.percentile(l_ch, 90) - np.percentile(l_ch, 10)),
            float(min(1.0, np.mean(mag) / 64.0)),
        ],
        dtype=np.float32,
    )

    sig = np.concatenate([hist_lab * 1.35, edge_feats * 0.85, moments * 0.50]).astype(np.float32)
    norm = float(np.linalg.norm(sig))
    return sig / norm if norm > 1e-8 else sig


def compute_zoom_tolerant_visual_similarity(
    full_a: np.ndarray,
    center_a: np.ndarray,
    inner_a: np.ndarray,
    full_b: np.ndarray,
    center_b: np.ndarray,
    inner_b: np.ndarray,
) -> float:
    """
    Compute visual similarity between two photos while catching 1.3x-2.0x optical zoom crops
    by comparing full-frame descriptors as well as center-66% / inner-50% crops against full frames.
    """
    sim_full = float(np.dot(full_a, full_b))
    sim_a_zoom_b = float(np.dot(center_a, full_b)) * 0.985
    sim_b_zoom_a = float(np.dot(full_a, center_b)) * 0.985
    sim_a_deep_b = float(np.dot(inner_a, full_b)) * 0.975
    sim_b_deep_a = float(np.dot(full_a, inner_b)) * 0.975
    return float(max(sim_full, sim_a_zoom_b, sim_b_zoom_a, sim_a_deep_b, sim_b_deep_a))


def cie_lab_harmony_score(
    lab_a: tuple[float, float, float],
    lab_b: tuple[float, float, float],
) -> float:
    """
    Score chromatic & luminance-temperature compatibility in [0.0, 1.0] between two CIE-Lab means.
    High scores (> 0.65) indicate harmonious palettes suitable for Diptych / Triptych pairing.
    """
    dl = lab_a[0] - lab_b[0]
    da = lab_a[1] - lab_b[1]
    db = lab_a[2] - lab_b[2]
    delta_e = math.sqrt(0.30 * dl * dl + 1.15 * da * da + 1.15 * db * db)
    return round(float(math.exp(-delta_e / 30.0)), 4)


_CASCADE_CACHE: dict[str, Any] = {}


def _get_haar_cascade(xml_name: str) -> Any:
    if xml_name in _CASCADE_CACHE:
        return _CASCADE_CACHE[xml_name]
    try:
        cascade_path = Path(cv2.data.haarcascades) / xml_name
        if cascade_path.exists():
            clf = cv2.CascadeClassifier(str(cascade_path))
            _CASCADE_CACHE[xml_name] = clf
            return clf
    except Exception:
        pass
    _CASCADE_CACHE[xml_name] = None
    return None


def _detect_faces_and_skin_heads(bgr: np.ndarray, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect human faces via OpenCV Haar cascades plus skin-tone head-oval fallback."""
    h, w = gray.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    min_side = max(32, int(min(h, w) * 0.10))

    for xml_name in ("haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
        clf = _get_haar_cascade(xml_name)
        if clf is not None:
            try:
                found = clf.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(min_side, min_side))
                for (fx, fy, fw, fh) in found:
                    boxes.append((int(fx), int(fy), int(fw), int(fh)))
            except Exception:
                pass
        if boxes:
            break

    if not boxes:
        # Fallback skin-ellipse portrait head detector (essential when closed/blinking eyes cause Haar frontalface to miss!)
        # Crop upper 60% portrait head zone so warm shirts/torsos never merge with the facial contour.
        head_h = max(32, int(h * 0.60))
        ycrcb = cv2.cvtColor(bgr[:head_h, :], cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(
            ycrcb,
            np.array([80, 135, 88], dtype=np.uint8),
            np.array([248, 170, 124], dtype=np.uint8),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (h * w) * 0.020 or area > (h * w) * 0.55:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            aspect = rh / max(1.0, float(rw))
            fill_ratio = area / max(1.0, float(rw * rh))
            if 0.68 <= aspect <= 1.90 and fill_ratio >= 0.45:
                boxes.append((int(rx), int(ry), int(rw), int(rh)))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:4]


def _evaluate_eye_open_and_smile(
    gray: np.ndarray,
    bgr: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> tuple[float, float]:
    """
    Estimate (eye_open_score, smile_score) in [0.0, 1.0] inside a detected face box.
    Distinguishes crisp open eyes (dark iris/pupil core + sclera contrast + vertical opening)
    from closed/blinking eyes (flat skin with thin horizontal eyelid crease).
    """
    h, w = gray.shape[:2]
    fx, fy, fw, fh = face_box
    if fw < 36 or fh < 36:
        return 0.85, 0.25

    # Evaluate left and right inner anatomical eye boxes [0.20..0.47] & [0.53..0.80]
    # so temple hair / sideburns along outer face edges never pollute eye openness metrics.
    ey0 = max(0, min(h - 4, int(fy + fh * 0.23)))
    ey1 = max(ey0 + 4, min(h, int(fy + fh * 0.49)))
    eye_clf = _get_haar_cascade("haarcascade_eye_tree_eyeglasses.xml") or _get_haar_cascade("haarcascade_eye.xml")

    per_eye_scores: list[float] = []
    for x_lo_frac, x_hi_frac in ((0.20, 0.47), (0.53, 0.80)):
        ex0 = max(0, min(w - 4, int(fx + fw * x_lo_frac)))
        ex1 = max(ex0 + 4, min(w, int(fx + fw * x_hi_frac)))
        sub_eye = gray[ey0:ey1, ex0:ex1]
        if sub_eye.size < 36:
            continue

        roi_f = sub_eye.astype(np.float32)
        p92 = float(np.percentile(roi_f, 92))
        p08 = float(np.percentile(roi_f, 8))
        contrast_span = p92 - p08
        gx = np.abs(cv2.Sobel(roi_f, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(roi_f, cv2.CV_32F, 0, 1, ksize=3))
        mean_gx = float(np.mean(gx)) + 1e-4
        mean_gy = float(np.mean(gy)) + 1e-4
        # Open eyes have dark rounded pupils/irises producing strong horizontal (gx) AND vertical (gy) edges,
        # whereas closed eyelids have only a thin horizontal line (gy >> gx) and weak dark-core contrast.
        hv_balance = min(1.3, mean_gx / mean_gy)
        dark_iris_frac = float(np.mean(roi_f < max(38.0, p92 - 55.0)))

        has_cascade_eye = False
        if eye_clf is not None and contrast_span >= 44.0 and hv_balance >= 0.34:
            try:
                min_eye = max(8, int(fw * 0.09))
                det_eyes = eye_clf.detectMultiScale(
                    sub_eye,
                    scaleFactor=1.10,
                    minNeighbors=3,
                    minSize=(min_eye, max(6, int(min_eye * 0.55))),
                )
                has_cascade_eye = len(det_eyes) > 0
            except Exception:
                has_cascade_eye = False

        if contrast_span < 38.0 or hv_balance < 0.28 or dark_iris_frac < 0.015:
            # Closed/blinking eyelid signature
            sub_score = 0.16 + min(0.14, hv_balance * 0.30) + min(0.08, max(0.0, contrast_span - 12.0) / 220.0)
        else:
            sub_score = (
                0.42
                + (0.24 if has_cascade_eye else 0.0)
                + min(0.22, max(0.0, (contrast_span - 38.0) / 110.0))
                + min(0.18, hv_balance * 0.22)
                + (0.12 if 0.02 <= dark_iris_frac <= 0.42 else 0.0)
            )
        per_eye_scores.append(float(np.clip(sub_score, 0.12, 1.0)))

    eye_open_score = round(float(np.mean(per_eye_scores)), 3) if per_eye_scores else 0.75

    # Lower-face smile band
    my0 = max(0, min(h - 4, int(fy + fh * 0.58)))
    my1 = max(my0 + 4, min(h, int(fy + fh * 0.92)))
    mx0 = max(0, min(w - 4, int(fx + fw * 0.20)))
    mx1 = max(mx0 + 4, min(w, int(fx + fw * 0.80)))
    mouth_roi = gray[my0:my1, mx0:mx1]

    smile_score = 0.20
    if mouth_roi.size >= 48:
        smile_clf = _get_haar_cascade("haarcascade_smile.xml")
        has_cascade_smile = False
        if smile_clf is not None:
            try:
                det_s = smile_clf.detectMultiScale(mouth_roi, scaleFactor=1.3, minNeighbors=10)
                has_cascade_smile = len(det_s) > 0
            except Exception:
                has_cascade_smile = False

        m_f = mouth_roi.astype(np.float32)
        teeth_bright_frac = float(np.mean(m_f > min(235.0, float(np.percentile(m_f, 70)) + 28.0)))
        m_std = float(np.std(m_f))
        raw_smile = 0.18 + (0.48 if has_cascade_smile else 0.0) + min(0.24, m_std / 85.0) + min(0.18, teeth_bright_frac * 1.4)
        smile_score = round(float(np.clip(raw_smile, 0.0, 1.0)), 3)

    return eye_open_score, smile_score


def analyze_human_and_aesthetic_perception(
    img_path: str | Path,
    filename: str = "",
) -> dict[str, Any]:
    """
    Comprehensive local vision perception pass on a photo or video keyframe:
    - Zoom-tolerant 128-D Lab + Spatial Edge-Orientation signatures (full, center-66%, inner-50%)
    - OpenCV Face/Head detection, Eye-Open (blink penalty), and Smile boost
    - Subject-ROI Laplacian sharpness vs. intentional shallow-DOF portrait bokeh
    - CIE-Lab palette mean for chromatic diptych/triptych harmony
    """
    p = Path(img_path)
    cache_engine = SemanticEmbeddingEngine.instance()
    mtime_key = 0
    if p.exists():
        try:
            mtime_key = p.stat().st_mtime_ns
        except Exception:
            mtime_key = 0
        cached = cache_engine._vis_cache.get((str(p.resolve()), mtime_key))
        if cached is not None:
            return dict(cached)

    try:
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            w_orig, h_orig = rgb.size
            scale = min(1.0, 640.0 / max(1, max(w_orig, h_orig)))
            new_w = max(64, int(round(w_orig * scale)))
            new_h = max(64, int(round(h_orig * scale)))
            rgb_work = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
            arr_rgb = np.asarray(rgb_work, dtype=np.uint8)
    except Exception:
        zeros_sig = np.zeros(128, dtype=np.float32)
        zeros_sig[0] = 1.0
        return {
            "dhash": "0000000000000000",
            "visual_signature": zeros_sig,
            "center_signature": zeros_sig,
            "inner_signature": zeros_sig,
            "lab_mean": (128.0, 128.0, 128.0),
            "face_boxes": [],
            "focal_point": (0.5, 0.5),
            "eye_open_score": 1.0,
            "smile_score": 0.2,
            "blur_type": "crisp_all",
            "saliency_score": 7.5,
            "composition_score": 7.5,
            "portrait_warmth_score": 7.0,
            "art_architecture_score": 7.5,
            "overall_taste_score": 7.5,
            "inferred_category": "landmark_architecture",
        }

    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    h, w = gray.shape[:2]

    dhash = compute_dhash_hex_from_gray(gray)
    full_sig = _compute_region_signature_128d(bgr)

    cy0, cy1 = int(h * 0.17), int(h * 0.83)
    cx0, cx1 = int(w * 0.17), int(w * 0.83)
    center_sig = _compute_region_signature_128d(bgr[cy0:cy1, cx0:cx1])

    iy0, iy1 = int(h * 0.25), int(h * 0.75)
    ix0, ix1 = int(w * 0.25), int(w * 0.75)
    inner_sig = _compute_region_signature_128d(bgr[iy0:iy1, ix0:ix1])

    lab_mean = (
        round(float(np.mean(lab[:, :, 0])), 2),
        round(float(np.mean(lab[:, :, 1])), 2),
        round(float(np.mean(lab[:, :, 2])), 2),
    )

    # 1. Face, blink & smile perception
    raw_faces = _detect_faces_and_skin_heads(bgr, gray)
    norm_faces: list[tuple[float, float, float, float]] = []
    eye_scores: list[float] = []
    smile_scores: list[float] = []
    for (fx, fy, fw, fh) in raw_faces:
        norm_faces.append(
            (
                round(fx / w, 4),
                round(fy / h, 4),
                round((fx + fw) / w, 4),
                round((fy + fh) / h, 4),
            )
        )
        eo, sm = _evaluate_eye_open_and_smile(gray, bgr, (fx, fy, fw, fh))
        eye_scores.append(eo)
        smile_scores.append(sm)

    eye_open_score = round(min(eye_scores), 3) if eye_scores else 1.0
    smile_score = round(max(smile_scores), 3) if smile_scores else 0.0

    # 2. Subject-ROI sharpness vs. background bokeh / motion blur
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_abs = np.abs(lap)
    if raw_faces:
        fx, fy, fw, fh = raw_faces[0]
        subj_roi = lap_abs[max(0, fy) : min(h, fy + fh), max(0, fx) : min(w, fx + fw)]
        focal_pt = (round((fx + 0.5 * fw) / w, 3), round((fy + 0.42 * fh) / h, 3))
    else:
        subj_roi = lap_abs[int(h * 0.18) : int(h * 0.80), int(w * 0.18) : int(w * 0.82)]
        # Gradient-energy centroid inside central 70% window
        gx_f = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy_f = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        e_map = np.hypot(gx_f, gy_f)
        ys, xs = np.mgrid[0:h, 0:w]
        tot_e = float(np.sum(e_map)) + 1e-5
        focal_pt = (
            round(float(np.clip(np.sum(xs * e_map) / (tot_e * w), 0.2, 0.8)), 3),
            round(float(np.clip(np.sum(ys * e_map) / (tot_e * h), 0.2, 0.8)), 3),
        )

    # For human faces, ~85% of facial area is smooth cheek/forehead skin while eyelashes, irises, and lips
    # sit in the top 4-6% edge-detail tail; p96 measures true focal-plane crispness on portraits!
    subj_pct = 96 if raw_faces else 90
    subj_sharp = float(np.percentile(subj_roi, subj_pct)) if subj_roi.size > 0 else float(np.percentile(lap_abs, 90))
    outer_mask = np.ones((h, w), dtype=bool)
    outer_mask[int(h * 0.18) : int(h * 0.82), int(w * 0.18) : int(w * 0.82)] = False
    outer_sharp = float(np.percentile(lap_abs[outer_mask], 90)) if np.any(outer_mask) else subj_sharp

    blur_type: Literal["crisp_all", "intentional_bokeh", "motion_or_defocus_blur"] = "crisp_all"
    blur_adjust = 0.0
    if subj_sharp < (16.0 if raw_faces else 11.5):
        blur_type = "motion_or_defocus_blur"
        blur_adjust = -1.65
    elif subj_sharp >= 24.0 and outer_sharp < subj_sharp * 0.52:
        blur_type = "intentional_bokeh"
        blur_adjust = 0.38

    # 3. Saliency, Composition, Portrait Warmth & Art/Architecture sub-scores
    arr_f = arr_rgb.astype(np.float32)
    r, g, b = arr_f[:, :, 0], arr_f[:, :, 1], arr_f[:, :, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    gy_l, gx_l = np.gradient(luma)
    energy = np.hypot(gx_l, gy_l)

    y0, y1 = int(h * 0.16), int(h * 0.80)
    x0, x1 = int(w * 0.16), int(w * 0.84)
    center_energy = float(np.mean(energy[y0:y1, x0:x1])) + 1e-5
    outer_energy = float(np.mean(energy)) + 1e-5
    bottom_strip_energy = float(np.mean(energy[int(h * 0.80) :, :])) + 1e-5

    focus_ratio = center_energy / outer_energy
    ground_penalty = max(0.0, (bottom_strip_energy / center_energy) - 1.08) * 1.8
    saliency_score = round(
        min(9.9, max(3.8, 7.35 + (focus_ratio - 0.95) * 3.6 - ground_penalty + min(0.55, subj_sharp / 85.0))),
        2,
    )

    luma_std = float(np.std(luma))
    luma_mean = float(np.mean(luma))
    clipped_frac = float(np.mean((luma > 250.0) | (luma < 6.0)))
    clip_penalty = max(0.0, clipped_frac - 0.16) * 2.5
    exposure_balance = 1.0 - min(1.0, abs(luma_mean - 122.0) / 110.0)
    composition_score = round(
        min(
            9.9,
            max(
                4.2,
                6.8 + min(1.8, luma_std / 34.0) + exposure_balance * 1.1 - ground_penalty * 0.5 - clip_penalty,
            ),
        ),
        2,
    )

    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    center_cb = cb[y0:y1, x0:x1]
    center_cr = cr[y0:y1, x0:x1]
    skin_mask = (center_cr >= 136.0) & (center_cr <= 172.0) & (center_cb >= 82.0) & (center_cb <= 124.0)
    skin_frac = float(np.mean(skin_mask))
    warm_diff = float(np.mean(r - b))
    face_bonus = 0.65 if norm_faces else 0.0
    portrait_warmth_score = round(
        min(
            9.9,
            max(
                5.0,
                6.7
                + min(2.0, skin_frac * 8.8)
                + min(0.85, max(0.0, warm_diff) / 28.0)
                + face_bonus
                + smile_score * 0.45,
            ),
        ),
        2,
    )

    sat = np.max(arr_f, axis=2) - np.min(arr_f, axis=2)
    color_richness = float(np.mean(sat[y0:y1, x0:x1]))
    border_warmth = float(np.mean((r - b)[int(h * 0.08) : int(h * 0.22), :]))
    art_arch_score = round(
        min(
            9.9,
            max(
                5.2,
                7.0
                + min(1.4, color_richness / 42.0)
                + min(0.9, max(0.0, border_warmth) / 30.0)
                + min(0.6, (focus_ratio - 0.9) * 2.0),
            ),
        ),
        2,
    )

    sky_blue = float(np.mean((b - r)[: int(h * 0.35), :]))
    if norm_faces or (skin_frac > 0.16 and h >= w):
        inferred_cat = "family_portrait"
    elif border_warmth > 18.0 and color_richness > 26.0:
        inferred_cat = "masterpiece_painting"
    elif sky_blue > 14.0 or w > h * 1.18:
        inferred_cat = "landmark_architecture"
    else:
        inferred_cat = "sculpture"

    # Human expression & blink adjustment
    blink_penalty = 0.0
    if norm_faces and eye_open_score < 0.48:
        blink_penalty = (0.52 - eye_open_score) * 3.4  # up to ~1.4 points penalty for closed eyes

    expression_bonus = smile_score * 0.42 if norm_faces else 0.0

    raw_overall = (
        0.33 * saliency_score
        + 0.27 * composition_score
        + 0.22 * art_arch_score
        + 0.18 * portrait_warmth_score
        + blur_adjust
        + expression_bonus
        - blink_penalty
    )
    overall = round(float(np.clip(raw_overall, 2.5, 9.95)), 2)

    res: dict[str, Any] = {
        "dhash": dhash,
        "visual_signature": full_sig,
        "center_signature": center_sig,
        "inner_signature": inner_sig,
        "lab_mean": lab_mean,
        "face_boxes": norm_faces,
        "focal_point": focal_pt,
        "eye_open_score": eye_open_score,
        "smile_score": smile_score,
        "blur_type": blur_type,
        "saliency_score": saliency_score,
        "composition_score": composition_score,
        "portrait_warmth_score": portrait_warmth_score,
        "art_architecture_score": art_arch_score,
        "overall_taste_score": overall,
        "inferred_category": inferred_cat,
    }
    if p.exists():
        if len(cache_engine._vis_cache) > 1024:
            cache_engine._vis_cache.clear()
        cache_engine._vis_cache[(str(p.resolve()), mtime_key)] = res
    return dict(res)
