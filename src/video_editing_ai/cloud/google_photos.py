"""Direct Google Photos Picker OAuth 2.0 PKCE + Two-Pass Smart Curation + Takeout Connector."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dt_time, timedelta, timezone
import hashlib
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, quote, urlencode, urlparse
import webbrowser
import zipfile
from zoneinfo import ZoneInfo

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..agents.photo_curator_agent import _evaluate_photo_taste_metrics, _hamming_distance_hex
from ..analyzer.media_probe import (
    PHOTO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    _extract_exif_camera_info,
    _extract_exif_gps_and_alt,
    _get_cjk_font,
    extract_exif_is_favorite,
    iter_media_files_recursive,
    probe_video_file,
)
from ..analyzer.semantic_intelligence import (
    analyze_human_and_aesthetic_perception,
    cie_lab_harmony_score,
    compute_zoom_tolerant_visual_similarity,
)
from ..hardware import get_io_worker_count
from ..models.schema import CloudMediaCandidateItem, GooglePhotosSyncManifest

PICKER_SCOPE = "https://www.googleapis.com/auth/photospicker.mediaitems.readonly"
DEFAULT_PICKER_API_BASE = "https://photospicker.googleapis.com/v1"
GOOGLE_AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"

_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
}
_EXT_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}


def get_default_google_photos_config_dir() -> Path:
    """Return per-user OS config directory outside the git repo for OAuth secrets/tokens."""
    env_dir = os.environ.get("GOOGLE_PHOTOS_CONFIG_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return (base / "video_editing_ai" / "google_photos").resolve()


def get_default_google_photos_cache_dir() -> Path:
    """Return persistent local cache directory for cloud thumbnails & downloaded originals."""
    env_dir = os.environ.get("GOOGLE_PHOTOS_CACHE_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (get_default_google_photos_config_dir() / "cache").resolve()


def generate_pkce_pair() -> tuple[str, str]:
    """Generate an RFC 7636 PKCE (code_verifier, S256 code_challenge) pair."""
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def build_autoclose_picker_uri(picker_uri: str) -> str:
    """Append `/autoclose` to a Picker session URI so the browser closes after Done."""
    uri = (picker_uri or "").strip()
    if not uri:
        return ""
    parsed = urlparse(uri)
    if parsed.path.rstrip("/").endswith("/autoclose") or parsed.query:
        return uri
    return uri.rstrip("/") + "/autoclose"


def format_picker_date_query(date_from: Optional[str], date_to: Optional[str]) -> str:
    """Format a copy-friendly Google Photos search bar date query string."""
    df = (date_from or "").strip()[:10]
    dt = (date_to or "").strip()[:10]
    if df and dt:
        return f"{df} to {dt}"
    if df:
        return f"after:{df}"
    if dt:
        return f"before:{dt}"
    return ""


def try_copy_to_clipboard(text: str) -> bool:
    """Best-effort clipboard helper across Windows, macOS, and Linux."""
    if not text:
        return False
    cmds: list[list[str]] = []
    if sys.platform == "win32":
        cmds.append(["clip"])
    elif sys.platform == "darwin":
        cmds.append(["pbcopy"])
    else:
        cmds.extend([["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]])
    for cmd in cmds:
        try:
            proc = subprocess.run(cmd, input=text, text=True, check=False, timeout=2, capture_output=True)
            if proc.returncode == 0:
                return True
        except Exception:
            continue
    return False


def _resolve_tzinfo(tz_name: str = "UTC"):
    name = (tz_name or "UTC").strip()
    if not name or name.upper() == "UTC":
        return timezone.utc
    if name.lower() == "local":
        return datetime.now().astimezone().tzinfo or timezone.utc
    return ZoneInfo(name)


def parse_iso_utc(raw_ts: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 / RFC 3339 or EXIF timestamp string into UTC datetime."""
    if not raw_ts:
        return None
    txt = str(raw_ts).strip()
    if not txt:
        return None
    norm = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", txt.replace("Z", "+00:00"))
    for fmt in (None, "%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            if fmt is None:
                dt = datetime.fromisoformat(norm)
            else:
                dt = datetime.strptime(txt[:19] if len(txt) >= 19 else txt, fmt)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return None


def _parse_single_bound(raw_val: str, is_end: bool, tz_name: str) -> datetime:
    txt = raw_val.strip()
    tzinfo = _resolve_tzinfo(tz_name)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", txt):
        d = datetime.strptime(txt, "%Y-%m-%d").date()
        t_part = dt_time(23, 59, 59, 999999) if is_end else dt_time(0, 0, 0, 0)
        return datetime.combine(d, t_part, tzinfo=tzinfo).astimezone(timezone.utc)
    norm = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", txt.replace("Z", "+00:00"))
    dt = datetime.fromisoformat(norm)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tzinfo)
    return dt.astimezone(timezone.utc)


def parse_date_range_bounds_utc(
    date_from: Optional[str],
    date_to: Optional[str],
    tz_name: str = "UTC",
) -> tuple[Optional[datetime], Optional[datetime]]:
    """Parse optional inclusive start/end date bounds into UTC datetimes."""
    start_utc = _parse_single_bound(date_from, is_end=False, tz_name=tz_name) if date_from else None
    end_utc = _parse_single_bound(date_to, is_end=True, tz_name=tz_name) if date_to else None
    if start_utc is not None and end_utc is not None and start_utc > end_utc:
        raise ValueError(f"date_from ({date_from}) must not be after date_to ({date_to}).")
    return start_utc, end_utc


def is_timestamp_in_range(
    iso_ts: Optional[str],
    start_utc: Optional[datetime],
    end_utc: Optional[datetime],
) -> bool:
    """Return True if timestamp falls inclusively inside [start_utc, end_utc]."""
    if start_utc is None and end_utc is None:
        return True
    dt = parse_iso_utc(iso_ts)
    if dt is None:
        return False
    if start_utc is not None and dt < start_utc:
        return False
    if end_utc is not None and dt > end_utc:
        return False
    return True


def filter_items_by_date_range(
    items: list[dict[str, Any]],
    date_from: Optional[str],
    date_to: Optional[str],
    tz_name: str = "UTC",
) -> list[dict[str, Any]]:
    """Filter raw Google Photos Picker PickedMediaItem payloads to [date_from, date_to]."""
    start_utc, end_utc = parse_date_range_bounds_utc(date_from, date_to, tz_name=tz_name)
    if start_utc is None and end_utc is None:
        return list(items)
    return [it for it in items if is_timestamp_in_range(it.get("createTime"), start_utc, end_utc)]


class GooglePhotosAuthManager:
    """Installed-App OAuth 2.0 PKCE token manager for the Google Photos Picker API."""

    def __init__(
        self,
        client_secrets_path: Optional[str | Path] = None,
        token_path: Optional[str | Path] = None,
        scope: str = PICKER_SCOPE,
    ) -> None:
        cfg_dir = get_default_google_photos_config_dir()
        env_secrets = os.environ.get("GOOGLE_PHOTOS_CLIENT_SECRETS", "").strip()
        env_token = os.environ.get("GOOGLE_PHOTOS_TOKEN_PATH", "").strip()
        self.client_secrets_path = Path(
            client_secrets_path or env_secrets or (cfg_dir / "client_secret.json")
        ).expanduser()
        self.token_path = Path(
            token_path or env_token or (cfg_dir / "token.json")
        ).expanduser()
        self.scope = scope

    def format_setup_instructions(self) -> str:
        return (
            "Google Photos OAuth 2.0 setup required (one-time personal Google Cloud configuration):\n"
            "  1. Open Google Cloud Console -> APIs & Services -> Library -> enable 'Google Photos Picker API'.\n"
            "  2. Configure OAuth Consent Screen (External -> Testing mode -> add your Gmail under Test Users).\n"
            "  3. Create Credentials -> OAuth Client ID -> Application type: 'Desktop app' -> Download JSON.\n"
            f"  4. Save that JSON to: {self.client_secrets_path}\n"
            "     (or set GOOGLE_PHOTOS_CLIENT_SECRETS / pass --client-secrets <path>).\n"
            "  5. Run: ai-editor google-photos-auth\n"
            "Alternative without Google Cloud OAuth: export via Google Takeout and pass --google-takeout <dir_or_zip>."
        )

    def _load_client_config(self) -> dict[str, str]:
        if not self.client_secrets_path.exists():
            raise FileNotFoundError(
                f"OAuth client_secret.json not found at {self.client_secrets_path}.\n\n"
                + self.format_setup_instructions()
            )
        raw = json.loads(self.client_secrets_path.read_text(encoding="utf-8"))
        section = raw.get("installed") or raw.get("web") or raw
        client_id = str(section.get("client_id") or "").strip()
        if not client_id:
            raise ValueError(f"Invalid OAuth client secrets file (missing client_id): {self.client_secrets_path}")
        return {
            "client_id": client_id,
            "client_secret": str(section.get("client_secret") or "").strip(),
            "auth_uri": str(section.get("auth_uri") or GOOGLE_AUTH_URI).strip(),
            "token_uri": str(section.get("token_uri") or GOOGLE_TOKEN_URI).strip(),
        }

    def _load_token_data(self) -> Optional[dict[str, Any]]:
        if not self.token_path.exists():
            return None
        try:
            data = json.loads(self.token_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _save_token_data(self, token_data: dict[str, Any]) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps(token_data, indent=2, ensure_ascii=False), encoding="utf-8")
        if os.name == "posix":
            try:
                os.chmod(self.token_path, 0o600)
            except Exception:
                pass

    def _exchange_code(
        self,
        cfg: dict[str, str],
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> dict[str, Any]:
        payload = {
            "client_id": cfg["client_id"],
            "code": code,
            "code_verifier": code_verifier,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }
        if cfg["client_secret"]:
            payload["client_secret"] = cfg["client_secret"]
        resp = httpx.post(cfg["token_uri"], data=payload, timeout=30.0)
        if resp.status_code >= 400:
            raise RuntimeError(f"OAuth token exchange failed ({resp.status_code}): {resp.text[:400]}")
        tok = dict(resp.json())
        tok["expires_at"] = time.time() + float(tok.get("expires_in", 3600))
        tok.setdefault("scope", self.scope)
        self._save_token_data(tok)
        return tok

    def authenticate_interactive(
        self,
        open_browser: bool = True,
        timeout_sec: float = 240.0,
        auth_url_callback: Optional[Callable[[str], None]] = None,
        manual_code: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run OAuth 2.0 Installed-App PKCE login via localhost loopback or callback hook."""
        cfg = self._load_client_config()
        verifier, challenge = generate_pkce_pair()
        expected_state = secrets.token_urlsafe(24)

        result_box: dict[str, Optional[str]] = {"code": manual_code, "error": None, "state": None}
        done_evt = threading.Event()
        if manual_code:
            done_evt.set()

        class _LoopbackHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                qs = parse_qs(parsed.query)
                code_val = (qs.get("code") or [None])[0]
                err_val = (qs.get("error") or [None])[0]
                st_val = (qs.get("state") or [None])[0]
                if st_val and st_val != expected_state:
                    result_box["error"] = "OAuth state mismatch."
                elif err_val:
                    result_box["error"] = err_val
                elif code_val:
                    result_box["code"] = code_val
                    result_box["state"] = st_val
                else:
                    result_box["error"] = "Missing authorization code in callback."

                ok = bool(result_box["code"] and not result_box["error"])
                body = (
                    "<html><body style='font-family:sans-serif;background:#0e141f;color:#eef2f7;padding:40px;'>"
                    f"<h2>{'Google Photos Connected' if ok else 'Authorization Failed'}</h2>"
                    f"<p>{'You may close this browser tab and return to the terminal.' if ok else escape(str(result_box['error']))}</p>"
                    "</body></html>"
                ).encode("utf-8")
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                done_evt.set()

        server = HTTPServer(("127.0.0.1", 0), _LoopbackHandler)
        redirect_uri = f"http://127.0.0.1:{server.server_address[1]}/callback"
        srv_thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        srv_thread.start()
        try:
            auth_params = {
                "client_id": cfg["client_id"],
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": self.scope,
                "access_type": "offline",
                "prompt": "consent",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": expected_state,
            }
            auth_url = f"{cfg['auth_uri']}?{urlencode(auth_params)}"
            if auth_url_callback is not None:
                auth_url_callback(auth_url)
            elif open_browser:
                try:
                    webbrowser.open(auth_url, new=2)
                except Exception:
                    pass

            if not done_evt.wait(timeout=max(1.0, float(timeout_sec))):
                raise TimeoutError(f"Timed out waiting for OAuth callback after {timeout_sec:.0f}s.")
            if result_box["error"]:
                raise RuntimeError(f"Google OAuth authorization failed: {result_box['error']}")
            if not result_box["code"]:
                raise RuntimeError("Did not receive OAuth authorization code.")
            return self._exchange_code(cfg, str(result_box["code"]), redirect_uri, verifier)
        finally:
            server.shutdown()
            server.server_close()

    def get_valid_access_token(self) -> str:
        """Return a fresh access token, refreshing automatically via refresh_token when near expiry."""
        tok = self._load_token_data()
        if not tok or not tok.get("access_token"):
            raise RuntimeError(
                "No Google Photos OAuth token found. Run 'ai-editor google-photos-auth' first."
            )
        expires_at = float(tok.get("expires_at") or 0.0)
        if expires_at - time.time() > 120.0:
            return str(tok["access_token"])

        refresh_tok = str(tok.get("refresh_token") or "").strip()
        if not refresh_tok:
            raise RuntimeError(
                "Google Photos access token expired and no refresh_token is stored. Re-run 'ai-editor google-photos-auth'."
            )
        cfg = self._load_client_config()
        payload = {
            "client_id": cfg["client_id"],
            "refresh_token": refresh_tok,
            "grant_type": "refresh_token",
        }
        if cfg["client_secret"]:
            payload["client_secret"] = cfg["client_secret"]
        resp = httpx.post(cfg["token_uri"], data=payload, timeout=30.0)
        if resp.status_code >= 400:
            raise RuntimeError(f"Google Photos token refresh failed ({resp.status_code}): {resp.text[:400]}")
        new_tok = dict(resp.json())
        new_tok["expires_at"] = time.time() + float(new_tok.get("expires_in", 3600))
        if not new_tok.get("refresh_token"):
            new_tok["refresh_token"] = refresh_tok
        new_tok.setdefault("scope", tok.get("scope", self.scope))
        self._save_token_data(new_tok)
        return str(new_tok["access_token"])

    def revoke_local_token(self) -> bool:
        if self.token_path.exists():
            self.token_path.unlink()
            return True
        return False

    def get_auth_status(self) -> dict[str, Any]:
        tok = self._load_token_data() or {}
        expires_at = float(tok.get("expires_at") or 0.0)
        has_access = bool(tok.get("access_token"))
        has_refresh = bool(tok.get("refresh_token"))
        not_expired = expires_at > (time.time() + 60.0)
        return {
            "config_dir": str(self.client_secrets_path.parent),
            "client_secrets_path": str(self.client_secrets_path),
            "client_secrets_configured": self.client_secrets_path.is_file(),
            "token_path": str(self.token_path),
            "token_present": self.token_path.is_file(),
            "token_valid": bool(has_access and not_expired),
            "refresh_token_present": has_refresh,
            "ready": bool(self.client_secrets_path.is_file() and has_access and (not_expired or has_refresh)),
            "scope": str(tok.get("scope") or self.scope),
        }


def _parse_duration_seconds(raw_val: Any, default_sec: float = 3.0) -> float:
    if isinstance(raw_val, (int, float)):
        return max(0.05, float(raw_val))
    txt = str(raw_val or "").strip().lower()
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*s?", txt)
    if m:
        return max(0.05, float(m.group(1)))
    return default_sec


class GooglePhotosPickerClient:
    """REST client for Google Photos Picker API (`photospicker.googleapis.com/v1`) with pooled HTTP/2 keepalive."""

    def __init__(
        self,
        auth_manager: GooglePhotosAuthManager,
        api_base_url: Optional[str] = None,
        timeout_sec: float = 45.0,
    ) -> None:
        self.auth_manager = auth_manager
        base = api_base_url or os.environ.get("GOOGLE_PHOTOS_API_BASE_URL", "").strip() or DEFAULT_PICKER_API_BASE
        self.api_base_url = base.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self._client_lock = threading.Lock()
        self._http_client: Optional[httpx.Client] = None

    def _client(self) -> httpx.Client:
        with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                limits = httpx.Limits(max_connections=24, max_keepalive_connections=16)
                try:
                    self._http_client = httpx.Client(http2=True, limits=limits, follow_redirects=True)
                except Exception:
                    self._http_client = httpx.Client(limits=limits, follow_redirects=True)
            return self._http_client

    def close(self) -> None:
        with self._client_lock:
            if self._http_client is not None and not self._http_client.is_closed:
                try:
                    self._http_client.close()
                except Exception:
                    pass
                self._http_client = None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.auth_manager.get_valid_access_token()}"}

    def create_session(self, max_item_count: int = 2000) -> dict[str, Any]:
        capped = max(1, min(2000, int(max_item_count)))
        resp = self._client().post(
            f"{self.api_base_url}/sessions",
            headers=self._headers(),
            json={"pickingConfig": {"maxItemCount": capped}},
            timeout=self.timeout_sec,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Picker sessions.create failed ({resp.status_code}): {resp.text[:400]}")
        return dict(resp.json())

    def get_session(self, session_id: str) -> dict[str, Any]:
        resp = self._client().get(
            f"{self.api_base_url}/sessions/{session_id}",
            headers=self._headers(),
            timeout=self.timeout_sec,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Picker sessions.get failed ({resp.status_code}): {resp.text[:400]}")
        return dict(resp.json())

    def poll_until_ready(
        self,
        session_id: str,
        timeout_sec: float = 600.0,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        while True:
            sess = self.get_session(session_id)
            if bool(sess.get("mediaItemsSet")):
                return sess
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Google Photos Picker selection (session={session_id}).")
            poll_cfg = sess.get("pollingConfig") or {}
            wait_s = min(5.0, _parse_duration_seconds(poll_cfg.get("pollInterval"), default_sec=3.0))
            if progress_callback:
                progress_callback(f"Waiting for Google Photos selection completion (poll {wait_s:.1f}s)...")
            time.sleep(wait_s)

    def list_session_media_items(self, session_id: str, page_size: int = 100) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        page_token: Optional[str] = None
        capped_size = max(1, min(100, int(page_size)))
        while True:
            params: dict[str, Any] = {"sessionId": session_id, "pageSize": capped_size}
            if page_token:
                params["pageToken"] = page_token
            resp = self._client().get(
                f"{self.api_base_url}/mediaItems",
                headers=self._headers(),
                params=params,
                timeout=self.timeout_sec,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"Picker mediaItems.list failed ({resp.status_code}): {resp.text[:400]}")
            data = resp.json()
            collected.extend(list(data.get("mediaItems") or []))
            page_token = data.get("nextPageToken") or None
            if not page_token:
                break
        return collected

    def delete_session(self, session_id: str) -> bool:
        try:
            resp = self._client().delete(
                f"{self.api_base_url}/sessions/{session_id}",
                headers=self._headers(),
                timeout=self.timeout_sec,
            )
            return resp.status_code < 400
        except Exception:
            return False

    def download_media_file(
        self,
        base_url: str,
        dest_path: Path,
        modifier: str,
        max_retries: int = 3,
    ) -> Path:
        clean_base = re.sub(r"=(?:d|dv|w\d+[^/?]*)$", "", str(base_url).strip())
        dl_url = f"{clean_base}={modifier}" if modifier else clean_base
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dest_path.with_suffix(dest_path.suffix + ".part")

        last_exc: Optional[Exception] = None
        for attempt in range(max(1, int(max_retries))):
            try:
                with self._client().stream(
                    "GET",
                    dl_url,
                    headers=self._headers(),
                    timeout=max(self.timeout_sec, 120.0),
                ) as resp:
                    if resp.status_code >= 400:
                        raise RuntimeError(f"HTTP {resp.status_code} downloading {dl_url}")
                    with part_path.open("wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                            if chunk:
                                fh.write(chunk)
                if part_path.exists() and part_path.stat().st_size > 0:
                    os.replace(part_path, dest_path)
                    return dest_path
                raise RuntimeError(f"Downloaded empty payload from {dl_url}")
            except Exception as exc:
                last_exc = exc
                if part_path.exists():
                    try:
                        part_path.unlink()
                    except Exception:
                        pass
                if attempt + 1 < max_retries:
                    time.sleep(0.25 * (2**attempt))
        raise RuntimeError(f"Failed downloading cloud asset after {max_retries} attempts: {last_exc}")


def _guess_extension(filename: str, mime_type: str, media_type: str) -> str:
    raw_ext = Path(filename or "").suffix
    if raw_ext.lower() in PHOTO_EXTENSIONS or raw_ext.lower() in VIDEO_EXTENSIONS:
        return raw_ext
    mime_ext = _MIME_TO_EXT.get((mime_type or "").lower().strip())
    if mime_ext:
        return mime_ext
    return ".mp4" if media_type == "video" else ".jpg"


def _safe_filename(raw_name: str, fallback_stem: str, mime_type: str, media_type: str) -> str:
    ext = _guess_extension(raw_name, mime_type, media_type)
    stem = Path(raw_name or "").stem or fallback_stem
    clean_stem = re.sub(r"[^a-zA-Z0-9._-]", "_", stem).strip("._-") or fallback_stem
    return f"{clean_stem[:80]}{ext}"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two WGS84 coordinates."""
    r_earth = 6371.0088
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    return r_earth * 2.0 * math.atan2(math.sqrt(max(0.0, a)), math.sqrt(max(0.0, 1.0 - a)))


def _infer_orientation(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unknown"
    if width > height * 1.06:
        return "landscape"
    if height > width * 1.06:
        return "portrait"
    return "square"


_DEDICATED_CAM_RE = re.compile(
    r"(sony|ilce|a7c|a7r|a7s|a7m|a6\d{3}|zv-?e|canon|eos|nikon|fujifilm|fuji|x-t\d|x100|leica|olympus|om-?\d|panasonic|lumix|ricoh|hasselblad|pentax|sigma|phase\s*one)",
    re.IGNORECASE,
)
_SCREENSHOT_NAME_RE = re.compile(
    r"(^|[\W_])(screenshot|screen[\s_-]?shot|screen[\s_-]?capture|screencap|screen[\s_-]?recording|whatsapp[\s_-]*image|received[\s_-]*\d+)([\W_]|$)",
    re.IGNORECASE,
)


def _is_dedicated_camera(camera_make_model: Optional[str], filename: str) -> bool:
    cam = (camera_make_model or "").strip()
    if cam and _DEDICATED_CAM_RE.search(cam):
        return True
    stem_up = Path(filename or "").stem.upper()
    return stem_up.startswith(("_DSC", "DSC0", "DSC_", "A7C", "DSCF", "CR3_", "NEF_"))


def _detect_screenshot_or_junk(item: CloudMediaCandidateItem, thumb_path: Optional[Path]) -> Optional[str]:
    """
    Detect phone screenshots, screen recordings & flat-UI junk while preserving starred items
    and camera-captured photographs (including photographed signs, museum plaques, and menus).
    """
    if item.is_favorite or item.media_type != "photo":
        return None
    has_camera_exif = bool((item.camera_make_model or "").strip() or (item.exif_summary or "").strip())
    fname_low = Path(item.filename or "").name.lower()
    if not has_camera_exif and _SCREENSHOT_NAME_RE.search(fname_low):
        return "screenshot_or_junk"

    # Camera-captured JPEGs/HEICs (even of signs/menus) are never discarded as screenshots
    ext = Path(item.filename or "").suffix.lower()
    if has_camera_exif or ext in (".heic", ".heif", ".dng", ".arw", ".cr3", ".nef"):
        return None

    # PNG / WebP images lacking camera EXIF: check for flat-UI / phone-screen concentration
    if ext in (".png", ".webp") and thumb_path and thumb_path.is_file():
        try:
            with Image.open(thumb_path) as im:
                small = im.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
                arr = np.asarray(small, dtype=np.uint8)
            quant = (arr >> 4).reshape(-1, 3)
            keys = (quant[:, 0].astype(np.int32) << 8) | (quant[:, 1].astype(np.int32) << 4) | quant[:, 2].astype(np.int32)
            counts = np.bincount(keys)
            top4_frac = float(np.sort(counts)[-4:].sum()) / float(max(1, keys.size))
            aspect = max(item.width, item.height) / float(max(1, min(item.width, item.height))) if (item.width > 0 and item.height > 0) else 1.0
            if top4_frac >= 0.58 and (aspect >= 1.72 or top4_frac >= 0.72):
                return "screenshot_or_junk"
        except Exception:
            pass
    return None


def _dedupe_and_filter_videos(
    candidates: list[CloudMediaCandidateItem],
    parsed_times: dict[int, Optional[datetime]],
) -> None:
    """Suppress Apple/Android Live Photo companion micro-.MOV clips and duplicate re-uploaded videos."""
    photo_stems: set[str] = set()
    photo_times: list[tuple[datetime, str]] = []
    for idx, it in enumerate(candidates):
        if it.media_type == "photo":
            photo_stems.add(Path(it.filename).stem.lower())
            pt = parsed_times.get(idx)
            if pt is not None:
                photo_times.append((pt, it.dhash))

    eligible_vid_indices: list[int] = []
    for idx, it in enumerate(candidates):
        if it.media_type != "video" or not it.selected_for_download:
            continue
        if not it.is_favorite:
            ext = Path(it.filename).suffix.lower()
            is_mov = ext == ".mov" or it.mime_type == "video/quicktime"
            stem_low = Path(it.filename).stem.lower()
            if is_mov and 0.0 < float(it.duration_sec or 0.0) <= 3.2:
                matched_photo = stem_low in photo_stems
                if not matched_photo and parsed_times.get(idx) is not None:
                    vt = parsed_times[idx]
                    assert vt is not None
                    for pt, p_hash in photo_times:
                        if abs((vt - pt).total_seconds()) <= 2.0 and (
                            not it.dhash or not p_hash or _hamming_distance_hex(it.dhash, p_hash) <= 18
                        ):
                            matched_photo = True
                            break
                if matched_photo:
                    it.selected_for_download = False
                    it.skip_reason = "live_photo_companion"
                    continue
        eligible_vid_indices.append(idx)

    # Deduplicate exact/re-uploaded videos (e.g. A7C2_20251222_0618.MP4 vs A7C2_20251222_0618_1.MP4)
    n_v = len(eligible_vid_indices)
    if n_v < 2:
        return
    parent = list(range(n_v))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    def _base_stem(fname: str) -> str:
        s = Path(fname).stem.lower()
        return re.sub(r"(?:_\d+|\(\d+\))$", "", s)

    for a in range(n_v):
        ia = eligible_vid_indices[a]
        va = candidates[ia]
        ta = parsed_times.get(ia)
        if ta is None:
            continue
        ba = _base_stem(va.filename)
        for b in range(a + 1, n_v):
            ib = eligible_vid_indices[b]
            vb = candidates[ib]
            tb = parsed_times.get(ib)
            if tb is None or abs((tb - ta).total_seconds()) > 2.0:
                continue
            same_hash = bool(va.dhash and vb.dhash and _hamming_distance_hex(va.dhash, vb.dhash) <= 6)
            same_stem = bool(ba and ba == _base_stem(vb.filename))
            if same_hash or same_stem:
                _union(a, b)

    groups: dict[int, list[int]] = {}
    for pos, idx_v in enumerate(eligible_vid_indices):
        groups.setdefault(_find(pos), []).append(idx_v)

    for members in groups.values():
        if len(members) <= 1:
            continue
        winner = max(
            members,
            key=lambda idx_m: (
                candidates[idx_m].is_favorite,
                int(candidates[idx_m].width) * int(candidates[idx_m].height),
                candidates[idx_m].taste_score,
                0 if re.search(r"(?:_\d+|\(\d+\))$", Path(candidates[idx_m].filename).stem) else 1,
                -idx_m,
            ),
        )
        for m_idx in members:
            if m_idx != winner:
                candidates[m_idx].selected_for_download = False
                candidates[m_idx].skip_reason = "duplicate_video"


def _inherit_dual_camera_gps(
    candidates: list[CloudMediaCandidateItem],
    photo_indices: list[int],
    parsed_times: dict[int, Optional[datetime]],
    clusters: dict[int, list[int]],
) -> None:
    """Hydrate missing GPS on dedicated-camera frames from nearby smartphone shots (same cluster or <=180s)."""
    # Pass A: within visual/cross-camera burst clusters
    for members in clusters.values():
        gps_members = [m for m in members if candidates[m].gps_lat is not None and candidates[m].gps_lon is not None]
        if not gps_members:
            continue
        for m in members:
            if candidates[m].gps_lat is not None and candidates[m].gps_lon is not None:
                continue
            tm = parsed_times.get(m)
            donor_idx = min(
                gps_members,
                key=lambda g: abs((parsed_times[g] - tm).total_seconds()) if (tm and parsed_times.get(g)) else 0.0,
            )
            candidates[m].gps_lat = candidates[donor_idx].gps_lat
            candidates[m].gps_lon = candidates[donor_idx].gps_lon
            candidates[m].gps_alt = candidates[donor_idx].gps_alt

    # Pass B: temporal proximity (<= 180s) across all GPS-tagged items (photos + videos)
    gps_donors: list[tuple[datetime, CloudMediaCandidateItem]] = []
    for idx, it in enumerate(candidates):
        if it.gps_lat is not None and it.gps_lon is not None:
            ts = parsed_times.get(idx) or parse_iso_utc(it.creation_time)
            if ts is not None:
                gps_donors.append((ts, it))
    if not gps_donors:
        return
    for idx in photo_indices:
        it = candidates[idx]
        if it.gps_lat is not None and it.gps_lon is not None:
            continue
        tm = parsed_times.get(idx)
        if tm is None:
            continue
        best_dt = 181.0
        best_donor: Optional[CloudMediaCandidateItem] = None
        for d_ts, d_it in gps_donors:
            dt_s = abs((d_ts - tm).total_seconds())
            if dt_s < best_dt:
                best_dt = dt_s
                best_donor = d_it
        if best_donor is not None and best_dt <= 180.0:
            it.gps_lat = best_donor.gps_lat
            it.gps_lon = best_donor.gps_lon
            it.gps_alt = best_donor.gps_alt


def _segment_event_clusters(
    items: list[tuple[int, CloudMediaCandidateItem]],
    parsed_times: dict[int, Optional[datetime]],
) -> dict[int, str]:
    """Group chronologically ordered items into geographic/temporal event clusters (dt > 25m or GPS > 0.5km)."""
    if not items:
        return {}
    ordered = sorted(
        items,
        key=lambda pair: (pair[1].creation_time or "9999-12-31T23:59:59Z", pair[1].filename, pair[0]),
    )
    event_map: dict[int, str] = {}
    ev_num = 1
    prev_it: Optional[CloudMediaCandidateItem] = None
    prev_ts: Optional[datetime] = None
    for idx, it in ordered:
        cur_ts = parsed_times.get(idx) or parse_iso_utc(it.creation_time)
        if prev_it is not None:
            time_break = bool(cur_ts is not None and prev_ts is not None and abs((cur_ts - prev_ts).total_seconds()) > 1500.0)
            gps_break = bool(
                it.gps_lat is not None
                and it.gps_lon is not None
                and prev_it.gps_lat is not None
                and prev_it.gps_lon is not None
                and _haversine_km(prev_it.gps_lat, prev_it.gps_lon, it.gps_lat, it.gps_lon) > 0.5
            )
            if time_break or gps_break:
                ev_num += 1
        ev_id = f"event_{ev_num:02d}"
        it.event_cluster_id = ev_id
        event_map[idx] = ev_id
        prev_it = it
        prev_ts = cur_ts
    return event_map


def _select_video_editing_mmr_portfolio(
    items: list[tuple[int, CloudMediaCandidateItem]],
    budget: int,
    sig_map: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]],
    event_map: dict[int, str],
) -> set[int]:
    """
    Event-Clustered Submodular MMR selection optimized for 16:9 video editing:
    - Guarantees 100% of Favorited items (up to budget)
    - Guarantees >= 2 diverse shots per event stop (preferring a 16:9 landscape establishing shot per stop)
    - Maximizes aesthetic taste + visual/category/event novelty + vertical portrait diptych pair completion
    """
    if len(items) <= budget:
        for _, it in items:
            it.mmr_score = round(min(1.0, it.taste_score / 10.0), 4)
        return {idx for idx, _ in items}
    if budget <= 0:
        return set()

    ordered = sorted(
        items,
        key=lambda pair: (pair[1].creation_time or "9999-12-31T23:59:59Z", pair[1].filename, pair[0]),
    )
    by_idx = {idx: it for idx, it in ordered}

    selected: set[int] = set()
    selected_sigs: list[np.ndarray] = []
    cat_counts: dict[str, int] = {}
    ev_counts: dict[str, int] = {}
    portrait_by_ev: dict[str, int] = {}
    landscape_total = 0

    def _record_pick(pick_idx: int, gain: float) -> None:
        nonlocal landscape_total
        it_p = by_idx[pick_idx]
        selected.add(pick_idx)
        it_p.mmr_score = round(float(gain), 4)
        if pick_idx in sig_map:
            selected_sigs.append(sig_map[pick_idx][0])
        cat_k = it_p.subject_category or "general"
        cat_counts[cat_k] = cat_counts.get(cat_k, 0) + 1
        ev_k = event_map.get(pick_idx, "event_01")
        ev_counts[ev_k] = ev_counts.get(ev_k, 0) + 1
        if it_p.orientation == "portrait":
            portrait_by_ev[ev_k] = portrait_by_ev.get(ev_k, 0) + 1
        elif it_p.orientation == "landscape":
            landscape_total += 1

    def _mmr_gain(cand_idx: int, remaining_slots: int) -> float:
        it_c = by_idx[cand_idx]
        ev_c = event_map.get(cand_idx, "event_01")
        cat_c = it_c.subject_category or "general"
        q_norm = min(1.15, float(it_c.taste_score) / 10.0)
        if selected_sigs and cand_idx in sig_map:
            sig_c = sig_map[cand_idx][0]
            max_sim = max(float(np.dot(sig_c, s_sig)) for s_sig in selected_sigs)
        else:
            max_sim = 0.0
        cat_nov = 1.0 / (1.0 + cat_counts.get(cat_c, 0))
        ev_nov = 1.0 / (1.0 + ev_counts.get(ev_c, 0))

        # Orientation balance for 16:9 timelines: encourage landscape anchors + complete vertical diptych pairs
        orient_bonus = 0.0
        n_sel = len(selected)
        p_in_ev = portrait_by_ev.get(ev_c, 0)
        if it_c.orientation == "portrait":
            if p_in_ev % 2 == 1:
                orient_bonus = 0.11  # Complete vertical diptych pair in same event!
            elif remaining_slots == 1 and p_in_ev == 0:
                orient_bonus = -0.05  # Avoid stranding a single lone vertical on the last slot
        elif it_c.orientation == "landscape":
            land_frac = landscape_total / float(max(1, n_sel))
            if n_sel == 0 or land_frac < 0.45:
                orient_bonus = 0.08

        return (
            0.46 * q_norm
            - 0.28 * max(0.0, max_sim)
            + 0.09 * cat_nov
            + 0.08 * ev_nov
            + orient_bonus
            + (0.35 if it_c.is_favorite else 0.0)
        )

    # 1. Always preserve Favorited items first up to budget
    fav_ordered = sorted(
        [(idx, it) for idx, it in ordered if it.is_favorite],
        key=lambda pair: (-pair[1].taste_score, pair[0]),
    )
    for f_idx, f_it in fav_ordered[:budget]:
        _record_pick(f_idx, min(1.2, f_it.taste_score / 10.0 + 0.35))
    if len(selected) >= budget:
        return selected

    # 2. Per-event minimum floor (preferring 1 landscape establishing anchor + 1 complementary shot per stop)
    by_event: dict[str, list[int]] = {}
    for idx, _ in ordered:
        by_event.setdefault(event_map.get(idx, "event_01"), []).append(idx)

    n_events = max(1, len(by_event))
    floor_per_event = 2 if budget >= n_events * 2 else 1
    for pass_round in range(floor_per_event):
        for ev_id in sorted(by_event):
            if len(selected) >= budget:
                break
            if ev_counts.get(ev_id, 0) > pass_round:
                continue
            avail = [i for i in by_event[ev_id] if i not in selected]
            if not avail:
                continue
            rem = budget - len(selected)
            if pass_round == 0:
                # Prefer a strong landscape establishing frame for the stop's opener if available
                lands = [i for i in avail if by_idx[i].orientation == "landscape"]
                pool_ev = lands if lands else avail
            else:
                pool_ev = avail
            best_i = max(pool_ev, key=lambda i: (_mmr_gain(i, rem), -i))
            _record_pick(best_i, _mmr_gain(best_i, rem))

    # 3. Fill remaining budget via global Submodular MMR
    while len(selected) < budget:
        rem = budget - len(selected)
        best_i: Optional[int] = None
        best_g = -1e9
        for idx, _ in ordered:
            if idx in selected:
                continue
            g = _mmr_gain(idx, rem)
            if g > best_g:
                best_g = g
                best_i = idx
        if best_i is None:
            break
        _record_pick(best_i, best_g)

    return selected


def _stratified_budget_indices(
    items: list[tuple[int, CloudMediaCandidateItem]],
    budget: int,
) -> set[int]:
    """Select up to `budget` indices preserving chronological coverage across day/time buckets."""
    if len(items) <= budget:
        return {idx for idx, _ in items}
    if budget <= 0:
        return set()

    ordered = sorted(
        items,
        key=lambda pair: (pair[1].creation_time or "9999-12-31T23:59:59Z", pair[1].filename, pair[0]),
    )
    fav_ordered = sorted(
        [(idx, it) for idx, it in ordered if it.is_favorite],
        key=lambda pair: (-pair[1].taste_score, pair[0]),
    )
    selected: set[int] = {idx for idx, _ in fav_ordered[:budget]}
    if len(selected) >= budget:
        return selected

    n = len(ordered)
    buckets: list[list[tuple[int, CloudMediaCandidateItem]]] = []
    for b in range(budget):
        lo = (b * n) // budget
        hi = max(lo + 1, ((b + 1) * n) // budget)
        buckets.append(ordered[lo:hi])

    for bucket in buckets:
        if len(selected) >= budget:
            break
        if any(idx in selected for idx, _ in bucket):
            continue
        avail = [pair for pair in bucket if pair[0] not in selected]
        if avail:
            best_idx, _ = max(avail, key=lambda pair: (pair[1].taste_score, -pair[0]))
            selected.add(best_idx)

    if len(selected) < budget:
        remaining = sorted(
            [pair for pair in ordered if pair[0] not in selected],
            key=lambda pair: (pair[1].taste_score, -pair[0]),
            reverse=True,
        )
        for idx, _ in remaining:
            selected.add(idx)
            if len(selected) >= budget:
                break
    return selected


def select_necessary_cloud_items(
    candidates: list[CloudMediaCandidateItem],
    target_duration_sec: float = 155.0,
    max_photos: Optional[int] = None,
    max_videos: Optional[int] = None,
    download_all: bool = False,
    headroom_multiplier: float = 1.6,
) -> list[CloudMediaCandidateItem]:
    """
    Parallel Video-Editing-Aware Pass-1 Smart Cloud Curation:
    - Parallel perception & taste evaluation (eye-open, smile, motion blur, saliency, Lab color)
    - Rejects phone screenshots/blank receipts (preserving camera-captured signs/menus & ★ Favorites)
    - Suppresses Live Photo companion micro-.MOVs & duplicate re-uploaded videos
    - Non-adjacent Union-Find burst & cross-camera dedup + smartphone-to-camera GPS inheritance
    - Event-clustered Submodular MMR with 1.6x recall headroom & 16:9 / vertical diptych pair balance
    """
    if not candidates:
        return []

    dur = max(30.0, min(600.0, float(target_duration_sec or 155.0)))
    h_mult = max(1.0, min(3.0, float(headroom_multiplier or 1.6)))
    photo_budget = (
        max(1, int(max_photos))
        if max_photos is not None
        else max(24, min(120, int(round(dur * 0.26 * h_mult))))
    )
    video_budget = (
        max(1, int(max_videos))
        if max_videos is not None
        else max(12, min(60, int(round(dur * 0.20))))
    )

    sig_map: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = {}
    eye_smile_map: dict[int, tuple[float, float]] = {}

    def _score_one(pair: tuple[int, CloudMediaCandidateItem]) -> tuple[
        int,
        float,
        str,
        str,
        str,
        Optional[str],
        Optional[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]],
        tuple[float, float],
    ]:
        idx_c, item = pair
        orient = _infer_orientation(item.width, item.height)
        thumb_p = Path(item.proxy_thumbnail_path) if item.proxy_thumbnail_path else None
        if orient == "unknown" and thumb_p and thumb_p.is_file():
            try:
                with Image.open(thumb_p) as im_sz:
                    tw_s, th_s = im_sz.size
                orient = _infer_orientation(tw_s, th_s)
            except Exception:
                pass

        junk_reason = _detect_screenshot_or_junk(item, thumb_p)
        if thumb_p and thumb_p.is_file():
            metrics = _evaluate_photo_taste_metrics(thumb_p, item.filename)
            perc = analyze_human_and_aesthetic_perception(thumb_p, item.filename)
            dhash_val = str(perc.get("dhash") or metrics.get("dhash") or "")
            eye_s = float(perc.get("eye_open_score", 1.0))
            smile_s = float(perc.get("smile_score", 0.0))
            blur_t = str(perc.get("blur_type") or "crisp_all")
            sub_cat = "screenshot_or_doc" if junk_reason else str(perc.get("inferred_category") or metrics.get("inferred_category") or "")
            taste = float(metrics.get("overall_taste_score", 7.5))
            if eye_s < 0.4:
                taste -= 1.8
            elif smile_s > 0.35:
                taste += 0.45
            if blur_t in ("global_motion_blur", "motion_or_defocus_blur"):
                taste -= 1.2
            if item.media_type == "video":
                res_bonus = 0.35 if min(item.width, item.height) >= 1080 else 0.0
                fps_bonus = 0.2 if item.fps >= 29.0 else 0.0
                taste += res_bonus + fps_bonus
            if item.is_favorite:
                taste += 1.25
            taste_clamped = round(max(1.0, min(9.99, taste)), 2)
            lab_mean = np.asarray(perc.get("lab_mean", [128.0, 128.0, 128.0]), dtype=np.float32)
            sig_t = (
                perc["visual_signature"],
                perc["center_signature"],
                perc["inner_signature"],
                eye_s,
                lab_mean,
            )
            return idx_c, taste_clamped, dhash_val, orient, sub_cat, junk_reason, sig_t, (eye_s, smile_s)

        base_t = 7.6 if item.media_type == "video" else 7.2
        taste_clamped = round(min(9.99, base_t + (1.25 if item.is_favorite else 0.0)), 2)
        sub_cat = "screenshot_or_doc" if junk_reason else ("video_clip" if item.media_type == "video" else "")
        return idx_c, taste_clamped, item.dhash or "", orient, sub_cat, junk_reason, None, (1.0, 0.0)

    workers = max(1, min(get_io_worker_count(12), len(candidates)))
    indexed_cands = list(enumerate(candidates))
    if workers == 1 or len(candidates) <= 3:
        scored_rows = [_score_one(p) for p in indexed_cands]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scored_rows = list(pool.map(_score_one, indexed_cands))

    for idx_c, taste_val, dhash_val, orient, sub_cat, junk_reason, sig_t, es_pair in scored_rows:
        item = candidates[idx_c]
        item.selected_for_download = True
        item.skip_reason = None
        item.burst_group_id = None
        item.burst_cluster_size = 1
        item.mmr_score = 0.0
        item.taste_score = taste_val
        item.dhash = dhash_val
        item.orientation = orient  # type: ignore[assignment]
        item.subject_category = sub_cat
        eye_smile_map[idx_c] = es_pair
        if sig_t is not None:
            sig_map[idx_c] = sig_t

        if item.media_type == "video" and item.video_processing_status not in ("", "READY", "UNSPECIFIED"):
            item.selected_for_download = False
            item.skip_reason = "video_processing_not_ready"
        elif junk_reason and not download_all:
            item.selected_for_download = False
            item.skip_reason = junk_reason

    parsed_times: dict[int, Optional[datetime]] = {
        i: parse_iso_utc(candidates[i].creation_time) for i in range(len(candidates))
    }

    if download_all:
        _segment_event_clusters(list(enumerate(candidates)), parsed_times)
        return candidates

    # Video Live-Photo companion suppression & duplicate video deduplication
    _dedupe_and_filter_videos(candidates, parsed_times)

    # Seed-Anchored Anti-Chaining Photo Burst & Cross-Camera Near-Duplicate Clustering
    # (Anchored to each cluster's seed frame so similar warm/gold museum rooms can NEVER
    # transitively chain distinct paintings or galleries into a single mega-cluster!)
    photo_indices = sorted(
        [i for i, it in enumerate(candidates) if it.media_type == "photo" and it.selected_for_download],
        key=lambda i: (candidates[i].creation_time or "9999", candidates[i].filename, i),
    )

    def _cam_bucket(idx_c: int) -> str:
        c = candidates[idx_c]
        return (c.camera_make_model or ("dedicated" if _is_dedicated_camera(c.camera_make_model, c.filename) else "generic")).strip().lower()

    by_burst: dict[int, list[int]] = {}
    cluster_seeds: list[tuple[int, int]] = []  # (cluster_id, seed_cand_idx)

    for idx_p in photo_indices:
        t_cur = parsed_times[idx_p]
        h_cur = candidates[idx_p].dhash
        sig_cur = sig_map.get(idx_p)
        cam_cur = _cam_bucket(idx_p)

        matched_cid: Optional[int] = None
        best_match_score = -1.0

        # Compare directly against active cluster seeds (prevents single-linkage transitive chaining!)
        for cid, seed_idx in reversed(cluster_seeds):
            t_seed = parsed_times[seed_idx]
            dt_sec: Optional[float] = None
            if t_cur is not None and t_seed is not None:
                dt_sec = abs((t_cur - t_seed).total_seconds())
                # Stop checking old seeds (> 10 min) unless exact dHash check on recent 25 clusters
                if dt_sec > 600.0 and (len(cluster_seeds) - cid) > 25:
                    continue

            h_seed = candidates[seed_idx].dhash
            h_dist = _hamming_distance_hex(h_cur, h_seed) if (h_cur and h_seed) else 64
            sig_seed = sig_map.get(seed_idx)
            lab_h = cie_lab_harmony_score(sig_cur[4], sig_seed[4]) if (sig_cur is not None and sig_seed is not None) else 0.75

            is_dup = False
            match_strength = 0.0
            within_15s = dt_sec is None or dt_sec <= 15.0
            within_7m = dt_sec is None or dt_sec <= 420.0
            diff_cam = cam_cur != _cam_bucket(seed_idx)

            # Rule 1: Near-identical duplicate regardless of time gap
            if h_dist <= 2 and lab_h >= 0.85:
                is_dup = True
                match_strength = 2.0 - h_dist * 0.1
            # Rule 2: Rapid burst sequence (<= 15s from cluster seed)
            elif within_15s and h_dist <= 10 and lab_h >= 0.65:
                is_dup = True
                match_strength = 1.6 - h_dist * 0.05
            elif within_15s and h_dist <= 16 and lab_h >= 0.65 and sig_cur is not None and sig_seed is not None:
                z_sim = compute_zoom_tolerant_visual_similarity(
                    sig_cur[0], sig_cur[1], sig_cur[2], sig_seed[0], sig_seed[1], sig_seed[2]
                )
                if z_sim >= 0.94:
                    is_dup = True
                    match_strength = z_sim
            # Rule 3: Non-adjacent re-frame or dual-camera (Sony + Pixel) match within 7 minutes
            elif within_7m and h_dist <= 6 and lab_h >= 0.78:
                is_dup = True
                match_strength = 1.4 - h_dist * 0.05
            elif within_7m and (diff_cam or (dt_sec is not None and dt_sec <= 60.0)) and h_dist <= 12 and lab_h >= 0.72 and sig_cur is not None and sig_seed is not None:
                z_sim = compute_zoom_tolerant_visual_similarity(
                    sig_cur[0], sig_cur[1], sig_cur[2], sig_seed[0], sig_seed[1], sig_seed[2]
                )
                if z_sim >= 0.955:
                    is_dup = True
                    match_strength = z_sim

            if is_dup and match_strength > best_match_score:
                best_match_score = match_strength
                matched_cid = cid

        if matched_cid is None:
            new_cid = len(cluster_seeds) + 1
            cluster_seeds.append((new_cid, idx_p))
            by_burst[new_cid] = [idx_p]
            candidates[idx_p].burst_group_id = f"burst_{new_cid:03d}"
        else:
            by_burst[matched_cid].append(idx_p)
            candidates[idx_p].burst_group_id = f"burst_{matched_cid:03d}"

    # Inherit smartphone GPS onto dedicated-camera frames before discarding burst losers
    _inherit_dual_camera_gps(candidates, photo_indices, parsed_times, by_burst)

    kept_photos: list[tuple[int, CloudMediaCandidateItem]] = []
    for cid in sorted(by_burst):
        members = by_burst[cid]
        best_i = max(
            members,
            key=lambda idx_m: (
                candidates[idx_m].is_favorite,
                round(
                    candidates[idx_m].taste_score
                    + (0.45 if _is_dedicated_camera(candidates[idx_m].camera_make_model, candidates[idx_m].filename) else 0.0),
                    2,
                ),
                eye_smile_map.get(idx_m, (1.0, 0.0))[0],
                eye_smile_map.get(idx_m, (1.0, 0.0))[1],
                int(candidates[idx_m].width) * int(candidates[idx_m].height),
                -idx_m,
            ),
        )
        candidates[best_i].burst_cluster_size = len(members)
        kept_photos.append((best_i, candidates[best_i]))
        for m_idx in members:
            if m_idx != best_i:
                candidates[m_idx].burst_cluster_size = len(members)
                candidates[m_idx].selected_for_download = False
                candidates[m_idx].skip_reason = "burst_duplicate"

    # Drop obvious unrecoverable low-score accidental blur shots when >=4 stronger photos exist
    if len(kept_photos) > 4:
        strong_count = sum(1 for _, it_p in kept_photos if it_p.taste_score >= 5.0)
        if strong_count >= 4:
            filtered_photos: list[tuple[int, CloudMediaCandidateItem]] = []
            for idx_p, it_p in kept_photos:
                if it_p.taste_score < 5.0 and not it_p.is_favorite:
                    it_p.selected_for_download = False
                    it_p.skip_reason = "low_taste_blur"
                else:
                    filtered_photos.append((idx_p, it_p))
            kept_photos = filtered_photos or kept_photos

    # Segment all surviving media into temporal/GPS Event Clusters
    eligible_videos = [
        (i, it) for i, it in enumerate(candidates) if it.media_type == "video" and it.selected_for_download
    ]
    event_map = _segment_event_clusters(kept_photos + eligible_videos, parsed_times)

    # Event-Clustered Submodular MMR portfolio selection for photos
    if len(kept_photos) > photo_budget:
        winner_ids = _select_video_editing_mmr_portfolio(kept_photos, photo_budget, sig_map, event_map)
        for idx_p, it_p in kept_photos:
            if idx_p not in winner_ids:
                it_p.selected_for_download = False
                it_p.skip_reason = "over_photo_budget"
    else:
        for _, it_p in kept_photos:
            it_p.mmr_score = round(min(1.0, it_p.taste_score / 10.0), 4)

    # Event-Clustered Submodular MMR portfolio selection for videos
    if len(eligible_videos) > video_budget:
        winner_vids = _select_video_editing_mmr_portfolio(eligible_videos, video_budget, sig_map, event_map)
        for idx_v, it_v in eligible_videos:
            if idx_v not in winner_vids:
                it_v.selected_for_download = False
                it_v.skip_reason = "over_video_budget"

    return candidates


def render_cloud_scan_contact_sheets(
    items: list[CloudMediaCandidateItem],
    output_dir: str | Path,
    title: str = "Google Photos Smart Cloud Scan",
    tiles_per_page: int = 20,
    include_burst_duplicates: bool = False,
) -> tuple[list[Path], Path]:
    """
    Render paginated 5x4 (20 tiles/page) cloud scan contact sheets covering ALL unique/deduped candidates
    (zero 40-tile truncation!), plus `cloud_scan_tile_map.json` for Pass-1.5 AI Vision rescues.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_p = out_dir / "cloud_scan_contact_sheet.jpg"

    if include_burst_duplicates or len(items) <= tiles_per_page:
        display_items = list(items)
    else:
        display_items = [it for it in items if it.skip_reason != "burst_duplicate"]
        if not display_items:
            display_items = list(items)

    # Sort chronologically for natural narrative scanning
    display_items.sort(key=lambda it: (it.creation_time or "9999", it.filename, it.cloud_item_id))

    per_page = max(4, int(tiles_per_page))
    cols = 5 if per_page >= 15 else 4
    total_pages = max(1, (max(1, len(display_items)) + per_page - 1) // per_page)
    sel_cnt = sum(1 for it in items if it.selected_for_download)
    burst_collapsed = sum(1 for it in items if it.skip_reason == "burst_duplicate")
    skip_cnt = len(items) - sel_cnt

    f_hdr = _get_cjk_font(22)
    f_sub = _get_cjk_font(14)
    f_badge = _get_cjk_font(12)

    tile_w, tile_h = 360, 245
    margin = 16
    header_h = 72
    sheet_paths: list[Path] = []
    tile_map: dict[str, dict[str, Any]] = {}

    for page_idx in range(total_pages):
        chunk = display_items[page_idx * per_page : (page_idx + 1) * per_page]
        rows = max(1, (max(1, len(chunk)) + cols - 1) // cols)
        canvas_w = cols * tile_w + (cols + 1) * margin
        canvas_h = header_h + rows * tile_h + (rows + 1) * margin

        canvas = Image.new("RGB", (canvas_w, canvas_h), (14, 18, 26))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, canvas_w, header_h - 4], fill=(20, 27, 39))
        draw.text((margin, 12), f"{title}  ·  Page {page_idx + 1}/{total_pages}", font=f_hdr, fill=(242, 246, 252))
        draw.text(
            (margin, 44),
            f"Scanned: {len(items)}  |  Selected: {sel_cnt}  |  Skipped: {skip_cnt} ({burst_collapsed} burst dups collapsed)  |  Showing unique heroes #{page_idx * per_page + 1}-{page_idx * per_page + len(chunk)}",
            font=f_sub,
            fill=(146, 206, 244),
        )

        for local_i, it in enumerate(chunk):
            global_num = page_idx * per_page + local_i + 1
            tile_id = f"#{global_num:02d}"
            tile_map[tile_id] = {
                "cloud_item_id": it.cloud_item_id,
                "filename": it.filename,
                "selected": it.selected_for_download,
                "skip_reason": it.skip_reason,
                "media_type": it.media_type,
                "event_cluster_id": it.event_cluster_id,
            }

            r, c = divmod(local_i, cols)
            x0 = margin + c * (tile_w + margin)
            y0 = header_h + margin + r * (tile_h + margin)
            is_sel = it.selected_for_download
            is_runner_up = (not is_sel) and (it.skip_reason in ("over_photo_budget", "over_video_budget"))
            border_col = (
                (250, 204, 21)
                if (is_sel and it.is_favorite)
                else ((54, 211, 153) if is_sel else ((96, 165, 250) if is_runner_up else (245, 158, 11)))
            )
            draw.rounded_rectangle([x0 - 2, y0 - 2, x0 + tile_w + 2, y0 + tile_h + 2], radius=8, outline=border_col, width=2)
            draw.rounded_rectangle([x0, y0, x0 + tile_w, y0 + tile_h], radius=6, fill=(24, 30, 42))

            img_h = tile_h - 54
            thumb_loaded = False
            for cand_p in (it.proxy_thumbnail_path, it.downloaded_path):
                if cand_p and Path(cand_p).is_file():
                    try:
                        with Image.open(cand_p) as im:
                            fitted = ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (tile_w - 8, img_h - 6))
                            if not is_sel:
                                fitted = Image.blend(fitted, Image.new("RGB", fitted.size, (18, 22, 30)), alpha=0.38 if is_runner_up else 0.52)
                            canvas.paste(fitted, (x0 + 4, y0 + 4))
                            thumb_loaded = True
                            break
                    except Exception:
                        continue
            if not thumb_loaded:
                draw.rectangle([x0 + 4, y0 + 4, x0 + tile_w - 4, y0 + img_h], fill=(32, 40, 56))
                draw.text((x0 + 16, y0 + img_h // 2 - 8), f"[{it.media_type.upper()}] {it.filename[:24]}", font=f_sub, fill=(190, 200, 216))

            skip_short = {
                "video_processing_not_ready": "processing",
                "over_photo_budget": "RUNNER-UP",
                "over_video_budget": "RUNNER-UP",
                "screenshot_or_junk": "SCREENSHOT",
                "live_photo_companion": "LIVE_MOV",
                "duplicate_video": "DUP_VID",
            }.get(it.skip_reason or "", (it.skip_reason or "filtered")[:14])
            fav_prefix = "★" if it.is_favorite else ""
            if is_sel:
                badge_txt = f"{tile_id} {fav_prefix}SEL"
                badge_bg = (16, 128, 86)
            elif is_runner_up:
                badge_txt = f"{tile_id} {fav_prefix}RUNNER-UP"
                badge_bg = (37, 99, 178)
            else:
                badge_txt = f"{tile_id} {fav_prefix}SKIP:{skip_short}"
                badge_bg = (146, 78, 12)
            bw = min(tile_w - 110, max(78, len(badge_txt) * 7 + 16))
            draw.rounded_rectangle([x0 + 8, y0 + 8, x0 + 8 + bw, y0 + 30], radius=5, fill=badge_bg)
            draw.text((x0 + 14, y0 + 12), badge_txt, font=f_badge, fill=(248, 250, 252))

            ori_tag = {"landscape": "16:9", "portrait": "9:16", "square": "1:1"}.get(it.orientation, "")
            kind_txt = f"{'VID' if it.media_type == 'video' else 'IMG'} {ori_tag} {it.taste_score:.1f}".replace("  ", " ").strip()
            kw = len(kind_txt) * 7 + 14
            draw.rounded_rectangle([x0 + tile_w - kw - 8, y0 + 8, x0 + tile_w - 8, y0 + 30], radius=5, fill=(15, 23, 38))
            draw.text((x0 + tile_w - kw - 2, y0 + 12), kind_txt, font=f_badge, fill=(196, 224, 248))

            if it.burst_cluster_size > 1:
                b_txt = f"+{it.burst_cluster_size - 1} burst"
                bbw = len(b_txt) * 7 + 12
                draw.rounded_rectangle([x0 + tile_w - bbw - 8, y0 + img_h - 26, x0 + tile_w - 8, y0 + img_h - 6], radius=4, fill=(15, 23, 38))
                draw.text((x0 + tile_w - bbw - 2, y0 + img_h - 23), b_txt, font=f_badge, fill=(250, 204, 21))

            ts_short = (it.creation_time or "")[:16].replace("T", " ")
            ev_short = (it.event_cluster_id or "").replace("event_", "E")
            cam_short = (it.camera_make_model or "")[:14]
            gps_mark = " GPS" if (it.gps_lat is not None and it.gps_lon is not None) else ""
            draw.text((x0 + 8, y0 + img_h + 7), f"{tile_id} {it.filename[:30]}", font=f_sub, fill=(236, 240, 246))
            draw.text((x0 + 8, y0 + img_h + 27), f"{ev_short} {ts_short} {cam_short}{gps_mark}".strip(), font=f_badge, fill=(148, 163, 184))

        page_p = out_dir / f"cloud_scan_sheet_{page_idx + 1:02d}.jpg"
        canvas.save(page_p, "JPEG", quality=90)
        sheet_paths.append(page_p.resolve())

    if sheet_paths:
        shutil.copy2(sheet_paths[0], legacy_p)
    (out_dir / "cloud_scan_tile_map.json").write_text(
        json.dumps(tile_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return sheet_paths, legacy_p.resolve()


def render_cloud_scan_contact_sheet(
    items: list[CloudMediaCandidateItem],
    output_jpg: str | Path,
    title: str = "Google Photos Smart Cloud Scan",
    max_tiles: int = 40,
) -> Path:
    """Render an inspection contact sheet contrasting SELECTED vs SKIPPED cloud candidates."""
    out_p = Path(output_jpg).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sheet_paths, legacy_p = render_cloud_scan_contact_sheets(
        items=items,
        output_dir=out_p.parent,
        title=title,
        tiles_per_page=max(20, min(40, int(max_tiles))),
        include_burst_duplicates=len(items) <= max_tiles,
    )
    if legacy_p != out_p and sheet_paths:
        shutil.copy2(sheet_paths[0], out_p)
    return out_p


def _stage_selected_items_into_cloud_media(
    items: list[CloudMediaCandidateItem],
    staged_dir: Path,
) -> Path:
    staged_dir = staged_dir.resolve()
    staged_dir.mkdir(parents=True, exist_ok=True)

    # Plan destination names BEFORE touching staged_dir so re-staging never unlinks an active source file
    planned_moves: list[tuple[CloudMediaCandidateItem, Path, Path]] = []
    keep_dst_resolved: set[Path] = set()
    used_names: set[str] = set()
    for idx, it in enumerate(items, start=1):
        if not it.selected_for_download or not it.downloaded_path:
            continue
        src = Path(it.downloaded_path).resolve()
        if not src.is_file():
            continue
        candidate_name = _safe_filename(it.filename, f"cloud_{idx:04d}", it.mime_type, it.media_type)
        if candidate_name.lower() in used_names:
            stem = Path(candidate_name).stem
            suffix = Path(candidate_name).suffix
            short_h = hashlib.sha1(it.cloud_item_id.encode("utf-8")).hexdigest()[:6]
            candidate_name = f"{stem}_{short_h}{suffix}"
        used_names.add(candidate_name.lower())
        dst = (staged_dir / candidate_name).resolve()
        keep_dst_resolved.add(dst)
        planned_moves.append((it, src, dst))

    for old_f in list(staged_dir.iterdir()):
        if (old_f.is_file() or old_f.is_symlink()) and old_f.resolve() not in keep_dst_resolved:
            old_f.unlink()

    for it, src, dst in planned_moves:
        if src == dst:
            it.downloaded_path = str(dst)
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
        it.downloaded_path = str(dst)
    return staged_dir


def _picker_item_to_candidate(raw: dict[str, Any]) -> CloudMediaCandidateItem:
    item_id = str(raw.get("id") or hashlib.sha1(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16])
    mf = raw.get("mediaFile") or {}
    meta = mf.get("mediaFileMetadata") or {}
    raw_type = str(raw.get("type") or "").upper()
    mime = str(mf.get("mimeType") or "").lower()
    if raw_type == "VIDEO" or mime.startswith("video/"):
        mtype = "video"
    else:
        mtype = "photo"
    filename = str(mf.get("filename") or f"{item_id}{'.mp4' if mtype == 'video' else '.jpg'}")
    make = str(meta.get("cameraMake") or "").strip()
    model = str(meta.get("cameraModel") or "").strip()
    cam = " ".join(p for p in (make, model) if p) or None
    vid_meta = meta.get("videoMetadata") or {}
    pho_meta = meta.get("photoMetadata") or {}
    fps = float(vid_meta.get("fps") or (30.0 if mtype == "video" else 0.0))
    proc_status = str(vid_meta.get("processingStatus") or "READY").upper()
    exif_parts: list[str] = []
    if pho_meta.get("focalLength"):
        exif_parts.append(f"{pho_meta['focalLength']}mm")
    if pho_meta.get("apertureFNumber"):
        exif_parts.append(f"f/{pho_meta['apertureFNumber']}")
    if pho_meta.get("iso"):
        exif_parts.append(f"ISO {pho_meta['iso']}")
    return CloudMediaCandidateItem(
        cloud_item_id=item_id,
        source_provider="google_photos_picker",
        filename=filename,
        mime_type=mime or ("video/mp4" if mtype == "video" else "image/jpeg"),
        media_type=mtype,
        creation_time=str(raw.get("createTime") or ""),
        width=int(meta.get("width") or 0),
        height=int(meta.get("height") or 0),
        fps=fps,
        video_processing_status=proc_status,
        camera_make_model=cam,
        exif_summary=" ".join(exif_parts) or None,
    )


def sync_google_photos_media(
    project_dir: str | Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tz_name: str = "UTC",
    target_duration_sec: float = 155.0,
    max_photos: Optional[int] = None,
    max_videos: Optional[int] = None,
    download_all: bool = False,
    headroom_multiplier: float = 1.6,
    plan_only: bool = False,
    session_id: Optional[str] = None,
    open_browser: bool = True,
    client_secrets_path: Optional[str | Path] = None,
    token_path: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    api_base_url: Optional[str] = None,
    picker_ready_callback: Optional[Callable[[str, str], None]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> GooglePhotosSyncManifest:
    """Connect to Google Photos Picker API, filter [date_from, date_to], run Two-Pass curation, and stage originals."""
    proj = Path(project_dir).resolve()
    proj.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_dir).resolve() if cache_dir else get_default_google_photos_cache_dir()
    thumbs_dir = cache_root / "thumbs"
    orig_dir = cache_root / "originals"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    orig_dir.mkdir(parents=True, exist_ok=True)

    auth = GooglePhotosAuthManager(client_secrets_path=client_secrets_path, token_path=token_path)
    client = GooglePhotosPickerClient(auth_manager=auth, api_base_url=api_base_url)

    created_here = False
    if not session_id:
        sess = client.create_session(max_item_count=2000)
        session_id = str(sess["id"])
        created_here = True
        picker_uri = build_autoclose_picker_uri(str(sess.get("pickerUri") or ""))
        search_hint = format_picker_date_query(date_from, date_to)
        if search_hint:
            try_copy_to_clipboard(search_hint)
        if picker_ready_callback is not None:
            picker_ready_callback(picker_uri, search_hint)
        elif open_browser and picker_uri:
            try:
                webbrowser.open(picker_uri, new=2)
            except Exception:
                pass
        if progress_callback:
            hint_msg = f" Paste '{search_hint}' into Google Photos search bar." if search_hint else ""
            progress_callback(f"Opened Picker session {session_id}.{hint_msg}")

    try:
        client.poll_until_ready(session_id, progress_callback=progress_callback)
        raw_items = client.list_session_media_items(session_id)
        in_range_raw = filter_items_by_date_range(raw_items, date_from, date_to, tz_name=tz_name)
        base_url_by_id = {
            str(it.get("id") or ""): str((it.get("mediaFile") or {}).get("baseUrl") or "")
            for it in in_range_raw
        }
        candidates = [_picker_item_to_candidate(it) for it in in_range_raw]

        # Pass 1: Pooled HTTP/2 16-worker 512px proxy thumbnail download for perceptual scoring & burst dedup
        def _fetch_thumb(cand: CloudMediaCandidateItem) -> None:
            b_url = base_url_by_id.get(cand.cloud_item_id, "")
            if not b_url:
                return
            h_id = hashlib.sha1(cand.cloud_item_id.encode("utf-8")).hexdigest()[:16]
            thumb_p = thumbs_dir / f"{h_id}.jpg"
            if not (thumb_p.is_file() and thumb_p.stat().st_size > 0):
                try:
                    client.download_media_file(b_url, thumb_p, modifier="w512-h512")
                except Exception:
                    return
            if thumb_p.is_file() and thumb_p.stat().st_size > 0:
                cand.proxy_thumbnail_path = str(thumb_p.resolve())

        if candidates:
            workers = max(1, min(get_io_worker_count(16), len(candidates)))
            if workers == 1:
                for c in candidates:
                    _fetch_thumb(c)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(_fetch_thumb, candidates))

        select_necessary_cloud_items(
            candidates,
            target_duration_sec=target_duration_sec,
            max_photos=max_photos,
            max_videos=max_videos,
            download_all=download_all,
            headroom_multiplier=headroom_multiplier,
        )

        # Render Pass-1.5 paginated 5x4 contact sheets covering ALL unique cluster heroes
        sheet_paths, legacy_sheet_p = render_cloud_scan_contact_sheets(
            candidates,
            proj / "contact_sheets",
            title=f"Google Photos Scan ({date_from or 'All'} -> {date_to or 'All'})",
        )

        # Persist session base URLs privately so Pass-1.5 apply_cloud_pick_overrides can execute Pass 2 cleanly
        picker_ctx_p = proj / "cache" / "picker_session_urls.json"
        picker_ctx_p.parent.mkdir(parents=True, exist_ok=True)
        picker_ctx_p.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "api_base_url": client.api_base_url,
                    "cache_dir": str(cache_root),
                    "client_secrets_path": str(auth.client_secrets_path),
                    "token_path": str(auth.token_path),
                    "base_urls": base_url_by_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        cache_hits = 0
        staged_dir = (proj / "cloud_media").resolve()
        if not plan_only:
            # Pass 2: Pooled parallel full-res original download ONLY for selected items
            selected_cands = [c for c in candidates if c.selected_for_download]

            def _fetch_original(cand: CloudMediaCandidateItem) -> None:
                b_url = base_url_by_id.get(cand.cloud_item_id, "")
                if not b_url:
                    cand.selected_for_download = False
                    cand.skip_reason = "missing_base_url"
                    return
                h_id = hashlib.sha1(cand.cloud_item_id.encode("utf-8")).hexdigest()[:16]
                safe_n = _safe_filename(cand.filename, h_id, cand.mime_type, cand.media_type)
                orig_p = orig_dir / f"{h_id}_{safe_n}"
                if orig_p.is_file() and orig_p.stat().st_size > 0:
                    cand.cache_hit = True
                else:
                    mod = "dv" if cand.media_type == "video" else "d"
                    client.download_media_file(b_url, orig_p, modifier=mod)
                cand.downloaded_path = str(orig_p.resolve())

            if selected_cands:
                workers = max(1, min(6, len(selected_cands)))
                if workers == 1:
                    for c in selected_cands:
                        _fetch_original(c)
                else:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        list(pool.map(_fetch_original, selected_cands))
                cache_hits = sum(1 for c in selected_cands if c.cache_hit)

            staged_dir = _stage_selected_items_into_cloud_media(candidates, proj / "cloud_media")

        sel_photos = sum(1 for c in candidates if c.selected_for_download and c.media_type == "photo")
        sel_videos = sum(1 for c in candidates if c.selected_for_download and c.media_type == "video")
        ev_cnt = len({c.event_cluster_id for c in candidates if c.event_cluster_id})

        manifest = GooglePhotosSyncManifest(
            source_mode="google_photos_picker",
            session_id=session_id,
            date_from=date_from,
            date_to=date_to,
            timezone_name=tz_name,
            download_policy="download_all" if download_all else "smart_two_pass",
            download_stage="planned_only" if plan_only else "completed",
            total_scanned_in_session=len(raw_items),
            in_date_range_count=len(candidates),
            selected_photos_count=sel_photos,
            selected_videos_count=sel_videos,
            skipped_count=len(candidates) - (sel_photos + sel_videos),
            cache_hits_count=cache_hits,
            event_clusters_count=ev_cnt,
            staged_media_dir=str(staged_dir),
            scan_contact_sheet_path=str(legacy_sheet_p),
            scan_contact_sheet_paths=[str(p) for p in sheet_paths],
            items=candidates,
        )
        manifest.to_json_file(proj / "google_photos_manifest.json")
        return manifest
    finally:
        if created_here and session_id and not plan_only:
            client.delete_session(session_id)
        client.close()


def _extract_takeout_json_fields(raw_json: dict[str, Any]) -> dict[str, Any]:
    """Parse photoTakenTime / creationTime, geoData, and description from a Takeout JSON sidecar."""
    taken = raw_json.get("photoTakenTime") or raw_json.get("creationTime") or {}
    ts_raw = taken.get("timestamp")
    iso_ts: Optional[str] = None
    if ts_raw is not None:
        try:
            iso_ts = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            iso_ts = None

    def _valid_geo(g: Any) -> Optional[tuple[float, float, Optional[float]]]:
        if not isinstance(g, dict):
            return None
        try:
            lat = float(g.get("latitude", 0.0))
            lon = float(g.get("longitude", 0.0))
            alt = g.get("altitude")
            alt_f = float(alt) if alt is not None else None
            if abs(lat) < 1e-6 and abs(lon) < 1e-6:
                return None
            return lat, lon, alt_f
        except Exception:
            return None

    geo = _valid_geo(raw_json.get("geoData")) or _valid_geo(raw_json.get("geoDataExif"))
    desc = str(raw_json.get("description") or "").strip() or None
    return {
        "creation_time": iso_ts,
        "gps_lat": geo[0] if geo else None,
        "gps_lon": geo[1] if geo else None,
        "gps_alt": geo[2] if geo else None,
        "description": desc,
        "is_favorite": bool(raw_json.get("favorited")),
    }


def _find_takeout_sidecar_key(media_key: str, json_keys_by_dir: dict[str, dict[str, str]]) -> Optional[str]:
    """Resolve classic `.json` and newer `.supplemental-metadata.json` / truncated Takeout sidecars."""
    p = Path(media_key)
    parent = p.parent.as_posix()
    dir_map = json_keys_by_dir.get(parent) or {}
    if not dir_map:
        return None
    name = p.name
    stem = p.stem
    suffix = p.suffix

    exact_candidates = [
        f"{name}.supplemental-metadata.json",
        f"{name}.supplemental-meta.json",
        f"{name}.suppl.json",
        f"{name}.json",
    ]
    # Handle Takeout collision names like `IMG_001(1).JPG` -> `IMG_001.JPG(1).json`
    m_dup = re.match(r"^(.*?)(\(\d+\))(\.[^.]+)$", name)
    if m_dup:
        base_n, dup_idx, ext_n = m_dup.groups()
        exact_candidates.extend(
            [
                f"{base_n}{ext_n}{dup_idx}.supplemental-metadata.json",
                f"{base_n}{ext_n}{dup_idx}.json",
            ]
        )

    for cand in exact_candidates:
        hit = dir_map.get(cand.lower())
        if hit:
            return hit

    # Prefix match for Takeout's ~46-char truncated sidecar filenames (e.g. `very_long_name.jpg.suppl.json`)
    prefix = f"{name[:34].lower()}"
    for lower_fname, orig_key in dir_map.items():
        if lower_fname.startswith(prefix) and ("supplemental" in lower_fname or lower_fname.startswith(name.lower())):
            return orig_key
    # Fallback by stem + suffix hint if unique
    stem_low = stem.lower()
    for lower_fname, orig_key in dir_map.items():
        if lower_fname.startswith(stem_low) and suffix.lower() in lower_fname:
            return orig_key
    return None


def discover_takeout_sidecar_metadata(media_path: Path) -> dict[str, Any]:
    """Read Google Takeout sidecar JSON metadata adjacent to an extracted local media file."""
    parent = media_path.parent
    if not parent.is_dir():
        return {}
    dir_map = {p.name.lower(): str(p) for p in parent.iterdir() if p.is_file() and p.suffix.lower() == ".json"}
    hit = _find_takeout_sidecar_key(media_path.name, {".": dir_map})
    if not hit:
        return {}
    try:
        raw_j = json.loads(Path(hit).read_text(encoding="utf-8"))
        return _extract_takeout_json_fields(raw_j)
    except Exception:
        return {}


def _extract_photo_or_video_thumb(src_path: Path, thumb_path: Path, media_type: str) -> Optional[Path]:
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    if thumb_path.is_file() and thumb_path.stat().st_size > 0:
        return thumb_path
    if media_type == "photo":
        try:
            with Image.open(src_path) as im:
                im2 = ImageOps.exif_transpose(im).convert("RGB")
                im2.thumbnail((512, 512), Image.Resampling.LANCZOS)
                im2.save(thumb_path, "JPEG", quality=88)
                return thumb_path
        except Exception:
            return None
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                "0.3",
                "-i",
                str(src_path),
                "-frames:v",
                "1",
                "-vf",
                "scale='min(512,iw)':-2",
                str(thumb_path),
            ],
            check=False,
            capture_output=True,
            timeout=20,
        )
        if proc.returncode == 0 and thumb_path.is_file() and thumb_path.stat().st_size > 0:
            return thumb_path
    except Exception:
        pass
    return None


def sync_google_takeout_media(
    takeout_source: str | Path,
    project_dir: str | Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tz_name: str = "UTC",
    target_duration_sec: float = 155.0,
    max_photos: Optional[int] = None,
    max_videos: Optional[int] = None,
    download_all: bool = False,
    headroom_multiplier: float = 1.6,
    plan_only: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> GooglePhotosSyncManifest:
    """
    Ingest an exported Google Takeout directory or `.zip`, filter [date_from, date_to], and run smart curation.
    Uses Lazy Two-Pass ZIP extraction so skipped/unselected videos are NEVER extracted to disk.
    """
    src = Path(takeout_source).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Google Takeout path does not exist: {src}")
    proj = Path(project_dir).resolve()
    proj.mkdir(parents=True, exist_ok=True)
    cache_root = proj / "cache" / "takeout_scan"
    thumbs_dir = cache_root / "thumbs"
    extracted_dir = cache_root / "extracted"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    start_utc, end_utc = parse_date_range_bounds_utc(date_from, date_to, tz_name=tz_name)
    candidates: list[CloudMediaCandidateItem] = []
    deferred_zip_videos: dict[str, tuple[str, Path]] = {}
    total_scanned = 0
    is_zip_src = bool(src.is_file() and src.suffix.lower() == ".zip")

    if is_zip_src:
        with zipfile.ZipFile(src, "r") as zf:
            infos = [zi for zi in zf.infolist() if not zi.is_dir()]
            json_by_dir: dict[str, dict[str, str]] = {}
            media_infos: list[zipfile.ZipInfo] = []
            for zi in infos:
                ext = Path(zi.filename).suffix.lower()
                if ext == ".json":
                    p_z = Path(zi.filename)
                    json_by_dir.setdefault(p_z.parent.as_posix(), {})[p_z.name.lower()] = zi.filename
                elif ext in PHOTO_EXTENSIONS or ext in VIDEO_EXTENSIONS:
                    media_infos.append(zi)

            total_scanned = len(media_infos)
            for idx, zi in enumerate(media_infos, start=1):
                ext = Path(zi.filename).suffix.lower()
                mtype = "video" if ext in VIDEO_EXTENSIONS else "photo"
                side_key = _find_takeout_sidecar_key(zi.filename, json_by_dir)
                side_meta: dict[str, Any] = {}
                if side_key:
                    try:
                        side_meta = _extract_takeout_json_fields(json.loads(zf.read(side_key).decode("utf-8")))
                    except Exception:
                        side_meta = {}
                c_time = side_meta.get("creation_time") or datetime(
                    *zi.date_time, tzinfo=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
                if not is_timestamp_in_range(c_time, start_utc, end_utc):
                    continue

                item_id = f"takeout_{hashlib.sha1(zi.filename.encode('utf-8')).hexdigest()[:14]}"
                out_file = extracted_dir / f"{idx:04d}_{Path(zi.filename).name}"
                is_fav = bool(side_meta.get("is_favorite"))

                if mtype == "video":
                    # Lazy Two-Pass: defer extracting heavy video payloads until AFTER Pass-1 selection!
                    deferred_zip_videos[item_id] = (zi.filename, out_file)
                    dl_str = str(out_file.resolve()) if (out_file.is_file() and out_file.stat().st_size == zi.file_size) else None
                    thumb_cached = thumbs_dir / f"{item_id}.jpg"
                    thumb_p = thumb_cached if (thumb_cached.is_file() and thumb_cached.stat().st_size > 0) else None
                    candidates.append(
                        CloudMediaCandidateItem(
                            cloud_item_id=item_id,
                            source_provider="google_takeout",
                            filename=Path(zi.filename).name,
                            mime_type=_EXT_TO_MIME.get(ext, "video/mp4"),
                            media_type="video",
                            creation_time=c_time,
                            gps_lat=side_meta.get("gps_lat"),
                            gps_lon=side_meta.get("gps_lon"),
                            gps_alt=side_meta.get("gps_alt"),
                            proxy_thumbnail_path=str(thumb_p.resolve()) if thumb_p else None,
                            downloaded_path=dl_str,
                            is_favorite=is_fav,
                        )
                    )
                    continue

                # Photos are small: extract for Pass-1 512px thumbnail perception & EXIF
                if not (out_file.is_file() and out_file.stat().st_size == zi.file_size):
                    with zf.open(zi, "r") as zsrc, out_file.open("wb") as dst_fh:
                        shutil.copyfileobj(zsrc, dst_fh, length=1024 * 1024)

                thumb_p = _extract_photo_or_video_thumb(out_file, thumbs_dir / f"{item_id}.jpg", "photo")
                cam = None
                w, h = 0, 0
                try:
                    with Image.open(out_file) as im:
                        w, h = im.size
                        cam, _ = _extract_exif_camera_info(im, out_file)
                        is_fav = is_fav or extract_exif_is_favorite(im, out_file)
                except Exception:
                    pass
                candidates.append(
                    CloudMediaCandidateItem(
                        cloud_item_id=item_id,
                        source_provider="google_takeout",
                        filename=Path(zi.filename).name,
                        mime_type=_EXT_TO_MIME.get(ext, "image/jpeg"),
                        media_type="photo",
                        creation_time=c_time,
                        width=w,
                        height=h,
                        camera_make_model=cam,
                        gps_lat=side_meta.get("gps_lat"),
                        gps_lon=side_meta.get("gps_lon"),
                        gps_alt=side_meta.get("gps_alt"),
                        proxy_thumbnail_path=str(thumb_p.resolve()) if thumb_p else None,
                        downloaded_path=str(out_file.resolve()),
                        is_favorite=is_fav,
                    )
                )
    else:
        all_files = iter_media_files_recursive(src)
        json_by_dir_fs: dict[str, dict[str, str]] = {}
        for jf in src.rglob("*.json"):
            if jf.is_file():
                json_by_dir_fs.setdefault(jf.parent.as_posix(), {})[jf.name.lower()] = str(jf)

        media_files = [f for f in all_files if f.suffix.lower() in PHOTO_EXTENSIONS or f.suffix.lower() in VIDEO_EXTENSIONS]
        total_scanned = len(media_files)

        def _inspect_takeout_fs_file(mf_path: Path) -> Optional[CloudMediaCandidateItem]:
            ext = mf_path.suffix.lower()
            mtype = "video" if ext in VIDEO_EXTENSIONS else "photo"
            side_key = _find_takeout_sidecar_key(str(mf_path), json_by_dir_fs)
            side_meta: dict[str, Any] = {}
            if side_key and Path(side_key).is_file():
                try:
                    side_meta = _extract_takeout_json_fields(json.loads(Path(side_key).read_text(encoding="utf-8")))
                except Exception:
                    side_meta = {}

            exif_ts: Optional[str] = None
            exif_lat: Optional[float] = None
            exif_lon: Optional[float] = None
            exif_alt: Optional[float] = None
            cam = None
            w, h, fps_v, dur_v = 0, 0, (30.0 if mtype == "video" else 0.0), 0.0
            is_fav = bool(side_meta.get("is_favorite"))
            if mtype == "photo":
                try:
                    with Image.open(mf_path) as im:
                        w, h = im.size
                        cam, _ = _extract_exif_camera_info(im, mf_path)
                        exif_lat, exif_lon, exif_alt = _extract_exif_gps_and_alt(im)
                        exif_ts = _extract_photo_exif_dt(im)
                        is_fav = is_fav or extract_exif_is_favorite(im, mf_path)
                except Exception:
                    pass
            else:
                try:
                    vi = probe_video_file(mf_path)
                    w = int(vi.get("width") or 0)
                    h = int(vi.get("height") or 0)
                    fps_v = float(vi.get("fps") or 30.0)
                    dur_v = float(vi.get("duration") or 0.0)
                    cam = vi.get("camera_make_model") or cam
                except Exception:
                    pass

            c_time = (
                side_meta.get("creation_time")
                or exif_ts
                or datetime.fromtimestamp(mf_path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            if not is_timestamp_in_range(c_time, start_utc, end_utc):
                return None

            item_id = f"takeout_{hashlib.sha1(str(mf_path).encode('utf-8')).hexdigest()[:14]}"
            thumb_p = _extract_photo_or_video_thumb(mf_path, thumbs_dir / f"{item_id}.jpg", mtype)
            return CloudMediaCandidateItem(
                cloud_item_id=item_id,
                source_provider="google_takeout",
                filename=mf_path.name,
                mime_type=_EXT_TO_MIME.get(ext, "video/mp4" if mtype == "video" else "image/jpeg"),
                media_type=mtype,
                creation_time=c_time,
                width=w,
                height=h,
                fps=fps_v,
                duration_sec=dur_v,
                camera_make_model=cam,
                gps_lat=side_meta.get("gps_lat") if side_meta.get("gps_lat") is not None else exif_lat,
                gps_lon=side_meta.get("gps_lon") if side_meta.get("gps_lon") is not None else exif_lon,
                gps_alt=side_meta.get("gps_alt") if side_meta.get("gps_alt") is not None else exif_alt,
                proxy_thumbnail_path=str(thumb_p.resolve()) if thumb_p else None,
                downloaded_path=str(mf_path.resolve()),
                is_favorite=is_fav,
            )

        workers_fs = max(1, min(get_io_worker_count(12), len(media_files)))
        if workers_fs == 1 or len(media_files) <= 3:
            fs_rows = [_inspect_takeout_fs_file(p) for p in media_files]
        else:
            with ThreadPoolExecutor(max_workers=workers_fs) as pool:
                fs_rows = list(pool.map(_inspect_takeout_fs_file, media_files))
        candidates = [c for c in fs_rows if c is not None]

    if progress_callback:
        progress_callback(f"Takeout scanned {total_scanned} items, {len(candidates)} in range [{date_from or '*'}..{date_to or '*'}].")

    select_necessary_cloud_items(
        candidates,
        target_duration_sec=target_duration_sec,
        max_photos=max_photos,
        max_videos=max_videos,
        download_all=download_all,
        headroom_multiplier=headroom_multiplier,
    )

    if is_zip_src and deferred_zip_videos:
        # Save zip member index so Pass-1.5 apply_cloud_pick_overrides can lazily extract rescued videos
        (cache_root / "zip_index.json").write_text(
            json.dumps(
                {
                    "takeout_zip": str(src),
                    "members": {
                        cid: {"zip_member": zm, "extract_path": str(ep.resolve())}
                        for cid, (zm, ep) in deferred_zip_videos.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if not plan_only:
            with zipfile.ZipFile(src, "r") as zf:
                for cand in candidates:
                    if not cand.selected_for_download or cand.cloud_item_id not in deferred_zip_videos:
                        continue
                    z_mem, out_f = deferred_zip_videos[cand.cloud_item_id]
                    zi_sel = zf.getinfo(z_mem)
                    if not (out_f.is_file() and out_f.stat().st_size == zi_sel.file_size):
                        with zf.open(zi_sel, "r") as zsrc, out_f.open("wb") as dst_fh:
                            shutil.copyfileobj(zsrc, dst_fh, length=1024 * 1024)
                    cand.downloaded_path = str(out_f.resolve())
                    thumb_p = _extract_photo_or_video_thumb(out_f, thumbs_dir / f"{cand.cloud_item_id}.jpg", "video")
                    if thumb_p:
                        cand.proxy_thumbnail_path = str(thumb_p.resolve())

    sheet_paths, legacy_sheet_p = render_cloud_scan_contact_sheets(
        candidates,
        proj / "contact_sheets",
        title=f"Google Takeout Scan ({date_from or 'All'} -> {date_to or 'All'})",
    )

    staged_dir = (proj / "cloud_media").resolve()
    if not plan_only:
        staged_dir = _stage_selected_items_into_cloud_media(candidates, proj / "cloud_media")

    sel_photos = sum(1 for c in candidates if c.selected_for_download and c.media_type == "photo")
    sel_videos = sum(1 for c in candidates if c.selected_for_download and c.media_type == "video")
    ev_cnt = len({c.event_cluster_id for c in candidates if c.event_cluster_id})
    manifest = GooglePhotosSyncManifest(
        source_mode="google_takeout",
        date_from=date_from,
        date_to=date_to,
        timezone_name=tz_name,
        download_policy="download_all" if download_all else "smart_two_pass",
        download_stage="planned_only" if plan_only else "completed",
        total_scanned_in_session=total_scanned,
        in_date_range_count=len(candidates),
        selected_photos_count=sel_photos,
        selected_videos_count=sel_videos,
        skipped_count=len(candidates) - (sel_photos + sel_videos),
        cache_hits_count=0,
        event_clusters_count=ev_cnt,
        staged_media_dir=str(staged_dir),
        scan_contact_sheet_path=str(legacy_sheet_p),
        scan_contact_sheet_paths=[str(p) for p in sheet_paths],
        items=candidates,
    )
    manifest.to_json_file(proj / "google_photos_manifest.json")
    return manifest


def apply_cloud_pick_overrides(
    project_dir: str | Path,
    overrides: dict[str, Any] | list[dict[str, Any]],
    execute_pass2: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> GooglePhotosSyncManifest:
    """
    Pass-1.5 AI Vision / User Rescue Gate:
    Applies select/skip/favorite overrides (by `#07` tile_id, `cloud_item_id`, or `filename`)
    to `google_photos_manifest.json`, then executes Pass-2 selective download / lazy Takeout extraction
    and stages `<project_dir>/cloud_media/`.
    """
    proj = Path(project_dir).resolve()
    manifest_p = proj / "google_photos_manifest.json"
    if not manifest_p.is_file():
        raise FileNotFoundError(f"google_photos_manifest.json not found in {proj}")
    manifest = GooglePhotosSyncManifest.from_json_file(manifest_p)

    tile_map_p = proj / "contact_sheets" / "cloud_scan_tile_map.json"
    tile_to_cid: dict[str, str] = {}
    if tile_map_p.is_file():
        try:
            raw_tm = json.loads(tile_map_p.read_text(encoding="utf-8"))
            if isinstance(raw_tm, dict):
                for tk, tv in raw_tm.items():
                    if isinstance(tv, dict) and tv.get("cloud_item_id"):
                        tile_to_cid[str(tk).strip().lower()] = str(tv["cloud_item_id"])
        except Exception:
            pass

    by_cid = {it.cloud_item_id: it for it in manifest.items}
    by_name = {it.filename.lower(): it for it in manifest.items}
    by_stem = {Path(it.filename).stem.lower(): it for it in manifest.items}

    def _resolve_item(token: str) -> Optional[CloudMediaCandidateItem]:
        raw_t = str(token or "").strip()
        if not raw_t:
            return None
        low_t = raw_t.lower()
        if low_t in tile_to_cid and tile_to_cid[low_t] in by_cid:
            return by_cid[tile_to_cid[low_t]]
        if not low_t.startswith("#") and f"#{low_t}" in tile_to_cid and tile_to_cid[f"#{low_t}"] in by_cid:
            return by_cid[tile_to_cid[f"#{low_t}"]]
        if raw_t in by_cid:
            return by_cid[raw_t]
        if low_t in by_name:
            return by_name[low_t]
        if Path(raw_t).stem.lower() in by_stem:
            return by_stem[Path(raw_t).stem.lower()]
        return None

    if isinstance(overrides, dict):
        for tok in overrides.get("select") or []:
            it = _resolve_item(str(tok))
            if it is not None:
                it.selected_for_download = True
                it.skip_reason = None
        for tok in overrides.get("skip") or []:
            it = _resolve_item(str(tok))
            if it is not None:
                it.selected_for_download = False
                it.skip_reason = "manual_skip"
        for tok in overrides.get("favorite") or []:
            it = _resolve_item(str(tok))
            if it is not None:
                it.is_favorite = True
                it.selected_for_download = True
                it.skip_reason = None
    elif isinstance(overrides, list):
        for row in overrides:
            if not isinstance(row, dict):
                continue
            tok = str(row.get("tile_id") or row.get("cloud_item_id") or row.get("filename") or "")
            it = _resolve_item(tok)
            if it is None:
                continue
            if "is_favorite" in row:
                it.is_favorite = bool(row["is_favorite"])
            if "selected" in row:
                sel = bool(row["selected"])
                it.selected_for_download = sel
                it.skip_reason = None if sel else str(row.get("skip_reason") or "manual_skip")

    cache_hits = manifest.cache_hits_count
    if execute_pass2:
        # 1. Picker API Pass-2 selective original download for any selected item not yet downloaded
        picker_ctx_p = proj / "cache" / "picker_session_urls.json"
        if manifest.source_mode == "google_photos_picker" and picker_ctx_p.is_file():
            p_ctx = json.loads(picker_ctx_p.read_text(encoding="utf-8"))
            base_urls: dict[str, str] = dict(p_ctx.get("base_urls") or {})
            cache_root = Path(p_ctx.get("cache_dir") or get_default_google_photos_cache_dir())
            orig_dir = cache_root / "originals"
            orig_dir.mkdir(parents=True, exist_ok=True)
            need_dl = [
                it
                for it in manifest.items
                if it.selected_for_download and (not it.downloaded_path or not Path(it.downloaded_path).is_file())
            ]
            if need_dl:
                auth = GooglePhotosAuthManager(
                    client_secrets_path=p_ctx.get("client_secrets_path"),
                    token_path=p_ctx.get("token_path"),
                )
                client = GooglePhotosPickerClient(auth_manager=auth, api_base_url=p_ctx.get("api_base_url"))
                try:
                    def _dl_one(cand: CloudMediaCandidateItem) -> None:
                        b_url = base_urls.get(cand.cloud_item_id, "")
                        if not b_url:
                            return
                        h_id = hashlib.sha1(cand.cloud_item_id.encode("utf-8")).hexdigest()[:16]
                        safe_n = _safe_filename(cand.filename, h_id, cand.mime_type, cand.media_type)
                        orig_p = orig_dir / f"{h_id}_{safe_n}"
                        if orig_p.is_file() and orig_p.stat().st_size > 0:
                            cand.cache_hit = True
                        else:
                            mod = "dv" if cand.media_type == "video" else "d"
                            client.download_media_file(b_url, orig_p, modifier=mod)
                        cand.downloaded_path = str(orig_p.resolve())

                    workers = max(1, min(6, len(need_dl)))
                    if workers == 1:
                        for c in need_dl:
                            _dl_one(c)
                    else:
                        with ThreadPoolExecutor(max_workers=workers) as pool:
                            list(pool.map(_dl_one, need_dl))
                finally:
                    if manifest.session_id:
                        client.delete_session(manifest.session_id)
                    client.close()
                cache_hits = sum(1 for c in manifest.items if c.selected_for_download and c.cache_hit)

        # 2. Takeout ZIP lazy extraction for any newly rescued video/photo not yet extracted
        zip_idx_p = proj / "cache" / "takeout_scan" / "zip_index.json"
        if manifest.source_mode == "google_takeout" and zip_idx_p.is_file():
            z_idx = json.loads(zip_idx_p.read_text(encoding="utf-8"))
            t_zip = Path(z_idx.get("takeout_zip") or "")
            members_map: dict[str, dict[str, str]] = dict(z_idx.get("members") or {})
            need_ex = [
                it
                for it in manifest.items
                if it.selected_for_download
                and it.cloud_item_id in members_map
                and (not it.downloaded_path or not Path(it.downloaded_path).is_file())
            ]
            if need_ex and t_zip.is_file():
                thumbs_dir = proj / "cache" / "takeout_scan" / "thumbs"
                with zipfile.ZipFile(t_zip, "r") as zf:
                    for it in need_ex:
                        rec = members_map[it.cloud_item_id]
                        out_f = Path(rec["extract_path"])
                        out_f.parent.mkdir(parents=True, exist_ok=True)
                        zi_sel = zf.getinfo(rec["zip_member"])
                        if not (out_f.is_file() and out_f.stat().st_size == zi_sel.file_size):
                            with zf.open(zi_sel, "r") as zsrc, out_f.open("wb") as dst_fh:
                                shutil.copyfileobj(zsrc, dst_fh, length=1024 * 1024)
                        it.downloaded_path = str(out_f.resolve())
                        thumb_p = _extract_photo_or_video_thumb(out_f, thumbs_dir / f"{it.cloud_item_id}.jpg", it.media_type)
                        if thumb_p:
                            it.proxy_thumbnail_path = str(thumb_p.resolve())

        staged_dir = _stage_selected_items_into_cloud_media(manifest.items, proj / "cloud_media")
        manifest.staged_media_dir = str(staged_dir)
        manifest.download_stage = "completed"

    sheet_paths, legacy_sheet_p = render_cloud_scan_contact_sheets(
        manifest.items,
        proj / "contact_sheets",
        title=f"Google Photos Curated Picks ({manifest.date_from or 'All'} -> {manifest.date_to or 'All'})",
    )
    manifest.selected_photos_count = sum(1 for c in manifest.items if c.selected_for_download and c.media_type == "photo")
    manifest.selected_videos_count = sum(1 for c in manifest.items if c.selected_for_download and c.media_type == "video")
    manifest.skipped_count = len(manifest.items) - (manifest.selected_photos_count + manifest.selected_videos_count)
    manifest.cache_hits_count = cache_hits
    manifest.scan_contact_sheet_path = str(legacy_sheet_p)
    manifest.scan_contact_sheet_paths = [str(p) for p in sheet_paths]
    manifest.to_json_file(manifest_p)
    if progress_callback:
        progress_callback(
            f"Applied cloud pick overrides: {manifest.selected_photos_count} photos, {manifest.selected_videos_count} videos staged in {manifest.staged_media_dir}."
        )
    return manifest


_TEMP_DOWNLOAD_SUFFIXES = {".crdownload", ".part", ".download", ".tmp"}


def _extract_photo_exif_dt(im: Image.Image) -> Optional[str]:
    """Read DateTimeOriginal / CreateDate from root EXIF or ExifIFD 0x8769 into ISO-8601 UTC."""
    try:
        ex = im.getexif()
    except Exception:
        return None
    raw_candidates: list[Any] = [ex.get(36867), ex.get(36868), ex.get(306)]
    try:
        sub = ex.get_ifd(0x8769)
        if sub:
            raw_candidates = [sub.get(36867), sub.get(36868), *raw_candidates]
    except Exception:
        pass
    for raw_val in raw_candidates:
        if raw_val:
            dt_p = parse_iso_utc(str(raw_val))
            if dt_p is not None:
                return dt_p.strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def build_google_photos_web_search_url(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    extra_query: str = "",
    favorites_only: bool = False,
    media_filter: Optional[str] = None,
    album_url: Optional[str] = None,
) -> str:
    """Build a zero-setup Google Photos web search or album deep-link URL."""
    if album_url and str(album_url).strip().startswith(("http://", "https://")):
        return str(album_url).strip()
    date_part = format_picker_date_query(date_from, date_to)
    parts: list[str] = [date_part] if date_part else []
    if favorites_only:
        parts.append("favorites")
    mf = (media_filter or "").strip().lower()
    if mf in ("videos", "video"):
        parts.append("videos")
    elif mf in ("photos", "photo"):
        parts.append("photos")
    extra = (extra_query or "").strip()
    if extra:
        parts.append(extra)
    query = " ".join(p for p in parts if p).strip()
    if not query:
        return "https://photos.google.com/favorites" if favorites_only else "https://photos.google.com"
    return f"https://photos.google.com/search/{quote(query, safe='')}"


def build_zero_setup_browser_plan(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    album_url: Optional[str] = None,
    favorites_only: bool = False,
) -> dict[str, Any]:
    """Build structured zero-setup Google Photos web deep-links & fast batch-picking playbook."""
    primary_url = build_google_photos_web_search_url(
        date_from=date_from,
        date_to=date_to,
        favorites_only=favorites_only,
        album_url=album_url,
    )
    return {
        "search_url": primary_url,
        "favorites_search_url": build_google_photos_web_search_url(date_from, date_to, favorites_only=True),
        "videos_search_url": build_google_photos_web_search_url(date_from, date_to, media_filter="videos"),
        "date_query": format_picker_date_query(date_from, date_to),
        "max_items_per_zip": 450,
        "recommended_curated_batch_size": "5-15",
        "recommended_pathway": "zero_setup_visual_curated_small_batch",
        "batch_strategy_hint": (
            "Proven Zero-Setup Visual Curation & Small-Batch Download Routine: "
            "1) Inspect overview grids directly on the main library timeline (https://photos.google.com) using the right-hand date scrubber plus the sidebar Favorites tab — immune to Google Photos search RPC (EzkLib) RESOURCE_EXHAUSTED rate-limits and skips utility/whiteboard/burst clutter before downloading; "
            "2) Work cleanly in ONE browser tab with 'Include: Shared with me' UNCHECKED when partner-shared photos are already saved in the main library; "
            "3) Visually scan each chapter/month (jumping briskly across months rather than day-by-day over 90 days) to understand the full story arc, then download only curated standout photos & videos in small batches of 5-15 items per Shift+D ZIP (avoids accounts.google.com download-redirect blocks triggered by oversized 50+ item ZIPs; fall back to single-item download if needed); "
            "4) Download starred Favorites from the requested date range as a dedicated small ZIP and pass their filenames via --favorites for guaranteed selection immunity; "
            "5) Pass the downloads folder directly to sync-google-photos --from-downloads <dir> --favorites <names> --download-all (or --plan-only for Pass-1.5 contact-sheet review) to retain 100% original EXIF timestamps & GPS."
        ),
    }


def get_default_browser_downloads_dir() -> Path:
    """Return the browser downloads folder (overrideable via GOOGLE_PHOTOS_DOWNLOADS_DIR)."""
    env_dir = os.environ.get("GOOGLE_PHOTOS_DOWNLOADS_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (Path.home() / "Downloads").resolve()


def snapshot_downloads_directory(downloads_dir: str | Path) -> dict[str, tuple[int, int]]:
    """Capture a baseline map of {resolved_path: (mtime_ns, size_bytes)} before browser login/download."""
    dl_dir = Path(downloads_dir)
    snap: dict[str, tuple[int, int]] = {}
    if not dl_dir.is_dir():
        return snap
    for p in dl_dir.iterdir():
        try:
            if p.is_file():
                st = p.stat()
                snap[str(p.resolve())] = (int(st.st_mtime_ns), int(st.st_size))
        except OSError:
            continue
    return snap


def watch_browser_downloads(
    downloads_dir: str | Path,
    started_after_epoch: float,
    baseline_files: Optional[dict[str, tuple[int, int]]] = None,
    timeout_sec: float = 600.0,
    settle_quiet_sec: float = 1.8,
    poll_interval_sec: float = 0.4,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[Path]:
    """
    Poll a browser downloads folder until newly downloaded Google Photos `.zip`/media files finish writing.
    Waits out `.crdownload` / `.part` / `.download` temp files and incomplete ZIP archives before returning.
    """
    dl_dir = Path(downloads_dir).expanduser().resolve()
    dl_dir.mkdir(parents=True, exist_ok=True)
    baseline = dict(baseline_files or {})
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    last_state_sig: Any = None
    last_change_mono = time.monotonic()
    last_log_mono = 0.0

    while True:
        active_temps: list[str] = []
        candidates: list[Path] = []
        state_items: list[tuple[str, int, int]] = []

        for p in dl_dir.iterdir():
            try:
                if not p.is_file():
                    continue
                st = p.stat()
                res_str = str(p.resolve())
                ext = p.suffix.lower()
            except OSError:
                continue

            if ext in _TEMP_DOWNLOAD_SUFFIXES:
                if res_str not in baseline or st.st_mtime >= (started_after_epoch - 2.0):
                    active_temps.append(p.name)
                    state_items.append((res_str, int(st.st_mtime_ns), int(st.st_size)))
                continue

            is_supported = ext == ".zip" or ext in PHOTO_EXTENSIONS or ext in VIDEO_EXTENSIONS
            if not is_supported or st.st_size <= 0:
                continue

            prev = baseline.get(res_str)
            is_new_or_updated = (prev is None) or ((int(st.st_mtime_ns), int(st.st_size)) != prev)
            if is_new_or_updated and (prev is None or st.st_mtime >= (started_after_epoch - 5.0)):
                candidates.append(p)
                state_items.append((res_str, int(st.st_mtime_ns), int(st.st_size)))

        state_sig = (tuple(sorted(state_items)), tuple(sorted(active_temps)))
        now_mono = time.monotonic()
        if state_sig != last_state_sig:
            last_state_sig = state_sig
            last_change_mono = now_mono

        if candidates and not active_temps and (now_mono - last_change_mono) >= settle_quiet_sec:
            zips_ok = all((c.suffix.lower() != ".zip" or zipfile.is_zipfile(c)) for c in candidates)
            if zips_ok:
                return sorted(candidates, key=lambda x: x.stat().st_mtime_ns)

        if now_mono >= deadline:
            raise TimeoutError(
                f"Timed out waiting {timeout_sec:.0f}s for Google Photos download in {dl_dir}. "
                "You can re-run with --from-downloads <downloaded_zip_or_file>."
            )

        if progress_callback and (now_mono - last_log_mono) >= 4.0:
            last_log_mono = now_mono
            if active_temps:
                progress_callback(f"Download in progress ({', '.join(active_temps[:2])})...")
            elif candidates:
                progress_callback(f"Detected {len(candidates)} completed file(s); settling...")
            else:
                progress_callback(f"Waiting for browser download in {dl_dir}...")

        time.sleep(max(0.05, float(poll_interval_sec)))


def _expand_browser_download_inputs(
    download_paths: list[str | Path] | str | Path,
) -> list[Path]:
    raw_items: list[Path] = []
    if isinstance(download_paths, (str, Path)):
        raw_str = str(download_paths)
        parts = [s.strip() for s in raw_str.split(",") if s.strip()] if isinstance(download_paths, str) else [raw_str]
        raw_items = [Path(s).expanduser().resolve() for s in parts]
    else:
        raw_items = [Path(p).expanduser().resolve() for p in download_paths]

    expanded: list[Path] = []
    for p in raw_items:
        if not p.exists():
            raise FileNotFoundError(f"Browser download path does not exist: {p}")
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and (
                    child.suffix.lower() == ".zip"
                    or child.suffix.lower() in PHOTO_EXTENSIONS
                    or child.suffix.lower() in VIDEO_EXTENSIONS
                ):
                    expanded.append(child)
        else:
            expanded.append(p)
    return expanded


def _normalize_favorite_lookup(
    favorite_filenames: Optional[list[str] | set[str] | str],
) -> tuple[set[str], set[str]]:
    """Normalize a list, set, comma-separated string, or JSON manifest/list into lowercase (names, stems)."""
    if not favorite_filenames:
        return set(), set()
    raw_entries: list[str] = []
    if isinstance(favorite_filenames, str):
        raw_s = favorite_filenames.strip()
        if raw_s:
            cand_p = Path(raw_s).expanduser()
            if cand_p.is_file():
                try:
                    parsed = json.loads(cand_p.read_text(encoding="utf-8"))
                    if isinstance(parsed, dict):
                        if parsed.get("favoritedFilenames"):
                            raw_entries.extend(str(x) for x in parsed["favoritedFilenames"])
                        elif parsed.get("manifest"):
                            for row in parsed["manifest"]:
                                if isinstance(row, dict) and row.get("isFavorite") and row.get("filename"):
                                    raw_entries.append(str(row["filename"]))
                    elif isinstance(parsed, list):
                        for row in parsed:
                            if isinstance(row, dict):
                                if row.get("isFavorite", True) and row.get("filename"):
                                    raw_entries.append(str(row["filename"]))
                            elif row:
                                raw_entries.append(str(row))
                except Exception:
                    raw_entries.extend(s.strip() for s in raw_s.split(",") if s.strip())
            elif raw_s.startswith("["):
                try:
                    parsed_l = json.loads(raw_s)
                    if isinstance(parsed_l, list):
                        raw_entries.extend(str(x) for x in parsed_l if x)
                except Exception:
                    raw_entries.extend(s.strip() for s in raw_s.split(",") if s.strip())
            else:
                raw_entries.extend(s.strip() for s in raw_s.split(",") if s.strip())
    else:
        raw_entries.extend(str(x).strip() for x in favorite_filenames if x)

    names_low = {Path(x).name.lower() for x in raw_entries if Path(x).name}
    stems_low = {Path(x).stem.lower() for x in raw_entries if Path(x).stem}
    return names_low, stems_low


def ingest_google_photos_browser_downloads(
    download_paths: list[str | Path] | str | Path,
    project_dir: str | Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tz_name: str = "UTC",
    target_duration_sec: float = 155.0,
    max_photos: Optional[int] = None,
    max_videos: Optional[int] = None,
    download_all: bool = False,
    headroom_multiplier: float = 1.6,
    plan_only: bool = False,
    keep_undated_items: bool = True,
    favorite_filenames: Optional[list[str] | set[str] | str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> GooglePhotosSyncManifest:
    """
    Unpack single or multi-batch browser-downloaded Google Photos `.zip`/media files in parallel,
    filter optional [date_from, date_to], run parallel video-editing-aware Pass-1 burst/junk/MMR curation,
    render paginated 5x4 contact sheets, and stage originals into `<project_dir>/cloud_media/`.
    """
    proj = Path(project_dir).resolve()
    proj.mkdir(parents=True, exist_ok=True)
    work_root = proj / "cache" / "browser_downloads"
    extract_root = work_root / "extracted"
    thumbs_dir = work_root / "thumbs"
    extract_root.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    inputs = _expand_browser_download_inputs(download_paths)
    collected_media: list[Path] = []
    seen_real: set[str] = set()

    for zip_idx, src_p in enumerate(inputs, start=1):
        ext = src_p.suffix.lower()
        if ext == ".zip" and zipfile.is_zipfile(src_p):
            dest_sub = extract_root / f"batch_{zip_idx:02d}_{src_p.stem[:24]}"
            dest_sub.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(src_p, "r") as zf:
                for mem_idx, zi in enumerate(zf.infolist(), start=1):
                    if zi.is_dir():
                        continue
                    rel_parts = Path(zi.filename).parts
                    if not rel_parts or Path(zi.filename).is_absolute() or ".." in rel_parts:
                        continue
                    m_name = Path(zi.filename).name
                    m_ext = Path(m_name).suffix.lower()
                    if m_ext not in PHOTO_EXTENSIONS and m_ext not in VIDEO_EXTENSIONS and m_ext != ".json":
                        continue
                    target_f = dest_sub / m_name
                    if target_f.exists() and m_ext != ".json" and target_f.stat().st_size != zi.file_size:
                        target_f = dest_sub / f"{Path(m_name).stem}_{mem_idx:03d}{Path(m_name).suffix}"
                    if not (target_f.is_file() and target_f.stat().st_size == zi.file_size):
                        with zf.open(zi, "r") as zsrc, target_f.open("wb") as dst_fh:
                            shutil.copyfileobj(zsrc, dst_fh, length=1024 * 1024)
                    if m_ext in PHOTO_EXTENSIONS or m_ext in VIDEO_EXTENSIONS:
                        res_k = str(target_f.resolve())
                        if res_k not in seen_real:
                            seen_real.add(res_k)
                            collected_media.append(target_f)
        elif ext in PHOTO_EXTENSIONS or ext in VIDEO_EXTENSIONS:
            res_k = str(src_p.resolve())
            if res_k not in seen_real:
                seen_real.add(res_k)
                collected_media.append(src_p)

    start_utc, end_utc = parse_date_range_bounds_utc(date_from, date_to, tz_name=tz_name)
    fav_names_low, fav_stems_low = _normalize_favorite_lookup(favorite_filenames)

    def _inspect_browser_file(mf_path: Path) -> Optional[CloudMediaCandidateItem]:
        ext = mf_path.suffix.lower()
        mtype = "video" if ext in VIDEO_EXTENSIONS else "photo"
        side_meta = discover_takeout_sidecar_metadata(mf_path)
        c_time: Optional[str] = side_meta.get("creation_time")
        cam: Optional[str] = None
        exif_sum: Optional[str] = None
        w, h, fps_val, dur_val = 0, 0, (30.0 if mtype == "video" else 0.0), 0.0
        g_lat, g_lon, g_alt = side_meta.get("gps_lat"), side_meta.get("gps_lon"), side_meta.get("gps_alt")
        is_fav = (
            bool(side_meta.get("is_favorite"))
            or mf_path.name.lower() in fav_names_low
            or mf_path.stem.lower() in fav_stems_low
        )

        if mtype == "photo":
            try:
                with Image.open(mf_path) as im:
                    im_t = ImageOps.exif_transpose(im)
                    w, h = im_t.size
                    cam, _ = _extract_exif_camera_info(im, mf_path)
                    ex_lat, ex_lon, ex_alt = _extract_exif_gps_and_alt(im)
                    if g_lat is None:
                        g_lat, g_lon, g_alt = ex_lat, ex_lon, ex_alt
                    if not c_time:
                        c_time = _extract_photo_exif_dt(im)
                    is_fav = is_fav or extract_exif_is_favorite(im, mf_path)
            except Exception:
                pass
        else:
            try:
                vi = probe_video_file(mf_path)
                w = int(vi.get("width") or 0)
                h = int(vi.get("height") or 0)
                fps_val = float(vi.get("fps") or 30.0)
                dur_val = float(vi.get("duration") or 0.0)
                cam = vi.get("camera_make_model") or cam
                if not c_time and vi.get("creation_time"):
                    dt_v = parse_iso_utc(str(vi["creation_time"]))
                    if dt_v:
                        c_time = dt_v.strftime("%Y-%m-%dT%H:%M:%SZ")
                if g_lat is None and vi.get("gps_lat") is not None:
                    g_lat = vi.get("gps_lat")
                    g_lon = vi.get("gps_lon")
                    g_alt = vi.get("gps_alt")
            except Exception:
                pass

        if start_utc is not None or end_utc is not None:
            if c_time:
                if not is_timestamp_in_range(c_time, start_utc, end_utc):
                    return None
            elif not keep_undated_items:
                return None

        item_id = f"browser_{hashlib.sha1(str(mf_path).encode('utf-8')).hexdigest()[:14]}"
        thumb_p = _extract_photo_or_video_thumb(mf_path, thumbs_dir / f"{item_id}.jpg", mtype)
        return CloudMediaCandidateItem(
            cloud_item_id=item_id,
            source_provider="google_photos_browser",
            filename=mf_path.name,
            mime_type=_EXT_TO_MIME.get(ext, "video/mp4" if mtype == "video" else "image/jpeg"),
            media_type=mtype,
            creation_time=c_time or "",
            width=w,
            height=h,
            fps=fps_val,
            duration_sec=dur_val,
            camera_make_model=cam,
            exif_summary=exif_sum,
            gps_lat=g_lat,
            gps_lon=g_lon,
            gps_alt=g_alt,
            proxy_thumbnail_path=str(thumb_p.resolve()) if thumb_p else None,
            downloaded_path=str(mf_path.resolve()),
            is_favorite=is_fav,
        )

    workers_io = max(1, min(get_io_worker_count(12), len(collected_media)))
    if workers_io == 1 or len(collected_media) <= 3:
        inspected = [_inspect_browser_file(p) for p in collected_media]
    else:
        with ThreadPoolExecutor(max_workers=workers_io) as pool:
            inspected = list(pool.map(_inspect_browser_file, collected_media))
    candidates = [c for c in inspected if c is not None]

    if progress_callback:
        progress_callback(
            f"Browser download scanned {len(collected_media)} files, {len(candidates)} kept in range [{date_from or '*'}..{date_to or '*'}]."
        )

    select_necessary_cloud_items(
        candidates,
        target_duration_sec=target_duration_sec,
        max_photos=max_photos,
        max_videos=max_videos,
        download_all=download_all,
        headroom_multiplier=headroom_multiplier,
    )
    sheet_paths, legacy_sheet_p = render_cloud_scan_contact_sheets(
        candidates,
        proj / "contact_sheets",
        title=f"Google Photos Browser Import ({date_from or 'All'} -> {date_to or 'All'})",
    )

    staged_dir = (proj / "cloud_media").resolve()
    if not plan_only:
        staged_dir = _stage_selected_items_into_cloud_media(candidates, proj / "cloud_media")

    sel_photos = sum(1 for c in candidates if c.selected_for_download and c.media_type == "photo")
    sel_videos = sum(1 for c in candidates if c.selected_for_download and c.media_type == "video")
    ev_cnt = len({c.event_cluster_id for c in candidates if c.event_cluster_id})
    manifest = GooglePhotosSyncManifest(
        source_mode="google_photos_browser",
        date_from=date_from,
        date_to=date_to,
        timezone_name=tz_name,
        download_policy="download_all" if download_all else "smart_two_pass",
        download_stage="planned_only" if plan_only else "completed",
        total_scanned_in_session=len(collected_media),
        in_date_range_count=len(candidates),
        selected_photos_count=sel_photos,
        selected_videos_count=sel_videos,
        skipped_count=len(candidates) - (sel_photos + sel_videos),
        cache_hits_count=0,
        event_clusters_count=ev_cnt,
        staged_media_dir=str(staged_dir),
        scan_contact_sheet_path=str(legacy_sheet_p),
        scan_contact_sheet_paths=[str(p) for p in sheet_paths],
        items=candidates,
    )
    manifest.to_json_file(proj / "google_photos_manifest.json")
    return manifest


def sync_google_photos_via_browser(
    project_dir: str | Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tz_name: str = "UTC",
    target_duration_sec: float = 155.0,
    max_photos: Optional[int] = None,
    max_videos: Optional[int] = None,
    download_all: bool = False,
    headroom_multiplier: float = 1.6,
    plan_only: bool = False,
    favorites_only: bool = False,
    album_url: Optional[str] = None,
    downloads_dir: Optional[str | Path] = None,
    timeout_sec: float = 600.0,
    settle_quiet_sec: float = 1.8,
    open_browser: bool = True,
    favorite_filenames: Optional[list[str] | set[str] | str] = None,
    browser_open_callback: Optional[Callable[[str, Path], None]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> GooglePhotosSyncManifest:
    """
    Zero-setup Google Photos Browser Login workflow:
    1. Opens `photos.google.com/search/<date_from> to <date_to>` (or Album / Favorites URL) in the user's browser.
    2. Watches `~/Downloads` for newly downloaded `.zip` / media files after sign-in & Day-Header/Shift+Click selection.
    3. Extracts in parallel, filters [date_from, date_to], runs video-editing Pass-1 curation, renders paginated sheets, and stages `cloud_media/`.
    """
    dl_dir = Path(downloads_dir).expanduser().resolve() if downloads_dir else get_default_browser_downloads_dir()
    dl_dir.mkdir(parents=True, exist_ok=True)
    baseline = snapshot_downloads_directory(dl_dir)
    started_at = time.time()
    search_url = build_google_photos_web_search_url(
        date_from,
        date_to,
        favorites_only=favorites_only,
        album_url=album_url,
    )

    if browser_open_callback is not None:
        browser_open_callback(search_url, dl_dir)
    elif open_browser:
        try:
            webbrowser.open(search_url, new=2)
        except Exception:
            pass

    if progress_callback:
        progress_callback(f"Opened {search_url}; watching {dl_dir} for finished download...")

    detected = watch_browser_downloads(
        dl_dir,
        started_after_epoch=started_at,
        baseline_files=baseline,
        timeout_sec=timeout_sec,
        settle_quiet_sec=settle_quiet_sec,
        progress_callback=progress_callback,
    )
    return ingest_google_photos_browser_downloads(
        download_paths=detected,
        project_dir=project_dir,
        date_from=date_from,
        date_to=date_to,
        tz_name=tz_name,
        target_duration_sec=target_duration_sec,
        max_photos=max_photos,
        max_videos=max_videos,
        download_all=download_all,
        headroom_multiplier=headroom_multiplier,
        plan_only=plan_only,
        favorite_filenames=favorite_filenames,
        progress_callback=progress_callback,
    )


def resolve_auth_mode(
    auth_mode: str = "auto",
    client_secrets_path: Optional[str | Path] = None,
    token_path: Optional[str | Path] = None,
) -> str:
    """Resolve effective Google Photos connection mode ('browser' by default unless OAuth token is ready)."""
    mode = (auth_mode or "auto").strip().lower()
    if mode in ("browser", "oauth"):
        return mode
    if mode != "auto":
        raise ValueError(f"Unsupported Google Photos auth_mode '{auth_mode}'; expected 'auto', 'browser', or 'oauth'.")
    st = GooglePhotosAuthManager(client_secrets_path=client_secrets_path, token_path=token_path).get_auth_status()
    return "oauth" if st.get("ready") else "browser"


def run_google_photos_sync(
    project_dir: str | Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tz_name: str = "UTC",
    target_duration_sec: float = 155.0,
    max_photos: Optional[int] = None,
    max_videos: Optional[int] = None,
    download_all: bool = False,
    headroom_multiplier: float = 1.6,
    plan_only: bool = False,
    favorites_only: bool = False,
    album_url: Optional[str] = None,
    auth_mode: str = "auto",
    from_downloads: Optional[str | Path | list[str | Path]] = None,
    takeout_source: Optional[str | Path] = None,
    downloads_dir: Optional[str | Path] = None,
    watch_timeout_sec: float = 600.0,
    session_id: Optional[str] = None,
    open_browser: bool = True,
    client_secrets_path: Optional[str | Path] = None,
    token_path: Optional[str | Path] = None,
    favorite_filenames: Optional[list[str] | set[str] | str] = None,
    browser_open_callback: Optional[Callable[[str, Path], None]] = None,
    picker_ready_callback: Optional[Callable[[str, str], None]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> GooglePhotosSyncManifest:
    """Unified dispatcher across Browser Downloads import, Takeout archive, OAuth Picker, and Browser Login watcher."""
    if from_downloads is not None:
        return ingest_google_photos_browser_downloads(
            download_paths=from_downloads,
            project_dir=project_dir,
            date_from=date_from,
            date_to=date_to,
            tz_name=tz_name,
            target_duration_sec=target_duration_sec,
            max_photos=max_photos,
            max_videos=max_videos,
            download_all=download_all,
            headroom_multiplier=headroom_multiplier,
            plan_only=plan_only,
            favorite_filenames=favorite_filenames,
            progress_callback=progress_callback,
        )
    if takeout_source is not None:
        return sync_google_takeout_media(
            takeout_source=takeout_source,
            project_dir=project_dir,
            date_from=date_from,
            date_to=date_to,
            tz_name=tz_name,
            target_duration_sec=target_duration_sec,
            max_photos=max_photos,
            max_videos=max_videos,
            download_all=download_all,
            headroom_multiplier=headroom_multiplier,
            plan_only=plan_only,
            progress_callback=progress_callback,
        )
    eff_mode = "oauth" if session_id else resolve_auth_mode(
        auth_mode=auth_mode,
        client_secrets_path=client_secrets_path,
        token_path=token_path,
    )
    if eff_mode == "oauth":
        return sync_google_photos_media(
            project_dir=project_dir,
            date_from=date_from,
            date_to=date_to,
            tz_name=tz_name,
            target_duration_sec=target_duration_sec,
            max_photos=max_photos,
            max_videos=max_videos,
            download_all=download_all,
            headroom_multiplier=headroom_multiplier,
            plan_only=plan_only,
            session_id=session_id,
            open_browser=open_browser,
            client_secrets_path=client_secrets_path,
            token_path=token_path,
            picker_ready_callback=picker_ready_callback,
            progress_callback=progress_callback,
        )
    return sync_google_photos_via_browser(
        project_dir=project_dir,
        date_from=date_from,
        date_to=date_to,
        tz_name=tz_name,
        target_duration_sec=target_duration_sec,
        max_photos=max_photos,
        max_videos=max_videos,
        download_all=download_all,
        headroom_multiplier=headroom_multiplier,
        plan_only=plan_only,
        favorites_only=favorites_only,
        album_url=album_url,
        downloads_dir=downloads_dir,
        timeout_sec=watch_timeout_sec,
        open_browser=open_browser,
        favorite_filenames=favorite_filenames,
        browser_open_callback=browser_open_callback,
        progress_callback=progress_callback,
    )
