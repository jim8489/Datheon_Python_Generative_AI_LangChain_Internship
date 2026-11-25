"""Signed anonymous browser identity for isolating persistent chat histories.

This is browser-level identity, not authentication. A copied cookie can be replayed;
use an authenticated identity provider for accounts or cross-device access.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.settings import (
    BROWSER_COOKIE_NAME,
    BROWSER_COOKIE_TTL_DAYS,
    SESSION_SIGNING_KEY,
    TASK4_ROOT,
)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def load_signing_key(
    key_path: str | Path | None = None,
    configured_key: str | None = SESSION_SIGNING_KEY,
) -> bytes:
    """Load a stable signing key, creating a private local key when not configured."""
    if configured_key:
        key = configured_key.encode("utf-8")
        if len(key) < 32:
            raise ValueError("SESSION_SIGNING_KEY must contain at least 32 bytes.")
        return key

    path = Path(key_path) if key_path else TASK4_ROOT / "data" / ".browser_signing_key"
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("xb") as key_file:
            key = secrets.token_bytes(32)
            key_file.write(key)
            key_file.flush()
            try:
                path.chmod(0o600)
            except OSError:
                # Windows and some mounted filesystems do not support POSIX modes.
                pass
            return key
    except FileExistsError:
        key = path.read_bytes()
        if len(key) < 32:
            raise ValueError(f"The browser signing key at {path} is invalid or truncated.")
        return key


def create_browser_token(
    browser_id: str,
    signing_key: bytes,
    issued_at: int | None = None,
) -> str:
    """Create a versioned HMAC-signed token for one anonymous browser identity."""
    timestamp = int(time.time()) if issued_at is None else int(issued_at)
    payload = f"v1:{browser_id}:{timestamp}".encode("utf-8")
    signature = hmac.new(signing_key, payload, hashlib.sha256).digest()
    return f"{_b64encode(payload)}.{_b64encode(signature)}"


def verify_browser_token(
    token: Any,
    signing_key: bytes,
    *,
    now: int | None = None,
    ttl_seconds: int | None = None,
) -> str | None:
    """Return the browser ID only when signature, format, and expiry are valid."""
    if not isinstance(token, str) or len(token) > 512 or "." not in token:
        return None

    try:
        payload_part, signature_part = token.split(".", 1)
        payload = _b64decode(payload_part)
        supplied_signature = _b64decode(signature_part)
        expected_signature = hmac.new(signing_key, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(supplied_signature, expected_signature):
            return None

        version, browser_id, issued_at_text = payload.decode("utf-8").split(":", 2)
        if version != "v1" or len(browser_id) != 32:
            return None
        int(browser_id, 16)
        issued_at = int(issued_at_text)
    except (ValueError, TypeError, UnicodeDecodeError):
        return None

    current_time = int(time.time()) if now is None else int(now)
    max_age = ttl_seconds or BROWSER_COOKIE_TTL_DAYS * 24 * 60 * 60
    if issued_at <= 0 or issued_at > current_time + 60:
        return None
    if current_time - issued_at > max_age:
        return None
    return browser_id


def get_or_create_browser_id(
    cookie_controller: Any,
    signing_key: bytes,
    *,
    secure: bool,
    cookie_name: str = BROWSER_COOKIE_NAME,
    ttl_days: int = BROWSER_COOKIE_TTL_DAYS,
    fallback_token: str | None = None,
) -> str:
    """Read a verified browser cookie or issue a new signed, opaque identifier."""
    ttl_seconds = ttl_days * 24 * 60 * 60
    existing_token = cookie_controller.get(cookie_name) or fallback_token
    browser_id = verify_browser_token(
        existing_token,
        signing_key,
        ttl_seconds=ttl_seconds,
    )
    if browser_id:
        return browser_id

    browser_id = uuid4().hex
    token = create_browser_token(browser_id, signing_key)
    cookie_controller.set(
        cookie_name,
        token,
        path="/",
        expires=datetime.now() + timedelta(seconds=ttl_seconds),
        max_age=ttl_seconds,
        secure=secure,
        same_site="strict",
    )
    return browser_id
