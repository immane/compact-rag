"""Admin configuration reader."""

from __future__ import annotations

import hashlib
import hmac
from functools import lru_cache

from compact_rag.config.settings import AdminSettings, get_settings


@lru_cache()
def get_admin_settings() -> AdminSettings:
    return get_settings().admin


def get_api_base_url(api_port: int = 8000) -> str:
    return f"http://127.0.0.1:{api_port}"


def is_auth_required() -> bool:
    """Check if admin password authentication is configured."""
    settings = get_admin_settings()
    return bool(settings.password)


def verify_password(password: str) -> bool:
    """Verify the admin password using constant-time comparison.

    Args:
        password: The password to check.

    Returns:
        True if the password matches the configured ADMIN_PASSWORD.
        If no password is configured, always returns True (auth disabled).
    """
    settings = get_admin_settings()
    if not settings.password:
        return True
    if not password:
        return False
    return hmac.compare_digest(password, settings.password)


def hash_password_for_storage(password: str) -> str:
    """NOT used in production — password is stored in env var, never persisted."""
    return hashlib.sha256(password.encode()).hexdigest()
