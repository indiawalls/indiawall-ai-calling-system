"""
Admin Authentication Service (Hardcoded Username & Password).
Handles secure credential verification and session tokens for the Web Console.
"""

import hmac
import logging
import secrets
import time
from typing import Dict, Optional

from backend.config import ADMIN_PASSWORD, ADMIN_USERNAME

logger = logging.getLogger("hindi-ai-calling.auth")

# In-memory store: session_token -> {username, created_at, expires_at}
_ACTIVE_SESSIONS: Dict[str, Dict] = {}


def verify_admin_credentials(username: str, password: str) -> bool:
    """Verify username and password against configured admin credentials."""
    expected_user = (ADMIN_USERNAME or "admin").strip()
    expected_pwd = (ADMIN_PASSWORD or "admin123").strip()

    valid_user = hmac.compare_digest(username.strip(), expected_user)
    valid_pwd = hmac.compare_digest(password.strip(), expected_pwd)

    return valid_user and valid_pwd


def get_admin_username() -> str:
    """Get active admin username."""
    return (ADMIN_USERNAME or "admin").strip()


def create_session(username: str, duration_hours: int = 24) -> str:
    """Generate a secure session token valid for 24 hours."""
    token = secrets.token_urlsafe(32)
    _ACTIVE_SESSIONS[token] = {
        "username": username.strip(),
        "created_at": time.time(),
        "expires_at": time.time() + (duration_hours * 3600),
    }
    return token


def validate_session(token: str) -> Optional[Dict]:
    """Validate a session token. Returns session dict if valid, else None."""
    if not token:
        return None
    session = _ACTIVE_SESSIONS.get(token)
    if not session:
        return None
    if time.time() > session["expires_at"]:
        _ACTIVE_SESSIONS.pop(token, None)
        return None
    return session


def revoke_session(token: str) -> bool:
    """Revoke/Logout a session token."""
    return bool(_ACTIVE_SESSIONS.pop(token, None))
