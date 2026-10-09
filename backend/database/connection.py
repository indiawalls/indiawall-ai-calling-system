"""
Database Connection Manager for IndiaWalls AI Voice Calling System.
Exclusively connects to Supabase Cloud PostgreSQL. Local SQLite is completely removed.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("hindi-ai-calling.db.connection")

_SUPABASE_CLIENT = None
_SUPABASE_INITIALIZED = False


def is_supabase_enabled() -> bool:
    """Check if Supabase credentials are configured in environment."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    return bool(url and key and not url.startswith("your-") and not key.startswith("your-"))


def get_supabase_client():
    """Lazily initialize and return the official Supabase client."""
    global _SUPABASE_CLIENT, _SUPABASE_INITIALIZED
    if _SUPABASE_INITIALIZED:
        return _SUPABASE_CLIENT

    if is_supabase_enabled():
        url = os.environ.get("SUPABASE_URL", "").strip()
        key = os.environ.get("SUPABASE_KEY", "").strip()
        try:
            from supabase import create_client
            _SUPABASE_CLIENT = create_client(url, key)
            _SUPABASE_INITIALIZED = True
            logger.info(f"Connected to Supabase PostgreSQL at {url[:30]}...")
            return _SUPABASE_CLIENT
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            _SUPABASE_INITIALIZED = True
            _SUPABASE_CLIENT = None
            return None

    _SUPABASE_INITIALIZED = True
    _SUPABASE_CLIENT = None
    logger.warning("SUPABASE_URL or SUPABASE_KEY is missing. Please set them in .env")
    return None
