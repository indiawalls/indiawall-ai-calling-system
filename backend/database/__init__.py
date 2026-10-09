"""Database module exports for IndiaWalls AI Voice Calling."""

from .connection import get_supabase_client, is_supabase_enabled
from .repository import (
    add_call_message,
    check_supabase_schema,
    clear_calls_db,
    create_call,
    delete_call,
    end_call,
    get_call_details,
    get_call_stats,
    get_calls_history,
    init_db,
    update_call_caller_id,
)

from .migrator import auto_migrate_supabase
from .seeder import verify_and_seed_database

__all__ = [
    "auto_migrate_supabase",
    "get_supabase_client",
    "is_supabase_enabled",
    "check_supabase_schema",
    "init_db",
    "verify_and_seed_database",
    "create_call",
    "update_call_caller_id",
    "add_call_message",
    "end_call",
    "get_calls_history",
    "get_call_details",
    "get_call_stats",
    "delete_call",
    "clear_calls_db",
]
