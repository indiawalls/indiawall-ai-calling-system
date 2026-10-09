"""
Call Session & Transcript Repository.
Exclusively powered by Supabase Cloud PostgreSQL.
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from .connection import get_supabase_client, is_supabase_enabled

logger = logging.getLogger("hindi-ai-calling.db.repository")

_supabase_schema_available: Optional[bool] = None
_warned_missing_schema = False


def check_supabase_schema(force_refresh: bool = False) -> bool:
    """Verify if Supabase 'calls' and 'call_messages' tables are created and accessible."""
    global _supabase_schema_available, _warned_missing_schema
    if _supabase_schema_available is True and not force_refresh:
        return True

    if not is_supabase_enabled():
        _supabase_schema_available = False
        return False

    client = get_supabase_client()
    if not client:
        _supabase_schema_available = False
        return False

    try:
        client.table("calls").select("id").limit(1).execute()
        client.table("call_messages").select("id").limit(1).execute()
        _supabase_schema_available = True
        logger.info("Supabase PostgreSQL tables ('calls', 'call_messages') verified & ready.")
        return True
    except Exception as e:
        _supabase_schema_available = False
        if not _warned_missing_schema:
            _warned_missing_schema = True
            logger.warning(
                "\n" + "=" * 70 + "\n"
                "[!] ACTION REQUIRED IN SUPABASE: Tables 'calls' & 'call_messages' do not exist!\n"
                "    Please run 'backend/database/schema.sql' in your Supabase SQL Editor:\n"
                "    --> https://supabase.com/dashboard/project/ntcpgtpholfrfymxdfsv/sql\n"
                + "=" * 70 + "\n"
            )
        return False


def init_db() -> bool:
    """Verify Supabase Cloud database connection and schema on server start."""
    if not is_supabase_enabled():
        logger.error("SUPABASE_URL and SUPABASE_KEY must be configured in .env.")
        return False
    return check_supabase_schema(force_refresh=True)


def create_call(
    call_id: str,
    channel: str = "web",
    from_number: str = "Web Visitor",
    to_number: str = "",
    call_sid: str = "",
) -> Dict[str, Any]:
    """Record the start of a new call session in Supabase."""
    now_iso = datetime.now().isoformat()
    client = get_supabase_client()

    if client and check_supabase_schema():
        try:
            client.table("calls").upsert({
                "id": call_id,
                "call_sid": call_sid or None,
                "channel": channel,
                "from_number": from_number or "Unknown",
                "to_number": to_number or None,
                "start_time": now_iso,
                "status": "in-progress",
                "turn_count": 0,
                "duration_seconds": 0.0,
            }).execute()
        except Exception as e:
            logger.error(f"Supabase error creating call {call_id}: {e}")

    return {
        "call_id": call_id,
        "channel": channel,
        "from_number": from_number,
        "start_time": now_iso,
    }


def update_call_caller_id(call_id: str, caller_number: str, call_sid: str = ""):
    """Update caller phone number in Supabase when detected from Exotel start event."""
    if not caller_number or caller_number == "Unknown":
        return

    client = get_supabase_client()
    if client and check_supabase_schema():
        try:
            update_payload: Dict[str, Any] = {"from_number": caller_number}
            if call_sid:
                update_payload["call_sid"] = call_sid
            client.table("calls").update(update_payload).eq("id", call_id).execute()
        except Exception as e:
            logger.error(f"Supabase error updating caller ID for {call_id}: {e}")


def add_call_message(
    call_id: str,
    role: str,
    content: str,
    latency_ms: int = 0,
):
    """Add a dialogue message to the call transcript in Supabase."""
    if not content or not content.strip():
        return

    now_iso = datetime.now().isoformat()
    client = get_supabase_client()

    if client and check_supabase_schema():
        try:
            client.table("call_messages").insert({
                "call_id": call_id,
                "role": role,
                "content": content.strip(),
                "timestamp": now_iso,
                "latency_ms": latency_ms,
            }).execute()

            # Increment turn_count on calls
            res = client.table("calls").select("turn_count").eq("id", call_id).execute()
            current_turns = res.data[0].get("turn_count", 0) if res.data else 0
            client.table("calls").update({"turn_count": (current_turns or 0) + 1}).eq("id", call_id).execute()
        except Exception as e:
            logger.error(f"Supabase error adding message to call {call_id}: {e}")


def end_call(call_id: str, status: str = "completed"):
    """Mark a call as completed and calculate total duration in Supabase."""
    now = datetime.now()
    now_iso = now.isoformat()
    duration_secs = 0.0

    client = get_supabase_client()
    if client and check_supabase_schema():
        try:
            res = client.table("calls").select("start_time").eq("id", call_id).execute()
            if res.data and res.data[0].get("start_time"):
                try:
                    start_dt = datetime.fromisoformat(res.data[0]["start_time"].replace("Z", "+00:00")).replace(tzinfo=None)
                    duration_secs = max(1.0, round((now - start_dt).total_seconds(), 1))
                except Exception:
                    pass

            client.table("calls").update({
                "status": status,
                "duration_seconds": duration_secs,
                "end_time": now_iso,
            }).eq("id", call_id).execute()
        except Exception as e:
            logger.error(f"Supabase error ending call {call_id}: {e}")


def cleanup_stale_calls(max_idle_seconds: int = 120):
    """Mark abandoned calls as 'completed' in Supabase."""
    client = get_supabase_client()
    if not client or not check_supabase_schema():
        return

    now = datetime.now()
    now_iso = now.isoformat()

    try:
        res = client.table("calls").select("id, start_time").in_("status", ["in-progress", "active"]).execute()
        for call in res.data or []:
            call_id = call["id"]
            start_time = call.get("start_time")
            msg_res = (
                client.table("call_messages")
                .select("timestamp")
                .eq("call_id", call_id)
                .order("id", desc=True)
                .limit(1)
                .execute()
            )
            last_time = msg_res.data[0]["timestamp"] if msg_res.data else start_time
            if last_time:
                try:
                    last_dt = datetime.fromisoformat(last_time.replace("Z", "+00:00")).replace(tzinfo=None)
                    if (now - last_dt).total_seconds() > max_idle_seconds:
                        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00")).replace(tzinfo=None)
                        dur = max(1.0, round((last_dt - start_dt).total_seconds(), 1))
                        client.table("calls").update({
                            "status": "completed",
                            "duration_seconds": dur,
                            "end_time": now_iso,
                        }).eq("id", call_id).execute()
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"Error cleaning up stale calls in Supabase: {e}")


def get_calls_history(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Retrieve recent calls with message previews exclusively from Supabase."""
    cleanup_stale_calls(max_idle_seconds=120)
    client = get_supabase_client()

    if not client or not check_supabase_schema():
        return []

    try:
        res = (
            client.table("calls")
            .select("id, call_sid, channel, from_number, to_number, start_time, end_time, duration_seconds, turn_count, status")
            .order("start_time", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        results = []
        for call in res.data or []:
            c = dict(call)
            msg_res = (
                client.table("call_messages")
                .select("content, role")
                .eq("call_id", c["id"])
                .order("id", desc=True)
                .limit(1)
                .execute()
            )
            if msg_res.data:
                c["last_message"] = msg_res.data[0].get("content", "")
                c["last_role"] = msg_res.data[0].get("role", "")
            else:
                c["last_message"] = ""
                c["last_role"] = ""
            results.append(c)
        return results
    except Exception as e:
        logger.error(f"Supabase error fetching calls history: {e}")
        return []


def get_call_details(call_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full call metadata and all dialogue messages from Supabase."""
    client = get_supabase_client()
    if not client or not check_supabase_schema():
        return None

    try:
        call_res = client.table("calls").select("*").eq("id", call_id).limit(1).execute()
        if call_res.data:
            call_data = dict(call_res.data[0])
            msg_res = (
                client.table("call_messages")
                .select("id, role, content, timestamp, latency_ms")
                .eq("call_id", call_id)
                .order("id", desc=False)
                .execute()
            )
            call_data["messages"] = msg_res.data or []
            return call_data
    except Exception as e:
        logger.error(f"Supabase error fetching call details for {call_id}: {e}")
    return None


def get_call_stats() -> Dict[str, Any]:
    """Calculate call analytics directly from Supabase Cloud."""
    client = get_supabase_client()
    if not client or not check_supabase_schema():
        return {
            "total_calls": 0,
            "total_duration_minutes": 0.0,
            "completed_calls": 0,
            "active_calls": 0,
            "channels": {},
        }

    try:
        res = client.table("calls").select("channel, duration_seconds, status").execute()
        calls = res.data or []
        total_calls = len(calls)
        total_secs = sum(c.get("duration_seconds") or 0.0 for c in calls)
        completed = sum(1 for c in calls if c.get("status") == "completed")
        active = sum(1 for c in calls if c.get("status") in ("in-progress", "active"))
        channels: Dict[str, int] = {}
        for c in calls:
            ch = c.get("channel") or "exotel"
            channels[ch] = channels.get(ch, 0) + 1

        return {
            "total_calls": total_calls,
            "total_duration_minutes": round(total_secs / 60.0, 1),
            "completed_calls": completed,
            "active_calls": active,
            "channels": channels,
        }
    except Exception as e:
        logger.error(f"Supabase error fetching stats: {e}")
        return {
            "total_calls": 0,
            "total_duration_minutes": 0.0,
            "completed_calls": 0,
            "active_calls": 0,
            "channels": {},
        }


def delete_call(call_id: str) -> bool:
    """Delete a call and its transcript messages from Supabase."""
    client = get_supabase_client()
    if client and check_supabase_schema():
        try:
            client.table("calls").delete().eq("id", call_id).execute()
            return True
        except Exception as e:
            logger.error(f"Supabase error deleting call {call_id}: {e}")
    return False


def clear_calls_db() -> bool:
    """Delete all calls and transcripts from Supabase."""
    client = get_supabase_client()
    if client and check_supabase_schema():
        try:
            client.table("calls").delete().neq("id", "___none___").execute()
            return True
        except Exception as e:
            logger.error(f"Supabase error clearing calls: {e}")
    return False
