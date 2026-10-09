"""
Monthly Call Archive & Lifecycle Management Service.
- Retains current month active calls in live Supabase database.
- Automatically or manually rolls over completed months into structured CSV files on the server.
- Allows Admin to view, download, or delete archived month CSV files.
"""

import calendar
import csv
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.config import ROOT_DIR
from backend.database.connection import get_supabase_client, is_supabase_enabled
from backend.database.repository import check_supabase_schema

logger = logging.getLogger("hindi-ai-calling.archive-service")

ARCHIVES_DIR = ROOT_DIR / "archives"
ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)


def format_file_size(size_bytes: int) -> str:
    """Format bytes into readable KB/MB."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def parse_month_name(year: int, month: int) -> str:
    """Return Month Name Year (e.g. September 2026)."""
    return f"{calendar.month_name[month]} {year}"


def sanitize_filename(filename: str) -> str:
    """Prevent directory traversal attacks."""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "", os.path.basename(filename))


def list_monthly_archives() -> List[Dict[str, Any]]:
    """List all available monthly CSV archive files stored on the server."""
    archives = []
    if not ARCHIVES_DIR.exists():
        return []

    for file in sorted(ARCHIVES_DIR.glob("*.csv"), reverse=True):
        stat = file.stat()
        name = file.name

        # Parse year and month from filename like indiawalls_calls_2026_09.csv
        match = re.search(r"(\d{4})_(\d{2})", name)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            month_label = parse_month_name(year, month)
        else:
            month_label = name.replace(".csv", "").replace("_", " ").title()

        # Count rows in CSV
        row_count = 0
        try:
            with open(file, "r", encoding="utf-8-sig", errors="ignore") as f:
                row_count = max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass

        archives.append({
            "filename": name,
            "month_label": month_label,
            "size_bytes": stat.st_size,
            "size_formatted": format_file_size(stat.st_size),
            "created_at": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
            "row_count": row_count,
        })

    return archives


def get_archive_file_path(filename: str) -> Optional[Path]:
    """Retrieve absolute Path to an archive file, validating safety."""
    clean_name = sanitize_filename(filename)
    path = ARCHIVES_DIR / clean_name
    if path.is_file() and path.exists():
        return path
    return None


def delete_archive(filename: str) -> bool:
    """Delete an archived CSV file from the server."""
    path = get_archive_file_path(filename)
    if path and path.exists():
        try:
            path.unlink()
            logger.info(f"Archive file deleted: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete archive {filename}: {e}")
    return False


def get_month_date_range(year: int, month: int) -> Tuple[str, str]:
    """Return ISO UTC bounds (start, end) for a given year and month."""
    _, last_day = calendar.monthrange(year, month)
    start_iso = f"{year:04d}-{month:02d}-01T00:00:00+00:00"
    end_iso = f"{year:04d}-{month:02d}-{last_day:02d}T23:59:59+00:00"
    return start_iso, end_iso


def export_month_to_csv(year: int, month: int, purge_from_db: bool = True) -> Dict[str, Any]:
    """
    Export all calls and messages from a specific month to a CSV file.
    Optionally purges those records from Supabase so the database begins fresh.
    """
    client = get_supabase_client()
    if not client or not check_supabase_schema():
        return {"success": False, "error": "Supabase database not connected."}

    start_iso, end_iso = get_month_date_range(year, month)
    month_name = parse_month_name(year, month)
    filename = f"indiawalls_calls_{year}_{month:02d}.csv"
    output_path = ARCHIVES_DIR / filename

    try:
        # 1. Fetch calls within date range
        calls_res = (
            client.table("calls")
            .select("id, call_sid, channel, from_number, to_number, start_time, end_time, duration_seconds, turn_count, status")
            .gte("start_time", start_iso)
            .lte("start_time", end_iso)
            .order("start_time", desc=False)
            .execute()
        )
        calls = calls_res.data or []

        if not calls:
            return {
                "success": True,
                "exported_calls": 0,
                "exported_messages": 0,
                "filename": filename,
                "message": f"No calls found for {month_name}.",
            }

        call_ids = [c["id"] for c in calls]
        calls_map = {c["id"]: c for c in calls}

        # 2. Fetch call messages for these calls
        all_messages: List[Dict[str, Any]] = []
        # Chunk requests if there are many calls
        chunk_size = 50
        for i in range(0, len(call_ids), chunk_size):
            chunk = call_ids[i : i + chunk_size]
            msg_res = (
                client.table("call_messages")
                .select("id, call_id, role, content, timestamp, latency_ms")
                .in_("call_id", chunk)
                .order("id", desc=False)
                .execute()
            )
            all_messages.extend(msg_res.data or [])

        # 3. Write Comprehensive CSV Export
        with open(output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.writer(csvfile)
            # Header
            writer.writerow([
                "Call ID",
                "Channel",
                "From Number",
                "To Number",
                "Call Status",
                "Call Start Time",
                "Call End Time",
                "Duration (Seconds)",
                "Turn Count",
                "Message ID",
                "Speaker Role",
                "Message Content / Speech",
                "Turn Latency (ms)",
                "Message Timestamp",
            ])

            if all_messages:
                for msg in all_messages:
                    cid = msg.get("call_id")
                    call = calls_map.get(cid, {})
                    writer.writerow([
                        cid,
                        call.get("channel", "web"),
                        call.get("from_number", "Unknown"),
                        call.get("to_number", ""),
                        call.get("status", "completed"),
                        call.get("start_time", ""),
                        call.get("end_time", ""),
                        call.get("duration_seconds", 0.0),
                        call.get("turn_count", 0),
                        msg.get("id", ""),
                        "Customer" if msg.get("role") == "user" else "Priya AI",
                        msg.get("content", ""),
                        msg.get("latency_ms", 0),
                        msg.get("timestamp", ""),
                    ])
            else:
                # If calls exist but no messages
                for call in calls:
                    writer.writerow([
                        call.get("id"),
                        call.get("channel", "web"),
                        call.get("from_number", "Unknown"),
                        call.get("to_number", ""),
                        call.get("status", "completed"),
                        call.get("start_time", ""),
                        call.get("end_time", ""),
                        call.get("duration_seconds", 0.0),
                        call.get("turn_count", 0),
                        "",
                        "",
                        "(No messages)",
                        0,
                        "",
                    ])

        logger.info(f"Exported {len(calls)} calls and {len(all_messages)} messages to {output_path}")

        # 4. Purge from database if requested
        purged_count = 0
        if purge_from_db:
            for i in range(0, len(call_ids), chunk_size):
                chunk = call_ids[i : i + chunk_size]
                client.table("calls").delete().in_("id", chunk).execute()
                purged_count += len(chunk)
            logger.info(f"Purged {purged_count} archived calls from Supabase for fresh month start.")

        return {
            "success": True,
            "exported_calls": len(calls),
            "exported_messages": len(all_messages),
            "filename": filename,
            "file_size": format_file_size(output_path.stat().st_size),
            "month_label": month_name,
            "purged_from_db": purge_from_db,
        }

    except Exception as e:
        logger.error(f"Error during month export for {month_name}: {e}")
        return {"success": False, "error": str(e)}


def check_and_run_monthly_rollover() -> Dict[str, Any]:
    """
    Automated Monthly Rollover:
    Scans Supabase for any calls dated in previous months.
    If found, exports them to monthly CSV files and deletes them from active Supabase tables,
    keeping only the current calendar month active.
    """
    client = get_supabase_client()
    if not client or not check_supabase_schema():
        return {"rollover_executed": False, "reason": "Supabase not connected"}

    now = datetime.now(timezone.utc)
    current_month_start = datetime(now.year, now.month, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat()

    try:
        # Check for calls strictly before the 1st day of the current month
        old_calls_res = (
            client.table("calls")
            .select("start_time")
            .lt("start_time", current_month_start)
            .limit(200)
            .execute()
        )
        old_calls = old_calls_res.data or []

        if not old_calls:
            return {"rollover_executed": False, "reason": "No previous month calls to archive"}

        # Find distinct (year, month) pairs
        distinct_months = set()
        for c in old_calls:
            st = c.get("start_time")
            if st:
                try:
                    dt = datetime.fromisoformat(st.replace("Z", "+00:00"))
                    distinct_months.add((dt.year, dt.month))
                except Exception:
                    pass

        results = []
        for year, month in distinct_months:
            res = export_month_to_csv(year, month, purge_from_db=True)
            results.append(res)

        return {
            "rollover_executed": True,
            "archived_months": results,
            "current_month_active": parse_month_name(now.year, now.month),
        }

    except Exception as e:
        logger.error(f"Error checking monthly rollover: {e}")
        return {"rollover_executed": False, "error": str(e)}
