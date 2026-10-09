"""
Call History and System Analytics REST API Router.
"""

from fastapi import APIRouter, Depends, HTTPException

import backend.database as db
from backend.api.auth import require_admin_auth
from backend.services.llm import gemini_rotator

router = APIRouter(prefix="/api", tags=["Calls & History"])


@router.get("/calls")
async def get_call_history(limit: int = 50, offset: int = 0, _admin: dict = Depends(require_admin_auth)):
    """Fetch past caller history with timestamps, durations, turn counts, and channels."""
    calls = db.get_calls_history(limit=limit, offset=offset)
    return {"calls": calls, "count": len(calls)}


@router.get("/calls/{call_id}")
async def get_call_record(call_id: str, _admin: dict = Depends(require_admin_auth)):
    """Fetch full transcript messages and metadata for a specific call session."""
    call = db.get_call_details(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call record not found")
    return call


@router.get("/stats")
async def get_system_stats(_admin: dict = Depends(require_admin_auth)):
    """Return dashboard analytics including total calls, durations, and Gemini key rotation status."""
    call_stats = db.get_call_stats()
    rotator_status = gemini_rotator.get_status()
    return {
        "calls": call_stats,
        "rotator": rotator_status,
        "database": "supabase",
    }


@router.post("/calls/clear")
async def clear_all_calls(_admin: dict = Depends(require_admin_auth)):
    """Clear call history and transcripts database."""
    db.clear_calls_db()
    return {"status": "success", "message": "Call history cleared"}


@router.delete("/calls/{call_id}")
async def delete_single_call(call_id: str, _admin: dict = Depends(require_admin_auth)):
    """Delete a single call session and its messages."""
    deleted = db.delete_call(call_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Call record not found")
    return {"status": "success", "message": f"Call {call_id} deleted"}
