"""
Admin Panel API Endpoints for IndiaWalls AI Voice Calling.
Handles:
- System Dashboard Metrics & Daily Call Volume Breakdown
- Real-Time STT/LLM/TTS Latency Profiling
- Dynamic LLM (Gemini) API Key Rotation Management
- Monthly Archive Lifecycle, CSV File Downloads & Cleanup
"""

import logging
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.auth import require_admin_auth
from backend.database import (
    get_call_stats,
    get_daily_call_stats,
    get_latency_analytics,
)
from backend.services.archive_service import (
    check_and_run_monthly_rollover,
    delete_archive,
    export_month_to_csv,
    get_archive_file_path,
    list_monthly_archives,
)
from backend.services.llm.gemini_rotator import gemini_rotator

logger = logging.getLogger("hindi-ai-calling.api.admin")
router = APIRouter(prefix="/api/admin", tags=["Admin Panel"])


# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------

class AddKeyRequest(BaseModel):
    key: str = Field(..., min_length=10, description="Google Gemini API Key string")


class TestKeyRequest(BaseModel):
    key: Optional[str] = Field(None, description="Optional key to test, or tests active key if omitted")


class RolloverRequest(BaseModel):
    year: Optional[int] = None
    month: Optional[int] = None
    purge: bool = True


# ---------------------------------------------------------
# 1. Executive Dashboard Summary & Daily Calls
# ---------------------------------------------------------

@router.get("/dashboard-summary")
async def get_dashboard_summary(_admin: dict = Depends(require_admin_auth)):
    """Comprehensive executive analytics for the Admin Panel."""
    call_stats = get_call_stats()
    daily_stats = get_daily_call_stats(days=14)
    latency_stats = get_latency_analytics()
    rotator_status = gemini_rotator.get_status()

    return {
        "calls": call_stats,
        "daily": daily_stats,
        "latency": latency_stats,
        "rotator": rotator_status,
        "timestamp": time.time(),
    }


@router.get("/daily-stats")
async def get_daily_stats(
    days: int = Query(14, ge=1, le=90),
    _admin: dict = Depends(require_admin_auth)
):
    """Daily call count distribution (Web vs Exotel, today/yesterday trend)."""
    return get_daily_call_stats(days=days)


# ---------------------------------------------------------
# 2. Latency Monitoring & Performance Analytics
# ---------------------------------------------------------

@router.get("/latency-stats")
async def get_latency_stats(_admin: dict = Depends(require_admin_auth)):
    """Real-time latency breakdown across STT, LLM, TTS and total turnaround."""
    return get_latency_analytics()


# ---------------------------------------------------------
# 3. LLM API Key Management
# ---------------------------------------------------------

@router.get("/llm-keys")
async def list_llm_keys(_admin: dict = Depends(require_admin_auth)):
    """List all registered Gemini keys, active pointer, cooldowns, and request counts."""
    return gemini_rotator.get_status()


@router.post("/llm-keys")
async def add_llm_key(req: AddKeyRequest, _admin: dict = Depends(require_admin_auth)):
    """Add a new Gemini API key dynamically to the rotator pool and persist to .env."""
    clean = req.key.strip()
    if not clean.startswith("AIza") and len(clean) < 20:
        raise HTTPException(
            status_code=400,
            detail="Invalid Gemini API key format. Must be a valid Google AI Studio key.",
        )

    success = gemini_rotator.add_key(clean, persist=True)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add key to rotator.")

    return {
        "success": True,
        "message": f"Successfully added key ending ...{clean[-6:]}",
        "status": gemini_rotator.get_status(),
    }


@router.delete("/llm-keys/{key_id}")
async def remove_llm_key(key_id: int, _admin: dict = Depends(require_admin_auth)):
    """Remove a Gemini API key by its 1-indexed identifier."""
    success = gemini_rotator.remove_key(key_id, persist=True)
    if not success:
        raise HTTPException(status_code=404, detail="Key ID not found in pool.")

    return {
        "success": True,
        "message": f"Successfully removed key #{key_id}",
        "status": gemini_rotator.get_status(),
    }


@router.post("/llm-keys/test")
async def test_llm_key(req: TestKeyRequest, _admin: dict = Depends(require_admin_auth)):
    """Perform an instantaneous test query against Google Gemini to verify key validity & latency."""
    test_key = req.key.strip() if req.key else gemini_rotator.get_current_key()
    if not test_key:
        raise HTTPException(status_code=400, detail="No Gemini API key available to test.")

    t0 = time.time()
    try:
        from google import genai
        client = genai.Client(api_key=test_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents="Say 'System ready' in 2 words.",
        )
        elapsed_ms = round((time.time() - t0) * 1000)
        reply = (resp.text or "").strip()
        return {
            "success": True,
            "latency_ms": elapsed_ms,
            "response": reply,
            "masked_key": f"...{test_key[-6:]}" if len(test_key) >= 6 else "***",
            "message": "Key is active and responsive!",
        }
    except Exception as e:
        elapsed_ms = round((time.time() - t0) * 1000)
        return {
            "success": False,
            "latency_ms": elapsed_ms,
            "error": str(e),
            "masked_key": f"...{test_key[-6:]}" if len(test_key) >= 6 else "***",
            "message": f"Verification failed: {e}",
        }


# ---------------------------------------------------------
# 4. Monthly Archive & Lifecycle Management
# ---------------------------------------------------------

@router.get("/archives")
async def get_monthly_archives(_admin: dict = Depends(require_admin_auth)):
    """List all available monthly CSV archive files stored on the server."""
    files = list_monthly_archives()
    return {
        "total_archives": len(files),
        "archives": files,
    }


@router.get("/archives/{filename}/download")
async def download_monthly_archive(filename: str, _admin: dict = Depends(require_admin_auth)):
    """Download a specific monthly CSV archive file from the server."""
    file_path = get_archive_file_path(filename)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Archive file '{filename}' not found.")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="text/csv",
    )


@router.delete("/archives/{filename}")
async def delete_monthly_archive(filename: str, _admin: dict = Depends(require_admin_auth)):
    """Remove a monthly archive CSV file from the server."""
    file_path = get_archive_file_path(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Archive file '{filename}' not found.")

    success = delete_archive(filename)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete archive file.")

    return {
        "success": True,
        "message": f"Successfully deleted archive file: {filename}",
    }


@router.post("/archives/rollover")
async def trigger_monthly_rollover(
    req: RolloverRequest,
    _admin: dict = Depends(require_admin_auth)
):
    """
    Export calls to monthly CSV archive and start with a fresh month in the database.
    If year and month are omitted, automatically archives all prior completed months.
    """
    if req.year and req.month:
        res = export_month_to_csv(year=req.year, month=req.month, purge_from_db=req.purge)
        return res
    else:
        res = check_and_run_monthly_rollover()
        return res
