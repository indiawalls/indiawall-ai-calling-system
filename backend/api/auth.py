"""
Admin Authentication Endpoints.
Protects web console with username and password login.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

from backend.services.auth_service import (
    create_session,
    get_admin_username,
    revoke_session,
    validate_session,
    verify_admin_credentials,
)

logger = logging.getLogger("hindi-ai-calling.api.auth")
router = APIRouter(prefix="/api/auth", tags=["Admin Auth & Security"])


class LoginPayload(BaseModel):
    username: str
    password: str


def require_admin_auth(authorization: Optional[str] = Header(None)) -> dict:
    """Dependency ensuring caller holds a valid admin session token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in with admin username and password.",
        )
    token = authorization.split("Bearer ", 1)[1].strip()
    session = validate_session(token)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid. Please log in again.",
        )
    return session


@router.post("/login")
async def login_admin(payload: LoginPayload):
    """Authenticate admin using username and password."""
    username = payload.username.strip()
    password = payload.password.strip()

    if not verify_admin_credentials(username, password):
        logger.warning(f"Failed login attempt for username: '{username}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    token = create_session(username)
    logger.info(f"Admin logged in successfully: '{username}'")
    return {
        "success": True,
        "token": token,
        "username": username,
        "message": "Login successful.",
    }


@router.get("/me")
async def get_current_user_profile(authorization: Optional[str] = Header(None)):
    """Check if client has an active authenticated admin session."""
    if not authorization or not authorization.startswith("Bearer "):
        return {"authenticated": False, "username": None}
    token = authorization.split("Bearer ", 1)[1].strip()
    session = validate_session(token)
    if not session:
        return {"authenticated": False, "username": None}
    return {
        "authenticated": True,
        "username": session.get("username", get_admin_username()),
    }


@router.post("/logout")
async def logout_admin(authorization: Optional[str] = Header(None)):
    """Revoke admin session token."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ", 1)[1].strip()
        revoke_session(token)
    return {"success": True, "message": "Logged out successfully."}
