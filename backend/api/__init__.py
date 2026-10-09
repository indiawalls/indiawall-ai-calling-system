"""API routes module."""

from .admin import router as admin_router
from .auth import router as auth_router
from .calls import router as calls_router
from .system import router as system_router
from .websockets import router as websockets_router

__all__ = ["admin_router", "auth_router", "calls_router", "system_router", "websockets_router"]
