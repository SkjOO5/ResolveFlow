"""
server/app.py — OpenEnv standard server module location.

The OpenEnv validator expects the FastAPI application to be importable
from `server.app`. This module re-exports the main app from the project root.
"""
from app import app  # noqa: F401

__all__ = ["app"]
