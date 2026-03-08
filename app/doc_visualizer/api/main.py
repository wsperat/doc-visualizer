"""ASGI entrypoint for running Doc Visualizer API with uvicorn."""

from doc_visualizer.api.app import app

__all__ = ["app"]
