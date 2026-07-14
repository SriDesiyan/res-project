"""
analytics/inference/__init__.py
================================
Public surface of the inference abstraction layer.

Import the factory function to obtain the correct backend:

    from analytics.inference import create_engine
    engine = create_engine(backend="auto")

Or import engine types directly for type-hinting:

    from analytics.inference import BaseInferenceEngine
"""
from analytics.inference.base_engine import BaseInferenceEngine
from analytics.inference.engine_factory import create_engine

__all__ = ["BaseInferenceEngine", "create_engine"]
