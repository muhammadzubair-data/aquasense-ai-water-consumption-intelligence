"""Backward-compatible module: the original ``from src.analytics import ...`` keeps working.

The implementation now lives in ``aquasense/core.py``.
"""
from aquasense.core import *  # noqa: F401,F403
from aquasense.core import (REQUIRED, add_anomalies, forecast_meter, generate_data, leak_signals,  # noqa: F401
                            validate_data)
