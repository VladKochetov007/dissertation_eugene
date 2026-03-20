"""
Offline merchant agents — Pearl's Level 2 (intervention).

These agents ingest data from the physical world: sensors, embodied robots,
and experimental apparatus. Unlike online merchants who passively observe,
offline merchants collect *interventional* data — measurements taken while
an agent acts upon the environment.

This is the epistemological argument for embodiment: online data is a biased
sample of reality (only what's been digitized). The physical world is where
causation LIVES. Offline merchants bridge that gap.

Available components:
- SensorDataRouter: FastAPI router for accepting sensor/robot data
- Physical measurement schemas (SensorReading, GPSCoordinate, etc.)
- InterventionLog: structured logging of physical-world interventions
"""

from .schemas import (
    EnvironmentMeasurement,
    GPSCoordinate,
    InterventionLog,
    InterventionType,
    SensorReading,
    SensorType,
)

__all__ = [
    "SensorReading",
    "SensorType",
    "GPSCoordinate",
    "EnvironmentMeasurement",
    "InterventionLog",
    "InterventionType",
]
