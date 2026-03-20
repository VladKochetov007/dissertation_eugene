"""
Physical Measurement Schemas — data models for offline / embodied merchant agents.

These schemas define the structured formats for data collected by physical
sensors, embodied robots, and experimental apparatus. Unlike online merchants
that observe digital traces, offline merchants operate at Pearl's Level 2
(intervention): they act upon the physical world and measure the consequences.

The epistemological argument for embodiment: online data is a biased sample
of reality — only what's been digitized. The physical world is where causation
LIVES. These schemas standardize how that ground-truth data enters the
knowledge graph.

Schemas defined:
- SensorReading: individual sensor measurements (temperature, pressure, etc.)
- GPSCoordinate: geographic location data
- EnvironmentMeasurement: composite environmental readings with location
- InterventionLog: structured log of a physical-world intervention and its outcome

All models use Pydantic v2 for validation, serialization, and schema generation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SensorType(str, Enum):
    """Type of physical sensor providing measurements.

    Covers common sensor modalities for environmental monitoring,
    industrial measurement, and robotic perception.
    """

    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    SOUND = "sound"
    ACCELERATION = "acceleration"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    AIR_QUALITY = "air_quality"
    CO2 = "co2"
    PARTICULATE = "particulate"
    RADIATION = "radiation"
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    RAINFALL = "rainfall"
    SOIL_MOISTURE = "soil_moisture"
    PH = "ph"
    CAMERA = "camera"
    LIDAR = "lidar"
    ULTRASONIC = "ultrasonic"
    CUSTOM = "custom"


class InterventionType(str, Enum):
    """Type of physical-world intervention performed by an embodied agent.

    Pearl's Level 2 (intervention) requires that the agent perform an
    action and observe the consequence. This enum classifies the kinds
    of interventions that embodied merchants can perform.
    """

    MEASUREMENT = "measurement"
    MANIPULATION = "manipulation"
    SAMPLING = "sampling"
    DEPLOYMENT = "deployment"
    ADJUSTMENT = "adjustment"
    OBSERVATION = "observation"
    EXPERIMENT = "experiment"
    CUSTOM = "custom"


class MeasurementUnit(str, Enum):
    """Physical units for sensor measurements.

    SI units and common derived units for the sensor types above.
    """

    # Temperature
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    KELVIN = "kelvin"

    # Pressure
    PASCAL = "pascal"
    BAR = "bar"
    PSI = "psi"
    ATM = "atm"

    # Distance / length
    METER = "meter"
    CENTIMETER = "centimeter"
    MILLIMETER = "millimeter"
    KILOMETER = "kilometer"

    # Speed
    METERS_PER_SECOND = "m/s"
    KILOMETERS_PER_HOUR = "km/h"

    # Mass
    KILOGRAM = "kilogram"
    GRAM = "gram"

    # Light
    LUX = "lux"
    LUMENS = "lumens"

    # Sound
    DECIBELS = "dB"

    # Electrical
    VOLTS = "volts"
    AMPS = "amps"
    WATTS = "watts"

    # Environmental
    PPM = "ppm"
    PPB = "ppb"
    PERCENT = "percent"
    UG_PER_M3 = "ug/m3"

    # Angular
    DEGREES = "degrees"
    RADIANS = "radians"

    # Rate
    MM_PER_HOUR = "mm/h"

    # Generic
    UNITLESS = "unitless"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Core schemas
# ---------------------------------------------------------------------------


class GPSCoordinate(BaseModel):
    """Geographic coordinate with optional altitude and accuracy.

    Standard WGS84 coordinate system used by GPS receivers and
    mapping services.
    """

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees (-90 to 90).",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees (-180 to 180).",
    )
    altitude: Optional[float] = Field(
        default=None,
        description="Altitude above sea level in meters.",
    )
    accuracy: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Horizontal accuracy in meters (GPS CEP or similar).",
    )


class SensorReading(BaseModel):
    """A single measurement from a physical sensor.

    The atomic unit of offline merchant data: one value, from one sensor,
    at one point in time, with full provenance and unit metadata.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this reading.",
    )
    sensor_id: str = Field(
        ...,
        description="Identifier of the physical sensor that produced this reading.",
    )
    sensor_type: SensorType = Field(
        ...,
        description="Type of sensor (temperature, pressure, etc.).",
    )
    value: float = Field(
        ...,
        description="The measured value.",
    )
    unit: MeasurementUnit = Field(
        ...,
        description="Physical unit of the measurement.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the measurement was taken.",
    )
    quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Data quality indicator (0 = unreliable, 1 = calibrated/verified).",
    )
    location: Optional[GPSCoordinate] = Field(
        default=None,
        description="Geographic location where the measurement was taken.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional sensor-specific metadata (calibration info, etc.).",
    )


class EnvironmentMeasurement(BaseModel):
    """A composite environmental measurement from multiple sensors.

    Groups related sensor readings taken at the same time and place,
    providing a snapshot of environmental conditions. This is the typical
    payload from a weather station, environmental monitor, or mobile
    sensing platform.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this measurement set.",
    )
    station_id: str = Field(
        ...,
        description="Identifier of the measurement station or robot.",
    )
    location: GPSCoordinate = Field(
        ...,
        description="Geographic location of the measurement.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the measurement.",
    )
    readings: list[SensorReading] = Field(
        default_factory=list,
        description="Individual sensor readings in this measurement set.",
    )
    conditions_summary: Optional[str] = Field(
        default=None,
        description="Human-readable summary of conditions (e.g., 'partly cloudy, light wind').",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (mission ID, operator, etc.).",
    )


class InterventionLog(BaseModel):
    """Structured log of a physical-world intervention and its outcome.

    This is the key data model for Pearl's Level 2 (intervention). An
    embodied agent performs an action (the intervention) and records
    what happened (the outcome). This is fundamentally different from
    observation: the agent is *causing* something and measuring the
    effect, not merely watching.

    Example: a robot adjusts a valve (intervention) and records the
    resulting pressure change (outcome). This generates interventional
    data that supports do-calculus reasoning.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this intervention log entry.",
    )
    agent_id: str = Field(
        ...,
        description="Identifier of the embodied agent that performed the intervention.",
    )
    intervention_type: InterventionType = Field(
        ...,
        description="Category of intervention performed.",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the intervention.",
    )
    target_variable: str = Field(
        ...,
        description=(
            "The variable being intervened on — in do-calculus notation, "
            "this is the X in do(X=x)."
        ),
    )
    intervention_value: Any = Field(
        ...,
        description="The value set by the intervention (the x in do(X=x)).",
    )
    timestamp_start: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the intervention began.",
    )
    timestamp_end: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when the intervention completed.",
    )
    location: Optional[GPSCoordinate] = Field(
        default=None,
        description="Geographic location of the intervention.",
    )
    pre_intervention_readings: list[SensorReading] = Field(
        default_factory=list,
        description="Sensor readings taken immediately before the intervention.",
    )
    post_intervention_readings: list[SensorReading] = Field(
        default_factory=list,
        description="Sensor readings taken after the intervention.",
    )
    outcome_description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the observed outcome.",
    )
    success: bool = Field(
        default=True,
        description="Whether the intervention was successfully executed.",
    )
    hypothesis_id: Optional[str] = Field(
        default=None,
        description="ID of the hypothesis this intervention was designed to test.",
    )
    experiment_id: Optional[str] = Field(
        default=None,
        description="ID of the experiment this intervention is part of.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (environmental conditions, equipment state, etc.).",
    )


# ---------------------------------------------------------------------------
# Batch ingestion models
# ---------------------------------------------------------------------------


class SensorBatch(BaseModel):
    """A batch of sensor readings for bulk ingestion.

    Used by the sensor API endpoint to accept multiple readings
    in a single request, reducing HTTP overhead for high-frequency
    data sources.
    """

    station_id: str = Field(
        ...,
        description="Identifier of the measurement station or robot.",
    )
    readings: list[SensorReading] = Field(
        ...,
        min_length=1,
        description="One or more sensor readings to ingest.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Batch-level metadata (mission ID, batch sequence, etc.).",
    )


class InterventionBatch(BaseModel):
    """A batch of intervention logs for bulk ingestion.

    Used by the sensor API to accept multiple intervention records
    in a single request.
    """

    agent_id: str = Field(
        ...,
        description="Identifier of the embodied agent.",
    )
    interventions: list[InterventionLog] = Field(
        ...,
        min_length=1,
        description="One or more intervention logs to ingest.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Batch-level metadata.",
    )
