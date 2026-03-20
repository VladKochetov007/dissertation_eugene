"""
Sensor Data Ingestion API — FastAPI router for accepting physical-world data.

Provides REST endpoints for embodied agents (sensors, robots, measurement
stations) to push data into the knowledge graph. This is the entry point
for Pearl's Level 2 (intervention) data: measurements taken while an agent
acts upon the environment.

The API accepts:
- Individual sensor readings
- Batched sensor readings (for high-frequency sources)
- Environmental measurements (composite multi-sensor snapshots)
- Intervention logs (do-calculus: agent intervened, measured outcome)

All data is validated, enriched with provenance metadata, and ingested
into the KnowledgeGraphStore as Variable entities with full lineage.

Usage:
    from fastapi import FastAPI
    from merchants.offline.sensor_api import create_sensor_router
    from graph.store import KnowledgeGraphStore

    app = FastAPI()
    store = KnowledgeGraphStore()
    router = create_sensor_router(store)
    app.include_router(router, prefix="/api/sensors")
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from graph.entities import DataSource, DataSourceType, Variable, VariableType
from graph.store import KnowledgeGraphStore

from .schemas import (
    EnvironmentMeasurement,
    InterventionBatch,
    InterventionLog,
    SensorBatch,
    SensorReading,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class IngestionResponse(BaseModel):
    """Response returned after successful data ingestion."""

    success: bool = Field(default=True, description="Whether ingestion succeeded.")
    records_ingested: int = Field(
        ..., description="Number of records successfully ingested."
    )
    variables_created: list[str] = Field(
        default_factory=list,
        description="IDs of Variable entities created during ingestion.",
    )
    message: str = Field(
        default="Data ingested successfully.",
        description="Human-readable status message.",
    )


class InterventionResponse(BaseModel):
    """Response returned after intervention log ingestion."""

    success: bool = Field(default=True, description="Whether ingestion succeeded.")
    intervention_id: str = Field(
        ..., description="ID of the ingested intervention log."
    )
    variables_created: list[str] = Field(
        default_factory=list,
        description="IDs of Variable entities created.",
    )
    message: str = Field(
        default="Intervention logged successfully.",
        description="Human-readable status message.",
    )


class SensorStatusResponse(BaseModel):
    """Status response for sensor data ingestion subsystem."""

    status: str = Field(..., description="Current status of the sensor API.")
    registered_stations: int = Field(
        ..., description="Number of registered measurement stations."
    )
    total_readings_ingested: int = Field(
        ..., description="Total sensor readings ingested since startup."
    )
    total_interventions_logged: int = Field(
        ..., description="Total intervention logs ingested since startup."
    )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_sensor_router(store: KnowledgeGraphStore) -> APIRouter:
    """Create a FastAPI router for sensor data ingestion.

    Factory function that creates router endpoints bound to a specific
    KnowledgeGraphStore instance.

    Args:
        store: The knowledge graph store to ingest data into.

    Returns:
        Configured APIRouter with sensor data endpoints.
    """
    router = APIRouter(tags=["sensors"])

    # Ingestion counters
    _state = {
        "total_readings": 0,
        "total_interventions": 0,
        "stations": set(),
    }

    # ------------------------------------------------------------------
    # Health / status
    # ------------------------------------------------------------------

    @router.get("/status", response_model=SensorStatusResponse)
    async def sensor_status() -> SensorStatusResponse:
        """Return the status of the sensor data ingestion subsystem.

        Returns:
            Current status with ingestion counters.
        """
        return SensorStatusResponse(
            status="operational",
            registered_stations=len(_state["stations"]),
            total_readings_ingested=_state["total_readings"],
            total_interventions_logged=_state["total_interventions"],
        )

    # ------------------------------------------------------------------
    # Single sensor reading
    # ------------------------------------------------------------------

    @router.post("/readings", response_model=IngestionResponse)
    async def ingest_reading(reading: SensorReading) -> IngestionResponse:
        """Ingest a single sensor reading into the knowledge graph.

        Creates a Variable entity for the sensor measurement and registers
        it in the knowledge graph store.

        Args:
            reading: The sensor reading to ingest.

        Returns:
            Ingestion response with created variable IDs.
        """
        try:
            var_name = f"sensor:{reading.sensor_id}:{reading.sensor_type.value}"
            variable = Variable(
                name=var_name,
                type=VariableType.OBSERVABLE,
                data_sources=[reading.sensor_id],
            )

            try:
                variable = store.add_variable(variable)
            except ValueError:
                # Variable already exists — find it
                for existing in store.variables.values():
                    if existing.name == var_name:
                        variable = existing
                        break

            _state["total_readings"] += 1
            _state["stations"].add(reading.sensor_id)

            logger.info(
                "Ingested sensor reading: sensor=%s, type=%s, value=%.4f %s",
                reading.sensor_id,
                reading.sensor_type.value,
                reading.value,
                reading.unit.value,
            )

            return IngestionResponse(
                records_ingested=1,
                variables_created=[variable.id],
                message=f"Reading from sensor {reading.sensor_id} ingested.",
            )

        except Exception as exc:
            logger.error("Failed to ingest sensor reading: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Batch sensor readings
    # ------------------------------------------------------------------

    @router.post("/readings/batch", response_model=IngestionResponse)
    async def ingest_reading_batch(batch: SensorBatch) -> IngestionResponse:
        """Ingest a batch of sensor readings.

        Accepts multiple readings in a single request for efficiency.
        Each reading creates or updates a Variable entity in the graph.

        Args:
            batch: Batch of sensor readings from a single station.

        Returns:
            Ingestion response with total records and variable IDs.
        """
        try:
            variables_created: list[str] = []
            ingested_count = 0

            for reading in batch.readings:
                var_name = f"sensor:{reading.sensor_id}:{reading.sensor_type.value}"
                variable = Variable(
                    name=var_name,
                    type=VariableType.OBSERVABLE,
                    data_sources=[batch.station_id],
                )

                try:
                    variable = store.add_variable(variable)
                except ValueError:
                    for existing in store.variables.values():
                        if existing.name == var_name:
                            variable = existing
                            break

                variables_created.append(variable.id)
                ingested_count += 1

            _state["total_readings"] += ingested_count
            _state["stations"].add(batch.station_id)

            logger.info(
                "Ingested batch: station=%s, readings=%d",
                batch.station_id,
                ingested_count,
            )

            return IngestionResponse(
                records_ingested=ingested_count,
                variables_created=list(set(variables_created)),
                message=(
                    f"Batch of {ingested_count} readings from station "
                    f"{batch.station_id} ingested."
                ),
            )

        except Exception as exc:
            logger.error("Failed to ingest sensor batch: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Batch ingestion failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Environment measurement
    # ------------------------------------------------------------------

    @router.post("/environment", response_model=IngestionResponse)
    async def ingest_environment(
        measurement: EnvironmentMeasurement,
    ) -> IngestionResponse:
        """Ingest a composite environmental measurement.

        Accepts a multi-sensor snapshot (temperature + humidity + pressure,
        etc.) taken at a single location and time.

        Args:
            measurement: Composite environmental measurement.

        Returns:
            Ingestion response with created variable IDs.
        """
        try:
            variables_created: list[str] = []
            ingested_count = 0

            for reading in measurement.readings:
                var_name = (
                    f"env:{measurement.station_id}"
                    f":{reading.sensor_type.value}"
                )
                variable = Variable(
                    name=var_name,
                    type=VariableType.OBSERVABLE,
                    data_sources=[measurement.station_id],
                )

                try:
                    variable = store.add_variable(variable)
                except ValueError:
                    for existing in store.variables.values():
                        if existing.name == var_name:
                            variable = existing
                            break

                variables_created.append(variable.id)
                ingested_count += 1

            _state["total_readings"] += ingested_count
            _state["stations"].add(measurement.station_id)

            logger.info(
                "Ingested environment measurement: station=%s, "
                "location=(%.4f, %.4f), readings=%d",
                measurement.station_id,
                measurement.location.latitude,
                measurement.location.longitude,
                ingested_count,
            )

            return IngestionResponse(
                records_ingested=ingested_count,
                variables_created=list(set(variables_created)),
                message=(
                    f"Environment measurement from station "
                    f"{measurement.station_id} ingested "
                    f"({ingested_count} readings)."
                ),
            )

        except Exception as exc:
            logger.error("Failed to ingest environment measurement: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Environment ingestion failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Intervention logs — Pearl's Level 2
    # ------------------------------------------------------------------

    @router.post("/interventions", response_model=InterventionResponse)
    async def log_intervention(
        intervention: InterventionLog,
    ) -> InterventionResponse:
        """Log a physical-world intervention.

        This is the core Pearl Level 2 endpoint. An embodied agent reports
        an intervention it performed (do(X=x)) and the observed outcome.
        This data supports causal inference via the do-calculus.

        Args:
            intervention: The intervention log entry.

        Returns:
            Intervention response with created variable IDs.
        """
        try:
            variables_created: list[str] = []

            # Create a variable for the intervention target
            target_var_name = (
                f"intervention:{intervention.agent_id}"
                f":{intervention.target_variable}"
            )
            target_variable = Variable(
                name=target_var_name,
                type=VariableType.INTERVENTION,
                data_sources=[intervention.agent_id],
            )

            try:
                target_variable = store.add_variable(target_variable)
            except ValueError:
                for existing in store.variables.values():
                    if existing.name == target_var_name:
                        target_variable = existing
                        break

            variables_created.append(target_variable.id)

            # Create variables for pre- and post-intervention readings
            for prefix, readings in [
                ("pre", intervention.pre_intervention_readings),
                ("post", intervention.post_intervention_readings),
            ]:
                for reading in readings:
                    var_name = (
                        f"intervention:{intervention.agent_id}"
                        f":{prefix}:{reading.sensor_type.value}"
                    )
                    variable = Variable(
                        name=var_name,
                        type=VariableType.OBSERVABLE,
                        data_sources=[intervention.agent_id],
                    )

                    try:
                        variable = store.add_variable(variable)
                    except ValueError:
                        for existing in store.variables.values():
                            if existing.name == var_name:
                                variable = existing
                                break

                    variables_created.append(variable.id)

            _state["total_interventions"] += 1

            logger.info(
                "Logged intervention: agent=%s, type=%s, target=%s, "
                "success=%s, pre_readings=%d, post_readings=%d",
                intervention.agent_id,
                intervention.intervention_type.value,
                intervention.target_variable,
                intervention.success,
                len(intervention.pre_intervention_readings),
                len(intervention.post_intervention_readings),
            )

            return InterventionResponse(
                intervention_id=intervention.id,
                variables_created=list(set(variables_created)),
                message=(
                    f"Intervention by agent {intervention.agent_id} logged "
                    f"(type={intervention.intervention_type.value}, "
                    f"target={intervention.target_variable})."
                ),
            )

        except Exception as exc:
            logger.error("Failed to log intervention: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Intervention logging failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Batch intervention logs
    # ------------------------------------------------------------------

    @router.post("/interventions/batch", response_model=IngestionResponse)
    async def log_intervention_batch(
        batch: InterventionBatch,
    ) -> IngestionResponse:
        """Log a batch of interventions from a single agent.

        Accepts multiple intervention logs in a single request.

        Args:
            batch: Batch of intervention logs.

        Returns:
            Ingestion response with total records.
        """
        try:
            all_variables: list[str] = []
            ingested_count = 0

            for intervention in batch.interventions:
                target_var_name = (
                    f"intervention:{intervention.agent_id}"
                    f":{intervention.target_variable}"
                )
                target_variable = Variable(
                    name=target_var_name,
                    type=VariableType.INTERVENTION,
                    data_sources=[batch.agent_id],
                )

                try:
                    target_variable = store.add_variable(target_variable)
                except ValueError:
                    for existing in store.variables.values():
                        if existing.name == target_var_name:
                            target_variable = existing
                            break

                all_variables.append(target_variable.id)
                ingested_count += 1

            _state["total_interventions"] += ingested_count

            logger.info(
                "Ingested intervention batch: agent=%s, interventions=%d",
                batch.agent_id,
                ingested_count,
            )

            return IngestionResponse(
                records_ingested=ingested_count,
                variables_created=list(set(all_variables)),
                message=(
                    f"Batch of {ingested_count} interventions from agent "
                    f"{batch.agent_id} ingested."
                ),
            )

        except Exception as exc:
            logger.error("Failed to ingest intervention batch: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Intervention batch ingestion failed: {exc}",
            )

    return router
