"""
Abstract Merchant Agent — the foundational contract for all data-gathering agents.

Every merchant agent in the Republic follows the same pipeline:

    discover() -> collect() -> validate() -> ingest()

- discover(): scan the source for available data (new markets, new datasets, etc.)
- collect(): pull raw data from the source
- validate(): check data integrity, schema conformance, freshness
- ingest(): transform validated data into Variable / DataSource entities and
  push them into the KnowledgeGraphStore

The base class provides:
- Rate limiting with configurable intervals to be a responsible API citizen
- Exponential backoff on errors with configurable retry limits
- Health check / status reporting for the MerchantRegistry
- Provenance metadata on all collected data (source, timestamp, confidence,
  collection method) — because data without provenance is rumor, not evidence
- Lifecycle management (start/stop/restart)

Usage:
    class MyMerchant(MerchantAgent):
        async def discover(self) -> list[str]:
            ...
        async def collect(self, targets: list[str]) -> list[CollectionResult]:
            ...
        async def validate(self, results: list[CollectionResult]) -> list[CollectionResult]:
            ...
        async def ingest(self, results: list[CollectionResult]) -> int:
            ...
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from graph.entities import DataSource, DataSourceType, Variable, VariableType
from graph.store import KnowledgeGraphStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MerchantStatus(str, Enum):
    """Lifecycle status of a merchant agent.

    Tracks where the agent is in its operational lifecycle, from
    initial registration through active data collection to shutdown.
    """

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class PearlLevel(str, Enum):
    """Pearl's causal hierarchy level for data classification.

    - ASSOCIATION (Level 1): observational data — online merchants watch and record
    - INTERVENTION (Level 2): interventional data — offline/embodied merchants act
      and measure the consequences
    """

    ASSOCIATION = "association"
    INTERVENTION = "intervention"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class MerchantConfig(BaseModel):
    """Configuration for a merchant agent.

    Controls polling behavior, rate limiting, retry logic, and provenance
    metadata. Each merchant instance carries its own config, allowing
    fine-grained tuning per data source.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable name for this merchant agent.",
    )
    source_type: DataSourceType = Field(
        ...,
        description="Category of data this merchant collects.",
    )
    pearl_level: PearlLevel = Field(
        default=PearlLevel.ASSOCIATION,
        description="Pearl's causal hierarchy level: association (Level 1) or intervention (Level 2).",
    )
    poll_interval_seconds: float = Field(
        default=300.0,
        ge=1.0,
        description="Seconds between polling cycles. Minimum 1 second.",
    )
    rate_limit_requests_per_minute: float = Field(
        default=60.0,
        ge=1.0,
        description="Maximum requests per minute to the data source.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts on transient failures.",
    )
    base_backoff_seconds: float = Field(
        default=1.0,
        ge=0.1,
        description="Base delay for exponential backoff between retries.",
    )
    max_backoff_seconds: float = Field(
        default=60.0,
        ge=1.0,
        description="Maximum backoff delay cap.",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this merchant is enabled for scheduling.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata for source-specific configuration.",
    )
    era_aware: bool = Field(
        default=True,
        description="Whether this merchant checks era parameters before collecting.",
    )


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ProvenanceMetadata(BaseModel):
    """Provenance metadata attached to every piece of collected data.

    Without provenance, data is rumor. This model ensures every data point
    carries a full chain of custody: who collected it, when, how, and
    with what confidence.

    The era_id field links every piece of data to the governance context
    in which it was collected — inspired by OpenForage's era system where
    all agent contributions are timestamped within an era's parameters.
    """

    source_name: str = Field(
        ..., description="Name of the data source."
    )
    source_url: Optional[str] = Field(
        default=None, description="URL or endpoint the data was collected from."
    )
    merchant_id: str = Field(
        ..., description="ID of the merchant agent that collected this data."
    )
    collected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of data collection.",
    )
    collection_method: str = Field(
        default="api_poll",
        description="How the data was collected (api_poll, websocket, rss, sensor, etc.).",
    )
    pearl_level: PearlLevel = Field(
        default=PearlLevel.ASSOCIATION,
        description="Pearl's causal hierarchy level of this data.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in data quality, from 0 (unreliable) to 1 (verified).",
    )
    raw_record_count: int = Field(
        default=0,
        ge=0,
        description="Number of raw records collected in this batch.",
    )
    era_id: Optional[str] = Field(
        default=None,
        description="ID of the era in which this data was collected. Links data to its governance context.",
    )


class CollectionResult(BaseModel):
    """Result of a single collection cycle.

    Encapsulates raw data alongside its provenance metadata, ready for
    validation and ingestion into the knowledge graph.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this collection result.",
    )
    target: str = Field(
        ..., description="The discovery target this data was collected for."
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw collected data as key-value pairs.",
    )
    provenance: ProvenanceMetadata = Field(
        ..., description="Full provenance metadata for this data."
    )
    is_valid: bool = Field(
        default=False,
        description="Whether this result passed validation.",
    )
    validation_errors: list[str] = Field(
        default_factory=list,
        description="List of validation error messages, if any.",
    )
    variables_created: list[str] = Field(
        default_factory=list,
        description="IDs of Variable entities created from this result during ingestion.",
    )


# ---------------------------------------------------------------------------
# Health snapshot
# ---------------------------------------------------------------------------


class MerchantHealthSnapshot(BaseModel):
    """Point-in-time health snapshot of a merchant agent.

    Used by the MerchantRegistry for monitoring and alerting.
    """

    merchant_id: str
    name: str
    status: MerchantStatus
    last_heartbeat: Optional[datetime] = None
    last_successful_collection: Optional[datetime] = None
    consecutive_errors: int = 0
    total_collections: int = 0
    total_records_ingested: int = 0
    total_errors: int = 0
    uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class MerchantAgent(ABC):
    """Abstract base class for all merchant agents in the Republic.

    Implements the standard merchant pipeline (discover -> collect -> validate
    -> ingest) with built-in rate limiting, exponential backoff, health
    reporting, and provenance tracking.

    Subclasses must implement the four abstract pipeline methods. The base
    class handles lifecycle management, error recovery, and integration
    with the KnowledgeGraphStore.

    Attributes:
        id: Unique identifier for this merchant instance.
        config: The merchant's configuration.
        store: Reference to the knowledge graph store for data ingestion.
        status: Current lifecycle status.
    """

    def __init__(
        self,
        config: MerchantConfig,
        store: KnowledgeGraphStore,
    ) -> None:
        """Initialize a merchant agent.

        Args:
            config: Configuration controlling polling, rate limiting, and retries.
            store: The knowledge graph store to ingest data into.
        """
        self.id: str = str(uuid.uuid4())
        self.config: MerchantConfig = config
        self.store: KnowledgeGraphStore = store
        self.status: MerchantStatus = MerchantStatus.IDLE

        # Internal state
        self._started_at: Optional[datetime] = None
        self._last_heartbeat: Optional[datetime] = None
        self._last_successful_collection: Optional[datetime] = None
        self._consecutive_errors: int = 0
        self._total_collections: int = 0
        self._total_records_ingested: int = 0
        self._total_errors: int = 0
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

        # Rate limiting state
        self._request_timestamps: list[float] = []

    # ------------------------------------------------------------------
    # Abstract pipeline methods — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    async def discover(self) -> list[str]:
        """Discover available data targets at the source.

        Scan the data source for items to collect: active markets, new
        datasets, recent articles, available sensors, etc.

        Returns:
            List of target identifiers (market IDs, dataset names, URLs, etc.)
            that should be collected in the next cycle.
        """
        ...

    @abstractmethod
    async def collect(self, targets: list[str]) -> list[CollectionResult]:
        """Collect raw data for the given targets.

        Pull data from the source for each target. Each result carries
        provenance metadata via the ``_build_provenance`` helper.

        Args:
            targets: List of target identifiers from ``discover()``.

        Returns:
            List of CollectionResult objects containing raw data and provenance.
        """
        ...

    @abstractmethod
    async def validate(self, results: list[CollectionResult]) -> list[CollectionResult]:
        """Validate collected data for integrity and schema conformance.

        Check that each result contains the expected fields, values are
        within reasonable ranges, timestamps are fresh, etc. Mark each
        result's ``is_valid`` flag and populate ``validation_errors``.

        Args:
            results: Collection results to validate.

        Returns:
            The same results with ``is_valid`` and ``validation_errors`` updated.
        """
        ...

    @abstractmethod
    async def ingest(self, results: list[CollectionResult]) -> int:
        """Ingest validated data into the knowledge graph.

        Transform valid CollectionResults into Variable and DataSource
        entities and register them in the KnowledgeGraphStore.

        Args:
            results: Validated collection results (only those with ``is_valid=True``
                     should be ingested).

        Returns:
            Number of records successfully ingested.
        """
        ...

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the merchant agent's polling loop.

        Transitions to RUNNING status and begins the discover-collect-
        validate-ingest cycle at the configured poll interval.
        """
        if self._running:
            logger.warning("Merchant '%s' is already running.", self.config.name)
            return

        self._running = True
        self.status = MerchantStatus.RUNNING
        self._started_at = datetime.now(timezone.utc)
        logger.info(
            "Starting merchant '%s' (id=%s, poll_interval=%.1fs)",
            self.config.name,
            self.id,
            self.config.poll_interval_seconds,
        )
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the merchant agent gracefully.

        Signals the polling loop to exit and waits for the current cycle
        to complete before transitioning to STOPPED status.
        """
        if not self._running:
            logger.warning("Merchant '%s' is not running.", self.config.name)
            return

        self._running = False
        self.status = MerchantStatus.STOPPED
        logger.info("Stopping merchant '%s' (id=%s)", self.config.name, self.id)

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def restart(self) -> None:
        """Restart the merchant agent.

        Stops and then starts the agent, resetting error counters.
        """
        await self.stop()
        self._consecutive_errors = 0
        await self.start()

    def pause(self) -> None:
        """Pause the merchant agent.

        The polling loop continues running but skips collection cycles
        until resumed.
        """
        self.status = MerchantStatus.PAUSED
        logger.info("Paused merchant '%s' (id=%s)", self.config.name, self.id)

    def resume(self) -> None:
        """Resume a paused merchant agent."""
        if self.status == MerchantStatus.PAUSED:
            self.status = MerchantStatus.RUNNING
            logger.info("Resumed merchant '%s' (id=%s)", self.config.name, self.id)

    # ------------------------------------------------------------------
    # Health reporting
    # ------------------------------------------------------------------

    def health(self) -> MerchantHealthSnapshot:
        """Return a point-in-time health snapshot.

        Used by the MerchantRegistry for monitoring and alerting.

        Returns:
            MerchantHealthSnapshot with current agent metrics.
        """
        uptime = 0.0
        if self._started_at is not None:
            uptime = (datetime.now(timezone.utc) - self._started_at).total_seconds()

        return MerchantHealthSnapshot(
            merchant_id=self.id,
            name=self.config.name,
            status=self.status,
            last_heartbeat=self._last_heartbeat,
            last_successful_collection=self._last_successful_collection,
            consecutive_errors=self._consecutive_errors,
            total_collections=self._total_collections,
            total_records_ingested=self._total_records_ingested,
            total_errors=self._total_errors,
            uptime_seconds=uptime,
        )

    def heartbeat(self) -> None:
        """Record a heartbeat timestamp for liveness monitoring."""
        self._last_heartbeat = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Pipeline execution with error handling
    # ------------------------------------------------------------------

    async def run_cycle(self) -> int:
        """Execute one full discover-collect-validate-ingest cycle.

        This is the core pipeline method. It handles rate limiting,
        error recovery with exponential backoff, and metric tracking.

        Returns:
            Number of records ingested in this cycle.

        Raises:
            Exception: Re-raises after max retries exhausted.
        """
        self.heartbeat()

        attempt = 0
        last_error: Optional[Exception] = None

        while attempt <= self.config.max_retries:
            try:
                # Phase 1: Discover
                targets = await self.discover()
                if not targets:
                    logger.debug(
                        "Merchant '%s': no targets discovered, skipping cycle.",
                        self.config.name,
                    )
                    return 0

                # Phase 2: Collect (with rate limiting)
                await self._enforce_rate_limit()
                results = await self.collect(targets)

                # Phase 3: Validate
                validated = await self.validate(results)
                valid_results = [r for r in validated if r.is_valid]

                if not valid_results:
                    logger.debug(
                        "Merchant '%s': no valid results after validation.",
                        self.config.name,
                    )
                    return 0

                # Phase 4: Ingest
                ingested = await self.ingest(valid_results)

                # Success — update metrics
                self._total_collections += 1
                self._total_records_ingested += ingested
                self._consecutive_errors = 0
                self._last_successful_collection = datetime.now(timezone.utc)
                logger.info(
                    "Merchant '%s': cycle complete — %d records ingested from %d targets.",
                    self.config.name,
                    ingested,
                    len(targets),
                )
                return ingested

            except Exception as exc:
                attempt += 1
                last_error = exc
                self._total_errors += 1
                self._consecutive_errors += 1

                if attempt > self.config.max_retries:
                    logger.error(
                        "Merchant '%s': max retries (%d) exhausted. Last error: %s",
                        self.config.name,
                        self.config.max_retries,
                        exc,
                    )
                    self.status = MerchantStatus.ERROR
                    raise

                backoff = min(
                    self.config.base_backoff_seconds * (2 ** (attempt - 1)),
                    self.config.max_backoff_seconds,
                )
                logger.warning(
                    "Merchant '%s': attempt %d/%d failed (%s). "
                    "Retrying in %.1fs.",
                    self.config.name,
                    attempt,
                    self.config.max_retries,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)

        # Should not reach here, but satisfy type checker
        if last_error is not None:
            raise last_error
        return 0

    # ------------------------------------------------------------------
    # Helper methods for subclasses
    # ------------------------------------------------------------------

    def _build_provenance(
        self,
        source_url: Optional[str] = None,
        collection_method: str = "api_poll",
        confidence: float = 0.5,
        raw_record_count: int = 0,
        era_id: Optional[str] = None,
    ) -> ProvenanceMetadata:
        """Build a ProvenanceMetadata object for a collection result.

        Convenience method for subclasses to create standardized provenance
        records with the merchant's identity and era context pre-filled.

        Args:
            source_url: URL or endpoint the data was collected from.
            collection_method: How the data was collected.
            confidence: Confidence score for data quality.
            raw_record_count: Number of raw records in this batch.
            era_id: ID of the current era (links data to its governance context).

        Returns:
            Fully populated ProvenanceMetadata.
        """
        return ProvenanceMetadata(
            source_name=self.config.name,
            source_url=source_url,
            merchant_id=self.id,
            collection_method=collection_method,
            pearl_level=self.config.pearl_level,
            confidence=confidence,
            raw_record_count=raw_record_count,
            era_id=era_id,
        )

    def _create_variable(
        self,
        name: str,
        variable_type: VariableType = VariableType.OBSERVABLE,
        embedding: Optional[list[float]] = None,
    ) -> Variable:
        """Create and register a Variable entity in the knowledge graph.

        Convenience method for subclasses to create variables with the
        merchant's data source pre-linked.

        Args:
            name: Human-readable variable name.
            variable_type: Observable, latent, or intervention.
            embedding: Optional semantic embedding vector.

        Returns:
            The registered Variable entity.
        """
        variable = Variable(
            name=name,
            type=variable_type,
            embedding=embedding,
            data_sources=[self.id],
        )
        try:
            return self.store.add_variable(variable)
        except ValueError:
            # Variable already exists — look it up by iterating
            for existing in self.store.variables.values():
                if existing.name == name and self.id in existing.data_sources:
                    return existing
            # Name matches but different source — add our source ID
            for existing in self.store.variables.values():
                if existing.name == name:
                    if self.id not in existing.data_sources:
                        existing.data_sources.append(self.id)
                    return existing
            # Fallback: return what we built (ID collision, extremely unlikely)
            return variable

    def _register_data_source(
        self,
        name: str,
        url: Optional[str] = None,
        schema_info: Optional[dict] = None,
    ) -> DataSource:
        """Create and register a DataSource entity in the knowledge graph.

        Called during initialization to register this merchant's data source
        in the knowledge graph for provenance tracking.

        Args:
            name: Human-readable data source name.
            url: URL or endpoint for the data source.
            schema_info: JSON schema or description of the data format.

        Returns:
            The registered DataSource entity.
        """
        data_source = DataSource(
            name=name,
            type=self.config.source_type,
            url=url,
            merchant_id=self.id,
            schema_info=schema_info,
        )
        return data_source

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Internal polling loop that runs discover-collect-validate-ingest cycles.

        Executes ``run_cycle()`` at the configured poll interval, skipping
        cycles when paused.
        """
        logger.info("Merchant '%s': polling loop started.", self.config.name)

        try:
            while self._running:
                if self.status == MerchantStatus.PAUSED:
                    await asyncio.sleep(1.0)
                    continue

                try:
                    await self.run_cycle()
                except Exception as exc:
                    logger.error(
                        "Merchant '%s': unrecoverable error in cycle: %s",
                        self.config.name,
                        exc,
                    )
                    # Don't crash the loop — stay in ERROR status and retry
                    # next interval
                    self.status = MerchantStatus.ERROR

                await asyncio.sleep(self.config.poll_interval_seconds)

        except asyncio.CancelledError:
            logger.info("Merchant '%s': polling loop cancelled.", self.config.name)
            raise

        logger.info("Merchant '%s': polling loop exited.", self.config.name)

    async def _enforce_rate_limit(self) -> None:
        """Enforce the configured rate limit by sleeping if necessary.

        Uses a sliding window approach: tracks timestamps of recent requests
        and sleeps if the window is full.
        """
        now = asyncio.get_event_loop().time()
        window_seconds = 60.0
        max_requests = self.config.rate_limit_requests_per_minute

        # Prune timestamps outside the window
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if now - ts < window_seconds
        ]

        if len(self._request_timestamps) >= max_requests:
            # Calculate how long to wait for the oldest request to leave the window
            oldest = self._request_timestamps[0]
            sleep_time = window_seconds - (now - oldest) + 0.1
            if sleep_time > 0:
                logger.debug(
                    "Merchant '%s': rate limit reached, sleeping %.1fs.",
                    self.config.name,
                    sleep_time,
                )
                await asyncio.sleep(sleep_time)

        self._request_timestamps.append(asyncio.get_event_loop().time())
