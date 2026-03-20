"""
Merchant Scheduler — async orchestration of merchant agent polling loops.

Manages the coordinated execution of multiple merchant agents, each with
its own polling interval and lifecycle. Uses asyncio for concurrent
scheduling with graceful shutdown support.

The scheduler works hand-in-hand with the MerchantRegistry:
- Registry handles registration, status tracking, and health monitoring
- Scheduler handles the actual async execution and coordination

Usage:
    registry = MerchantRegistry.instance()
    registry.register(polymarket_merchant)
    registry.register(financial_merchant)

    scheduler = MerchantScheduler(registry)
    await scheduler.start()
    # ... merchants run concurrently ...
    await scheduler.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .base import MerchantStatus
from .registry import MerchantRegistry

from eras import EraManager

logger = logging.getLogger(__name__)


class MerchantScheduler:
    """Async scheduler for coordinating merchant agent polling loops.

    Starts all enabled merchants in the registry, monitors their health,
    and provides graceful shutdown. Optionally runs a periodic health
    check that logs warnings for unhealthy merchants.

    Attributes:
        registry: The MerchantRegistry containing agents to schedule.
        health_check_interval: Seconds between health check sweeps.
    """

    def __init__(
        self,
        registry: Optional[MerchantRegistry] = None,
        era_manager: Optional[EraManager] = None,
        health_check_interval: float = 60.0,
    ) -> None:
        """Initialize the scheduler.

        Args:
            registry: The MerchantRegistry to schedule agents from.
                     If None, uses the singleton instance.
            era_manager: Optional EraManager for era-aware scheduling.
                When provided, the scheduler synchronizes merchants to era
                transitions — inspired by OpenForage's era system where all
                agents must sync to new parameters on era change.
            health_check_interval: Seconds between periodic health checks.
                                  Set to 0 to disable health monitoring.
        """
        self.registry: MerchantRegistry = registry or MerchantRegistry.instance()
        self.era_manager: Optional[EraManager] = era_manager
        self.health_check_interval: float = health_check_interval
        self._running: bool = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_era_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the scheduler: launch all enabled merchants and begin health monitoring.

        Starts all registered and enabled merchant agents via the registry,
        then optionally begins a periodic health check loop.
        """
        if self._running:
            logger.warning("Scheduler is already running.")
            return

        self._running = True
        logger.info(
            "Starting MerchantScheduler with %d registered merchants.",
            self.registry.count,
        )

        # Start all enabled merchants
        await self.registry.start_all()

        # Start health monitoring if configured
        if self.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )

        logger.info("MerchantScheduler started successfully.")

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Gracefully shut down the scheduler and all running merchants.

        Stops all merchant agents and the health check loop, waiting up
        to ``timeout`` seconds for graceful completion.

        Args:
            timeout: Maximum seconds to wait for graceful shutdown.
        """
        if not self._running:
            logger.warning("Scheduler is not running.")
            return

        logger.info("Shutting down MerchantScheduler (timeout=%.1fs)...", timeout)
        self._running = False

        # Cancel health check
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await asyncio.wait_for(self._health_check_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._health_check_task = None

        # Stop all merchants
        try:
            await asyncio.wait_for(
                self.registry.stop_all(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Shutdown timed out after %.1fs — some merchants may not have stopped cleanly.",
                timeout,
            )

        logger.info("MerchantScheduler shut down.")

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is currently running."""
        return self._running

    # ------------------------------------------------------------------
    # Dynamic agent management
    # ------------------------------------------------------------------

    async def add_and_start(self, merchant_id: str) -> None:
        """Start a specific merchant that was registered after scheduler startup.

        Allows hot-adding merchants to a running scheduler without
        restarting the entire system.

        Args:
            merchant_id: The ID of the already-registered merchant to start.

        Raises:
            KeyError: If the merchant is not registered.
        """
        await self.registry.start_merchant(merchant_id)
        logger.info("Dynamically started merchant '%s'.", merchant_id)

    async def remove_and_stop(self, merchant_id: str) -> None:
        """Stop and unregister a merchant from the running scheduler.

        Args:
            merchant_id: The ID of the merchant to remove.

        Raises:
            KeyError: If the merchant is not registered.
        """
        await self.registry.stop_merchant(merchant_id)
        self.registry.unregister(merchant_id)
        logger.info("Dynamically removed merchant '%s'.", merchant_id)

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    def status_summary(self) -> dict[str, int]:
        """Return a summary of merchant statuses.

        Returns:
            Dictionary mapping MerchantStatus values to counts of
            merchants in each status.
        """
        summary: dict[str, int] = {}
        for snapshot in self.registry.list_merchants():
            status_key = snapshot.status.value
            summary[status_key] = summary.get(status_key, 0) + 1
        return summary

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    async def _health_check_loop(self) -> None:
        """Periodic health check loop that logs warnings for unhealthy merchants.

        Runs every ``health_check_interval`` seconds, scanning all merchants
        for error conditions. Automatically attempts to restart merchants
        that have been in ERROR status for too long.
        """
        logger.info(
            "Health check loop started (interval=%.1fs).",
            self.health_check_interval,
        )

        try:
            while self._running:
                await asyncio.sleep(self.health_check_interval)

                if not self._running:
                    break

                # Check for era transitions — OpenForage-inspired synchronization.
                # When the era changes, all merchants must acknowledge the new
                # paradigm parameters before their next collection cycle.
                await self._check_era_transition()

                unhealthy = self.registry.get_unhealthy(max_consecutive_errors=5)
                if unhealthy:
                    for snapshot in unhealthy:
                        logger.warning(
                            "Unhealthy merchant: '%s' (id=%s, status=%s, "
                            "consecutive_errors=%d, total_errors=%d)",
                            snapshot.name,
                            snapshot.merchant_id,
                            snapshot.status.value,
                            snapshot.consecutive_errors,
                            snapshot.total_errors,
                        )

                        # Attempt auto-restart for ERROR status merchants
                        if snapshot.status == MerchantStatus.ERROR:
                            try:
                                await self.registry.restart_merchant(
                                    snapshot.merchant_id
                                )
                                logger.info(
                                    "Auto-restarted unhealthy merchant '%s' (id=%s).",
                                    snapshot.name,
                                    snapshot.merchant_id,
                                )
                            except Exception as exc:
                                logger.error(
                                    "Failed to auto-restart merchant '%s': %s",
                                    snapshot.name,
                                    exc,
                                )
                else:
                    running = [
                        s for s in self.registry.list_merchants()
                        if s.status == MerchantStatus.RUNNING
                    ]
                    logger.debug(
                        "Health check OK: %d merchants running, 0 unhealthy.",
                        len(running),
                    )

        except asyncio.CancelledError:
            logger.info("Health check loop cancelled.")
            raise

        logger.info("Health check loop exited.")

    async def _check_era_transition(self) -> None:
        """Detect era transitions and synchronize merchants.

        When the EraManager reports a new active era, the scheduler logs
        the transition and records a contribution for each running merchant
        to acknowledge the new parameters. This mirrors OpenForage's era
        system where all agents must sync to updated evaluation criteria,
        valid data sources, and signal thresholds on era change.
        """
        if self.era_manager is None:
            return

        current_era = self.era_manager.current_era
        if current_era is None:
            return

        current_era_id = current_era.id
        if current_era_id == self._last_era_id:
            return

        # Era has changed — synchronize
        old_era_id = self._last_era_id
        self._last_era_id = current_era_id

        running_merchants = [
            s for s in self.registry.list_merchants()
            if s.status == MerchantStatus.RUNNING
        ]

        logger.info(
            "ERA TRANSITION detected: %s → %s (era #%d '%s'). "
            "Synchronizing %d running merchants to new parameters.",
            old_era_id or "genesis",
            current_era_id,
            current_era.number,
            current_era.name,
            len(running_merchants),
        )

        # Log valid data sources for the new era
        valid_sources = current_era.config.valid_data_sources
        if valid_sources:
            logger.info(
                "New era valid data sources: %s",
                ", ".join(valid_sources),
            )
        logger.info(
            "New era parameters: min_evidence=%d, min_support_ratio=%.2f, "
            "min_independent_sources=%d, signal_correlation_threshold=%.2f",
            current_era.config.min_evidence_count,
            current_era.config.min_supporting_ratio,
            current_era.config.min_independent_sources,
            current_era.config.signal_correlation_threshold,
        )
