"""
Merchant Registry — agent registration, lifecycle management, and health monitoring.

The MerchantRegistry is the central authority that tracks all merchant agents
in the Republic. It handles:

- Registration / deregistration of merchant agents
- Start / stop / restart lifecycle operations
- Health monitoring with heartbeat tracking, error counts, and data volume
- Listing all active merchants with their current status

Implemented as a singleton to ensure a single source of truth for merchant
state across the application.

Usage:
    registry = MerchantRegistry.instance()
    registry.register(my_merchant)
    await registry.start_merchant(my_merchant.id)
    health = registry.get_health(my_merchant.id)
    all_agents = registry.list_merchants()
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .base import MerchantAgent, MerchantHealthSnapshot, MerchantStatus

logger = logging.getLogger(__name__)


class MerchantRegistry:
    """Singleton registry managing all merchant agent lifecycles.

    Provides centralized control over merchant agents: registration,
    lifecycle transitions (start/stop/pause/resume), and health monitoring.
    The singleton pattern ensures a single source of truth across the
    application.

    Attributes:
        _merchants: Internal mapping of merchant ID to MerchantAgent instance.
        _registered_at: Mapping of merchant ID to registration timestamp.
    """

    _instance: Optional["MerchantRegistry"] = None

    def __init__(self) -> None:
        """Initialize an empty registry.

        Use ``MerchantRegistry.instance()`` for singleton access.
        """
        self._merchants: dict[str, MerchantAgent] = {}
        self._registered_at: dict[str, datetime] = {}

    @classmethod
    def instance(cls) -> "MerchantRegistry":
        """Return the singleton MerchantRegistry instance.

        Creates the instance on first call, returns the same instance
        on subsequent calls.

        Returns:
            The global MerchantRegistry singleton.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance.

        Primarily for testing — clears all registered merchants and
        allows a fresh registry to be created.
        """
        cls._instance = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, merchant: MerchantAgent) -> None:
        """Register a merchant agent with the registry.

        The agent must not already be registered. After registration,
        the agent can be started, stopped, and monitored through
        the registry.

        Args:
            merchant: The MerchantAgent instance to register.

        Raises:
            ValueError: If a merchant with the same ID is already registered.
        """
        if merchant.id in self._merchants:
            raise ValueError(
                f"Merchant with id '{merchant.id}' is already registered."
            )

        self._merchants[merchant.id] = merchant
        self._registered_at[merchant.id] = datetime.now(timezone.utc)
        logger.info(
            "Registered merchant '%s' (id=%s, type=%s)",
            merchant.config.name,
            merchant.id,
            merchant.config.source_type.value,
        )

    def unregister(self, merchant_id: str) -> MerchantAgent:
        """Remove a merchant agent from the registry.

        The agent is stopped if running, then removed from all tracking
        data structures.

        Args:
            merchant_id: The ID of the merchant to unregister.

        Returns:
            The unregistered MerchantAgent instance.

        Raises:
            KeyError: If no merchant with the given ID is registered.
        """
        if merchant_id not in self._merchants:
            raise KeyError(f"Merchant '{merchant_id}' is not registered.")

        merchant = self._merchants.pop(merchant_id)
        self._registered_at.pop(merchant_id, None)

        logger.info(
            "Unregistered merchant '%s' (id=%s)",
            merchant.config.name,
            merchant_id,
        )
        return merchant

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    async def start_merchant(self, merchant_id: str) -> None:
        """Start a registered merchant agent.

        Args:
            merchant_id: The ID of the merchant to start.

        Raises:
            KeyError: If the merchant is not registered.
        """
        merchant = self._get_merchant(merchant_id)
        await merchant.start()
        logger.info("Started merchant '%s' (id=%s)", merchant.config.name, merchant_id)

    async def stop_merchant(self, merchant_id: str) -> None:
        """Stop a registered merchant agent.

        Args:
            merchant_id: The ID of the merchant to stop.

        Raises:
            KeyError: If the merchant is not registered.
        """
        merchant = self._get_merchant(merchant_id)
        await merchant.stop()
        logger.info("Stopped merchant '%s' (id=%s)", merchant.config.name, merchant_id)

    async def restart_merchant(self, merchant_id: str) -> None:
        """Restart a registered merchant agent.

        Stops and re-starts the agent, resetting error counters.

        Args:
            merchant_id: The ID of the merchant to restart.

        Raises:
            KeyError: If the merchant is not registered.
        """
        merchant = self._get_merchant(merchant_id)
        await merchant.restart()
        logger.info("Restarted merchant '%s' (id=%s)", merchant.config.name, merchant_id)

    def pause_merchant(self, merchant_id: str) -> None:
        """Pause a registered merchant agent.

        The agent's polling loop continues but skips collection cycles.

        Args:
            merchant_id: The ID of the merchant to pause.

        Raises:
            KeyError: If the merchant is not registered.
        """
        merchant = self._get_merchant(merchant_id)
        merchant.pause()

    def resume_merchant(self, merchant_id: str) -> None:
        """Resume a paused merchant agent.

        Args:
            merchant_id: The ID of the merchant to resume.

        Raises:
            KeyError: If the merchant is not registered.
        """
        merchant = self._get_merchant(merchant_id)
        merchant.resume()

    async def start_all(self) -> None:
        """Start all registered and enabled merchant agents.

        Only starts agents whose ``config.enabled`` is True and whose
        current status is IDLE or STOPPED.
        """
        for merchant in self._merchants.values():
            if merchant.config.enabled and merchant.status in (
                MerchantStatus.IDLE,
                MerchantStatus.STOPPED,
            ):
                await merchant.start()

    async def stop_all(self) -> None:
        """Stop all running merchant agents gracefully."""
        for merchant in self._merchants.values():
            if merchant.status in (
                MerchantStatus.RUNNING,
                MerchantStatus.PAUSED,
                MerchantStatus.ERROR,
            ):
                await merchant.stop()

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    def get_health(self, merchant_id: str) -> MerchantHealthSnapshot:
        """Get the health snapshot for a specific merchant.

        Args:
            merchant_id: The ID of the merchant to check.

        Returns:
            MerchantHealthSnapshot with current metrics.

        Raises:
            KeyError: If the merchant is not registered.
        """
        merchant = self._get_merchant(merchant_id)
        return merchant.health()

    def get_all_health(self) -> list[MerchantHealthSnapshot]:
        """Get health snapshots for all registered merchants.

        Returns:
            List of MerchantHealthSnapshot objects, one per merchant.
        """
        return [m.health() for m in self._merchants.values()]

    def get_unhealthy(
        self,
        max_consecutive_errors: int = 5,
    ) -> list[MerchantHealthSnapshot]:
        """Find merchants that may need attention.

        Returns merchants in ERROR status or with consecutive errors
        exceeding the threshold.

        Args:
            max_consecutive_errors: Error count threshold for flagging.

        Returns:
            List of health snapshots for unhealthy merchants.
        """
        unhealthy = []
        for merchant in self._merchants.values():
            snapshot = merchant.health()
            if (
                snapshot.status == MerchantStatus.ERROR
                or snapshot.consecutive_errors >= max_consecutive_errors
            ):
                unhealthy.append(snapshot)
        return unhealthy

    # ------------------------------------------------------------------
    # Listing and lookup
    # ------------------------------------------------------------------

    def list_merchants(
        self,
        status: Optional[MerchantStatus] = None,
    ) -> list[MerchantHealthSnapshot]:
        """List all registered merchants, optionally filtered by status.

        Args:
            status: If provided, only return merchants with this status.

        Returns:
            List of MerchantHealthSnapshot objects for matching merchants.
        """
        snapshots = []
        for merchant in self._merchants.values():
            snapshot = merchant.health()
            if status is None or snapshot.status == status:
                snapshots.append(snapshot)
        return snapshots

    def get_merchant(self, merchant_id: str) -> Optional[MerchantAgent]:
        """Retrieve a merchant agent by ID.

        Args:
            merchant_id: The unique identifier of the merchant.

        Returns:
            The MerchantAgent if found, None otherwise.
        """
        return self._merchants.get(merchant_id)

    @property
    def count(self) -> int:
        """Return the number of registered merchants."""
        return len(self._merchants)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_merchant(self, merchant_id: str) -> MerchantAgent:
        """Retrieve a merchant by ID, raising KeyError if not found.

        Args:
            merchant_id: The ID of the merchant to look up.

        Returns:
            The MerchantAgent instance.

        Raises:
            KeyError: If the merchant is not registered.
        """
        if merchant_id not in self._merchants:
            raise KeyError(f"Merchant '{merchant_id}' is not registered.")
        return self._merchants[merchant_id]
