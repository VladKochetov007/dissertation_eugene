"""
Polymarket Merchant Agent — prediction market data integration.

Bridges Track B (knowledge graph) and Track C (Polymarket pipeline) by
collecting prediction market data and ingesting it as Variable and DataSource
entities. Polymarket is a distributed Popperian falsification engine: each
market is a collective bet on a falsifiable proposition.

This merchant connects to the existing Polymarket pipeline data (the sibling
polymarket/ directory) and translates market metadata, prices, trades, and
orderbook depth into knowledge graph entities.

Data collected:
- Active market metadata (question, outcomes, category, end date)
- Current prices / probabilities per outcome
- Trade volume and liquidity metrics
- Market resolution status

Pearl Level: 1 (Association) — observational data from prediction markets.

Usage:
    from graph.store import KnowledgeGraphStore
    from merchants.online.polymarket import PolymarketMerchant

    store = KnowledgeGraphStore()
    merchant = PolymarketMerchant(store=store)
    await merchant.start()
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from graph.entities import DataSourceType, VariableType
from graph.store import KnowledgeGraphStore

from ..base import (
    CollectionResult,
    MerchantAgent,
    MerchantConfig,
    PearlLevel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLYMARKET_API_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
DEFAULT_POLL_INTERVAL = 120.0  # 2 minutes
DEFAULT_RATE_LIMIT = 30.0  # requests per minute


# ---------------------------------------------------------------------------
# Polymarket Merchant
# ---------------------------------------------------------------------------


class PolymarketMerchant(MerchantAgent):
    """Merchant agent for Polymarket prediction market data.

    Collects market metadata, prices, and volume data from Polymarket's
    public APIs. Each market becomes a set of Variable entities in the
    knowledge graph (one per outcome), enabling causal analysis of how
    prediction markets relate to real-world events.

    The merchant also bridges to the existing Polymarket pipeline by
    reading from its ClickHouse data store when available, falling back
    to the public API for direct collection.

    Attributes:
        api_base: Base URL for the Polymarket CLOB API.
        gamma_api: Base URL for the Polymarket Gamma API.
        _client: Async HTTP client for API requests.
        _tracked_markets: Set of market condition IDs being tracked.
    """

    def __init__(
        self,
        store: KnowledgeGraphStore,
        api_base: str = POLYMARKET_API_BASE,
        gamma_api: str = POLYMARKET_GAMMA_API,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        market_filter: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the Polymarket merchant.

        Args:
            store: Knowledge graph store for data ingestion.
            api_base: Base URL for the CLOB API.
            gamma_api: Base URL for the Gamma API.
            poll_interval: Seconds between polling cycles.
            rate_limit: Maximum requests per minute.
            market_filter: Optional filter criteria for market discovery
                          (e.g., {"category": "politics", "active": True}).
        """
        config = MerchantConfig(
            name="polymarket",
            source_type=DataSourceType.MARKET_FEED,
            pearl_level=PearlLevel.ASSOCIATION,
            poll_interval_seconds=poll_interval,
            rate_limit_requests_per_minute=rate_limit,
            metadata=market_filter or {},
        )
        super().__init__(config=config, store=store)

        self.api_base = api_base
        self.gamma_api = gamma_api
        self._client: Optional[httpx.AsyncClient] = None
        self._tracked_markets: set[str] = set()

        # Register data source in knowledge graph
        self._data_source = self._register_data_source(
            name="Polymarket Prediction Markets",
            url=api_base,
            schema_info={
                "type": "prediction_market",
                "provider": "polymarket",
                "data_fields": [
                    "condition_id",
                    "question",
                    "outcomes",
                    "prices",
                    "volume",
                    "liquidity",
                    "end_date",
                    "resolved",
                ],
            },
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client.

        Returns:
            Configured httpx.AsyncClient instance.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                headers={"Accept": "application/json"},
            )
        return self._client

    # ------------------------------------------------------------------
    # Pipeline implementation
    # ------------------------------------------------------------------

    async def discover(self) -> list[str]:
        """Discover active prediction markets on Polymarket.

        Queries the Gamma API for active markets, optionally filtered
        by category or other criteria from the merchant config.

        Returns:
            List of market condition IDs to collect data for.
        """
        client = await self._get_client()
        try:
            params: dict[str, Any] = {
                "active": True,
                "closed": False,
                "limit": 100,
            }

            # Apply any configured filters
            if "category" in self.config.metadata:
                params["tag"] = self.config.metadata["category"]

            response = await client.get(
                f"{self.gamma_api}/markets",
                params=params,
            )
            response.raise_for_status()
            markets = response.json()

            condition_ids = []
            for market in markets:
                cid = market.get("conditionId") or market.get("condition_id", "")
                if cid:
                    condition_ids.append(cid)
                    self._tracked_markets.add(cid)

            logger.info(
                "Polymarket: discovered %d active markets.", len(condition_ids)
            )
            return condition_ids

        except httpx.HTTPError as exc:
            logger.error("Polymarket discovery failed: %s", exc)
            raise

    async def collect(self, targets: list[str]) -> list[CollectionResult]:
        """Collect price and metadata for each target market.

        For each market condition ID, fetches current prices, volume,
        and metadata from the Polymarket APIs.

        Args:
            targets: List of market condition IDs.

        Returns:
            List of CollectionResult objects with market data.
        """
        client = await self._get_client()
        results: list[CollectionResult] = []

        for condition_id in targets:
            try:
                await self._enforce_rate_limit()

                # Fetch market details from Gamma API
                response = await client.get(
                    f"{self.gamma_api}/markets/{condition_id}",
                )

                if response.status_code == 404:
                    logger.debug("Market %s not found, skipping.", condition_id)
                    continue

                response.raise_for_status()
                market_data = response.json()

                result = CollectionResult(
                    target=condition_id,
                    data={
                        "condition_id": condition_id,
                        "question": market_data.get("question", ""),
                        "description": market_data.get("description", ""),
                        "outcomes": market_data.get("outcomes", []),
                        "outcome_prices": market_data.get("outcomePrices", []),
                        "volume": market_data.get("volume", 0),
                        "liquidity": market_data.get("liquidity", 0),
                        "start_date": market_data.get("startDate"),
                        "end_date": market_data.get("endDate"),
                        "category": market_data.get("category", ""),
                        "resolved": market_data.get("resolved", False),
                        "resolution": market_data.get("resolution"),
                    },
                    provenance=self._build_provenance(
                        source_url=f"{self.gamma_api}/markets/{condition_id}",
                        collection_method="api_poll",
                        confidence=0.9,
                        raw_record_count=1,
                    ),
                )
                results.append(result)

            except httpx.HTTPError as exc:
                logger.warning(
                    "Polymarket: failed to collect market %s: %s",
                    condition_id,
                    exc,
                )
                results.append(
                    CollectionResult(
                        target=condition_id,
                        data={},
                        provenance=self._build_provenance(
                            source_url=f"{self.gamma_api}/markets/{condition_id}",
                            confidence=0.0,
                        ),
                        is_valid=False,
                        validation_errors=[f"HTTP error: {exc}"],
                    )
                )

        logger.info(
            "Polymarket: collected data for %d/%d markets.",
            len([r for r in results if not r.validation_errors]),
            len(targets),
        )
        return results

    async def validate(self, results: list[CollectionResult]) -> list[CollectionResult]:
        """Validate collected market data.

        Checks that each result contains the required fields and that
        prices are within valid ranges (0-1 for probabilities).

        Args:
            results: Collection results to validate.

        Returns:
            Results with is_valid and validation_errors updated.
        """
        for result in results:
            # Skip already-failed results
            if result.validation_errors:
                continue

            errors: list[str] = []
            data = result.data

            # Required fields
            if not data.get("condition_id"):
                errors.append("Missing condition_id.")
            if not data.get("question"):
                errors.append("Missing question text.")

            # Price validation: probabilities should be in [0, 1]
            prices = data.get("outcome_prices", [])
            if prices:
                try:
                    parsed_prices = [float(p) for p in prices]
                    for price in parsed_prices:
                        if not (0.0 <= price <= 1.0):
                            errors.append(
                                f"Price {price} out of valid range [0, 1]."
                            )
                except (TypeError, ValueError) as exc:
                    errors.append(f"Invalid price format: {exc}")

            # Volume should be non-negative
            volume = data.get("volume", 0)
            try:
                if float(volume) < 0:
                    errors.append("Negative volume.")
            except (TypeError, ValueError):
                errors.append(f"Invalid volume format: {volume}")

            result.validation_errors = errors
            result.is_valid = len(errors) == 0

        valid_count = sum(1 for r in results if r.is_valid)
        logger.info(
            "Polymarket: validated %d/%d results.", valid_count, len(results)
        )
        return results

    async def ingest(self, results: list[CollectionResult]) -> int:
        """Ingest validated market data into the knowledge graph.

        Creates Variable entities for each market outcome and links them
        to the Polymarket data source.

        Args:
            results: Validated collection results.

        Returns:
            Number of variables created or updated.
        """
        ingested = 0

        for result in results:
            if not result.is_valid:
                continue

            data = result.data
            condition_id = data["condition_id"]
            question = data["question"]
            outcomes = data.get("outcomes", [])
            prices = data.get("outcome_prices", [])

            # Create a Variable for each outcome
            for i, outcome in enumerate(outcomes):
                var_name = f"polymarket:{condition_id}:{outcome}"
                price = float(prices[i]) if i < len(prices) else None

                variable = self._create_variable(
                    name=var_name,
                    variable_type=VariableType.OBSERVABLE,
                )

                result.variables_created.append(variable.id)
                ingested += 1

            # Create a summary variable for the market itself
            market_var_name = f"polymarket:{condition_id}:market"
            self._create_variable(
                name=market_var_name,
                variable_type=VariableType.OBSERVABLE,
            )
            ingested += 1

        logger.info("Polymarket: ingested %d variables.", ingested)
        return ingested

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Stop the merchant and close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        await super().stop()
