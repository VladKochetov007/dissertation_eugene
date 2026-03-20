"""
Financial Market Merchant Agent — stock, forex, and commodity data integration.

Collects financial market data via yfinance and maps financial instruments
to Variable entities in the knowledge graph. Financial markets are the
purest expression of aggregated human judgment on asset values — a rich
source of observational data for causal analysis.

Data collected:
- Stock prices (open, high, low, close, volume)
- Forex exchange rates
- Commodity prices
- Key financial metrics (market cap, P/E ratio, etc.)

Pearl Level: 1 (Association) — observational data from financial markets.

Usage:
    from graph.store import KnowledgeGraphStore
    from merchants.online.financial import FinancialMerchant

    store = KnowledgeGraphStore()
    merchant = FinancialMerchant(
        store=store,
        symbols=["AAPL", "MSFT", "EURUSD=X", "GC=F"],
    )
    await merchant.start()
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

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

DEFAULT_POLL_INTERVAL = 300.0  # 5 minutes
DEFAULT_RATE_LIMIT = 20.0  # requests per minute (yfinance is rate-sensitive)

# Common symbol categories for organized discovery
DEFAULT_STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "V",
]
DEFAULT_FOREX_SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
]
DEFAULT_COMMODITY_SYMBOLS = [
    "GC=F",   # Gold
    "SI=F",   # Silver
    "CL=F",   # Crude Oil
    "NG=F",   # Natural Gas
    "ZW=F",   # Wheat
]


# ---------------------------------------------------------------------------
# Financial Merchant
# ---------------------------------------------------------------------------


class FinancialMerchant(MerchantAgent):
    """Merchant agent for financial market data via yfinance.

    Collects stock prices, forex rates, and commodity prices, converting
    each financial instrument into Variable entities for causal analysis
    in the knowledge graph.

    Note: yfinance is a synchronous library. This merchant uses a thread
    pool executor to avoid blocking the async event loop.

    Attributes:
        symbols: List of ticker symbols to track.
        _executor: Thread pool for running synchronous yfinance calls.
    """

    def __init__(
        self,
        store: KnowledgeGraphStore,
        symbols: Optional[list[str]] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        include_stocks: bool = True,
        include_forex: bool = True,
        include_commodities: bool = True,
    ) -> None:
        """Initialize the financial market merchant.

        Args:
            store: Knowledge graph store for data ingestion.
            symbols: Explicit list of ticker symbols. If None, uses defaults
                    based on the include_* flags.
            poll_interval: Seconds between polling cycles.
            rate_limit: Maximum requests per minute.
            include_stocks: Whether to include default stock symbols.
            include_forex: Whether to include default forex symbols.
            include_commodities: Whether to include default commodity symbols.
        """
        config = MerchantConfig(
            name="financial-markets",
            source_type=DataSourceType.MARKET_FEED,
            pearl_level=PearlLevel.ASSOCIATION,
            poll_interval_seconds=poll_interval,
            rate_limit_requests_per_minute=rate_limit,
            metadata={
                "include_stocks": include_stocks,
                "include_forex": include_forex,
                "include_commodities": include_commodities,
            },
        )
        super().__init__(config=config, store=store)

        # Build symbol list
        if symbols is not None:
            self.symbols = list(symbols)
        else:
            self.symbols = []
            if include_stocks:
                self.symbols.extend(DEFAULT_STOCK_SYMBOLS)
            if include_forex:
                self.symbols.extend(DEFAULT_FOREX_SYMBOLS)
            if include_commodities:
                self.symbols.extend(DEFAULT_COMMODITY_SYMBOLS)

        self._executor = ThreadPoolExecutor(max_workers=4)

        # Register data source
        self._data_source = self._register_data_source(
            name="Financial Markets (yfinance)",
            url="https://finance.yahoo.com",
            schema_info={
                "type": "financial_market",
                "provider": "yfinance",
                "data_fields": [
                    "symbol",
                    "price",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "market_cap",
                    "pe_ratio",
                    "currency",
                ],
            },
        )

    # ------------------------------------------------------------------
    # Pipeline implementation
    # ------------------------------------------------------------------

    async def discover(self) -> list[str]:
        """Return the configured list of financial instrument symbols.

        For financial markets, discovery is static: the symbols are
        configured at initialization. Dynamic discovery (screening for
        new tickers) could be added in the future.

        Returns:
            List of ticker symbols to collect data for.
        """
        logger.info(
            "Financial: tracking %d symbols.", len(self.symbols)
        )
        return list(self.symbols)

    async def collect(self, targets: list[str]) -> list[CollectionResult]:
        """Collect current price data for each financial instrument.

        Uses yfinance in a thread pool to avoid blocking the async loop.
        Collects the most recent available quote for each symbol.

        Args:
            targets: List of ticker symbols.

        Returns:
            List of CollectionResult objects with price data.
        """
        import asyncio

        results: list[CollectionResult] = []
        loop = asyncio.get_event_loop()

        for symbol in targets:
            try:
                await self._enforce_rate_limit()
                data = await loop.run_in_executor(
                    self._executor, self._fetch_symbol_data, symbol
                )

                result = CollectionResult(
                    target=symbol,
                    data=data,
                    provenance=self._build_provenance(
                        source_url=f"https://finance.yahoo.com/quote/{symbol}",
                        collection_method="yfinance_api",
                        confidence=0.85,
                        raw_record_count=1,
                    ),
                )
                results.append(result)

            except Exception as exc:
                logger.warning("Financial: failed to collect %s: %s", symbol, exc)
                results.append(
                    CollectionResult(
                        target=symbol,
                        data={},
                        provenance=self._build_provenance(
                            source_url=f"https://finance.yahoo.com/quote/{symbol}",
                            confidence=0.0,
                        ),
                        is_valid=False,
                        validation_errors=[f"Collection error: {exc}"],
                    )
                )

        logger.info(
            "Financial: collected data for %d/%d symbols.",
            len([r for r in results if not r.validation_errors]),
            len(targets),
        )
        return results

    async def validate(self, results: list[CollectionResult]) -> list[CollectionResult]:
        """Validate collected financial data.

        Checks for required fields, reasonable price ranges, and
        non-negative volumes.

        Args:
            results: Collection results to validate.

        Returns:
            Results with is_valid and validation_errors updated.
        """
        for result in results:
            if result.validation_errors:
                continue

            errors: list[str] = []
            data = result.data

            if not data.get("symbol"):
                errors.append("Missing symbol.")

            price = data.get("price")
            if price is not None:
                try:
                    p = float(price)
                    if p < 0:
                        errors.append(f"Negative price: {p}")
                except (TypeError, ValueError):
                    errors.append(f"Invalid price format: {price}")
            else:
                errors.append("Missing price data.")

            volume = data.get("volume")
            if volume is not None:
                try:
                    v = float(volume)
                    if v < 0:
                        errors.append(f"Negative volume: {v}")
                except (TypeError, ValueError):
                    # Volume can be None for some instruments (forex)
                    pass

            result.validation_errors = errors
            result.is_valid = len(errors) == 0

        valid_count = sum(1 for r in results if r.is_valid)
        logger.info(
            "Financial: validated %d/%d results.", valid_count, len(results)
        )
        return results

    async def ingest(self, results: list[CollectionResult]) -> int:
        """Ingest validated financial data into the knowledge graph.

        Creates a Variable entity for each financial instrument with
        the current price as metadata.

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
            symbol = data["symbol"]

            # Create price variable
            var_name = f"finance:{symbol}:price"
            variable = self._create_variable(
                name=var_name,
                variable_type=VariableType.OBSERVABLE,
            )
            result.variables_created.append(variable.id)
            ingested += 1

            # Create volume variable if available
            if data.get("volume") is not None:
                vol_name = f"finance:{symbol}:volume"
                vol_var = self._create_variable(
                    name=vol_name,
                    variable_type=VariableType.OBSERVABLE,
                )
                result.variables_created.append(vol_var.id)
                ingested += 1

        logger.info("Financial: ingested %d variables.", ingested)
        return ingested

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_symbol_data(symbol: str) -> dict[str, Any]:
        """Fetch current data for a single symbol using yfinance.

        This runs in a thread pool executor since yfinance is synchronous.

        Args:
            symbol: The ticker symbol to fetch.

        Returns:
            Dictionary with price, volume, and metadata fields.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error(
                "yfinance is not installed. Install it with: pip install yfinance"
            )
            raise ImportError(
                "yfinance is required for the FinancialMerchant. "
                "Install with: pip install yfinance"
            )

        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        # Extract the most relevant price field
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        return {
            "symbol": symbol,
            "price": price,
            "open": info.get("open") or info.get("regularMarketOpen"),
            "high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
            "low": info.get("dayLow") or info.get("regularMarketDayLow"),
            "close": info.get("previousClose"),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange"),
            "name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

    async def stop(self) -> None:
        """Stop the merchant and shut down the thread pool."""
        self._executor.shutdown(wait=False)
        await super().stop()
