"""Data collection scheduling and orchestration."""

from __future__ import annotations

import logging
from typing import Optional

from .world_bank import WorldBankCollector
from .un_comtrade import UNComtradeCollector
from .imf import IMFCollector
from .afdb import AfDBCollector
from .wits import WITSCollector

logger = logging.getLogger(__name__)


class CollectionScheduler:
    """Orchestrate data collection from all sources."""

    def __init__(self, comtrade_api_key: str = "") -> None:
        self.world_bank = WorldBankCollector()
        self.comtrade = UNComtradeCollector(api_key=comtrade_api_key)
        self.imf = IMFCollector()
        self.afdb = AfDBCollector()
        self.wits = WITSCollector()

    def collect_all(
        self,
        start_year: int = 2000,
        end_year: int = 2025,
    ) -> dict[str, list[dict]]:
        """Run all collectors and return combined results."""
        results: dict[str, list[dict]] = {}

        logger.info("Collecting World Bank indicators...")
        try:
            results["world_bank"] = self.world_bank.collect(
                start_year=start_year, end_year=end_year
            )
        except Exception as e:
            logger.error(f"World Bank collection failed: {e}")
            results["world_bank"] = []

        logger.info("Collecting UN Comtrade trade flows...")
        try:
            results["comtrade"] = self.comtrade.collect(year=end_year)
        except Exception as e:
            logger.error(f"UN Comtrade collection failed: {e}")
            results["comtrade"] = []

        logger.info("Collecting IMF financial data...")
        try:
            results["imf"] = self.imf.collect(
                start_year=start_year, end_year=end_year
            )
        except Exception as e:
            logger.error(f"IMF collection failed: {e}")
            results["imf"] = []

        logger.info("Collecting AfDB infrastructure index...")
        try:
            results["afdb"] = self.afdb.collect(
                start_year=start_year, end_year=end_year
            )
            if not results["afdb"]:
                logger.info("No AfDB CSV data found, generating synthetic data...")
                results["afdb"] = self.afdb.generate_synthetic(
                    start_year=start_year, end_year=end_year
                )
        except Exception as e:
            logger.error(f"AfDB collection failed: {e}")
            try:
                logger.info("Falling back to synthetic AfDB data...")
                results["afdb"] = self.afdb.generate_synthetic(
                    start_year=start_year, end_year=end_year
                )
            except Exception as e2:
                logger.error(f"AfDB synthetic generation also failed: {e2}")
                results["afdb"] = []

        logger.info("Collecting WITS tariff data...")
        try:
            results["wits"] = self.wits.collect_simple_tariffs(
                start_year=max(start_year, 2015), end_year=end_year
            )
        except Exception as e:
            logger.error(f"WITS collection failed: {e}")
            results["wits"] = []

        total = sum(len(v) for v in results.values())
        logger.info(f"Collection complete: {total} total records")
        return results

    def collect_latest(self) -> dict[str, dict]:
        """Collect only the most recent data from each source."""
        return self.world_bank.collect_latest()
