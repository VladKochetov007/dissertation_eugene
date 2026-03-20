"""World Bank Indicators API collector."""

from __future__ import annotations

import logging
from typing import Optional

from .base import AbstractCollector

logger = logging.getLogger(__name__)

# West African + external country ISO3 codes
COUNTRY_CODES = [
    "NGA", "GHA", "SEN", "CIV", "MLI", "BFA", "GIN", "NER",
    "BEN", "TGO", "SLE", "LBR", "GNB", "GMB", "CPV",
    "MRT", "CMR", "MAR",
]

INDICATORS = {
    "NY.GDP.PCAP.CD": "gdp_per_capita",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "NE.TRD.GNFS.ZS": "trade_openness_pct",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_pct_gdp",
    "FP.CPI.TOTL.ZG": "inflation",
    "TM.TAX.MRCH.SM.AR.ZS": "avg_tariff",
    "BN.CAB.XOKA.GD.ZS": "current_account_pct_gdp",
    "BX.TRF.PWKR.DT.GD.ZS": "remittances_pct_gdp",
}


class WorldBankCollector(AbstractCollector):
    """Collect economic indicators from the World Bank API."""

    BASE_URL = "https://api.worldbank.org/v2"

    def __init__(self, **kwargs):
        super().__init__(rate_limit_per_second=5.0, cache_ttl_hours=48, **kwargs)

    @property
    def source_name(self) -> str:
        return "World Bank"

    def collect(
        self,
        countries: Optional[list[str]] = None,
        indicators: Optional[dict[str, str]] = None,
        start_year: int = 2000,
        end_year: int = 2025,
    ) -> list[dict]:
        """Collect indicators for all countries.

        Returns list of dicts with keys: country_iso3, year, indicator_name, value.
        """
        countries = countries or COUNTRY_CODES
        indicators = indicators or INDICATORS
        records = []

        for iso3 in countries:
            for ind_code, ind_name in indicators.items():
                try:
                    data = self._fetch_indicator(iso3, ind_code, start_year, end_year)
                    for entry in data:
                        if entry.get("value") is not None:
                            records.append({
                                "country_iso3": iso3,
                                "year": int(entry["date"]),
                                "indicator_code": ind_code,
                                "indicator_name": ind_name,
                                "value": float(entry["value"]),
                                "source": "worldbank",
                            })
                except Exception as e:
                    logger.warning(f"Failed to fetch {ind_code} for {iso3}: {e}")

        logger.info(f"Collected {len(records)} records from World Bank")
        return records

    def _fetch_indicator(
        self, country: str, indicator: str, start_year: int, end_year: int
    ) -> list[dict]:
        """Fetch a single indicator for a single country."""
        url = f"{self.BASE_URL}/country/{country}/indicator/{indicator}"
        params = {
            "format": "json",
            "date": f"{start_year}:{end_year}",
            "per_page": 100,
        }
        result = self.fetch_json(url, params)
        if result and len(result) >= 2:
            return result[1] or []
        return []

    def collect_latest(self, countries: Optional[list[str]] = None) -> dict[str, dict]:
        """Collect only the latest available value per country per indicator.

        Returns: {country_iso3: {indicator_name: value, ...}, ...}
        """
        records = self.collect(countries=countries, start_year=2020, end_year=2025)
        latest: dict[str, dict] = {}
        for r in sorted(records, key=lambda x: x["year"]):
            country = r["country_iso3"]
            latest.setdefault(country, {})[r["indicator_name"]] = r["value"]
        return latest
