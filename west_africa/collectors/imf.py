"""IMF Data API collector for exchange rates and financial indicators."""

from __future__ import annotations

import logging
from typing import Optional

from .base import AbstractCollector

logger = logging.getLogger(__name__)

COUNTRY_CODES = [
    "NGA", "GHA", "SEN", "CIV", "MLI", "BFA", "GIN", "NER",
    "BEN", "TGO", "SLE", "LBR", "GNB", "GMB", "CPV",
    "MRT", "CMR", "MAR",
]


class IMFCollector(AbstractCollector):
    """Collect financial data from IMF Data API."""

    BASE_URL = "https://dataservices.imf.org/REST/SDMX_JSON.svc"

    def __init__(self, **kwargs):
        super().__init__(rate_limit_per_second=2.0, cache_ttl_hours=24, **kwargs)

    @property
    def source_name(self) -> str:
        return "IMF"

    def collect(
        self,
        countries: Optional[list[str]] = None,
        start_year: int = 2015,
        end_year: int = 2025,
    ) -> list[dict]:
        """Collect exchange rate and financial data.

        Returns list of dicts with: country_iso3, period, indicator, value.
        """
        countries = countries or COUNTRY_CODES
        records = []

        for iso3 in countries:
            try:
                fx_data = self._fetch_exchange_rate(iso3)
                for period, value in fx_data.items():
                    year = int(period[:4]) if len(period) >= 4 else 0
                    if start_year <= year <= end_year:
                        records.append({
                            "country_iso3": iso3,
                            "period": period,
                            "indicator": "exchange_rate_usd",
                            "value": value,
                            "source": "imf",
                        })
            except Exception as e:
                logger.warning(f"Failed to fetch IMF data for {iso3}: {e}")

        logger.info(f"Collected {len(records)} records from IMF")
        return records

    def _fetch_exchange_rate(self, country_iso3: str) -> dict[str, float]:
        """Fetch exchange rate time series for a country."""
        url = f"{self.BASE_URL}/CompactData/IFS/M.{country_iso3}.ENDA_XDC_USD_RATE"
        result = self.fetch_json(url)

        rates = {}
        if result:
            try:
                series = result.get("CompactData", {}).get("DataSet", {}).get("Series", {})
                observations = series.get("Obs", [])
                if isinstance(observations, dict):
                    observations = [observations]
                for obs in observations:
                    period = obs.get("@TIME_PERIOD", "")
                    value = obs.get("@OBS_VALUE")
                    if period and value:
                        rates[period] = float(value)
            except (KeyError, TypeError, ValueError) as e:
                logger.debug(f"Parse error for {country_iso3}: {e}")

        return rates
