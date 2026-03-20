"""UN Comtrade API collector for bilateral trade flows."""

from __future__ import annotations

import logging
from typing import Optional

from .base import AbstractCollector

logger = logging.getLogger(__name__)

# ISO3 to UN Comtrade numeric codes (M49)
COUNTRY_M49 = {
    "NGA": "566", "GHA": "288", "SEN": "686", "CIV": "384",
    "MLI": "466", "BFA": "854", "GIN": "324", "NER": "562",
    "BEN": "204", "TGO": "768", "SLE": "694", "LBR": "430",
    "GNB": "624", "GMB": "270", "CPV": "132",
    "MRT": "478", "CMR": "120", "MAR": "504",
}


class UNComtradeCollector(AbstractCollector):
    """Collect bilateral trade flow data from UN Comtrade."""

    BASE_URL = "https://comtradeapi.un.org/data/v1/get/C/A/HS"

    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(rate_limit_per_second=0.1, cache_ttl_hours=168, **kwargs)  # 1 week cache
        self.api_key = api_key

    @property
    def source_name(self) -> str:
        return "UN Comtrade"

    def collect(
        self,
        reporter_iso3: Optional[str] = None,
        partner_iso3: Optional[str] = None,
        year: int = 2023,
    ) -> list[dict]:
        """Collect bilateral trade data.

        Returns list of dicts with: reporter_iso3, partner_iso3, year,
        trade_value_usd, flow_type (import/export).
        """
        reporters = [reporter_iso3] if reporter_iso3 else list(COUNTRY_M49.keys())
        records = []

        for rep_iso3 in reporters:
            rep_m49 = COUNTRY_M49.get(rep_iso3)
            if not rep_m49:
                continue

            try:
                data = self._fetch_trade(rep_m49, year, partner_iso3)
                for entry in data:
                    partner_m49 = str(entry.get("partnerCode", ""))
                    # Reverse lookup M49 to ISO3
                    partner = next(
                        (k for k, v in COUNTRY_M49.items() if v == partner_m49), None
                    )
                    if partner:
                        records.append({
                            "reporter_iso3": rep_iso3,
                            "partner_iso3": partner,
                            "year": year,
                            "trade_value_usd": entry.get("primaryValue", 0),
                            "flow_type": "export" if entry.get("flowCode") == "X" else "import",
                            "commodity_code": entry.get("cmdCode", "TOTAL"),
                            "source": "comtrade",
                        })
            except Exception as e:
                logger.warning(f"Failed to fetch trade for {rep_iso3}: {e}")

        logger.info(f"Collected {len(records)} trade records from UN Comtrade")
        return records

    def _fetch_trade(
        self, reporter_m49: str, year: int, partner_iso3: Optional[str] = None
    ) -> list[dict]:
        """Fetch trade data for a reporter country."""
        params = {
            "reporterCode": reporter_m49,
            "period": str(year),
            "flowCode": "X,M",  # exports and imports
            "cmdCode": "TOTAL",
        }
        if partner_iso3 and partner_iso3 in COUNTRY_M49:
            params["partnerCode"] = COUNTRY_M49[partner_iso3]

        if self.api_key:
            params["subscription-key"] = self.api_key

        result = self.fetch_json(self.BASE_URL, params)
        if result and "data" in result:
            return result["data"]
        return []

    def collect_bilateral_matrix(self, year: int = 2023) -> dict[tuple[str, str], float]:
        """Build a bilateral trade matrix for all country pairs.

        Returns: {(reporter, partner): trade_value_usd, ...}
        """
        records = self.collect(year=year)
        matrix: dict[tuple[str, str], float] = {}
        for r in records:
            key = (r["reporter_iso3"], r["partner_iso3"])
            matrix[key] = matrix.get(key, 0) + r["trade_value_usd"]
        return matrix
