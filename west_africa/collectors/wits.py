"""World Integrated Trade Solution (WITS) tariff data collector."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
import urllib.request
import urllib.error
from typing import Optional

from .base import AbstractCollector

logger = logging.getLogger(__name__)

# West African + external country ISO3 codes (same as world_bank.py)
COUNTRY_CODES = [
    "NGA", "GHA", "SEN", "CIV", "MLI", "BFA", "GIN", "NER",
    "BEN", "TGO", "SLE", "LBR", "GNB", "GMB", "CPV",
    "MRT", "CMR", "MAR",
]

# Major product groups for tariff disaggregation
PRODUCT_GROUPS = {
    "ALL": "All Products",
    "AGR": "Agricultural Products",
    "MFG": "Manufactured Goods",
    "FUE": "Fuels",
    "MIN": "Mining Products",
    "TEX": "Textiles and Clothing",
    "FOO": "Food Products",
}


class WITSCollector(AbstractCollector):
    """Collect applied tariff rate data from the WITS SDMX API.

    WITS provides tariff data through a SDMX-based REST API.  Responses are
    XML, which this collector parses into flat records.
    """

    BASE_URL = "https://wits.worldbank.org/API/V1/SDMX/V21"

    def __init__(self, **kwargs):
        super().__init__(rate_limit_per_second=1.0, cache_ttl_hours=72, **kwargs)

    @property
    def source_name(self) -> str:
        return "WITS"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_xml(
        self,
        url: str,
        max_retries: int = 3,
    ) -> Optional[ET.Element]:
        """Fetch and parse an XML response from the WITS API."""
        import time

        cache_key = self._cache_key(url, {})
        cached = self._get_cached(cache_key)
        if cached is not None:
            # Cached as raw XML string
            return ET.fromstring(cached)

        for attempt in range(max_retries):
            self._rate_limit()
            try:
                req = urllib.request.Request(url)
                req.add_header("User-Agent", "WestAfricaFTZ-Research/1.0")
                req.add_header("Accept", "application/xml")
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read().decode()
                    self._set_cache(cache_key, raw)
                    return ET.fromstring(raw)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"WITS rate limited, waiting {wait}s...")
                    time.sleep(wait)
                elif e.code >= 500:
                    wait = 2 ** attempt
                    logger.warning(f"WITS server error {e.code}, retry in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"WITS HTTP error {e.code}: {url}")
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Error fetching WITS {url}: {e}, retry in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"WITS failed after {max_retries} attempts: {url}")
                    raise

        return None

    @staticmethod
    def _parse_tariff_observations(root: ET.Element) -> list[dict]:
        """Extract observation values from a WITS SDMX XML response.

        The SDMX structure varies; this handles the common GenericData layout
        returned by the WITS V21 endpoint.
        """
        records: list[dict] = []
        # Strip namespaces for easier XPath
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]

        for obs in root.iter("Obs"):
            entry: dict[str, str] = {}
            for child in obs:
                tag = child.tag
                if tag == "ObsDimension":
                    entry["year"] = child.attrib.get("value", "")
                elif tag == "ObsValue":
                    entry["value"] = child.attrib.get("value", "")
            # Also look at series-level keys from the parent
            series = obs.find("..")
            if series is not None:
                for key in series.iter("Value"):
                    concept = key.attrib.get("id", key.attrib.get("concept", ""))
                    val = key.attrib.get("value", "")
                    if concept:
                        entry[concept] = val
            if entry.get("value"):
                records.append(entry)

        return records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        reporters: Optional[list[str]] = None,
        partners: Optional[list[str]] = None,
        year: int = 2022,
    ) -> list[dict]:
        """Fetch applied MFN tariff rates for reporter-partner country pairs.

        Returns list of dicts with keys:
            reporter_iso3, partner_iso3, year, product_group, tariff_rate, source
        """
        reporters = reporters or COUNTRY_CODES
        partners = partners or COUNTRY_CODES
        records: list[dict] = []

        for reporter in reporters:
            for partner in partners:
                if reporter == partner:
                    continue
                try:
                    url = (
                        f"{self.BASE_URL}/datasource/tradestats-tariff"
                        f"/reporter/{reporter}"
                        f"/partner/{partner}"
                        f"/product/all"
                        f"/indicator/AHS-WGHTD-AVRG"
                        f"/year/{year}"
                    )
                    root = self._fetch_xml(url)
                    if root is None:
                        continue

                    obs_list = self._parse_tariff_observations(root)
                    for obs in obs_list:
                        try:
                            tariff_val = float(obs.get("value", ""))
                        except (ValueError, TypeError):
                            continue
                        product = obs.get("PRODUCT", obs.get("product", "ALL"))
                        records.append({
                            "reporter_iso3": reporter,
                            "partner_iso3": partner,
                            "year": int(obs.get("year", year)),
                            "product_group": product,
                            "tariff_rate": tariff_val,
                            "source": "wits",
                        })
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch WITS tariff {reporter}->{partner}: {e}"
                    )

        logger.info(f"Collected {len(records)} tariff records from WITS")
        return records

    def collect_simple_tariffs(
        self,
        countries: Optional[list[str]] = None,
        start_year: int = 2015,
        end_year: int = 2023,
    ) -> list[dict]:
        """Fetch overall simple average applied tariff per country.

        This is a lighter query that retrieves the aggregate MFN simple-average
        tariff for each country across the requested year range.

        Returns list of dicts with keys:
            country_iso3, year, indicator_name, tariff_rate, source
        """
        countries = countries or COUNTRY_CODES
        records: list[dict] = []

        for iso3 in countries:
            try:
                url = (
                    f"{self.BASE_URL}/datasource/tradestats-tariff"
                    f"/reporter/{iso3}"
                    f"/partner/WLD"
                    f"/product/all"
                    f"/indicator/MFN-SMPL-AVRG"
                    f"/year/{start_year}:{end_year}"
                )
                root = self._fetch_xml(url)
                if root is None:
                    continue

                obs_list = self._parse_tariff_observations(root)
                for obs in obs_list:
                    try:
                        tariff_val = float(obs.get("value", ""))
                    except (ValueError, TypeError):
                        continue
                    records.append({
                        "country_iso3": iso3,
                        "year": int(obs.get("year", end_year)),
                        "indicator_name": "mfn_simple_avg_tariff",
                        "tariff_rate": tariff_val,
                        "source": "wits",
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch WITS simple tariff for {iso3}: {e}")

        logger.info(f"Collected {len(records)} simple tariff records from WITS")
        return records
