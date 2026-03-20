"""African Development Bank Infrastructure Index collector."""

from __future__ import annotations

import csv
import logging
import pathlib
import random
from typing import Optional

from .base import AbstractCollector, CACHE_DIR

logger = logging.getLogger(__name__)

# West African + external country ISO3 codes
COUNTRY_CODES = [
    "NGA", "GHA", "SEN", "CIV", "MLI", "BFA", "GIN", "NER",
    "BEN", "TGO", "SLE", "LBR", "GNB", "GMB", "CPV",
    "MRT", "CMR", "MAR",
]

# Approximate infrastructure quality baselines (0-100 scale) based on known
# AfDB and World Economic Forum infrastructure assessments.  These anchor the
# synthetic data generator so that values are plausible.
_INFRASTRUCTURE_BASELINES: dict[str, float] = {
    "NGA": 32.0,  # Nigeria — large but uneven infrastructure
    "GHA": 45.0,  # Ghana — relatively well-developed
    "SEN": 43.0,  # Senegal — strong urban infra, weaker rural
    "CIV": 38.0,  # Cote d'Ivoire — recovering, Abidjan strong
    "MLI": 20.0,  # Mali — landlocked, limited infra
    "BFA": 18.0,  # Burkina Faso — landlocked, limited
    "GIN": 15.0,  # Guinea — mineral wealth, poor infra
    "NER": 12.0,  # Niger — among the lowest globally
    "BEN": 28.0,  # Benin — small but improving (Cotonou port)
    "TGO": 26.0,  # Togo — Lome port hub
    "SLE": 16.0,  # Sierra Leone — post-conflict rebuilding
    "LBR": 14.0,  # Liberia — post-conflict rebuilding
    "GNB": 10.0,  # Guinea-Bissau — very limited
    "GMB": 22.0,  # Gambia — small, basic network
    "CPV": 50.0,  # Cabo Verde — small island, well-managed
    "MRT": 24.0,  # Mauritania — sparse, mining corridors
    "CMR": 30.0,  # Cameroon — Douala port, uneven
    "MAR": 62.0,  # Morocco — best infrastructure in the group
}


class AfDBCollector(AbstractCollector):
    """Collect AfDB Infrastructure Index data from local CSV or synthetic generation.

    The AfDB does not provide a clean public API for infrastructure index data.
    This collector reads from a manually-maintained CSV when available and falls
    back to synthetic generation for research/prototyping purposes.
    """

    CSV_PATH = CACHE_DIR / "afdb_infrastructure.csv"

    def __init__(self, **kwargs):
        super().__init__(rate_limit_per_second=1.0, cache_ttl_hours=168, **kwargs)

    @property
    def source_name(self) -> str:
        return "African Development Bank"

    def collect(
        self,
        countries: Optional[list[str]] = None,
        start_year: int = 2010,
        end_year: int = 2025,
    ) -> list[dict]:
        """Read infrastructure index data from the local CSV file.

        Expected CSV columns: country_iso3, year, infrastructure_index

        Returns list of dicts with keys:
            country_iso3, year, indicator_name, value, source
        """
        countries = countries or COUNTRY_CODES
        country_set = set(countries)

        if not self.CSV_PATH.exists():
            logger.warning(
                f"AfDB CSV not found at {self.CSV_PATH}. "
                "Use generate_synthetic() to create placeholder data."
            )
            return []

        records: list[dict] = []
        with open(self.CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iso3 = row.get("country_iso3", "").strip().upper()
                if iso3 not in country_set:
                    continue
                try:
                    year = int(row["year"])
                except (KeyError, ValueError):
                    continue
                if year < start_year or year > end_year:
                    continue
                try:
                    value = float(row["infrastructure_index"])
                except (KeyError, ValueError):
                    continue
                records.append({
                    "country_iso3": iso3,
                    "year": year,
                    "indicator_name": "infrastructure_index",
                    "value": value,
                    "source": "afdb",
                })

        logger.info(f"Collected {len(records)} records from AfDB CSV")
        return records

    def generate_synthetic(
        self,
        countries: Optional[list[str]] = None,
        start_year: int = 2010,
        end_year: int = 2025,
        seed: int = 42,
    ) -> list[dict]:
        """Generate plausible synthetic infrastructure index values.

        Values are anchored to known baselines with a slight upward trend and
        year-over-year noise.  Useful for prototyping when real AfDB data is
        unavailable.

        Also writes the generated data to the CSV path so that future calls to
        ``collect()`` can read it directly.

        Returns list of dicts with the same schema as ``collect()``.
        """
        countries = countries or COUNTRY_CODES
        rng = random.Random(seed)
        records: list[dict] = []

        for iso3 in countries:
            baseline = _INFRASTRUCTURE_BASELINES.get(iso3, 25.0)
            value = baseline
            for year in range(start_year, end_year + 1):
                # Small upward trend (~0.3-0.8 pts/year) + noise
                trend = rng.uniform(0.3, 0.8)
                noise = rng.gauss(0, 1.5)
                value = max(0.0, min(100.0, value + trend + noise))
                records.append({
                    "country_iso3": iso3,
                    "year": year,
                    "indicator_name": "infrastructure_index",
                    "value": round(value, 2),
                    "source": "afdb",
                })

        # Persist to CSV for future collect() calls
        self.CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["country_iso3", "year", "infrastructure_index"])
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "country_iso3": r["country_iso3"],
                    "year": r["year"],
                    "infrastructure_index": r["value"],
                })

        logger.info(
            f"Generated {len(records)} synthetic AfDB infrastructure records "
            f"and saved to {self.CSV_PATH}"
        )
        return records
