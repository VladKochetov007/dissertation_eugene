"""Data classes and enums for the West Africa Free Trade Zone network model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BlocMembership(str, Enum):
    """Economic bloc membership status."""
    ECOWAS = "ECOWAS"           # Active ECOWAS member (non-CFA)
    UEMOA = "UEMOA"             # ECOWAS + UEMOA (CFA franc zone)
    SUSPENDED = "SUSPENDED"     # Suspended from ECOWAS (Mali, Burkina, Niger, Guinea)
    EXTERNAL = "EXTERNAL"       # External trade partner (Cameroon, Morocco)


class ConnectionType(str, Enum):
    """Type of connection between cities."""
    TRADE = "TRADE"                     # Bilateral goods trade
    POLITICAL = "POLITICAL"             # Diplomatic relations, shared institutions
    CULTURAL = "CULTURAL"               # Linguistic, ethnic, historical ties
    MIGRATORY = "MIGRATORY"             # Migration corridors, diaspora
    LABOUR = "LABOUR"                   # Labour mobility, remittance flows
    INFRASTRUCTURE = "INFRASTRUCTURE"   # Roads, rail, ports, airports
    FINANCIAL = "FINANCIAL"             # Banking, CFA franc zone, financial flows


@dataclass
class City:
    """A node in the West Africa multigraph."""
    id: str
    name: str
    lat: float
    lng: float
    country: str                          # e.g., "Nigeria"
    country_iso3: str                     # e.g., "NGA"
    bloc: BlocMembership
    population: int = 0
    is_port: bool = False                 # Major seaport
    is_capital: bool = False              # National capital
    gdp_per_capita: float = 0.0           # USD, latest available
    trade_openness: float = 0.0           # (exports + imports) / GDP
    ease_of_business: float = 0.0         # World Bank score (0-100)
    cfa_zone: bool = False                # Part of CFA franc monetary zone
    is_ftz_target: bool = False           # City to score for FTZ impact
    tags: list[str] = field(default_factory=list)


@dataclass
class TradeEdge:
    """An edge in the West Africa multigraph."""
    source: str
    target: str
    edge_type: ConnectionType
    weight: float = 1.0                   # Normalized importance (lower = easier)
    volume: float = 0.0                   # Trade volume USD millions / flow quantity
    distance_km: float = 0.0
    is_active: bool = True
    tariff_rate: float = 0.0              # Applied tariff (0-1)
    description: str = ""


@dataclass
class EconomicState:
    """Dynamic overlay for economic indicators — updated periodically."""
    city_id: str
    bloc_override: Optional[BlocMembership] = None
    gdp_growth_rate: float = 0.0          # Year-over-year %
    trade_volume_change: float = 0.0      # Quarter-over-quarter change
    fdi_inflow_change: float = 0.0        # Quarter-over-quarter
    inflation_rate: float = 0.0           # Annual %
    political_stability: float = 0.5      # 0 = unstable, 1 = stable
    tariff_change: float = 0.0            # Avg tariff delta
    last_updated: str = ""                # ISO timestamp


@dataclass
class TradeImpactScore:
    """Composite FTZ impact assessment for a city."""
    city_id: str
    connectivity_score: float = 0.0       # From graph centrality
    port_access_score: float = 0.0        # Distance/routes to nearest major port
    diversification_score: float = 0.0    # Trade partner diversity
    tariff_exposure_score: float = 0.0    # How much current tariffs affect this city
    border_proximity_score: float = 0.0   # Proximity to international borders
    trade_volume_score: float = 0.0       # Current trade volume relative to peers
    stability_score: float = 0.0          # Political stability index
    composite: float = 0.0

    WEIGHTS: dict = field(default_factory=lambda: {
        "connectivity": 0.20,
        "port_access": 0.20,
        "tariff_exposure": 0.15,
        "trade_volume": 0.15,
        "diversification": 0.10,
        "border_proximity": 0.10,
        "stability": 0.10,
    })

    def compute(self) -> float:
        w = self.WEIGHTS
        self.composite = (
            w["connectivity"] * self.connectivity_score
            + w["port_access"] * self.port_access_score
            + w["tariff_exposure"] * self.tariff_exposure_score
            + w["trade_volume"] * self.trade_volume_score
            + w["diversification"] * self.diversification_score
            + w["border_proximity"] * self.border_proximity_score
            + w["stability"] * self.stability_score
        )
        return self.composite


@dataclass
class OpportunitySignal:
    """Trade opportunity signal for a city."""
    city_id: str
    model_trade_flow: float       # Model's predicted normalized trade volume
    actual_trade_flow: float      # Actual observed
    gap: float                    # model - actual (positive = untapped potential)
    direction: str = ""           # OPPORTUNITY / RISK / NEUTRAL
    confidence: float = 0.0       # 0-1
