"""Trade opportunity signal generation -- identify untapped trade potential."""

from __future__ import annotations
from typing import TYPE_CHECKING
from ..core.types import OpportunitySignal

if TYPE_CHECKING:
    from ..core.graph import WestAfricaGraph
    from ..core.metrics import GraphMetrics


class OpportunitySignalGenerator:
    """Generate trade opportunity signals by comparing model predictions to actual flows."""

    OPPORTUNITY_THRESHOLD = 0.10   # 10% gap = opportunity
    RISK_THRESHOLD = -0.10         # -10% gap = risk

    def __init__(self, graph: "WestAfricaGraph", metrics: "GraphMetrics") -> None:
        self.wag = graph
        self.metrics = metrics

    def model_trade_flow(self, city_id: str) -> float:
        """Predict normalized trade flow based on structural indicators.

        Combines: 40% connectivity, 30% port access, 20% stability, 10% base.
        Returns a 0-1 normalized score.
        """
        betweenness = self.metrics.betweenness_centrality()
        city = self.wag.cities.get(city_id)
        if not city:
            return 0.0

        connectivity = min(betweenness.get(city_id, 0.0) / 0.3, 1.0)
        port_bonus = 0.3 if city.is_port else 0.0
        es = self.wag.economic_states.get(city_id)
        stability = es.political_stability if es else 0.5
        base = 0.3  # baseline trade expectation

        return 0.40 * connectivity + 0.30 * port_bonus + 0.20 * stability + 0.10 * base

    def actual_trade_flow(self, city_id: str) -> float:
        """Compute normalized actual trade flow from edge volumes."""
        total_vol = sum(
            d.get("volume", 0.0) for _, _, d in self.wag.G.edges(city_id, data=True)
            if d.get("edge_type") == "TRADE"
        )
        # Normalize by max observed trade volume in network
        max_vol = max(
            (sum(d.get("volume", 0.0) for _, _, d in self.wag.G.edges(cid, data=True)
                 if d.get("edge_type") == "TRADE")
             for cid in self.wag.cities),
            default=1.0
        )
        return total_vol / max(max_vol, 1.0)

    def generate_signal(self, city_id: str) -> OpportunitySignal:
        """Generate opportunity signal for a city."""
        model = self.model_trade_flow(city_id)
        actual = self.actual_trade_flow(city_id)
        gap = model - actual

        if gap > self.OPPORTUNITY_THRESHOLD:
            direction = "OPPORTUNITY"
        elif gap < self.RISK_THRESHOLD:
            direction = "RISK"
        else:
            direction = "NEUTRAL"

        confidence = min(abs(gap) / 0.30, 1.0)

        return OpportunitySignal(
            city_id=city_id,
            model_trade_flow=round(model, 4),
            actual_trade_flow=round(actual, 4),
            gap=round(gap, 4),
            direction=direction,
            confidence=round(confidence, 4),
        )

    def generate_all_signals(self) -> dict[str, OpportunitySignal]:
        """Generate signals for all FTZ target cities."""
        targets = self.wag.get_ftz_targets()
        return {t.id: self.generate_signal(t.id) for t in targets}

    def top_opportunities(self, top_n: int = 10) -> list[OpportunitySignal]:
        """Return top-N cities by trade gap (untapped potential)."""
        signals = self.generate_all_signals()
        ranked = sorted(signals.values(), key=lambda s: s.gap, reverse=True)
        return ranked[:top_n]
