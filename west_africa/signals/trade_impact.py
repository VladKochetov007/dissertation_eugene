"""FTZ impact scoring for cities based on graph + economic indicators."""

from __future__ import annotations
from typing import TYPE_CHECKING
from ..core.types import BlocMembership, TradeImpactScore

if TYPE_CHECKING:
    from ..core.graph import WestAfricaGraph
    from ..core.metrics import GraphMetrics


class TradeImpactAnalyzer:
    """Compute composite FTZ impact scores for cities."""

    PORT_CITIES = ["lagos", "abidjan", "dakar", "tema", "lome", "cotonou", "conakry",
                   "freetown", "monrovia", "port_harcourt", "douala", "bissau", "banjul", "praia",
                   "casablanca", "accra", "nouakchott"]

    def __init__(self, graph: "WestAfricaGraph", metrics: "GraphMetrics") -> None:
        self.wag = graph
        self.metrics = metrics

    def score_all(self) -> dict[str, TradeImpactScore]:
        """Score all FTZ target cities."""
        betweenness = self.metrics.betweenness_centrality()
        results: dict[str, TradeImpactScore] = {}

        # Compute max trade volume for normalization
        max_trade_vol = max(
            (sum(d.get("volume", 0) for _, _, d in self.wag.G.edges(cid, data=True))
             for cid in self.wag.cities),
            default=1.0
        )

        for cid, city in self.wag.cities.items():
            if not city.is_ftz_target:
                continue

            es = self.wag.economic_states.get(cid)
            ts = TradeImpactScore(city_id=cid)

            # 1. Connectivity: higher betweenness = more central to trade network
            ts.connectivity_score = min(betweenness.get(cid, 0.0) / 0.3, 1.0)

            # 2. Port access: port cities score 1.0, others based on distance to nearest port
            if city.is_port:
                ts.port_access_score = 1.0
            else:
                # Approximate: use shortest path weight to nearest port as proxy
                ts.port_access_score = self._port_access(cid)

            # 3. Economic diversification: number of unique trade partner countries
            partner_countries = set()
            for _, neighbor, data in self.wag.G.edges(cid, data=True):
                if data.get("edge_type") == "TRADE":
                    neighbor_city = self.wag.cities.get(neighbor)
                    if neighbor_city:
                        partner_countries.add(neighbor_city.country_iso3)
            ts.diversification_score = min(len(partner_countries) / 10.0, 1.0)

            # 4. Tariff exposure: average tariff on trade edges (higher = more to gain from FTZ)
            tariffs = [d.get("tariff_rate", 0.0)
                       for _, _, d in self.wag.G.edges(cid, data=True)
                       if d.get("edge_type") == "TRADE" and d.get("tariff_rate", 0) > 0]
            ts.tariff_exposure_score = sum(tariffs) / len(tariffs) / 0.20 if tariffs else 0.0
            ts.tariff_exposure_score = min(ts.tariff_exposure_score, 1.0)

            # 5. Border proximity: cities closer to international borders benefit more
            cross_border_edges = sum(
                1 for _, neighbor, _ in self.wag.G.edges(cid, data=True)
                if self.wag.cities.get(neighbor) and
                self.wag.cities[neighbor].country_iso3 != city.country_iso3
            )
            ts.border_proximity_score = min(cross_border_edges / 8.0, 1.0)

            # 6. Trade volume: current trade volume relative to network max
            city_trade_vol = sum(
                d.get("volume", 0.0) for _, _, d in self.wag.G.edges(cid, data=True)
                if d.get("edge_type") == "TRADE"
            )
            ts.trade_volume_score = min(city_trade_vol / max(max_trade_vol, 1.0), 1.0)

            # 7. Political stability
            ts.stability_score = es.political_stability if es else 0.5

            ts.compute()
            results[cid] = ts

        return results

    def _port_access(self, cid: str) -> float:
        """Compute port access score based on shortest path to nearest port."""
        import networkx as nx
        simple = self.wag.get_active_edges_graph()
        best_cost = float("inf")
        for port_id in self.PORT_CITIES:
            if port_id in simple and cid in simple:
                try:
                    cost = nx.shortest_path_length(simple, cid, port_id, weight="weight")
                    best_cost = min(best_cost, cost)
                except nx.NetworkXNoPath:
                    pass
        # Normalize: cost of 0 = score 1.0, cost of 5.0+ = score 0.0
        if best_cost == float("inf"):
            return 0.0
        return max(0.0, 1.0 - best_cost / 5.0)

    def top_impact(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Return top-N cities by FTZ impact score."""
        scores = self.score_all()
        ranked = sorted(scores.items(), key=lambda x: x[1].composite, reverse=True)
        return [(cid, ts.composite) for cid, ts in ranked[:top_n]]
