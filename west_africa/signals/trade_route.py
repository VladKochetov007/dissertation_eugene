"""Trade route analysis -- corridor analysis, bottlenecks, port access."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import networkx as nx
from ..core.types import BlocMembership

if TYPE_CHECKING:
    from ..core.graph import WestAfricaGraph


class TradeRouteAnalyzer:
    """Analyze trade route vulnerability from ports to inland cities."""

    # Major port cities serving as trade origins
    TRADE_ORIGINS = ["lagos", "abidjan", "dakar", "tema", "lome", "cotonou", "conakry"]

    def __init__(self, graph: "WestAfricaGraph") -> None:
        self.wag = graph

    def shortest_trade_route(
        self, target: str, origin: Optional[str] = None
    ) -> tuple[list[str], float]:
        """Find shortest weighted path from port to target city."""
        origins = [origin] if origin else self.TRADE_ORIGINS
        simple = self.wag.get_active_edges_graph()
        best_path, best_cost = [], float("inf")
        for src in origins:
            try:
                path = nx.shortest_path(simple, src, target, weight="weight")
                cost = nx.shortest_path_length(simple, src, target, weight="weight")
                if cost < best_cost:
                    best_path, best_cost = path, cost
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        return best_path, best_cost

    def all_simple_routes(
        self, target: str, origin: Optional[str] = None, cutoff: int = 8
    ) -> list[list[str]]:
        """Find all simple routes (up to cutoff length) -- measures redundancy."""
        origins = [origin] if origin else self.TRADE_ORIGINS[:3]  # Limit for performance
        simple = self.wag.get_active_edges_graph()
        all_paths = []
        for src in origins:
            try:
                paths = list(nx.all_simple_paths(simple, src, target, cutoff=cutoff))
                all_paths.extend(paths)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        return all_paths

    def route_redundancy(
        self, target: str, origin: Optional[str] = None, cutoff: int = 8
    ) -> int:
        """Count of distinct trade routes -- higher = more resilient."""
        return len(self.all_simple_routes(target, origin, cutoff))

    def min_cut_nodes(
        self, target: str, origin: Optional[str] = None
    ) -> tuple[int, set[str]]:
        """Minimum node cut between origin and target.

        Low cut_size = fragile trade corridor.
        """
        if origin:
            origins = [origin]
        else:
            origins = self.TRADE_ORIGINS
        simple = self.wag.get_active_edges_graph()
        best_cut_value = float("inf")
        best_cut_nodes: set[str] = set()
        for src in origins:
            try:
                cut_value = nx.node_connectivity(simple, src, target)
                if cut_value < best_cut_value:
                    best_cut_value = cut_value
                    best_cut_nodes = nx.minimum_node_cut(simple, src, target)
            except (nx.NetworkXError, nx.NodeNotFound):
                continue
        return int(best_cut_value) if best_cut_value != float("inf") else 0, best_cut_nodes

    def trade_route_risk(
        self, target: str, origin: Optional[str] = None
    ) -> dict:
        """Composite trade route risk assessment for a city."""
        path, cost = self.shortest_trade_route(target, origin)
        redundancy = self.route_redundancy(target, origin)
        cut_size, cut_nodes = self.min_cut_nodes(target, origin)

        risk = 0.0
        if redundancy > 0:
            risk = 1.0 - min(redundancy / 10.0, 1.0)
        if cut_size > 0:
            risk = max(risk, 1.0 - min(cut_size / 5.0, 1.0))

        return {
            "target": target,
            "origin": origin or "nearest_port",
            "shortest_path": path,
            "shortest_cost": cost,
            "route_redundancy": redundancy,
            "min_cut_size": cut_size,
            "min_cut_nodes": list(cut_nodes),
            "trade_route_risk": round(risk, 3),
        }

    def score_all_targets(self) -> list[dict]:
        """Trade route risk for all FTZ target cities."""
        targets = self.wag.get_ftz_targets()
        results = []
        for t in targets:
            if not t.is_port:  # Only score non-port cities (ports have direct access)
                results.append(self.trade_route_risk(t.id))
        return sorted(results, key=lambda x: x["trade_route_risk"], reverse=True)
