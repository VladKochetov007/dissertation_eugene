"""Economic cascade simulation -- 'what if country X exits/joins FTZ?' analysis."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import networkx as nx
from ..core.types import BlocMembership

if TYPE_CHECKING:
    from ..core.graph import WestAfricaGraph


@dataclass
class CascadeResult:
    """Result of an economic cascade simulation."""
    trigger_node: str
    scenario_type: str               # 'exit', 'entry', 'tariff_removal'
    affected_nodes: list[str] = field(default_factory=list)
    isolated_nodes: list[str] = field(default_factory=list)
    trade_disrupted_nodes: list[str] = field(default_factory=list)
    new_component_count: int = 0
    trade_volume_affected: float = 0.0  # USD millions
    severity: float = 0.0              # 0-1


class EconomicCascadeSimulator:
    """Simulate cascading effects of FTZ membership changes."""

    def __init__(self, graph: "WestAfricaGraph") -> None:
        self.wag = graph

    def simulate_exit(self, node_id: str) -> CascadeResult:
        """Simulate what happens if a city's country exits ECOWAS/FTZ.

        Steps:
        1. Remove all TRADE and FINANCIAL edges to/from the node's country
        2. Check if ECOWAS subgraph fragments
        3. Identify trade disruption for connected nodes
        4. Score severity by affected trade volume
        """
        result = CascadeResult(trigger_node=node_id, scenario_type="exit")

        city = self.wag.get_city(node_id)
        if not city:
            return result

        # Find all cities in the same country
        country_cities = {c.id for c in self.wag.get_by_country(city.country_iso3)}
        result.affected_nodes = sorted(country_cities)

        # Build subgraph without the exiting country's trade/financial edges
        ecowas_sub = self.wag.get_ecowas_subgraph()

        # Remove nodes from exiting country
        for cid in country_cities:
            if cid in ecowas_sub:
                ecowas_sub.remove_node(cid)

        simple = nx.Graph(ecowas_sub)
        components = list(nx.connected_components(simple))
        result.new_component_count = len(components)

        # Find largest component
        if components:
            largest = max(components, key=len)
            all_remaining = set(simple.nodes())
            isolated = all_remaining - largest
            result.isolated_nodes = sorted(isolated)

        # Calculate trade volume affected
        total_trade = 0.0
        for u, v, data in self.wag.G.edges(data=True):
            if data.get("edge_type") == "TRADE":
                if u in country_cities or v in country_cities:
                    total_trade += data.get("volume", 0.0)
        result.trade_volume_affected = total_trade

        # Identify trade-disrupted nodes (neighbors that lose trade connections)
        disrupted = set()
        for cid in country_cities:
            for _, neighbor, data in self.wag.G.edges(cid, data=True):
                if neighbor not in country_cities and data.get("edge_type") == "TRADE":
                    disrupted.add(neighbor)
        result.trade_disrupted_nodes = sorted(disrupted)

        # Severity: fraction of total ECOWAS trade affected
        total_ecowas_trade = sum(
            d.get("volume", 0.0) for _, _, d in self.wag.G.edges(data=True)
            if d.get("edge_type") == "TRADE"
        )
        if total_ecowas_trade > 0:
            result.severity = min(total_trade / total_ecowas_trade, 1.0)

        # Bonus severity for isolated FTZ targets
        ftz_isolated = [
            n for n in result.isolated_nodes
            if n in self.wag.cities and self.wag.cities[n].is_ftz_target
        ]
        result.severity = min(result.severity + 0.05 * len(ftz_isolated), 1.0)

        return result

    def simulate_entry(self, node_id: str) -> CascadeResult:
        """Simulate positive effects of a suspended country re-joining."""
        result = CascadeResult(trigger_node=node_id, scenario_type="entry")

        city = self.wag.get_city(node_id)
        if not city:
            return result

        country_cities = {c.id for c in self.wag.get_by_country(city.country_iso3)}
        result.affected_nodes = sorted(country_cities)

        # Count trade volume that would be restored
        restored_trade = 0.0
        benefited = set()
        for cid in country_cities:
            for _, neighbor, data in self.wag.G.edges(cid, data=True):
                if data.get("edge_type") == "TRADE" and neighbor not in country_cities:
                    restored_trade += data.get("volume", 0.0)
                    benefited.add(neighbor)

        result.trade_volume_affected = restored_trade
        result.trade_disrupted_nodes = sorted(benefited)  # reusing as "benefited"

        total_ecowas_trade = sum(
            d.get("volume", 0.0) for _, _, d in self.wag.G.edges(data=True)
            if d.get("edge_type") == "TRADE"
        )
        if total_ecowas_trade > 0:
            result.severity = min(restored_trade / total_ecowas_trade, 1.0)

        return result

    def simulate_multi_exit(self, node_ids: list[str]) -> CascadeResult:
        """Simulate multiple countries exiting simultaneously."""
        result = CascadeResult(
            trigger_node=node_ids[0] if node_ids else "",
            scenario_type="exit",
        )

        all_country_cities: set[str] = set()
        for nid in node_ids:
            city = self.wag.get_city(nid)
            if city:
                all_country_cities |= {c.id for c in self.wag.get_by_country(city.country_iso3)}

        result.affected_nodes = sorted(all_country_cities)

        ecowas_sub = self.wag.get_ecowas_subgraph()
        for cid in all_country_cities:
            if cid in ecowas_sub:
                ecowas_sub.remove_node(cid)

        simple = nx.Graph(ecowas_sub)
        components = list(nx.connected_components(simple))
        result.new_component_count = len(components)

        if components:
            largest = max(components, key=len)
            isolated = set(simple.nodes()) - largest
            result.isolated_nodes = sorted(isolated)

        total_trade = sum(
            d.get("volume", 0.0) for u, v, d in self.wag.G.edges(data=True)
            if d.get("edge_type") == "TRADE" and (u in all_country_cities or v in all_country_cities)
        )
        result.trade_volume_affected = total_trade

        total_ecowas_trade = sum(
            d.get("volume", 0.0) for _, _, d in self.wag.G.edges(data=True)
            if d.get("edge_type") == "TRADE"
        )
        result.severity = min(total_trade / max(total_ecowas_trade, 1), 1.0)

        return result

    def scenario_report(self, node_id: str, scenario_type: str = "exit") -> dict:
        """Generate a human-readable scenario report."""
        if scenario_type == "exit":
            r = self.simulate_exit(node_id)
        else:
            r = self.simulate_entry(node_id)

        city = self.wag.cities.get(node_id)
        name = city.name if city else node_id
        country = city.country if city else "Unknown"

        return {
            "scenario": f"If {country} ({name}) {'exits' if scenario_type == 'exit' else 'joins'} FTZ",
            "affected_cities": r.affected_nodes,
            "affected_city_names": [
                self.wag.cities[n].name for n in r.affected_nodes if n in self.wag.cities
            ],
            "isolated_cities": r.isolated_nodes,
            "trade_disrupted_cities": r.trade_disrupted_nodes,
            "trade_volume_affected_usd_m": round(r.trade_volume_affected, 1),
            "new_components": r.new_component_count,
            "severity": round(r.severity, 3),
        }
