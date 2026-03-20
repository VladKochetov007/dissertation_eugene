"""Core West Africa multigraph built on NetworkX."""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import networkx as nx

from .types import BlocMembership, ConnectionType, City, TradeEdge, EconomicState


DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


class WestAfricaGraph:
    """Multigraph of West African cities and connections."""

    def __init__(self) -> None:
        self.G: nx.MultiGraph = nx.MultiGraph()
        self.cities: dict[str, City] = {}
        self.economic_states: dict[str, EconomicState] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_seed_data(cls, data_dir: Optional[pathlib.Path] = None) -> "WestAfricaGraph":
        """Build graph from JSON seed files."""
        d = data_dir or DATA_DIR
        g = cls()
        g._load_cities(d / "cities.json")
        g._load_edges(d / "edges.json")
        g._load_economic_state(d / "economic_state.json")
        return g

    def _load_cities(self, path: pathlib.Path) -> None:
        with open(path) as f:
            raw = json.load(f)
        for c in raw:
            city = City(
                id=c["id"],
                name=c["name"],
                lat=c["lat"],
                lng=c["lng"],
                country=c["country"],
                country_iso3=c["country_iso3"],
                bloc=BlocMembership(c["bloc"]),
                population=c.get("population", 0),
                is_port=c.get("is_port", False),
                is_capital=c.get("is_capital", False),
                gdp_per_capita=c.get("gdp_per_capita", 0.0),
                trade_openness=c.get("trade_openness", 0.0),
                ease_of_business=c.get("ease_of_business", 0.0),
                cfa_zone=c.get("cfa_zone", False),
                is_ftz_target=c.get("is_ftz_target", False),
                tags=c.get("tags", []),
            )
            self.cities[city.id] = city
            self.G.add_node(
                city.id,
                name=city.name,
                country=city.country,
                bloc=city.bloc.value,
                lat=city.lat,
                lng=city.lng,
                population=city.population,
                is_port=city.is_port,
                is_capital=city.is_capital,
                gdp_per_capita=city.gdp_per_capita,
                trade_openness=city.trade_openness,
                cfa_zone=city.cfa_zone,
                is_ftz_target=city.is_ftz_target,
            )

    def _load_edges(self, path: pathlib.Path) -> None:
        with open(path) as f:
            raw = json.load(f)
        for e in raw:
            edge = TradeEdge(
                source=e["source"],
                target=e["target"],
                edge_type=ConnectionType(e["edge_type"]),
                weight=e.get("weight", 1.0),
                volume=e.get("volume", 0.0),
                distance_km=e.get("distance_km", 0.0),
                is_active=e.get("is_active", True),
                tariff_rate=e.get("tariff_rate", 0.0),
                description=e.get("description", ""),
            )
            self.G.add_edge(
                edge.source,
                edge.target,
                key=edge.edge_type.value,
                weight=edge.weight,
                volume=edge.volume,
                distance_km=edge.distance_km,
                is_active=edge.is_active,
                tariff_rate=edge.tariff_rate,
                edge_type=edge.edge_type.value,
            )

    def _load_economic_state(self, path: pathlib.Path) -> None:
        with open(path) as f:
            raw = json.load(f)
        for es in raw:
            state = EconomicState(
                city_id=es["city_id"],
                bloc_override=(
                    BlocMembership(es["bloc_override"])
                    if es.get("bloc_override")
                    else None
                ),
                gdp_growth_rate=es.get("gdp_growth_rate", 0.0),
                trade_volume_change=es.get("trade_volume_change", 0.0),
                fdi_inflow_change=es.get("fdi_inflow_change", 0.0),
                inflation_rate=es.get("inflation_rate", 0.0),
                political_stability=es.get("political_stability", 0.5),
                tariff_change=es.get("tariff_change", 0.0),
                last_updated=es.get("last_updated", ""),
            )
            self.economic_states[state.city_id] = state
            # apply bloc override to graph
            if state.bloc_override and state.city_id in self.G:
                self.G.nodes[state.city_id]["bloc"] = state.bloc_override.value

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_city(self, cid: str) -> Optional[City]:
        return self.cities.get(cid)

    def get_effective_bloc(self, cid: str) -> BlocMembership:
        es = self.economic_states.get(cid)
        if es and es.bloc_override:
            return es.bloc_override
        c = self.cities.get(cid)
        return c.bloc if c else BlocMembership.EXTERNAL

    def get_ftz_targets(self) -> list[City]:
        return [c for c in self.cities.values() if c.is_ftz_target]

    def get_port_cities(self) -> list[City]:
        return [c for c in self.cities.values() if c.is_port]

    def get_by_bloc(self, status: BlocMembership) -> list[City]:
        return [
            c for c in self.cities.values()
            if self.get_effective_bloc(c.id) == status
        ]

    def get_ecowas_subgraph(self) -> nx.MultiGraph:
        """Subgraph of active ECOWAS + UEMOA members."""
        nodes = [
            c.id for c in self.cities.values()
            if self.get_effective_bloc(c.id) in (BlocMembership.ECOWAS, BlocMembership.UEMOA)
        ]
        return self.G.subgraph(nodes).copy()

    def get_by_country(self, country_iso3: str) -> list[City]:
        return [c for c in self.cities.values() if c.country_iso3 == country_iso3]

    def get_active_edges_graph(self) -> nx.Graph:
        """Simple graph with only active edges (for pathfinding)."""
        simple = nx.Graph()
        for u, v, key, data in self.G.edges(keys=True, data=True):
            if data.get("is_active", True):
                w = data.get("weight", 1.0)
                if simple.has_edge(u, v):
                    if w < simple[u][v]["weight"]:
                        simple[u][v]["weight"] = w
                else:
                    simple.add_edge(u, v, weight=w, volume=data.get("volume", 0.0))
        return simple

    def get_edges_by_type(self, edge_type: ConnectionType) -> list[tuple[str, str, dict]]:
        """Get all edges of a specific type."""
        result = []
        for u, v, key, data in self.G.edges(keys=True, data=True):
            if data.get("edge_type") == edge_type.value:
                result.append((u, v, data))
        return result

    @property
    def node_count(self) -> int:
        return self.G.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.G.number_of_edges()

    def summary(self) -> dict:
        ecowas = len(self.get_by_bloc(BlocMembership.ECOWAS))
        uemoa = len(self.get_by_bloc(BlocMembership.UEMOA))
        suspended = len(self.get_by_bloc(BlocMembership.SUSPENDED))
        external = len(self.get_by_bloc(BlocMembership.EXTERNAL))
        ports = len(self.get_port_cities())
        targets = len(self.get_ftz_targets())
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "ecowas_active": ecowas,
            "uemoa_cfa": uemoa,
            "suspended": suspended,
            "external": external,
            "port_cities": ports,
            "ftz_targets": targets,
        }
