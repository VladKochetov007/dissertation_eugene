"""Smoke tests — graph loads, metrics compute, analyzers run."""

import sys
import pathlib

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from west_africa.core.graph import WestAfricaGraph
from west_africa.core.metrics import GraphMetrics
from west_africa.core.types import BlocMembership
from west_africa.signals.trade_impact import TradeImpactAnalyzer
from west_africa.signals.trade_route import TradeRouteAnalyzer
from west_africa.signals.cascade import EconomicCascadeSimulator
from west_africa.signals.opportunity_signal import OpportunitySignalGenerator


def _build_graph() -> WestAfricaGraph:
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    return WestAfricaGraph.from_seed_data(data_dir)


class TestGraphLoads:
    def test_node_count(self):
        g = _build_graph()
        assert g.node_count == 45, f"Expected 45 nodes, got {g.node_count}"

    def test_edge_count(self):
        g = _build_graph()
        assert g.edge_count >= 150, f"Expected >= 150 edges, got {g.edge_count}"

    def test_ftz_targets(self):
        g = _build_graph()
        targets = g.get_ftz_targets()
        assert len(targets) >= 20, f"Expected >= 20 FTZ targets, got {len(targets)}"

    def test_port_cities(self):
        g = _build_graph()
        ports = g.get_port_cities()
        assert len(ports) >= 10, f"Expected >= 10 port cities, got {len(ports)}"

    def test_bloc_membership(self):
        g = _build_graph()
        # Suspended countries should have SUSPENDED status
        assert g.get_effective_bloc("bamako") == BlocMembership.SUSPENDED
        assert g.get_effective_bloc("ouagadougou") == BlocMembership.SUSPENDED
        assert g.get_effective_bloc("niamey") == BlocMembership.SUSPENDED
        # Active ECOWAS
        assert g.get_effective_bloc("lagos") == BlocMembership.ECOWAS
        assert g.get_effective_bloc("accra") == BlocMembership.ECOWAS
        # UEMOA/CFA
        assert g.get_effective_bloc("dakar") == BlocMembership.UEMOA
        assert g.get_effective_bloc("abidjan") == BlocMembership.UEMOA

    def test_ecowas_subgraph(self):
        g = _build_graph()
        sub = g.get_ecowas_subgraph()
        # Should exclude suspended and external cities
        assert "bamako" not in sub
        assert "casablanca" not in sub
        assert "lagos" in sub
        assert "dakar" in sub

    def test_summary(self):
        g = _build_graph()
        s = g.summary()
        assert s["nodes"] == 45
        assert s["ecowas_active"] > 0
        assert s["uemoa_cfa"] > 0
        assert s["suspended"] > 0
        assert s["port_cities"] > 0
        assert s["ftz_targets"] > 0


class TestMetrics:
    def test_betweenness(self):
        g = _build_graph()
        m = GraphMetrics(g)
        bc = m.betweenness_centrality()
        assert len(bc) > 0
        assert all(0 <= v <= 1 for v in bc.values())

    def test_top_centrality(self):
        g = _build_graph()
        m = GraphMetrics(g)
        top = m.top_centrality("betweenness", top_n=5)
        assert len(top) == 5
        # Lagos or Accra should be in top 5 by betweenness
        top_ids = [t[0] for t in top]
        assert any(city in top_ids for city in ["lagos", "accra", "abidjan", "lome", "cotonou"]), \
            f"Expected a major hub in top 5, got {top_ids}"

    def test_articulation_points(self):
        g = _build_graph()
        m = GraphMetrics(g)
        ap = m.articulation_points()
        # Should return a list (may be empty if graph is well-connected)
        assert isinstance(ap, list)

    def test_component_count(self):
        g = _build_graph()
        m = GraphMetrics(g)
        cc = m.component_count()
        # Graph should be connected or nearly connected
        assert cc >= 1


class TestTradeImpact:
    def test_scores_in_range(self):
        g = _build_graph()
        m = GraphMetrics(g)
        analyzer = TradeImpactAnalyzer(g, m)
        scores = analyzer.score_all()
        assert len(scores) > 0
        for cid, ts in scores.items():
            assert 0.0 <= ts.composite <= 1.0, f"{cid}: composite {ts.composite} out of range"

    def test_top_impact(self):
        g = _build_graph()
        m = GraphMetrics(g)
        analyzer = TradeImpactAnalyzer(g, m)
        top = analyzer.top_impact(top_n=5)
        assert len(top) == 5
        # Scores should be descending
        for i in range(len(top) - 1):
            assert top[i][1] >= top[i + 1][1]


class TestTradeRoute:
    def test_shortest_route(self):
        g = _build_graph()
        analyzer = TradeRouteAnalyzer(g)
        path, cost = analyzer.shortest_trade_route("ouagadougou")
        assert len(path) > 0, "Should find a route to Ouagadougou"
        assert cost < float("inf")

    def test_route_unreachable(self):
        g = _build_graph()
        analyzer = TradeRouteAnalyzer(g)
        path, cost = analyzer.shortest_trade_route("nonexistent_city")
        assert len(path) == 0
        assert cost == float("inf")

    def test_route_risk_range(self):
        g = _build_graph()
        analyzer = TradeRouteAnalyzer(g)
        risk = analyzer.trade_route_risk("ouagadougou")
        assert 0.0 <= risk["trade_route_risk"] <= 1.0


class TestCascade:
    def test_exit_severity_range(self):
        g = _build_graph()
        sim = EconomicCascadeSimulator(g)
        result = sim.simulate_exit("lagos")
        assert 0.0 <= result.severity <= 1.0

    def test_exit_affects_country(self):
        g = _build_graph()
        sim = EconomicCascadeSimulator(g)
        result = sim.simulate_exit("bamako")
        # Should affect all Mali cities
        mali_cities = {c.id for c in g.get_by_country("MLI")}
        assert mali_cities.issubset(set(result.affected_nodes))

    def test_multi_exit(self):
        g = _build_graph()
        sim = EconomicCascadeSimulator(g)
        # Simulate the Alliance of Sahel States all exiting
        result = sim.simulate_multi_exit(["bamako", "ouagadougou", "niamey"])
        assert result.severity > 0

    def test_scenario_report(self):
        g = _build_graph()
        sim = EconomicCascadeSimulator(g)
        report = sim.scenario_report("lagos", "exit")
        assert "scenario" in report
        assert "severity" in report
        assert report["severity"] >= 0


class TestOpportunitySignals:
    def test_signals_generated(self):
        g = _build_graph()
        m = GraphMetrics(g)
        gen = OpportunitySignalGenerator(g, m)
        signals = gen.generate_all_signals()
        assert len(signals) > 0

    def test_signal_direction(self):
        g = _build_graph()
        m = GraphMetrics(g)
        gen = OpportunitySignalGenerator(g, m)
        signals = gen.generate_all_signals()
        for cid, sig in signals.items():
            assert sig.direction in ("OPPORTUNITY", "RISK", "NEUTRAL")
            assert 0.0 <= sig.confidence <= 1.0

    def test_top_opportunities(self):
        g = _build_graph()
        m = GraphMetrics(g)
        gen = OpportunitySignalGenerator(g, m)
        top = gen.top_opportunities(top_n=5)
        assert len(top) <= 5
        # Should be sorted by gap descending
        for i in range(len(top) - 1):
            assert top[i].gap >= top[i + 1].gap
