"""Export the full West Africa FTZ analysis pipeline to a JSON file
consumed by the Next.js dashboard.

Usage:
    cd /Users/meuge/Coding/maynard && python -m west_africa.scripts.export_dashboard_data
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

# Ensure project root is on sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from west_africa.core.graph import WestAfricaGraph
from west_africa.core.metrics import GraphMetrics
from west_africa.signals.trade_impact import TradeImpactAnalyzer
from west_africa.signals.trade_route import TradeRouteAnalyzer
from west_africa.signals.cascade import EconomicCascadeSimulator
from west_africa.signals.opportunity_signal import OpportunitySignalGenerator

# Paths
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "lib"
OUTPUT_FILE = OUTPUT_DIR / "analysis-data.json"


def _round_dict(d: dict[str, float], decimals: int = 4) -> dict[str, float]:
    """Round all float values in a dict."""
    return {k: round(v, decimals) for k, v in d.items()}


def build_summary(graph: WestAfricaGraph) -> dict:
    """Build the top-level summary from the graph."""
    return graph.summary()


def build_metrics(metrics: GraphMetrics) -> dict:
    """Compute all structural graph metrics."""
    return {
        "betweenness": _round_dict(metrics.betweenness_centrality()),
        "degree": _round_dict(metrics.degree_centrality()),
        "closeness": _round_dict(metrics.closeness_centrality()),
        "articulation_points": metrics.articulation_points(),
        "bridges": [list(pair) for pair in metrics.bridges()],
        "ecowas_cut_vertices": metrics.ecowas_articulation_points(),
        "component_count": metrics.component_count(),
    }


def build_ftz_impact(graph: WestAfricaGraph, metrics: GraphMetrics) -> dict:
    """Compute FTZ trade impact scores for all target cities."""
    analyzer = TradeImpactAnalyzer(graph, metrics)
    scores = analyzer.score_all()

    result: dict = {}
    for cid, ts in scores.items():
        result[cid] = {
            "composite": round(ts.composite, 4),
            "connectivity": round(ts.connectivity_score, 4),
            "port_access": round(ts.port_access_score, 4),
            "tariff_exposure": round(ts.tariff_exposure_score, 4),
            "trade_volume": round(ts.trade_volume_score, 4),
            "diversification": round(ts.diversification_score, 4),
            "border_proximity": round(ts.border_proximity_score, 4),
            "stability": round(ts.stability_score, 4),
        }
    return result


def build_trade_routes(graph: WestAfricaGraph) -> dict:
    """Compute trade route vulnerability for all FTZ target cities."""
    analyzer = TradeRouteAnalyzer(graph)
    targets = graph.get_ftz_targets()

    result: dict = {}
    for city in targets:
        risk_data = analyzer.trade_route_risk(city.id)
        # Handle infinite cost (unreachable)
        cost = risk_data["shortest_cost"]
        if cost == float("inf"):
            cost = None

        result[city.id] = {
            "risk": risk_data["trade_route_risk"],
            "redundancy": risk_data["route_redundancy"],
            "min_cut": risk_data["min_cut_size"],
            "shortest_path": risk_data["shortest_path"],
            "shortest_cost": cost,
            "min_cut_nodes": risk_data["min_cut_nodes"],
        }
    return result


def _cascade_to_dict(result, name: str) -> dict:
    """Convert a CascadeResult dataclass to a JSON-serializable dict."""
    return {
        "name": name,
        "trigger": result.trigger_node,
        "type": result.scenario_type,
        "affected_cities": result.affected_nodes,
        "isolated_cities": result.isolated_nodes,
        "trade_disrupted_cities": result.trade_disrupted_nodes,
        "new_components": result.new_component_count,
        "trade_volume_affected": round(result.trade_volume_affected, 1),
        "severity": round(result.severity, 4),
    }


def build_cascades(graph: WestAfricaGraph) -> list[dict]:
    """Run cascade simulations and return results."""
    sim = EconomicCascadeSimulator(graph)
    cascades: list[dict] = []

    # Nigeria exits ECOWAS
    nga_cities = graph.get_by_country("NGA")
    if nga_cities:
        r = sim.simulate_exit(nga_cities[0].id)
        cascades.append(_cascade_to_dict(r, "Nigeria exits ECOWAS"))

    # Mali exits ECOWAS
    mli_cities = graph.get_by_country("MLI")
    if mli_cities:
        r = sim.simulate_exit(mli_cities[0].id)
        cascades.append(_cascade_to_dict(r, "Mali exits ECOWAS"))

    # Alliance of Sahel States multi-exit (Mali + Burkina Faso + Niger)
    sahel_triggers = []
    for iso3 in ("MLI", "BFA", "NER"):
        country_cities = graph.get_by_country(iso3)
        if country_cities:
            sahel_triggers.append(country_cities[0].id)
    if sahel_triggers:
        r = sim.simulate_multi_exit(sahel_triggers)
        cascades.append(
            _cascade_to_dict(r, "Alliance of Sahel States multi-exit")
        )

    # Guinea exits ECOWAS
    gin_cities = graph.get_by_country("GIN")
    if gin_cities:
        r = sim.simulate_exit(gin_cities[0].id)
        cascades.append(_cascade_to_dict(r, "Guinea exits ECOWAS"))

    return cascades


def build_opportunities(
    graph: WestAfricaGraph, metrics: GraphMetrics
) -> list[dict]:
    """Generate opportunity signals for all FTZ target cities."""
    gen = OpportunitySignalGenerator(graph, metrics)
    all_signals = gen.generate_all_signals()

    opportunities: list[dict] = []
    for cid, sig in sorted(all_signals.items(), key=lambda x: x[1].gap, reverse=True):
        city = graph.cities.get(cid)
        opportunities.append({
            "city_id": cid,
            "city_name": city.name if city else cid,
            "country": city.country if city else "Unknown",
            "signal_type": sig.direction,
            "gap": round(sig.gap, 4),
            "model_score": round(sig.model_trade_flow, 4),
            "actual_score": round(sig.actual_trade_flow, 4),
            "confidence": round(sig.confidence, 4),
        })
    return opportunities


def main() -> None:
    print("West Africa FTZ -- Exporting dashboard data")
    print("=" * 60)

    # Step 1: Load graph
    print("  [1/6] Loading graph from seed data...")
    graph = WestAfricaGraph.from_seed_data()
    print(f"         {graph.node_count} nodes, {graph.edge_count} edges")

    # Step 2: Compute metrics
    print("  [2/6] Computing graph metrics...")
    metrics = GraphMetrics(graph)

    # Step 3: FTZ impact
    print("  [3/6] Scoring FTZ trade impact...")
    ftz_impact = build_ftz_impact(graph, metrics)
    print(f"         {len(ftz_impact)} cities scored")

    # Step 4: Trade routes
    print("  [4/6] Analyzing trade route vulnerability...")
    trade_routes = build_trade_routes(graph)
    print(f"         {len(trade_routes)} routes analyzed")

    # Step 5: Cascade simulations
    print("  [5/6] Running cascade simulations...")
    cascades = build_cascades(graph)
    print(f"         {len(cascades)} scenarios simulated")

    # Step 6: Opportunity signals
    print("  [6/6] Generating opportunity signals...")
    opportunities = build_opportunities(graph, metrics)
    opp_count = sum(1 for o in opportunities if o["signal_type"] == "OPPORTUNITY")
    risk_count = sum(1 for o in opportunities if o["signal_type"] == "RISK")
    neutral_count = sum(1 for o in opportunities if o["signal_type"] == "NEUTRAL")
    print(f"         {opp_count} opportunities, {risk_count} risks, {neutral_count} neutral")

    # Load raw seed data for inclusion
    with open(DATA_DIR / "cities.json") as f:
        raw_cities = json.load(f)
    with open(DATA_DIR / "edges.json") as f:
        raw_edges = json.load(f)

    # Assemble final JSON
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": build_summary(graph),
        "cities": raw_cities,
        "edges": raw_edges,
        "metrics": build_metrics(metrics),
        "ftz_impact": ftz_impact,
        "trade_routes": trade_routes,
        "cascades": cascades,
        "opportunities": opportunities,
    }

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    file_size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\n  Output written to: {OUTPUT_FILE}")
    print(f"  File size: {file_size_kb:.1f} KB")
    print("\nExport complete.")


if __name__ == "__main__":
    main()
