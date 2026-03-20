"""Full analysis pipeline: load graph, compute metrics, run all signal analyzers.

Usage:
    python -m west_africa.scripts.run_analysis
    python -m west_africa.scripts.run_analysis --top-n 15
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

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


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def _subheader(title: str) -> None:
    print(f"\n  --- {title} ---")


def _table(headers: list[str], rows: list[list[Any]], col_widths: list[int] | None = None) -> None:
    """Print a simple text table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)

    # Header
    header_line = "  "
    sep_line = "  "
    for h, w in zip(headers, col_widths):
        header_line += str(h).ljust(w)
        sep_line += "-" * (w - 1) + " "
    print(header_line)
    print(sep_line)

    # Rows
    for row in rows:
        line = "  "
        for val, w in zip(row, col_widths):
            line += str(val).ljust(w)
        print(line)


def _cascade_summary(result: Any, graph: WestAfricaGraph, label: str) -> None:
    """Print cascade simulation result."""
    _subheader(label)

    affected_names = [
        graph.cities[n].name for n in result.affected_nodes if n in graph.cities
    ]
    isolated_names = [
        graph.cities[n].name for n in result.isolated_nodes if n in graph.cities
    ]
    disrupted_names = [
        graph.cities[n].name for n in result.trade_disrupted_nodes if n in graph.cities
    ]

    print(f"  Scenario:            {result.scenario_type}")
    print(f"  Trigger:             {result.trigger_node}")
    print(f"  Affected cities:     {len(result.affected_nodes)} ({', '.join(affected_names[:5])}{'...' if len(affected_names) > 5 else ''})")
    print(f"  Isolated cities:     {len(result.isolated_nodes)} ({', '.join(isolated_names[:5])}{'...' if len(isolated_names) > 5 else ''})")
    print(f"  Trade-disrupted:     {len(result.trade_disrupted_nodes)} ({', '.join(disrupted_names[:5])}{'...' if len(disrupted_names) > 5 else ''})")
    print(f"  New components:      {result.new_component_count}")
    print(f"  Trade vol affected:  ${result.trade_volume_affected:,.1f}M")
    print(f"  Severity:            {result.severity:.3f}")


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full West Africa FTZ analysis pipeline."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top results to show in each ranking (default: 10)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    top_n = args.top_n

    print("West Africa FTZ -- Full Analysis Pipeline")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Load graph
    # ------------------------------------------------------------------
    _header("STEP 1: LOAD GRAPH")
    graph = WestAfricaGraph.from_seed_data()
    summary = graph.summary()
    print(f"  Nodes:          {summary['nodes']}")
    print(f"  Edges:          {summary['edges']}")
    print(f"  ECOWAS active:  {summary['ecowas_active']}")
    print(f"  UEMOA (CFA):    {summary['uemoa_cfa']}")
    print(f"  Suspended:      {summary['suspended']}")
    print(f"  External:       {summary['external']}")
    print(f"  Port cities:    {summary['port_cities']}")
    print(f"  FTZ targets:    {summary['ftz_targets']}")

    # ------------------------------------------------------------------
    # Step 2: Compute graph metrics
    # ------------------------------------------------------------------
    _header("STEP 2: GRAPH METRICS")
    metrics = GraphMetrics(graph)

    print(f"  Components:           {metrics.component_count()}")
    art_points = metrics.articulation_points()
    art_names = [graph.cities[n].name for n in art_points if n in graph.cities]
    print(f"  Articulation points:  {len(art_points)} ({', '.join(art_names[:5])}{'...' if len(art_names) > 5 else ''})")

    bridges = metrics.bridges()
    print(f"  Bridges:              {len(bridges)}")

    ecowas_art = metrics.ecowas_articulation_points()
    ecowas_art_names = [graph.cities[n].name for n in ecowas_art if n in graph.cities]
    print(f"  ECOWAS cut vertices:  {len(ecowas_art)} ({', '.join(ecowas_art_names[:5])}{'...' if len(ecowas_art_names) > 5 else ''})")

    _subheader(f"Top {top_n} Betweenness Centrality")
    top_bc = metrics.top_centrality("betweenness", top_n)
    rows_bc = []
    for rank, (cid, score) in enumerate(top_bc, 1):
        city = graph.cities.get(cid)
        name = city.name if city else cid
        country = city.country if city else "?"
        rows_bc.append([rank, name, country, f"{score:.4f}"])
    _table(["#", "City", "Country", "Betweenness"], rows_bc)

    # ------------------------------------------------------------------
    # Step 3: FTZ Trade Impact
    # ------------------------------------------------------------------
    _header("STEP 3: FTZ TRADE IMPACT SCORES")
    impact_analyzer = TradeImpactAnalyzer(graph, metrics)
    impact_scores = impact_analyzer.score_all()

    ranked_impact = sorted(impact_scores.items(), key=lambda x: x[1].composite, reverse=True)
    rows_impact = []
    for rank, (cid, ts) in enumerate(ranked_impact[:top_n], 1):
        city = graph.cities.get(cid)
        name = city.name if city else cid
        country = city.country if city else "?"
        rows_impact.append([
            rank,
            name,
            country,
            f"{ts.composite:.3f}",
            f"{ts.connectivity_score:.2f}",
            f"{ts.port_access_score:.2f}",
            f"{ts.tariff_exposure_score:.2f}",
            f"{ts.trade_volume_score:.2f}",
            f"{ts.stability_score:.2f}",
        ])
    _table(
        ["#", "City", "Country", "Score", "Conn", "Port", "Tariff", "Trade", "Stab"],
        rows_impact,
    )

    # ------------------------------------------------------------------
    # Step 4: Trade Route Analysis
    # ------------------------------------------------------------------
    _header("STEP 4: TRADE ROUTE VULNERABILITY")
    route_analyzer = TradeRouteAnalyzer(graph)
    route_results = route_analyzer.score_all_targets()

    rows_route = []
    for rank, r in enumerate(route_results[:top_n], 1):
        city = graph.cities.get(r["target"])
        name = city.name if city else r["target"]
        path_str = " -> ".join(
            graph.cities[n].name if n in graph.cities else n
            for n in r["shortest_path"][:4]
        )
        if len(r["shortest_path"]) > 4:
            path_str += " -> ..."
        rows_route.append([
            rank,
            name,
            f"{r['trade_route_risk']:.3f}",
            r["route_redundancy"],
            r["min_cut_size"],
            path_str,
        ])
    _table(
        ["#", "City", "Risk", "Redund.", "MinCut", "Shortest Route"],
        rows_route,
    )

    # ------------------------------------------------------------------
    # Step 5: Economic Cascade Simulations
    # ------------------------------------------------------------------
    _header("STEP 5: ECONOMIC CASCADE SIMULATIONS")
    cascade_sim = EconomicCascadeSimulator(graph)

    # 5a. Nigeria exit
    nigeria_cities = graph.get_by_country("NGA")
    if nigeria_cities:
        nigeria_result = cascade_sim.simulate_exit(nigeria_cities[0].id)
        _cascade_summary(nigeria_result, graph, "Scenario: Nigeria exits ECOWAS")

    # 5b. Mali exit
    mali_cities = graph.get_by_country("MLI")
    if mali_cities:
        mali_result = cascade_sim.simulate_exit(mali_cities[0].id)
        _cascade_summary(mali_result, graph, "Scenario: Mali exits ECOWAS")

    # 5c. Alliance of Sahel States (Mali + Burkina Faso + Niger) multi-exit
    sahel_triggers = []
    for iso3 in ("MLI", "BFA", "NER"):
        country_cities = graph.get_by_country(iso3)
        if country_cities:
            sahel_triggers.append(country_cities[0].id)

    if sahel_triggers:
        sahel_result = cascade_sim.simulate_multi_exit(sahel_triggers)
        _cascade_summary(
            sahel_result,
            graph,
            "Scenario: Alliance of Sahel States (Mali + Burkina + Niger) multi-exit",
        )

    # ------------------------------------------------------------------
    # Step 6: Opportunity Signals
    # ------------------------------------------------------------------
    _header("STEP 6: TRADE OPPORTUNITY SIGNALS")
    signal_gen = OpportunitySignalGenerator(graph, metrics)
    top_signals = signal_gen.top_opportunities(top_n)

    rows_signals = []
    for rank, sig in enumerate(top_signals, 1):
        city = graph.cities.get(sig.city_id)
        name = city.name if city else sig.city_id
        country = city.country if city else "?"
        rows_signals.append([
            rank,
            name,
            country,
            sig.direction,
            f"{sig.gap:+.4f}",
            f"{sig.model_trade_flow:.4f}",
            f"{sig.actual_trade_flow:.4f}",
            f"{sig.confidence:.2f}",
        ])
    _table(
        ["#", "City", "Country", "Signal", "Gap", "Model", "Actual", "Conf"],
        rows_signals,
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    _header("PIPELINE SUMMARY")

    # Count opportunity vs risk vs neutral signals
    all_signals = signal_gen.generate_all_signals()
    opp_count = sum(1 for s in all_signals.values() if s.direction == "OPPORTUNITY")
    risk_count = sum(1 for s in all_signals.values() if s.direction == "RISK")
    neutral_count = sum(1 for s in all_signals.values() if s.direction == "NEUTRAL")

    top_impact_city = ranked_impact[0] if ranked_impact else None
    top_risk_route = route_results[0] if route_results else None

    summary_rows = [
        ["Graph nodes", str(summary["nodes"])],
        ["Graph edges", str(summary["edges"])],
        ["FTZ target cities scored", str(len(impact_scores))],
        ["Highest FTZ impact", f"{top_impact_city[1].composite:.3f} ({graph.cities[top_impact_city[0]].name})" if top_impact_city else "N/A"],
        ["Highest route risk", f"{top_risk_route['trade_route_risk']:.3f} ({graph.cities[top_risk_route['target']].name})" if top_risk_route and top_risk_route["target"] in graph.cities else "N/A"],
        ["Opportunity signals", str(opp_count)],
        ["Risk signals", str(risk_count)],
        ["Neutral signals", str(neutral_count)],
        ["Cascade scenarios run", "3"],
    ]
    _table(["Metric", "Value"], summary_rows, col_widths=[30, 42])

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
