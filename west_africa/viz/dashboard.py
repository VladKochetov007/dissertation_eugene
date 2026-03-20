"""Combined dashboard builder — generates a single-page HTML with all visualizations."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Optional

import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..core.graph import WestAfricaGraph
    from ..core.metrics import GraphMetrics


class DashboardBuilder:
    """Generate a combined interactive HTML dashboard from all analysis outputs."""

    def __init__(self, graph: "WestAfricaGraph", metrics: "GraphMetrics") -> None:
        self.wag = graph
        self.metrics = metrics

    def build(self, output_dir: pathlib.Path) -> pathlib.Path:
        """Run all visualizations and generate a combined dashboard page.

        Args:
            output_dir: Directory to write HTML files into.

        Returns:
            Path to the main dashboard HTML file.
        """
        from ..signals.trade_impact import TradeImpactAnalyzer
        from ..signals.trade_route import TradeRouteAnalyzer
        from ..signals.cascade import EconomicCascadeSimulator
        from ..signals.opportunity_signal import OpportunitySignalGenerator
        from .network_map import NetworkMapViz
        from .impact_charts import ImpactChartsViz
        from .route_viz import RouteViz
        from .cascade_viz import CascadeViz
        from .opportunity_viz import OpportunityViz

        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Instantiate analyzers ───────────────────────────────
        impact_analyzer = TradeImpactAnalyzer(self.wag, self.metrics)
        route_analyzer = TradeRouteAnalyzer(self.wag)
        cascade_sim = EconomicCascadeSimulator(self.wag)
        opp_gen = OpportunitySignalGenerator(self.wag, self.metrics)

        # ── Generate individual charts ──────────────────────────
        files: dict[str, pathlib.Path] = {}

        # 1. Network map
        net_viz = NetworkMapViz(self.wag, self.metrics)
        f = output_dir / "network_map.html"
        net_viz.full_network(size_by="centrality", output_path=f)
        files["network_map"] = f

        # 2. Impact charts
        imp_viz = ImpactChartsViz(impact_analyzer, self.wag)
        f = output_dir / "impact_stacked.html"
        imp_viz.stacked_bar(top_n=15, output_path=f)
        files["impact_stacked"] = f

        f = output_dir / "impact_radar.html"
        imp_viz.radar_comparison(top_n=5, output_path=f)
        files["impact_radar"] = f

        f = output_dir / "impact_heatmap.html"
        imp_viz.score_heatmap(output_path=f)
        files["impact_heatmap"] = f

        # 3. Route vulnerability
        rt_viz = RouteViz(route_analyzer, self.wag)
        f = output_dir / "route_risk.html"
        rt_viz.risk_ranking(top_n=15, output_path=f)
        files["route_risk"] = f

        f = output_dir / "route_bubble.html"
        rt_viz.redundancy_bubble(output_path=f)
        files["route_bubble"] = f

        # Route maps for top 3 vulnerable cities
        top_routes = route_analyzer.score_all_targets()[:3]
        for i, r in enumerate(top_routes):
            target = r["target"]
            city_name = self.wag.cities[target].name
            f = output_dir / f"route_map_{target}.html"
            rt_viz.route_map(target, output_path=f)
            files[f"route_map_{city_name}"] = f

        # 4. Cascade simulations
        cas_viz = CascadeViz(cascade_sim, self.wag)
        scenarios = {
            "Nigeria exit": cascade_sim.simulate_exit("lagos"),
            "Mali exit": cascade_sim.simulate_exit("bamako"),
            "Sahel multi-exit": cascade_sim.simulate_multi_exit(["bamako", "ouagadougou", "niamey"]),
        }

        f = output_dir / "cascade_comparison.html"
        cas_viz.scenario_comparison(scenarios, output_path=f)
        files["cascade_comparison"] = f

        for label, result in scenarios.items():
            slug = label.lower().replace(" ", "_")
            f = output_dir / f"cascade_map_{slug}.html"
            cas_viz.impact_map(result, scenario_label=label, output_path=f)
            files[f"cascade_{slug}"] = f

        f = output_dir / "cascade_severity.html"
        cas_viz.severity_waterfall(scenarios, output_path=f)
        files["cascade_severity"] = f

        # 5. Opportunity signals
        opp_viz = OpportunityViz(opp_gen, self.wag)
        f = output_dir / "opportunity_scatter.html"
        opp_viz.gap_scatter(output_path=f)
        files["opportunity_scatter"] = f

        f = output_dir / "opportunity_bar.html"
        opp_viz.gap_bar(top_n=15, output_path=f)
        files["opportunity_bar"] = f

        f = output_dir / "opportunity_confidence.html"
        opp_viz.confidence_heatmap(output_path=f)
        files["opportunity_confidence"] = f

        # ── Generate index page ─────────────────────────────────
        index_path = output_dir / "dashboard.html"
        self._write_index(index_path, files)

        return index_path

    def _write_index(self, path: pathlib.Path, files: dict[str, pathlib.Path]) -> None:
        """Generate an HTML index page with iframe-embedded charts."""
        summary = self.wag.summary()

        sections = {
            "Network Overview": ["network_map"],
            "FTZ Impact Scores": ["impact_stacked", "impact_radar", "impact_heatmap"],
            "Route Vulnerability": ["route_risk", "route_bubble"] + [
                k for k in files if k.startswith("route_map_")
            ],
            "Cascade Simulations": ["cascade_comparison", "cascade_severity"] + [
                k for k in files if k.startswith("cascade_") and k not in ("cascade_comparison", "cascade_severity")
            ],
            "Opportunity Signals": ["opportunity_scatter", "opportunity_bar", "opportunity_confidence"],
        }

        nav_html = ""
        content_html = ""

        for section_title, chart_keys in sections.items():
            section_id = section_title.lower().replace(" ", "-")
            nav_html += f'<a href="#{section_id}" class="nav-link">{section_title}</a>\n'

            content_html += f'<h2 id="{section_id}">{section_title}</h2>\n'
            for key in chart_keys:
                if key not in files:
                    continue
                filename = files[key].name
                label = key.replace("_", " ").title()
                content_html += f"""
                <div class="chart-card">
                    <h3>{label}</h3>
                    <iframe src="{filename}" class="chart-frame"></iframe>
                </div>
                """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>West Africa FTZ Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #f5f6fa; color: #333; }}
        .header {{
            background: linear-gradient(135deg, #1B2A4A 0%, #2E75B6 100%);
            color: white; padding: 24px 32px;
        }}
        .header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
        .header .subtitle {{ font-size: 14px; opacity: 0.85; }}
        .kpi-row {{
            display: flex; gap: 16px; padding: 20px 32px;
            flex-wrap: wrap;
        }}
        .kpi {{
            background: white; border-radius: 8px; padding: 16px 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            flex: 1; min-width: 140px;
        }}
        .kpi .value {{ font-size: 28px; font-weight: 700; color: #1B2A4A; }}
        .kpi .label {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
        nav {{
            background: white; padding: 12px 32px;
            border-bottom: 1px solid #e0e0e0;
            position: sticky; top: 0; z-index: 100;
            display: flex; gap: 8px; flex-wrap: wrap;
        }}
        .nav-link {{
            text-decoration: none; color: #2E75B6; font-size: 14px; font-weight: 500;
            padding: 6px 14px; border-radius: 20px; background: #E8F0FE;
            transition: all 0.2s;
        }}
        .nav-link:hover {{ background: #2E75B6; color: white; }}
        .content {{ padding: 24px 32px; max-width: 1400px; margin: 0 auto; }}
        h2 {{
            font-size: 22px; color: #1B2A4A; margin: 32px 0 16px;
            padding-bottom: 8px; border-bottom: 2px solid #2E75B6;
        }}
        .chart-card {{
            background: white; border-radius: 8px; margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden;
        }}
        .chart-card h3 {{
            font-size: 14px; color: #555; padding: 12px 16px;
            border-bottom: 1px solid #f0f0f0; background: #fafafa;
        }}
        .chart-frame {{
            width: 100%; height: 650px; border: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>West Africa Free Trade Zone Dashboard</h1>
        <div class="subtitle">Interactive network analysis and model findings</div>
    </div>

    <div class="kpi-row">
        <div class="kpi"><div class="value">{summary['nodes']}</div><div class="label">Cities</div></div>
        <div class="kpi"><div class="value">{summary['edges']}</div><div class="label">Connections</div></div>
        <div class="kpi"><div class="value">{summary['ftz_targets']}</div><div class="label">FTZ Targets</div></div>
        <div class="kpi"><div class="value">{summary['ecowas_active']}</div><div class="label">ECOWAS Active</div></div>
        <div class="kpi"><div class="value">{summary['suspended']}</div><div class="label">Suspended</div></div>
        <div class="kpi"><div class="value">{summary['port_cities']}</div><div class="label">Port Cities</div></div>
    </div>

    <nav>{nav_html}</nav>

    <div class="content">
        {content_html}
    </div>
</body>
</html>"""

        path.write_text(html)
