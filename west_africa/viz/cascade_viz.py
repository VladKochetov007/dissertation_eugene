"""Economic cascade simulation visualizations."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core.types import BlocMembership

if TYPE_CHECKING:
    from ..core.graph import WestAfricaGraph
    from ..signals.cascade import CascadeResult, EconomicCascadeSimulator


class CascadeViz:
    """Visualize economic cascade simulation results."""

    def __init__(self, simulator: "EconomicCascadeSimulator", graph: "WestAfricaGraph") -> None:
        self.simulator = simulator
        self.wag = graph

    def scenario_comparison(
        self,
        scenarios: dict[str, "CascadeResult"],
        output_path: Optional[pathlib.Path] = None,
    ) -> go.Figure:
        """Multi-metric bar comparison of cascade scenarios.

        Args:
            scenarios: Dict of {label: CascadeResult}.
        """
        labels = list(scenarios.keys())
        severities = [r.severity for r in scenarios.values()]
        trade_vols = [r.trade_volume_affected for r in scenarios.values()]
        affected = [len(r.affected_nodes) for r in scenarios.values()]
        isolated = [len(r.isolated_nodes) for r in scenarios.values()]
        components = [r.new_component_count for r in scenarios.values()]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Severity (0–1)", "Trade Volume Affected ($M)",
                            "Cities Affected / Isolated", "Network Components"],
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        colours = ["#2E75B6", "#E74C3C", "#E67E22", "#27AE60", "#8E44AD"]

        # Severity
        fig.add_trace(go.Bar(
            x=labels, y=severities,
            marker_color=[colours[i % len(colours)] for i in range(len(labels))],
            showlegend=False,
            hovertemplate="%{x}: %{y:.3f}<extra>Severity</extra>",
        ), row=1, col=1)

        # Trade volume
        fig.add_trace(go.Bar(
            x=labels, y=trade_vols,
            marker_color=[colours[i % len(colours)] for i in range(len(labels))],
            showlegend=False,
            hovertemplate="%{x}: $%{y:,.0f}M<extra>Trade Vol</extra>",
        ), row=1, col=2)

        # Affected + isolated (grouped)
        fig.add_trace(go.Bar(
            x=labels, y=affected, name="Affected",
            marker_color="#2E75B6",
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=labels, y=isolated, name="Isolated",
            marker_color="#E74C3C",
        ), row=2, col=1)

        # Components
        fig.add_trace(go.Bar(
            x=labels, y=components,
            marker_color=[colours[i % len(colours)] for i in range(len(labels))],
            showlegend=False,
            hovertemplate="%{x}: %{y} components<extra></extra>",
        ), row=2, col=2)

        fig.update_layout(
            title=dict(text="Economic Cascade Scenario Comparison", font=dict(size=18, color="#1B2A4A")),
            barmode="group",
            height=600,
            template="plotly_white",
            legend=dict(x=0.35, y=0.42),
        )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")
        return fig

    def impact_map(
        self,
        result: "CascadeResult",
        scenario_label: str = "",
        output_path: Optional[pathlib.Path] = None,
    ) -> go.Figure:
        """Geographic map showing cascade impact: affected, isolated, disrupted cities."""
        fig = go.Figure()

        # Background edges
        seen: set[tuple[str, str]] = set()
        for u, v, data in self.wag.G.edges(data=True):
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            c1, c2 = self.wag.cities.get(u), self.wag.cities.get(v)
            if c1 and c2:
                # Dim edges connected to affected nodes
                is_disrupted = u in result.affected_nodes or v in result.affected_nodes
                fig.add_trace(go.Scattergeo(
                    lon=[c1.lng, c2.lng], lat=[c1.lat, c2.lat],
                    mode="lines",
                    line=dict(
                        width=1.5 if is_disrupted else 0.5,
                        color="#E74C3C" if is_disrupted else "#EEEEEE",
                        dash="dash" if is_disrupted else "solid",
                    ),
                    opacity=0.5 if is_disrupted else 0.3,
                    hoverinfo="skip", showlegend=False,
                ))

        # Classification groups
        groups = [
            ("Affected (exiting)", result.affected_nodes, "#E74C3C", 14, "x"),
            ("Isolated", result.isolated_nodes, "#F39C12", 12, "diamond"),
            ("Trade disrupted", result.trade_disrupted_nodes, "#E67E22", 10, "triangle-up"),
        ]

        classified = set(result.affected_nodes + result.isolated_nodes + result.trade_disrupted_nodes)

        # Unaffected cities first
        unaffected = [c for c in self.wag.cities.values() if c.id not in classified]
        if unaffected:
            fig.add_trace(go.Scattergeo(
                lon=[c.lng for c in unaffected],
                lat=[c.lat for c in unaffected],
                mode="markers",
                marker=dict(size=6, color="#CCCCCC", line=dict(width=0.5, color="white")),
                hoverinfo="text",
                hovertext=[f"{c.name} ({c.country})" for c in unaffected],
                name="Unaffected",
            ))

        for group_name, node_ids, colour, size, symbol in groups:
            cities_in_group = [self.wag.cities[n] for n in node_ids if n in self.wag.cities]
            if not cities_in_group:
                continue
            fig.add_trace(go.Scattergeo(
                lon=[c.lng for c in cities_in_group],
                lat=[c.lat for c in cities_in_group],
                mode="markers+text",
                marker=dict(size=size, color=colour, symbol=symbol,
                            line=dict(width=1, color="white")),
                text=[c.name for c in cities_in_group],
                textposition="top center",
                textfont=dict(size=9, color=colour),
                hoverinfo="text",
                hovertext=[
                    f"<b>{c.name}</b> ({c.country})<br>Status: {group_name}"
                    for c in cities_in_group
                ],
                name=group_name,
            ))

        sev_pct = f"{result.severity * 100:.1f}%"
        fig.update_layout(
            title=dict(
                text=f"Cascade Impact: {scenario_label}<br>"
                     f"<sub>Severity: {sev_pct} | Trade affected: ${result.trade_volume_affected:,.0f}M | "
                     f"Components: {result.new_component_count}</sub>",
                font=dict(size=16, color="#1B2A4A"),
            ),
            geo=dict(
                scope="africa",
                showland=True, landcolor="#F7F7F7",
                showocean=True, oceancolor="#E8F0FE",
                showcountries=True, countrycolor="#CCCCCC",
                lonaxis=dict(range=[-20, 16]),
                lataxis=dict(range=[3, 38]),
                projection_type="mercator",
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)", font=dict(size=11)),
            margin=dict(l=0, r=0, t=80, b=0),
            height=650,
            template="plotly_white",
        )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")
        return fig

    def severity_waterfall(
        self,
        scenarios: dict[str, "CascadeResult"],
        output_path: Optional[pathlib.Path] = None,
    ) -> go.Figure:
        """Waterfall chart of severity contributions."""
        labels = list(scenarios.keys())
        severities = [r.severity for r in scenarios.values()]

        fig = go.Figure()
        fig.add_trace(go.Waterfall(
            x=labels,
            y=severities,
            measure=["absolute"] * len(labels),
            connector_line_color="#CCCCCC",
            increasing_marker_color="#E74C3C",
            decreasing_marker_color="#27AE60",
            totals_marker_color="#2E75B6",
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text="Cascade Severity by Scenario", font=dict(size=18, color="#1B2A4A")),
            yaxis=dict(title="Severity (0–1)", range=[0, 1]),
            height=400,
            template="plotly_white",
        )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")
        return fig
