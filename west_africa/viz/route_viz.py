"""Trade route vulnerability visualizations."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Optional

import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..core.graph import WestAfricaGraph
    from ..signals.trade_route import TradeRouteAnalyzer


class RouteViz:
    """Visualize trade route vulnerability and corridor analysis."""

    def __init__(self, analyzer: "TradeRouteAnalyzer", graph: "WestAfricaGraph") -> None:
        self.analyzer = analyzer
        self.wag = graph

    def risk_ranking(
        self, top_n: int = 15, output_path: Optional[pathlib.Path] = None
    ) -> go.Figure:
        """Bar chart of trade route risk scores for inland cities."""
        results = self.analyzer.score_all_targets()[:top_n]

        cities = [self.wag.cities[r["target"]].name for r in results]
        risks = [r["trade_route_risk"] for r in results]
        redundancies = [r["route_redundancy"] for r in results]
        min_cuts = [r["min_cut_size"] for r in results]
        routes = [" → ".join(self.wag.cities[n].name for n in r["shortest_path"]) for r in results]

        # Colour by risk level
        colours = [
            "#E74C3C" if r >= 0.5 else "#E67E22" if r >= 0.3 else "#27AE60"
            for r in risks
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cities,
            y=risks,
            marker_color=colours,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Risk: %{y:.3f}<br>"
                "%{customdata}<extra></extra>"
            ),
            customdata=[
                f"Redundancy: {red} routes<br>Min-cut: {mc} nodes<br>Route: {rt}"
                for red, mc, rt in zip(redundancies, min_cuts, routes)
            ],
        ))

        fig.update_layout(
            title=dict(text="Trade Route Vulnerability Ranking", font=dict(size=18, color="#1B2A4A")),
            xaxis=dict(title="", tickangle=-45),
            yaxis=dict(title="Risk Score (0–1)", range=[0, 1]),
            height=450,
            margin=dict(l=60, r=40, t=60, b=100),
            template="plotly_white",
        )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")
        return fig

    def route_map(
        self, target: str, output_path: Optional[pathlib.Path] = None
    ) -> go.Figure:
        """Map showing the shortest trade route to a target city with highlights."""
        path, cost = self.analyzer.shortest_trade_route(target)
        if not path:
            fig = go.Figure()
            fig.add_annotation(text="No route found", x=0.5, y=0.5, showarrow=False)
            return fig

        target_city = self.wag.cities.get(target)
        fig = go.Figure()

        # Background edges (light)
        seen: set[tuple[str, str]] = set()
        for u, v, data in self.wag.G.edges(data=True):
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            c1, c2 = self.wag.cities.get(u), self.wag.cities.get(v)
            if c1 and c2:
                fig.add_trace(go.Scattergeo(
                    lon=[c1.lng, c2.lng], lat=[c1.lat, c2.lat],
                    mode="lines", line=dict(width=0.5, color="#DDDDDD"),
                    hoverinfo="skip", showlegend=False,
                ))

        # Route path (bold)
        for i in range(len(path) - 1):
            c1 = self.wag.cities.get(path[i])
            c2 = self.wag.cities.get(path[i + 1])
            if c1 and c2:
                fig.add_trace(go.Scattergeo(
                    lon=[c1.lng, c2.lng], lat=[c1.lat, c2.lat],
                    mode="lines",
                    line=dict(width=4, color="#E74C3C"),
                    hoverinfo="text",
                    text=f"{c1.name} → {c2.name}",
                    showlegend=False,
                ))

        # All cities (small)
        for c in self.wag.cities.values():
            colour = "#2E75B6" if c.id in path else "#CCCCCC"
            size = 10 if c.id in path else 5
            fig.add_trace(go.Scattergeo(
                lon=[c.lng], lat=[c.lat],
                mode="markers+text",
                marker=dict(size=size, color=colour, line=dict(width=1, color="white")),
                text=c.name if c.id in path else "",
                textposition="top center",
                textfont=dict(size=9),
                hoverinfo="text",
                hovertext=f"<b>{c.name}</b> ({c.country})",
                showlegend=False,
            ))

        route_names = " → ".join(self.wag.cities[n].name for n in path)
        fig.update_layout(
            title=dict(
                text=f"Trade Route to {target_city.name if target_city else target}<br>"
                     f"<sub>{route_names} (cost: {cost:.1f})</sub>",
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
            margin=dict(l=0, r=0, t=80, b=0),
            height=600,
            template="plotly_white",
        )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")
        return fig

    def redundancy_bubble(
        self, output_path: Optional[pathlib.Path] = None
    ) -> go.Figure:
        """Bubble chart: x=min-cut, y=risk, size=redundancy."""
        results = self.analyzer.score_all_targets()

        cities = [self.wag.cities[r["target"]].name for r in results]
        risks = [r["trade_route_risk"] for r in results]
        min_cuts = [r["min_cut_size"] for r in results]
        redundancies = [r["route_redundancy"] for r in results]

        # Normalise bubble sizes
        max_red = max(redundancies) if redundancies else 1
        sizes = [max(8, (r / max_red) * 50) for r in redundancies]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=min_cuts,
            y=risks,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=risks,
                colorscale="RdYlGn_r",
                cmin=0, cmax=1,
                showscale=True,
                colorbar=dict(title="Risk"),
                line=dict(width=1, color="white"),
            ),
            text=cities,
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Min-cut: %{x} nodes<br>"
                "Risk: %{y:.3f}<br>"
                "Redundancy: %{customdata} routes<extra></extra>"
            ),
            customdata=redundancies,
        ))

        fig.update_layout(
            title=dict(text="Route Resilience: Min-Cut vs Risk", font=dict(size=18, color="#1B2A4A")),
            xaxis=dict(title="Minimum Node Cut", dtick=1),
            yaxis=dict(title="Route Risk (0–1)", range=[-0.05, 1.05]),
            height=500,
            template="plotly_white",
        )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")
        return fig
