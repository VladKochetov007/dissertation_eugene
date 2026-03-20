"""Geographic network map visualization using Plotly."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Optional

import plotly.graph_objects as go

from ..core.types import BlocMembership, ConnectionType

if TYPE_CHECKING:
    from ..core.graph import WestAfricaGraph
    from ..core.metrics import GraphMetrics


# ── Colour palette ──────────────────────────────────────────────────
BLOC_COLOURS = {
    BlocMembership.ECOWAS: "#2E75B6",
    BlocMembership.UEMOA: "#27AE60",
    BlocMembership.SUSPENDED: "#E74C3C",
    BlocMembership.EXTERNAL: "#95A5A6",
}

EDGE_TYPE_COLOURS = {
    "TRADE": "#2E75B6",
    "INFRASTRUCTURE": "#E67E22",
    "POLITICAL": "#8E44AD",
    "CULTURAL": "#16A085",
    "FINANCIAL": "#F1C40F",
    "MIGRATORY": "#E74C3C",
    "LABOUR": "#3498DB",
}

EDGE_TYPE_WIDTHS = {
    "TRADE": 2.0,
    "INFRASTRUCTURE": 1.8,
    "POLITICAL": 1.0,
    "CULTURAL": 0.8,
    "FINANCIAL": 1.5,
    "MIGRATORY": 1.0,
    "LABOUR": 0.8,
}


class NetworkMapViz:
    """Generate interactive geographic network maps."""

    def __init__(self, graph: "WestAfricaGraph", metrics: Optional["GraphMetrics"] = None) -> None:
        self.wag = graph
        self.metrics = metrics

    def full_network(
        self,
        edge_types: Optional[list[str]] = None,
        highlight_ftz: bool = True,
        size_by: str = "population",
        output_path: Optional[pathlib.Path] = None,
    ) -> go.Figure:
        """Render full geographic network with all cities and edges.

        Args:
            edge_types: Filter to these edge types (None = all).
            highlight_ftz: Ring highlight on FTZ target cities.
            size_by: 'population', 'gdp', or 'centrality'.
            output_path: If provided, save as standalone HTML.
        """
        fig = go.Figure()

        # ── Edges ────────────────────────────────────────────────
        seen_edges: set[tuple[str, str, str]] = set()
        for u, v, data in self.wag.G.edges(data=True):
            etype = data.get("edge_type", "TRADE")
            if edge_types and etype not in edge_types:
                continue

            key = (min(u, v), max(u, v), etype)
            if key in seen_edges:
                continue
            seen_edges.add(key)

            c1 = self.wag.cities.get(u)
            c2 = self.wag.cities.get(v)
            if not c1 or not c2:
                continue

            colour = EDGE_TYPE_COLOURS.get(etype, "#CCCCCC")
            width = EDGE_TYPE_WIDTHS.get(etype, 1.0)
            vol = data.get("volume", 0)
            desc = data.get("description", "") or f"{etype} link"

            fig.add_trace(go.Scattergeo(
                lon=[c1.lng, c2.lng],
                lat=[c1.lat, c2.lat],
                mode="lines",
                line=dict(width=width, color=colour),
                opacity=0.45,
                hoverinfo="text",
                text=f"{c1.name} ↔ {c2.name}<br>{etype}<br>Vol: ${vol}M<br>{desc}",
                showlegend=False,
            ))

        # ── Edge type legend (invisible markers) ─────────────────
        for etype, colour in EDGE_TYPE_COLOURS.items():
            if edge_types and etype not in edge_types:
                continue
            fig.add_trace(go.Scattergeo(
                lon=[None], lat=[None],
                mode="markers",
                marker=dict(size=8, color=colour),
                name=etype.title(),
                showlegend=True,
            ))

        # ── Nodes ────────────────────────────────────────────────
        betweenness = {}
        if self.metrics and size_by == "centrality":
            betweenness = self.metrics.betweenness_centrality()

        for bloc, colour in BLOC_COLOURS.items():
            cities_in_bloc = [
                c for c in self.wag.cities.values()
                if self.wag.get_effective_bloc(c.id) == bloc
            ]
            if not cities_in_bloc:
                continue

            lats = [c.lat for c in cities_in_bloc]
            lons = [c.lng for c in cities_in_bloc]
            names = [c.name for c in cities_in_bloc]
            sizes = []
            hover_texts = []

            for c in cities_in_bloc:
                if size_by == "centrality" and betweenness:
                    s = 8 + betweenness.get(c.id, 0) * 80
                elif size_by == "gdp":
                    s = max(6, min(c.gdp_per_capita / 200, 30))
                else:
                    s = max(6, min(c.population / 500_000, 30))
                sizes.append(s)

                es = self.wag.economic_states.get(c.id)
                stability = f"{es.political_stability:.2f}" if es else "N/A"
                tags = ", ".join(c.tags) if c.tags else "none"
                hover_texts.append(
                    f"<b>{c.name}</b> ({c.country})<br>"
                    f"Bloc: {bloc.value}<br>"
                    f"Pop: {c.population:,}<br>"
                    f"GDP/cap: ${c.gdp_per_capita:,.0f}<br>"
                    f"Port: {'Yes' if c.is_port else 'No'} | "
                    f"Capital: {'Yes' if c.is_capital else 'No'}<br>"
                    f"FTZ target: {'Yes' if c.is_ftz_target else 'No'}<br>"
                    f"Stability: {stability}<br>"
                    f"Tags: {tags}"
                )

            # FTZ target ring
            symbols = [
                "circle-open" if (c.is_ftz_target and highlight_ftz) else "circle"
                for c in cities_in_bloc
            ]

            fig.add_trace(go.Scattergeo(
                lon=lons, lat=lats,
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color=colour,
                    symbol=symbols,
                    line=dict(width=1, color="white"),
                ),
                text=names,
                textposition="top center",
                textfont=dict(size=9, color="#333333"),
                hoverinfo="text",
                hovertext=hover_texts,
                name=bloc.value,
            ))

        # ── Layout ───────────────────────────────────────────────
        fig.update_layout(
            title=dict(
                text="West Africa FTZ Network",
                font=dict(size=20, color="#1B2A4A"),
            ),
            geo=dict(
                scope="africa",
                showland=True,
                landcolor="#F7F7F7",
                showocean=True,
                oceancolor="#E8F0FE",
                showcountries=True,
                countrycolor="#CCCCCC",
                showcoastlines=True,
                coastlinecolor="#AAAAAA",
                showlakes=True,
                lakecolor="#E8F0FE",
                lonaxis=dict(range=[-20, 16]),
                lataxis=dict(range=[3, 38]),
                projection_type="mercator",
            ),
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#CCCCCC",
                borderwidth=1,
                font=dict(size=11),
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=700,
            template="plotly_white",
        )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")

        return fig

    def bloc_comparison(self, output_path: Optional[pathlib.Path] = None) -> go.Figure:
        """Side-by-side map showing ECOWAS vs suspended vs external."""
        from plotly.subplots import make_subplots

        fig = self.full_network(edge_types=["TRADE", "INFRASTRUCTURE"])

        # Add annotations for suspended countries
        for c in self.wag.cities.values():
            if self.wag.get_effective_bloc(c.id) == BlocMembership.SUSPENDED:
                fig.add_annotation(
                    x=c.lng, y=c.lat,
                    text="⚠",
                    showarrow=False,
                    font=dict(size=14),
                    xref="x", yref="y",
                )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")

        return fig

    def edge_type_filter(
        self, edge_type: str, output_path: Optional[pathlib.Path] = None
    ) -> go.Figure:
        """Render network filtered to a single edge type."""
        return self.full_network(
            edge_types=[edge_type],
            output_path=output_path,
        )
