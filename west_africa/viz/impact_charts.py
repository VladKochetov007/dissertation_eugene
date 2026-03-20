"""FTZ impact score visualizations."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core.types import TradeImpactScore

if TYPE_CHECKING:
    from ..core.graph import WestAfricaGraph
    from ..core.metrics import GraphMetrics
    from ..signals.trade_impact import TradeImpactAnalyzer


COMPONENT_COLOURS = {
    "connectivity": "#2E75B6",
    "port_access": "#27AE60",
    "tariff_exposure": "#E67E22",
    "trade_volume": "#8E44AD",
    "diversification": "#16A085",
    "border_proximity": "#F1C40F",
    "stability": "#E74C3C",
}

COMPONENT_LABELS = {
    "connectivity": "Connectivity",
    "port_access": "Port Access",
    "tariff_exposure": "Tariff Exposure",
    "trade_volume": "Trade Volume",
    "diversification": "Diversification",
    "border_proximity": "Border Proximity",
    "stability": "Stability",
}


class ImpactChartsViz:
    """Visualize FTZ trade impact scores."""

    def __init__(self, analyzer: "TradeImpactAnalyzer", graph: "WestAfricaGraph") -> None:
        self.analyzer = analyzer
        self.wag = graph
        self._scores: Optional[dict[str, TradeImpactScore]] = None

    @property
    def scores(self) -> dict[str, TradeImpactScore]:
        if self._scores is None:
            self._scores = self.analyzer.score_all()
        return self._scores

    def _ranked(self, top_n: int = 29) -> list[tuple[str, TradeImpactScore]]:
        return sorted(self.scores.items(), key=lambda x: x[1].composite, reverse=True)[:top_n]

    def stacked_bar(
        self, top_n: int = 15, output_path: Optional[pathlib.Path] = None
    ) -> go.Figure:
        """Stacked horizontal bar chart showing score decomposition per city."""
        ranked = self._ranked(top_n)
        ranked.reverse()  # bottom-up for horizontal bars

        city_names = [self.wag.cities[cid].name for cid, _ in ranked]
        components = [
            ("connectivity", [ts.connectivity_score * ts.WEIGHTS["connectivity"] for _, ts in ranked]),
            ("port_access", [ts.port_access_score * ts.WEIGHTS["port_access"] for _, ts in ranked]),
            ("tariff_exposure", [ts.tariff_exposure_score * ts.WEIGHTS["tariff_exposure"] for _, ts in ranked]),
            ("trade_volume", [ts.trade_volume_score * ts.WEIGHTS["trade_volume"] for _, ts in ranked]),
            ("diversification", [ts.diversification_score * ts.WEIGHTS["diversification"] for _, ts in ranked]),
            ("border_proximity", [ts.border_proximity_score * ts.WEIGHTS["border_proximity"] for _, ts in ranked]),
            ("stability", [ts.stability_score * ts.WEIGHTS["stability"] for _, ts in ranked]),
        ]

        fig = go.Figure()
        for comp_name, values in components:
            fig.add_trace(go.Bar(
                y=city_names,
                x=values,
                name=COMPONENT_LABELS[comp_name],
                marker_color=COMPONENT_COLOURS[comp_name],
                orientation="h",
                hovertemplate="%{y}: %{x:.3f}<extra>" + COMPONENT_LABELS[comp_name] + "</extra>",
            ))

        fig.update_layout(
            barmode="stack",
            title=dict(text="FTZ Impact Score Decomposition", font=dict(size=18, color="#1B2A4A")),
            xaxis=dict(title="Composite Score", range=[0, 0.75]),
            yaxis=dict(title=""),
            legend=dict(x=1.02, y=1, font=dict(size=10)),
            margin=dict(l=120, r=200, t=60, b=40),
            height=max(400, top_n * 32),
            template="plotly_white",
        )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")
        return fig

    def radar_comparison(
        self, city_ids: Optional[list[str]] = None, top_n: int = 5,
        output_path: Optional[pathlib.Path] = None
    ) -> go.Figure:
        """Radar (spider) chart comparing top cities across all 7 dimensions."""
        if city_ids is None:
            city_ids = [cid for cid, _ in self._ranked(top_n)]

        categories = list(COMPONENT_LABELS.values())
        colours = ["#2E75B6", "#E74C3C", "#27AE60", "#E67E22", "#8E44AD",
                    "#16A085", "#F1C40F", "#3498DB"]

        fig = go.Figure()
        for i, cid in enumerate(city_ids):
            ts = self.scores.get(cid)
            if not ts:
                continue
            city = self.wag.cities.get(cid)
            name = city.name if city else cid

            values = [
                ts.connectivity_score,
                ts.port_access_score,
                ts.tariff_exposure_score,
                ts.trade_volume_score,
                ts.diversification_score,
                ts.border_proximity_score,
                ts.stability_score,
            ]
            values.append(values[0])  # close the polygon

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=name,
                line=dict(color=colours[i % len(colours)], width=2),
                fill="toself",
                opacity=0.3,
            ))

        fig.update_layout(
            title=dict(text="FTZ Impact Radar — Top Cities", font=dict(size=18, color="#1B2A4A")),
            polar=dict(
                radialaxis=dict(range=[0, 1], showticklabels=True, tickfont=dict(size=9)),
            ),
            legend=dict(x=1.1, y=1, font=dict(size=11)),
            height=550,
            template="plotly_white",
        )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")
        return fig

    def score_heatmap(
        self, output_path: Optional[pathlib.Path] = None
    ) -> go.Figure:
        """Heatmap of raw component scores for all FTZ cities."""
        ranked = self._ranked(29)
        city_names = [self.wag.cities[cid].name for cid, _ in ranked]
        comp_names = list(COMPONENT_LABELS.values())

        z = []
        for _, ts in ranked:
            z.append([
                ts.connectivity_score,
                ts.port_access_score,
                ts.tariff_exposure_score,
                ts.trade_volume_score,
                ts.diversification_score,
                ts.border_proximity_score,
                ts.stability_score,
            ])

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=comp_names,
            y=city_names,
            colorscale="Blues",
            zmin=0, zmax=1,
            hovertemplate="%{y} — %{x}: %{z:.2f}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text="FTZ Component Scores Heatmap", font=dict(size=18, color="#1B2A4A")),
            xaxis=dict(tickangle=-45),
            margin=dict(l=120, r=40, t=60, b=80),
            height=max(500, len(city_names) * 22),
            template="plotly_white",
        )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")
        return fig
