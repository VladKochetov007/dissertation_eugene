"""Trade opportunity signal visualizations."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Optional

import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..core.graph import WestAfricaGraph
    from ..signals.opportunity_signal import OpportunitySignalGenerator


SIGNAL_COLOURS = {
    "OPPORTUNITY": "#27AE60",
    "RISK": "#E74C3C",
    "NEUTRAL": "#95A5A6",
}


class OpportunityViz:
    """Visualize trade opportunity signals."""

    def __init__(self, generator: "OpportunitySignalGenerator", graph: "WestAfricaGraph") -> None:
        self.generator = generator
        self.wag = graph

    def gap_scatter(
        self, output_path: Optional[pathlib.Path] = None
    ) -> go.Figure:
        """Scatter: model vs actual trade flow, coloured by signal direction."""
        signals = self.generator.generate_all_signals()

        fig = go.Figure()

        # 45-degree reference line
        fig.add_trace(go.Scatter(
            x=[0, 0.7], y=[0, 0.7],
            mode="lines",
            line=dict(dash="dash", color="#CCCCCC", width=1),
            hoverinfo="skip",
            showlegend=False,
        ))

        for direction, colour in SIGNAL_COLOURS.items():
            subset = [(cid, s) for cid, s in signals.items() if s.direction == direction]
            if not subset:
                continue

            names = [self.wag.cities[cid].name for cid, _ in subset]
            actuals = [s.actual_trade_flow for _, s in subset]
            models = [s.model_trade_flow for _, s in subset]
            gaps = [s.gap for _, s in subset]
            confs = [s.confidence for _, s in subset]

            fig.add_trace(go.Scatter(
                x=actuals,
                y=models,
                mode="markers+text",
                marker=dict(
                    size=[max(8, c * 25) for c in confs],
                    color=colour,
                    line=dict(width=1, color="white"),
                    opacity=0.8,
                ),
                text=names,
                textposition="top center",
                textfont=dict(size=8, color="#555555"),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Actual: %{x:.3f}<br>"
                    "Model: %{y:.3f}<br>"
                    "Gap: %{customdata[0]:+.3f}<br>"
                    "Confidence: %{customdata[1]:.2f}<extra>" + direction + "</extra>"
                ),
                customdata=list(zip(gaps, confs)),
                name=direction.title(),
            ))

        fig.add_annotation(
            x=0.05, y=0.65,
            text="← Untapped potential<br>(model > actual)",
            showarrow=False, font=dict(size=10, color="#27AE60"),
            xref="x", yref="y",
        )

        fig.update_layout(
            title=dict(text="Trade Opportunity Signals: Model vs Actual Flow",
                       font=dict(size=18, color="#1B2A4A")),
            xaxis=dict(title="Actual Trade Flow (normalized)", range=[-0.02, 0.55]),
            yaxis=dict(title="Model-Predicted Flow (normalized)", range=[-0.02, 0.7]),
            legend=dict(x=0.75, y=0.15, font=dict(size=11)),
            height=550,
            template="plotly_white",
        )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")
        return fig

    def gap_bar(
        self, top_n: int = 15, output_path: Optional[pathlib.Path] = None
    ) -> go.Figure:
        """Horizontal bar chart of trade gaps, sorted by gap size."""
        signals = self.generator.generate_all_signals()
        ranked = sorted(signals.values(), key=lambda s: s.gap, reverse=True)[:top_n]
        ranked.reverse()

        city_names = [self.wag.cities[s.city_id].name for s in ranked]
        gaps = [s.gap for s in ranked]
        directions = [s.direction for s in ranked]
        colours = [SIGNAL_COLOURS.get(d, "#95A5A6") for d in directions]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=city_names,
            x=gaps,
            orientation="h",
            marker_color=colours,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Gap: %{x:+.3f}<br>"
                "%{customdata}<extra></extra>"
            ),
            customdata=[
                f"Model: {s.model_trade_flow:.3f} | Actual: {s.actual_trade_flow:.3f} | Conf: {s.confidence:.2f}"
                for s in ranked
            ],
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="#999999")
        fig.add_vline(x=0.10, line_dash="dot", line_color="#27AE60", opacity=0.5,
                      annotation_text="Opportunity threshold", annotation_position="top right")
        fig.add_vline(x=-0.10, line_dash="dot", line_color="#E74C3C", opacity=0.5,
                      annotation_text="Risk threshold", annotation_position="top left")

        fig.update_layout(
            title=dict(text="Trade Gap Analysis (Model − Actual)", font=dict(size=18, color="#1B2A4A")),
            xaxis=dict(title="Gap (positive = untapped potential)"),
            yaxis=dict(title=""),
            margin=dict(l=120, r=40, t=60, b=40),
            height=max(400, top_n * 30),
            template="plotly_white",
        )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")
        return fig

    def confidence_heatmap(
        self, output_path: Optional[pathlib.Path] = None
    ) -> go.Figure:
        """Heatmap showing gap size and confidence for all FTZ cities."""
        signals = self.generator.generate_all_signals()
        ranked = sorted(signals.values(), key=lambda s: s.gap, reverse=True)

        city_names = [self.wag.cities[s.city_id].name for s in ranked]
        gaps = [s.gap for s in ranked]
        confs = [s.confidence for s in ranked]
        actuals = [s.actual_trade_flow for s in ranked]
        models = [s.model_trade_flow for s in ranked]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gaps,
            y=city_names,
            mode="markers",
            marker=dict(
                size=14,
                color=confs,
                colorscale="Viridis",
                cmin=0, cmax=1,
                showscale=True,
                colorbar=dict(title="Confidence"),
                line=dict(width=1, color="white"),
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Gap: %{x:+.3f}<br>"
                "Confidence: %{marker.color:.2f}<br>"
                "%{customdata}<extra></extra>"
            ),
            customdata=[
                f"Model: {m:.3f} | Actual: {a:.3f}"
                for m, a in zip(models, actuals)
            ],
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="#999999")

        fig.update_layout(
            title=dict(text="Opportunity Signals: Gap vs Confidence", font=dict(size=18, color="#1B2A4A")),
            xaxis=dict(title="Trade Gap (model − actual)"),
            yaxis=dict(title=""),
            margin=dict(l=120, r=40, t=60, b=40),
            height=max(500, len(city_names) * 22),
            template="plotly_white",
        )

        if output_path:
            fig.write_html(str(output_path), include_plotlyjs="cdn")
        return fig
