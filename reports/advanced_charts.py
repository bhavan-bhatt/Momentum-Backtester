# reports/advanced_charts.py
# ============================================================
# PHASE 2 CHART FUNCTIONS
# ============================================================

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import BacktestConfig


def plot_sharpe_confidence_interval(
    point_estimate: float,
    lower: float,
    upper: float,
    config: BacktestConfig,
) -> plt.Figure:
    """Bootstrap Sharpe CI as a horizontal error bar."""
    fig, ax = plt.subplots(figsize=(8, 3))
    color = "green" if lower > 0 else ("orange" if upper > 0 else "red")
    ax.errorbar(
        point_estimate,
        0,
        xerr=[[point_estimate - lower], [upper - point_estimate]],
        fmt="o",
        color=color,
        capsize=8,
        markersize=10,
    )
    ax.axvline(0, color="gray", linestyle="--", label="No edge")
    ax.set_yticks([])
    ax.set_xlabel("Annualised Sharpe Ratio")
    ax.set_title("Sharpe Ratio — Bootstrap Confidence Interval")
    ax.annotate(
        f"{point_estimate:.2f} [{lower:.2f}, {upper:.2f}] at "
        f"{config.advanced.statistics.confidence_level:.0%} CI",
        xy=(0.02, 0.85),
        xycoords="axes fraction",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_parameter_stability_heatmap(
    stability_summary: dict, config: BacktestConfig
) -> plt.Figure:
    """Bar chart of coefficient of variation per parameter."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if not stability_summary:
        ax.text(0.5, 0.5, "No stability data", ha="center", va="center")
        return fig

    names = list(stability_summary.keys())
    cvs = [stability_summary[n].get("cv") or 0 for n in names]
    colors = ["green" if cv < 0.25 else ("gold" if cv < 0.5 else "red") for cv in cvs]

    ax.barh(names, cvs, color=colors)
    ax.axvline(0.25, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Coefficient of Variation (lower = more stable)")
    ax.set_title("Parameter Stability Across Walk-Forward Splits (lower = better)")
    fig.tight_layout()
    return fig


def plot_regime_shaded_equity(
    equity_curve: pd.Series,
    regime_series: pd.Series,
    config: BacktestConfig,
) -> plt.Figure:
    """Equity curve with regime background shading."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity_curve.index, equity_curve.values, color="steelblue", linewidth=1.5)

    aligned = regime_series.reindex(equity_curve.index, method="ffill")
    if not aligned.empty:
        block_start = aligned.index[0]
        current = aligned.iloc[0]
        for ts, regime in aligned.items():
            if regime != current:
                color = "lightblue" if current == "trend" else "lightyellow"
                ax.axvspan(block_start, ts, color=color, alpha=0.3)
                block_start = ts
                current = regime
        color = "lightblue" if current == "trend" else "lightyellow"
        ax.axvspan(block_start, aligned.index[-1], color=color, alpha=0.3)

    ax.set_title("Equity Curve with Market Regime Shading")
    ax.set_ylabel("Portfolio Value")
    ax.set_xlabel("Date")
    fig.tight_layout()
    return fig


def plot_ensemble_weight_allocation(weight_history: pd.DataFrame) -> plt.Figure:
    """Stacked area chart of sleeve capital allocation."""
    fig, ax = plt.subplots(figsize=(14, 5))
    if weight_history.empty:
        ax.text(0.5, 0.5, "No weight history", ha="center", va="center")
        return fig

    ax.stackplot(
        weight_history.index,
        *[weight_history[col] for col in weight_history.columns],
        labels=weight_history.columns.tolist(),
    )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_title("Ensemble Capital Allocation Over Time")
    ax.set_ylabel("Weight")
    fig.tight_layout()
    return fig


def plot_cost_impact_waterfall(
    gross_return: float,
    slippage_cost: float,
    commission_cost: float,
    stt_cost: float,
    net_return: float,
) -> plt.Figure:
    """Waterfall chart decomposing gross return into cost components."""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ["Gross", "Slippage", "Commission", "STT", "Net"]
    values = [gross_return, -slippage_cost, -commission_cost, -stt_cost, net_return]
    running = [gross_return]
    for v in values[1:-1]:
        running.append(running[-1] + v)
    bottoms = [0] + running[:-1]
    colors = ["steelblue", "salmon", "salmon", "salmon", "seagreen"]

    for i, (label, val, bottom, color) in enumerate(
        zip(labels, values, bottoms + [0], colors)
    ):
        height = val if i == 0 else abs(val)
        bar_bottom = bottom if i not in (0, len(labels) - 1) else (0 if i == 0 else net_return)
        if i == len(labels) - 1:
            ax.bar(label, net_return, bottom=0, color=color)
        elif i == 0:
            ax.bar(label, gross_return, color=color)
        else:
            ax.bar(label, height, bottom=bottom + val, color=color)

    ax.set_title("Where Returns Go: Cost Attribution")
    ax.set_ylabel("Cumulative Return")
    fig.tight_layout()
    return fig


def plot_capacity_analysis(capacity_report: pd.DataFrame) -> plt.Figure:
    """Scatter of capacity-constrained orders over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    if capacity_report is None or capacity_report.empty:
        ax.text(
            0.5,
            0.5,
            "No capacity constraints triggered — strategy well within liquidity "
            "limits at tested capital level.",
            ha="center",
            va="center",
            wrap=True,
        )
        ax.set_axis_off()
        return fig

    for symbol, grp in capacity_report.groupby("symbol"):
        ax.scatter(
            grp["date"],
            grp["participation_pct"] * 100,
            label=symbol,
            alpha=0.7,
        )
    ax.set_title("Capacity-Constrained Orders Over Time")
    ax.set_ylabel("Participation %")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    return fig


def plotly_dsr_gauge(dsr_result: dict) -> go.Figure:
    """Interactive gauge for Deflated Sharpe Ratio."""
    dsr = dsr_result.get("deflated_sharpe_ratio", 0.0) * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=dsr,
            title={"text": "Deflated Sharpe Ratio (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 50], "color": "lightcoral"},
                    {"range": [50, 95], "color": "lightyellow"},
                    {"range": [95, 100], "color": "lightgreen"},
                ],
            },
        )
    )
    fig.add_annotation(
        text=dsr_result.get("interpretation", ""),
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.1,
        showarrow=False,
        font={"size": 12},
    )
    return fig
