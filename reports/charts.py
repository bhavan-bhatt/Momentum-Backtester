# reports/charts.py
# ============================================================
# CHART FUNCTIONS — MATPLOTLIB + PLOTLY
# All functions accept pre-computed data (DataFrames/Series)
# and return figure objects. Callers decide whether to save or show.
# ============================================================

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy imports so the module can be imported even if libs are missing
def _get_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for report generation
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.gridspec import GridSpec
        return matplotlib, plt, mdates, GridSpec
    except ImportError as e:
        raise ImportError(f"matplotlib not installed. Run: pip install matplotlib. Error: {e}")


def _get_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.subplots as sp
        return go, sp
    except ImportError as e:
        raise ImportError(f"plotly not installed. Run: pip install plotly. Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB CHARTS (static PNGs)
# ══════════════════════════════════════════════════════════════════════════════

def equity_curve_chart(
    equity: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
    strategy_name: str = "Strategy",
    chart_style: str = "seaborn-v0_8",
    save_path: Optional[str] = None,
):
    """
    Plot equity curve with drawdown panel.

    Parameters
    ----------
    equity           : pd.Series indexed by datetime — portfolio equity.
    benchmark_equity : pd.Series or None — benchmark equity (normalised to same start).
    strategy_name    : str — chart title label.
    chart_style      : str — matplotlib style name.
    save_path        : str or None — if given, save PNG to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    matplotlib, plt, mdates, GridSpec = _get_mpl()

    try:
        plt.style.use(chart_style)
    except OSError:
        plt.style.use("seaborn-v0_8" if "seaborn" not in chart_style else "ggplot")

    fig = plt.figure(figsize=(14, 8))
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1], hspace=0.08)

    ax_eq  = fig.add_subplot(gs[0])
    ax_dd  = fig.add_subplot(gs[1], sharex=ax_eq)
    ax_ret = fig.add_subplot(gs[2], sharex=ax_eq)

    # ── Equity curve ─────────────────────────────────────────────────────
    norm_equity = equity / equity.iloc[0] * 100
    ax_eq.plot(equity.index, norm_equity, label=strategy_name, linewidth=1.8, color="#2196F3")

    if benchmark_equity is not None and len(benchmark_equity) > 1:
        norm_bench = benchmark_equity / benchmark_equity.iloc[0] * 100
        ax_eq.plot(
            benchmark_equity.index, norm_bench,
            label="Benchmark", linewidth=1.2, color="#FF9800", alpha=0.8, linestyle="--"
        )

    ax_eq.axhline(100, color="gray", linewidth=0.8, linestyle=":")
    ax_eq.set_ylabel("Normalised Value (base=100)")
    ax_eq.set_title(f"{strategy_name} — Equity Curve & Drawdown", fontsize=13, fontweight="bold")
    ax_eq.legend(loc="upper left", framealpha=0.8)
    ax_eq.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))

    # ── Drawdown ─────────────────────────────────────────────────────────
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max * 100
    ax_dd.fill_between(drawdown.index, drawdown, 0, color="#F44336", alpha=0.5, label="Drawdown")
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.legend(loc="lower left", framealpha=0.8)

    # ── Daily returns ─────────────────────────────────────────────────────
    daily_ret = equity.pct_change().dropna() * 100
    colors = ["#4CAF50" if r >= 0 else "#F44336" for r in daily_ret]
    ax_ret.bar(daily_ret.index, daily_ret, color=colors, width=1, alpha=0.7)
    ax_ret.axhline(0, color="gray", linewidth=0.6)
    ax_ret.set_ylabel("Daily Ret %")

    # ── X-axis formatting ─────────────────────────────────────────────────
    ax_ret.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_ret.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_eq.get_xticklabels(), visible=False)
    plt.setp(ax_dd.get_xticklabels(), visible=False)
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved equity chart → %s", save_path)

    return fig


def monthly_returns_heatmap(
    equity: pd.Series,
    chart_style: str = "seaborn-v0_8",
    save_path: Optional[str] = None,
):
    """
    Monthly returns heatmap (seaborn-style table).

    Rows = year, Columns = month, Cells = monthly return %.

    Returns
    -------
    matplotlib Figure object.
    """
    matplotlib, plt, mdates, GridSpec = _get_mpl()
    try:
        import seaborn as sns
        _have_sns = True
    except ImportError:
        _have_sns = False

    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    monthly.index = pd.DatetimeIndex(monthly.index)

    df = pd.DataFrame({
        "year":  monthly.index.year,
        "month": monthly.index.month,
        "ret":   monthly.values,
    })
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6)))
    if _have_sns:
        sns.heatmap(
            pivot, annot=True, fmt=".1f", center=0,
            cmap="RdYlGn", linewidths=0.5, ax=ax,
            cbar_kws={"label": "Return %"},
        )
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label="Return %")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)

    ax.set_title("Monthly Returns Heatmap (%)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved monthly heatmap → %s", save_path)

    return fig


def trade_analysis_chart(
    trade_log: pd.DataFrame,
    chart_style: str = "seaborn-v0_8",
    save_path: Optional[str] = None,
):
    """
    3-panel trade analysis chart:
      - PnL distribution histogram
      - Cumulative PnL over time
      - Rolling win-rate (20-trade window)

    Returns
    -------
    matplotlib Figure object.
    """
    matplotlib, plt, mdates, GridSpec = _get_mpl()

    exits = trade_log[trade_log["type"] == "EXIT"].copy()
    if exits.empty:
        logger.warning("No exit trades found — skipping trade analysis chart.")
        return None

    exits = exits.sort_values("date")
    pnl   = exits["realised_pnl"].values

    try:
        plt.style.use(chart_style)
    except OSError:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Trade Analysis", fontsize=13, fontweight="bold")

    # ── PnL Distribution ─────────────────────────────────────────────────
    ax = axes[0]
    colors = ["#4CAF50" if p >= 0 else "#F44336" for p in pnl]
    ax.hist(pnl, bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.axvline(pnl.mean(), color="orange", linewidth=1.2, linestyle="-.", label=f"Mean={pnl.mean():.0f}")
    ax.set_xlabel("Realised PnL (₹)")
    ax.set_ylabel("Frequency")
    ax.set_title("PnL Distribution")
    ax.legend()

    # ── Cumulative PnL ───────────────────────────────────────────────────
    ax = axes[1]
    cum_pnl = pd.Series(pnl).cumsum()
    cum_pnl.index = range(len(cum_pnl))
    pos_mask = cum_pnl >= 0
    ax.fill_between(cum_pnl.index, cum_pnl, 0,
                    where=pos_mask, color="#4CAF50", alpha=0.5, label="Profit")
    ax.fill_between(cum_pnl.index, cum_pnl, 0,
                    where=~pos_mask, color="#F44336", alpha=0.5, label="Loss")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL (₹)")
    ax.set_title("Cumulative PnL")
    ax.legend()

    # ── Rolling Win-Rate ─────────────────────────────────────────────────
    ax = axes[2]
    wins        = (pd.Series(pnl) > 0).astype(float)
    rolling_wr  = wins.rolling(window=20, min_periods=5).mean() * 100
    ax.plot(rolling_wr.values, color="#9C27B0", linewidth=1.5)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", label="50%")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Win Rate %")
    ax.set_title("Rolling Win-Rate (20 trades)")
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved trade analysis chart → %s", save_path)

    return fig


def walk_forward_summary_chart(
    wf_summary: pd.DataFrame,
    metric: str = "sharpe_ratio",
    chart_style: str = "seaborn-v0_8",
    save_path: Optional[str] = None,
):
    """
    Bar chart of a chosen metric across walk-forward splits.

    Returns
    -------
    matplotlib Figure object.
    """
    matplotlib, plt, mdates, GridSpec = _get_mpl()

    df = wf_summary[wf_summary["split"] != "MEAN"].copy()
    if metric not in df.columns:
        logger.warning("Metric '%s' not found in walk-forward summary.", metric)
        return None

    try:
        plt.style.use(chart_style)
    except OSError:
        pass

    fig, ax = plt.subplots(figsize=(12, 5))
    vals   = df[metric].astype(float)
    labels = [f"Split {int(s)}" for s in df["split"]]
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in vals]

    bars = ax.bar(labels, vals, color=colors, edgecolor="white", alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8)

    mean_val = wf_summary[wf_summary["split"] == "MEAN"][metric].values
    if len(mean_val):
        ax.axhline(
            float(mean_val[0]), color="orange", linewidth=1.5,
            linestyle="--", label=f"Mean={float(mean_val[0]):.3f}"
        )

    ax.set_xlabel("Walk-Forward Split")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Walk-Forward {metric.replace('_', ' ').title()} per Split", fontweight="bold")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved walk-forward chart → %s", save_path)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY INTERACTIVE CHART
# ══════════════════════════════════════════════════════════════════════════════

def interactive_equity_chart(
    equity: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
    trade_log: Optional[pd.DataFrame] = None,
    strategy_name: str = "Strategy",
    save_path: Optional[str] = None,
):
    """
    Plotly interactive equity curve with drawdown, volume-of-trades, and
    optional trade markers.

    Parameters
    ----------
    equity           : pd.Series — portfolio equity indexed by datetime.
    benchmark_equity : pd.Series or None.
    trade_log        : pd.DataFrame or None — to overlay entry/exit markers.
    strategy_name    : str.
    save_path        : str or None — if given, save as HTML file.

    Returns
    -------
    plotly Figure object.
    """
    go, sp = _get_plotly()

    rows = 3
    fig  = sp.make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=["Equity Curve", "Drawdown", "Daily Returns"],
    )

    # ── Equity ───────────────────────────────────────────────────────────
    norm = equity / equity.iloc[0] * 100
    fig.add_trace(
        go.Scatter(
            x=equity.index, y=norm,
            name=strategy_name, line=dict(color="#2196F3", width=2),
        ),
        row=1, col=1,
    )

    if benchmark_equity is not None and len(benchmark_equity) > 1:
        norm_b = benchmark_equity / benchmark_equity.iloc[0] * 100
        fig.add_trace(
            go.Scatter(
                x=benchmark_equity.index, y=norm_b,
                name="Benchmark", line=dict(color="#FF9800", width=1.5, dash="dash"),
                opacity=0.8,
            ),
            row=1, col=1,
        )

    # Trade markers
    if trade_log is not None and not trade_log.empty:
        entries = trade_log[trade_log["type"] == "ENTRY"]
        exits   = trade_log[trade_log["type"] == "EXIT"]

        def _equity_at(dates):
            ts = pd.to_datetime(dates)
            return [
                float(equity.reindex([t], method="nearest").values[0])
                / equity.iloc[0] * 100
                if len(equity.reindex([t], method="nearest")) > 0 else None
                for t in ts
            ]

        if not entries.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(entries["date"]),
                y=_equity_at(entries["date"]),
                mode="markers",
                name="Entry",
                marker=dict(symbol="triangle-up", size=8, color="#4CAF50"),
            ), row=1, col=1)

        if not exits.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(exits["date"]),
                y=_equity_at(exits["date"]),
                mode="markers",
                name="Exit",
                marker=dict(symbol="triangle-down", size=8, color="#F44336"),
            ), row=1, col=1)

    # ── Drawdown ─────────────────────────────────────────────────────────
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max * 100
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown,
            name="Drawdown", fill="tozeroy",
            line=dict(color="#F44336", width=1),
            fillcolor="rgba(244,67,54,0.3)",
        ),
        row=2, col=1,
    )

    # ── Daily returns ─────────────────────────────────────────────────────
    daily_ret = equity.pct_change().dropna() * 100
    colors    = ["#4CAF50" if r >= 0 else "#F44336" for r in daily_ret]
    fig.add_trace(
        go.Bar(
            x=daily_ret.index, y=daily_ret,
            name="Daily Ret %", marker_color=colors, opacity=0.6,
        ),
        row=3, col=1,
    )

    fig.update_layout(
        title=f"<b>{strategy_name}</b> — Interactive Performance",
        height=700,
        hovermode="x unified",
        showlegend=True,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Value (base=100)", row=1, col=1)
    fig.update_yaxes(title_text="DD %",  row=2, col=1)
    fig.update_yaxes(title_text="Ret %", row=3, col=1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path, include_plotlyjs="cdn")
        logger.info("Saved interactive chart → %s", save_path)

    return fig
