# reports/charts.py
# ============================================================
# ALL CHART FUNCTIONS — MATPLOTLIB (PNG) AND PLOTLY (INTERACTIVE)
# Each function accepts processed data and returns a figure.
# ============================================================

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for file saving

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from config import BacktestConfig

_PALETTE = {
    "portfolio": "#1f77b4",
    "benchmark": "#9e9e9e",
    "green":     "#2ca02c",
    "red":       "#d62728",
    "yellow":    "#ff7f0e",
}


# ═════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB CHARTS  (return plt.Figure, saved by caller)
# ═════════════════════════════════════════════════════════════════════════════

def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series],
    config: BacktestConfig,
    title: str = "Portfolio Equity Curve",
) -> plt.Figure:
    """Portfolio equity curve vs benchmark, indexed to 100 at start."""
    style = getattr(config.report, "chart_style", "seaborn-v0_8-darkgrid")
    try:
        plt.style.use(style)
    except Exception:
        plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(14, 6))

    port_idx = equity_curve / equity_curve.iloc[0] * 100
    ax.plot(port_idx.index, port_idx.values, color=_PALETTE["portfolio"],
            linewidth=1.8, label="Portfolio")
    ax.fill_between(port_idx.index, port_idx.values, 100,
                    alpha=0.10, color=_PALETTE["portfolio"])

    if benchmark_curve is not None and len(benchmark_curve) > 0:
        bm_idx = benchmark_curve / benchmark_curve.iloc[0] * 100
        # Align to portfolio dates
        bm_aligned = bm_idx.reindex(port_idx.index, method="ffill")
        ax.plot(bm_aligned.index, bm_aligned.values, color=_PALETTE["benchmark"],
                linewidth=1.2, linestyle="--", label="Nifty 50")

    ax.axhline(y=100, color="black", linewidth=0.8, linestyle=":")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Indexed Value (Base 100)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_drawdown(
    equity_curve: pd.Series,
    config: BacktestConfig,
    title: str = "Drawdown Over Time",
) -> plt.Figure:
    """Drawdown percentage as a filled red area chart."""
    running_max = equity_curve.cummax()
    drawdown    = (equity_curve - running_max) / running_max

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(drawdown.index, drawdown.values * 100,
            color=_PALETTE["red"], linewidth=1.2, label="Drawdown")
    ax.fill_between(drawdown.index, drawdown.values * 100, 0,
                    alpha=0.30, color=_PALETTE["red"])

    trough_date = drawdown.idxmin()
    max_dd      = drawdown.min()
    ax.axvline(x=trough_date, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.annotate(
        f"Max DD: {max_dd:.1%}",
        xy=(trough_date, max_dd * 100),
        xytext=(20, -10),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "black"},
    )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_monthly_returns_heatmap(
    monthly_returns: pd.DataFrame,
    config: BacktestConfig,
    title: str = "Monthly Returns Heatmap",
) -> plt.Figure:
    """Calendar heatmap — rows = years, columns = months."""
    n_rows = max(4, len(monthly_returns))
    fig, ax = plt.subplots(figsize=(13, n_rows * 0.65 + 2))

    sns.heatmap(
        monthly_returns * 100,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Return (%)", "shrink": 0.6},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def plot_trade_distribution(
    trade_log: List[dict],
    config: BacktestConfig,
    title: str = "Trade P&L Distribution",
) -> plt.Figure:
    """Histogram of trade P&L with win/loss breakdown."""
    if not trade_log:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, "No completed trades", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig

    pnls   = [t["net_pnl"] for t in trade_log]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    # Histogram of P&L
    colors = [_PALETTE["green"] if p > 0 else _PALETTE["red"] for p in pnls]
    ax1.hist(pnls, bins=min(30, len(pnls)), edgecolor="white", color=_PALETTE["portfolio"])
    ax1.axvline(x=0, color="black", linewidth=1.2, linestyle="--")
    ax1.set_title("P&L Distribution")
    ax1.set_xlabel("Net P&L (₹)")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.4)

    # Win/Loss bar count
    counts = [len(wins), len(losses)]
    bar_colors = [_PALETTE["green"], _PALETTE["red"]]
    bars = ax2.bar(["Wins", "Losses"], counts, color=bar_colors, edgecolor="white")
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 str(cnt), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_title("Win / Loss Count")
    ax2.set_ylabel("Number of Trades")
    ax2.grid(True, alpha=0.4, axis="y")

    # Stats annotation
    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win  = float(np.mean(wins))   if wins   else 0
    avg_loss = float(np.mean(losses)) if losses else 0
    pf       = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float("inf")
    stats    = (
        f"Win Rate:       {win_rate:.1%}\n"
        f"Profit Factor:  {pf:.2f}\n"
        f"Avg Win:  ₹{avg_win:>10,.0f}\n"
        f"Avg Loss: ₹{avg_loss:>10,.0f}"
    )
    ax1.annotate(
        stats, xy=(0.98, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(
    rolling_sharpe: pd.Series,
    config: BacktestConfig,
    title: str = "Rolling 252-Day Sharpe Ratio",
) -> plt.Figure:
    """Rolling Sharpe over time — shows strategy consistency."""
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
            color=_PALETTE["portfolio"], linewidth=1.5, label="Rolling Sharpe")
    ax.axhline(y=0,   color="black",  linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(y=1.0, color=_PALETTE["green"], linewidth=0.8,
               linestyle="--", alpha=0.7, label="Sharpe = 1.0")

    ax.fill_between(
        rolling_sharpe.index, rolling_sharpe.values, 0,
        where=rolling_sharpe.values >= 0,
        alpha=0.15, color=_PALETTE["green"],
    )
    ax.fill_between(
        rolling_sharpe.index, rolling_sharpe.values, 0,
        where=rolling_sharpe.values < 0,
        alpha=0.15, color=_PALETTE["red"],
    )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_walk_forward_results(
    wf_df: pd.DataFrame,
    config: BacktestConfig,
    title: str = "Walk-Forward Out-of-Sample Results",
) -> plt.Figure:
    """Grouped bar chart of Sharpe and CAGR per walk-forward test window."""
    # Drop the aggregate 'MEAN' row for plotting
    df = wf_df[wf_df["Split"] != "MEAN"].copy() if "Split" in wf_df.columns else wf_df.copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    x_labels = df.get("Test Start", df.index.astype(str)).tolist()
    x        = np.arange(len(x_labels))

    def _colour(val: float) -> str:
        if val >= 1.0: return _PALETTE["green"]
        if val >= 0.0: return _PALETTE["yellow"]
        return _PALETTE["red"]

    sharpe_col = "Sharpe" if "Sharpe" in df.columns else "sharpe_ratio"
    cagr_col   = "CAGR"   if "CAGR"   in df.columns else "cagr"

    if sharpe_col in df.columns:
        sharpe_vals = df[sharpe_col].astype(float)
        ax1.bar(x, sharpe_vals, color=[_colour(v) for v in sharpe_vals], edgecolor="white")
        ax1.axhline(sharpe_vals.mean(), color="black", linewidth=1.2, linestyle="--",
                    label=f"Mean = {sharpe_vals.mean():.2f}")
        ax1.set_ylabel("Sharpe Ratio")
        ax1.set_title("Sharpe Ratio per Test Window")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.4, axis="y")

    if cagr_col in df.columns:
        cagr_vals = df[cagr_col].astype(float)
        ax2.bar(x, cagr_vals * 100, color=[_colour(v) for v in cagr_vals], edgecolor="white")
        ax2.axhline(cagr_vals.mean() * 100, color="black", linewidth=1.2, linestyle="--",
                    label=f"Mean = {cagr_vals.mean():.1%}")
        ax2.set_ylabel("CAGR (%)")
        ax2.set_title("CAGR per Test Window")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.4, axis="y")

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(lbl)[:7] for lbl in x_labels], rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_ma_signals(
    bars: pd.DataFrame,
    fast_ma: pd.Series,
    slow_ma: pd.Series,
    trade_log: List[dict],
    symbol: str,
    config: BacktestConfig,
) -> plt.Figure:
    """Debug chart: close price with MA lines and entry/exit trade markers."""
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(bars.index, bars["close"], color="black", linewidth=1.0, label="Close")
    ax.plot(fast_ma.index, fast_ma.values, color=_PALETTE["portfolio"],
            linewidth=1.2, label=f"Fast MA ({config.strategy.fast_ma_window})")
    ax.plot(slow_ma.index, slow_ma.values, color=_PALETTE["yellow"],
            linewidth=1.2, label=f"Slow MA ({config.strategy.slow_ma_window})")

    for t in trade_log:
        if t.get("symbol") != symbol:
            continue
        try:
            entry_dt = pd.Timestamp(t["entry_date"])
            exit_dt  = pd.Timestamp(t["exit_date"])
            ax.annotate(
                "▲", xy=(entry_dt, t["entry_price"]),
                fontsize=12, color=_PALETTE["green"],
                ha="center", va="bottom",
            )
            ax.annotate(
                "▼", xy=(exit_dt, t["exit_price"]),
                fontsize=12, color=_PALETTE["red"],
                ha="center", va="top",
            )
        except Exception:
            pass

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(f"Dual MA Signals — {symbol}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_rsi_signals(
    bars: pd.DataFrame,
    rsi: pd.Series,
    trade_log: List[dict],
    symbol: str,
    config: BacktestConfig,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> plt.Figure:
    """Debug chart: price with trade markers (top) and RSI with bands (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 2]})

    ax1.plot(bars.index, bars["close"], color="black", linewidth=1.0, label="Close")
    for t in trade_log:
        if t.get("symbol") != symbol:
            continue
        try:
            ax1.annotate(
                "▲", xy=(pd.Timestamp(t["entry_date"]), t["entry_price"]),
                fontsize=12, color=_PALETTE["green"], ha="center", va="bottom",
            )
            ax1.annotate(
                "▼", xy=(pd.Timestamp(t["exit_date"]), t["exit_price"]),
                fontsize=12, color=_PALETTE["red"], ha="center", va="top",
            )
        except Exception:
            pass
    ax1.set_title(f"RSI Signals — {symbol}", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Price (₹)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.4)

    ax2.plot(rsi.index, rsi.values, color=_PALETTE["portfolio"], linewidth=1.2)
    ax2.axhline(overbought, color=_PALETTE["red"],   linestyle="--", linewidth=0.9)
    ax2.axhline(oversold,   color=_PALETTE["green"], linestyle="--", linewidth=0.9)
    ax2.fill_between(rsi.index, rsi.values, overbought,
                     where=rsi.values >= overbought, alpha=0.20, color=_PALETTE["red"])
    ax2.fill_between(rsi.index, rsi.values, oversold,
                     where=rsi.values <= oversold,   alpha=0.20, color=_PALETTE["green"])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.4)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# PLOTLY INTERACTIVE CHARTS
# ═════════════════════════════════════════════════════════════════════════════

def plotly_equity_curve(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series],
    metrics: dict,
) -> go.Figure:
    """Interactive equity curve with hover tooltips and metric annotation."""
    port_idx = equity_curve / equity_curve.iloc[0] * 100
    fig      = go.Figure()

    fig.add_trace(go.Scatter(
        x=port_idx.index, y=port_idx.values,
        name="Portfolio", mode="lines",
        line={"color": _PALETTE["portfolio"], "width": 2},
        hovertemplate="<b>Portfolio</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.1f}<extra></extra>",
    ))

    if benchmark_curve is not None and len(benchmark_curve) > 0:
        bm_idx = benchmark_curve / benchmark_curve.iloc[0] * 100
        bm_idx = bm_idx.reindex(port_idx.index, method="ffill")
        fig.add_trace(go.Scatter(
            x=bm_idx.index, y=bm_idx.values,
            name="Nifty 50", mode="lines",
            line={"color": _PALETTE["benchmark"], "width": 1.5, "dash": "dash"},
            hovertemplate="<b>Nifty 50</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.1f}<extra></extra>",
        ))

    sharpe = metrics.get("sharpe_ratio", 0)
    cagr   = metrics.get("cagr", 0)
    max_dd = metrics.get("max_drawdown_pct", 0)

    fig.add_annotation(
        x=0.01, y=0.97, xref="paper", yref="paper",
        text=(
            f"<b>Sharpe:</b> {sharpe:.2f}  |  "
            f"<b>CAGR:</b> {cagr:.1%}  |  "
            f"<b>Max DD:</b> {max_dd:.1%}"
        ),
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="lightgrey", borderwidth=1,
        font={"size": 12},
    )

    fig.update_layout(
        title="Portfolio Equity Curve vs Benchmark (Indexed to 100)",
        xaxis_title="Date",
        yaxis_title="Indexed Value",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        height=450,
        template="plotly_white",
    )
    return fig


def plotly_drawdown(equity_curve: pd.Series) -> go.Figure:
    """Interactive drawdown area chart."""
    running_max = equity_curve.cummax()
    drawdown    = (equity_curve - running_max) / running_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        name="Drawdown",
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(214,39,40,0.2)",
        line={"color": "rgba(214,39,40,0.8)", "width": 1.2},
        hovertemplate="<b>Drawdown</b><br>Date: %{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        height=350,
        template="plotly_white",
    )
    return fig


def plotly_monthly_heatmap(monthly_returns: pd.DataFrame) -> go.Figure:
    """Interactive Plotly heatmap of monthly returns."""
    data_pct  = (monthly_returns * 100).round(2)
    text_vals = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
                 for row in data_pct.values]

    fig = go.Figure(go.Heatmap(
        z=data_pct.values,
        x=data_pct.columns.tolist(),
        y=data_pct.index.tolist(),
        text=text_vals,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        colorbar={"title": "Return (%)"},
        hovertemplate=(
            "<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Monthly Returns Heatmap",
        xaxis_title="Month",
        yaxis_title="Year",
        height=max(250, len(monthly_returns) * 40 + 100),
        template="plotly_white",
    )
    return fig


def plotly_combined_dashboard(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series],
    trade_log: List[dict],
    metrics: dict,
    monthly_returns: pd.DataFrame,
) -> go.Figure:
    """
    A single multi-panel Plotly dashboard (2 × 2 grid):
      Row 1, Col 1 — Equity curve vs benchmark
      Row 1, Col 2 — Monthly returns heatmap
      Row 2, Col 1 — Drawdown
      Row 2, Col 2 — Trade P&L histogram
    """
    specs = [
        [{"type": "xy"},      {"type": "heatmap"}],
        [{"type": "xy"},      {"type": "xy"}],
    ]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Equity Curve vs Benchmark",
            "Monthly Returns Heatmap",
            "Drawdown",
            "Trade P&L Distribution",
        ),
        specs=specs,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # ── Equity curve (1,1) ────────────────────────────────────────────────
    port_idx = equity_curve / equity_curve.iloc[0] * 100
    fig.add_trace(
        go.Scatter(x=port_idx.index, y=port_idx.values,
                   name="Portfolio", mode="lines",
                   line={"color": _PALETTE["portfolio"], "width": 2}),
        row=1, col=1,
    )
    if benchmark_curve is not None and len(benchmark_curve) > 0:
        bm_idx = (benchmark_curve / benchmark_curve.iloc[0] * 100).reindex(
            port_idx.index, method="ffill"
        )
        fig.add_trace(
            go.Scatter(x=bm_idx.index, y=bm_idx.values,
                       name="Nifty 50", mode="lines",
                       line={"color": _PALETTE["benchmark"], "width": 1.2, "dash": "dash"}),
            row=1, col=1,
        )

    # ── Monthly heatmap (1,2) ─────────────────────────────────────────────
    data_pct  = (monthly_returns * 100).round(2)
    text_vals = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
                 for row in data_pct.values]
    fig.add_trace(
        go.Heatmap(
            z=data_pct.values, x=data_pct.columns.tolist(),
            y=data_pct.index.tolist(), text=text_vals,
            texttemplate="%{text}", colorscale="RdYlGn", zmid=0, showscale=False,
        ),
        row=1, col=2,
    )

    # ── Drawdown (2,1) ────────────────────────────────────────────────────
    running_max = equity_curve.cummax()
    drawdown    = (equity_curve - running_max) / running_max * 100
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values,
                   name="Drawdown", mode="lines",
                   fill="tozeroy",
                   fillcolor="rgba(214,39,40,0.15)",
                   line={"color": "rgba(214,39,40,0.8)", "width": 1}),
        row=2, col=1,
    )

    # ── Trade P&L histogram (2,2) ─────────────────────────────────────────
    if trade_log:
        pnls = [t["net_pnl"] for t in trade_log]
        colors = [_PALETTE["green"] if p > 0 else _PALETTE["red"] for p in pnls]
        fig.add_trace(
            go.Bar(x=list(range(len(pnls))), y=pnls,
                   name="Trade P&L", marker_color=colors, showlegend=False),
            row=2, col=2,
        )

    # ── Metric tiles as title annotations ─────────────────────────────────
    sharpe  = metrics.get("sharpe_ratio", 0)
    cagr    = metrics.get("cagr", 0)
    max_dd  = metrics.get("max_drawdown_pct", 0)
    wr      = metrics.get("win_rate", 0)
    pf      = metrics.get("profit_factor", 0)
    tiles   = (
        f"<b>Sharpe</b> {sharpe:.2f}  ·  "
        f"<b>CAGR</b> {cagr:.1%}  ·  "
        f"<b>Max DD</b> {max_dd:.1%}  ·  "
        f"<b>Win Rate</b> {wr:.1%}  ·  "
        f"<b>PF</b> {pf:.2f}"
    )
    fig.update_layout(
        title={
            "text": f"Backtest Dashboard<br><sup>{tiles}</sup>",
            "x": 0.5, "xanchor": "center",
        },
        height=800,
        hovermode="x unified",
        template="plotly_white",
        showlegend=True,
    )
    return fig
