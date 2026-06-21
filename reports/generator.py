# reports/generator.py
# ============================================================
# HTML REPORT BUILDER
# Assembles a self-contained HTML report from:
#   - Performance metrics table
#   - Embedded plotly interactive chart
#   - Trade log table
#   - Monthly returns heatmap (base64-encoded PNG)
# ============================================================

import base64
import io
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import BacktestConfig

logger = logging.getLogger(__name__)


def _metric_fmt(key: str, val: float) -> str:
    """Format a metric value for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    pct_keys = {
        "total_return", "cagr", "annualised_volatility",
        "max_drawdown", "var_95", "cvar_95", "alpha",
        "benchmark_return", "win_rate",
    }
    ratio_keys = {
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "profit_factor", "beta", "information_ratio",
    }
    if key in pct_keys:
        return f"{val * 100:.2f}%"
    elif key in ratio_keys:
        return f"{val:.3f}"
    elif key == "num_trades":
        return f"{int(val)}"
    elif key == "max_dd_duration_days":
        return f"{int(val)} days"
    elif key in ("avg_trade_pnl", "expectancy"):
        return f"₹{val:,.0f}"
    return f"{val:.4f}"


def _fig_to_base64_png(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return encoded


def _plotly_to_html_div(fig) -> str:
    """Render a plotly figure as an HTML div (no full page wrapper)."""
    try:
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception as e:
        logger.warning("Could not render plotly figure: %s", e)
        return "<p><em>Interactive chart unavailable.</em></p>"


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0f1117;
      --surface: #1a1d27;
      --border: #2d3148;
      --text: #e8eaf6;
      --muted: #9fa8da;
      --green: #4caf50;
      --red: #ef5350;
      --accent: #5c6bc0;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; padding: 24px; }}
    h1 {{ font-size: 1.8rem; color: var(--accent); margin-bottom: 4px; }}
    h2 {{ font-size: 1.15rem; color: var(--muted); margin: 28px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
    .meta {{ font-size: 0.85rem; color: var(--muted); margin-bottom: 24px; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 24px; }}
    .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; }}
    .metric-label {{ font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .metric-value {{ font-size: 1.4rem; font-weight: 600; margin-top: 4px; }}
    .positive {{ color: var(--green); }}
    .negative {{ color: var(--red); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    th {{ background: var(--surface); color: var(--muted); padding: 8px 12px; text-align: left; font-weight: 500; border-bottom: 2px solid var(--border); }}
    td {{ padding: 7px 12px; border-bottom: 1px solid var(--border); }}
    tr:hover td {{ background: #1e2135; }}
    .chart-container {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
    img.chart {{ max-width: 100%; border-radius: 6px; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
    .badge-green {{ background: rgba(76,175,80,0.15); color: var(--green); }}
    .badge-red   {{ background: rgba(239,83,80,0.15); color: var(--red); }}
    footer {{ margin-top: 40px; font-size: 0.75rem; color: var(--muted); text-align: center; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta">Generated: {generated_at} &nbsp;|&nbsp; Strategy: {strategy_id} &nbsp;|&nbsp; Period: {period}</p>

  <h2>Key Metrics</h2>
  <div class="grid-2">
    {metric_cards}
  </div>

  <h2>Interactive Equity Curve</h2>
  <div class="chart-container">
    {plotly_div}
  </div>

  {monthly_heatmap_section}

  <h2>Full Metrics</h2>
  <div class="card">
    {metrics_table}
  </div>

  <h2>Trade Log</h2>
  <div class="card" style="overflow-x:auto">
    {trade_table}
  </div>

  <footer>
    Backtester &mdash; for research purposes only. Not financial advice.
  </footer>
</body>
</html>
"""


def _make_metric_card(label: str, raw_key: str, val: float) -> str:
    formatted = _metric_fmt(raw_key, val)
    is_pos    = val >= 0 if (isinstance(val, float) and not np.isnan(val)) else True
    css       = "positive" if is_pos else "negative"
    return (
        f'<div class="card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value {css}">{formatted}</div>'
        f'</div>'
    )


def _metrics_table_html(metrics: Dict[str, float]) -> str:
    rows = ""
    for key, val in metrics.items():
        label = key.replace("_", " ").title()
        fmt   = _metric_fmt(key, val)
        if isinstance(val, float) and not np.isnan(val):
            css = "positive" if val >= 0 else "negative"
        else:
            css = ""
        rows += f"<tr><td>{label}</td><td class='{css}'><strong>{fmt}</strong></td></tr>"
    return f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"


def _trade_table_html(trade_log: pd.DataFrame, max_rows: int = 200) -> str:
    if trade_log is None or trade_log.empty:
        return "<p><em>No trades recorded.</em></p>"

    df = trade_log.head(max_rows).copy()
    for col in ("fill_price", "commission", "slippage"):
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "")
    if "realised_pnl" in df.columns:
        def _pnl(x):
            if pd.isna(x) or x == 0:
                return ""
            badge = "badge-green" if x > 0 else "badge-red"
            return f'<span class="badge {badge}">₹{x:,.0f}</span>'
        df["realised_pnl"] = df["realised_pnl"].map(_pnl)

    # Date formatting
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    cols_order = [
        c for c in ["date","symbol","type","direction","quantity",
                     "fill_price","commission","realised_pnl","strategy_id"]
        if c in df.columns
    ]
    df = df[cols_order]

    header = "".join(f"<th>{c.replace('_',' ').title()}</th>" for c in df.columns)
    body   = ""
    for _, row in df.iterrows():
        body += "<tr>" + "".join(f"<td>{v}</td>" for v in row.values) + "</tr>"

    if len(trade_log) > max_rows:
        note = f"<p><em>Showing first {max_rows} of {len(trade_log)} trades.</em></p>"
    else:
        note = ""

    return f"{note}<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


class ReportGenerator:
    """
    Builds and saves a self-contained HTML performance report.

    Usage
    -----
    gen = ReportGenerator(config)
    gen.generate(
        equity=equity_series,
        trade_log=trade_df,
        metrics=metrics_dict,
        strategy_id="DualMA_SMA_20_50",
        benchmark_equity=bench_series,   # optional
    )
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config

    def generate(
        self,
        equity: pd.Series,
        trade_log: pd.DataFrame,
        metrics: Dict[str, float],
        strategy_id: str,
        benchmark_equity: Optional[pd.Series] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Build the HTML report and write it to disk.

        Parameters
        ----------
        equity           : pd.Series — portfolio equity curve.
        trade_log        : pd.DataFrame — trade log from PortfolioManager.
        metrics          : dict — output of compute_all_metrics().
        strategy_id      : str — displayed in the report header.
        benchmark_equity : pd.Series or None.
        filename         : str or None — output filename (without directory).
                           Defaults to "{strategy_id}_{date}.html".

        Returns
        -------
        str — path to the saved HTML file.
        """
        rcfg     = self._config.report
        out_dir  = os.path.join(rcfg.output_dir, "reports")
        os.makedirs(out_dir, exist_ok=True)

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename is None:
            safe_sid  = strategy_id.replace("/", "_").replace(" ", "_")
            filename  = f"{safe_sid}_{timestamp}.html"
        out_path = os.path.join(out_dir, filename)

        # ── Build interactive plotly chart ───────────────────────────────
        plotly_div = "<p><em>Plotly not available.</em></p>"
        try:
            from reports.charts import interactive_equity_chart
            plotly_fig = interactive_equity_chart(
                equity, benchmark_equity, trade_log, strategy_id
            )
            plotly_div = _plotly_to_html_div(plotly_fig)
        except Exception as e:
            logger.warning("Could not build interactive chart: %s", e)

        # ── Build monthly heatmap PNG (embedded) ──────────────────────────
        monthly_heatmap_section = ""
        if rcfg.save_charts_png and len(equity) > 30:
            try:
                from reports.charts import monthly_returns_heatmap
                hm_fig   = monthly_returns_heatmap(equity, rcfg.chart_style)
                b64      = _fig_to_base64_png(hm_fig)
                monthly_heatmap_section = (
                    f'<h2>Monthly Returns Heatmap</h2>'
                    f'<div class="chart-container">'
                    f'<img class="chart" src="data:image/png;base64,{b64}" alt="Monthly Returns" />'
                    f'</div>'
                )
                import matplotlib.pyplot as plt
                plt.close(hm_fig)
            except Exception as e:
                logger.warning("Could not build monthly heatmap: %s", e)

        # ── Key metric cards ─────────────────────────────────────────────
        highlight = [
            ("Total Return",  "total_return"),
            ("CAGR",          "cagr"),
            ("Sharpe Ratio",  "sharpe_ratio"),
            ("Max Drawdown",  "max_drawdown"),
            ("Sortino Ratio", "sortino_ratio"),
            ("Win Rate",      "win_rate"),
        ]
        cards_html = "".join(
            _make_metric_card(label, key, metrics.get(key, np.nan))
            for label, key in highlight
        )

        # ── Period string ─────────────────────────────────────────────────
        period = (
            f"{self._config.data.start_date} → {self._config.data.end_date}"
        )

        html = _HTML_TEMPLATE.format(
            title              = f"Backtest Report — {strategy_id}",
            generated_at       = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            strategy_id        = strategy_id,
            period             = period,
            metric_cards       = cards_html,
            plotly_div         = plotly_div,
            monthly_heatmap_section = monthly_heatmap_section,
            metrics_table      = _metrics_table_html(metrics),
            trade_table        = _trade_table_html(trade_log),
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("HTML report saved → %s", out_path)
        if self._config.verbose:
            print(f"[Report] Saved → {out_path}")

        return out_path

    def save_csv_artifacts(
        self,
        equity: pd.Series,
        trade_log: pd.DataFrame,
        strategy_id: str,
    ) -> None:
        """Save equity curve and trade log as CSV files."""
        rcfg    = self._config.report
        out_dir = rcfg.output_dir
        os.makedirs(out_dir, exist_ok=True)

        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id = strategy_id.replace("/", "_").replace(" ", "_")

        if rcfg.save_equity_curve_csv and not equity.empty:
            path = os.path.join(out_dir, f"{safe_id}_equity_{ts}.csv")
            equity.to_csv(path, header=True)
            logger.info("Equity curve saved → %s", path)

        if rcfg.save_csv_trades and not trade_log.empty:
            path = os.path.join(out_dir, f"{safe_id}_trades_{ts}.csv")
            trade_log.to_csv(path, index=False)
            logger.info("Trade log saved → %s", path)

    def save_static_charts(
        self,
        equity: pd.Series,
        trade_log: pd.DataFrame,
        strategy_id: str,
        benchmark_equity: Optional[pd.Series] = None,
    ) -> None:
        """Save static PNG chart files to output/charts/."""
        rcfg    = self._config.report
        if not rcfg.save_charts_png:
            return

        charts_dir = os.path.join(rcfg.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        safe_id = strategy_id.replace("/", "_").replace(" ", "_")
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            import matplotlib.pyplot as plt
            from reports.charts import equity_curve_chart, trade_analysis_chart, monthly_returns_heatmap

            fig = equity_curve_chart(
                equity, benchmark_equity, strategy_id, rcfg.chart_style,
                save_path=os.path.join(charts_dir, f"{safe_id}_equity_{ts}.png"),
            )
            plt.close(fig)

            if not trade_log.empty:
                fig = trade_analysis_chart(
                    trade_log, rcfg.chart_style,
                    save_path=os.path.join(charts_dir, f"{safe_id}_trades_{ts}.png"),
                )
                if fig:
                    plt.close(fig)

            fig = monthly_returns_heatmap(
                equity, rcfg.chart_style,
                save_path=os.path.join(charts_dir, f"{safe_id}_monthly_{ts}.png"),
            )
            plt.close(fig)

        except Exception as e:
            logger.warning("Static chart generation failed: %s", e)
