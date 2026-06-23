# reports/generator.py
# ============================================================
# GENERATES HTML REPORTS, CSV EXPORTS, AND CONSOLE SUMMARIES
# ============================================================

import base64
import io
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from config import BacktestConfig
from performance.metrics import PerformanceMetrics
from reports.charts import (
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_returns_heatmap,
    plot_trade_distribution,
    plot_rolling_sharpe,
    plot_walk_forward_results,
    plotly_combined_dashboard,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Orchestrates the creation of all output files from a completed backtest.

    Output files
    ------------
    output/reports/{run_id}_report.html      — Full standalone HTML report
    output/reports/{run_id}_dashboard.html   — Interactive Plotly dashboard
    output/charts/{run_id}_equity.png        — Equity curve PNG
    output/charts/{run_id}_drawdown.png      — Drawdown PNG
    output/charts/{run_id}_monthly.png       — Monthly heatmap PNG
    output/charts/{run_id}_trades.png        — Trade distribution PNG
    output/charts/{run_id}_rolling_sharpe.png
    output/logs/{run_id}_trades.csv          — Trade log CSV
    output/logs/{run_id}_equity.csv          — Equity curve CSV
    output/logs/{run_id}_metrics.csv         — Metrics summary CSV
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        out          = config.report.output_dir

        self._dirs = {
            "reports": os.path.join(out, "reports"),
            "charts":  os.path.join(out, "charts"),
            "logs":    os.path.join(out, "logs"),
        }
        for d in self._dirs.values():
            os.makedirs(d, exist_ok=True)

        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = ts  # overridden in generate_all with strategy name

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────

    def generate_all(
        self,
        equity_curve: pd.Series,
        trade_log: List[dict],
        metrics: dict,
        benchmark_curve: Optional[pd.Series] = None,
        wf_results: Optional[List[dict]] = None,
        strategy_name: str = "Strategy",
    ) -> str:
        """
        Generate every output file and return the path to the main HTML report.
        """
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{strategy_name.replace(' ', '_')}_{ts}"

        perf            = PerformanceMetrics(self._config)
        monthly_returns = perf.calculate_monthly_returns(equity_curve)
        rolling_sharpe  = perf.calculate_rolling_sharpe(equity_curve)

        chart_paths: dict = {}

        # ── Matplotlib PNGs ───────────────────────────────────────────────
        if self._config.report.save_charts_png:
            chart_specs = [
                ("equity",         lambda: plot_equity_curve(equity_curve, benchmark_curve, self._config)),
                ("drawdown",       lambda: plot_drawdown(equity_curve, self._config)),
                ("monthly",        lambda: plot_monthly_returns_heatmap(monthly_returns, self._config)),
                ("trades",         lambda: plot_trade_distribution(trade_log, self._config)),
                ("rolling_sharpe", lambda: plot_rolling_sharpe(rolling_sharpe, self._config)),
            ]
            for name, fn in chart_specs:
                path = os.path.join(self._dirs["charts"], f"{self.run_id}_{name}.png")
                try:
                    fig = fn()
                    fig.savefig(path, dpi=120, bbox_inches="tight")
                    fig.clf()
                    import matplotlib.pyplot as plt
                    plt.close("all")
                    chart_paths[name] = path
                    logger.debug("Chart saved: %s", path)
                except Exception as exc:
                    logger.warning("Failed to generate chart '%s': %s", name, exc)

        # ── Walk-forward chart ────────────────────────────────────────────
        if wf_results:
            wf_engine = _build_wf_report_df(wf_results)
            try:
                path = os.path.join(self._dirs["charts"], f"{self.run_id}_walkforward.png")
                fig  = plot_walk_forward_results(wf_engine, self._config)
                fig.savefig(path, dpi=120, bbox_inches="tight")
                import matplotlib.pyplot as plt
                plt.close("all")
                chart_paths["walkforward"] = path
            except Exception as exc:
                logger.warning("Walk-forward chart failed: %s", exc)

        # ── Plotly dashboard HTML ─────────────────────────────────────────
        dash_path = os.path.join(self._dirs["reports"], f"{self.run_id}_dashboard.html")
        try:
            fig_dash = plotly_combined_dashboard(
                equity_curve, benchmark_curve, trade_log, metrics, monthly_returns
            )
            fig_dash.write_html(dash_path, include_plotlyjs="cdn")
            logger.info("Dashboard saved: %s", dash_path)
        except Exception as exc:
            logger.warning("Plotly dashboard failed: %s", exc)

        # ── Full HTML report ──────────────────────────────────────────────
        html     = self._build_html_report(
            equity_curve, trade_log, metrics, monthly_returns,
            benchmark_curve, wf_results, strategy_name, chart_paths,
        )
        rep_path = os.path.join(self._dirs["reports"], f"{self.run_id}_report.html")
        with open(rep_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("HTML report saved: %s", rep_path)

        # ── CSV exports ───────────────────────────────────────────────────
        if self._config.report.save_csv_trades:
            csv_path = os.path.join(self._dirs["logs"], f"{self.run_id}_trades.csv")
            self.save_trades_csv(trade_log, csv_path)

        if self._config.report.save_equity_curve_csv:
            csv_path = os.path.join(self._dirs["logs"], f"{self.run_id}_equity.csv")
            self.save_equity_csv(equity_curve, csv_path)

        metrics_path = os.path.join(self._dirs["logs"], f"{self.run_id}_metrics.csv")
        self.save_metrics_csv(metrics, metrics_path)

        self._print_console_summary(metrics, strategy_name)

        return rep_path

    def generate_advanced_sections(
        self,
        report_path: str,
        advanced_analysis: dict,
        audit_log=None,
    ) -> None:
        """Append Phase 2 analysis sections to an existing HTML report."""
        if not os.path.exists(report_path):
            logger.warning("Report not found for advanced sections: %s", report_path)
            return

        dsr = advanced_analysis.get("deflated_sharpe", {})
        ci = advanced_analysis.get("sharpe_ci", {})
        jb = advanced_analysis.get("jarque_bera", {})
        turnover = advanced_analysis.get("turnover", {})
        capacity = advanced_analysis.get("capacity", {})
        bench_rows = advanced_analysis.get("benchmark_comparisons", [])

        bench_html = ""
        for b in bench_rows:
            bench_html += (
                f"<tr><td>{b.get('benchmark_label','')}</td>"
                f"<td>{b.get('strategy_total_return',0):.2%}</td>"
                f"<td>{b.get('benchmark_total_return',0):.2%}</td>"
                f"<td>{b.get('outperformance',0):+.2%}</td></tr>"
            )

        extra = f"""
<section>
  <h2>Statistical Significance</h2>
  <table class="data-table">
    <tr><td>Bootstrap Sharpe CI</td>
        <td>{ci.get('point',0):.2f} [{ci.get('lower',0):.2f}, {ci.get('upper',0):.2f}]</td></tr>
    <tr><td>Deflated Sharpe Ratio</td>
        <td>{dsr.get('deflated_sharpe_ratio',0):.1%} — {dsr.get('interpretation','')}</td></tr>
    <tr><td>Trials assumed</td><td>{dsr.get('n_trials_assumed','')}</td></tr>
    <tr><td>Jarque-Bera</td><td>{jb.get('interpretation','')}</td></tr>
    <tr><td>Tail Ratio</td><td>{advanced_analysis.get('tail_ratio','N/A')}</td></tr>
    <tr><td>Omega Ratio</td><td>{advanced_analysis.get('omega_ratio','N/A')}</td></tr>
    <tr><td>Worst Recovery (days)</td><td>{advanced_analysis.get('worst_recovery_days','N/A')}</td></tr>
    <tr><td>Annual Turnover</td><td>{turnover.get('annual_turnover','N/A')}</td></tr>
    <tr><td>Est. Capacity (INR)</td><td>{capacity.get('estimated_capacity_inr','N/A')}</td></tr>
  </table>
</section>
<section>
  <h2>Benchmark Comparison</h2>
  <table class="data-table">
    <thead><tr><th>Benchmark</th><th>Strategy</th><th>Benchmark</th><th>Alpha</th></tr></thead>
    <tbody>{bench_html}</tbody>
  </table>
</section>
"""

        with open(report_path, "r", encoding="utf-8") as f:
            html = f.read()

        html = html.replace("</main>", extra + "\n</main>")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        chart_dir = self._dirs["charts"]
        try:
            from reports.advanced_charts import plot_sharpe_confidence_interval
            fig = plot_sharpe_confidence_interval(
                ci.get("point", 0),
                ci.get("lower", 0),
                ci.get("upper", 0),
                self._config,
            )
            path = os.path.join(chart_dir, f"{self.run_id}_sharpe_ci.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as exc:
            logger.debug("Advanced chart generation skipped: %s", exc)

        logger.info("Advanced sections appended to %s", report_path)

    # ──────────────────────────────────────────────────────────────────────
    # HTML REPORT BUILDER
    # ──────────────────────────────────────────────────────────────────────

    def _build_html_report(
        self,
        equity_curve: pd.Series,
        trade_log: List[dict],
        metrics: dict,
        monthly_returns: pd.DataFrame,
        benchmark_curve: Optional[pd.Series],
        wf_results: Optional[List[dict]],
        strategy_name: str,
        chart_paths: dict,
    ) -> str:
        """Build a fully self-contained HTML report (base64-embedded images)."""

        def _b64(path: str) -> str:
            try:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except Exception:
                return ""

        def _img(key: str) -> str:
            if key not in chart_paths:
                return ""
            b64 = _b64(chart_paths[key])
            if not b64:
                return ""
            return f'<img src="data:image/png;base64,{b64}" class="chart-img">'

        def _fmt(val, fmt_str: str = ".2%", fallback: str = "N/A") -> str:
            if val is None:
                return fallback
            try:
                return format(val, fmt_str)
            except Exception:
                return str(val)

        # ── Metric tiles ──────────────────────────────────────────────────
        tiles_html = _metric_tiles(metrics)

        # ── Metrics table ─────────────────────────────────────────────────
        table_rows = ""
        labels = {
            "total_return_pct":   ("Total Return",        ".2%"),
            "cagr":               ("CAGR",                ".2%"),
            "sharpe_ratio":       ("Sharpe Ratio",        ".3f"),
            "sortino_ratio":      ("Sortino Ratio",       ".3f"),
            "calmar_ratio":       ("Calmar Ratio",        ".3f"),
            "max_drawdown_pct":   ("Max Drawdown",        ".2%"),
            "max_dd_peak_date":   ("Peak Date",           "s"),
            "max_dd_trough_date": ("Trough Date",         "s"),
            "total_trades":       ("Total Trades",        ".0f"),
            "win_rate":           ("Win Rate",            ".2%"),
            "profit_factor":      ("Profit Factor",       ".3f"),
            "avg_win_loss_ratio": ("Avg Win/Loss Ratio",  ".3f"),
            "alpha":              ("Alpha (annual)",      ".2%"),
            "beta":               ("Beta",               ".3f"),
            "benchmark_cagr":     ("Benchmark CAGR",     ".2%"),
            "start_date":         ("Start Date",          "s"),
            "end_date":           ("End Date",            "s"),
            "start_capital":      ("Start Capital",      ",.0f"),
            "end_capital":        ("End Capital",        ",.0f"),
        }
        for key, (label, fmt_str) in labels.items():
            val  = metrics.get(key)
            disp = _fmt(val, fmt_str)
            row_class = ""
            if key in ("total_return_pct", "cagr") and val is not None:
                row_class = 'class="positive"' if val > 0 else 'class="negative"'
            table_rows += f"<tr {row_class}><td>{label}</td><td>{disp}</td></tr>\n"

        # ── Trade log table ────────────────────────────────────────────────
        trade_rows = ""
        for t in sorted(trade_log, key=lambda x: x.get("entry_date", "")):
            css = "win-row" if t.get("net_pnl", 0) > 0 else "loss-row"
            trade_rows += (
                f"<tr class='{css}'>"
                f"<td>{t.get('symbol','')}</td>"
                f"<td>{_ts(t.get('entry_date'))}</td>"
                f"<td>{_ts(t.get('exit_date'))}</td>"
                f"<td>{t.get('direction','')}</td>"
                f"<td>{t.get('quantity','')}</td>"
                f"<td>{_fmt(t.get('entry_price',0),',.2f')}</td>"
                f"<td>{_fmt(t.get('exit_price',0),',.2f')}</td>"
                f"<td>{_fmt(t.get('gross_pnl',0),',.2f')}</td>"
                f"<td>{_fmt(t.get('commission',0),',.2f')}</td>"
                f"<td>{_fmt(t.get('net_pnl',0),',.2f')}</td>"
                f"<td>{_fmt(t.get('return_pct',0),'.2%')}</td>"
                f"</tr>\n"
            )

        # ── Walk-forward section ───────────────────────────────────────────
        wf_section = ""
        if wf_results:
            df    = _build_wf_report_df(wf_results)
            wf_section = (
                "<h2>Walk-Forward Validation Results</h2>"
                + df.to_html(index=False, classes="data-table", border=0)
            )

        # ── Config section ─────────────────────────────────────────────────
        cfg = self._config
        cfg_lines = [
            f"Initial Capital: ₹{cfg.portfolio.initial_capital:,.0f}",
            f"Strategy: {strategy_name}",
            f"Date Range: {cfg.data.start_date} → {cfg.data.end_date}",
            f"Symbols: {', '.join(cfg.data.symbols)}",
            f"Fast Window: {cfg.strategy.fast_ma_window} | Slow Window: {cfg.strategy.slow_ma_window}",
            f"Sizing Method: {cfg.portfolio.sizing_method}",
            f"Risk Per Trade: {cfg.portfolio.risk_per_trade_pct:.1%}",
            f"Slippage: {cfg.execution.slippage_pct:.4%}",
            f"Commission: {cfg.execution.commission_pct:.4%}",
            f"STT: {cfg.execution.stt_pct:.4%}",
        ]
        cfg_html = "<br>".join(cfg_lines)

        run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Backtest Report — {strategy_name}</title>
  <style>
    :root {{
      --primary: #1f77b4; --green: #2ca02c; --red: #d62728;
      --bg: #f8f9fa; --card-bg: #ffffff; --border: #dee2e6;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body  {{ font-family: "Segoe UI", Arial, sans-serif; background: var(--bg);
              color: #333; line-height: 1.55; font-size: 14px; }}
    header {{ background: var(--primary); color: white; padding: 24px 32px; }}
    header h1 {{ font-size: 1.8rem; font-weight: 700; }}
    header p  {{ opacity: 0.85; margin-top: 4px; }}
    main  {{ max-width: 1400px; margin: 0 auto; padding: 28px 24px; }}
    section {{ margin-bottom: 36px; }}
    h2    {{ font-size: 1.15rem; font-weight: 600; color: var(--primary);
             border-bottom: 2px solid var(--border); padding-bottom: 6px;
             margin-bottom: 16px; }}
    .tiles {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }}
    .tile  {{ background: var(--card-bg); border: 1px solid var(--border);
              border-radius: 8px; padding: 16px 22px; min-width: 160px;
              box-shadow: 0 1px 4px rgba(0,0,0,.06); text-align: center; flex: 1; }}
    .tile .label {{ font-size: .8rem; color: #666; text-transform: uppercase;
                    letter-spacing: .5px; }}
    .tile .value {{ font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
    .tile.good  .value {{ color: var(--green); }}
    .tile.bad   .value {{ color: var(--red);   }}
    .tile.neutral .value {{ color: var(--primary); }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    table.data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    table.data-table th {{ background: var(--primary); color: white;
                           padding: 8px 12px; text-align: left; }}
    table.data-table td {{ padding: 6px 12px; border-bottom: 1px solid var(--border); }}
    table.data-table tr:nth-child(even) td {{ background: #f2f6fb; }}
    .win-row  td {{ background: rgba(44,160,44,0.07)  !important; }}
    .loss-row td {{ background: rgba(214,39,40,0.07)  !important; }}
    .positive {{ color: var(--green); }}
    .negative {{ color: var(--red);   }}
    .chart-img {{ width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
    details summary {{ cursor: pointer; font-weight: 600; color: var(--primary);
                       padding: 6px 0; user-select: none; }}
    .meta {{ font-size: .8rem; color: #888; margin-top: 8px; }}
  </style>
</head>
<body>

<header>
  <h1>📊 Backtest Report — {strategy_name}</h1>
  <p>
    Period: {metrics.get('start_date','—')} → {metrics.get('end_date','—')}
    &nbsp;|&nbsp; Capital: ₹{metrics.get('start_capital',0):,.0f}
    &nbsp;|&nbsp; Generated: {run_ts}
  </p>
</header>

<main>

  <!-- SECTION 1: KEY METRIC TILES -->
  <section>
    <h2>Key Performance Metrics</h2>
    {tiles_html}
  </section>

  <!-- SECTION 2: DETAILED METRICS TABLE -->
  <section>
    <h2>Detailed Metrics</h2>
    <div class="two-col">
      <table class="data-table">
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
  </section>

  <!-- SECTION 3: CHARTS -->
  <section>
    <h2>Charts</h2>
    {_img('equity')}
    {_img('drawdown')}
    {_img('monthly')}
    {_img('trades')}
    {_img('rolling_sharpe')}
    {_img('walkforward')}
  </section>

  <!-- SECTION 4: TRADE LOG -->
  <section>
    <h2>Trade Log ({len(trade_log)} completed trades)</h2>
    <div style="overflow-x:auto">
    <table class="data-table">
      <thead>
        <tr>
          <th>Symbol</th><th>Entry Date</th><th>Exit Date</th>
          <th>Dir</th><th>Qty</th>
          <th>Entry ₹</th><th>Exit ₹</th>
          <th>Gross P&amp;L</th><th>Commission</th>
          <th>Net P&amp;L</th><th>Return %</th>
        </tr>
      </thead>
      <tbody>{trade_rows}</tbody>
    </table>
    </div>
  </section>

  <!-- SECTION 5: WALK-FORWARD -->
  {wf_section}

  <!-- SECTION 6: CONFIG -->
  <section>
    <details>
      <summary>Configuration</summary>
      <p style="margin-top:12px; font-size:13px; line-height:2">{cfg_html}</p>
    </details>
  </section>

</main>
</body>
</html>"""
        return html

    # ──────────────────────────────────────────────────────────────────────
    # CONSOLE SUMMARY
    # ──────────────────────────────────────────────────────────────────────

    def _print_console_summary(self, metrics: dict, strategy_name: str) -> None:
        """Print a boxed performance summary to stdout."""
        def pct(v):  return f"{v:.2%}" if v is not None else "N/A"
        def f3(v):   return f"{v:.3f}" if v is not None else "N/A"
        def num(v):  return f"{int(v)}" if v is not None else "N/A"

        width = 56
        line  = "═" * width
        print(f"\n╔{line}╗")
        print(f"║{'  BACKTEST RESULTS — ' + strategy_name:^{width}}║")
        period = (
            metrics.get("start_date","?")
            + " → "
            + metrics.get("end_date","?")
        )
        print(f"║{'  Period: ' + period:^{width}}║")
        print(f"╠{line}╣")

        rows = [
            ("CAGR",            pct(metrics.get("cagr"))),
            ("Total Return",    pct(metrics.get("total_return_pct"))),
            ("Sharpe Ratio",    f3(metrics.get("sharpe_ratio"))),
            ("Sortino Ratio",   f3(metrics.get("sortino_ratio"))),
            ("Calmar Ratio",    f3(metrics.get("calmar_ratio"))),
            ("Max Drawdown",    pct(metrics.get("max_drawdown_pct"))),
            ("Win Rate",        pct(metrics.get("win_rate"))),
            ("Profit Factor",   f3(metrics.get("profit_factor"))),
            ("Total Trades",    num(metrics.get("total_trades"))),
            ("Alpha (annual)",  pct(metrics.get("alpha"))),
            ("Beta",            f3(metrics.get("beta"))),
            ("Benchmark CAGR",  pct(metrics.get("benchmark_cagr"))),
        ]
        for label, val in rows:
            content = f"  {label:<22} {val}"
            print(f"║{content:<{width}}║")

        print(f"╚{line}╝\n")

    # ──────────────────────────────────────────────────────────────────────
    # CSV EXPORTS
    # ──────────────────────────────────────────────────────────────────────

    def save_trades_csv(self, trade_log: List[dict], filepath: str) -> None:
        """Save trade log as CSV, sorted by entry date."""
        if not trade_log:
            logger.info("No completed trades — skipping trade CSV.")
            return
        df = pd.DataFrame(trade_log)
        if "entry_date" in df.columns:
            df = df.sort_values("entry_date")
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].round(4)
        df.to_csv(filepath, index=False)
        logger.info("Trade log saved: %d trades → %s", len(df), filepath)

    def save_equity_csv(self, equity_curve: pd.Series, filepath: str) -> None:
        """Save equity curve as CSV with daily_return and cumulative_return columns."""
        df = equity_curve.reset_index()
        df.columns = ["date", "portfolio_value"]
        df["daily_return"]       = equity_curve.pct_change().values
        df["cumulative_return"]  = equity_curve / equity_curve.iloc[0] - 1
        df.to_csv(filepath, index=False)
        logger.info("Equity curve saved: %d rows → %s", len(df), filepath)

    def save_metrics_csv(self, metrics: dict, filepath: str) -> None:
        """Save metrics dict as a two-column CSV (metric_name, value)."""
        rows = [{"metric": k, "value": v} for k, v in metrics.items()]
        pd.DataFrame(rows).to_csv(filepath, index=False)
        logger.info("Metrics saved: %d metrics → %s", len(rows), filepath)


# ──────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _metric_tiles(metrics: dict) -> str:
    """Build the 5 large metric tiles as an HTML string."""
    def _pct(v): return f"{v:.2%}" if v is not None else "N/A"
    def _f3(v):  return f"{v:.2f}" if v is not None else "N/A"

    cagr    = metrics.get("cagr", 0) or 0
    sharpe  = metrics.get("sharpe_ratio", 0) or 0
    max_dd  = metrics.get("max_drawdown_pct", 0) or 0
    wr      = metrics.get("win_rate", 0) or 0
    pf      = metrics.get("profit_factor", 0) or 0

    def _css(val: float, lo: float, hi: float) -> str:
        if val >= hi: return "good"
        if val < lo:  return "bad"
        return "neutral"

    tiles = [
        ("CAGR",         _pct(cagr),    _css(cagr, 0, 0.12)),
        ("Sharpe Ratio", _f3(sharpe),   _css(sharpe, 0, 1.0)),
        ("Max Drawdown", _pct(max_dd),  "bad" if max_dd < -0.20 else "neutral"),
        ("Win Rate",     _pct(wr),      _css(wr, 0, 0.50)),
        ("Profit Factor",_f3(pf),       _css(pf, 1.0, 1.5)),
    ]
    html = '<div class="tiles">'
    for label, val, css in tiles:
        html += (
            f'<div class="tile {css}">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{val}</div>'
            f"</div>"
        )
    html += "</div>"
    return html


def _ts(val) -> str:
    """Convert a datetime-like value to a short date string."""
    if val is None:
        return ""
    try:
        return pd.Timestamp(val).strftime("%Y-%m-%d")
    except Exception:
        return str(val)


def _build_wf_report_df(wf_results: List[dict]) -> pd.DataFrame:
    """Build a formatted DataFrame from walk-forward split results."""
    if not wf_results:
        return pd.DataFrame()
    df = pd.DataFrame(wf_results)
    col_map = {
        "split_index":      "Split",
        "test_start":       "Test Start",
        "test_end":         "Test End",
        "cagr":             "CAGR",
        "sharpe_ratio":     "Sharpe",
        "max_drawdown_pct": "Max DD",
        "sortino_ratio":    "Sortino",
        "win_rate":         "Win Rate",
        "total_trades":     "# Trades",
    }
    avail = {k: v for k, v in col_map.items() if k in df.columns}
    df    = df[list(avail.keys())].rename(columns=avail)
    return df
