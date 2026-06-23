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
    plotly_multi_benchmark_equity,
    plotly_underwater,
    plotly_monthly_heatmap,
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
        bench_comparisons: Optional[List[dict]] = None,
        benchmark_curves: Optional[Dict[str, pd.Series]] = None,
        advanced_analysis: Optional[dict] = None,
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

        # ── Embedded Plotly fragments for main report ─────────────────────
        plotly_fragments: Dict[str, str] = {}
        try:
            curves = benchmark_curves or {}
            if benchmark_curve is not None and "Nifty 50" not in curves:
                curves = {"Nifty 50": benchmark_curve, **curves}
            fig_eq = plotly_multi_benchmark_equity(equity_curve, curves, metrics)
            plotly_fragments["equity"] = fig_eq.to_html(
                full_html=False, include_plotlyjs=False, config={"displayModeBar": False}
            )
            fig_dd = plotly_underwater(equity_curve)
            plotly_fragments["drawdown"] = fig_dd.to_html(
                full_html=False, include_plotlyjs=False, config={"displayModeBar": False}
            )
            fig_hm = plotly_monthly_heatmap(monthly_returns)
            plotly_fragments["monthly"] = fig_hm.to_html(
                full_html=False, include_plotlyjs=False, config={"displayModeBar": False}
            )
        except Exception as exc:
            logger.warning("Embedded Plotly charts failed: %s", exc)

        # ── Full HTML report ──────────────────────────────────────────────
        html     = self._build_html_report(
            equity_curve, trade_log, metrics, monthly_returns,
            benchmark_curve, wf_results, strategy_name, chart_paths,
            bench_comparisons=bench_comparisons or [],
            plotly_fragments=plotly_fragments,
            advanced_analysis=advanced_analysis,
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
        """Legacy hook — advanced sections are now built into the main report."""
        if not os.path.exists(report_path):
            logger.warning("Report not found for advanced sections: %s", report_path)
            return

        chart_dir = self._dirs["charts"]
        ci = advanced_analysis.get("sharpe_ci", {})
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

        logger.info("Advanced sections included in %s", report_path)

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
        bench_comparisons: Optional[List[dict]] = None,
        plotly_fragments: Optional[Dict[str, str]] = None,
        advanced_analysis: Optional[dict] = None,
    ) -> str:
        """Build a minimalist standalone HTML report with embedded Plotly charts."""

        def _fmt(val, fmt_str: str = ".2%", fallback: str = "N/A") -> str:
            if val is None:
                return fallback
            try:
                return format(val, fmt_str)
            except Exception:
                return str(val)

        bench_comparisons = bench_comparisons or []
        plotly_fragments = plotly_fragments or {}
        advanced_analysis = advanced_analysis or {}

        n_symbols = len(self._config.data.symbols)
        run_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        # ── Hero metrics ──────────────────────────────────────────────────
        hero = _hero_metrics(metrics, bench_comparisons)

        # ── Benchmark cards ───────────────────────────────────────────────
        bench_cards = _benchmark_cards(bench_comparisons)

        # ── Stats grid ────────────────────────────────────────────────────
        stats_html = _stats_grid(metrics, advanced_analysis)

        # ── Plotly chart panels ───────────────────────────────────────────
        equity_plot = plotly_fragments.get("equity", "")
        dd_plot = plotly_fragments.get("drawdown", "")
        monthly_plot = plotly_fragments.get("monthly", "")

        # ── Trade log (collapsed) ─────────────────────────────────────────
        trade_rows = ""
        for t in sorted(trade_log, key=lambda x: x.get("entry_date", ""))[-200:]:
            pnl = t.get("net_pnl", 0)
            css = "pos" if pnl > 0 else "neg"
            trade_rows += (
                f"<tr class='{css}'>"
                f"<td>{t.get('symbol','')}</td>"
                f"<td>{_ts(t.get('entry_date'))}</td>"
                f"<td>{_ts(t.get('exit_date'))}</td>"
                f"<td>{t.get('strategy_id','')[:20]}</td>"
                f"<td class='num'>{_fmt(t.get('net_pnl',0),',.0f')}</td>"
                f"<td class='num'>{_fmt(t.get('return_pct',0),'.2%')}</td>"
                f"</tr>\n"
            )
        trade_note = ""
        if len(trade_log) > 200:
            trade_note = f"<p class='muted'>Showing last 200 of {len(trade_log)} trades.</p>"

        # ── Walk-forward ──────────────────────────────────────────────────
        wf_section = ""
        if wf_results:
            df = _build_wf_report_df(wf_results)
            wf_section = (
                "<section class='panel'><h2>Walk-Forward Validation</h2>"
                + df.to_html(index=False, classes="compact-table", border=0)
                + "</section>"
            )

        cfg = self._config
        cfg_html = (
            f"{n_symbols} symbols · {cfg.data.start_date} → {cfg.data.end_date} · "
            f"₹{cfg.portfolio.initial_capital:,.0f} capital · "
            f"{cfg.portfolio.sizing_method} sizing · "
            f"{', '.join(cfg.advanced.ensemble.enabled_strategies)}"
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{strategy_name} — Research Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #f8fafc; --surface: #ffffff; --border: #e2e8f0;
      --text: #0f172a; --muted: #64748b; --accent: #2563eb;
      --positive: #059669; --negative: #dc2626;
      --radius: 12px; --shadow: 0 1px 3px rgba(15,23,42,.06);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.5; font-size: 14px;
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 32px 24px 64px; }}
    header {{
      margin-bottom: 32px; padding-bottom: 24px; border-bottom: 1px solid var(--border);
    }}
    header h1 {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }}
    header .meta {{ color: var(--muted); font-size: 0.875rem; margin-top: 6px; }}
    .hero {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 16px; margin-bottom: 28px;
    }}
    .hero-card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 20px; box-shadow: var(--shadow);
    }}
    .hero-card .label {{
      font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--muted); font-weight: 500;
    }}
    .hero-card .value {{
      font-size: 1.75rem; font-weight: 700; margin-top: 4px; letter-spacing: -0.03em;
    }}
    .hero-card .sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 2px; }}
    .hero-card.up .value {{ color: var(--positive); }}
    .hero-card.down .value {{ color: var(--negative); }}
    .hero-card.neutral .value {{ color: var(--text); }}
    .panel {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 24px; margin-bottom: 20px; box-shadow: var(--shadow);
    }}
    h2 {{
      font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.08em; color: var(--muted); margin-bottom: 16px;
    }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 900px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
    .chart-box {{ min-height: 280px; }}
    .bench-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px;
    }}
    .bench-card {{
      border: 1px solid var(--border); border-radius: 10px; padding: 16px;
      background: #fafbfc;
    }}
    .bench-card .name {{ font-weight: 600; font-size: 0.85rem; margin-bottom: 8px; }}
    .bench-card .row {{
      display: flex; justify-content: space-between; font-size: 0.8rem;
      color: var(--muted); margin-top: 4px;
    }}
    .bench-card .alpha {{
      margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border);
      font-weight: 600; font-size: 0.9rem;
    }}
    .bench-card .alpha.pos {{ color: var(--positive); }}
    .bench-card .alpha.neg {{ color: var(--negative); }}
    .stats-grid {{
      display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1px;
      background: var(--border); border: 1px solid var(--border); border-radius: 10px;
      overflow: hidden;
    }}
    .stat-cell {{ background: var(--surface); padding: 12px 16px; }}
    .stat-cell .k {{ font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .stat-cell .v {{ font-size: 0.95rem; font-weight: 600; margin-top: 2px; }}
    table.compact-table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    table.compact-table th {{
      text-align: left; padding: 8px 10px; background: #f1f5f9;
      font-weight: 600; color: var(--muted); font-size: 0.72rem; text-transform: uppercase;
    }}
    table.compact-table td {{ padding: 7px 10px; border-top: 1px solid var(--border); }}
    table.compact-table tr.pos td:last-child {{ color: var(--positive); }}
    table.compact-table tr.neg td:last-child {{ color: var(--negative); }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    details summary {{
      cursor: pointer; font-weight: 600; color: var(--accent); padding: 4px 0;
    }}
    .muted {{ color: var(--muted); font-size: 0.85rem; }}
    footer {{ margin-top: 40px; text-align: center; color: var(--muted); font-size: 0.75rem; }}
  </style>
</head>
<body>
<div class="wrap">

<header>
  <h1>{strategy_name}</h1>
  <p class="meta">
    {metrics.get('start_date','—')} → {metrics.get('end_date','—')}
    · {n_symbols} stocks · Generated {run_ts}
  </p>
</header>

{hero}

<section class="panel">
  <h2>Benchmark Comparison</h2>
  <div class="bench-grid">{bench_cards}</div>
</section>

<section class="panel">
  <h2>Performance</h2>
  <div class="chart-box">{equity_plot}</div>
</section>

<div class="chart-grid">
  <section class="panel">
    <h2>Drawdown</h2>
    <div class="chart-box">{dd_plot}</div>
  </section>
  <section class="panel">
    <h2>Monthly Returns</h2>
    <div class="chart-box">{monthly_plot}</div>
  </section>
</div>

<section class="panel">
  <h2>Risk &amp; Statistics</h2>
  {stats_html}
</section>

{wf_section}

<section class="panel">
  <details>
    <summary>Trade Log ({len(trade_log)} trades)</summary>
    {trade_note}
    <div style="overflow-x:auto; margin-top:12px">
      <table class="compact-table">
        <thead><tr>
          <th>Symbol</th><th>Entry</th><th>Exit</th><th>Strategy</th>
          <th>Net P&amp;L</th><th>Return</th>
        </tr></thead>
        <tbody>{trade_rows}</tbody>
      </table>
    </div>
  </details>
  <details style="margin-top:16px">
    <summary>Configuration</summary>
    <p class="muted" style="margin-top:8px">{cfg_html}</p>
  </details>
</section>

<footer>Quant Research Backtester · {self.run_id}</footer>
</div>
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

def _hero_metrics(metrics: dict, bench_comparisons: List[dict]) -> str:
    """Top hero metric cards."""
    def _pct(v):
        return f"{v:.2%}" if v is not None else "N/A"

    cagr = metrics.get("cagr", 0) or 0
    total = metrics.get("total_return_pct", 0) or 0
    sharpe = metrics.get("sharpe_ratio", 0) or 0
    alpha = metrics.get("alpha", 0) or 0

    ew_alpha = None
    for b in bench_comparisons:
        if "Equal-Weight" in b.get("benchmark_label", "") and "cost-adjusted" in b.get("benchmark_label", ""):
            ew_alpha = b.get("outperformance")
            break
    if ew_alpha is None:
        for b in bench_comparisons:
            if "Equal-Weight" in b.get("benchmark_label", ""):
                ew_alpha = b.get("outperformance")
                break

    cards = [
        ("CAGR", _pct(cagr), "Compound annual growth", "up" if cagr > 0.10 else "neutral"),
        ("Total Return", _pct(total), f"₹{metrics.get('end_capital',0):,.0f} ending value", "up" if total > 0 else "down"),
        ("Sharpe Ratio", f"{sharpe:.2f}", "Risk-adjusted return", "up" if sharpe > 0.5 else "neutral"),
        ("Alpha vs Nifty 50", _pct(alpha), "Annual excess return", "up" if alpha > 0 else "down"),
    ]
    if ew_alpha is not None:
        cards.append((
            "vs Equal-Weight",
            f"{ew_alpha:+.2%}",
            "Total return alpha",
            "up" if ew_alpha > 0 else "down",
        ))

    html = '<div class="hero">'
    for label, val, sub, css in cards:
        html += (
            f'<div class="hero-card {css}">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{val}</div>'
            f'<div class="sub">{sub}</div></div>'
        )
    html += "</div>"
    return html


def _benchmark_cards(bench_comparisons: List[dict]) -> str:
    """Styled benchmark comparison cards."""
    if not bench_comparisons:
        return "<p class='muted'>No benchmark data available.</p>"

    html = ""
    for b in bench_comparisons:
        label = b.get("benchmark_label", "")
        strat = b.get("strategy_total_return", 0)
        bench = b.get("benchmark_total_return", 0)
        alpha = b.get("outperformance", 0)
        css = "pos" if alpha > 0 else "neg"
        short = label.replace(" Buy & Hold", "").replace(" (cost-adjusted)", " *")
        html += f"""
<div class="bench-card">
  <div class="name">{short}</div>
  <div class="row"><span>Strategy</span><span>{strat:.2%}</span></div>
  <div class="row"><span>Benchmark</span><span>{bench:.2%}</span></div>
  <div class="alpha {css}">Alpha {alpha:+.2%}</div>
</div>"""
    return html


def _stats_grid(metrics: dict, advanced: dict) -> str:
    """Compact statistics grid."""
    dsr = advanced.get("deflated_sharpe", {})
    ci = advanced.get("sharpe_ci", {})
    turnover = advanced.get("turnover", {})

    def _fmt(v, f=".2%"):
        if v is None:
            return "N/A"
        try:
            return format(v, f)
        except Exception:
            return str(v)

    cells = [
        ("Max Drawdown", _fmt(metrics.get("max_drawdown_pct"))),
        ("Sortino", _fmt(metrics.get("sortino_ratio"), ".2f")),
        ("Calmar", _fmt(metrics.get("calmar_ratio"), ".2f")),
        ("Win Rate", _fmt(metrics.get("win_rate"))),
        ("Profit Factor", _fmt(metrics.get("profit_factor"), ".2f")),
        ("Beta", _fmt(metrics.get("beta"), ".2f")),
        ("Total Trades", str(int(metrics.get("total_trades", 0)))),
        ("Annual Turnover", str(turnover.get("annual_turnover", "N/A"))),
        ("Deflated Sharpe", _fmt(dsr.get("deflated_sharpe_ratio"))),
        ("Sharpe CI", f"{ci.get('point',0):.2f} [{ci.get('lower',0):.2f}, {ci.get('upper',0):.2f}]"),
        ("Tail Ratio", str(advanced.get("tail_ratio", "N/A"))),
        ("Omega Ratio", str(advanced.get("omega_ratio", "N/A"))),
    ]
    html = '<div class="stats-grid">'
    for k, v in cells:
        html += f'<div class="stat-cell"><div class="k">{k}</div><div class="v">{v}</div></div>'
    html += "</div>"
    return html


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
