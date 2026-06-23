# Configuration Files

YAML configs override `config.py` defaults via `config_loader.py`.

| File | Purpose |
|------|---------|
| `base.yaml` | **Primary research config** — 150-stock cross-sectional momentum |
| `symbols_n150.txt` | 150 liquid NSE tickers (`.NS` suffix for yfinance) |
| `symbols_n100.txt` | 100-stock subset (faster runs) |
| `dual_ma.yaml` | Single-sleeve Dual MA research run |
| `cross_sectional_momentum.yaml` | Cross-sectional momentum only |
| `pairs_stat_arb.yaml` | Pairs trading (needs `allow_short: true`) |
| `ensemble_all.yaml` | Full 4-sleeve ensemble with advanced execution |

Usage:

```bash
./scripts/run_research.sh --config configs/base.yaml
python run_research.py --config configs/ensemble_all.yaml --with-stability
```

Symbol lists are referenced from YAML via `data.symbols_file`.

Unknown YAML keys raise `ValueError` immediately — typos are not silently ignored.
