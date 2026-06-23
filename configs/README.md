# Configuration Files

YAML configs override `config.py` defaults via `config_loader.py`.

| File | Purpose |
|------|---------|
| `base.yaml` | Default ensemble of 4 trend/range strategies |
| `dual_ma.yaml` | Single-sleeve Dual MA research run |
| `cross_sectional_momentum.yaml` | Cross-sectional momentum only |
| `pairs_stat_arb.yaml` | Pairs trading (needs `allow_short: true`) |
| `ensemble_all.yaml` | Full 4-sleeve ensemble with advanced execution |

Usage:

```bash
python run_research.py --config configs/base.yaml
python run_research.py --config configs/ensemble_all.yaml --with-stability
```

Unknown YAML keys raise `ValueError` immediately — typos are not silently ignored.
