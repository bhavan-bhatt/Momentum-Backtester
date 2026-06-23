# engine/audit_log.py
# ============================================================
# STRUCTURED AUDIT TRAIL FOR EVERY DECISION
# ============================================================

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """One structured record of a decision point in the backtest."""

    timestamp: datetime
    event_type: str
    strategy_id: Optional[str]
    symbol: str
    reason: str
    values: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[float] = None


class AuditLog:
    """Append-only log of meaningful system decisions during a run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.records: List[AuditRecord] = []

    def record(
        self,
        timestamp: datetime,
        event_type: str,
        symbol: str,
        reason: str,
        strategy_id: Optional[str] = None,
        values: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Append a new audit record and return it."""
        rec = AuditRecord(
            timestamp=timestamp,
            event_type=event_type,
            strategy_id=strategy_id,
            symbol=symbol,
            reason=reason,
            values=values or {},
        )
        self.records.append(rec)
        return rec

    def link_outcome(self, record: AuditRecord, net_pnl: float) -> None:
        """Attach realized P&L to a prior signal record."""
        record.outcome = net_pnl

    def to_dataframe(self) -> pd.DataFrame:
        """Convert audit log to a flat DataFrame."""
        if not self.records:
            return pd.DataFrame()

        rows = []
        for rec in self.records:
            row = asdict(rec)
            vals = row.pop("values", {}) or {}
            for k, v in vals.items():
                row[f"val_{k}"] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def save_jsonl(self, filepath: str) -> None:
        """Save audit log as JSON Lines."""
        with open(filepath, "w", encoding="utf-8") as f:
            for rec in self.records:
                record_dict = asdict(rec)
                ts = record_dict["timestamp"]
                record_dict["timestamp"] = (
                    ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                )
                f.write(json.dumps(record_dict, default=str) + "\n")
        logger.info(
            "Audit log saved: %d records → %s", len(self.records), filepath
        )

    def filter_by(
        self,
        event_type: Optional[str] = None,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[AuditRecord]:
        """Query audit log by optional filters."""
        results = []
        for rec in self.records:
            if event_type is not None and rec.event_type != event_type:
                continue
            if strategy_id is not None and rec.strategy_id != strategy_id:
                continue
            if symbol is not None and symbol not in rec.symbol:
                continue
            results.append(rec)
        return results
