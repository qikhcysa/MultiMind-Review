"""End-to-end audit trail: records every agent decision with full context."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models import AuditEntry


class AuditTrail:
    """
    Persists every agent decision to a JSONL log file and keeps an in-memory
    index for fast querying.

    File layout: ``<log_dir>/audit_<YYYY-MM-DD>.jsonl``
    """

    def __init__(self, log_dir: str | None = None, enabled: bool = True) -> None:
        self.log_dir = Path(log_dir or os.getenv("AUDIT_LOG_DIR", "./audit_logs"))
        self.enabled = enabled
        self._entries: list[AuditEntry] = []
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def record(self, entry: AuditEntry) -> None:
        """Persist an audit entry both in-memory and to disk."""
        self._entries.append(entry)
        if self.enabled:
            log_path = self._today_log_path()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")

    def record_many(self, entries: list[AuditEntry]) -> None:
        """Persist multiple entries."""
        for entry in entries:
            self.record(entry)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_all(self) -> list[AuditEntry]:
        """Return all in-memory audit entries."""
        return list(self._entries)

    def get_by_review(self, review_id: str) -> list[AuditEntry]:
        """Return all audit entries for a specific review."""
        return [e for e in self._entries if e.review_id == review_id]

    def get_by_stage(self, stage: str) -> list[AuditEntry]:
        """Return all audit entries for a specific pipeline stage."""
        return [e for e in self._entries if e.stage == stage]

    def load_from_date(self, date: str) -> list[AuditEntry]:
        """
        Load entries from a specific date's log file.

        Args:
            date: ISO date string, e.g. ``"2024-01-15"``.
        """
        log_path = self.log_dir / f"audit_{date}.jsonl"
        if not log_path.exists():
            return []
        entries: list[AuditEntry] = []
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(AuditEntry.model_validate_json(line))
                    except Exception:
                        continue
        return entries

    def to_dataframe(self, entries: list[AuditEntry] | None = None):
        """Convert audit entries to a pandas DataFrame."""
        import pandas as pd

        data = entries if entries is not None else self._entries
        rows = []
        for e in data:
            rows.append(
                {
                    "entry_id": e.entry_id,
                    "review_id": e.review_id,
                    "stage": e.stage,
                    "agent_name": e.agent_name,
                    "reasoning": e.reasoning,
                    "top_similarity": e.similarities[0] if e.similarities else None,
                    "num_neighbors": len(e.retrieved_neighbors),
                    "timestamp": e.timestamp.isoformat(),
                }
            )
        return pd.DataFrame(rows)

    def available_dates(self) -> list[str]:
        """Return sorted list of dates for which log files exist."""
        if not self.log_dir.exists():
            return []
        dates = []
        for f in sorted(self.log_dir.glob("audit_*.jsonl")):
            dates.append(f.stem.replace("audit_", ""))
        return dates

    def clear_memory(self) -> None:
        """Clear in-memory entries (does not affect persisted logs)."""
        self._entries.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _today_log_path(self) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{today}.jsonl"
