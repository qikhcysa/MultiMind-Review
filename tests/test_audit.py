"""Tests for the audit trail module."""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.models import AuditEntry
from src.audit import AuditTrail


def make_entry(review_id: str = "test_review", stage: str = "product_recognition") -> AuditEntry:
    return AuditEntry(
        entry_id=str(uuid.uuid4()),
        review_id=review_id,
        stage=stage,
        agent_name="TestAgent",
        input_data={"text": "sample review"},
        output_data={"result": "ok"},
        retrieved_neighbors=[
            {"id": "prod_001", "document": "some product", "similarity": 0.85}
        ],
        similarities=[0.85],
        reasoning="Test reasoning",
    )


@pytest.fixture
def audit(tmp_path):
    return AuditTrail(log_dir=str(tmp_path / "audit"), enabled=True)


def test_record_and_retrieve(audit):
    """Recorded entries should be retrievable from memory."""
    rid = str(uuid.uuid4())
    entry = make_entry(review_id=rid)
    audit.record(entry)

    all_entries = audit.get_all()
    assert any(e.entry_id == entry.entry_id for e in all_entries)


def test_get_by_review(audit):
    """Filtering by review_id should return only matching entries."""
    rid = str(uuid.uuid4())
    other_rid = str(uuid.uuid4())
    audit.record(make_entry(review_id=rid, stage="product_recognition"))
    audit.record(make_entry(review_id=rid, stage="dimension_detection"))
    audit.record(make_entry(review_id=other_rid, stage="product_recognition"))

    entries = audit.get_by_review(rid)
    assert len(entries) == 2
    assert all(e.review_id == rid for e in entries)


def test_get_by_stage(audit):
    """Filtering by stage should return only matching entries."""
    audit.record(make_entry(stage="product_recognition"))
    audit.record(make_entry(stage="product_recognition"))
    audit.record(make_entry(stage="dimension_detection"))

    pr_entries = audit.get_by_stage("product_recognition")
    assert len(pr_entries) >= 2
    assert all(e.stage == "product_recognition" for e in pr_entries)


def test_persist_and_reload(tmp_path):
    """Entries should be persisted to disk and reloadable."""
    log_dir = str(tmp_path / "audit_reload")
    audit = AuditTrail(log_dir=log_dir, enabled=True)

    rid = str(uuid.uuid4())
    entry = make_entry(review_id=rid)
    audit.record(entry)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    audit2 = AuditTrail(log_dir=log_dir, enabled=True)
    loaded = audit2.load_from_date(today)
    assert any(e.entry_id == entry.entry_id for e in loaded)


def test_to_dataframe(audit):
    """to_dataframe should return a DataFrame with expected columns."""
    audit.record(make_entry())
    df = audit.to_dataframe()
    assert not df.empty
    assert "review_id" in df.columns
    assert "stage" in df.columns
    assert "top_similarity" in df.columns


def test_clear_memory(audit):
    """clear_memory should empty in-memory entries."""
    audit.record(make_entry())
    audit.record(make_entry())
    assert len(audit.get_all()) >= 2
    audit.clear_memory()
    assert audit.get_all() == []


def test_available_dates(tmp_path):
    """available_dates should list dates of existing log files."""
    log_dir = str(tmp_path / "audit_dates")
    audit = AuditTrail(log_dir=log_dir, enabled=True)
    audit.record(make_entry())
    dates = audit.available_dates()
    assert len(dates) >= 1
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert today in dates
