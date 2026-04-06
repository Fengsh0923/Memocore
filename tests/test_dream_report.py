"""Tests for DreamReport from memocore.core.dream"""

from datetime import datetime, timezone, timedelta

from memocore.core.dream import DreamReport


class TestDreamReport:
    def test_defaults(self):
        r = DreamReport(agent_id="test")
        assert r.status == "running"
        assert r.total_nodes == 0
        assert r.merged == 0
        assert r.compiled_pages == 0
        assert r.lint_contradictions == 0
        assert r.error is None

    def test_summary_running(self):
        r = DreamReport(agent_id="test")
        s = r.summary()
        assert "test" in s
        assert "running" in s

    def test_summary_done_with_elapsed(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(seconds=12.5)
        r = DreamReport(
            agent_id="demo",
            started_at=start,
            finished_at=end,
            status="done",
            total_nodes=100,
            total_edges=50,
            merged=3,
            compiled_pages=10,
        )
        s = r.summary()
        assert "done" in s
        assert "12.5s" in s
        assert "100" in s
        assert "compiled=10" in s

    def test_summary_failed(self):
        r = DreamReport(agent_id="x", status="failed", error="boom")
        s = r.summary()
        assert "failed" in s
