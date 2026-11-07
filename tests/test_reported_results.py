"""REPORT.md, the committed output of the full run, against the paper's claim."""

from __future__ import annotations

import re

import pytest


def row(report: str, name: str) -> list[float]:
    match = re.search(rf"\| \*\*{re.escape(name)}\*\* \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", report)
    assert match, f"{name} row missing from REPORT.md"
    return [float(g) for g in match.groups()]


def test_sqa_improves_on_the_baseline(report):
    """Abstract: SQA significantly improves performance. DEVIATIONS D6 expects no gain in-domain."""
    assert row(report, "+ SQA")[0] > row(report, "Baseline")[0]


def test_published_values(report, targets):
    if targets["published_values"] is None:
        pytest.skip("paper tables not recorded in tests/paper_targets.yaml")
    expected = targets["published_values"]
    assert row(report, "+ SQA")[:2] == pytest.approx([expected["map"], expected["rank1"]], abs=1.0)
