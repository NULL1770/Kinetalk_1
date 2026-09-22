import json
from pathlib import Path

import pytest

from scripts.aggregate_paper_tables import build


def test_build_extracts_known_metric_layouts_and_keeps_missing_pending(tmp_path: Path):
    report = {
        "schema": "arkit_benchmark_report_v1",
        "test_loaded": True,
        "rows": [
            {"mode": "full", "arkit_mbe": 0.2, "arkit_lbe": 0.3,
             "arkit_fdd_absolute": 0.4, "supp_upper9_fdd_absolute": 0.5},
        ],
    }
    source = tmp_path / "report.json"
    source.write_text(json.dumps(report), encoding="utf8")
    result = build({"ours": source}, tmp_path / "out", require_test=True)
    main = result["tables"]["main"]["rows"]
    assert len(main) == 1
    assert main[0]["MBE"] == pytest.approx(0.2)
    assert main[0]["LBE"] == pytest.approx(0.3)
    assert main[0]["FDD_abs"] == pytest.approx(0.4)
    assert main[0]["dynamic_correlation"] is None
    assert main[0]["status"] == "computed"
    assert (tmp_path / "out" / "table_main.csv").is_file()


def test_summaries_layout_and_regional_correlation(tmp_path: Path):
    report = {
        "schema": "mixed",
        "test_loaded": True,
        "summaries": {
            "audio": {
                "arkit_mbe": {"mean": 1.0},
                "upper_intensity_mae": {"mean": 2.0},
            },
        },
        "aggregate": {
            "brow": {"pearson": 0.4},
            "eye": {"pearson": 0.6},
        },
    }
    source = tmp_path / "report.json"
    source.write_text(json.dumps(report), encoding="utf8")
    result = build({"audio": source}, tmp_path / "out", require_test=True)
    row = result["tables"]["main"]["rows"][0]
    assert row["MBE"] == pytest.approx(1.0)
    assert row["Upper9_intensity_error"] == pytest.approx(2.0)
    # The summary row itself has no aggregate, so correlation is explicitly missing.
    assert row["dynamic_correlation"] is None


def test_regional_report_populates_dynamic_correlation_and_delay(tmp_path: Path):
    report = {
        "schema": "regional_receiver_metrics_v1",
        "test_loaded": True,
        "aggregate": {
            "brow": {
                "pearson": 0.4,
                "dominant_peak_abs_delay_frames": 2.0,
            },
            "eye": {
                "pearson": 0.6,
                "dominant_peak_abs_delay_frames": 4.0,
            },
        },
    }
    source = tmp_path / "regional.json"
    source.write_text(json.dumps(report), encoding="utf8")
    result = build({"full_audio": source}, tmp_path / "out", require_test=True)
    row = result["tables"]["dynamic"]["rows"][0]
    assert row["temporal_correlation"] == pytest.approx(0.5)
    assert row["peak_delay_frames"] == pytest.approx(3.0)


def test_modes_report_expands_conditions_and_reads_nested_metrics(tmp_path: Path):
    report = {
        "schema": "dynamics",
        "test_loaded": True,
        "modes": {
            "42/full": {"metrics": {
                "brows": {"centered_correlation": 0.2},
                "eyes_expression": {"centered_correlation": 0.4},
            }},
            "42/reverse_audio": {"metrics": {
                "brows": {"centered_correlation": -0.1},
                "eyes_expression": {"centered_correlation": 0.1},
            }},
        },
    }
    source = tmp_path / "modes.json"
    source.write_text(json.dumps(report), encoding="utf8")
    result = build({"dynamic": source}, tmp_path / "out", require_test=True)
    rows = result["tables"]["dynamic"]["rows"]
    assert [row["condition"] for row in rows] == ["42/full", "42/reverse_audio"]
    assert rows[0]["temporal_correlation"] == pytest.approx(0.3)
    assert rows[1]["temporal_correlation"] == pytest.approx(0.0)


def test_require_test_rejects_development_report(tmp_path: Path):
    source = tmp_path / "dev.json"
    source.write_text(json.dumps({"schema": "dev", "test_loaded": False}), encoding="utf8")
    with pytest.raises(ValueError, match="test_loaded=true"):
        build({"dev": source}, tmp_path / "out", require_test=True)


def test_explicit_table_labels_route_sources_without_cross_table_duplication(tmp_path: Path):
    main = tmp_path / "main.json"
    dynamic = tmp_path / "dynamic.json"
    main.write_text(json.dumps({"schema": "main", "test_loaded": True,
                                "rows": [{"mode": "ours", "arkit_mbe": 1.}]}), encoding="utf8")
    dynamic.write_text(json.dumps({"schema": "dynamic", "test_loaded": True,
                                   "rows": [{"mode": "full", "pearson": .5}]}), encoding="utf8")
    result = build({"main": main, "dynamic": dynamic}, tmp_path / "out", require_test=True)
    assert len(result["tables"]["main"]["rows"]) == 1
    assert len(result["tables"]["dynamic"]["rows"]) == 1
    assert not result["tables"]["emotion"]["rows"]
    assert not result["tables"]["ablation"]["rows"]
