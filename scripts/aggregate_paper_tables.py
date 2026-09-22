"""Build the four paper tables from completed evaluator reports.

This module is deliberately an *aggregation* layer.  It never computes a
metric from a target array and it never fills a missing value with zero.  A
report can therefore be replaced by a final test report without changing the
table protocol.  Inputs are supplied as repeated ``--input ROLE=PATH``
arguments; ``ROLE`` is only a human-readable label and does not change the
numbers in the source report.

The supported source layouts are the reports emitted by
``arkit_benchmark_report``, ``evaluate_paper_coefficients``,
``regional_receiver_metrics`` and the emotion/dynamic audit scripts.  Unknown
fields remain empty in CSV and are recorded as ``pending`` in JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


TABLE_COLUMNS = {
    "main": (
        "method", "MBE", "LBE", "FDD_abs", "Upper9_intensity_error",
        "dynamic_correlation",
    ),
    "ablation": (
        "variant", "residual", "mouth_calibration", "emotion_teacher_student",
        "dynamic_receiver", "MBE", "LBE", "Upper9_intensity_error",
    ),
    "emotion": (
        "method", "emotion_input", "accuracy", "macro_f1",
        "upper_face_intensity_error", "tsne",
    ),
    "dynamic": (
        "condition", "Upper9_intensity_error", "temporal_correlation",
        "peak_delay_frames", "std_gap", "LBE",
    ),
}


def _finite(value: Any) -> float | None:
    """Convert a scalar metric to JSON-safe float, preserving missing values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _scalar(value: Any) -> float | str | None:
    """Unwrap common evaluator scalar records.

    ``evaluate_paper_coefficients`` stores ``{"mean": ...}``, while the
    coefficient evaluator stores ``{"value": ...}``.  Strings (for example a
    t-SNE image path) are retained; arbitrary objects are not.
    """
    if isinstance(value, Mapping):
        for key in ("value", "mean", "score", "metric"):
            if key in value:
                result = _scalar(value[key])
                if result is not None:
                    return result
        return None
    number = _finite(value)
    if number is not None:
        return number
    return value if isinstance(value, str) else None


def _walk(obj: Any, key: str) -> Iterable[Any]:
    """Yield values for an exact key, recursively, without fuzzy matching."""
    if isinstance(obj, Mapping):
        for name, value in obj.items():
            if name == key:
                yield value
            yield from _walk(value, key)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk(value, key)


def _first(obj: Any, *keys: str) -> float | str | None:
    for key in keys:
        for value in _walk(obj, key):
            value = _scalar(value)
            if value is not None:
                return value
    return None


def _records(report: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    """Normalize known evaluator report layouts into one row per method/arm."""
    rows = report.get("rows")
    if isinstance(rows, list):
        return [{"method": label, **row} for row in rows if isinstance(row, Mapping)]

    summaries = report.get("summaries")
    if isinstance(summaries, Mapping):
        result = []
        for name, row in summaries.items():
            if isinstance(row, Mapping):
                result.append({"method": str(name), **row})
        if result:
            return result

    # A regional receiver report has no method rows.  Keep it as one record;
    # _first() below can still read aggregate/brow/eye values.
    return [{"method": label, **report}]


def _regional_correlation(row: Mapping[str, Any]) -> float | None:
    direct = _first(row, "dynamic_correlation", "pooled_centered_correlation")
    if isinstance(direct, (float, int)):
        return float(direct)
    aggregate = row.get("aggregate")
    if isinstance(aggregate, Mapping):
        values = []
        for region in ("brow", "eye"):
            value = aggregate.get(region)
            if isinstance(value, Mapping):
                value = _scalar(value.get("pearson", value.get("xcorr_best_corr")))
                if isinstance(value, (float, int)):
                    values.append(float(value))
        if values:
            return sum(values) / len(values)
    return None


def _regional_mean(row: Mapping[str, Any], *keys: str) -> float | None:
    """Average a scalar over brow/eye aggregate regions when available."""
    aggregate = row.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return None
    values = []
    for region in ("brow", "eye"):
        value = aggregate.get(region)
        if not isinstance(value, Mapping):
            continue
        value = _first(value, *keys)
        if isinstance(value, (float, int)):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _main_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": str(row.get("method") or row.get("arm") or row.get("mode") or "unknown"),
        "MBE": _first(row, "MBE", "arkit_mbe"),
        "LBE": _first(row, "LBE", "arkit_lbe", "supp_lip23_lbe"),
        "FDD_abs": _first(row, "FDD_abs", "arkit_fdd_absolute", "supp_upper9_fdd_absolute"),
        "Upper9_intensity_error": _first(
            row, "Upper9_intensity_error", "upper_intensity_mae", "upper_face_intensity_error",
        ),
        "dynamic_correlation": _regional_correlation(row),
    }


def _ablation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "variant": str(row.get("variant") or row.get("method") or row.get("arm") or row.get("mode") or "unknown"),
        "residual": row.get("residual", row.get("residual_branch")),
        "mouth_calibration": row.get("mouth_calibration", row.get("static_mouth_calibration")),
        "emotion_teacher_student": row.get("emotion_teacher_student", row.get("emotion_teacher")),
        "dynamic_receiver": row.get("dynamic_receiver", row.get("bounded_center")),
        "MBE": _first(row, "MBE", "arkit_mbe"),
        "LBE": _first(row, "LBE", "arkit_lbe", "supp_lip23_lbe"),
        "Upper9_intensity_error": _first(
            row, "Upper9_intensity_error", "upper_intensity_mae", "upper_face_intensity_error",
        ),
    }


def _emotion_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": str(row.get("method") or row.get("arm") or row.get("mode") or "unknown"),
        "emotion_input": row.get("emotion_input", row.get("input", row.get("condition"))),
        "accuracy": _first(row, "accuracy", "emotion_accuracy", "emotion_readout_accuracy"),
        "macro_f1": _first(row, "macro_f1", "f1", "emotion_macro_f1"),
        "upper_face_intensity_error": _first(
            row, "upper_face_intensity_error", "upper_intensity_mae", "envelope_mse",
        ),
        "tsne": _first(row, "tsne", "plot", "tsne_path"),
    }


def _dynamic_row(row: Mapping[str, Any]) -> dict[str, Any]:
    correlation = _regional_correlation(row)
    delay = _regional_mean(row, "dominant_peak_abs_delay_frames", "xcorr_best_lag_frames")
    return {
        "condition": str(row.get("condition", row.get("method", row.get("arm", row.get("mode", "unknown"))))),
        "Upper9_intensity_error": _first(
            row, "Upper9_intensity_error", "upper_intensity_mae", "upper_face_intensity_error",
            "envelope_mse",
        ),
        "temporal_correlation": correlation if correlation is not None else _first(
            row, "temporal_correlation", "dynamic_correlation", "pooled_centered_correlation",
            "xcorr_best_corr", "pearson",
        ),
        "peak_delay_frames": delay if delay is not None else _first(
            row, "peak_delay_frames", "dominant_peak_abs_delay_frames", "xcorr_best_lag_frames",
        ),
        "std_gap": _first(row, "std_gap", "upper_std_absolute_gap", "std_absolute_gap"),
        "LBE": _first(row, "LBE", "arkit_lbe", "supp_lip23_lbe"),
    }


def _rows_for_table(table: str, reports: list[tuple[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    mapper = {"main": _main_row, "ablation": _ablation_row,
              "emotion": _emotion_row, "dynamic": _dynamic_row}[table]
    result = []
    for label, report in reports:
        for row in _records(report, label):
            normalized = mapper(row)
            normalized["source"] = label
            normalized["status"] = "computed" if any(
                _finite(v) is not None for key, v in normalized.items()
                if key not in ("source", "status")
            ) else "pending"
            result.append(normalized)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*columns, "source", "status"])
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in rows)


def build(inputs: Mapping[str, Path], output: Path, *, require_test: bool = False) -> dict[str, Any]:
    """Read reports and write JSON/CSV files for all four paper tables."""
    loaded: list[tuple[str, Mapping[str, Any]]] = []
    provenance = []
    for label, path in inputs.items():
        path = Path(path)
        report = json.loads(path.read_text(encoding="utf8"))
        if not isinstance(report, Mapping):
            raise ValueError(f"Report must be a JSON object: {path}")
        test_loaded = report.get("test_loaded")
        if require_test and test_loaded is not True:
            raise ValueError(f"Final table requires test_loaded=true: {path}")
        loaded.append((label, report))
        provenance.append({"label": label, "path": str(path.resolve()),
                           "schema": report.get("schema"), "test_loaded": test_loaded})

    output.mkdir(parents=True, exist_ok=True)
    tables = {}
    for table, columns in TABLE_COLUMNS.items():
        rows = _rows_for_table(table, loaded)
        tables[table] = {"columns": list(columns), "rows": rows,
                         "scope": "final_test" if require_test else "source_report_scope"}
        _write_csv(output / f"table_{table}.csv", rows, columns)

    result = {"schema": "paper_four_tables_v1", "scope": "final_test" if require_test else "source_report_scope",
              "provenance": provenance, "tables": tables,
              "missing_values": "null/pending; no metric is replaced with zero"}
    (output / "paper_tables.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8"
    )
    return result


def _parse_input(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--input must be LABEL=REPORT.json")
        label, path = value.split("=", 1)
        if not label or not path or label in result:
            raise ValueError("--input labels must be nonempty and unique")
        result[label] = Path(path)
    if not result:
        raise ValueError("At least one --input LABEL=REPORT.json is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[], metavar="LABEL=REPORT.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-test", action="store_true",
                        help="Reject reports that do not explicitly set test_loaded=true")
    args = parser.parse_args()
    result = build(_parse_input(args.input), args.output, require_test=args.require_test)
    print(json.dumps({"output": str(args.output.resolve()),
                      "scope": result["scope"],
                      "tables": {k: len(v["rows"]) for k, v in result["tables"].items()}},
               ensure_ascii=False))


if __name__ == "__main__":
    main()
