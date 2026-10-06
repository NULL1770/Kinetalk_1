"""Build the three paper tables from protocol-matched final reports.

Table 1 is mesh-space (LVE/EVE/FDD), Table 2 is ARKit52 coefficient-space
(MBE/LBE), and Table 3 is KineTalk ablation. Every input report must carry
``test_loaded=true`` when ``--require-test`` is used. Missing metrics remain
pending and are never replaced by zeros or development values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


TABLES = {
    # Unified blendshape-space table.  This is the common output space before
    # applying the fixed ARKit-to-mesh rig used by the vertex table.
    "table0_bs": ("method", "MBE", "LBE", "FDD_abs"),
    # FDD is an energy-std difference and therefore has squared distance
    # units.  Keep the unit in the column name so a raw coefficient FDD or a
    # paper's dimensionless coefficient diagnostic cannot be mixed in.
    "table1_vertex": ("method", "LVE_mm", "EVE_mm", "FDD_mm2"),
    "table2_arkit": ("method", "MBE", "LBE"),
    "table3_ablation": (
        "variant", "MBE", "LBE", "LVE_mm", "EVE_mm", "FDD_mm2", "emotion_macro_f1"
    ),
}

EXPECTED_ROWS = {
    "table0_bs": {"VOCA", "FaceFormer", "EmoTalk", "FaceDiffuser", "Ours"},
    "table1_vertex": {"VOCA", "FaceFormer", "EmoTalk", "FaceDiffuser", "Ours"},
    "table2_arkit": {"EmoTalk", "FaceDiffuser", "Ours"},
    "table3_ablation": {
        "Full", "w/o residual factorization", "w/o mouth protection/calibration",
        "w/o emotion teacher-student",
    },
}

ADAPTED_LABELS = {
    'VOCA': 'VOCA-core (ARKit, shared audio)',
    'EmoTalk': 'EmoTalk-core (ARKit, shared audio)',
    'FaceFormer': 'FaceFormer (ARKit, shared audio)',
    'FaceDiffuser': 'FaceDiffuser (ARKit, frozen KineTalk conditions)',
}


def _read(path: Path) -> Mapping[str, Any]:
    report = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(report, Mapping):
        raise ValueError(f"report must be a JSON object: {path}")
    return report


def _num(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("value", "mean", "score", "metric"):
            if key in value:
                return _num(value[key])
        return None
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _lookup(obj: Mapping[str, Any], *names: str) -> float | None:
    # Reports in this repository use both flat summaries and nested
    # ``summary -> benchmark -> coefficient`` layouts.  Walk only a shallow,
    # deterministic tree so a per-clip diagnostic cannot silently become the
    # aggregate row.
    pending = [(obj, 0)]
    while pending:
        container, depth = pending.pop(0)
        if not isinstance(container, Mapping):
            continue
        for name in names:
            if name in container:
                value = _num(container[name])
                if value is not None:
                    return value
        if depth < 3:
            for key in ("metrics", "summary", "benchmark", "coefficient", "vertex", "arkit"):
                child = container.get(key)
                if isinstance(child, Mapping):
                    pending.append((child, depth + 1))
    return None


def _require_scope(path: Path, report: Mapping[str, Any], require_test: bool) -> None:
    if require_test and report.get("test_loaded") is not True:
        raise ValueError(f"publishable table input lacks test_loaded=true: {path}")
    if report.get("status") in ("failed", "pending"):
        raise ValueError(f"failed/pending report cannot enter final table: {path}")
    if require_test and report.get("method") in ("voca", "emotalk"):
        adaptation = report.get("baseline_adaptation", {})
        if adaptation.get("version") != "core_arkit_v1":
            raise ValueError(f"Legacy simplified baseline is internal only: {path}")


def _row_vertex(method: str, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": method,
        # EmoTalk-style vertex metrics are max-over-region per frame, then
        # averaged over time.  The evaluator emits both this diagnostic and a
        # mean-region alternative; keep the max variant first so Table 1 is
        # unambiguous and consistent with the requested protocol.
        "LVE_mm": _lookup(report, "LVE_mm", "vertex_lve_sqrt_mm", "vertex_lve_sqrt_mean",
                           "vertex_lve_mm", "lve_mm"),
        "EVE_mm": _lookup(report, "EVE_mm", "eye_forehead_eve_sqrt_mm", "eye_forehead_eve_sqrt_mean",
                           "vertex_eve_mm", "eye_forehead_eve_mm", "eve_mm"),
        "FDD_mm2": _lookup(
            report,
            "FDD_mm2",
            "vertex_fdd_absolute_mm2_mean",
            "vertex_fdd_absolute_mm2",
            "vertex_fdd_mm2",
            "fdd_mm2",
        ),
    }


def _row_bs(method: str, report: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the shared 52-D blendshape-space metrics.

    The aliases cover both the FaceDiffuser-style nested benchmark reports and
    the flat paper reports emitted by the final-test evaluator.  No vertex
    metric is inferred from BS values here; Table 1 remains a separate
    fixed-rig conversion table.
    """
    return {
        "method": method,
        "MBE": _lookup(report, "MBE", "mbe", "arkit_mbe"),
        "LBE": _lookup(report, "LBE", "lbe", "arkit_lbe"),
        "FDD_abs": _lookup(report, "FDD_abs", "FDD_absolute", "arkit_fdd_absolute",
                           "supp_upper9_fdd_absolute"),
    }


def _row_arkit(method: str, report: Mapping[str, Any]) -> dict[str, Any]:
    return {"method": method, "MBE": _lookup(report, "MBE", "mbe", "arkit_mbe"),
            "LBE": _lookup(report, "LBE", "lbe", "arkit_lbe")}


def _row_ablation(variant: str, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "variant": variant,
        "MBE": _lookup(report, "MBE", "mbe", "arkit_mbe"),
        "LBE": _lookup(report, "LBE", "lbe", "arkit_lbe"),
        "LVE_mm": _lookup(report, "LVE_mm", "vertex_lve_sqrt_mm", "vertex_lve_sqrt_mean",
                           "vertex_lve_mm", "lve_mm"),
        "EVE_mm": _lookup(report, "EVE_mm", "eye_forehead_eve_sqrt_mm", "eye_forehead_eve_sqrt_mean",
                           "vertex_eve_mm", "eye_forehead_eve_mm", "eve_mm"),
        "FDD_mm2": _lookup(
            report,
            "FDD_mm2",
            "vertex_fdd_absolute_mm2_mean",
            "vertex_fdd_absolute_mm2",
            "vertex_fdd_mm2",
            "fdd_mm2",
        ),
        "emotion_macro_f1": _lookup(report, "emotion_macro_f1", "macro_f1", "emotion_f1", "f1"),
    }


def _pairs(values: list[str], option: str) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--{option} expects LABEL=REPORT.json")
        label, raw = value.split("=", 1)
        if not label or not raw:
            raise ValueError(f"--{option} has an empty label/path")
        result.append((label, Path(raw)))
    return result


def _check_complete(rows: list[dict[str, Any]], columns: tuple[str, ...], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} has no rows")
    missing = [(row.get(columns[0]), col) for row in rows for col in columns[1:] if row.get(col) is None]
    if missing:
        preview = ", ".join(f"{label}:{col}" for label, col in missing[:8])
        raise ValueError(f"{name} contains missing metrics ({preview}); final tables cannot be emitted")


def _check_labels(rows: list[dict[str, Any]], name: str) -> None:
    labels = {row[next(iter(TABLES[name]))] for row in rows}
    expected = EXPECTED_ROWS[name]
    missing = sorted(expected - labels)
    if missing:
        raise ValueError(f"{name} is missing required rows: {', '.join(missing)}")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows({col: row.get(col) for col in columns} for row in rows)


def _fmt(value: Any) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def _latex(caption: str, label: str, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    body = [" & ".join(str(row[c]) if c == columns[0] else _fmt(row[c]) for c in columns) + r" \\" for row in rows]
    return "\n".join([
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}", r"\toprule",
        " & ".join(columns) + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])


def build(vertex, arkit, ablation, output: Path, *, require_test: bool, tsne: Path | None = None, bs=None):
    output.mkdir(parents=True, exist_ok=True)
    provenance, tables = [], {}
    rows = {"table0_bs": [], "table1_vertex": [], "table2_arkit": [], "table3_ablation": []}
    specs = (("table0_bs", bs if bs is not None else arkit, _row_bs), ("table1_vertex", vertex, _row_vertex), ("table2_arkit", arkit, _row_arkit),
             ("table3_ablation", ablation, _row_ablation))
    for name, pairs, mapper in specs:
        for label, path in pairs:
            report = _read(path)
            _require_scope(path, report, require_test)
            rows[name].append(mapper(label, report))
            provenance.append({"table": name, "label": label, "path": str(path.resolve()),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "test_loaded": report.get("test_loaded"),
                               "baseline_adaptation": report.get("baseline_adaptation")})
        _check_complete(rows[name], TABLES[name], name)
        _check_labels(rows[name], name)
        for row in rows[name]:
            if row.get('method') in ADAPTED_LABELS:
                row['method'] = ADAPTED_LABELS[row['method']]
        tables[name] = {"columns": list(TABLES[name]), "rows": rows[name]}
        _write_csv(output / f"{name}.csv", rows[name], TABLES[name])

    tex = {
        "table0_bs": _latex("Unified 52-D blendshape-space errors on the sealed MEAD test split.", "tab:bs_main", rows["table0_bs"], TABLES["table0_bs"]),
        "table1_vertex": _latex("Mesh-space errors after the fixed ARKit-to-mesh rig; LVE/EVE in mm and FDD in mm$^2$.", "tab:vertex_main", rows["table1_vertex"], TABLES["table1_vertex"]),
        "table2_arkit": _latex("ARKit52 coefficient errors on the sealed MEAD test split.", "tab:arkit_main", rows["table2_arkit"], TABLES["table2_arkit"]),
        "table3_ablation": _latex("KineTalk ablations under the same sealed-test protocol.", "tab:ablation", rows["table3_ablation"], TABLES["table3_ablation"]),
    }
    for name, content in tex.items():
        (output / f"{name}.tex").write_text(content + "\n", encoding="utf8")
    result = {"schema": "kinetalk_three_paper_tables_v1", "scope": "sealed_test" if require_test else "source_report_scope",
              "test_loaded": bool(require_test), "tables": tables, "provenance": provenance,
              "tsne": str(tsne.resolve()) if tsne else None,
              "notes": ["No development, smoke, oracle, or predicted values are accepted.",
                        "Baselines are declared MEAD/ARKit adaptations, not official end-to-end reproductions; see baseline_adaptation_20260923.md.",
                        "All methods must share rig, masks, frame rate, split, and evaluator version."]}
    (output / "paper_tables.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertex-report", action="append", default=[])
    parser.add_argument("--bs-report", action="append", default=[],
                        help="LABEL=REPORT.json for the unified 52-D blendshape table")
    parser.add_argument("--arkit-report", action="append", default=[])
    parser.add_argument("--ablation-report", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsne", type=Path)
    parser.add_argument("--require-test", action="store_true")
    args = parser.parse_args()
    result = build(_pairs(args.vertex_report, "vertex-report"), _pairs(args.arkit_report, "arkit-report"),
                   _pairs(args.ablation_report, "ablation-report"), args.output,
                   bs=_pairs(args.bs_report, "bs-report") if args.bs_report else None,
                   require_test=args.require_test, tsne=args.tsne)
    print(json.dumps({"output": str(args.output.resolve()), "tables": {k: len(v["rows"]) for k, v in result["tables"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
