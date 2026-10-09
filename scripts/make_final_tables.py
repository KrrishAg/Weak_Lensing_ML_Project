#!/usr/bin/env python
"""Collect measured experiment outputs into report-ready Markdown and CSV tables.

This script never invents missing values. Rows are emitted only for result files that exist.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

RUNS = [
    ("baseline_realization", "Baseline (realization split)"),
    ("proposed_realization", "Proposed M=3 (realization split)"),
    ("proposed_heldout_cosmology", "Proposed M=3 (held-out cosmologies)"),
]


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def fmt(x, digits=5):
    if x is None:
        return "-"
    return f"{float(x):.{digits}f}"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    inference_rows = []
    for tag, label in RUNS:
        p = RESULTS / f"{tag}_metrics.json"
        if not p.exists():
            continue
        pack = load_json(p)
        raw = pack["test_raw"]
        cal = pack.get("test_calibrated") or raw
        nuisance_on = float(pack.get("nuisance_weight", 0.0)) > 0
        inference_rows.append({
            "run": label,
            "tag": tag,
            "ensemble_size": int(pack["ensemble_size"]),
            "nuisance_weight": float(pack["nuisance_weight"]),
            "mae_omega_m": cal["mae_omega_m"],
            "mae_s8": cal["mae_s8"],
            "rmse_omega_m": cal["rmse_omega_m"],
            "rmse_s8": cal["rmse_s8"],
            "raw_coverage68": raw["coverage68_mean"],
            "cal_coverage68": cal["coverage68_mean"],
            "phase1_score": cal.get("phase1_score"),
            "mae_tagn": cal.get("mae_tagn") if nuisance_on else None,
            "mae_f0": cal.get("mae_f0") if nuisance_on else None,
            "mae_delta_z": cal.get("mae_delta_z") if nuisance_on else None,
        })

    md = ["# Measured final tables", "", "Only completed runs are shown; missing experiments are not imputed.", ""]
    if inference_rows:
        md += [
            "## Cosmological inference and calibration",
            "",
            "| Run | M | nuisance wt | MAE Om | MAE S8 | RMSE Om | RMSE S8 | raw cov68 | calibrated cov68 | Phase-1 ref. score | TAGN MAE | f0 MAE | dz MAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in inference_rows:
            md.append(
                f"| {r['run']} | {r['ensemble_size']} | {r['nuisance_weight']:.3f} | "
                f"{fmt(r['mae_omega_m'])} | {fmt(r['mae_s8'])} | {fmt(r['rmse_omega_m'])} | "
                f"{fmt(r['rmse_s8'])} | {fmt(r['raw_coverage68'],3)} | {fmt(r['cal_coverage68'],3)} | "
                f"{fmt(r['phase1_score'],4)} | {fmt(r['mae_tagn'],4)} | {fmt(r['mae_f0'],5)} | {fmt(r['mae_delta_z'],5)} |"
            )
        md.append("")

        csv_path = RESULTS / "final_inference_table.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(inference_rows[0]))
            w.writeheader(); w.writerows(inference_rows)

    ood_path = RESULTS / "proposed_realization_ood_metrics.json"
    if ood_path.exists():
        d = load_json(ood_path)
        methods = ["ensemble_disagreement", "nuisance_prior", "power_spectrum", "hybrid"]
        md += [
            "## Offline controlled-shift OoD evaluation",
            "",
            "These are diagnostic synthetic shifts, not the hidden competition OoD distribution.",
            "",
            "| Detector | " + " | ".join(d.keys()) + " | Mean low-FPR TPR |",
            "|---|" + "---:|" * (len(d) + 1),
        ]
        ood_rows = []
        for method in methods:
            vals = [float(d[shift][method]["low_fpr_tpr"]) for shift in d]
            md.append("| " + method + " | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {sum(vals)/len(vals):.3f} |")
            row = {"detector": method, **{shift: v for shift, v in zip(d, vals)}, "mean_low_fpr_tpr": sum(vals)/len(vals)}
            ood_rows.append(row)
        md.append("")
        with (RESULTS / "final_ood_table.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ood_rows[0]))
            w.writeheader(); w.writerows(ood_rows)

    if len(md) == 4:
        md.append("No measured result files were found yet. Run the training/evaluation scripts first.")

    out = RESULTS / "final_tables.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    print("wrote", out)


if __name__ == "__main__":
    main()
