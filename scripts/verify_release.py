#!/usr/bin/env python3
"""Verify checksums and the numerical claims in the compact public release."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PRIMARY = ARTIFACTS / "dml_weighted_residual_gamma_selection_ablation_20260903"
PENALTY = ARTIFACTS / "dml_section4_same_sample_selection_penalty_20260909_v3"
BOUNDS = ARTIFACTS / "dml_section3_bounds_diagnostic_20260908_theorem_fit_public_v1"
TMLE_BOUNDS = ARTIFACTS / "dml_section3_bounds_diagnostic_fixed_floor_tmle_20260908_theorem_fit_public_v1"
EXPECTED_PRIMARY = {
    "aipw": 8.95,
    "selective_ml": 5.19,
    "ma_dr_bc": 7.66,
    "c_tmle": 4.98,
}
FORBIDDEN_MARKERS = (
    "manifold" + "://",
    "/data/" + "users/",
    "/" + "Users/",
    "/home/" + "atavory",
    "USH" + "MOO_",
    "fb" + "source",
    "over" + "leaf",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_checksums(bundle: Path) -> int:
    checksum_file = bundle / "SHA256SUMS"
    if not checksum_file.is_file():
        raise SystemExit(f"missing checksum manifest: {checksum_file}")
    checked = 0
    for line in checksum_file.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = bundle / name
        if not path.is_file():
            raise SystemExit(f"missing checksum target: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"checksum mismatch: {path}")
        checked += 1
    return checked


def verify_primary() -> None:
    rows = read_csv(PRIMARY / "weighted_residual_gamma_primary_table.csv")
    if len(rows) != 24:
        raise SystemExit(f"expected 24 primary settings, found {len(rows)}")
    for stem, expected in EXPECTED_PRIMARY.items():
        observed = sum(float(row[f"{stem}_gain_pct"]) for row in rows) / len(rows)
        if not math.isclose(observed, expected, abs_tol=0.005):
            raise SystemExit(
                f"unexpected equal-setting gain for {stem}: {observed:.6f}"
            )


def verify_penalty() -> None:
    rows = {
        row["scope"]: row
        for row in read_csv(PENALTY / "same_sample_penalty_equal_setting_summary.csv")
    }
    expected = {
        "primary": (6.693018611041604, 4.898684923532671, 1.794333687508933),
        "fixed_floor_tmle": (-7.669510135132373, 3.7411372461262995, -11.410647381258672),
    }
    fields = (
        "same_sample_mse_gain_pct",
        "fresh_risk_gain_pct",
        "gain_gap_same_minus_fresh_pct",
    )
    for scope, values in expected.items():
        for field, value in zip(fields, values):
            observed = float(rows[scope][field])
            if not math.isclose(observed, value, rel_tol=0.0, abs_tol=1e-12):
                raise SystemExit(f"unexpected {scope} {field}: {observed}")


def verify_bounds() -> None:
    for bundle, expected_rows in ((BOUNDS, 9216), (TMLE_BOUNDS, 2304)):
        rows = read_csv(bundle / "section3_bounds_overall_summary.csv")
        if len(rows) != 1 or int(rows[0]["rows"]) != expected_rows:
            raise SystemExit(f"unexpected bound-audit coverage: {bundle}")
        if int(rows[0]["variance_violations"]) or int(rows[0]["bias_violations"]):
            raise SystemExit(f"bound violation recorded in {bundle}")


def verify_no_internal_identifiers() -> None:
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or path.resolve() == Path(__file__).resolve()
        ):
            continue
        if path.suffix.lower() not in {".json", ".md", ".py", ".txt", ".tex", ".csv"}:
            continue
        text = path.read_text(errors="replace").lower()
        for marker in FORBIDDEN_MARKERS:
            if marker.lower() in text:
                raise SystemExit(f"internal identifier {marker!r} in {path}")


def main() -> None:
    bundles = sorted(path for path in ARTIFACTS.iterdir() if path.is_dir())
    checksum_count = sum(verify_checksums(bundle) for bundle in bundles)
    verify_primary()
    verify_penalty()
    verify_bounds()
    verify_no_internal_identifiers()
    print(
        f"VERIFIED bundles={len(bundles)} checksums={checksum_count} "
        "primary_settings=24 bound_rows=11520"
    )


if __name__ == "__main__":
    main()
