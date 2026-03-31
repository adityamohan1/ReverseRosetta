"""Output tables, Excel/CSV export, and optional per-sequence JSON audit logs."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from reverserosetta.splice import SpliceScanResult

import pandas as pd

from reverserosetta.optimize import OptimizedSequence

logger = logging.getLogger(__name__)


def results_to_dataframe(rows: list[OptimizedSequence]) -> pd.DataFrame:
    """Build the main results table for display and export."""
    records: list[dict[str, Any]] = []
    for r in rows:
        splice_b = r.splice_before
        splice_a = r.splice_after
        rep_b = r.repeat_before
        rep_a = r.repeat_after
        records.append(
            {
                "row_index": r.row_index,
                "excel_row": r.excel_row,
                "original_aa": r.aa_sequence,
                "optimized_dna": r.final_dna,
                "dna_length_nt": len(r.final_dna),
                "restriction_sites_removed": r.restriction_sites_removed,
                "had_restriction_sites_initially": r.had_restriction_sites_initially,
                "splice_max_donor_before": getattr(splice_b, "max_donor", None),
                "splice_max_acceptor_before": getattr(splice_b, "max_acceptor", None),
                "splice_max_donor_after": getattr(splice_a, "max_donor", None),
                "splice_max_acceptor_after": getattr(splice_a, "max_acceptor", None),
                "splice_risk_before": getattr(splice_b, "risk_score", None),
                "splice_risk_after": getattr(splice_a, "risk_score", None),
                "repeat_score_before": rep_b.total if rep_b else None,
                "repeat_score_after": rep_a.total if rep_a else None,
                "validation_ok": r.validation_ok,
                "validation_errors": "; ".join(r.validation_errors),
                "validation_warnings": "; ".join(r.validation_warnings),
            }
        )
    return pd.DataFrame.from_records(records)


def write_excel_csv(df: pd.DataFrame, output_xlsx: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    """Write Excel and CSV next to ``output_xlsx`` or under ``output_dir``."""
    output_xlsx = output_xlsx.resolve()
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    csv_path = (
        (output_dir / (output_xlsx.stem + ".csv")) if output_dir else output_xlsx.with_suffix(".csv")
    )
    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_xlsx, index=False, engine="openpyxl")
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s and %s", output_xlsx, csv_path)
    return output_xlsx, csv_path


def write_per_sequence_json_reports(rows: list[OptimizedSequence], out_dir: Path) -> None:
    """Emit one JSON audit file per optimized sequence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        path = out_dir / f"sequence_row{r.row_index}_excel{r.excel_row}.json"
        payload: dict[str, Any] = {
            "row_index": r.row_index,
            "excel_row": r.excel_row,
            "original_aa": r.aa_sequence,
            "initial_dna": r.initial_dna,
            "final_dna": r.final_dna,
            "edits": r.edits_log,
            "had_restriction_sites_initially": r.had_restriction_sites_initially,
            "restriction_sites_removed": r.restriction_sites_removed,
            "splice_before": None
            if r.splice_before is None
            else _splice_to_dict(r.splice_before),
            "splice_after": None if r.splice_after is None else _splice_to_dict(r.splice_after),
            "repeat_before": asdict(r.repeat_before) if r.repeat_before else None,
            "repeat_after": asdict(r.repeat_after) if r.repeat_after else None,
            "validation_ok": r.validation_ok,
            "validation_errors": r.validation_errors,
            "validation_warnings": r.validation_warnings,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _splice_to_dict(sr: SpliceScanResult) -> dict[str, Any]:
    top = sorted(sr.signals, key=lambda s: s.score, reverse=True)[:24]
    return {
        "max_donor": sr.max_donor,
        "max_acceptor": sr.max_acceptor,
        "risk_score": sr.risk_score,
        "num_signals": len(sr.signals),
        "top_signals": [asdict(x) for x in top],
    }


def print_results_table(df: pd.DataFrame) -> None:
    """Log/preview a concise table (full width may wrap in terminals)."""
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))
