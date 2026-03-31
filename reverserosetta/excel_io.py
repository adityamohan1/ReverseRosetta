"""Excel input: read amino acid sequences from a user-specified column and sheet."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from reverserosetta.utils import VALID_PROTEIN

logger = logging.getLogger(__name__)


class InvalidAminoAcidError(ValueError):
    """Raised when a sequence contains characters outside the allowed protein alphabet."""


def read_amino_acid_column(
    path: Path,
    sheet_name: str,
    column_index: int,
) -> pd.DataFrame:
    """
    Read an Excel sheet and return a DataFrame with row index and cleaned sequences.

    Parameters
    ----------
    path:
        Path to .xlsx file.
    sheet_name:
        Sheet to read.
    column_index:
        1-based column index (e.g. 4 for column D).

    Returns
    -------
    DataFrame with columns: ``excel_row`` (1-based sheet row), ``aa_sequence``.
    Blank cells are dropped. Sequences are stripped and uppercased.
    """
    if column_index < 1:
        raise ValueError("column_index must be >= 1 (1-based).")
    col_zero = column_index - 1
    if not path.is_file():
        raise FileNotFoundError(f"Excel file not found: {path}")

    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=object)
    # Trailing empty columns are often omitted by Excel I/O; pad through the requested index.
    while raw.shape[1] <= col_zero:
        raw[raw.shape[1]] = None
    if col_zero < 0:
        raise ValueError("Invalid column index.")

    series = raw.iloc[:, col_zero]
    out_rows: list[tuple[int, str]] = []
    for i, val in enumerate(series.tolist()):
        excel_row = i + 1  # 1-based row in sheet (including header rows if any)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        text = str(val).strip()
        if not text:
            continue
        aa = text.upper().replace(" ", "")
        validate_amino_acid_string(aa)
        out_rows.append((excel_row, aa))

    df = pd.DataFrame(out_rows, columns=["excel_row", "aa_sequence"])
    logger.info("Loaded %d non-blank sequences from %s sheet %r", len(df), path, sheet_name)
    return df.reset_index(drop=True)


def validate_amino_acid_string(aa: str) -> None:
    """
    Ensure `aa` uses only standard one-letter amino acid codes (no stop by default).

    Raises
    ------
    InvalidAminoAcidError
        If any character is not in the allowed set.
    """
    bad = sorted({c for c in aa if c not in VALID_PROTEIN})
    if bad:
        raise InvalidAminoAcidError(
            f"Invalid amino acid character(s) {bad!r} in sequence (allowed: {sorted(VALID_PROTEIN)})."
        )


def write_template_excel(path: Path, sheet_name: str = "Sheet1") -> None:
    """Create a minimal example workbook with sequences in column D (index 4)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "id": ["ex1", "ex2"],
            "note": ["short", "restriction-prone"],
            "extra": ["", ""],
            "aa_sequence": ["MKWVTFIS", "MGEAAAKV"],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
