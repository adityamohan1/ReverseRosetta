#!/usr/bin/env python3
"""Create the example Excel template (if missing) and print a sample CLI invocation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEMPLATE = ROOT / "examples" / "ReverseRosettaTemplate.xlsx"


def main() -> None:
    from reverserosetta.excel_io import write_template_excel

    TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
    if not TEMPLATE.is_file():
        write_template_excel(TEMPLATE, sheet_name="Sheet1")
        print(f"Wrote template: {TEMPLATE}")
    else:
        print(f"Template already exists: {TEMPLATE}")
    print("\nExample run:")
    print(
        f"  python -m reverserosetta --input {TEMPLATE} "
        "--sheet Sheet1 --output results/reverserosetta_output.xlsx --host human -v"
    )


if __name__ == "__main__":
    main()
