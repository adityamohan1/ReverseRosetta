"""Command-line interface for ReverseRosetta."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from reverserosetta import __version__
from reverserosetta.config import ReverseRosettaConfig, load_yaml_config, merge_config
from reverserosetta.excel_io import read_amino_acid_column
from reverserosetta.optimize import optimize_sequences_from_dataframe
from reverserosetta.reporting import (
    print_results_table,
    results_to_dataframe,
    write_excel_csv,
    write_per_sequence_json_reports,
)
from reverserosetta.utils import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        help="Path to input .xlsx file.",
    ),
    sheet: str = typer.Option("Sheet1", "--sheet", help="Worksheet name."),
    output: Path = typer.Option(
        Path("results/reverserosetta_output.xlsx"),
        "--output",
        help="Primary Excel output path (.xlsx).",
    ),
    host: str = typer.Option("human", "--host", help="Host species label (currently: human)."),
    column_index: int = typer.Option(4, "--column-index", help="1-based column index for AA sequences."),
    splice_donor_threshold: float = typer.Option(
        0.65, "--splice-donor-threshold", help="Heuristic/CNN donor flag threshold."
    ),
    splice_acceptor_threshold: float = typer.Option(
        0.65, "--splice-acceptor-threshold", help="Heuristic/CNN acceptor flag threshold."
    ),
    max_iterations: int = typer.Option(500, "--max-iterations", help="Global refinement iteration budget."),
    emit_stop_codon: bool = typer.Option(
        False, "--emit-stop-codon", help="Append a terminal stop codon after optimization."
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Directory for CSV sidecar and optional JSON audits (defaults next to --output).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
    config: Optional[Path] = typer.Option(None, "--config", help="Optional YAML configuration file."),
    use_splicefinder_keras: bool = typer.Option(
        False,
        "--use-splicefinder-keras",
        help="Load SpliceFinder CNN.h5 from splicefinder_model_dir (requires tensorflow).",
    ),
    splicefinder_model_dir: Optional[Path] = typer.Option(
        None,
        "--splicefinder-model-dir",
        help="Directory containing CNN.h5 (and optional donor_dis.h5, acceptor_dis.h5).",
    ),
    emit_json_reports: bool = typer.Option(
        False,
        "--emit-json-reports",
        help="Write per-sequence JSON audit files under output_dir / audits.",
    ),
) -> None:
    """Optimize coding DNA from protein sequences in an Excel column."""
    if input_path is None:
        typer.echo(ctx.get_help(), err=True)
        raise typer.Exit(code=2)

    setup_logging(verbose)
    log = logging.getLogger("reverserosetta.cli")
    base = ReverseRosettaConfig(
        host=host,
        column_index=column_index,
        emit_stop_codon=emit_stop_codon,
        max_iterations=max_iterations,
        splice_donor_threshold=splice_donor_threshold,
        splice_acceptor_threshold=splice_acceptor_threshold,
        use_splicefinder_keras=use_splicefinder_keras,
        splicefinder_model_dir=splicefinder_model_dir,
    )
    if config is not None:
        overrides = load_yaml_config(config)
        cfg = merge_config(base, overrides)
    else:
        cfg = base

    log.info("ReverseRosetta %s", __version__)
    df_in = read_amino_acid_column(input_path, sheet_name=sheet, column_index=cfg.column_index)
    if df_in.empty:
        raise typer.BadParameter("No sequences found in the selected sheet/column.")

    results = optimize_sequences_from_dataframe(df_in, cfg)
    df_out = results_to_dataframe(results)
    out_dir = output_dir if output_dir is not None else output.parent
    write_excel_csv(df_out, output, output_dir=out_dir)
    print_results_table(df_out)
    if emit_json_reports:
        write_per_sequence_json_reports(results, out_dir / "audits")


def run_cli() -> None:
    """Setuptools / ``reverserosetta`` console script entrypoint."""
    app()


if __name__ == "__main__":
    app()
