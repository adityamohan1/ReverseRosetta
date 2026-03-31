"""Hard validation: translation, ORF, stops, restriction sites, DNA alphabet."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from reverserosetta.config import ReverseRosettaConfig
from reverserosetta.restriction import has_forbidden_site
from reverserosetta.splice import SpliceScanResult, significant_signals
from reverserosetta.utils import CODON_TO_AA, split_codons, translate_dna


@dataclass
class ValidationReport:
    """Structured validation outcome for one sequence."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_dna_alphabet(dna: str) -> None:
    """Raise ValueError if any character is not A/C/G/T."""
    if not re.fullmatch(r"[ACGTacgt]*", dna):
        bad = sorted({c for c in dna.upper() if c not in "ACGT"})
        raise ValueError(f"DNA must contain only A/C/G/T; found: {bad}")


def validate_orf(dna: str) -> None:
    """ORF length divisible by 3."""
    if len(dna) % 3 != 0:
        raise ValueError(f"ORF length {len(dna)} is not divisible by 3.")


def validate_translation_matches(
    dna: str,
    expected_aa_no_stop: str,
    *,
    emit_stop_codon: bool,
) -> None:
    """
    Ensure DNA translates to ``expected_aa_no_stop``, optionally with exactly one C-terminal stop.

    Internal stop codons are always rejected.
    """
    trans = translate_dna(dna)
    if emit_stop_codon:
        if not trans.endswith("*"):
            raise ValueError("Expected a single terminal stop codon in DNA when emit_stop_codon is set.")
        core = trans[:-1]
        if core != expected_aa_no_stop:
            raise ValueError("Translated amino acids (before stop) do not match input protein.")
        if "*" in core:
            raise ValueError("Internal stop codon(s) are not allowed.")
        return
    if trans != expected_aa_no_stop:
        raise ValueError("Translated product does not match input amino acid sequence.")


def assert_codons_translate_to(
    dna: str,
    expected_aa_no_stop: str,
    *,
    emit_stop: bool,
) -> None:
    """Raise ``ValueError`` if translation does not match expectations."""
    validate_translation_matches(
        dna,
        expected_aa_no_stop,
        emit_stop_codon=emit_stop,
    )


def validate_final_sequence(
    dna: str,
    expected_aa_no_stop: str,
    cfg: ReverseRosettaConfig,
    splice: SpliceScanResult | None = None,
) -> ValidationReport:
    """
    Run all hard checks and attach soft-threshold warnings for residual splice risk.

    Returns a :class:`ValidationReport` (``ok`` False on any hard failure).
    """
    rep = ValidationReport(ok=True)
    try:
        validate_dna_alphabet(dna)
    except ValueError as e:
        rep.add_error(str(e))
        return rep
    try:
        validate_orf(dna)
    except ValueError as e:
        rep.add_error(str(e))
        return rep
    try:
        validate_translation_matches(
            dna,
            expected_aa_no_stop,
            emit_stop_codon=cfg.emit_stop_codon,
        )
    except ValueError as e:
        rep.add_error(str(e))
        return rep
    if has_forbidden_site(dna):
        rep.add_error("Forbidden restriction site still present in final DNA.")
    if splice is not None:
        bad = significant_signals(splice, cfg)
        if bad:
            rep.add_warning(
                f"{len(bad)} splice signal(s) remain above threshold after optimization."
            )
    return rep


def assert_codons_translate_to(dna: str, expected_aa_no_stop: str, *, emit_stop: bool) -> None:
    """Fast check used inside optimization loops."""
    validate_translation_matches(dna, expected_aa_no_stop, emit_stop_codon=emit_stop)
