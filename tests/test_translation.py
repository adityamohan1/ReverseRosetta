"""Translation and validation checks."""

from __future__ import annotations

import pytest

from reverserosetta.utils import apply_codon_replace, split_codons, translate_dna
from reverserosetta.validate import (
    ValidationReport,
    assert_codons_translate_to,
    validate_final_sequence,
    validate_orf,
    validate_translation_matches,
)
from reverserosetta.config import ReverseRosettaConfig


def test_translate_roundtrip() -> None:
    aa = "MKT"
    dna = "ATGAAAACG"
    assert translate_dna(dna) == aa


def test_validate_emit_stop() -> None:
    aa = "MK"
    dna = "ATG" + "AAG" + "TAA"
    validate_translation_matches(dna, aa, emit_stop_codon=True)


def test_validate_rejects_internal_stop() -> None:
    aa = "MK"
    dna = "ATG" + "TAA" + "AAG"
    with pytest.raises(ValueError):
        validate_translation_matches(dna, aa, emit_stop_codon=True)


def test_assert_codons_translate_to() -> None:
    dna = "ATGGGC"
    assert_codons_translate_to(dna, "MG", emit_stop=False)


def test_validate_orf_divisible() -> None:
    validate_orf("ATGATG")
    with pytest.raises(ValueError):
        validate_orf("ATGA")


def test_validate_final_sequence_csv_config() -> None:
    cfg = ReverseRosettaConfig(emit_stop_codon=False)
    dna = "ATGGGC" * 3  # MG MG MG
    aa = "MGMGMG"
    rep = validate_final_sequence(dna, aa, cfg, splice=None)
    assert isinstance(rep, ValidationReport)
