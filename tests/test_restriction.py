"""Restriction motif scan and synonymous edit proposals."""

from __future__ import annotations

import pytest

from reverserosetta.restriction import (
    has_forbidden_site,
    list_restriction_hits,
    propose_synonymous_edits_overlapping_hit,
)
from reverserosetta.utils import apply_codon_replace


def test_ecori_forward_hit() -> None:
    dna = "GCG" * 5 + "GAATTC" + "GCG" * 5
    hits = list_restriction_hits(dna)
    assert any(h.enzyme == "EcoRI" and h.matched == "GAATTC" for h in hits)


def test_bsaI_detected_via_reverse_complement_on_watson() -> None:
    # BsaI recognition GGTCTC; the opposite strand 5'→3' projects as rc(GGTCTC)=GAGACC on Watson.
    dna = "GAGACC"
    hits = list_restriction_hits(dna)
    assert any(h.enzyme == "BsaI" for h in hits)


def test_propose_edits_on_embedded_site() -> None:
    from reverserosetta.utils import translate_dna

    # M ATG, K AAG contains no EcoRI; use explicit DNA with GAA TTC as two codons: E F
    aa = "MEF"
    dna = "ATG" + "GAA" + "TTC"
    assert translate_dna(dna) == aa
    hits = list_restriction_hits(dna)
    eco = [h for h in hits if h.enzyme == "EcoRI"]
    assert len(eco) == 1
    edits = propose_synonymous_edits_overlapping_hit(dna, eco[0], aa)
    assert edits
    for e in edits:
        cand = apply_codon_replace(dna, e.codon_index, e.new_codon)
        assert translate_dna(cand) == aa


def test_removing_site_may_create_another_is_detected() -> None:
    # Regression guard: scanner finds all sites independently.
    dna = "GAATTC" + "GAATTC"
    assert len([h for h in list_restriction_hits(dna) if h.enzyme == "EcoRI"]) >= 2


def test_has_forbidden_site() -> None:
    assert has_forbidden_site("GGATCC")
    assert not has_forbidden_site("AAAAAA")
