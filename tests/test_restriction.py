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


# Canonical recognition sequences (NEB/REBASE) for every enzyme in the motif tables.
# Regression guard: a wrong core or a pattern the scanner cannot match makes an enzyme
# silently never fire, which reads as "no forbidden sites" instead of as a failure.
CANONICAL_SITES: dict[str, str] = {
    "AgeI": "ACCGGT",
    "BamHI": "GGATCC",
    "BbsI": "GAAGAC",
    "BsaI": "GGTCTC",
    "BseRI": "GAGGAG",
    "BstBI": "TTCGAA",
    "BstEII": "GGTNACC",
    "BstXI": "CCANNNNNNTGG",
    "ClaI": "ATCGAT",
    "EagI": "CGGCCG",
    "EcoRI": "GAATTC",
    "EcoRV": "GATATC",
    "Esp3I": "CGTCTC",
    "HindIII": "AAGCTT",
    "KpnI": "GGTACC",
    "MfeI": "CAATTG",
    "MluI": "ACGCGT",
    "NcoI": "CCATGG",
    "NheI": "GCTAGC",
    "NotI": "GCGGCCGC",
    "NruI": "TCGCGA",
    "NsiI": "ATGCAT",
    "PaqCI": "CACCTGC",
    "PmeI": "GTTTAAAC",
    "SacI": "GAGCTC",
    "SalI": "GTCGAC",
    "SapI": "GCTCTTC",
    "SbfI": "CCTGCAGG",
    "ScaI": "AGTACT",
    "SfiI": "GGCCNNNNNGGCC",
    "SmaI": "CCCGGG",
    "SpeI": "ACTAGT",
    "XbaI": "TCTAGA",
    "XhoI": "CTCGAG",
    "XmaI": "CCCGGG",
}


def _expand_n(site: str) -> list[str]:
    out = [""]
    for ch in site:
        bases = "ACGT" if ch == "N" else ch
        out = [o + b for o in out for b in bases]
    return out


def test_every_enzyme_in_tables_has_a_canonical_site() -> None:
    from reverserosetta.restriction import RESTRICTION_LITERALS, RESTRICTION_REGEX

    coded = set(RESTRICTION_LITERALS) | {e for e, _ in RESTRICTION_REGEX}
    assert coded == set(CANONICAL_SITES)


@pytest.mark.parametrize("enzyme", sorted(CANONICAL_SITES))
def test_canonical_site_detected_on_both_strands(enzyme: str) -> None:
    from reverserosetta.utils import reverse_complement

    for site in _expand_n(CANONICAL_SITES[enzyme]):
        for oriented in (site, reverse_complement(site)):
            dna = "TTTT" + oriented + "TTTT"
            hits = [h for h in list_restriction_hits(dna) if h.enzyme == enzyme]
            assert hits, f"{enzyme} missed {oriented}"
            assert dna[hits[0].start : hits[0].end] == hits[0].matched


def test_overlapping_degenerate_sites_are_all_reported() -> None:
    # Two SfiI sites sharing their middle GGCC; a non-overlapping scan reports only one.
    dna = "GGCCAAAAAGGCCAAAAAGGCC"
    spans = [(h.start, h.end) for h in list_restriction_hits(dna) if h.enzyme == "SfiI"]
    assert spans == [(0, 13), (9, 22)]
