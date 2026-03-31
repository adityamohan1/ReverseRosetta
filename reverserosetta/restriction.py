"""Restriction enzyme motifs, scanning (both strands), and synonymous edit proposals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from reverserosetta.utils import AA_TO_CODONS, reverse_complement, split_codons


@dataclass(frozen=True, slots=True)
class RestrictionHit:
    """Occurrence of a forbidden motif in a DNA sequence (forward strand coordinates)."""

    enzyme: str
    start: int
    end: int
    matched: str
    orientation: str  # "forward" or "reverse"


# Literal recognition sequences (IUPAC-free). Both strand search uses P and rc(P) substring scan.
RESTRICTION_LITERALS: dict[str, tuple[str, ...]] = {
    "AgeI": ("ACCGGT",),
    "BamHI": ("GGATCC",),
    "BbsI": ("GAAGAC",),
    "BsaI": ("GGTCTC",),
    "BstBI": ("TTCGAA",),
    "ClaI": ("ATCGAT",),
    "EagI": ("CGGCCG",),
    "EcoRI": ("GAATTC",),
    "EcoRV": ("GATATC",),
    "Esp3I": ("CGTCTC",),
    "HindIII": ("AAGCTT",),
    "KpnI": ("GGTACC",),
    "MfeI": ("CAATTG",),
    "MluI": ("ACGCGT",),
    "NcoI": ("CCATGG",),
    "NheI": ("GCTAGC",),
    "NotI": ("GCGGCCGC",),
    "NruI": ("TCGCGA",),
    "NsiI": ("ATGCAT",),
    "PaqCI": ("GCTGTCC",),
    "PmeI": ("GTTTAAAC",),
    "SacI": ("GAGCTC",),
    "SalI": ("GTCGAC",),
    "SapI": ("GCTCTTC",),
    "SbfI": ("CCTGCAGG",),
    "ScaI": ("AGTACT",),
    "SmaI": ("CCCGGG",),
    "SpeI": ("ACTAGT",),
    "XbaI": ("TCTAGA",),
    "XhoI": ("CTCGAG",),
    "XmaI": ("CCCGGG",),
}

# Degenerate / variable-length: fixed window fullmatch on forward strand or on rc(window).
RESTRICTION_REGEX: list[tuple[str, str, int]] = [
    ("BseRI", r"CC[AT]GG", 6),
    ("BstEII", r"GGT[ACGT]ACC", 9),
    ("BstXI", r"CCA[ACGT]{6}TGG", 10),
    ("SfiI", r"GGCC[ACGT]{5}GGCC", 13),
]


def iter_restriction_hits(dna: str) -> Iterator[RestrictionHit]:
    """
    Yield forbidden motif hits. Literals are matched as substring on both strands.

    For each literal ``P``, occurrences of ``P`` or ``reverse_complement(P)`` in the
    forward sequence are reported (standard dsDNA cloning interpretation).

    Regex motifs use a sliding window of fixed width; a window matches if the forward
    pattern or the same pattern applied to ``reverse_complement(window)`` full-matches.
    """
    seq = dna.upper()
    n = len(seq)
    seen: set[tuple[int, int, str, str]] = set()

    def add_hit(enzyme: str, i: int, j: int, matched: str, ori: str) -> Iterator[RestrictionHit]:
        key = (i, j, matched, enzyme)
        if key in seen:
            return
        seen.add(key)
        yield RestrictionHit(
            enzyme=enzyme, start=i, end=j, matched=matched, orientation=ori
        )

    for enzyme, pats in RESTRICTION_LITERALS.items():
        for pat in pats:
            P = pat.upper()
            R = reverse_complement(P)
            start = 0
            while True:
                idx = seq.find(P, start)
                if idx == -1:
                    break
                yield from add_hit(enzyme, idx, idx + len(P), P, "forward")
                start = idx + 1
            start = 0
            while True:
                idx = seq.find(R, start)
                if idx == -1:
                    break
                yield from add_hit(enzyme, idx, idx + len(R), seq[idx : idx + len(R)], "reverse")
                start = idx + 1

    for enzyme, pat, width in RESTRICTION_REGEX:
        rx = re.compile(pat)
        if width > n:
            continue
        for i in range(0, n - width + 1):
            w = seq[i : i + width]
            if rx.fullmatch(w):
                yield from add_hit(enzyme, i, i + width, w, "forward")
                continue
            wrc = reverse_complement(w)
            if rx.fullmatch(wrc):
                yield from add_hit(enzyme, i, i + width, w, "reverse")


def list_restriction_hits(dna: str) -> list[RestrictionHit]:
    """Return all restriction hits, sorted by start then end."""
    hits = list(iter_restriction_hits(dna))
    hits.sort(key=lambda h: (h.start, h.end, h.enzyme))
    return hits


def has_forbidden_site(dna: str) -> bool:
    """True if any forbidden motif is present."""
    return next(iter_restriction_hits(dna), None) is not None


def codon_indices_overlapping_span(nt_start: int, nt_end: int, seq_len: int) -> list[int]:
    """Zero-based codon indices overlapping [nt_start, nt_end)."""
    if nt_start < 0 or nt_end > seq_len or nt_start >= nt_end:
        raise ValueError("Invalid nucleotide span.")
    if seq_len % 3 != 0:
        raise ValueError("seq_len must be multiple of 3.")
    first = nt_start // 3
    last = (nt_end - 1) // 3
    return list(range(first, last + 1))


@dataclass(frozen=True, slots=True)
class SynonymousEdit:
    """Replace codon at index with a synonymous alternative."""

    codon_index: int
    new_codon: str


def propose_synonymous_edits_overlapping_hit(
    dna: str,
    hit: RestrictionHit,
    aa_sequence: str,
) -> list[SynonymousEdit]:
    """
    Enumerate synonymous codon replacements for codons overlapping a restriction hit.

    Parameters
    ----------
    dna:
        Current DNA (multiple of 3).
    hit:
        Restriction hit on this DNA.
    aa_sequence:
        Expected amino acid sequence (same length as number of codons).
    """
    codons = split_codons(dna)
    if len(codons) != len(aa_sequence):
        raise ValueError("DNA codon count must match amino acid length.")
    idxs = codon_indices_overlapping_span(hit.start, hit.end, len(dna))
    edits: list[SynonymousEdit] = []
    for ci in idxs:
        aa = aa_sequence[ci]
        cur = codons[ci]
        for alt in AA_TO_CODONS[aa]:
            if alt != cur:
                edits.append(SynonymousEdit(codon_index=ci, new_codon=alt))
    return edits
