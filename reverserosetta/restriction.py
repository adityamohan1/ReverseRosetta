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
    "BseRI": ("GAGGAG",),
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
    "PaqCI": ("CACCTGC",),
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

# Degenerate motifs. Scanned with overlap-aware search on both strands; no hand-kept widths.
RESTRICTION_REGEX: list[tuple[str, str]] = [
    ("BstEII", r"GGT[ACGT]ACC"),
    ("BstXI", r"CCA[ACGT]{6}TGG"),
    ("SfiI", r"GGCC[ACGT]{5}GGCC"),
]


def iter_restriction_hits(dna: str) -> Iterator[RestrictionHit]:
    """
    Yield forbidden motif hits. Literals are matched as substring on both strands.

    For each literal ``P``, occurrences of ``P`` or ``reverse_complement(P)`` in the
    forward sequence are reported (standard dsDNA cloning interpretation).

    Degenerate motifs are searched with the same both-strand convention: the pattern is
    matched against the forward sequence and against its reverse complement, with reverse
    hits reported in forward-strand coordinates. Overlapping occurrences are all reported.
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

    rc_seq = reverse_complement(seq)
    for enzyme, pat in RESTRICTION_REGEX:
        # Lookahead keeps overlapping occurrences, which plain finditer would skip.
        rx = re.compile(f"(?=({pat}))")
        for m in rx.finditer(seq):
            i = m.start(1)
            yield from add_hit(enzyme, i, i + len(m.group(1)), m.group(1), "forward")
        for m in rx.finditer(rc_seq):
            end = n - m.start(1)
            i = end - len(m.group(1))
            yield from add_hit(enzyme, i, end, seq[i:end], "reverse")


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
