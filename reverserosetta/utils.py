"""Shared helpers: logging, DNA operations, genetic code, codon usage."""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from typing import Final

# Standard genetic code: amino acid -> synonymous codons (DNA, uppercase).
AA_TO_CODONS: dict[str, list[str]] = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "N": ["AAT", "AAC"],
    "D": ["GAT", "GAC"],
    "C": ["TGT", "TGC"],
    "Q": ["CAA", "CAG"],
    "E": ["GAA", "GAG"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "K": ["AAA", "AAG"],
    "M": ["ATG"],
    "F": ["TTT", "TTC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "W": ["TGG"],
    "Y": ["TAT", "TAC"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "*": ["TAA", "TAG", "TGA"],
}

# Build codon -> aa
CODON_TO_AA: dict[str, str] = {}
for aa, codons in AA_TO_CODONS.items():
    for c in codons:
        CODON_TO_AA[c] = aa

# Homo sapiens relative synonymous codon usage (approximate; sums to aa family).
# Source: Kazusa Codon Usage Database style frequencies, normalized per amino acid in code.
HUMAN_CODON_WEIGHT: dict[str, float] = {
    "GCT": 0.26,
    "GCC": 0.40,
    "GCA": 0.23,
    "GCG": 0.11,
    "CGT": 0.08,
    "CGC": 0.19,
    "CGA": 0.11,
    "CGG": 0.21,
    "AGA": 0.21,
    "AGG": 0.20,
    "AAT": 0.47,
    "AAC": 0.53,
    "GAT": 0.46,
    "GAC": 0.54,
    "TGT": 0.45,
    "TGC": 0.55,
    "CAA": 0.25,
    "CAG": 0.75,
    "GAA": 0.42,
    "GAG": 0.58,
    "GGT": 0.16,
    "GGC": 0.34,
    "GGA": 0.25,
    "GGG": 0.25,
    "CAT": 0.42,
    "CAC": 0.58,
    "ATT": 0.36,
    "ATC": 0.48,
    "ATA": 0.16,
    "TTA": 0.08,
    "TTG": 0.13,
    "CTT": 0.13,
    "CTC": 0.20,
    "CTA": 0.07,
    "CTG": 0.41,
    "AAA": 0.43,
    "AAG": 0.57,
    "ATG": 1.0,
    "TTT": 0.45,
    "TTC": 0.55,
    "CCT": 0.29,
    "CCC": 0.32,
    "CCA": 0.28,
    "CCG": 0.11,
    "TCT": 0.19,
    "TCC": 0.22,
    "TCA": 0.15,
    "TCG": 0.06,
    "AGT": 0.15,
    "AGC": 0.24,
    "ACT": 0.25,
    "ACC": 0.36,
    "ACA": 0.28,
    "ACG": 0.11,
    "TGG": 1.0,
    "TAT": 0.43,
    "TAC": 0.57,
    "GTT": 0.18,
    "GTC": 0.24,
    "GTA": 0.11,
    "GTG": 0.47,
    "TAA": 0.28,
    "TAG": 0.20,
    "TGA": 0.52,
}

_COMP: dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}

VALID_PROTEIN: Final[set[str]] = set(AA_TO_CODONS.keys()) - {"*"}


def setup_logging(verbose: bool) -> None:
    """Configure root logging for CLI runs."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA string (uppercase A/C/G/T only)."""
    s = seq.upper()
    try:
        return "".join(_COMP[b] for b in reversed(s))
    except KeyError as e:
        raise ValueError(f"Invalid DNA base in sequence: {e!s}") from e


def split_codons(dna: str) -> list[str]:
    """Split DNA into codon triplets; fails if length not divisible by 3."""
    if len(dna) % 3 != 0:
        raise ValueError("DNA length must be divisible by 3.")
    d = dna.upper()
    return [d[i : i + 3] for i in range(0, len(d), 3)]


def translate_dna(dna: str) -> str:
    """Translate coding DNA to a one-letter amino acid string (no stop handling special)."""
    aas: list[str] = []
    for c in split_codons(dna):
        if c not in CODON_TO_AA:
            raise ValueError(f"Unknown codon: {c}")
        aas.append(CODON_TO_AA[c])
    return "".join(aas)


def codon_preference_score(dna: str) -> float:
    """Sum log human codon weights (higher is more human-favored)."""
    import math

    total = 0.0
    for c in split_codons(dna):
        w = HUMAN_CODON_WEIGHT.get(c, 1e-6)
        total += math.log(max(w, 1e-9))
    return total


def synonymous_codons_for(aa: str) -> list[str]:
    """Return all codons encoding amino acid `aa` (including stop '*')."""
    if aa not in AA_TO_CODONS:
        raise KeyError(f"No codons for amino acid: {aa!r}")
    return list(AA_TO_CODONS[aa])


def apply_codon_replace(dna: str, codon_index: int, new_codon: str) -> str:
    """Replace codon at zero-based index `codon_index`."""
    codons = split_codons(dna)
    if codon_index < 0 or codon_index >= len(codons):
        raise IndexError("codon_index out of range.")
    codons[codon_index] = new_codon.upper()
    return "".join(codons)


def kmer_counts(seq: str, k: int) -> dict[str, int]:
    """Count k-mers in DNA string."""
    s = seq.upper()
    out: dict[str, int] = defaultdict(int)
    for i in range(0, max(0, len(s) - k + 1)):
        out[s[i : i + k]] += 1
    return dict(out)
