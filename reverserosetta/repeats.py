"""Repeat burden and synthesis-risk heuristics for DNA coding sequences."""

from __future__ import annotations

from dataclasses import dataclass

from reverserosetta.utils import kmer_counts


@dataclass(frozen=True, slots=True)
class RepeatScoreBreakdown:
    """Component scores before aggregation."""

    homopolymer_penalty: float
    tandem_penalty: float
    kmer_penalty: float
    complexity_penalty: float

    @property
    def total(self) -> float:
        return (
            self.homopolymer_penalty
            + self.tandem_penalty
            + self.kmer_penalty
            + self.complexity_penalty
        )


def longest_homopolymer_run(seq: str) -> int:
    """Length of the longest homopolymeric run (case-insensitive)."""
    s = seq.upper()
    if not s:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def tandem_repeat_score(seq: str, unit_min: int = 2, unit_max: int = 5, min_repeats: int = 4) -> float:
    """
    Penalize short tandem repeats (e.g. ATCATCATC...).

    Returns a nonnegative score; higher is worse.
    """
    s = seq.upper()
    n = len(s)
    penalty = 0.0
    for u in range(unit_min, unit_max + 1):
        i = 0
        while i + u * min_repeats <= n:
            unit = s[i : i + u]
            j = i
            reps = 0
            while j + u <= n and s[j : j + u] == unit:
                reps += 1
                j += u
            if reps >= min_repeats:
                penalty += float(reps - min_repeats + 1) * 0.15
            i += 1
    return penalty


def kmer_repeat_penalty(seq: str, k: int) -> float:
    """Penalize overrepresented k-mers via max count above expectation."""
    if k < 3 or len(seq) < k:
        return 0.0
    counts = kmer_counts(seq, k)
    if not counts:
        return 0.0
    max_c = max(counts.values())
    # Soft cap: expected ~ L / 4^k for random DNA; penalize excess multiplicity.
    expected = max(1.0, len(seq) / (4**k))
    excess = max(0.0, float(max_c) - 3 * expected)
    return excess


def local_complexity_penalty(seq: str, window: int = 30) -> float:
    """
    Penalize low Shannon entropy sliding windows (simple sequences).

    Uses log2 alphabet size 2 bits max per position for DNA.
    """
    s = seq.upper()
    n = len(s)
    if n < window:
        window = n
    if window == 0:
        return 0.0
    import math

    pen = 0.0
    for i in range(0, n - window + 1):
        w = s[i : i + window]
        freqs: dict[str, int] = {}
        for ch in w:
            freqs[ch] = freqs.get(ch, 0) + 1
        h = 0.0
        for c in freqs.values():
            p = c / window
            h -= p * math.log2(p)
        # Max entropy for DNA = 2.0
        pen += max(0.0, 1.6 - h)
    return pen / max(1, n - window + 1)


def compute_repeat_score(
    dna: str,
    *,
    homopolymer_max: int = 8,
    kmer_size: int = 6,
) -> RepeatScoreBreakdown:
    """
    Compute a composite repeat / manufacturability burden score.

    Parameters
    ----------
    dna:
        DNA string (uppercase).
    homopolymer_max:
        Runs longer than this contribute quadratically.
    kmer_size:
        k for k-mer repetition term.
    """
    run = longest_homopolymer_run(dna)
    homo = max(0.0, float(run - homopolymer_max)) ** 2 * 0.05
    tandem = tandem_repeat_score(dna)
    kmer = kmer_repeat_penalty(dna, kmer_size)
    comp = local_complexity_penalty(dna)
    return RepeatScoreBreakdown(
        homopolymer_penalty=homo,
        tandem_penalty=tandem,
        kmer_penalty=kmer,
        complexity_penalty=comp,
    )
