"""Repeat and synthesis-risk scoring."""

from __future__ import annotations

from reverserosetta.repeats import (
    compute_repeat_score,
    longest_homopolymer_run,
    tandem_repeat_score,
)


def test_homopolymer() -> None:
    assert longest_homopolymer_run("AAABBB") == 3
    assert longest_homopolymer_run("") == 0


def test_tandem_penalty() -> None:
    s = "AT" * 10
    assert tandem_repeat_score(s, unit_min=2, unit_max=2, min_repeats=4) > 0


def test_repeat_score_positive() -> None:
    r = compute_repeat_score("AAAAAAAAAA", homopolymer_max=4, kmer_size=6)
    assert r.total > 0
