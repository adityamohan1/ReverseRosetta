"""Multi-stage optimizer: codon assignment, restriction, splice, repeats, validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from reverserosetta.codon_opt import optimize_protein_to_human_dna
from reverserosetta.config import ReverseRosettaConfig
from reverserosetta.repeats import RepeatScoreBreakdown, compute_repeat_score
from reverserosetta.restriction import (
    SynonymousEdit,
    list_restriction_hits,
    propose_synonymous_edits_overlapping_hit,
)
from reverserosetta.splice import SpliceBackend, SpliceScanResult, get_splice_backend, scan_splice, significant_signals
from reverserosetta.utils import AA_TO_CODONS, apply_codon_replace, codon_preference_score, split_codons
from reverserosetta.validate import assert_codons_translate_to, validate_final_sequence

logger = logging.getLogger(__name__)


@dataclass
class OptimizedSequence:
    """Per-row optimization outcome and audit fields."""

    row_index: int
    excel_row: int
    aa_sequence: str
    initial_dna: str
    final_dna: str
    edits_log: list[dict[str, Any]] = field(default_factory=list)
    had_restriction_sites_initially: bool = False
    restriction_sites_removed: bool = True
    splice_before: SpliceScanResult | None = None
    splice_after: SpliceScanResult | None = None
    repeat_before: RepeatScoreBreakdown | None = None
    repeat_after: RepeatScoreBreakdown | None = None
    validation_ok: bool = False
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)


def _soft_score(
    dna: str,
    cfg: ReverseRosettaConfig,
    splice_res: SpliceScanResult,
    rep: RepeatScoreBreakdown,
) -> float:
    """Higher is better (soft composite)."""
    w = cfg.weights
    cod = codon_preference_score(dna)
    return (
        w.w_codon * cod
        - w.w_splice * splice_res.risk_score
        - w.w_repeat * rep.total
    )


def _count_hits(dna: str) -> int:
    return len(list_restriction_hits(dna))


def _emit_stop(dna_body: str) -> str:
    """Append a stop codon, preferring one that does not create a forbidden site."""
    from reverserosetta.restriction import has_forbidden_site

    for stop in ("TAA", "TAG", "TGA"):
        cand = dna_body + stop
        if not has_forbidden_site(cand):
            return cand
    return dna_body + "TAA"


def _stage_restriction(
    dna: str,
    aa: str,
    cfg: ReverseRosettaConfig,
    splice_backend: SpliceBackend,
    edits_log: list[dict[str, Any]],
    iter_budget: list[int],
    *,
    emit_stop: bool,
) -> str:
    cur = dna
    while iter_budget[0] < cfg.max_iterations:
        hits = list_restriction_hits(cur)
        if not hits:
            break
        hit = hits[0]
        proposals = propose_synonymous_edits_overlapping_hit(cur, hit, aa)
        best: tuple[int, float, str, SynonymousEdit] | None = None
        for ed in proposals:
            iter_budget[0] += 1
            if iter_budget[0] > cfg.max_iterations:
                break
            cand = apply_codon_replace(cur, ed.codon_index, ed.new_codon)
            try:
                assert_codons_translate_to(cand, aa, emit_stop=emit_stop)
            except ValueError:
                continue
            nh = _count_hits(cand)
            sp = scan_splice(cand, cfg, splice_backend)
            rp = compute_repeat_score(
                cand,
                homopolymer_max=cfg.repeat_homopolymer_max,
                kmer_size=cfg.repeat_kmer_size,
            )
            obj = _soft_score(cand, cfg, sp, rp)
            if best is None or nh < best[0] or (nh == best[0] and obj > best[1]):
                best = (nh, obj, cand, ed)
        if best is None:
            logger.warning(
                "No synonymous edit removes restriction hit at %s:%s (enzyme=%s)",
                hit.start,
                hit.end,
                hit.enzyme,
            )
            break
        nh, _obj, cand, ed = best
        edits_log.append(
            {
                "stage": "restriction",
                "enzyme": hit.enzyme,
                "hit_start": hit.start,
                "hit_end": hit.end,
                "codon_index": ed.codon_index,
                "new_codon": ed.new_codon,
                "hits_remaining": nh,
            }
        )
        cur = cand
        if nh == 0:
            break
    return cur


def _codon_indices_in_nt_window(seq_len: int, ws: int, we: int) -> list[int]:
    ws = max(0, ws)
    we = min(seq_len, we)
    if we <= ws:
        return []
    first = ws // 3
    last = (we - 1) // 3
    return list(range(first, last + 1))


def _stage_splice(
    dna: str,
    aa: str,
    cfg: ReverseRosettaConfig,
    splice_backend: SpliceBackend,
    edits_log: list[dict[str, Any]],
    iter_budget: list[int],
    *,
    emit_stop: bool,
) -> str:
    cur = dna
    while iter_budget[0] < cfg.max_iterations:
        scan = scan_splice(cur, cfg, splice_backend)
        bad = significant_signals(scan, cfg)
        if not bad:
            break
        bad.sort(key=lambda s: s.score, reverse=True)
        sig = bad[0]
        cur_codons = split_codons(cur)
        base_risk = scan.risk_score
        best_cand: str | None = None
        best_key: tuple[float, float] | None = None
        for ci in _codon_indices_in_nt_window(len(cur), sig.window_start, sig.window_end):
            for alt in AA_TO_CODONS[aa[ci]]:
                if alt == cur_codons[ci]:
                    continue
                iter_budget[0] += 1
                if iter_budget[0] > cfg.max_iterations:
                    break
                cand = apply_codon_replace(cur, ci, alt)
                try:
                    assert_codons_translate_to(cand, aa, emit_stop=emit_stop)
                except ValueError:
                    continue
                if _count_hits(cand) > 0:
                    continue
                sp = scan_splice(cand, cfg, splice_backend)
                rp = compute_repeat_score(
                    cand,
                    homopolymer_max=cfg.repeat_homopolymer_max,
                    kmer_size=cfg.repeat_kmer_size,
                )
                obj = _soft_score(cand, cfg, sp, rp)
                key = (sp.risk_score, -obj)
                if sp.risk_score < base_risk - 1e-9 and (best_key is None or key < best_key):
                    best_key = key
                    best_cand = cand
        if best_cand is None:
            break
        edits_log.append({"stage": "splice", "signal_center": sig.center, "kind": sig.kind})
        cur = best_cand
    return cur


def _stage_repeats(
    dna: str,
    aa: str,
    cfg: ReverseRosettaConfig,
    splice_backend: SpliceBackend,
    edits_log: list[dict[str, Any]],
    iter_budget: list[int],
    *,
    emit_stop: bool,
) -> str:
    cur = dna
    while iter_budget[0] < cfg.max_iterations:
        rep = compute_repeat_score(
            cur,
            homopolymer_max=cfg.repeat_homopolymer_max,
            kmer_size=cfg.repeat_kmer_size,
        )
        scan = scan_splice(cur, cfg, splice_backend)
        base_obj = _soft_score(cur, cfg, scan, rep)
        cur_codons = split_codons(cur)
        best_cand: str | None = None
        best_obj = base_obj
        n = len(aa)
        for ci in range(n):
            for alt in AA_TO_CODONS[aa[ci]]:
                if alt == cur_codons[ci]:
                    continue
                iter_budget[0] += 1
                if iter_budget[0] > cfg.max_iterations:
                    break
                cand = apply_codon_replace(cur, ci, alt)
                try:
                    assert_codons_translate_to(cand, aa, emit_stop=emit_stop)
                except ValueError:
                    continue
                if _count_hits(cand) > 0:
                    continue
                sp = scan_splice(cand, cfg, splice_backend)
                rp = compute_repeat_score(
                    cand,
                    homopolymer_max=cfg.repeat_homopolymer_max,
                    kmer_size=cfg.repeat_kmer_size,
                )
                if rp.total >= rep.total:
                    continue
                obj = _soft_score(cand, cfg, sp, rp)
                if obj > best_obj:
                    best_obj = obj
                    best_cand = cand
        if best_cand is None:
            break
        edits_log.append({"stage": "repeat", "improved_repeat_score": True})
        cur = best_cand
    return cur


def optimize_one_sequence(
    aa_sequence: str,
    cfg: ReverseRosettaConfig,
    *,
    excel_row: int,
    row_index: int,
    splice_backend: SpliceBackend | None = None,
    codon_fn: Any | None = None,
) -> OptimizedSequence:
    """
    Run the full multi-stage optimization for a single amino acid sequence.

    Parameters
    ----------
    aa_sequence:
        Input protein (no stop letter; terminal stop is appended when ``emit_stop_codon``).
    cfg:
        Pipeline configuration.
    excel_row:
        Original 1-based Excel row label for traceability.
    row_index:
        Zero-based output table index.
    splice_backend:
        Optional injected backend (tests); default from ``get_splice_backend(cfg)``.
    codon_fn:
        Optional replacement for :func:`optimize_protein_to_human_dna` (tests).
    """
    be = splice_backend or get_splice_backend(cfg)
    cf = codon_fn or optimize_protein_to_human_dna
    edits: list[dict[str, Any]] = []
    iter_budget = [0]

    logger.info("Stage 1: CodonTransformer initial DNA (row %s)", excel_row)
    initial = cf(
        aa_sequence,
        host=cfg.host,
        deterministic=True,
        match_protein=True,
    )
    dna = initial
    had_rest = _count_hits(dna) > 0

    logger.info("Stage 2: restriction removal (row %s)", excel_row)
    dna = _stage_restriction(dna, aa_sequence, cfg, be, edits, iter_budget, emit_stop=False)

    logger.info("Stage 3: splice minimization (row %s)", excel_row)
    splice_before = scan_splice(dna, cfg, be)
    dna = _stage_splice(dna, aa_sequence, cfg, be, edits, iter_budget, emit_stop=False)

    logger.info("Stage 4: repeat minimization (row %s)", excel_row)
    repeat_before = compute_repeat_score(
        dna,
        homopolymer_max=cfg.repeat_homopolymer_max,
        kmer_size=cfg.repeat_kmer_size,
    )
    dna = _stage_repeats(dna, aa_sequence, cfg, be, edits, iter_budget, emit_stop=False)

    aa_for_opt = aa_sequence
    if cfg.emit_stop_codon:
        dna = _emit_stop(dna)
        aa_for_opt = aa_sequence + "*"
        if _count_hits(dna) > 0:
            logger.info("Post-stop restriction cleanup (row %s)", excel_row)
            dna = _stage_restriction(dna, aa_for_opt, cfg, be, edits, iter_budget, emit_stop=False)

    splice_after = scan_splice(dna, cfg, be)
    repeat_after = compute_repeat_score(
        dna,
        homopolymer_max=cfg.repeat_homopolymer_max,
        kmer_size=cfg.repeat_kmer_size,
    )

    val = validate_final_sequence(dna, aa_sequence, cfg, splice=splice_after)

    return OptimizedSequence(
        row_index=row_index,
        excel_row=excel_row,
        aa_sequence=aa_sequence,
        initial_dna=initial,
        final_dna=dna,
        edits_log=edits,
        had_restriction_sites_initially=had_rest,
        restriction_sites_removed=_count_hits(dna) == 0,
        splice_before=splice_before,
        splice_after=splice_after,
        repeat_before=repeat_before,
        repeat_after=repeat_after,
        validation_ok=val.ok,
        validation_errors=val.errors,
        validation_warnings=val.warnings,
    )


def optimize_sequences_from_dataframe(
    df,
    cfg: ReverseRosettaConfig,
    *,
    splice_backend: SpliceBackend | None = None,
    codon_fn: Any | None = None,
) -> list[OptimizedSequence]:
    """
    Optimize each row of a DataFrame from :func:`reverserosetta.excel_io.read_amino_acid_column`.

    Expects columns ``excel_row`` and ``aa_sequence``.
    """
    out: list[OptimizedSequence] = []
    be = splice_backend or get_splice_backend(cfg)
    for i, row in df.iterrows():
        seq = str(row["aa_sequence"])
        er = int(row["excel_row"])
        out.append(
            optimize_one_sequence(
                seq,
                cfg,
                excel_row=er,
                row_index=int(i),
                splice_backend=be,
                codon_fn=codon_fn,
            )
        )
    return out
