"""Splice-site risk scoring: SpliceFinder Keras adapter + deterministic heuristic fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from reverserosetta.config import ReverseRosettaConfig

logger = logging.getLogger(__name__)

SpliceKind = Literal["donor", "acceptor"]


@dataclass(frozen=True, slots=True)
class SpliceSignal:
    """A localized donor/acceptor-like prediction on the forward strand."""

    center: int
    kind: SpliceKind
    score: float
    window_start: int
    window_end: int


@dataclass
class SpliceScanResult:
    """Full-sequence splice scan summary."""

    signals: list[SpliceSignal] = field(default_factory=list)
    max_donor: float = 0.0
    max_acceptor: float = 0.0

    @property
    def risk_score(self) -> float:
        """Scalar soft objective term (higher = worse)."""
        return float(self.max_donor + self.max_acceptor)


class SpliceBackend:
    """Protocol-like base; swap implementations via config."""

    def scan(self, dna: str, cfg: ReverseRosettaConfig) -> SpliceScanResult:
        raise NotImplementedError


def _base_one_hot(seq: str) -> np.ndarray:
    """Shape (L, 4) in A,C,G,T column order."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    s = seq.upper()
    out = np.zeros((len(s), 4), dtype=np.float32)
    for i, ch in enumerate(s):
        out[i, mapping[ch]] = 1.0
    return out


def _window_around_center(seq: str, center: int, width: int = 400) -> str:
    """Extract ``width`` nt centered at ``center``, pad with ``A`` if needed."""
    n = len(seq)
    half = width // 2
    lo = center - half
    hi = center + half
    if width % 2 == 1:
        hi += 1
    parts: list[str] = []
    if lo < 0:
        parts.append("A" * (-lo))
        lo = 0
    if hi > n:
        parts.append(seq[lo:n])
        parts.append("A" * (hi - n))
    else:
        parts.append(seq[lo:hi])
    w = "".join(parts)
    if len(w) != width:
        # Trim or pad edge case for odd lengths
        if len(w) > width:
            w = w[:width]
        else:
            w = w + "A" * (width - len(w))
    return w.upper()


class HeuristicSpliceBackend(SpliceBackend):
    """
    Lightweight cryptic splice surrogate (no trained weights).

    Penalizes strong GT donors and AG acceptors in a pyrimidine-rich / polypurine context.
    Scores are in ``[0, 1]`` per position, comparable order-of-magnitude for thresholding.
    """

    def scan(self, dna: str, cfg: ReverseRosettaConfig) -> SpliceScanResult:
        seq = dna.upper()
        n = len(seq)
        sigs: list[SpliceSignal] = []
        max_d = 0.0
        max_a = 0.0
        for i in range(n - 1):
            din = seq[i : i + 2]
            if din == "GT":
                upstream = seq[max(0, i - 20) : i]
                py = sum(1 for b in upstream if b in "CT")
                score = min(1.0, 0.35 + 0.02 * py)
                ws = max(0, i - cfg.splice_window_radius)
                we = min(n, i + cfg.splice_window_radius)
                sigs.append(
                    SpliceSignal(
                        center=i,
                        kind="donor",
                        score=score,
                        window_start=ws,
                        window_end=we,
                    )
                )
                max_d = max(max_d, score)
            if din == "AG":
                downstream = seq[i + 2 : min(n, i + 22)]
                pu = sum(1 for b in downstream if b in "AG")
                score = min(1.0, 0.35 + 0.02 * pu)
                ws = max(0, i - cfg.splice_window_radius)
                we = min(n, i + cfg.splice_window_radius)
                sigs.append(
                    SpliceSignal(
                        center=i,
                        kind="acceptor",
                        score=score,
                        window_start=ws,
                        window_end=we,
                    )
                )
                max_a = max(max_a, score)
        return SpliceScanResult(signals=sigs, max_donor=max_d, max_acceptor=max_a)


class KerasSpliceFinderBackend(SpliceBackend):
    """
    Optional SpliceFinder CNN (TensorFlow Keras ``.h5``).

    Expects ``CNN.h5`` (3-class donor/acceptor/non-splice) in ``splicefinder_model_dir``.
    If ``donor_dis.h5`` / ``acceptor_dis.h5`` exist, they are applied as in the upstream
    ``test_Cla.py`` refinement (binary discrimination).
    """

    def __init__(self) -> None:
        self._main = None
        self._donor = None
        self._acceptor = None

    def _lazy_load(self, cfg: ReverseRosettaConfig) -> None:
        if self._main is not None:
            return
        if not cfg.splicefinder_model_dir:
            raise ValueError("splicefinder_model_dir required for Keras backend.")
        from pathlib import Path

        from tensorflow.keras.models import load_model  # type: ignore[import-untyped]

        d = Path(cfg.splicefinder_model_dir)
        main_p = d / "CNN.h5"
        if not main_p.is_file():
            raise FileNotFoundError(f"Missing SpliceFinder CNN.h5 under {d}")
        self._main = load_model(main_p)
        dp, ap = d / "donor_dis.h5", d / "acceptor_dis.h5"
        if dp.is_file():
            self._donor = load_model(dp)
        if ap.is_file():
            self._acceptor = load_model(ap)
        logger.info("Loaded SpliceFinder Keras models from %s", d)

    def scan(self, dna: str, cfg: ReverseRosettaConfig) -> SpliceScanResult:
        self._lazy_load(cfg)
        assert self._main is not None
        seq = dna.upper()
        n = len(seq)
        width = 400
        if n < 2:
            return SpliceScanResult()
        sigs: list[SpliceSignal] = []
        max_d = 0.0
        max_a = 0.0
        step = max(1, cfg.splice_window_radius // 4)
        for center in range(1, n - 1, step):
            w = _window_around_center(seq, center, width)
            x = _base_one_hot(w).reshape(1, width, 4)
            probs = np.asarray(self._main.predict(x, verbose=0), dtype=np.float64).reshape(-1)
            cls = int(np.argmax(probs))
            conf = float(probs[cls])
            if cls == 2:
                continue
            kind: SpliceKind = "acceptor" if cls == 0 else "donor"
            refined = conf
            if kind == "donor" and self._donor is not None:
                dprob = np.asarray(self._donor.predict(x, verbose=0), dtype=np.float64).reshape(-1)
                dcls = int(np.argmax(dprob))
                if dcls == 1:
                    continue
            if kind == "acceptor" and self._acceptor is not None:
                aprob = np.asarray(self._acceptor.predict(x, verbose=0), dtype=np.float64).reshape(-1)
                acls = int(np.argmax(aprob))
                if acls == 1:
                    continue
            ws = max(0, center - cfg.splice_window_radius)
            we = min(n, center + cfg.splice_window_radius)
            sigs.append(
                SpliceSignal(
                    center=center,
                    kind=kind,
                    score=refined,
                    window_start=ws,
                    window_end=we,
                )
            )
            if kind == "donor":
                max_d = max(max_d, refined)
            else:
                max_a = max(max_a, refined)
        return SpliceScanResult(signals=sigs, max_donor=max_d, max_acceptor=max_a)


def get_splice_backend(cfg: ReverseRosettaConfig) -> SpliceBackend:
    """Choose backend from configuration."""
    if cfg.use_splicefinder_keras and cfg.splicefinder_model_dir:
        try:
            return KerasSpliceFinderBackend()
        except Exception as e:
            logger.warning("Falling back to heuristic splice backend: %s", e)
            return HeuristicSpliceBackend()
    return HeuristicSpliceBackend()


def scan_splice(dna: str, cfg: ReverseRosettaConfig, backend: SpliceBackend | None = None) -> SpliceScanResult:
    """Run configured splice scan."""
    be = backend or get_splice_backend(cfg)
    return be.scan(dna, cfg)


def significant_signals(
    result: SpliceScanResult,
    cfg: ReverseRosettaConfig,
) -> list[SpliceSignal]:
    """Filter scan to signals exceeding configured thresholds."""
    out: list[SpliceSignal] = []
    for s in result.signals:
        if s.kind == "donor" and s.score >= cfg.splice_donor_threshold:
            out.append(s)
        if s.kind == "acceptor" and s.score >= cfg.splice_acceptor_threshold:
            out.append(s)
    return out


def evaluate_edit_splice_delta(
    dna_before: str,
    dna_after: str,
    cfg: ReverseRosettaConfig,
    backend: SpliceBackend | None = None,
) -> tuple[float, float]:
    """
    Return ``(risk_before, risk_after)`` using the aggregate risk score.

    Full-sequence rescoring keeps the adapter simple and correct.
    """
    be = backend or get_splice_backend(cfg)
    r0 = be.scan(dna_before, cfg).risk_score
    r1 = be.scan(dna_after, cfg).risk_score
    return r0, r1
