"""CodonTransformer adapter: protein -> human-optimized coding DNA."""

from __future__ import annotations

import logging
import re
from typing import Any

from reverserosetta.config import host_to_codon_transformer_organism
from reverserosetta.utils import CODON_TO_AA, translate_dna

logger = logging.getLogger(__name__)

_tokenizer: Any = None
_model: Any = None
_device: Any = None


def _get_device() -> Any:
    import torch

    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _load_codonsformer() -> tuple[Any, Any]:
    """Lazy-load HuggingFace tokenizer and BigBird model (singleton)."""
    from transformers import AutoTokenizer, BigBirdForMaskedLM

    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logger.info("Loading CodonTransformer from HuggingFace (first call may download weights).")
        _tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
        _model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")
        _model.eval()
        _model.to(_get_device())
    return _tokenizer, _model


def _strip_trailing_stop_codons(dna: str, aa_seq: str) -> str:
    """
    CodonTransformer sometimes appends a terminal stop triplet even when the input protein omits '*'.

    Remove trailing TAA/TAG/TGA codons until length is ``3 * len(aa_seq)`` or no stop remains.
    """
    expected_nt = 3 * len(aa_seq)
    out = dna
    while len(out) > expected_nt and len(out) >= 3:
        last = out[-3:]
        if CODON_TO_AA.get(last) != "*":
            break
        out = out[:-3]
    return out


def optimize_protein_to_human_dna(
    aa_seq: str,
    *,
    host: str = "human",
    deterministic: bool = True,
    match_protein: bool = True,
    attention_type: str = "original_full",
) -> str:
    """
    Generate a human-optimized DNA coding sequence for a protein using CodonTransformer.

    Uses ``match_protein=True`` so logits are restricted to synonymous codons per position,
    preserving the amino acid sequence from the model side.

    Parameters
    ----------
    aa_seq:
        One-letter amino acid sequence (no stop unless you include '*', which is discouraged
        for this pipeline; use ``emit_stop_codon`` at orchestration level instead).
    host:
        Logical host; ``human`` maps to CodonTransformer organism ``Homo sapiens``.
    deterministic:
        Greedy argmax decoding if True.
    match_protein:
        Restrict predictions to codons compatible with each amino acid position.
    attention_type:
        BigBird attention implementation (``original_full`` is stable on CPU/GPU).

    Returns
    -------
    str
        DNA string (length 3 * len(aa_seq)), uppercase A/C/G/T only.
    """
    from CodonTransformer.CodonPrediction import predict_dna_sequence

    if not aa_seq:
        raise ValueError("Empty amino acid sequence.")
    organism = host_to_codon_transformer_organism(host)
    tokenizer, model = _load_codonsformer()
    device = _get_device()
    model.bert.set_attention_type(attention_type)

    out = predict_dna_sequence(
        protein=aa_seq.strip().upper(),
        organism=organism,
        device=device,
        tokenizer=tokenizer,
        model=model,
        attention_type=attention_type,
        deterministic=deterministic,
        match_protein=match_protein,
    )
    # predict_dna_sequence returns DNASequencePrediction or list
    pred = out[0] if isinstance(out, list) else out
    dna = pred.predicted_dna
    dna = re.sub(r"\s+", "", dna).upper()
    if not re.fullmatch(r"[ACGT]+", dna):
        raise RuntimeError(f"CodonTransformer returned non-DNA: {dna[:80]!r}...")
    dna = _strip_trailing_stop_codons(dna, aa_seq.strip().upper())
    if len(dna) != 3 * len(aa_seq):
        raise RuntimeError(
            f"DNA length {len(dna)} does not match 3 * protein length ({3 * len(aa_seq)})."
        )
    if translate_dna(dna) != aa_seq.strip().upper():
        raise RuntimeError("CodonTransformer DNA does not translate to the input protein after cleanup.")
    return dna


def reset_model_cache_for_tests() -> None:
    """Clear singleton model (testing only)."""
    global _tokenizer, _model, _device
    _tokenizer = None
    _model = None
    _device = None
