"""Typed configuration for ReverseRosetta (CLI + YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ObjectiveWeights(BaseModel):
    """Weights for the soft composite objective (larger codon score is better)."""

    w_codon: float = Field(default=1.0, ge=0.0, description="Weight for codon preference.")
    w_splice: float = Field(default=1.0, ge=0.0, description="Penalty weight for splice risk.")
    w_repeat: float = Field(default=1.0, ge=0.0, description="Penalty weight for repeat burden.")


class ReverseRosettaConfig(BaseModel):
    """Full pipeline configuration (merge CLI + optional YAML file)."""

    host: str = "human"
    column_index: int = Field(
        default=4,
        ge=1,
        description="1-based Excel column index for amino acid sequences.",
    )
    emit_stop_codon: bool = False
    max_iterations: int = Field(default=500, ge=1, le=100_000)
    splice_donor_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Flag donor windows with score above this (backend-specific scale).",
    )
    splice_acceptor_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Flag acceptor windows with score above this.",
    )
    splice_window_radius: int = Field(
        default=80,
        ge=8,
        le=400,
        description="Half-width (nt) around a hit for local synonymous search.",
    )
    splicefinder_model_dir: Path | None = Field(
        default=None,
        description="Directory with CNN.h5 (and optional donor_dis.h5, acceptor_dis.h5).",
    )
    use_splicefinder_keras: bool = Field(
        default=False,
        description="If True and model files exist, use Keras SpliceFinder; else heuristic.",
    )
    repeat_homopolymer_max: int = Field(default=8, ge=4, le=30)
    repeat_kmer_size: int = Field(default=6, ge=3, le=12)
    weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights)
    random_seed: int | None = Field(default=None, description="Optional RNG seed for tie-breaks.")

    @field_validator("splicefinder_model_dir", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Any:
        if v == "" or v is None:
            return None
        return v


class CliOverrides(BaseSettings):
    """Environment-style settings are unused; reserved for future use."""

    model_config = SettingsConfigDict(env_prefix="REVERSEROSETTA_", extra="ignore")


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file into a plain dict for merging."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("YAML root must be a mapping.")
    return raw


def merge_config(base: ReverseRosettaConfig, overrides: dict[str, Any]) -> ReverseRosettaConfig:
    """Merge dict overrides (e.g. from YAML) into an existing config."""
    data = base.model_dump()
    _deep_merge(data, overrides)
    return ReverseRosettaConfig.model_validate(data)


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)  # type: ignore[arg-type]
        else:
            dst[k] = v


def host_to_codon_transformer_organism(host: str) -> str:
    """Map high-level host label to CodonTransformer organism name."""
    h = host.strip().lower()
    if h in ("human", "homo_sapiens", "homo sapiens"):
        return "Homo sapiens"
    raise ValueError(f"Unsupported host: {host!r} (use 'human').")
