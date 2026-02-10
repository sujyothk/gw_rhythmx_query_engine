from __future__ import annotations

import pickle
from dataclasses import dataclass

from .fhir_loader import load_fhir_dir
from .normalize import normalize
from .retrieval import build_hybrid_tfidf, HybridTfidfIndex
from .types import Normalized


@dataclass
class EngineIndex:
    normalized: Normalized
    retrieval: HybridTfidfIndex


def build_index(data_dir: str) -> EngineIndex:
    resources = load_fhir_dir(data_dir)
    norm = normalize(resources)
    retrieval = build_hybrid_tfidf(norm.chunks)
    return EngineIndex(normalized=norm, retrieval=retrieval)


def build_and_save_index(data_dir: str, index_path: str) -> None:
    idx = build_index(data_dir)
    with open(index_path, "wb") as f:
        pickle.dump(idx, f)


def load_index(index_path: str) -> EngineIndex:
    with open(index_path, "rb") as f:
        return pickle.load(f)
