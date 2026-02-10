from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .types import Chunk


@dataclass
class HybridTfidfIndex:
    chunks: List[Chunk]
    vec_word: TfidfVectorizer
    vec_char: TfidfVectorizer
    mat_word: np.ndarray
    mat_char: np.ndarray
    alpha: float = 0.7

    def search(self, query: str, top_k: int = 8) -> List[Tuple[Chunk, float]]:
        qw = self.vec_word.transform([query]).toarray().astype(np.float32)[0]
        qc = self.vec_char.transform([query]).toarray().astype(np.float32)[0]

        sw = cosine_sim(self.mat_word, qw)
        sc = cosine_sim(self.mat_char, qc)

        sims = self.alpha * sw + (1.0 - self.alpha) * sc
        idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[int(i)], float(sims[int(i)])) for i in idx]


def build_hybrid_tfidf(chunks: List[Chunk]) -> HybridTfidfIndex:
    texts = [c.text for c in chunks]
    vec_word = TfidfVectorizer(stop_words="english", max_features=60000)
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=80000)

    mat_word = vec_word.fit_transform(texts).toarray().astype(np.float32)
    mat_char = vec_char.fit_transform(texts).toarray().astype(np.float32)

    return HybridTfidfIndex(
        chunks=chunks,
        vec_word=vec_word,
        vec_char=vec_char,
        mat_word=mat_word,
        mat_char=mat_char,
        alpha=0.7,
    )


def cosine_sim(M: np.ndarray, q: np.ndarray) -> np.ndarray:
    qn = float(np.linalg.norm(q) + 1e-12)
    Mn = np.linalg.norm(M, axis=1) + 1e-12
    return (M @ q) / (Mn * qn)
