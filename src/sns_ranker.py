from typing import Dict, Iterable, List, Tuple
import math

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SNSSimilarityRanker:
    def __init__(self, model_name: str = "princeton-nlp/sup-simcse-bert-base-uncased") -> None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for SNSSimilarityRanker. Please install it.")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=False)
        return np.array(emb, dtype=np.float32)

    def rank_neighbors(self,
                       query_text: str,
                       candidate_texts: List[str],
                       top_k: int = 5,
                       similarity_threshold: float = 0.0) -> List[Tuple[int, float]]:
        if len(candidate_texts) == 0:
            return []
        all_texts = [query_text] + candidate_texts
        emb = self.encode(all_texts)
        q = emb[0]
        cands = emb[1:]
        sims = [(_idx, _cosine_sim(q, cands[_idx])) for _idx in range(len(candidate_texts))]
        sims = [s for s in sims if s[1] >= similarity_threshold]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]


