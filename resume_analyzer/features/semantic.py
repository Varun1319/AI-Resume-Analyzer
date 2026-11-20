from functools import lru_cache

import numpy as np

from resume_analyzer.config import BERT_MODEL_NAME


class BertResumeMatcher:
    def __init__(self, model_name: str = BERT_MODEL_NAME):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def pair_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Cosine similarity for each (job_desc, resume) pair.
        Every distinct text is encoded once, in a single batched call.
        """
        texts = list(dict.fromkeys(t for pair in pairs for t in pair))
        index = {t: i for i, t in enumerate(texts)}
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return [float(np.dot(embeddings[index[a]], embeddings[index[b]])) for a, b in pairs]


@lru_cache(maxsize=1)
def get_bert_matcher() -> BertResumeMatcher:
    """Shared instance; the model loads on first use rather than at import."""
    return BertResumeMatcher()
