import torch
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from config import BERT_MODEL_NAME



class BertResumeMatcher:
    def __init__(self, model_name: str = BERT_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def _cosine(self, a, b) -> float:
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-12))

    def score(self, job_desc: str, resume_text: str) -> float:
        """
        Single resume semantic similarity.
        Returns value between 0 and 1.
        """
        embeddings = self.model.encode(
            [job_desc, resume_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return self._cosine(embeddings[0], embeddings[1])

    def batch_score(self, job_desc: str, resume_texts: list[str]) -> list[float]:
        """
        Multi-resume semantic similarity.
        """
        all_texts = [job_desc] + resume_texts
        embeddings = self.model.encode(
            all_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        jd_vec = embeddings[0]
        resume_vecs = embeddings[1:]
        return [self._cosine(jd_vec, v) for v in resume_vecs]
