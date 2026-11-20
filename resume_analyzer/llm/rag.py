# llm/rag.py
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI

from resume_analyzer.config import CHAT_MODEL, EMBEDDING_MODEL, RETRIEVAL_TOP_K
from resume_analyzer.llm.prompts import ANSWER_TEMPLATE, build_prompts


def _cosine_sim_matrix(matrix, vector):
    if matrix.size == 0 or vector.size == 0:
        return np.array([])
    mat_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    vec_norm = vector / (np.linalg.norm(vector) + 1e-12)
    return np.dot(mat_norm, vec_norm)


def answer_prompts(
    client: OpenAI, chunks: list[str], prompts: dict[str, str], top_k: int = RETRIEVAL_TOP_K
) -> dict[str, str]:
    """Answer each prompt from its top-k most similar chunks: one embeddings request, chat calls in parallel."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks + list(prompts.values()))
    vectors = np.array([d.embedding for d in sorted(resp.data, key=lambda d: d.index)], dtype=float)
    chunk_vecs, prompt_vecs = vectors[:len(chunks)], vectors[len(chunks):]

    contexts = []
    for vec in prompt_vecs:
        sims = _cosine_sim_matrix(chunk_vecs, vec)
        top_idx = np.argsort(-sims)[:top_k]
        contexts.append("\n\n---\n\n".join(chunks[i] for i in top_idx))

    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        answers = pool.map(_ask, [client] * len(prompts), prompts.values(), contexts)
        return dict(zip(prompts, answers))


def _ask(client: OpenAI, prompt: str, context: str) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": ANSWER_TEMPLATE.format(context=context, prompt=prompt)}],
        temperature=0.5,
        max_tokens=1500
    )
    return resp.choices[0].message.content


def analyze_resume(client: OpenAI, chunks: list[str]) -> dict[str, str]:
    """Summary, strengths, weaknesses and suggested job titles for a chunked resume."""
    # Use only first few chunks to keep prompts compact
    base_text = " ".join(chunks[:6])
    return answer_prompts(client, chunks, build_prompts(base_text))
