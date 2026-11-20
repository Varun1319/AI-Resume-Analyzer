import threading
import time
import types

from resume_analyzer.llm.rag import analyze_resume, answer_prompts


class FakeOpenAI:
    """Embeddings count keywords; each chat answer echoes the start of its instruction."""

    KEYWORDS = ["python", "sales", "design"]

    def __init__(self, chat_delay=0.0):
        self.embedding_requests, self.chat_requests = [], []
        self.chat_delay = chat_delay
        self._lock = threading.Lock()
        self.embeddings = types.SimpleNamespace(create=self._embed)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._chat))

    def _embed(self, model, input):
        self.embedding_requests.append(list(input))
        data = [
            types.SimpleNamespace(index=i, embedding=[t.lower().count(k) + 1e-3 for k in self.KEYWORDS])
            for i, t in enumerate(input)
        ]
        return types.SimpleNamespace(data=data[::-1])  # order isn't guaranteed; callers must sort by index

    def _chat(self, model, messages, temperature, max_tokens):
        content = messages[0]["content"]
        with self._lock:
            self.chat_requests.append(content)
        time.sleep(self.chat_delay)
        answer = content.split("Instruction:\n")[1][:30]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=answer))])


def test_each_prompt_gets_its_most_similar_chunks():
    client = FakeOpenAI()
    chunks = ["python python", "sales sales", "design design"]
    answer_prompts(client, chunks, {"a": "python question", "b": "sales question", "c": "design question"}, top_k=1)

    contexts = {p.split("Instruction:\n")[1].split("\n")[0]: p.split('"""')[1] for p in client.chat_requests}
    assert contexts == {
        "python question": "python python",
        "sales question": "sales sales",
        "design question": "design design",
    }


def test_one_embeddings_request_and_answers_keyed_by_section():
    client = FakeOpenAI()
    chunks = ["python python", "sales sales"]
    result = analyze_resume(client, chunks)

    assert len(client.embedding_requests) == 1
    assert client.embedding_requests[0][:2] == chunks and len(client.embedding_requests[0]) == 6
    assert len(client.chat_requests) == 4
    assert result["summary"].startswith("Provide a **comprehensive")
    assert result["strengths"].startswith("Analyze this resume")
    assert result["weaknesses"].startswith("Critically analyze")
    assert result["job_titles"].startswith("Based on this resume")


def test_chat_calls_run_in_parallel():
    client = FakeOpenAI(chat_delay=0.3)
    start = time.perf_counter()
    analyze_resume(client, ["python python"])
    assert time.perf_counter() - start < 0.9  # four sequential calls would take 1.2s
