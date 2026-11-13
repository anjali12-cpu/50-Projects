"""Local RAG helper: retrieve from the playbook, then ask Ollama."""

from pathlib import Path

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
PLAYBOOK = ROOT / "docs" / "retention_playbook.md"


def retrieve(question: str, count: int = 3) -> list[str]:
    text = PLAYBOOK.read_text(encoding="utf-8")
    passages = [part.strip() for part in text.split("\n## ") if part.strip()]
    if not passages:
        return []
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(passages + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    order = scores.argsort()[::-1][:count]
    return [passages[i] for i in order if scores[i] > 0]


def ask_local_model(question: str, evidence: list[str], model: str = "llama3.2") -> str:
    if not evidence:
        evidence = ["The supplied playbook has no clearly matching section."]
    reference_text = "\n\n".join(evidence)
    prompt = (
        "Answer the user's question using only the reference notes below. "
        "If the notes do not support an answer, say so. This is a learning demo; "
        "do not claim certainty about a real customer.\n\n"
        f"REFERENCE NOTES:\n{reference_text}\n\nQUESTION:\n{question}"
    )
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]
