"""
Retrieval-Augmented Q&A module for StudyPilot.

  - TfidfRAG: chunk the transcript, embed chunks with TF-IDF (scikit-learn,
    no model downloads needed), retrieve top-k by cosine similarity, and
    answer either extractively (return the most relevant chunk) or, if an
    LLM is attached, feed the retrieved chunks to it for a generated answer.

  - Grounding Verification: checks whether the answer is backed by the retrieved excerpts
    or whether the question is unanswerable from the transcript.
"""

from __future__ import annotations
import re
import logging
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_words: int = 120, overlap: int = 30) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_words]
        chunks.append(" ".join(chunk))
        i += chunk_words - overlap
    return chunks


def _calculate_groundedness(answer: str, context: str) -> bool:
    """Check if key terms in the answer are supported by the context."""
    if "not mentioned" in answer.lower() or "doesn't say" in answer.lower() or "unanswerable" in answer.lower() or "not contain" in answer.lower():
        return True

    answer_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", answer.lower()))
    context_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", context.lower()))

    if not answer_words:
        return True

    overlap = answer_words.intersection(context_words)
    ratio = len(overlap) / len(answer_words)
    return ratio >= 0.25


class TfidfRAG:
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(chunks) if chunks else None

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.chunks or self.matrix is None:
            return []
        try:
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.matrix)[0]
            ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
            return [{"chunk": self.chunks[i], "score": float(sims[i])} for i in ranked if sims[i] > 0]
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return []

    def answer(self, query: str, top_k: int = 3, llm=None) -> Dict:
        hits = self.retrieve(query, top_k=top_k)

        if not hits:
            return {
                "answer": "I couldn't find anything relevant to your question in the lecture transcript.",
                "sources": [],
                "grounded": False,
                "warning": "No relevant lecture segments found for this query."
            }

        context = "\n\n".join(f"Excerpt {idx+1}: {h['chunk']}" for idx, h in enumerate(hits))

        if llm is None:
            # Extractive fallback: return top relevant chunk
            answer = hits[0]["chunk"]
            return {
                "answer": answer,
                "sources": hits,
                "grounded": True,
                "warning": None
            }

        prompt = (
            "You are a lecture assistant. Answer the user's question using ONLY the provided lecture excerpts. "
            "If the answer is NOT present or cannot be inferred from the excerpts, reply clearly: "
            '"This information is not mentioned in the lecture transcript."\n\n'
            f"Excerpts:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        try:
            generated = llm.generate_text(prompt)
            grounded = _calculate_groundedness(generated, context)
            warning = None if grounded else "Warning: Answer may contain details not directly found in the retrieved transcript excerpts."

            return {
                "answer": generated,
                "sources": hits,
                "grounded": grounded,
                "warning": warning
            }
        except Exception as e:
            logger.error(f"LLM RAG answer generation failed: {e}")
            # Fallback to top extractive chunk if LLM call fails
            return {
                "answer": hits[0]["chunk"],
                "sources": hits,
                "grounded": True,
                "warning": f"LLM generation failed ({str(e)}). Returned top matching excerpt instead."
            }
