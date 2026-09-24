"""The full RAG pipeline: query expansion -> multi-query retrieval -> rerank -> grounded answer."""

import os
import re
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from rag.document_processor import file_hash, split_pdf
from rag.retrieval import Reranker, VectorStore

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"  # used when GROQ_MODEL is not set in .env
NUM_EXPANDED_QUERIES = 3
CHUNKS_PER_QUERY = 8  # retrieved per query before deduplication and reranking

NOT_FOUND_MESSAGE = "I could not find this information in the uploaded documents."

EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You rewrite search questions. Given a question, write {n} alternative versions "
        "that use different wording or focus on different aspects, so a document search "
        "finds more relevant passages. Keep the same meaning. "
        "Output only the questions, one per line, with no numbering and no extra text.",
    ),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You answer questions using ONLY the numbered context passages from the user's PDF documents.\n"
        "Rules:\n"
        "- Use only information found in the context. Never use outside knowledge.\n"
        "- Cite the passages you use with their numbers, like [1] or [2][3], right after the "
        "sentence they support.\n"
        "- If the context does not contain the answer, reply exactly: \"{not_found}\"\n"
        "- Be clear and concise.",
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def parse_queries(text: str, original: str, limit: int) -> List[str]:
    """Turn the LLM's reply into a clean list of questions.

    Handles numbering ("1.", "2)"), bullets, quotes and intro lines like
    "Here are 3 versions:" instead of assuming one exact format.
    """
    queries = []
    seen = {original.strip().lower()}

    for line in text.splitlines():
        line = re.sub(r"^\s*(?:\d+[.):]|[-*•])\s*", "", line).strip().strip('"').strip()
        if not line or line.endswith(":"):
            continue
        if line.lower() in seen:
            continue
        seen.add(line.lower())
        queries.append(line)

    return queries[:limit]


class RAGPipeline:
    """Holds the API clients once and runs indexing and question answering."""

    def __init__(self, persist_dir: str, groq_api_key: str, cohere_api_key: str):
        self.vector_store = VectorStore(persist_dir, cohere_api_key)
        self.reranker = Reranker(cohere_api_key)
        # Read at runtime (not import time) so the value from .env is already loaded.
        model = os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL
        self.expansion_llm = ChatGroq(model=model, temperature=0.3, api_key=groq_api_key)
        self.answer_llm = ChatGroq(model=model, temperature=0, api_key=groq_api_key)

    # ---------- Indexing ----------

    def add_pdf(self, file_name: str, data: bytes) -> int:
        """Index one PDF. Returns the number of chunks added (0 if it was already indexed)."""
        if self.vector_store.contains_file(file_hash(data)):
            return 0
        chunks = split_pdf(file_name, data)
        if chunks:
            self.vector_store.add_chunks(chunks)
        return len(chunks)

    def reset(self) -> None:
        self.vector_store.reset()

    # ---------- Question answering ----------

    def expand_query(self, question: str) -> List[str]:
        """Return the original question followed by up to 3 alternative versions."""
        try:
            response = (EXPANSION_PROMPT | self.expansion_llm).invoke(
                {"question": question, "n": NUM_EXPANDED_QUERIES}
            )
            alternatives = parse_queries(response.content, question, NUM_EXPANDED_QUERIES)
        except Exception:
            alternatives = []  # expansion is optional; fall back to the original question
        return [question] + alternatives

    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        queries = self.expand_query(question)
        candidates = self.vector_store.multi_query_search(queries, CHUNKS_PER_QUERY)

        # Rerank against the ORIGINAL question, not the expanded ones.
        ranked = self.reranker.rerank(question, candidates, top_n=top_k)

        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "score": score,
                "text": doc.page_content,
            }
            for doc, score in ranked
        ]

        if not sources:
            answer = NOT_FOUND_MESSAGE
        else:
            # The full chunk text is sent to the LLM, numbered to match the citations.
            context = "\n\n".join(
                f"[{i}] (File: {s['source']}, page {s['page']})\n{s['text']}"
                for i, s in enumerate(sources, start=1)
            )
            response = (ANSWER_PROMPT | self.answer_llm).invoke(
                {"context": context, "question": question, "not_found": NOT_FOUND_MESSAGE}
            )
            answer = response.content

        return {
            "question": question,
            "answer": answer,
            "queries": queries,
            "num_candidates": len(candidates),
            "sources": sources,
        }
