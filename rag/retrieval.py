"""Vector storage (ChromaDB + Cohere embeddings), multi-query search and Cohere reranking."""

from typing import List, Tuple

import chromadb
import cohere
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document

EMBEDDING_MODEL = "embed-english-v3.0"
RERANK_MODEL = "rerank-v3.5"
COLLECTION_NAME = "pdf_chunks"


class VectorStore:
    """Stores PDF chunks in a local ChromaDB collection."""

    def __init__(self, persist_dir: str, cohere_api_key: str):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        embeddings = CohereEmbeddings(model=EMBEDDING_MODEL, cohere_api_key=cohere_api_key)
        self.store = Chroma(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )

    def count(self) -> int:
        return self.client.get_or_create_collection(COLLECTION_NAME).count()

    def contains_file(self, file_hash: str) -> bool:
        return len(self.store.get(where={"file_hash": file_hash}, limit=1)["ids"]) > 0

    def add_chunks(self, chunks: List[Document]) -> None:
        # Chroma upserts by ID, so a chunk that already exists is overwritten, not duplicated.
        self.store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])

    def reset(self) -> None:
        # Delete the collection through Chroma instead of deleting the folder:
        # on Windows the database files stay locked while the app runs.
        self.store.reset_collection()

    def multi_query_search(self, queries: List[str], k_per_query: int) -> List[Document]:
        """Search with every query, merge the results and drop duplicate chunks."""
        unique_chunks = {}
        seen_texts = set()

        for query in queries:
            for doc in self.store.similarity_search(query, k=k_per_query):
                chunk_id = doc.metadata.get("chunk_id")
                text_key = " ".join(doc.page_content.split()).lower()
                # Skip chunks already found by another query (same ID or identical text).
                if chunk_id in unique_chunks or text_key in seen_texts:
                    continue
                unique_chunks[chunk_id] = doc
                seen_texts.add(text_key)

        return list(unique_chunks.values())


class Reranker:
    """Reorders chunks by how well they answer the question, using Cohere Rerank."""

    def __init__(self, cohere_api_key: str):
        self.client = cohere.ClientV2(api_key=cohere_api_key)

    def rerank(self, question: str, docs: List[Document], top_n: int) -> List[Tuple[Document, float]]:
        if not docs:
            return []
        response = self.client.rerank(
            model=RERANK_MODEL,
            query=question,
            documents=[doc.page_content for doc in docs],
            top_n=min(top_n, len(docs)),
        )
        return [(docs[r.index], r.relevance_score) for r in response.results]
