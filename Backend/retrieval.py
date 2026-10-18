
from pathlib import Path
from typing import Any
import uuid

import faiss
import numpy as np
import pymupdf as fitz
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# A small, general-purpose embedding model.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DocumentStore:
    def __init__(self):
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunks: list[dict[str, Any]] = []
        if hasattr(self.encoder, "get_embedding_dimension"):
            self.dimension = self.encoder.get_embedding_dimension()
        else:
            self.dimension = self.encoder.get_sentence_embedding_dimension()

    def _split_text(self, text: str, chunk_size: int = 900, overlap: int = 150):
        text = " ".join(text.split())
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap

        return chunks

    def add_pdf(self, file_path: Path, document_id: str):
        passages = []

        with fitz.open(file_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                for chunk in self._split_text(text):
                    passages.append({
                        "document_id": document_id,
                        "filename": file_path.name,
                        "page": page_num,
                        "text": chunk,
                    })

        if not passages:
            raise ValueError(
                "No selectable text found in this PDF. "
                "It may be scanned and require OCR."
            )

        texts = [p["text"] for p in passages]
        vectors = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        # Inner product on normalized vectors = cosine similarity.
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(vectors)
        self.chunks.extend(passages)

        return {
            "document_id": document_id,
            "filename": file_path.name,
            "pages_indexed": len(set(p["page"] for p in passages)),
            "chunks_indexed": len(passages),
        }

    def search(self, query: str, top_k: int = 5):
        if self.index is None or not self.chunks:
            return []

        query_vector = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        k = min(max(1, top_k), len(self.chunks))
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)].copy()
            chunk["similarity"] = round(float(score), 4)
            results.append(chunk)

        return results


document_store = DocumentStore()
