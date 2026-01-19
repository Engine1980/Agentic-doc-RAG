import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.texts = []
        self.meta = []
        self.index = None

    def add_chunks(self, chunks, metadata):
        vectors = self.embedder.encode(chunks, normalize_embeddings=True)
        vectors = np.array(vectors).astype("float32")

        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(vectors)
        self.texts.extend(chunks)
        self.meta.extend(metadata)

    def search(self, query: str, k: int = 5):
        qv = self.embedder.encode([query], normalize_embeddings=True)
        qv = np.array(qv).astype("float32")

        scores, idxs = self.index.search(qv, k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            results.append({
                "score": float(score),
                "text": self.texts[i],
                "meta": self.meta[i],
            })
        return results

