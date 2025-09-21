# store.py
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer

DIM = 384  # depends on embedding model (all-MiniLM-L6-v2 -> 384)

class DocStore:
    def __init__(self, index_path="faiss.index", meta_path="meta.json", dim=DIM):
        self.index_path = index_path
        self.meta_path = meta_path
        self.dim = dim
        self.id_to_meta = {}   # id (int) -> metadata
        self.next_id = 0
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        # Load existing if present
        if os.path.exists(self.meta_path) and os.path.exists(self.index_path):
            self._load()

    def add(self, embeddings: np.ndarray, metadatas: list):
        """
        embeddings: np.ndarray shape (N, dim)
        metadatas: list of dicts length N
        """
        embeddings = embeddings.astype('float32')
        # normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        ids = np.arange(self.next_id, self.next_id + len(metadatas)).astype('int64')
        self.index.add_with_ids(embeddings, ids)
        for i, meta in enumerate(metadatas):
            self.id_to_meta[int(ids[i])] = meta
        self.next_id += len(metadatas)
        self._save()

    def search(self, query_emb: np.ndarray, top_k=5):
        q = query_emb.astype('float32')
        q = q / (np.linalg.norm(q) + 1e-12)
        D, I = self.index.search(np.array([q]), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.id_to_meta.get(int(idx))
            results.append({"id": int(idx), "score": float(score), "meta": meta})
        return results

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump({"next_id": self.next_id, "id_to_meta": self.id_to_meta}, f)

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.next_id = data.get("next_id", 0)
            self.id_to_meta = {int(k): v for k, v in data.get("id_to_meta", {}).items()}
