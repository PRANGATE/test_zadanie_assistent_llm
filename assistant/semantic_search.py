"""Optional local multilingual embedding search, loaded lazily on first search.

If the embedding model cannot be downloaded/loaded (for example, offline on first
run), callers can transparently fall back to the repository's lexical search.
"""
from __future__ import annotations

import threading
from typing import Iterable


class SemanticSearch:
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, encoder=None):
        self._encoder = encoder
        self._lock = threading.Lock()
        self._load_error = None

    def _load(self):
        if self._encoder is not None:
            return
        with self._lock:
            if self._encoder is not None:
                return
            if self._load_error is not None:
                raise RuntimeError("Семантическая модель ранее не загрузилась") from self._load_error
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
                model = AutoModel.from_pretrained(self.MODEL_NAME)
                model.to("cpu")
                model.eval()

                def encode(texts):
                    batch = tokenizer(list(texts), padding=True, truncation=True,
                                      max_length=128, return_tensors="pt")
                    with torch.inference_mode():
                        output = model(**batch).last_hidden_state
                        mask = batch["attention_mask"].unsqueeze(-1).expand(output.size()).float()
                        pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    return pooled.cpu().tolist()

                self._encoder = encode
            except Exception as exc:  # network/cache/compatibility failures should not break basic search
                self._load_error = exc
                raise

    def rank(self, query: str, rows: Iterable[dict], limit: int = 10):
        rows = list(rows)
        if not rows:
            return []
        self._load()
        texts = [query] + [f"{row.get('kind', '')}. {row.get('title', '')}" for row in rows]
        vectors = self._encoder(texts)
        query_vector = vectors[0]
        scored = []
        for row, vector in zip(rows, vectors[1:]):
            score = sum(a * b for a, b in zip(query_vector, vector))
            scored.append((float(score), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        # Avoid returning unrelated rows just because the caller asked a question.
        return [row for score, row in scored[:limit] if score >= 0.28]
