from __future__ import annotations

import re

import jieba
from rank_bm25 import BM25Okapi

from compact_rag.common.logger import get_logger

logger = get_logger(__name__)

_CHINESE_CHAR = re.compile(r"[\u4e00-\u9fff]")


class BM25Retriever:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self.documents: list[str] = []
        self.doc_ids: list[str] = []
        self.metadatas: list[dict] = []
        self._doc_lookup: dict[str, tuple[str, dict]] = {}
        self._is_indexed = False

    def _tokenize(self, text: str) -> list[str]:
        if _CHINESE_CHAR.search(text):
            return jieba.lcut(text)
        return text.split()

    def index(
        self,
        documents: list[str],
        doc_ids: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        if len(documents) != len(doc_ids):
            raise ValueError("documents and doc_ids must have the same length")

        if not documents:
            self.clear()
            logger.info("BM25 index cleared due to empty documents")
            return

        if metadatas is None:
            metadatas = [{} for _ in documents]
        if len(metadatas) != len(documents):
            raise ValueError("metadatas must match documents length")

        tokenized = [self._tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
        self.documents = documents
        self.doc_ids = doc_ids
        self.metadatas = metadatas
        self._doc_lookup = {
            doc_id: (documents[i], metadatas[i])
            for i, doc_id in enumerate(doc_ids)
        }
        self._is_indexed = True
        logger.info(f"BM25 index built with {len(documents)} documents")

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        if not self._is_indexed or self._bm25 is None:
            return []
        tokenized = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.doc_ids[idx], float(score)) for idx, score in ranked if score > 0]

    def rebuild_index(
        self,
        documents: list[str],
        doc_ids: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        self.index(documents, doc_ids, metadatas)

    def clear(self) -> None:
        self._bm25 = None
        self.documents = []
        self.doc_ids = []
        self.metadatas = []
        self._doc_lookup = {}
        self._is_indexed = False

    def get_document(self, doc_id: str) -> tuple[str, dict] | None:
        return self._doc_lookup.get(doc_id)

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed
