from __future__ import annotations

import numpy as np

from compact_rag.common.exceptions import ConfigurationError, EmbeddingError
from compact_rag.common.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Singleton embedding service wrapping sentence-transformers."""

    _instance: EmbeddingService | None = None

    def __new__(cls, settings=None) -> EmbeddingService:
        if cls._instance is None:
            instance = super().__new__(cls)
            cls._instance = instance
        return cls._instance

    def __init__(self, settings=None) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return

        from compact_rag.config.settings import EmbeddingSettings

        self._settings = settings if settings is not None else EmbeddingSettings()
        self._model = None
        self._dimension: int | None = None
        self._initialized = True

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = self.model.get_sentence_embedding_dimension() or 0
        return self._dimension

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ConfigurationError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )

        logger.info("Loading embedding model", model_name=self._settings.model_name)
        try:
            model = SentenceTransformer(
                self._settings.model_name,
                device=self._settings.device,
            )
            if self._settings.use_onnx:
                self._export_onnx(model)
            self._model = model
            logger.info("Embedding model loaded", dimension=model.get_sentence_embedding_dimension())
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load embedding model '{self._settings.model_name}': {e}",
                cause=e,
            )

    def _export_onnx(self, model) -> None:
        try:
            import torch

            dummy_input = {
                "input_ids": torch.zeros((1, self._settings.max_seq_length), dtype=torch.long),
                "attention_mask": torch.zeros((1, self._settings.max_seq_length), dtype=torch.long),
            }
            model._first_module().eval()
            torch.onnx.export(
                model._first_module(),
                (dummy_input,),
                f"{self._settings.model_name.replace('/', '_')}.onnx",
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "sentence_embedding": {0: "batch_size"},
                },
                opset_version=14,
            )
            logger.info("ONNX model exported successfully")
        except ImportError:
            logger.warning("ONNX export requires torch. Continuing without ONNX.")
        except Exception as e:
            logger.warning("ONNX export failed, continuing without ONNX", error=str(e))

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self._settings.batch_size,
                normalize_embeddings=self._settings.normalize,
                show_progress_bar=False,
            )
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"Failed to encode texts: {e}", cause=e)

    def encode_query(self, query: str) -> np.ndarray:
        if not query:
            raise EmbeddingError("Query text is empty")
        try:
            embedding = self.model.encode(
                query,
                normalize_embeddings=self._settings.normalize,
                show_progress_bar=False,
            )
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            return embedding
        except Exception as e:
            raise EmbeddingError(f"Failed to encode query: {e}", cause=e)
