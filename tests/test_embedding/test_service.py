from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from compact_rag.common.exceptions import ConfigurationError, EmbeddingError


@pytest.fixture
def reset_singleton():
    """Reset the EmbeddingService singleton between tests."""
    from compact_rag.embedding.service import EmbeddingService

    EmbeddingService._instance = None
    yield
    EmbeddingService._instance = None


@pytest.fixture
def mock_sentence_transformer(mocker):
    """Mock sentence_transformers.SentenceTransformer."""
    mock_st = mocker.patch(
        "sentence_transformers.SentenceTransformer", autospec=True
    )
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = np.random.randn(2, 384).astype(np.float32)
    mock_st.return_value = mock_model
    return mock_st


@pytest.fixture
def mock_embedding_service(mocker, mock_sentence_transformer, reset_singleton):
    """Create a fresh EmbeddingService with mocked SentenceTransformer."""
    from compact_rag.embedding.service import EmbeddingService

    service = EmbeddingService()
    _ = service.model
    return service


class TestSingleton:
    def test_same_instance_returned_on_second_new(self, mock_sentence_transformer, reset_singleton):
        from compact_rag.embedding.service import EmbeddingService

        first = EmbeddingService()
        second = EmbeddingService()
        assert first is second

    def test_different_instances_after_reset(self, mock_sentence_transformer, reset_singleton):
        from compact_rag.embedding.service import EmbeddingService

        first = EmbeddingService()
        EmbeddingService._instance = None
        second = EmbeddingService()
        assert first is not second


class TestEncode:
    def test_encode_empty_list_returns_empty_array(self, mock_embedding_service):
        result = mock_embedding_service.encode([])
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_encode_reshapes_1d_result_to_2d(self, mock_embedding_service):
        mock_embedding_service._model.encode.return_value = np.array(
            [1.0, 2.0, 3.0], dtype=np.float32
        )
        result = mock_embedding_service.encode(["hello"])
        assert result.ndim == 2
        assert result.shape == (1, 3)

    def test_encode_preserves_2d_result(self, mock_embedding_service):
        mock_embedding_service._model.encode.return_value = np.array(
            [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
        )
        result = mock_embedding_service.encode(["hello", "world"])
        assert result.ndim == 2
        assert result.shape == (2, 2)

    def test_encode_raises_embedding_error_on_failure(self, mock_embedding_service):
        mock_embedding_service._model.encode.side_effect = RuntimeError("GPU OOM")
        with pytest.raises(EmbeddingError, match="Failed to encode texts"):
            mock_embedding_service.encode(["hello"])

    def test_encode_passes_batch_size_to_model(self, mock_embedding_service):
        mock_embedding_service._settings.batch_size = 64
        mock_embedding_service.encode(["text"])
        mock_embedding_service._model.encode.assert_called_once()
        call_kwargs = mock_embedding_service._model.encode.call_args.kwargs
        assert call_kwargs["batch_size"] == 64

    def test_encode_passes_normalize_to_model(self, mock_embedding_service):
        mock_embedding_service._settings.normalize = True
        mock_embedding_service.encode(["text"])
        call_kwargs = mock_embedding_service._model.encode.call_args.kwargs
        assert call_kwargs["normalize_embeddings"] is True

    def test_encode_passes_show_progress_bar_false(self, mock_embedding_service):
        mock_embedding_service.encode(["text"])
        call_kwargs = mock_embedding_service._model.encode.call_args.kwargs
        assert call_kwargs["show_progress_bar"] is False


class TestEncodeQuery:
    def test_encode_query_with_empty_string_raises_embedding_error(self, mock_embedding_service):
        with pytest.raises(EmbeddingError, match="Query text is empty"):
            mock_embedding_service.encode_query("")

    def test_encode_query_reshapes_1d_result_to_2d(self, mock_embedding_service):
        mock_embedding_service._model.encode.return_value = np.array(
            [1.0, 2.0, 3.0], dtype=np.float32
        )
        result = mock_embedding_service.encode_query("test")
        assert result.ndim == 2
        assert result.shape == (1, 3)

    def test_encode_query_preserves_2d_result(self, mock_embedding_service):
        mock_embedding_service._model.encode.return_value = np.array(
            [[1.0, 2.0, 3.0]], dtype=np.float32
        )
        result = mock_embedding_service.encode_query("test")
        assert result.ndim == 2
        assert result.shape == (1, 3)

    def test_encode_query_passes_normalize_to_model(self, mock_embedding_service):
        mock_embedding_service._settings.normalize = False
        mock_embedding_service.encode_query("test")
        call_kwargs = mock_embedding_service._model.encode.call_args.kwargs
        assert call_kwargs["normalize_embeddings"] is False

    def test_encode_query_passes_show_progress_bar_false(self, mock_embedding_service):
        mock_embedding_service.encode_query("test")
        call_kwargs = mock_embedding_service._model.encode.call_args.kwargs
        assert call_kwargs["show_progress_bar"] is False

    def test_encode_query_raises_embedding_error_on_failure(self, mock_embedding_service):
        mock_embedding_service._model.encode.side_effect = RuntimeError("GPU OOM")
        with pytest.raises(EmbeddingError, match="Failed to encode query"):
            mock_embedding_service.encode_query("test")


class TestDimension:
    def test_dimension_lazily_loads_model(self, mocker, reset_singleton):
        mock_st = mocker.patch(
            "sentence_transformers.SentenceTransformer", autospec=True
        )
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model

        from compact_rag.embedding.service import EmbeddingService

        service = EmbeddingService()
        assert service._model is None
        assert service._dimension is None

        dim = service.dimension
        assert dim == 768
        assert service._model is not None

    def test_dimension_returns_zero_when_model_returns_none(self, mocker, reset_singleton):
        mock_st = mocker.patch(
            "sentence_transformers.SentenceTransformer", autospec=True
        )
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = None
        mock_st.return_value = mock_model

        from compact_rag.embedding.service import EmbeddingService

        service = EmbeddingService()
        assert service.dimension == 0


class TestLoadModel:
    def test_model_load_failure_raises_configuration_error(self, mocker, reset_singleton):
        mocker.patch(
            "sentence_transformers.SentenceTransformer",
            side_effect=RuntimeError("Model not found"),
        )

        from compact_rag.embedding.service import EmbeddingService

        service = EmbeddingService()
        with pytest.raises(ConfigurationError, match="Failed to load embedding model"):
            _ = service.model

    def test_import_error_raises_configuration_error(self, mocker, reset_singleton):
        mocker.patch.dict(sys.modules, {"sentence_transformers": None})

        from compact_rag.embedding.service import EmbeddingService

        service = EmbeddingService()
        with pytest.raises(ConfigurationError, match="sentence-transformers not installed"):
            service._load_model()

    def test_use_onnx_flag_calls_export_onnx(self, mocker, reset_singleton):
        mocker.patch(
            "sentence_transformers.SentenceTransformer",
            return_value=MagicMock(get_sentence_embedding_dimension=lambda: 384),
        )

        from compact_rag.embedding.service import EmbeddingService

        service = EmbeddingService()
        service._settings.use_onnx = True
        service._export_onnx = MagicMock()
        service._load_model()
        service._export_onnx.assert_called_once()


class TestOnnxExport:
    def test_onnx_export_handles_torch_import_error(self, mocker, reset_singleton):
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named torch")
            return original_import(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=mock_import)

        from compact_rag.embedding.service import EmbeddingService

        service = EmbeddingService()
        mock_model = MagicMock()
        service._export_onnx(mock_model)

    def test_onnx_export_catches_export_exception(self, mocker, reset_singleton):
        import torch

        mocker.patch.object(torch.onnx, "export", side_effect=RuntimeError("export failed"))

        from compact_rag.embedding.service import EmbeddingService

        service = EmbeddingService()
        mock_model = MagicMock()
        mock_model._first_module.return_value = MagicMock()
        service._export_onnx(mock_model)
