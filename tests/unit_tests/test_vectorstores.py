"""Unit tests for LambdaDBVectorStore and create-if-not-exists behavior."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.embeddings import Embeddings

from langchain_lambdadb.vectorstores import (
    LambdaDBVectorStore,
    _default_index_configs,
    _is_collection_not_found_error,
)

# Patch LambdaDB so isinstance(client, LambdaDB) is True for our MagicMock client
_LAMBDADB_PATCH = "langchain_lambdadb.vectorstores.LambdaDB"


class _FakeEmbeddings(Embeddings):
    """Minimal Embeddings that returns fixed-dimension vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]


def test_default_index_configs_uses_embedding_dimension() -> None:
    """_default_index_configs uses dimension from embedding.embed_query."""
    emb = _FakeEmbeddings(dimension=8)
    configs = _default_index_configs(emb, text_field="page_content", vector_field="vec")
    assert configs["vec"]["type"] == "vector"
    assert configs["vec"]["dimensions"] == 8
    assert configs["vec"]["similarity"] == "cosine"
    assert configs["page_content"]["type"] == "text"
    assert "english" in configs["page_content"]["analyzers"]


def test_default_index_configs_respects_field_names() -> None:
    """_default_index_configs uses given text_field and vector_field."""
    emb = _FakeEmbeddings(dimension=3)
    configs = _default_index_configs(emb, text_field="body", vector_field="embedding")
    assert "body" in configs
    assert configs["body"]["type"] == "text"
    assert "embedding" in configs
    assert configs["embedding"]["dimensions"] == 3


def test_is_collection_not_found_error_message_not_found() -> None:
    """_is_collection_not_found_error returns True for 'not found' in message."""
    assert _is_collection_not_found_error(ValueError("Collection not found")) is True
    assert _is_collection_not_found_error(RuntimeError("resource not found")) is True


def test_is_collection_not_found_error_message_404() -> None:
    """_is_collection_not_found_error returns True for 404 in message."""
    assert _is_collection_not_found_error(ValueError("404")) is True
    assert _is_collection_not_found_error(ValueError("Error 404")) is True


def test_is_collection_not_found_error_other_raises_false() -> None:
    """_is_collection_not_found_error returns False for unrelated errors."""
    assert _is_collection_not_found_error(ValueError("bad request")) is False
    assert _is_collection_not_found_error(ConnectionError("timeout")) is False


def test_init_create_if_not_exists_false_raises_when_missing() -> None:
    """When create_if_not_exists=False and collection missing, ValueError is raised."""
    with patch(_LAMBDADB_PATCH, MagicMock):
        client = MagicMock()
        client.collections.get.side_effect = ValueError("Collection not found")
        embedding = _FakeEmbeddings(dimension=4)

        with pytest.raises(ValueError) as exc_info:
            LambdaDBVectorStore(
                client=client,
                collection_name="missing",
                embedding=embedding,
                create_if_not_exists=False,
            )
        assert "does not exist" in str(exc_info.value)
        assert "index_configs" in str(exc_info.value) or "create_if_not_exists" in str(
            exc_info.value
        )
        client.collections.create.assert_not_called()


def test_init_create_if_not_exists_true_creates_with_default_config() -> None:
    """When collection is missing and create_if_not_exists=True, create is called."""
    with patch(_LAMBDADB_PATCH, MagicMock):
        client = MagicMock()
        active_mock = MagicMock()
        active_mock.collection.collection_status.value = "ACTIVE"
        client.collections.get.side_effect = [
            ValueError("not found"),
            active_mock,
        ]
        embedding = _FakeEmbeddings(dimension=5)

        store = LambdaDBVectorStore(
            client=client,
            collection_name="new_coll",
            embedding=embedding,
            text_field="text",
            vector_field="vector",
            create_if_not_exists=True,
        )

        client.collections.create.assert_called_once()
        call_kw = client.collections.create.call_args[1]
        assert call_kw["collection_name"] == "new_coll"
        assert "index_configs" in call_kw
        configs = call_kw["index_configs"]
        assert configs["vector"]["type"] == "vector"
        assert configs["vector"]["dimensions"] == 5
        assert configs["text"]["type"] == "text"
        assert store._collection_name == "new_coll"


def test_init_create_if_not_exists_true_with_custom_index_configs() -> None:
    """When creating, custom index_configs are passed through."""
    with patch(_LAMBDADB_PATCH, MagicMock):
        client = MagicMock()
        active_mock = MagicMock()
        active_mock.collection.collection_status.value = "ACTIVE"
        client.collections.get.side_effect = [ValueError("not found"), active_mock]
        embedding = _FakeEmbeddings(dimension=2)
        custom_configs = {
            "vector": {"type": "vector", "dimensions": 10, "similarity": "cosine"},
            "text": {"type": "text", "analyzers": ["english"]},
        }

        LambdaDBVectorStore(
            client=client,
            collection_name="custom_coll",
            embedding=embedding,
            index_configs=custom_configs,
            create_if_not_exists=True,
        )

        call_kw = client.collections.create.call_args[1]
        assert call_kw["index_configs"] == custom_configs


def test_init_create_if_not_exists_true_with_partition_config() -> None:
    """When creating, partition_config is passed when provided."""
    with patch(_LAMBDADB_PATCH, MagicMock):
        client = MagicMock()
        active_mock = MagicMock()
        active_mock.collection.collection_status.value = "ACTIVE"
        client.collections.get.side_effect = [ValueError("not found"), active_mock]
        embedding = _FakeEmbeddings(dimension=2)
        partition = {"field_name": "url", "data_type": "keyword", "num_partitions": 4}

        LambdaDBVectorStore(
            client=client,
            collection_name="part_coll",
            embedding=embedding,
            partition_config=partition,
            create_if_not_exists=True,
        )

        call_kw = client.collections.create.call_args[1]
        assert call_kw.get("partition_config") == partition


def test_from_texts_passes_index_configs_and_create_if_not_exists() -> None:
    """from_texts passes index_configs, partition_config, create_if_not_exists."""
    with patch(_LAMBDADB_PATCH, MagicMock):
        client = MagicMock()
        active_mock = MagicMock()
        active_mock.collection.collection_status.value = "ACTIVE"
        client.collections.get.side_effect = [ValueError("not found"), active_mock]
        embedding = _FakeEmbeddings(dimension=3)
        custom_configs = {
            "vector": {"type": "vector", "dimensions": 3},
            "text": {"type": "text"},
        }

        with patch.object(LambdaDBVectorStore, "add_texts", MagicMock(return_value=[])):
            LambdaDBVectorStore.from_texts(
                texts=["hello"],
                embedding=embedding,
                client=client,
                collection_name="from_texts_coll",
                index_configs=custom_configs,
                create_if_not_exists=True,
            )

        call_kw = client.collections.create.call_args[1]
        assert call_kw["index_configs"] == custom_configs
