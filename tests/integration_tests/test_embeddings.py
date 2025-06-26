"""Test LambdaDB embeddings."""

from typing import Type

from langchain_lambdadb.embeddings import LambdaDBEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[LambdaDBEmbeddings]:
        return LambdaDBEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
